"""PyTorch module for EM algorithm for Dynamic Factor Models.

This module provides EMAlgorithm, a PyTorch nn.Module for the Expectation-Maximization
algorithm.

Algorithm Structure:
    E-step: Uses PyTorch Kalman smoother (all matrix operations → GPU optimal)
    M-step: Closed-form OLS regression (no autograd needed, pure matrix ops)
    
    The EM algorithm uses closed-form updates, so PyTorch autograd is not needed.
    All operations are matrix multiplications, inversions, and regressions that
    benefit greatly from GPU acceleration.

Numerical Stability:
    All matrix inversions and solves use adaptive regularization based on condition
    number to prevent singular matrix errors. This is critical for GPU stability, as
    PyTorch can throw RuntimeError for near-singular matrices.
    
    Root Cause of Divergence (Fixed):
    - Fixed regularization (1e-6) was insufficient when condition number of sum_EZZ
      (factor covariance) grows over iterations, leading to numerical instability in
      C matrix solve operation
    - Solution: Adaptive regularization scales up proportionally when condition number
      exceeds 1e8, preventing ill-conditioned matrix inversions
    
    Known Limitations:
    - Some target series may still exhibit numerical instability if data quality is
      poor (high collinearity, extreme missing data, small effective sample size)
    - The package includes multiple stability measures (adaptive regularization,
      matrix normalization, spectral radius capping, Q matrix floor), but some
      data/model combinations may still fail due to inherent numerical properties
      of the data structure
    - When EM algorithm fails for a specific target, consider:
      * Using DDFM (nonlinear encoder) as an alternative
      * Adjusting regularization_scale or other hyperparameters
      * Checking data quality (outliers, missing patterns, collinearity)

Performance:
    GPU acceleration provides significant speedup for large-scale problems.
    The E-step (Kalman smoother) and M-step (matrix regressions) are both
    highly parallelizable on GPU.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ..logger import get_logger
from .kalman import KalmanFilter
from .utils import ensure_positive_definite, ensure_symmetric, cap_max_eigenval

_logger = get_logger(__name__)


@dataclass
class EMStepParams:
    """Parameters for a single EM step using PyTorch tensors.
    
    This dataclass groups all parameters needed for one EM iteration.
    """
    y: torch.Tensor
    A: torch.Tensor
    C: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    Z_0: torch.Tensor
    V_0: torch.Tensor
    r: torch.Tensor
    p: int
    R_mat: Optional[torch.Tensor]
    q: Optional[torch.Tensor]
    nQ: int
    i_idio: torch.Tensor
    blocks: torch.Tensor
    tent_weights_dict: Dict[str, torch.Tensor]
    clock: str
    frequencies: Optional[torch.Tensor]
    idio_chain_lengths: torch.Tensor
    config: Any  # DFMConfig


class EMAlgorithm(nn.Module):
    """PyTorch module for EM algorithm.
    
    This module implements the Expectation-Maximization algorithm for Dynamic
    Factor Models. It composes a KalmanFilter for the E-step and performs
    closed-form parameter updates in the M-step.
    
    Parameters
    ----------
    kalman : KalmanFilter, optional
        KalmanFilter instance to use for E-step. If None, creates a new instance.
    regularization_scale : float, default 1e-6
        Regularization scale for matrix operations in M-step
    """
    
    def __init__(
        self,
        kalman: Optional[KalmanFilter] = None,
        regularization_scale: float = 1e-6
    ):
        super().__init__()
        # Compose KalmanFilter (create if not provided)
        if kalman is None:
            self.kalman = KalmanFilter()
        else:
            self.kalman = kalman
        self.register_buffer('regularization_scale', torch.tensor(regularization_scale))
    
    def _cap_max_eigenval(self, M: torch.Tensor, max_eigenval: float = 1e6) -> torch.Tensor:
        """Cap maximum eigenvalue of matrix to prevent numerical explosion."""
        return cap_max_eigenval(M, max_eigenval=max_eigenval, warn=False)
    
    def _compute_adaptive_regularization(
        self, 
        M: torch.Tensor, 
        matrix_name: str = "matrix",
        min_reg: float = 1e-3
    ) -> float:
        """Compute adaptive regularization based on condition number.
        
        Parameters
        ----------
        M : torch.Tensor
            Matrix to compute regularization for
        matrix_name : str
            Name for logging
        min_reg : float
            Minimum regularization value
            
        Returns
        -------
        float
            Regularization scale
        """
        base_reg = self.regularization_scale.item()
        try:
            eigenvals = torch.linalg.eigvalsh(M)
            eigenvals = eigenvals[eigenvals > 1e-12]
            if len(eigenvals) > 0:
                max_eig = torch.max(eigenvals)
                min_eig = torch.min(eigenvals)
                cond_num = max_eig / min_eig if min_eig > 1e-12 else float('inf')
                
                if cond_num > 1e8:
                    reg_scale = base_reg * (cond_num / 1e8)
                    _logger.debug(f"EM: {matrix_name} ill-conditioned (cond={cond_num:.2e}), reg={reg_scale:.2e}")
                else:
                    reg_scale = base_reg
            else:
                reg_scale = max(base_reg * 100, min_reg)
                _logger.warning(f"EM: {matrix_name} has no valid eigenvalues, using reg={reg_scale:.2e}")
        except (RuntimeError, ValueError) as e:
            reg_scale = max(base_reg * 10, min_reg)
            _logger.warning(f"EM: Failed to compute condition number for {matrix_name} ({e}), using reg={reg_scale:.2e}")
        
        return reg_scale
    
    def forward(
        self,
        params: EMStepParams
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Perform EM step. Main entry point.
        
        Parameters
        ----------
        params : EMStepParams
            Parameters for this EM step
            
        Returns
        -------
        C : torch.Tensor
            Updated observation matrix (N x m)
        R : torch.Tensor
            Updated observation covariance (N x N)
        A : torch.Tensor
            Updated transition matrix (m x m)
        Q : torch.Tensor
            Updated process noise covariance (m x m)
        Z_0 : torch.Tensor
            Updated initial state (m,)
        V_0 : torch.Tensor
            Updated initial covariance (m x m)
        loglik : float
            Log-likelihood value
        """
        device = params.y.device
        dtype = params.y.dtype
        
        # E-step: Kalman smoother (uses self.kalman)
        zsmooth, Vsmooth, VVsmooth, loglik = self.kalman(
            params.y, params.A, params.C, params.Q, params.R, params.Z_0, params.V_0
        )
        
        # zsmooth is m x (T+1), transpose to (T+1) x m
        Zsmooth = zsmooth.T
        
        T = params.y.shape[1]
        m = params.A.shape[0]
        N = params.C.shape[0]
        
        # Extract smoothed moments needed for M-step
        # E[Z_t | y_{1:T}]: smoothed factor means
        EZ = Zsmooth[1:, :]  # T x m (skip initial state)
        
        # E[Z_t Z_t^T | y_{1:T}]: smoothed factor covariances (vectorized)
        # Vsmooth is (m, m, T+1), EZ is (T, m)
        # Vectorized: EZZ[t] = Vsmooth[:, :, t+1] + outer(EZ[t], EZ[t])
        EZZ = Vsmooth[:, :, 1:].permute(2, 0, 1) + torch.bmm(EZ[:, :, None], EZ[:, None, :])
        
        # E[Z_t Z_{t-1}^T | y_{1:T}]: lag-1 cross-covariances (vectorized)
        # VVsmooth is (m, m, T), EZ is (T, m)
        # Vectorized: EZZ_lag1[t] = VVsmooth[:, :, t] + outer(EZ[t+1], EZ[t])
        # EZ[1:] is (T-1, m) for t+1, EZ[:-1] is (T-1, m) for t
        EZZ_lag1 = VVsmooth[:, :, :T-1].permute(2, 0, 1) + torch.bmm(
            EZ[1:, :, None],  # (T-1, m, 1)
            EZ[:-1, :, None].transpose(1, 2)  # (T-1, 1, m) -> (T-1, m, m)
        )
        
        # M-step: Update parameters via regressions
        
        # Update A (transition matrix): regression of Z_t on Z_{t-1}
        if T > 1:
            # Prepare data: Y = Z_t, X = Z_{t-1}
            Y_A = EZ[1:, :]  # (T-1) x m
            X_A = EZ[:-1, :]  # (T-1) x m
            
            # OLS: A = (X'X)^{-1} X'Y
            try:
                # Compute XTX = sum_t X_t X_t^T (vectorized: batch outer products)
                XTX_A = torch.sum(X_A[:, :, None] * X_A[:, None, :], dim=0)
                # Compute XTY = sum_t X_t Y_t^T (vectorized: batch outer products)
                XTY_A = torch.sum(X_A[:, :, None] * Y_A[:, None, :], dim=0)
                
                # Adaptive regularization based on condition number
                reg_scale = self._compute_adaptive_regularization(XTX_A, "XTX_A", min_reg=1e-6)
                XTX_A_reg = XTX_A + torch.eye(m, device=device, dtype=dtype) * reg_scale
                A_new = torch.linalg.solve(XTX_A_reg, XTY_A).T
                
                # Ensure stability
                eigenvals_A = torch.linalg.eigvals(A_new)
                max_eigenval = torch.max(torch.abs(eigenvals_A))
                if max_eigenval >= 0.99:
                    A_new = A_new * (0.99 / max_eigenval)
                
                # Clean and clip
                A_new = torch.nan_to_num(A_new, nan=0.0, posinf=0.99, neginf=-0.99)
                A_new = torch.clamp(A_new, min=-0.99, max=0.99)
            except (RuntimeError, ValueError):
                A_new = params.A.clone()
                # Still apply clipping even if eigvals failed
                A_new = torch.nan_to_num(A_new, nan=0.0, posinf=0.99, neginf=-0.99)
                A_new = torch.clamp(A_new, min=-0.99, max=0.99)
        else:
            A_new = params.A.clone()
        
        # Update C (observation matrix): regression of y_t on Z_t
        # Set NaN to 0 for M-step calculations
        y_for_mstep = params.y.clone()
        y_for_mstep[torch.isnan(y_for_mstep)] = 0.0
        
        # C = (sum_t y_t E[Z_t^T]) (sum_t E[Z_t Z_t^T])^{-1}
        try:
            # Compute sum_yEZ = sum_t y_t E[Z_t^T] (vectorized: batch outer products)
            # params.y is (N, T), EZ is (T, m)
            # Transpose y to (T, N) for batch operations
            # Use y_for_mstep (NaN replaced with 0) for M-step
            sum_yEZ = torch.sum(y_for_mstep.T[:, :, None] * EZ[:, None, :], dim=0)  # (N, m)
            
            # Remove NaN/Inf from EZZ before computing sum_EZZ
            EZZ_clean = EZZ.clone()
            nan_inf_mask = torch.isnan(EZZ_clean) | torch.isinf(EZZ_clean)
            if torch.any(nan_inf_mask):
                EZZ_clean[nan_inf_mask] = 0.0
                corrupted_count = torch.sum(nan_inf_mask).item()
                _logger.warning(f"EM: EZZ contains {corrupted_count}/{EZZ_clean.numel()} NaN/Inf, excluding from sum")
            
            sum_EZZ = torch.sum(EZZ_clean, dim=0)
            
            # Cap maximum eigenvalue to prevent condition number explosion
            sum_EZZ = self._cap_max_eigenval(sum_EZZ, max_eigenval=1e6)
            
            # Adaptive regularization based on condition number
            reg_scale = self._compute_adaptive_regularization(sum_EZZ, "sum_EZZ", min_reg=1e-3)
            sum_EZZ_reg = sum_EZZ + torch.eye(m, device=device, dtype=dtype) * reg_scale
            
            # Use pseudo-inverse as fallback when solve fails
            try:
                C_new = torch.linalg.solve(sum_EZZ_reg.T, sum_yEZ.T).T
            except RuntimeError as e:
                _logger.warning(f"EM: solve failed for C matrix ({e}), using pseudo-inverse fallback")
                C_new = (torch.linalg.pinv(sum_EZZ_reg.T) @ sum_yEZ.T).T
            
            # Handle NaN in C_new
            if torch.any(torch.isnan(C_new)):
                nan_mask = torch.isnan(C_new)
                nan_count = torch.sum(nan_mask).item()
                nan_ratio = nan_count / C_new.numel()
                _logger.warning(f"EM: C matrix contains {nan_count}/{C_new.numel()} NaN ({nan_ratio:.1%})")
                
                # Preserve previous iteration values if available
                if not torch.any(torch.isnan(params.C)):
                    C_new[nan_mask] = params.C[nan_mask]
                    if nan_ratio > 0.1:
                        _logger.warning(f"EM: Preserved {nan_count} NaN values from previous iteration")
                else:
                        # If previous C also has NaN, set to zero as last resort
                        C_new[nan_mask] = 0.0
                        _logger.warning(
                            f"Previous C matrix also contains NaN. Set {torch.sum(nan_mask).item()} NaN values to zero."
                        )
            
            # Normalize C columns (factor loadings)
            for j in range(m):
                norm = torch.linalg.norm(C_new[:, j])
                if norm > 1e-8:
                    C_new[:, j] = C_new[:, j] / norm
                elif norm < 1e-8:
                    # Very small norm: set column to zero to avoid division issues
                    C_new[:, j] = 0.0
                    _logger.debug(f"C matrix column {j} has very small norm ({norm:.2e}), set to zero.")
        except (RuntimeError, ValueError) as e:
            _logger.warning(f"EM algorithm: Error updating C matrix: {e}. Keeping previous C matrix.")
            C_new = params.C.clone()
            # Check if previous C also contains NaN
            if torch.any(torch.isnan(C_new)):
                _logger.error(
                    f"EM algorithm: Previous C matrix also contains NaN. "
                    f"This indicates the model cannot be trained with current data/parameters."
                )
        
        # Update Q (process noise covariance): residual covariance from transition
        if T > 1:
            # Vectorized: residuals_Q = EZ[1:] - (A_new @ EZ[:-1].T).T
            residuals_Q = EZ[1:, :] - (A_new @ EZ[:-1, :].T).T
            # Handle single factor case (torch.cov returns 0-D tensor for single variable)
            if residuals_Q.shape[1] == 1:
                var_val = torch.var(residuals_Q, dim=0, unbiased=False)
                Q_new = var_val.unsqueeze(0).unsqueeze(0)  # (1, 1)
            else:
                Q_new = torch.cov(residuals_Q.T)
                Q_new = ensure_symmetric(Q_new)
            # Ensure positive definite with robust eigenvalue computation
                Q_new = ensure_symmetric(Q_new)
            try:
                eigenvals_Q = torch.linalg.eigvalsh(Q_new)
                min_eigenval = torch.min(eigenvals_Q)
                if min_eigenval < 1e-8:
                    Q_new = Q_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
                    Q_new = ensure_symmetric(Q_new)  # Re-apply symmetry after regularization
            except (RuntimeError, ValueError) as e:
                _logger.warning(f"eigvalsh failed for Q matrix ({e}), applying stronger regularization")
                Q_new = Q_new + torch.eye(m, device=device, dtype=dtype) * 1e-6
                Q_new = ensure_symmetric(Q_new)
                try:
                    eigenvals_Q = torch.linalg.eigvalsh(Q_new)
                    min_eigenval = torch.min(eigenvals_Q)
                    if min_eigenval < 1e-8:
                        Q_new = Q_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
                        Q_new = ensure_symmetric(Q_new)
                except (RuntimeError, ValueError):
                    diag_Q = torch.diag(Q_new)
                    diag_Q = torch.maximum(diag_Q, torch.ones_like(diag_Q) * 1e-6)
                    Q_new = torch.diag(diag_Q)
            # Floor for Q and clean
            Q_new = torch.maximum(Q_new, torch.eye(m, device=device, dtype=dtype) * 0.01)
            Q_new = ensure_symmetric(Q_new)
            Q_new = torch.nan_to_num(Q_new, nan=0.01, posinf=1e6, neginf=0.01)
            # Final check: ensure positive definite with stronger regularization
            Q_new = ensure_positive_definite(Q_new, min_eigenval=1e-6, warn=False)
        else:
            Q_new = params.Q.clone()
        
        # Update R (observation covariance): residual covariance from observation
        # Vectorized: residuals_R = params.y.T - (C_new @ EZ.T).T
        # params.y is (N, T), EZ is (T, m), C_new is (N, m)
        # Use y_for_mstep (NaN replaced with 0) for M-step (MATLAB behavior)
        residuals_R = y_for_mstep.T - (C_new @ EZ.T).T  # (T, N)
        # Handle single series case (torch.cov returns 0-D tensor for single variable)
        if residuals_R.shape[1] == 1:
            var_val = torch.var(residuals_R, dim=0, unbiased=False)
            R_new = var_val.unsqueeze(0).unsqueeze(0)  # (1, 1)
        else:
            R_new = torch.cov(residuals_R.T)
            R_new = (R_new + R_new.T) / 2
        
        # Ensure R is diagonal (idiosyncratic variances only)
        if R_new.ndim > 2:
            _logger.warning(f"R_new has unexpected shape: {R_new.shape}, reshaping")
            R_new = R_new.reshape(-1, R_new.shape[-1])[-R_new.shape[-1]:, :]
        elif R_new.ndim == 1:
            R_new = R_new.unsqueeze(0)
        
        # Extract diagonal and create diagonal matrix
        diag_R = torch.diag(R_new) if R_new.ndim == 2 else R_new
        if diag_R.ndim > 1:
            diag_R = diag_R.flatten()
        
        # Clean and clamp diagonal
        # Use 1e-4 as minimum (MATLAB default) instead of 1e-6 for better numerical stability
        diag_R = torch.nan_to_num(diag_R, nan=1e-4, posinf=1e4, neginf=1e-4)  # Clean NaN/Inf
        diag_R = torch.clamp(diag_R, min=1e-4, max=1e4)
        R_new = torch.diag(diag_R)
        
        # Ensure positive definite (minimum variance floor)
        R_new = torch.maximum(R_new, torch.eye(N, device=device, dtype=dtype) * 1e-4)
        
        # Update Z_0 and V_0 (use first smoothed state)
        Z_0_new = Zsmooth[0, :]  # Initial state
        V_0_new = Vsmooth[:, :, 0]  # Initial covariance
        
        # Check for NaN in all updated parameters before returning
        params_to_check = {
            'C': C_new,
            'A': A_new,
            'Q': Q_new,
            'R': R_new,
            'Z_0': Z_0_new,
            'V_0': V_0_new
        }
        
        nan_detected = False
        for param_name, param_tensor in params_to_check.items():
            if torch.any(torch.isnan(param_tensor)):
                nan_count = torch.sum(torch.isnan(param_tensor)).item()
                nan_ratio = nan_count / param_tensor.numel()
                _logger.warning(
                    f"EM algorithm: {param_name} matrix contains {nan_count}/{param_tensor.numel()} NaN values "
                    f"({nan_ratio:.1%}) after M-step. This indicates numerical instability."
                )
                nan_detected = True
                # Replace NaN with previous value or zero
                if param_name == 'C':
                    # For C, we already handled NaN above, but check again
                    if torch.any(torch.isnan(C_new)):
                        C_new = torch.nan_to_num(C_new, nan=0.0)
                elif param_name == 'A':
                    A_new = torch.nan_to_num(A_new, nan=0.0)
                elif param_name == 'Q':
                    Q_new = torch.nan_to_num(Q_new, nan=params.Q.clone())
                elif param_name == 'R':
                    R_new = torch.nan_to_num(R_new, nan=params.R.clone())
                elif param_name == 'Z_0':
                    Z_0_new = torch.nan_to_num(Z_0_new, nan=params.Z_0.clone())
                elif param_name == 'V_0':
                    V_0_new = torch.nan_to_num(V_0_new, nan=params.V_0.clone())
        
        if nan_detected:
            _logger.error(
                "EM algorithm: NaN detected in parameter updates. "
                "This usually indicates: (1) singular matrix in solve operations, "
                "(2) extreme data values, (3) insufficient regularization, or "
                "(4) numerical precision issues. Consider increasing regularization_scale "
                "or checking data quality."
            )
        
        # Ensure V_0 is positive definite
        # Ensure V_0 is positive definite with robust eigenvalue computation
        try:
            eigenvals_V0 = torch.linalg.eigvalsh(V_0_new)
            min_eigenval = torch.min(eigenvals_V0)
            if min_eigenval < 1e-8:
                V_0_new = V_0_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
        except (RuntimeError, ValueError) as e:
            # eigvalsh failed due to ill-conditioning - apply stronger regularization
            _logger.warning(f"eigvalsh failed for V_0_new matrix: {e}. Applying stronger regularization.")
            V_0_new = V_0_new + torch.eye(m, device=device, dtype=dtype) * 1e-6
            try:
                eigenvals_V0 = torch.linalg.eigvalsh(V_0_new)
                min_eigenval = torch.min(eigenvals_V0)
                if min_eigenval < 1e-8:
                    V_0_new = V_0_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
            except (RuntimeError, ValueError):
                # Still failing - use diagonal matrix with variance estimates
                diag_V0 = torch.diag(V_0_new)
                diag_V0 = torch.maximum(diag_V0, torch.ones_like(diag_V0) * 1e-6)
                V_0_new = torch.diag(diag_V0)
        
        return C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik
    
    def initialize_parameters(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        p: int,
        blocks: torch.Tensor,
        opt_nan: Dict[str, Any],
        R_mat: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        nQ: int = 0,
        i_idio: Optional[torch.Tensor] = None,
        clock: str = 'm',
        tent_weights_dict: Optional[Dict[str, torch.Tensor]] = None,
        frequencies: Optional[torch.Tensor] = None,
        idio_chain_lengths: Optional[torch.Tensor] = None,
        config: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize DFM parameters using residual-based PCA (matching MATLAB InitCond).
        
        This method implements the MATLAB InitCond() approach:
        1. Start with residuals = spline-interpolated data
        2. For each block: compute PCA on residuals, extract factors, update residuals
        3. Build transition matrices block-by-block
        4. Handle idiosyncratic components (monthly AR(1) and quarterly 5-state chains)
        
        Parameters
        ----------
        x : torch.Tensor
            Standardized data matrix (T x N)
        r : torch.Tensor
            Number of factors per block (n_blocks,)
        p : int
            AR lag order (typically 1)
        blocks : torch.Tensor
            Block structure array (N x n_blocks)
        opt_nan : dict
            Missing data handling options {'method': int, 'k': int}
        R_mat : torch.Tensor, optional
            Constraint matrix for tent kernel aggregation
        q : torch.Tensor, optional
            Constraint vector for tent kernel aggregation
        nQ : int
            Number of slower-frequency series
        i_idio : torch.Tensor, optional
            Indicator array (1 for clock frequency, 0 for slower frequencies)
        clock : str
            Clock frequency ('d', 'w', 'm', 'q', 'sa', 'a')
        tent_weights_dict : dict, optional
            Dictionary mapping frequency pairs to tent weights
        frequencies : torch.Tensor, optional
            Array of frequencies for each series
        idio_chain_lengths : torch.Tensor, optional
            Array of idiosyncratic chain lengths per series
        config : Any, optional
            Configuration object
            
        Returns
        -------
        A : torch.Tensor
            Initial transition matrix (m x m)
        C : torch.Tensor
            Initial observation/loading matrix (N x m)
        Q : torch.Tensor
            Initial process noise covariance (m x m)
        R : torch.Tensor
            Initial observation noise covariance (N x N)
        Z_0 : torch.Tensor
            Initial state vector (m,)
        V_0 : torch.Tensor
            Initial state covariance (m x m)
        """
        T, N = x.shape
        device = x.device
        dtype = x.dtype
        
        n_blocks = blocks.shape[1]
        nM = N - nQ  # Number of monthly series
        
        # Handle missing data for initialization using GPU-accelerated PyTorch version
        from ..utils.data import rem_nans_spline_torch
        x_clean, indNaN = rem_nans_spline_torch(x, method=opt_nan.get('method', 2), k=opt_nan.get('k', 3))
        
        # Remove any remaining NaN/inf
        x_clean = torch.where(torch.isfinite(x_clean), x_clean, torch.tensor(0.0, device=device, dtype=dtype))
        
        # Initialize residuals: res = x_clean (spline-interpolated data)
        # This matches MATLAB: res = xBal; resNaN = xNaN;
        res = x_clean.clone()  # T x N
        resNaN = x_clean.clone()
        resNaN[indNaN] = torch.nan
        
        # Determine tent kernel size (pC) for slower-frequency aggregation
        pC = 5  # Default: quarterly to monthly uses 5 periods [1,2,3,2,1]
        if R_mat is not None:
            pC = R_mat.shape[1]
        elif tent_weights_dict is not None and 'q' in tent_weights_dict:
            pC = len(tent_weights_dict['q'])
        ppC = max(p, pC)  # max(p, pC) for lag structure
        
        # Set first pC-1 observations as NaN for quarterly-monthly aggregation scheme
        if pC > 1:
            resNaN[:pC-1, :] = torch.nan
        
        # Initialize output matrices
        C_list = []  # Will concatenate block loadings
        A_list = []  # Will build block-diagonal transition matrix
        Q_list = []  # Will build block-diagonal process noise
        V_0_list = []  # Will build block-diagonal initial covariance
        
        # Process each block sequentially (residual-based approach)
        _logger.info(f"Processing {n_blocks} blocks. r tensor: {r}, shape: {r.shape}")
        for i in range(n_blocks):
            r_i = int(r[i].item())  # Number of factors for this block
            _logger.info(f"Block {i}: r_i={r_i}, ppC={ppC}, expected size={r_i * ppC}")
            
            # Find series indices loading on this block
            idx_i = torch.where(blocks[:, i] > 0)[0]  # Series loading block i
            idx_iM = idx_i[idx_i < nM]  # Monthly series indices
            idx_iQ = idx_i[idx_i >= nM]  # Quarterly series indices
            
            # Initialize observation matrix for this block
            C_i = torch.zeros(N, r_i * ppC, device=device, dtype=dtype)
            
            if len(idx_iM) > 0:
                # === MONTHLY SERIES: PCA on residuals ===
                # Compute covariance of residuals for monthly series in this block
                res_M = res[:, idx_iM]  # T x n_iM
                # Center the data
                res_M_centered = res_M - res_M.mean(dim=0, keepdim=True)
                # Compute covariance
                n_iM = len(idx_iM)
                if res_M_centered.shape[0] > 1 and n_iM > 1:
                    # Multiple series: use torch.cov
                    cov_res = torch.cov(res_M_centered.T)  # n_iM x n_iM
                    cov_res = (cov_res + cov_res.T) / 2  # Symmetrize
                elif res_M_centered.shape[0] > 1 and n_iM == 1:
                    # Single series: use torch.var and convert to 2D matrix
                    var_val = torch.var(res_M_centered, dim=0, unbiased=False)
                    cov_res = var_val.unsqueeze(0).unsqueeze(0)  # (1, 1)
                else:
                    # Not enough data: use identity
                    cov_res = torch.eye(n_iM, device=device, dtype=dtype)
                
                # Compute PCA: extract r_i principal components
                from ..encoder.pca import compute_principal_components_torch
                try:
                    eigenvalues, eigenvectors = compute_principal_components_torch(cov_res, r_i)
                    v = eigenvectors  # n_iM x r_i
                    
                    # Sign flipping for cleaner output (MATLAB: if sum(v) < 0, v = -v)
                    v_sum = torch.sum(v, dim=0)
                    v = torch.where(v_sum < 0, -v, v)
                except (RuntimeError, ValueError):
                    # Fallback: use identity
                    v = torch.eye(len(idx_iM), device=device, dtype=dtype)[:, :r_i]
                
                # Set loadings for monthly series
                C_i[idx_iM, :r_i] = v
                
                # Extract factors: f = res(:,idx_iM) * v
                f = res[:, idx_iM] @ v  # T x r_i
            else:
                # No monthly series in this block, use zeros
                f = torch.zeros(T, r_i, device=device, dtype=dtype)
            
            # Build lag matrix F for quarterly series (and transition equation)
            # MATLAB: for kk = 0:max(p+1,pC)-1, F = [F f(pC-kk:end-kk,:)]
            F = torch.zeros(T, 0, device=device, dtype=dtype)
            max_lag = max(p + 1, pC)
            _logger.debug(f"Block {i}: Building F matrix with max_lag={max_lag}, f shape={f.shape}, r_i={r_i}")
            for kk in range(max_lag):
                start_idx = pC - kk
                end_idx = T - kk
                if start_idx < 0:
                    start_idx = 0
                if end_idx > T:
                    end_idx = T
                if start_idx < end_idx:
                    f_lag = f[start_idx:end_idx, :]
                    # Ensure f_lag has correct number of columns (r_i)
                    if f_lag.shape[1] != r_i:
                        _logger.warning(f"Block {i}, kk={kk}: f_lag shape mismatch: {f_lag.shape}, expected (?, {r_i}). Adjusting...")
                        if f_lag.shape[1] < r_i:
                            # Pad columns
                            f_lag = torch.cat([f_lag, torch.zeros(f_lag.shape[0], r_i - f_lag.shape[1], device=device, dtype=dtype)], dim=1)
                        else:
                            # Trim columns
                            f_lag = f_lag[:, :r_i]
                    # Pad to match T
                    if start_idx > 0:
                        f_lag = torch.cat([torch.zeros(start_idx, r_i, device=device, dtype=dtype), f_lag], dim=0)
                    if len(f_lag) < T:
                        f_lag = torch.cat([f_lag, torch.zeros(T - len(f_lag), r_i, device=device, dtype=dtype)], dim=0)
                    # Verify f_lag shape before concatenation
                    if f_lag.shape[1] != r_i:
                        _logger.error(f"Block {i}, kk={kk}: f_lag still has wrong shape: {f_lag.shape}, expected (T, {r_i})")
                        f_lag = f_lag[:, :r_i] if f_lag.shape[1] > r_i else torch.cat([f_lag, torch.zeros(f_lag.shape[0], r_i - f_lag.shape[1], device=device, dtype=dtype)], dim=1)
                    F = torch.cat([F, f_lag], dim=1)  # T x (r_i * (kk+1))
                    _logger.debug(f"Block {i}, kk={kk}: f_lag shape={f_lag.shape}, F shape after cat={F.shape}")
            
            # Extract ff for quarterly series: ff = F(:, 1:r_i*pC)
            ff = F[:, :r_i * pC] if F.shape[1] >= r_i * pC else F
            
            # === QUARTERLY SERIES: Constrained least squares with tent kernel ===
            if R_mat is not None and q is not None and len(idx_iQ) > 0:
                Rcon_i = torch.kron(R_mat, torch.eye(r_i, device=device, dtype=dtype))
                q_i = torch.kron(q, torch.zeros(r_i, device=device, dtype=dtype))
                
                for j in idx_iQ:
                    j_idx = int(j.item())
                    # Extract series j data (drop first pC observations for lag structure)
                    xx_j = resNaN[pC:, j_idx]
                    
                    # Check if enough non-NaN observations
                    non_nan_mask = ~torch.isnan(xx_j)
                    if torch.sum(non_nan_mask) < ff.shape[1] + 2:
                        # Use spline data if too many NaNs
                        xx_j = res[pC:, j_idx]
                        non_nan_mask = torch.ones(len(xx_j), dtype=torch.bool, device=device)
                    
                    # Extract non-NaN rows
                    ff_j = ff[pC:][non_nan_mask, :]
                    xx_j_clean = xx_j[non_nan_mask]
                    
                    if len(ff_j) > 0 and ff_j.shape[0] >= ff_j.shape[1]:
                        try:
                            # OLS: Cc = (ff_j'*ff_j)^{-1} * ff_j' * xx_j
                            iff_j = torch.linalg.pinv(ff_j.T @ ff_j)
                            Cc = iff_j @ ff_j.T @ xx_j  # r_i*pC
                            
                            # Apply tent kernel constraint: Cc = Cc - iff_j*Rcon_i'*inv(Rcon_i*iff_j*Rcon_i')*(Rcon_i*Cc-q_i)
                            Rcon_iff = Rcon_i @ iff_j
                            Rcon_iff_RconT = Rcon_iff @ Rcon_i.T
                            try:
                                Cc_constrained = Cc - iff_j @ Rcon_i.T @ torch.linalg.solve(
                                    Rcon_iff_RconT + torch.eye(Rcon_iff_RconT.shape[0], device=device, dtype=dtype) * 1e-6,
                                    Rcon_i @ Cc - q_i
                                )
                            except (RuntimeError, ValueError):
                                Cc_constrained = Cc
                            
                            # Set loadings for quarterly series
                            C_i[j_idx, :r_i * pC] = Cc_constrained
                        except (RuntimeError, ValueError):
                            # Fallback: use zeros
                            pass
            
            # Pad ff with zeros for first pC-1 entries (MATLAB: ff = [zeros(pC-1,pC*r_i);ff])
            if pC > 1:
                ff_padded = torch.cat([
                    torch.zeros(pC - 1, r_i * pC, device=device, dtype=dtype),
                    ff[:T - (pC - 1), :r_i * pC] if T > (pC - 1) else ff[:, :r_i * pC]
                ], dim=0)
                if len(ff_padded) < T:
                    ff_padded = torch.cat([
                        ff_padded,
                        torch.zeros(T - len(ff_padded), r_i * pC, device=device, dtype=dtype)
                    ], dim=0)
                ff = ff_padded[:T, :]
            
            # Update residuals: res = res - ff * C_i'
            # MATLAB: res = res - ff*C_i'
            # Ensure dimensions match (ff should be T x (r_i * pC), res should be T x N)
            if res.shape[0] != ff.shape[0]:
                # Pad or trim ff to match res
                if res.shape[0] < ff.shape[0]:
                    ff = ff[:res.shape[0], :]
                else:
                    ff = torch.cat([ff, torch.zeros(res.shape[0] - ff.shape[0], ff.shape[1], device=device, dtype=dtype)], dim=0)
            res = res - ff @ C_i[:, :r_i * pC].T
            resNaN = res.clone()
            resNaN[indNaN] = torch.nan
            
            # Accumulate C
            C_list.append(C_i)
            
            # === TRANSITION EQUATION for this block ===
            # MATLAB: z = F(:,1:r_i), Z = F(:,r_i+1:r_i*(p+1))
            z = F[:, :r_i] if F.shape[1] >= r_i else torch.zeros(T, r_i, device=device, dtype=dtype)
            Z = F[:, r_i:r_i * (p + 1)] if F.shape[1] >= r_i * (p + 1) else torch.zeros(T, r_i * p, device=device, dtype=dtype)
            
            # Initialize transition matrix for this block
            A_i = torch.zeros(r_i * ppC, r_i * ppC, device=device, dtype=dtype)
            
            if T > p and Z.shape[1] > 0:
                try:
                    # OLS: A_temp = inv(Z'*Z)*Z'*z
                    ZTZ = Z.T @ Z
                    reg_scale = self.regularization_scale.item()
                    ZTZ_reg = ZTZ + torch.eye(ZTZ.shape[0], device=device, dtype=dtype) * reg_scale
                    z_T = z.T if z.ndim > 1 else z.unsqueeze(0)  # Ensure z is 2D: (r_i, T)
                    ZTz = Z.T @ z  # (r_i*p, r_i) or (r_i*p, T) depending on z shape
                    A_temp = torch.linalg.solve(ZTZ_reg, ZTz).T  # r_i x (r_i*p)
                    
                    # Ensure A_temp has correct shape
                    # Defensive coding: Shape mismatch can occur if z has unexpected dimensions
                    # despite z_T transformation, or if Z.T @ z produces unexpected shape.
                    # This adjustment pads with zeros (safe, just zero-initializes some parameters)
                    # and prevents crashes, but the warning log helps identify when this occurs.
                    # If this warning appears frequently, investigate z/Z shape construction.
                    if A_temp.shape != (r_i, r_i * p):
                        _logger.warning(f"Block {i}: A_temp shape mismatch: {A_temp.shape}, expected ({r_i}, {r_i * p}). Adjusting...")
                        A_temp_new = torch.zeros(r_i, r_i * p, device=device, dtype=dtype)
                        min_rows = min(A_temp.shape[0], r_i)
                        min_cols = min(A_temp.shape[1], r_i * p)
                        A_temp_new[:min_rows, :min_cols] = A_temp[:min_rows, :min_cols]
                        A_temp = A_temp_new
                    
                    # Set transition matrix: A_i(1:r_i,1:r_i*p) = A_temp'
                    A_i[:r_i, :r_i * p] = A_temp
                    # Identity matrices for lag structure: A_i(r_i+1:end,1:r_i*(ppC-1)) = eye
                    if r_i * (ppC - 1) > 0:
                        A_i[r_i:, :r_i * (ppC - 1)] = torch.eye(r_i * (ppC - 1), device=device, dtype=dtype)
                except (RuntimeError, ValueError):
                    # Fallback: use identity for AR(1) part
                    A_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.9
                    if r_i * (ppC - 1) > 0:
                        A_i[r_i:, :r_i * (ppC - 1)] = torch.eye(r_i * (ppC - 1), device=device, dtype=dtype)
            else:
                # Not enough data: use identity
                A_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.9
                if r_i * (ppC - 1) > 0:
                    A_i[r_i:, :r_i * (ppC - 1)] = torch.eye(r_i * (ppC - 1), device=device, dtype=dtype)
            
            # Initialize Q_i (process noise covariance) for this block
            Q_i = torch.zeros(r_i * ppC, r_i * ppC, device=device, dtype=dtype)
            if T > p:
                # Compute VAR residuals: e = z - Z*A_temp
                if Z.shape[1] > 0:
                    try:
                        e = z[p:, :] - (Z[p:, :] @ A_i[:r_i, :r_i * p].T)
                        if e.shape[0] > 1:
                            # Handle single factor case (torch.cov returns 0-D tensor for single variable)
                            if e.shape[1] == 1:
                                var_val = torch.var(e, dim=0, unbiased=False)
                                Q_i[:r_i, :r_i] = var_val.unsqueeze(0).unsqueeze(0)  # (1, 1)
                            else:
                                Q_i[:r_i, :r_i] = torch.cov(e.T)
                                Q_i[:r_i, :r_i] = (Q_i[:r_i, :r_i] + Q_i[:r_i, :r_i].T) / 2
                        else:
                            Q_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.1
                    except (RuntimeError, ValueError):
                        Q_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.1
                else:
                    Q_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.1
            else:
                Q_i[:r_i, :r_i] = torch.eye(r_i, device=device, dtype=dtype) * 0.1
            
            # Ensure Q_i is positive definite
            # Ensure Q_i is positive definite with robust eigenvalue computation
            try:
                # Ensure Q_i is positive definite with robust eigenvalue computation
                try:
                    eigenvals_Qi = torch.linalg.eigvalsh(Q_i[:r_i, :r_i])
                    min_eigenval = torch.min(eigenvals_Qi)
                    if min_eigenval < 1e-8:
                        Q_i[:r_i, :r_i] = Q_i[:r_i, :r_i] + torch.eye(r_i, device=device, dtype=dtype) * (1e-8 - min_eigenval)
                except (RuntimeError, ValueError) as e:
                    # eigvalsh failed - apply stronger regularization
                    _logger.warning(f"eigvalsh failed for Q_i block in initialization: {e}. Applying stronger regularization.")
                    Q_i[:r_i, :r_i] = Q_i[:r_i, :r_i] + torch.eye(r_i, device=device, dtype=dtype) * 1e-6
                    try:
                        eigenvals_Qi = torch.linalg.eigvalsh(Q_i[:r_i, :r_i])
                        min_eigenval = torch.min(eigenvals_Qi)
                        if min_eigenval < 1e-8:
                            Q_i[:r_i, :r_i] = Q_i[:r_i, :r_i] + torch.eye(r_i, device=device, dtype=dtype) * (1e-8 - min_eigenval)
                    except (RuntimeError, ValueError):
                        # Still failing - use diagonal matrix
                        diag_Qi = torch.diag(Q_i[:r_i, :r_i])
                        diag_Qi = torch.maximum(diag_Qi, torch.ones_like(diag_Qi) * 1e-6)
                        Q_i[:r_i, :r_i] = torch.diag(diag_Qi)
            except (RuntimeError, ValueError) as e:
                # eigvalsh failed - apply stronger regularization
                _logger.warning(f"eigvalsh failed for Q_i block: {e}. Applying stronger regularization.")
                Q_i[:r_i, :r_i] = Q_i[:r_i, :r_i] + torch.eye(r_i, device=device, dtype=dtype) * 1e-6
                try:
                    eigenvals_Qi = torch.linalg.eigvalsh(Q_i[:r_i, :r_i])
                    min_eigenval = torch.min(eigenvals_Qi)
                    if min_eigenval < 1e-8:
                        Q_i[:r_i, :r_i] = Q_i[:r_i, :r_i] + torch.eye(r_i, device=device, dtype=dtype) * (1e-8 - min_eigenval)
                except (RuntimeError, ValueError):
                    # Still failing - use diagonal matrix
                    diag_Qi = torch.diag(Q_i[:r_i, :r_i])
                    diag_Qi = torch.maximum(diag_Qi, torch.ones_like(diag_Qi) * 1e-6)
                    Q_i[:r_i, :r_i] = torch.diag(diag_Qi)
            
            # Initial covariance for this block: initV_i = inv(eye - kron(A_i, A_i)) * Q_i(:)
            try:
                A_i_block = A_i[:r_i * ppC, :r_i * ppC]
                kron_AA = torch.kron(A_i_block, A_i_block)
                eye_kron = torch.eye((r_i * ppC) ** 2, device=device, dtype=dtype)
                initV_i_flat = torch.linalg.solve(
                    eye_kron - kron_AA + torch.eye((r_i * ppC) ** 2, device=device, dtype=dtype) * 1e-6,
                    Q_i[:r_i * ppC, :r_i * ppC].flatten()
                )
                initV_i = initV_i_flat.reshape(r_i * ppC, r_i * ppC)
            except (RuntimeError, ValueError):
                initV_i = Q_i[:r_i * ppC, :r_i * ppC].clone()
            
            # Accumulate block matrices
            # Each block can have different sizes (r_i * ppC), which is fine for block_diag
            # Just ensure each matrix is square within its block
            block_size = r_i * ppC
            _logger.info(f"Block {i}: Creating matrices of size {block_size}x{block_size}. A_i shape={A_i.shape}, Q_i shape={Q_i.shape}, initV_i shape={initV_i.shape}")
            
            # Extract or create square matrices of correct size
            if A_i.shape[0] >= block_size and A_i.shape[1] >= block_size:
                A_i_final = A_i[:block_size, :block_size]
            else:
                _logger.warning(f"Block {i}: A_i too small: {A_i.shape}, creating {block_size}x{block_size} matrix")
                A_i_final = torch.zeros(block_size, block_size, device=device, dtype=dtype)
                min_rows = min(A_i.shape[0], block_size)
                min_cols = min(A_i.shape[1], block_size)
                A_i_final[:min_rows, :min_cols] = A_i[:min_rows, :min_cols]
            
            if Q_i.shape[0] >= block_size and Q_i.shape[1] >= block_size:
                Q_i_final = Q_i[:block_size, :block_size]
            else:
                _logger.warning(f"Block {i}: Q_i too small: {Q_i.shape}, creating {block_size}x{block_size} matrix")
                Q_i_final = torch.zeros(block_size, block_size, device=device, dtype=dtype)
                min_rows = min(Q_i.shape[0], block_size)
                min_cols = min(Q_i.shape[1], block_size)
                Q_i_final[:min_rows, :min_cols] = Q_i[:min_rows, :min_cols]
            
            if initV_i.shape[0] >= block_size and initV_i.shape[1] >= block_size:
                V_0_i_final = initV_i[:block_size, :block_size]
            else:
                _logger.warning(f"Block {i}: initV_i too small: {initV_i.shape}, creating {block_size}x{block_size} matrix")
                V_0_i_final = torch.zeros(block_size, block_size, device=device, dtype=dtype)
                min_rows = min(initV_i.shape[0], block_size)
                min_cols = min(initV_i.shape[1], block_size)
                V_0_i_final[:min_rows, :min_cols] = initV_i[:min_rows, :min_cols]
            
            # Verify all matrices are square and correct size
            assert A_i_final.shape == (block_size, block_size), f"Block {i}: A_i_final shape {A_i_final.shape} != ({block_size}, {block_size})"
            assert Q_i_final.shape == (block_size, block_size), f"Block {i}: Q_i_final shape {Q_i_final.shape} != ({block_size}, {block_size})"
            assert V_0_i_final.shape == (block_size, block_size), f"Block {i}: V_0_i_final shape {V_0_i_final.shape} != ({block_size}, {block_size})"
            
            # Log dimensions for debugging
            _logger.info(f"Block {i}: Final matrices - A={A_i_final.shape}, Q={Q_i_final.shape}, V_0={V_0_i_final.shape}")
            
            A_list.append(A_i_final)
            Q_list.append(Q_i_final)
            V_0_list.append(V_0_i_final)
        
        # Concatenate C matrices
        C = torch.cat(C_list, dim=1) if C_list else torch.zeros(N, 0, device=device, dtype=dtype)
        
        # Build block-diagonal A, Q, V_0
        if A_list:
            # Log dimensions before block_diag for debugging
            _logger.info(f"Building block-diagonal matrices from {len(A_list)} blocks")
            for i, (a, q, v) in enumerate(zip(A_list, Q_list, V_0_list)):
                _logger.info(f"  Block {i}: A={a.shape}, Q={q.shape}, V_0={v.shape}")
                # Ensure all matrices are square
                if a.shape[0] != a.shape[1]:
                    _logger.error(f"Block {i}: A is not square: {a.shape}")
                if q.shape[0] != q.shape[1]:
                    _logger.error(f"Block {i}: Q is not square: {q.shape}")
                if v.shape[0] != v.shape[1]:
                    _logger.error(f"Block {i}: V_0 is not square: {v.shape}")
            
            try:
                A_factors = torch.block_diag(*A_list)
                Q_factors = torch.block_diag(*Q_list)
                V_0_factors = torch.block_diag(*V_0_list)
                _logger.info(f"Successfully created block-diagonal matrices: A_factors={A_factors.shape}, Q_factors={Q_factors.shape}, V_0_factors={V_0_factors.shape}")
            except RuntimeError as e:
                # Log dimension information for debugging
                _logger.error(f"block_diag failed for A_list/Q_list/V_0_list: {e}")
                for i, (a, q, v) in enumerate(zip(A_list, Q_list, V_0_list)):
                    _logger.error(f"Block {i}: A shape={a.shape}, Q shape={q.shape}, V_0 shape={v.shape}")
                raise
        else:
            A_factors = torch.zeros(0, 0, device=device, dtype=dtype)
            Q_factors = torch.zeros(0, 0, device=device, dtype=dtype)
            V_0_factors = torch.zeros(0, 0, device=device, dtype=dtype)
        
        # === IDIOSYNCRATIC COMPONENTS ===
        # Add identity matrix for monthly idiosyncratic series
        if i_idio is not None:
            eyeN = torch.eye(N, device=device, dtype=dtype)
            # Remove columns for non-idiosyncratic series
            i_idio_bool = i_idio.bool()
            eyeN_idio = eyeN[:, i_idio_bool]  # N x n_idio
            C = torch.cat([C, eyeN_idio], dim=1)
        else:
            # Default: all monthly series have idiosyncratic components
            eyeN_monthly = torch.eye(N, device=device, dtype=dtype)[:, :nM] if nM > 0 else torch.zeros(N, 0, device=device, dtype=dtype)
            C = torch.cat([C, eyeN_monthly], dim=1)
        
        # Add quarterly idiosyncratic chains (5-state: [1, 2, 3, 2, 1])
        if nQ > 0:
            # Quarterly tent weights: [1, 2, 3, 2, 1]
            tent_q = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=device, dtype=dtype)
            # C_quarterly = [zeros(nM, 5*nQ); kron(eye(nQ), tent_q)]
            C_quarterly = torch.zeros(N, 5 * nQ, device=device, dtype=dtype)
            C_quarterly[nM:, :] = torch.kron(torch.eye(nQ, device=device, dtype=dtype), tent_q.unsqueeze(0))
            C = torch.cat([C, C_quarterly], dim=1)
        
        # Initialize R (observation noise covariance) from final residuals
        # Ensure resNaN is 2D: (T, N)
        if resNaN.ndim > 2:
            _logger.warning(f"resNaN has unexpected shape: {resNaN.shape}, reshaping to 2D...")
            resNaN = resNaN.reshape(-1, resNaN.shape[-1])
        elif resNaN.ndim == 1:
            _logger.warning(f"resNaN is 1D: {resNaN.shape}, reshaping to 2D...")
            resNaN = resNaN.unsqueeze(0)
        
        # Check degrees of freedom before computing variance
        T_res, N_res = resNaN.shape
        if T_res <= 1:
            # Not enough data for variance calculation, use fallback
            _logger.warning(f"resNaN has T={T_res} <= 1, using fallback variance values")
            var_res = torch.ones(N_res, device=device, dtype=dtype) * 1e-4
        else:
            # Count valid (non-NaN) values per column
            valid_counts = torch.sum(torch.isfinite(resNaN), dim=0)
            # Compute variance only for columns with at least 2 valid values
            var_res = torch.zeros(N_res, device=device, dtype=dtype)
            for i in range(N_res):
                if valid_counts[i] > 1:
                    col_data = resNaN[:, i]
                    col_valid = col_data[torch.isfinite(col_data)]
                    if len(col_valid) > 1:
                        var_res[i] = torch.var(col_valid, unbiased=False)
                    else:
                        var_res[i] = 1e-4
                else:
                    var_res[i] = 1e-4
            
            # Handle any remaining NaN/Inf values
            var_res = torch.where(torch.isfinite(var_res), var_res, torch.tensor(1e-4, device=device, dtype=dtype))
        
        # Ensure var_res is 1D
        if var_res.ndim > 1:
            _logger.warning(f"var_res has unexpected shape: {var_res.shape}, flattening...")
            var_res = var_res.flatten()
        elif var_res.ndim == 0:
            # Single value, expand to (N,)
            var_res = var_res.unsqueeze(0)
        
        R = torch.diag(var_res)  # (N, N)
        R = torch.where(torch.isfinite(R), R, torch.tensor(1e-4, device=device, dtype=dtype))
        
        # Set monthly idiosyncratic variances to 1e-4 (MATLAB: R(ii_idio(i),ii_idio(i)) = 1e-04)
        if i_idio is not None:
            i_idio_indices = torch.where(i_idio > 0)[0]
            for idx in i_idio_indices:
                R[idx, idx] = 1e-4
        else:
            # Default: all monthly series
            for idx in range(nM):
                R[idx, idx] = 1e-4
        
        # Set quarterly variances: Rdiag(nM+1:N) = 1e-04
        for idx in range(nM, N):
            R[idx, idx] = 1e-4
        
        # === IDIOSYNCRATIC TRANSITION MATRICES ===
        # Monthly: AR(1) for each series
        n_idio_M = nM if i_idio is None else int(torch.sum(i_idio).item())
        BM = torch.zeros(n_idio_M, n_idio_M, device=device, dtype=dtype)
        SM = torch.zeros(n_idio_M, n_idio_M, device=device, dtype=dtype)
        
        if i_idio is not None:
            ii_idio = torch.where(i_idio > 0)[0]
        else:
            ii_idio = torch.arange(nM, device=device, dtype=torch.long)
        
        for i, idx in enumerate(ii_idio):
            res_i = resNaN[:, idx]
            # Find leading and trailing NaNs
            non_nan_mask = ~torch.isnan(res_i)
            if torch.sum(non_nan_mask) > 1:
                first_non_nan = torch.where(non_nan_mask)[0][0] if torch.any(non_nan_mask) else 0
                last_non_nan = torch.where(non_nan_mask)[0][-1] if torch.any(non_nan_mask) else T - 1
                res_i_clean = res[first_non_nan:last_non_nan + 1, idx]
                
                if len(res_i_clean) > 1:
                    # AR(1): res_i(t) = BM * res_i(t-1) + error
                    y_ar = res_i_clean[1:]
                    x_ar = res_i_clean[:-1].unsqueeze(1)
                    try:
                        BM[i, i] = torch.linalg.solve(
                            x_ar.T @ x_ar + torch.eye(1, device=device, dtype=dtype) * 1e-6,
                            x_ar.T @ y_ar
                        ).item()
                        # Residual covariance
                        residuals_ar = y_ar - x_ar.squeeze() * BM[i, i]
                        if len(residuals_ar) > 1:
                            SM[i, i] = torch.var(residuals_ar, unbiased=False).item()
                        else:
                            SM[i, i] = 0.1
                    except (RuntimeError, ValueError):
                        BM[i, i] = 0.1
                        SM[i, i] = 0.1
                else:
                    BM[i, i] = 0.1
                    SM[i, i] = 0.1
            else:
                BM[i, i] = 0.1
                SM[i, i] = 0.1
        
        # Quarterly: 5-state chain with rho0 = 0.1
        rho0 = 0.1
        if nQ > 0:
            # sig_e = Rdiag(nM+1:N)/19 (MATLAB approximation)
            sig_e = R[nM:, nM:].diag() / 19.0
            sig_e = torch.where(torch.isfinite(sig_e), sig_e, torch.tensor(1e-4, device=device, dtype=dtype))
            
            # temp = zeros(5); temp(1,1) = 1
            temp = torch.zeros(5, 5, device=device, dtype=dtype)
            temp[0, 0] = 1.0
            
            # SQ = kron(diag((1-rho0^2)*sig_e), temp)
            SQ = torch.kron(torch.diag((1 - rho0 ** 2) * sig_e), temp)
            
            # BQ = kron(eye(nQ), [[rho0 zeros(1,4)];[eye(4),zeros(4,1)]])
            BQ_block = torch.zeros(5, 5, device=device, dtype=dtype)
            BQ_block[0, 0] = rho0
            BQ_block[1:, :4] = torch.eye(4, device=device, dtype=dtype)
            BQ = torch.kron(torch.eye(nQ, device=device, dtype=dtype), BQ_block)
            
            # initViQ = reshape(inv(eye - kron(BQ,BQ))*SQ(:), 5*nQ, 5*nQ)
            try:
                kron_BQBQ = torch.kron(BQ, BQ)
                eye_kron = torch.eye((5 * nQ) ** 2, device=device, dtype=dtype)
                initViQ_flat = torch.linalg.solve(
                    eye_kron - kron_BQBQ + torch.eye((5 * nQ) ** 2, device=device, dtype=dtype) * 1e-6,
                    SQ.flatten()
                )
                initViQ = initViQ_flat.reshape(5 * nQ, 5 * nQ)
            except (RuntimeError, ValueError):
                initViQ = SQ.clone()
        else:
            BQ = torch.zeros(0, 0, device=device, dtype=dtype)
            SQ = torch.zeros(0, 0, device=device, dtype=dtype)
            initViQ = torch.zeros(0, 0, device=device, dtype=dtype)
        
        # Monthly initial covariance: initViM = diag(1./diag(eye - BM.^2)).*SM
        try:
            eye_BM = torch.eye(n_idio_M, device=device, dtype=dtype)
            BM_sq = BM ** 2
            diag_inv = 1.0 / torch.diag(eye_BM - BM_sq)
            diag_inv = torch.where(torch.isfinite(diag_inv), diag_inv, torch.ones_like(diag_inv))
            initViM = torch.diag(diag_inv) @ SM
        except (RuntimeError, ValueError):
            initViM = SM.clone()
        
        # Combine all transition matrices: A = blkdiag(A_factors, BM, BQ)
        # Log dimensions before block_diag for debugging
        _logger.debug(f"Before final block_diag:")
        _logger.debug(f"  A_factors shape: {A_factors.shape if A_factors.numel() > 0 else 'empty'}")
        _logger.debug(f"  BM shape: {BM.shape if BM.numel() > 0 else 'empty'}, n_idio_M={n_idio_M}")
        _logger.debug(f"  BQ shape: {BQ.shape if BQ.numel() > 0 else 'empty'}, nQ={nQ}")
        _logger.debug(f"  Q_factors shape: {Q_factors.shape if Q_factors.numel() > 0 else 'empty'}")
        _logger.debug(f"  SM shape: {SM.shape if SM.numel() > 0 else 'empty'}")
        _logger.debug(f"  SQ shape: {SQ.shape if SQ.numel() > 0 else 'empty'}")
        _logger.debug(f"  V_0_factors shape: {V_0_factors.shape if V_0_factors.numel() > 0 else 'empty'}")
        _logger.debug(f"  initViM shape: {initViM.shape if initViM.numel() > 0 else 'empty'}")
        _logger.debug(f"  initViQ shape: {initViQ.shape if initViQ.numel() > 0 else 'empty'}")
        
        # Check if BM and SM have correct dimensions
        if BM.shape[0] != n_idio_M or BM.shape[1] != n_idio_M:
            _logger.warning(f"BM shape mismatch: expected ({n_idio_M}, {n_idio_M}), got {BM.shape}. Resizing...")
            BM_new = torch.zeros(n_idio_M, n_idio_M, device=device, dtype=dtype)
            min_dim = min(BM.shape[0], n_idio_M, BM.shape[1], n_idio_M)
            BM_new[:min_dim, :min_dim] = BM[:min_dim, :min_dim]
            BM = BM_new
        if SM.shape[0] != n_idio_M or SM.shape[1] != n_idio_M:
            _logger.warning(f"SM shape mismatch: expected ({n_idio_M}, {n_idio_M}), got {SM.shape}. Resizing...")
            SM_new = torch.zeros(n_idio_M, n_idio_M, device=device, dtype=dtype)
            min_dim = min(SM.shape[0], n_idio_M, SM.shape[1], n_idio_M)
            SM_new[:min_dim, :min_dim] = SM[:min_dim, :min_dim]
            SM = SM_new
        if initViM.shape[0] != n_idio_M or initViM.shape[1] != n_idio_M:
            _logger.warning(f"initViM shape mismatch: expected ({n_idio_M}, {n_idio_M}), got {initViM.shape}. Resizing...")
            initViM_new = torch.zeros(n_idio_M, n_idio_M, device=device, dtype=dtype)
            min_dim = min(initViM.shape[0], n_idio_M, initViM.shape[1], n_idio_M)
            initViM_new[:min_dim, :min_dim] = initViM[:min_dim, :min_dim]
            initViM = initViM_new
        
        # Check dimensions before block_diag to debug size mismatches
        try:
            A = torch.block_diag(A_factors, BM, BQ)
            Q = torch.block_diag(Q_factors, SM, SQ)
            V_0 = torch.block_diag(V_0_factors, initViM, initViQ)
            _logger.debug(f"Successfully created final block-diagonal matrices: A={A.shape}, Q={Q.shape}, V_0={V_0.shape}")
        except RuntimeError as e:
            # Log dimension information for debugging
            _logger.error(f"block_diag failed: {e}")
            _logger.error(f"A_factors shape: {A_factors.shape if A_factors.numel() > 0 else 'empty'}")
            _logger.error(f"BM shape: {BM.shape if BM.numel() > 0 else 'empty'}")
            _logger.error(f"BQ shape: {BQ.shape if BQ.numel() > 0 else 'empty'}")
            _logger.error(f"Q_factors shape: {Q_factors.shape if Q_factors.numel() > 0 else 'empty'}")
            _logger.error(f"SM shape: {SM.shape if SM.numel() > 0 else 'empty'}")
            _logger.error(f"SQ shape: {SQ.shape if SQ.numel() > 0 else 'empty'}")
            _logger.error(f"V_0_factors shape: {V_0_factors.shape if V_0_factors.numel() > 0 else 'empty'}")
            _logger.error(f"initViM shape: {initViM.shape if initViM.numel() > 0 else 'empty'}")
            _logger.error(f"initViQ shape: {initViQ.shape if initViQ.numel() > 0 else 'empty'}")
            raise
        
        # Initial state: Z_0 = zeros
        m = A.shape[0]
        Z_0 = torch.zeros(m, device=device, dtype=dtype)
        
        # Ensure all matrices are positive definite
        # Ensure V_0 is positive definite with robust eigenvalue computation
        try:
            eigenvals_V0 = torch.linalg.eigvalsh(V_0)
            min_eigenval = torch.min(eigenvals_V0)
            if min_eigenval < 1e-8:
                V_0 = V_0 + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
        except (RuntimeError, ValueError) as e:
            # eigvalsh failed - apply stronger regularization
            _logger.warning(f"eigvalsh failed for V_0 in initialization: {e}. Applying stronger regularization.")
            V_0 = V_0 + torch.eye(m, device=device, dtype=dtype) * 1e-6
            try:
                eigenvals_V0 = torch.linalg.eigvalsh(V_0)
                min_eigenval = torch.min(eigenvals_V0)
                if min_eigenval < 1e-8:
                    V_0 = V_0 + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
            except (RuntimeError, ValueError):
                # Still failing - use diagonal matrix
                diag_V0 = torch.diag(V_0)
                diag_V0 = torch.maximum(diag_V0, torch.ones_like(diag_V0) * 1e-6)
                V_0 = torch.diag(diag_V0)
        
        return A, C, Q, R, Z_0, V_0
    
    def check_convergence(
        self,
        loglik: float,
        previous_loglik: float,
        threshold: float,
        verbose: bool = False
    ) -> Tuple[bool, float]:
        """Check EM convergence.
        
        Parameters
        ----------
        loglik : float
            Current log-likelihood value
        previous_loglik : float
            Previous log-likelihood value
        threshold : float
            Convergence threshold (typically 1e-4 to 1e-5)
        verbose : bool
            Whether to log convergence status
            
        Returns
        -------
        converged : bool
            Whether convergence was achieved
        change : float
            Relative change in log-likelihood
        """
        if previous_loglik == float('-inf'):
            return False, 0.0
        
        if abs(previous_loglik) < 1e-10:
            # Previous loglik is essentially zero, use absolute change
            change = abs(loglik - previous_loglik)
        else:
            # Relative change
            change = abs((loglik - previous_loglik) / previous_loglik)
        
        converged = change < threshold
        
        if verbose and converged:
            _logger.info(f'EM algorithm converged: loglik change = {change:.2e} < {threshold:.2e}')
        
        return converged, change

