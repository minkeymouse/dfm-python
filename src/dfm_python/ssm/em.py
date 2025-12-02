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
    All matrix inversions and solves use regularization (1e-6) to prevent
    singular matrix errors. This is critical for GPU stability, as PyTorch
    can throw RuntimeError for near-singular matrices.

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
                # Regularization prevents singular matrix errors (critical for GPU stability)
                reg_scale = self.regularization_scale.item()
                XTX_A_reg = XTX_A + torch.eye(m, device=device, dtype=dtype) * reg_scale
                A_new = torch.linalg.solve(XTX_A_reg, XTY_A).T
                
                # Ensure stability
                eigenvals_A = torch.linalg.eigvals(A_new)
                max_eigenval = torch.max(torch.abs(eigenvals_A))
                if max_eigenval >= 0.99:
                    A_new = A_new * (0.99 / max_eigenval)
            except (RuntimeError, ValueError):
                A_new = params.A.clone()
        else:
            A_new = params.A.clone()
        
        # Update C (observation matrix): regression of y_t on Z_t
        # C = (sum_t y_t E[Z_t^T]) (sum_t E[Z_t Z_t^T])^{-1}
        try:
            # Compute sum_yEZ = sum_t y_t E[Z_t^T] (vectorized: batch outer products)
            # params.y is (N, T), EZ is (T, m)
            # Transpose y to (T, N) for batch operations
            sum_yEZ = torch.sum(params.y.T[:, :, None] * EZ[:, None, :], dim=0)  # (N, m)
            # Compute sum_EZZ = sum_t E[Z_t Z_t^T]
            sum_EZZ = torch.sum(EZZ, dim=0)
            # Regularization prevents singular matrix errors (critical for GPU stability)
            reg_scale = self.regularization_scale.item()
            sum_EZZ_reg = sum_EZZ + torch.eye(m, device=device, dtype=dtype) * reg_scale
            C_new = torch.linalg.solve(sum_EZZ_reg.T, sum_yEZ.T).T
            
            # Normalize C columns (factor loadings)
            for j in range(m):
                norm = torch.linalg.norm(C_new[:, j])
                if norm > 1e-8:
                    C_new[:, j] = C_new[:, j] / norm
        except (RuntimeError, ValueError):
            C_new = params.C.clone()
        
        # Update Q (process noise covariance): residual covariance from transition
        if T > 1:
            # Vectorized: residuals_Q = EZ[1:] - (A_new @ EZ[:-1].T).T
            residuals_Q = EZ[1:, :] - (A_new @ EZ[:-1, :].T).T
            Q_new = torch.cov(residuals_Q.T)
            Q_new = (Q_new + Q_new.T) / 2
            # Ensure positive definite
            eigenvals_Q = torch.linalg.eigvalsh(Q_new)
            min_eigenval = torch.min(eigenvals_Q)
            if min_eigenval < 1e-8:
                Q_new = Q_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
            # Floor for Q
            Q_new = torch.maximum(Q_new, torch.eye(m, device=device, dtype=dtype) * 0.01)
        else:
            Q_new = params.Q.clone()
        
        # Update R (observation covariance): residual covariance from observation
        # Vectorized: residuals_R = params.y.T - (C_new @ EZ.T).T
        # params.y is (N, T), EZ is (T, m), C_new is (N, m)
        residuals_R = params.y.T - (C_new @ EZ.T).T  # (T, N)
        R_new = torch.cov(residuals_R.T)
        R_new = (R_new + R_new.T) / 2
        
        # Ensure R is diagonal (idiosyncratic variances only)
        R_new = torch.diag(torch.diag(R_new))
        
        # Ensure positive definite
        R_new = torch.maximum(R_new, torch.eye(N, device=device, dtype=dtype) * 1e-8)
        
        # Update Z_0 and V_0 (use first smoothed state)
        Z_0_new = Zsmooth[0, :]  # Initial state
        V_0_new = Vsmooth[:, :, 0]  # Initial covariance
        
        # Ensure V_0 is positive definite
        eigenvals_V0 = torch.linalg.eigvalsh(V_0_new)
        min_eigenval = torch.min(eigenvals_V0)
        if min_eigenval < 1e-8:
            V_0_new = V_0_new + torch.eye(m, device=device, dtype=dtype) * (1e-8 - min_eigenval)
        
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
        """Initialize DFM parameters using PCA and OLS.
        
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
        total_factors = int(torch.sum(r).item())
        
        # Handle missing data for initialization using GPU-accelerated PyTorch version
        from ..utils.data import rem_nans_spline_torch
        x_clean, _ = rem_nans_spline_torch(x, method=opt_nan.get('method', 2), k=opt_nan.get('k', 3))
        
        # Compute covariance matrix
        # Remove any remaining NaN/inf
        x_clean = torch.where(torch.isfinite(x_clean), x_clean, torch.tensor(0.0, device=device, dtype=dtype))
        
        # Compute covariance: cov expects (N, T) format
        x_clean_T = x_clean.T  # (N, T)
        cov_matrix = torch.cov(x_clean_T)
        
        # Ensure covariance is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        eigenvals = torch.linalg.eigvalsh(cov_matrix)
        if torch.any(eigenvals < 0):
            # Add small regularization
            cov_matrix = cov_matrix + torch.eye(N, device=device, dtype=dtype) * 1e-8
        
        # Initialize C (loading matrix) via PCA
        # Extract first total_factors principal components
        from ..encoder.pca import compute_principal_components_torch
        eigenvalues, eigenvectors = compute_principal_components_torch(cov_matrix, total_factors)
        C = eigenvectors  # N x total_factors
        
        # Normalize C columns (factor loadings)
        for j in range(total_factors):
            norm = torch.linalg.norm(C[:, j])
            if norm > 1e-8:
                C[:, j] = C[:, j] / norm
        
        # Extract initial factors via projection
        factors_init = x_clean @ C  # T x total_factors
        
        # Initialize A (transition matrix) via OLS
        # For AR(p): f_t = A_1 f_{t-1} + ... + A_p f_{t-p}
        if T > p:
            # Prepare data for OLS
            Y = factors_init[p:, :]  # (T-p) x total_factors
            X_list = []
            for lag in range(1, p + 1):
                X_list.append(factors_init[p - lag:-lag, :])
            X = torch.cat(X_list, dim=1)  # (T-p) x (p * total_factors)
            
            # OLS: A = (X'X)^{-1} X'Y
            # Use regularized solve to prevent singular matrix errors on GPU
            try:
                XTX = X.T @ X
                # Regularization prevents RuntimeError: cholesky_cpu: U(0,0) is zero
                reg_scale = self.regularization_scale.item()
                XTX_reg = XTX + torch.eye(XTX.shape[0], device=device, dtype=dtype) * reg_scale
                A_flat = torch.linalg.solve(XTX_reg, X.T @ Y).T  # total_factors x (p * total_factors)
                
                # Reshape to VAR(p) format
                if p == 1:
                    A = A_flat  # total_factors x total_factors
                else:
                    # For VAR(p), A is block matrix [A_1, A_2, ..., A_p]
                    A = A_flat  # Keep as is for now
            except (RuntimeError, ValueError):
                # Fallback: use identity for AR(1) part, zeros for higher lags
                if p == 1:
                    A = torch.eye(total_factors, device=device, dtype=dtype) * 0.9  # Slightly less than 1 for stability
                else:
                    A = torch.cat([torch.eye(total_factors, device=device, dtype=dtype) * 0.9] + 
                                  [torch.zeros((total_factors, total_factors), device=device, dtype=dtype)] * (p - 1), dim=1)
        else:
            # Not enough data, use identity
            if p == 1:
                A = torch.eye(total_factors, device=device, dtype=dtype) * 0.9
            else:
                A = torch.cat([torch.eye(total_factors, device=device, dtype=dtype) * 0.9] + 
                              [torch.zeros((total_factors, total_factors), device=device, dtype=dtype)] * (p - 1), dim=1)
        
        # Ensure stability: clip eigenvalues
        if p == 1:
            eigenvals_A = torch.linalg.eigvals(A)
            max_eigenval = torch.max(torch.abs(eigenvals_A))
            if max_eigenval >= 0.99:
                A = A * (0.99 / max_eigenval)
        
        # Initialize Q (process noise covariance) from factor residuals
        if T > p:
            if p == 1:
                residuals_f = Y - X @ A.T
            else:
                # For VAR(p), need to handle block structure
                A1 = A[:, :total_factors]  # First block
                residuals_f = Y - X[:, :total_factors] @ A1.T
            Q = torch.cov(residuals_f.T)
            Q = (Q + Q.T) / 2  # Symmetrize
            # Ensure positive definite
            eigenvals_Q = torch.linalg.eigvalsh(Q)
            min_eigenval = torch.min(eigenvals_Q)
            if min_eigenval < 1e-8:
                Q = Q + torch.eye(total_factors, device=device, dtype=dtype) * (1e-8 - min_eigenval)
            # Floor for Q
            Q = torch.maximum(Q, torch.eye(total_factors, device=device, dtype=dtype) * 0.01)
        else:
            Q = torch.eye(total_factors, device=device, dtype=dtype) * 0.1
        
        # Initialize R (observation noise covariance) from observation residuals
        reconstructed = factors_init @ C.T  # T x N
        residuals_obs = x_clean - reconstructed
        R = torch.cov(residuals_obs.T)
        R = (R + R.T) / 2  # Symmetrize
        
        # Ensure R is diagonal (idiosyncratic variances only)
        R = torch.diag(torch.diag(R))
        
        # Ensure positive definite
        R = torch.maximum(R, torch.eye(N, device=device, dtype=dtype) * 1e-8)
        
        # Initialize Z_0 (initial state) and V_0 (initial covariance)
        if T > 0:
            Z_0 = factors_init[0, :]  # Use first period factors
        else:
            Z_0 = torch.zeros(total_factors, device=device, dtype=dtype)
        
        # V_0: initial uncertainty (use factor covariance)
        V_0 = Q.clone()
        # Ensure positive definite
        eigenvals_V0 = torch.linalg.eigvalsh(V_0)
        min_eigenval = torch.min(eigenvals_V0)
        if min_eigenval < 1e-8:
            V_0 = V_0 + torch.eye(total_factors, device=device, dtype=dtype) * (1e-8 - min_eigenval)
        
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

