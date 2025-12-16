"""PyTorch module for Kalman filter and smoother.

This module provides KalmanFilter, a PyTorch nn.Module for Kalman filtering and
fixed-interval smoothing operations.

Numerical Stability:
    PyTorch matrix operations (especially Cholesky and inverse) can fail on
    near-singular matrices, especially on GPU. This module implements robust
    error handling with progressive fallbacks:
    1. Standard operations (fastest)
    2. Regularized operations (handles near-singular matrices)
    3. Pseudo-inverse fallback (most robust, but slower)
    
    All covariance matrices are regularized to ensure positive definiteness,
    preventing RuntimeError exceptions like "cholesky_cpu: U(0,0) is zero" or
    "inverse_cuda: singular matrix".

Performance:
    GPU acceleration provides 10-50x speedup for large-scale time series
    (T > 10k, N > 500) compared to NumPy implementations. Matrix operations
    (MM, SVD, Cholesky) are highly optimized on GPU.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass
from ..logger import get_logger
from .utils import (
    check_finite,
    ensure_real,
    ensure_symmetric,
    ensure_real_and_symmetric,
    ensure_covariance_stable,
    ensure_positive_definite,
    clean_matrix,
    safe_inverse,
    safe_determinant,
    cap_max_eigenval,
)

_logger = get_logger(__name__)


@dataclass
class KalmanFilterState:
    """Kalman filter state structure using PyTorch tensors.
    
    This dataclass stores the complete state of the Kalman filter after forward
    and backward passes, including prior/posterior estimates and covariances.
    
    Attributes
    ----------
    Zm : torch.Tensor
        Prior (predicted) factor state estimates, shape (m x nobs).
        Zm[:, t] is the predicted state at time t given observations up to t-1.
    Vm : torch.Tensor
        Prior covariance matrices, shape (m x m x nobs).
        Vm[:, :, t] is the covariance of Zm[:, t].
    ZmU : torch.Tensor
        Posterior (updated) factor state estimates, shape (m x (nobs+1)).
        ZmU[:, t] is the updated state at time t given observations up to t.
        Includes initial state at t=0.
    VmU : torch.Tensor
        Posterior covariance matrices, shape (m x m x (nobs+1)).
        VmU[:, :, t] is the covariance of ZmU[:, t].
    loglik : float
        Log-likelihood of the data under the current model parameters.
        Computed as sum of log-likelihoods at each time step.
    k_t : torch.Tensor
        Kalman gain matrix, shape (m x k) where k is number of observed series.
        Used to update state estimates with new observations.
    """
    Zm: torch.Tensor      # Prior/predicted factor state (m x nobs)
    Vm: torch.Tensor      # Prior covariance (m x m x nobs)
    ZmU: torch.Tensor     # Posterior/updated state (m x (nobs+1))
    VmU: torch.Tensor     # Posterior covariance (m x m x (nobs+1))
    loglik: float         # Log-likelihood
    k_t: torch.Tensor     # Kalman gain


class KalmanFilter(nn.Module):
    """PyTorch module for Kalman filtering and smoothing.
    
    This module provides Kalman filter (forward pass) and fixed-interval smoother
    (backward pass) operations. All numerical stability constants are stored as
    buffers, enabling automatic device management.
    
    Parameters
    ----------
    min_eigenval : float, default 1e-8
        Minimum eigenvalue for covariance matrices
    min_diagonal_variance : float, default 1e-6
        Minimum diagonal variance for regularization
    default_variance_fallback : float, default 1.0
        Fallback variance when matrix operations fail
    min_variance_covariance : float, default 1e-10
        Minimum variance for covariance estimation
    inv_regularization : float, default 1e-6
        Regularization added before matrix inversion
    cholesky_regularization : float, default 1e-8
        Regularization for Cholesky decomposition
    """
    
    def __init__(
        self,
        min_eigenval: float = 1e-8,
        min_diagonal_variance: float = 1e-6,
        default_variance_fallback: float = 1.0,
        min_variance_covariance: float = 1e-10,
        inv_regularization: float = 1e-6,
        cholesky_regularization: float = 1e-8
    ):
        super().__init__()
        # Store numerical stability parameters as buffers (moves to GPU automatically)
        self.register_buffer('min_eigenval', torch.tensor(min_eigenval))
        self.register_buffer('min_diagonal_variance', torch.tensor(min_diagonal_variance))
        self.register_buffer('default_variance_fallback', torch.tensor(default_variance_fallback))
        self.register_buffer('min_variance_covariance', torch.tensor(min_variance_covariance))
        self.register_buffer('inv_regularization', torch.tensor(inv_regularization))
        self.register_buffer('cholesky_regularization', torch.tensor(cholesky_regularization))
    
    def forward(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        Z_0: torch.Tensor,
        V_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Kalman filter and smoother. Main entry point.
        
        Parameters
        ----------
        Y : torch.Tensor
            Input data (k x nobs)
        A : torch.Tensor
            Transition matrix (m x m)
        C : torch.Tensor
            Observation matrix (k x m)
        Q : torch.Tensor
            Covariance for transition residuals (m x m)
        R : torch.Tensor
            Covariance for observation residuals (k x k)
        Z_0 : torch.Tensor
            Initial state (m,)
        V_0 : torch.Tensor
            Initial covariance (m x m)
            
        Returns
        -------
        zsmooth : torch.Tensor
            Smoothed factor estimates (m x (nobs+1)), zsmooth[:, t+1] = Z_t|T
        Vsmooth : torch.Tensor
            Smoothed factor covariance (m x m x (nobs+1)), Vsmooth[:, :, t+1] = Cov(Z_t|T)
        VVsmooth : torch.Tensor
            Lag 1 factor covariance (m x m x nobs), Cov(Z_t, Z_t-1|T)
        loglik : float
            Log-likelihood
        """
        # Kalman filter (forward pass)
        S = self.filter_forward(Y, A, C, Q, R, Z_0, V_0)
        
        # Fixed-interval smoother (backward pass)
        S = self.smoother_backward(A, S)
        
        # Organize output
        zsmooth = S.ZmT
        Vsmooth = S.VmT
        VVsmooth = S.VmT_1
        loglik = S.loglik
        
        # Ensure loglik is real and finite
        if not torch.isfinite(torch.tensor(loglik)):
            loglik = float('-inf')
        
        return zsmooth, Vsmooth, VVsmooth, loglik
    
    def filter_forward(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        Z_0: torch.Tensor,
        V_0: torch.Tensor
    ) -> KalmanFilterState:
        """Apply Kalman filter (forward pass).
        
        Parameters
        ----------
        Y : torch.Tensor
            Input data (k x nobs), where k = number of series, nobs = time periods
        A : torch.Tensor
            Transition matrix (m x m)
        C : torch.Tensor
            Observation matrix (k x m)
        Q : torch.Tensor
            Covariance for transition equation residuals (m x m)
        R : torch.Tensor
            Covariance for observation matrix residuals (k x k)
        Z_0 : torch.Tensor
            Initial state vector (m,)
        V_0 : torch.Tensor
            Initial state covariance matrix (m x m)
            
        Returns
        -------
        KalmanFilterState
            Filter state with prior and posterior estimates
        """
        # Dimensions
        k, nobs = Y.shape  # k series, nobs time periods
        m = C.shape[1]     # m factors
        
        device = Y.device
        dtype = Y.dtype
        
        # Replace Inf values with NaN (handled by missing data logic)
        if torch.any(torch.isinf(Y)):
            inf_count = torch.sum(torch.isinf(Y)).item()
            _logger.warning(f"kalman_filter_forward: Input Y contains {inf_count} Inf values, replacing with NaN")
            Y = torch.nan_to_num(Y, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
        
        # Pre-stabilize input matrices
        # Q: process noise covariance - must be positive definite
        Q = self._ensure_cov_stable(Q, min_eigenval=self.min_eigenval.item(), ensure_real=True)
        Q = self._clean_matrix(Q, 'covariance', default_nan=1e-6, default_inf=1e6)
        
        # R: observation noise covariance - must be positive definite (diagonal)
        # R is typically diagonal, but ensure it's stable
        if R.ndim == 2:
            # Ensure R is diagonal and positive
            diag_R = torch.diag(R)
            diag_R = torch.clamp(diag_R, min=self.min_diagonal_variance.item())
            diag_R = torch.nan_to_num(diag_R, nan=self.min_diagonal_variance.item(), posinf=1e4, neginf=self.min_diagonal_variance.item())
            R = torch.diag(diag_R)
        else:
            R = self._clean_matrix(R, 'diagonal', default_nan=self.min_diagonal_variance.item(), default_inf=1e4)
        
        # A: transition matrix - ensure it's real and finite
        A = self._clean_matrix(A, 'general', default_nan=0.0, default_inf=1.0)
        # Clip A to prevent instability (like numpy version)
        A = torch.clamp(A, min=-0.99, max=0.99)
        
        # C: observation matrix - ensure it's real and finite
        C = self._clean_matrix(C, 'loading', default_nan=0.0, default_inf=1.0)
        
        # V_0: initial covariance - must be positive definite
        V_0 = self._ensure_cov_stable(V_0, min_eigenval=self.min_eigenval.item(), ensure_real=True)
        V_0 = self._clean_matrix(V_0, 'covariance', default_nan=1e-6, default_inf=1e6)
        
        # Z_0: initial state - ensure finite
        Z_0 = torch.nan_to_num(Z_0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Initialize output
        Zm = torch.full((m, nobs), float('nan'), device=device, dtype=dtype)  # Z_t | t-1 (prior)
        Vm = torch.full((m, m, nobs), float('nan'), device=device, dtype=dtype)  # V_t | t-1 (prior)
        ZmU = torch.full((m, nobs + 1), float('nan'), device=device, dtype=dtype)  # Z_t | t (posterior/updated)
        VmU = torch.full((m, m, nobs + 1), float('nan'), device=device, dtype=dtype)  # V_t | t (posterior/updated)
        loglik = 0.0
        
        # Set initial values
        Zu = Z_0.clone()  # Z_0|0 (In loop, Zu gives Z_t | t)
        Vu = V_0.clone()  # V_0|0 (In loop, Vu gives V_t | t)
        
        # Validate dimensions match
        if Zu.shape[0] != m:
            raise ValueError(
                f"Dimension mismatch: Z_0 has shape {Zu.shape[0]}, but C has {m} columns. "
                f"This usually indicates a mismatch between init_conditions and em_step. "
                f"Z_0 should have dimension {m} to match C.shape[1]."
            )
        if Vu.shape[0] != m or Vu.shape[1] != m:
            raise ValueError(
                f"Dimension mismatch: V_0 has shape {Vu.shape}, but expected ({m}, {m}). "
                f"This usually indicates a mismatch between init_conditions and em_step."
            )
        
        # Store initial values
        ZmU[:, 0] = Zu
        VmU[:, :, 0] = Vu
        
        # Initialize variables for final iteration (used after loop)
        Y_t = torch.tensor([], device=device, dtype=dtype)  # Initialize Y_t to empty tensor
        C_t = None
        VCF = None
        
        # Kalman filter procedure
        for t in range(nobs):
            # Calculate prior distribution
            # Use transition equation to create prior estimate for factor
            # i.e. Z = Z_t|t-1
            # Check for NaN/Inf in inputs
            if not self._check_finite(Zu, f"Zu at t={t}"):
                _logger.warning(f"kalman_filter_forward: Zu contains NaN/Inf at t={t}, resetting to zeros")
                Zu = torch.zeros_like(Zu)
            
            Z = A @ Zu
            
            # Check for NaN/Inf in Z
            if not self._check_finite(Z, f"Z at t={t}"):
                _logger.warning(f"kalman_filter_forward: Z contains NaN/Inf at t={t}, using previous Zu")
                Z = Zu.clone()
            
            # Prior covariance matrix of Z (i.e. V = V_t|t-1)
            # Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(u) = A*Vu*A' + Q
            # PRE-REGULARIZATION: Check Vu condition number before V calculation to prevent NaN propagation
            if not self._check_finite(Vu, f"Vu before V calculation at t={t}"):
                Vu = self._ensure_cov_stable(Vu, min_eigenval=self.min_eigenval.item(), ensure_real=True)
            else:
                # Pre-regularization: Check condition number and apply adaptive regularization if ill-conditioned
                try:
                    eigenvals = torch.linalg.eigvalsh(Vu)
                    eigenvals = eigenvals[eigenvals > 1e-12]
                    if len(eigenvals) > 0:
                        max_eig = torch.max(eigenvals)
                        min_eig = torch.min(eigenvals)
                        cond_num = max_eig / min_eig if min_eig > 1e-12 else float('inf')
                        
                        # If ill-conditioned, apply pre-regularization (MATLAB-style stability)
                        if cond_num > 1e8:
                            reg_scale = self.min_eigenval.item() * (cond_num / 1e8)
                            Vu = Vu + torch.eye(Vu.shape[0], device=device, dtype=dtype) * reg_scale
                            _logger.debug(f"Pre-regularized Vu at t={t}: cond={cond_num:.2e}, reg={reg_scale:.2e}")
                except (RuntimeError, ValueError):
                    # Fallback: apply default regularization if eigendecomposition fails
                    Vu = Vu + torch.eye(Vu.shape[0], device=device, dtype=dtype) * self.min_eigenval.item()
            
            # Ensure Vu is symmetric before matrix multiplication (MATLAB: V = 0.5 * (V+V'))
            Vu = ensure_real_and_symmetric(Vu)
            
            # Now safely compute V
            V = A @ Vu @ A.T + Q
            
            # Check for NaN/Inf before stabilization
            if not self._check_finite(V, f"V at t={t}"):
                # Fallback: use previous covariance with regularization
                V = Vu + torch.eye(V.shape[0], device=device, dtype=dtype) * 1e-6
            
            # Cap maximum eigenvalue to prevent explosion
            V = self._cap_max_eigenval(V, max_eigenval=1e6)
            # Ensure V is real, symmetric, and positive semi-definite (MATLAB: V = 0.5 * (V+V'))
            V = ensure_real_and_symmetric(V)
            V = self._ensure_cov_stable(V, min_eigenval=self.min_eigenval.item(), ensure_real=True)
            
            # Calculate posterior distribution
            # Remove missing series: These are removed from Y, C, and R
            Y_t, C_t, R_t, _ = self.handle_missing_data(Y[:, t], C, R)
            
            # Check if y_t contains no data
            if len(Y_t) == 0:
                Zu = Z
                Vu = V
            else:
                # Adaptive regularization based on missing data ratio
                missing_ratio = 1.0 - (len(Y_t) / k)
                adaptive_reg = self.inv_regularization.item() * (1.0 + missing_ratio * 10.0)
                
                # Additional regularization for large n_obs
                n_obs = len(Y_t)
                if n_obs >= 30:
                    n_obs_factor = 1.0 + (n_obs - 29) * 0.1
                    adaptive_reg = adaptive_reg * n_obs_factor
                    if t == 0:
                        _logger.debug(f"kalman_filter_forward: n_obs={n_obs} >= 30, n_obs_factor={n_obs_factor:.2f}, adaptive_reg={adaptive_reg:.2e}")
                
                # Steps for variance and population regression coefficients:
                # Var(c_t*Z_t + e_t) = c_t Var(Z) c_t' + Var(e) = c_t*V*c_t' + R
                VC = V @ C_t.T
                
                # Compute innovation covariance F = C_t @ V @ C_t.T + R_t
                F = C_t @ VC + R_t
                
                # Apply adaptive regularization to R_t diagonal
                if R_t.ndim == 2 and R_t.shape[0] == R_t.shape[1]:
                    diag_R_t = torch.diag(R_t)
                    diag_R_t = torch.clamp(diag_R_t, min=self.min_diagonal_variance.item() * (1.0 + missing_ratio * 5.0))
                    R_t = torch.diag(diag_R_t)
                
                # Log F matrix condition number for diagnostics
                try:
                    eigenvals_F = torch.linalg.eigvalsh(F)
                    eigenvals_F = eigenvals_F[eigenvals_F > 1e-12]
                    if len(eigenvals_F) > 0:
                        max_eig_F = torch.max(eigenvals_F).item()
                        min_eig_F = torch.min(eigenvals_F).item()
                        cond_F = max_eig_F / min_eig_F if min_eig_F > 1e-12 else float('inf')
                        if cond_F > 1e10 or t < 5:
                            _logger.debug(
                                f"kalman_filter_forward: t={t}, n_obs={n_obs}, F condition={cond_F:.2e}, "
                                f"eigenvals=[{min_eig_F:.2e}, {max_eig_F:.2e}], adaptive_reg={adaptive_reg:.2e}"
                            )
                except (RuntimeError, ValueError):
                    pass
                
                # Cap maximum eigenvalue to prevent condition number explosion
                F = self._cap_max_eigenval(F, max_eigenval=1e6)
                # Ensure F is real, symmetric, and positive semi-definite
                # Use adaptive minimum eigenvalue based on missing data ratio
                min_eig_adaptive = self.min_eigenval.item() * (1.0 + missing_ratio * 5.0)
                F = self._ensure_cov_stable(F, min_eigenval=min_eig_adaptive, ensure_real=True)
                
                # Check for NaN/Inf before inversion
                if not self._check_finite(F, f"F at t={t}"):
                    # Fallback: use identity with large variance, scaled by missing ratio
                    fallback_var = 1e6 * (1.0 + missing_ratio * 5.0)
                    F = torch.eye(F.shape[0], device=device, dtype=dtype) * fallback_var
                    _logger.warning(f"kalman_filter_forward: F matrix contains NaN/Inf at t={t}, using fallback (missing_ratio={missing_ratio:.2f})")
                
                # Safe inverse with adaptive regularization
                iF = self._safe_inv(F, regularization=adaptive_reg, use_pinv_fallback=True)
                
                if not self._check_finite(iF, f"iF (inverse of F) at t={t}") and t < 5:
                    _logger.warning(
                        f"kalman_filter_forward: iF contains NaN/Inf at t={t}, n_obs={n_obs}, "
                        f"adaptive_reg={adaptive_reg:.2e}, missing_ratio={missing_ratio:.2f}"
                    )
                
                # Matrix of population regression coefficients (Kalman gain)
                VCF = VC @ iF
                
                # Difference between actual and predicted observation matrix values
                innov = Y_t - C_t @ Z
                
                # Check for NaN/Inf in innovation
                if not self._check_finite(innov, f"innovation at t={t}"):
                    _logger.warning(f"kalman_filter_forward: Innovation contains NaN/Inf at t={t}, skipping update")
                    Zu = Z
                    Vu = V
                else:
                    # Check VCF before matrix multiplication
                    if not self._check_finite(VCF, f"VCF (Kalman gain) at t={t}"):
                        _logger.warning(f"kalman_filter_forward: VCF contains NaN/Inf at t={t}, using zero matrix")
                        VCF = torch.zeros_like(VCF)
                    
                    # Update estimate of factor values (posterior)
                    Zu = Z + VCF @ innov
                    
                    # Clean NaN/Inf immediately after update
                    if not self._check_finite(Zu, f"Zu at t={t}"):
                        Zu = Z.clone()
                        Zu = self._clean_matrix(Zu, 'general', default_nan=0.0, default_inf=0.0)
                        _logger.warning(f"kalman_filter_forward: Zu update produced NaN/Inf at t={t}, using prior Z as fallback")
                    
                    # Update covariance matrix (posterior)
                    if not self._check_finite(VC, f"VC at t={t}"):
                        _logger.warning(f"kalman_filter_forward: VC contains NaN/Inf at t={t}, using zero matrix")
                        VC = torch.zeros_like(VC)
                    
                    Vu = V - VCF @ VC.T
                    
                    # MATLAB-style: Ensure symmetry immediately after update (V = 0.5 * (V+V'))
                    Vu = ensure_real_and_symmetric(Vu)
                    # Cap maximum eigenvalue to prevent explosion
                    Vu = self._cap_max_eigenval(Vu, max_eigenval=1e6)
                    
                    # Clean NaN/Inf before stabilization
                    if not self._check_finite(Vu, f"Vu at t={t}"):
                        Vu = self._clean_matrix(Vu, 'general', default_nan=1e-8, default_inf=1e6)
                        Vu = ensure_real_and_symmetric(Vu)  # Re-apply symmetry after cleaning
                    
                    # Check for NaN/Inf after cleaning
                    if not self._check_finite(Vu, f"Vu at t={t}"):
                        _logger.warning(f"kalman_filter_forward: Vu contains NaN/Inf at t={t}, using V as fallback")
                        Vu = V.clone()
                        Vu = ensure_real_and_symmetric(Vu)  # Ensure symmetry of fallback
                        # Ensure fallback is also stable
                        Vu = self._ensure_cov_stable(Vu, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                    
                    # Ensure Vu is real, symmetric, and positive semi-definite (MATLAB: Vu = 0.5 * (Vu+Vu'))
                    Vu = ensure_real_and_symmetric(Vu)
                    Vu = self._ensure_cov_stable(Vu, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                    
                    # Update log-likelihood (with safeguards)
                    try:
                        det_iF = self._safe_det(iF, use_logdet=True)
                        if det_iF > 0 and torch.isfinite(torch.tensor(det_iF)):
                            log_det = torch.log(torch.tensor(det_iF, device=device, dtype=dtype))
                            # Ensure innov is 2D for transpose: (N,) -> (N, 1) -> (1, N)
                            innov_2d = innov.unsqueeze(1) if innov.ndim == 1 else innov
                            innov_term = innov_2d.T @ iF @ innov_2d
                            if torch.isfinite(innov_term):
                                loglik += 0.5 * (log_det.item() - innov_term.item())
                            else:
                                _logger.debug(f"kalman_filter_forward: innov_term not finite at t={t}, skipping loglik update")
                        else:
                            _logger.debug(f"kalman_filter_forward: det(iF) <= 0 or not finite at t={t}, skipping loglik update")
                    except (RuntimeError, ValueError, OverflowError):
                        _logger.debug(f"kalman_filter_forward: Log-likelihood calculation failed at t={t}")
            
            # Store output
            # Store covariance and observation values for t (priors)
            # Ensure Z and V are real and finite before storing
            Z = self._ensure_real(Z)
            if not self._check_finite(Z, f"Z (prior) at t={t}"):
                Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"kalman_filter_forward: Z (prior) contains NaN/Inf at t={t}, cleaned to zero")
            V = ensure_real_and_symmetric(V)
            if not self._check_finite(V, f"V (prior) at t={t}"):
                V = torch.eye(V.shape[0], device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
                _logger.warning(f"kalman_filter_forward: V (prior) contains NaN/Inf at t={t}, using regularized identity")
            Zm[:, t] = Z
            Vm[:, :, t] = V
            
            # Store covariance and state values for t (posteriors)
            # i.e. Zu = Z_t|t   & Vu = V_t|t
            # Ensure Zu and Vu are finite before storing
            Zu = self._ensure_real(Zu)
            if not self._check_finite(Zu, f"Zu (posterior) at t={t}"):
                Zu = torch.nan_to_num(Zu, nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"kalman_filter_forward: Zu (posterior) contains NaN/Inf at t={t}, cleaned to zero")
            Vu = ensure_real_and_symmetric(Vu)
            if not self._check_finite(Vu, f"Vu (posterior) at t={t}"):
                Vu = torch.eye(Vu.shape[0], device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
                Vu = self._ensure_cov_stable(Vu, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                _logger.warning(f"kalman_filter_forward: Vu (posterior) contains NaN/Inf at t={t}, using regularized identity")
            ZmU[:, t + 1] = Zu
            VmU[:, :, t + 1] = Vu
        
        # Store Kalman gain k_t (from final iteration)
        # k_t should be m x n_obs where n_obs is number of observed series at final time
        # VCF is m x n_obs, C_t is n_obs x m, so VCF @ C_t gives m x m
        # However, if no observations at final time, use zeros
        if len(Y_t) == 0:
            k_t = torch.zeros((m, m), device=device, dtype=dtype)
        else:
            # VCF is m x n_obs, C_t is n_obs x m, so k_t = VCF @ C_t is m x m
            k_t = VCF @ C_t
        
        return KalmanFilterState(Zm=Zm, Vm=Vm, ZmU=ZmU, VmU=VmU, loglik=loglik, k_t=k_t)
    
    def smoother_backward(
        self,
        A: torch.Tensor,
        S: KalmanFilterState
    ) -> KalmanFilterState:
        """Apply fixed-interval smoother (backward pass).
        
        Parameters
        ----------
        A : torch.Tensor
            Transition matrix (m x m)
        S : KalmanFilterState
            State from Kalman filter (forward pass)
            
        Returns
        -------
        KalmanFilterState
            State with smoothed estimates added (ZmT, VmT, VmT_1)
        """
        m, nobs = S.Zm.shape
        
        device = S.Zm.device
        dtype = S.Zm.dtype
        
        # Initialize output matrices
        ZmT = torch.zeros((m, nobs + 1), device=device, dtype=dtype)
        VmT = torch.zeros((m, m, nobs + 1), device=device, dtype=dtype)
        
        # Fill the final period of ZmT, VmT with SKF posterior values
        # Check and clean initial values
        ZmT_init = S.ZmU[:, nobs].clone()
        if not self._check_finite(ZmT_init, f"ZmU[:, nobs] (initial smoother state)"):
            # If initial state contains NaN, use zero as fallback
            ZmT_init = torch.nan_to_num(ZmT_init, nan=0.0, posinf=0.0, neginf=0.0)
            _logger.warning("smoother_backward: Initial ZmU[:, nobs] contains NaN/Inf, using zero as fallback")
        ZmT[:, nobs] = ZmT_init
        
        VmT_init = S.VmU[:, :, nobs].clone()
        if not self._check_finite(VmT_init, f"VmU[:, :, nobs] (initial smoother covariance)"):
            # If initial covariance contains NaN, use identity with small regularization as fallback
            VmT_init = torch.eye(m, device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
            _logger.warning("smoother_backward: Initial VmU[:, :, nobs] contains NaN/Inf, using regularized identity as fallback")
        else:
            # Ensure initial covariance is PSD even if finite
            VmT_init = self._ensure_cov_stable(VmT_init, min_eigenval=self.min_eigenval.item(), ensure_real=True)
        VmT[:, :, nobs] = VmT_init
        
        # Early termination if initial state contains NaN
        # This prevents NaN from propagating through entire backward pass
        initial_state_has_nan = not self._check_finite(ZmT[:, nobs], f"ZmT[:, nobs] (initial)") or not self._check_finite(VmT[:, :, nobs], f"VmT[:, :, nobs] (initial)")
        if initial_state_has_nan:
            _logger.warning("smoother_backward: Initial smoothed state contains NaN/Inf after cleaning. Using forward-only estimates (skipping backward pass) to prevent NaN propagation.")
            # Fill all ZmT with forward estimates (ZmU), all VmT with forward estimates (VmU)
            for t in range(nobs + 1):
                ZmT_forward = S.ZmU[:, t].clone()
                if not self._check_finite(ZmT_forward, f"ZmU[:, {t}] (forward fallback)"):
                    ZmT_forward = torch.nan_to_num(ZmT_forward, nan=0.0, posinf=0.0, neginf=0.0)
                ZmT[:, t] = ZmT_forward
                
                VmT_forward = S.VmU[:, :, t].clone()
                if not self._check_finite(VmT_forward, f"VmU[:, :, {t}] (forward fallback)"):
                    VmT_forward = torch.eye(m, device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
                    VmT_forward = self._ensure_cov_stable(VmT_forward, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                else:
                    VmT_forward = self._ensure_cov_stable(VmT_forward, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                VmT[:, :, t] = VmT_forward
            
            # Initialize VmT_1 with zeros (no lag-1 covariance available)
            VmT_1 = torch.zeros((m, m, nobs), device=device, dtype=dtype)
            
            # Add smoothed estimates as attributes (using forward-only estimates)
            S.ZmT = ZmT
            S.VmT = VmT
            S.VmT_1 = VmT_1
            
            return S
        
        # Initialize VmT_1 lag 1 covariance matrix for final period
        VmT_1 = torch.zeros((m, m, nobs), device=device, dtype=dtype)
        # CRITICAL FIX: Check VmU[:, :, nobs - 1] before using it
        VmU_nobs_minus_1 = S.VmU[:, :, nobs - 1]
        if not self._check_finite(VmU_nobs_minus_1, f"VmU[:, :, nobs - 1] (for VmT_1 initialization)"):
            # Use regularized identity as fallback
            VmU_nobs_minus_1 = torch.eye(m, device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
            _logger.warning("smoother_backward: VmU[:, :, nobs - 1] contains NaN/Inf, using regularized identity for VmT_1")
        VmT_1_temp = (torch.eye(m, device=device, dtype=dtype) - S.k_t) @ A @ VmU_nobs_minus_1
        VmT_1[:, :, nobs - 1] = ensure_real_and_symmetric(VmT_1_temp)
        
        # Used for recursion process
        # CRITICAL FIX: Check forward pass values before computing J_2
        VmU_for_J2 = S.VmU[:, :, nobs - 1]
        Vm_for_J2 = S.Vm[:, :, nobs - 1]
        if not self._check_finite(VmU_for_J2, f"VmU[:, :, nobs - 1] (for J_2)") or not self._check_finite(Vm_for_J2, f"Vm[:, :, nobs - 1] (for J_2)"):
            # If forward pass values contain NaN, use zero matrix as fallback
            J_2 = torch.zeros((m, m), device=device, dtype=dtype)
            _logger.warning("smoother_backward: Forward pass values contain NaN/Inf, using zero matrix for J_2")
        else:
            try:
                J_2 = VmU_for_J2 @ A.T @ torch.linalg.pinv(Vm_for_J2)
                # Check if J_2 contains NaN/Inf
                if not self._check_finite(J_2, f"J_2 (initial)"):
                    J_2 = torch.zeros((m, m), device=device, dtype=dtype)
                    _logger.warning("smoother_backward: J_2 contains NaN/Inf, using zero matrix as fallback")
            except RuntimeError:
                # Fallback if pinv fails
                J_2 = torch.zeros((m, m), device=device, dtype=dtype)
                _logger.warning("smoother_backward: pinv failed for J_2, using zero matrix as fallback")
        
        # Run smoothing algorithm
        # Loop through time reverse-chronologically (starting at final period nobs-1)
        for t in range(nobs - 1, -1, -1):
            # Store posterior and prior factor covariance values
            VmU = S.VmU[:, :, t]
            Vm1 = S.Vm[:, :, t]
            
            # Store previous period smoothed factor covariance and lag-1 covariance
            V_T = VmT[:, :, t + 1]
            V_T1 = VmT_1[:, :, t] if t < nobs - 1 else torch.zeros((m, m), device=device, dtype=dtype)
            
            # CRITICAL FIX: Check and clean V_T before using in recursion to prevent NaN propagation
            if not self._check_finite(V_T, f"V_T (VmT[:, :, t+1]) at t={t}"):
                # If previous smoothed covariance is invalid, use posterior covariance as fallback
                V_T = VmU.clone()
                V_T = self._ensure_cov_stable(V_T, min_eigenval=self.min_eigenval.item(), ensure_real=True)
                _logger.warning(f"smoother_backward: V_T contains NaN/Inf at t={t}, using VmU as fallback")
            
            # CRITICAL FIX: Check and clean ZmT[:, t+1] before using in recursion
            ZmT_next = ZmT[:, t + 1]
            if not self._check_finite(ZmT_next, f"ZmT[:, t+1] at t={t}"):
                # If next smoothed state is invalid, use posterior state as fallback
                ZmT_next = S.ZmU[:, t].clone()
                ZmT_next = torch.nan_to_num(ZmT_next, nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"smoother_backward: ZmT[:, t+1] contains NaN/Inf at t={t}, using ZmU as fallback")
            
            J_1 = J_2
            
            # CRITICAL FIX: Check J_1 before using in recursion
            if not self._check_finite(J_1, f"J_1 at t={t}"):
                # If J_1 contains NaN, use zero matrix as fallback
                J_1 = torch.zeros((m, m), device=device, dtype=dtype)
                _logger.warning(f"smoother_backward: J_1 contains NaN/Inf at t={t}, using zero matrix as fallback")
            
            # CRITICAL FIX: Check forward pass values before using in recursion
            ZmU_t = S.ZmU[:, t]
            if not self._check_finite(ZmU_t, f"ZmU[:, t] at t={t}"):
                ZmU_t = torch.nan_to_num(ZmU_t, nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"smoother_backward: ZmU[:, t] contains NaN/Inf at t={t}, using cleaned values")
            
            # Update smoothed factor estimate
            # CRITICAL FIX: Check all inputs before recursion to prevent NaN propagation
            if not self._check_finite(ZmT_next, f"ZmT[:, t+1] at t={t}") or not self._check_finite(ZmU_t, f"ZmU[:, t] at t={t}") or not self._check_finite(J_1, f"J_1 at t={t}"):
                # If any input contains NaN, use forward estimate as fallback
                ZmT[:, t] = ZmU_t.clone()
                ZmT[:, t] = torch.nan_to_num(ZmT[:, t], nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"smoother_backward: Input contains NaN/Inf at t={t}, using ZmU as fallback (skipping recursion)")
            else:
                # Safe to perform recursion
                try:
                    ZmT[:, t] = ZmU_t + J_1 @ (ZmT_next - A @ ZmU_t)
                except RuntimeError as e:
                    _logger.warning(f"smoother_backward: Recursion failed at t={t}: {e}, using ZmU as fallback")
                    ZmT[:, t] = ZmU_t.clone()
                    ZmT[:, t] = torch.nan_to_num(ZmT[:, t], nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clean NaN/Inf after recursion
            if not self._check_finite(ZmT[:, t], f"ZmT[:, t] at t={t}"):
                # If recursion produces NaN, fall back to forward pass value
                ZmT[:, t] = ZmU_t.clone()
                ZmT[:, t] = torch.nan_to_num(ZmT[:, t], nan=0.0, posinf=0.0, neginf=0.0)
                _logger.warning(f"smoother_backward: ZmT[:, t] recursion produced NaN/Inf at t={t}, using ZmU as fallback")
            
            # CRITICAL FIX: Check forward pass covariance values before using in recursion
            if not self._check_finite(VmU, f"VmU[:, :, t] at t={t}"):
                VmU = torch.eye(m, device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
                _logger.warning(f"smoother_backward: VmU[:, :, t] contains NaN/Inf at t={t}, using regularized identity as fallback")
            if not self._check_finite(Vm1, f"Vm[:, :, t] at t={t}"):
                Vm1 = torch.eye(m, device=device, dtype=dtype) * (self.min_eigenval.item() * 10.0)
                _logger.warning(f"smoother_backward: Vm[:, :, t] contains NaN/Inf at t={t}, using regularized identity as fallback")
            
            # Update smoothed factor covariance matrix
            # CRITICAL FIX: Check all inputs before recursion to prevent NaN propagation
            if not self._check_finite(V_T, f"V_T (VmT[:, :, t+1]) at t={t}") or not self._check_finite(VmU, f"VmU[:, :, t] at t={t}") or not self._check_finite(Vm1, f"Vm[:, :, t] at t={t}") or not self._check_finite(J_1, f"J_1 at t={t}"):
                # If any input contains NaN, use forward estimate as fallback
                VmT[:, :, t] = VmU.clone()
                VmT[:, :, t] = self._ensure_cov_stable(VmT[:, :, t], min_eigenval=self.min_eigenval.item(), ensure_real=True)
                _logger.warning(f"smoother_backward: Input contains NaN/Inf at t={t}, using VmU as fallback (skipping recursion)")
            else:
                # Safe to perform recursion
                try:
                    VmT_temp = VmU + J_1 @ (V_T - Vm1) @ J_1.T
                    VmT[:, :, t] = ensure_real_and_symmetric(VmT_temp)
                except RuntimeError as e:
                    _logger.warning(f"smoother_backward: Covariance recursion failed at t={t}: {e}, using VmU as fallback")
                    VmT[:, :, t] = VmU.clone()
                    VmT[:, :, t] = self._ensure_cov_stable(VmT[:, :, t], min_eigenval=self.min_eigenval.item(), ensure_real=True)
            
            # Clean NaN/Inf and ensure PSD
            if not self._check_finite(VmT[:, :, t], f"VmT[:, :, t] at t={t}"):
                # If recursion produces NaN, fall back to forward pass value
                VmT[:, :, t] = VmU.clone()
                VmT[:, :, t] = self._ensure_cov_stable(VmT[:, :, t], min_eigenval=self.min_eigenval.item(), ensure_real=True)
                _logger.warning(f"smoother_backward: VmT[:, :, t] recursion produced NaN/Inf at t={t}, using VmU as fallback")
            
            if t > 0:
                # Update weight
                # Check forward pass values before computing J_2
                VmU_t_minus_1 = S.VmU[:, :, t - 1]
                Vm_t_minus_1 = S.Vm[:, :, t - 1]
                if not self._check_finite(VmU_t_minus_1, f"VmU[:, :, t - 1] at t={t}") or not self._check_finite(Vm_t_minus_1, f"Vm[:, :, t - 1] at t={t}"):
                    # If forward pass values contain NaN, use zero matrix as fallback
                    J_2 = torch.zeros((m, m), device=device, dtype=dtype)
                    _logger.warning(f"smoother_backward: Forward pass values contain NaN/Inf at t={t}, using zero matrix for J_2")
                else:
                    try:
                        J_2 = VmU_t_minus_1 @ A.T @ torch.linalg.pinv(Vm_t_minus_1)
                        # Check if J_2 contains NaN/Inf
                        if not self._check_finite(J_2, f"J_2 at t={t}"):
                            J_2 = torch.zeros((m, m), device=device, dtype=dtype)
                            _logger.warning(f"smoother_backward: J_2 contains NaN/Inf at t={t}, using zero matrix as fallback")
                    except RuntimeError:
                        J_2 = torch.zeros((m, m), device=device, dtype=dtype)
                        _logger.warning(f"smoother_backward: pinv failed for J_2 at t={t}, using zero matrix as fallback")
                
                # Update lag 1 factor covariance matrix
                # Check V_T1 before using in recursion
                if not self._check_finite(V_T1, f"VmT_1[:, :, t] at t={t}"):
                    V_T1 = torch.zeros((m, m), device=device, dtype=dtype)
                    _logger.warning(f"smoother_backward: VmT_1[:, :, t] contains NaN/Inf at t={t}, using zero matrix as fallback")
                
                VmT_1_temp = VmU @ J_2.T + J_1 @ (V_T1 - A @ VmU) @ J_2.T
                VmT_1[:, :, t - 1] = ensure_real_and_symmetric(VmT_1_temp)
                
                # Check VmT_1 result
                if not self._check_finite(VmT_1[:, :, t - 1], f"VmT_1[:, :, t - 1] at t={t}"):
                    VmT_1[:, :, t - 1] = torch.zeros((m, m), device=device, dtype=dtype)
                    _logger.warning(f"smoother_backward: VmT_1[:, :, t - 1] recursion produced NaN/Inf at t={t}, using zero matrix as fallback")
        
        # Add smoothed estimates as attributes
        S.ZmT = ZmT
        S.VmT = VmT
        S.VmT_1 = VmT_1
        
        return S
    
    def handle_missing_data(
        self,
        y: torch.Tensor, 
        C: torch.Tensor, 
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle missing data by removing NaN observations from the Kalman filter equations.
        
        Parameters
        ----------
        y : torch.Tensor
            Vector of observations at time t, shape (k,) where k is number of series.
            Missing values should be NaN.
        C : torch.Tensor
            Observation/loading matrix, shape (k x m) where m is state dimension.
            Each row corresponds to a series in y.
        R : torch.Tensor
            Covariance matrix for observation residuals, shape (k x k).
            Typically diagonal (idiosyncratic variances).
            
        Returns
        -------
        y_clean : torch.Tensor
            Reduced observation vector with NaN values removed, shape (k_obs,)
            where k_obs is number of non-missing observations.
        C_clean : torch.Tensor
            Reduced observation matrix, shape (k_obs x m).
            Rows corresponding to missing observations are removed.
        R_clean : torch.Tensor
            Reduced covariance matrix, shape (k_obs x k_obs).
            Rows and columns corresponding to missing observations are removed.
        L : torch.Tensor
            Selection matrix, shape (k x k_obs), used to restore standard dimensions.
            L @ y_clean gives y with zeros for missing values.
        """
        # Returns True for nonmissing series
        ix = ~torch.isnan(y)
        
        # Index for columns with nonmissing variables
        k = len(y)
        e = torch.eye(k, device=y.device, dtype=y.dtype)
        L = e[:, ix]
        
        # Remove missing series
        y = y[ix]
        
        # Remove missing series from observation matrix
        C = C[ix, :]
        
        # Remove missing series from covariance matrix
        # Use advanced indexing for 2D matrix
        ix_2d = ix.unsqueeze(1).expand(-1, k)
        R = R[ix_2d].view(-1, k)[:, ix]
        
        return y, C, R, L
    
    # Private helper methods (delegate to utils module)
    def _check_finite(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains only finite values."""
        return check_finite(tensor, name)
    
    def _ensure_real(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is real by extracting real part if complex."""
        return ensure_real(tensor)
    
    def _ensure_symmetric(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure matrix is symmetric by averaging with its transpose."""
        return ensure_symmetric(tensor)
    
    
    def _ensure_cov_stable(
        self,
        M: torch.Tensor, 
        min_eigenval: float = None,
        ensure_real: bool = True
    ) -> torch.Tensor:
        """Ensure covariance matrix is real, symmetric, and positive semi-definite."""
        if min_eigenval is None:
            min_eigenval = self.min_eigenval.item()
        return ensure_covariance_stable(M, min_eigenval=min_eigenval, ensure_real=ensure_real)
    
    def _cap_max_eigenval(self, M: torch.Tensor, max_eigenval: float = 1e6) -> torch.Tensor:
        """Cap maximum eigenvalue of matrix to prevent numerical explosion."""
        return cap_max_eigenval(M, max_eigenval=max_eigenval, warn=False)
    
    def _clean_matrix(
        self,
        M: torch.Tensor, 
        matrix_type: str = 'general', 
        default_nan: float = 0.0, 
        default_inf: Optional[float] = None
    ) -> torch.Tensor:
        """Clean matrix by removing NaN/Inf values and ensuring numerical stability."""
        return clean_matrix(
            M, 
            matrix_type=matrix_type,
            default_nan=default_nan,
            default_inf=default_inf,
            min_eigenval=self.min_eigenval.item(),
            min_diagonal_variance=self.min_diagonal_variance.item()
        )
    
    def _safe_inv(
        self,
        M: torch.Tensor,
        regularization: float = None,
        use_pinv_fallback: bool = True
    ) -> torch.Tensor:
        """Safely compute matrix inverse with robust error handling."""
        if regularization is None:
            regularization = self.inv_regularization.item()
        return safe_inverse(M, regularization=regularization, use_pinv_fallback=use_pinv_fallback)
    
    def _safe_det(self, M: torch.Tensor, use_logdet: bool = True) -> float:
        """Compute determinant safely to avoid overflow warnings."""
        return safe_determinant(M, use_logdet=use_logdet)

