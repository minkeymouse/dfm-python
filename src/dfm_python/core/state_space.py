"""State-space operations: Kalman filter, smoother, and numeric utilities.

This module consolidates Kalman filtering, fixed-interval smoothing, and
numeric operations for state-space models. It combines functionality from
the original kalman.py and numeric.py modules.
"""

import numpy as np
from scipy.linalg import inv, pinv
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
import warnings

_logger = logging.getLogger(__name__)

# Numerical stability constants
MIN_EIGENVAL_CLEAN = 1e-8  # Minimum eigenvalue for matrix cleaning operations
MIN_DIAGONAL_VARIANCE = 1e-6  # Minimum diagonal variance for diagonal matrix cleaning
DEFAULT_VARIANCE_FALLBACK = 1.0  # Default variance when computation fails or result is invalid
MIN_VARIANCE_COVARIANCE = 1e-10  # Minimum variance threshold for covariance matrix diagonal


# ============================================================================
# Kalman Filter and Smoother
# ============================================================================

@dataclass
class KalmanFilterState:
    """Kalman filter state structure.
    
    This dataclass stores the complete state of the Kalman filter after forward
    and backward passes, including prior/posterior estimates and covariances.
    
    Attributes
    ----------
    Zm : np.ndarray
        Prior (predicted) factor state estimates, shape (m x nobs).
        Zm[:, t] is the predicted state at time t given observations up to t-1.
    Vm : np.ndarray
        Prior covariance matrices, shape (m x m x nobs).
        Vm[:, :, t] is the covariance of Zm[:, t].
    ZmU : np.ndarray
        Posterior (updated) factor state estimates, shape (m x (nobs+1)).
        ZmU[:, t] is the updated state at time t given observations up to t.
        Includes initial state at t=0.
    VmU : np.ndarray
        Posterior covariance matrices, shape (m x m x (nobs+1)).
        VmU[:, :, t] is the covariance of ZmU[:, t].
    loglik : float
        Log-likelihood of the data under the current model parameters.
        Computed as sum of log-likelihoods at each time step.
    k_t : np.ndarray
        Kalman gain matrix, shape (m x k) where k is number of observed series.
        Used to update state estimates with new observations.
    """
    Zm: np.ndarray      # Prior/predicted factor state (m x nobs)
    Vm: np.ndarray      # Prior covariance (m x m x nobs)
    ZmU: np.ndarray     # Posterior/updated state (m x (nobs+1))
    VmU: np.ndarray     # Posterior covariance (m x m x (nobs+1))
    loglik: float       # Log-likelihood
    k_t: np.ndarray     # Kalman gain


def miss_data(y: np.ndarray, C: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handle missing data by removing NaN observations from the Kalman filter equations.
    
    This function implements the standard approach to missing data in Kalman filtering:
    observations with NaN values are removed from the observation vector, observation
    matrix, and covariance matrix. A selection matrix L is returned to restore standard
    dimensions if needed.
    
    Parameters
    ----------
    y : np.ndarray
        Vector of observations at time t, shape (k,) where k is number of series.
        Missing values should be NaN.
    C : np.ndarray
        Observation/loading matrix, shape (k x m) where m is state dimension.
        Each row corresponds to a series in y.
    R : np.ndarray
        Covariance matrix for observation residuals, shape (k x k).
        Typically diagonal (idiosyncratic variances).
        
    Returns
    -------
    y_clean : np.ndarray
        Reduced observation vector with NaN values removed, shape (k_obs,)
        where k_obs is number of non-missing observations.
    C_clean : np.ndarray
        Reduced observation matrix, shape (k_obs x m).
        Rows corresponding to missing observations are removed.
    R_clean : np.ndarray
        Reduced covariance matrix, shape (k_obs x k_obs).
        Rows and columns corresponding to missing observations are removed.
    L : np.ndarray
        Selection matrix, shape (k x k_obs), used to restore standard dimensions.
        L @ y_clean gives y with zeros for missing values.
        
    Notes
    -----
    This function is called at each time step in the Kalman filter to handle
    missing observations. The selection matrix L allows reconstruction of the
    full-dimensional vectors if needed for downstream processing.
    
    Examples
    --------
    >>> y = np.array([1.0, np.nan, 3.0])
    >>> C = np.array([[1, 0], [0, 1], [1, 1]])
    >>> R = np.eye(3)
    >>> y_clean, C_clean, R_clean, L = miss_data(y, C, R)
    >>> # y_clean = [1.0, 3.0], C_clean has 2 rows, R_clean is 2x2
    """
    # Returns True for nonmissing series
    ix = ~np.isnan(y)
    
    # Index for columns with nonmissing variables
    e = np.eye(len(y))
    L = e[:, ix]
    
    # Remove missing series
    y = y[ix]
    
    # Remove missing series from observation matrix
    C = C[ix, :]
    
    # Remove missing series from covariance matrix
    R = R[np.ix_(ix, ix)]
    
    return y, C, R, L


def skf(Y: np.ndarray, A: np.ndarray, C: np.ndarray, Q: np.ndarray, 
        R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> KalmanFilterState:
    """Apply Kalman filter (forward pass).
    
    Parameters:
    -----------
    Y : np.ndarray
        Input data (k x nobs), where k = number of series, nobs = time periods
    A : np.ndarray
        Transition matrix (m x m)
    C : np.ndarray
        Observation matrix (k x m)
    Q : np.ndarray
        Covariance for transition equation residuals (m x m)
    R : np.ndarray
        Covariance for observation matrix residuals (k x k)
    Z_0 : np.ndarray
        Initial state vector (m,)
    V_0 : np.ndarray
        Initial state covariance matrix (m x m)
        
    Returns:
    --------
    KalmanFilterState
        Filter state with prior and posterior estimates
    """
    # Dimensions
    k, nobs = Y.shape  # k series, nobs time periods
    m = C.shape[1]     # m factors
    
    # Initialize output
    Zm = np.full((m, nobs), np.nan)       # Z_t | t-1 (prior)
    Vm = np.full((m, m, nobs), np.nan)    # V_t | t-1 (prior)
    ZmU = np.full((m, nobs + 1), np.nan)  # Z_t | t (posterior/updated)
    VmU = np.full((m, m, nobs + 1), np.nan)  # V_t | t (posterior/updated)
    loglik = 0.0
    
    # Set initial values
    Zu = Z_0.copy()  # Z_0|0 (In loop, Zu gives Z_t | t)
    Vu = V_0.copy()  # V_0|0 (In loop, Vu gives V_t | t)
    
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
    Y_t = np.array([])  # Initialize Y_t to empty array
    C_t = None
    VCF = None
    
    # Kalman filter procedure
    for t in range(nobs):
        # Calculate prior distribution
        # Use transition equation to create prior estimate for factor
        # i.e. Z = Z_t|t-1
        # Check for NaN/Inf in inputs
        if not _check_finite(Zu, f"Zu at t={t}"):
            _logger.warning(f"skf: Zu contains NaN/Inf at t={t}, resetting to zeros")
            Zu = np.zeros_like(Zu)
        
        Z = A @ Zu
        
        # Check for NaN/Inf in Z
        if not _check_finite(Z, f"Z at t={t}"):
            _logger.warning(f"skf: Z contains NaN/Inf at t={t}, using previous Zu")
            Z = Zu.copy()
        
        # Prior covariance matrix of Z (i.e. V = V_t|t-1)
        # Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(u) = A*Vu*A' + Q
        V = A @ Vu @ A.T + Q
        
        # Check for NaN/Inf before stabilization
        if not _check_finite(V, f"V at t={t}"):
            # Fallback: use previous covariance with regularization
            V = Vu + np.eye(V.shape[0]) * 1e-6
        
        # Ensure V is real, symmetric, and positive semi-definite
        V = _ensure_covariance_stable(V, min_eigenval=1e-8, ensure_real=True)
        
        # Calculate posterior distribution
        # Remove missing series: These are removed from Y, C, and R
        Y_t, C_t, R_t, _ = miss_data(Y[:, t], C, R)
        
        # Check if y_t contains no data
        if len(Y_t) == 0:
            Zu = Z
            Vu = V
        else:
            # Steps for variance and population regression coefficients:
            # Var(c_t*Z_t + e_t) = c_t Var(Z) c_t' + Var(e) = c_t*V*c_t' + R
            VC = V @ C_t.T
            
            # Compute innovation covariance F = C_t @ V @ C_t.T + R_t
            F = C_t @ VC + R_t
            
            # Ensure F is real, symmetric, and positive semi-definite
            F = _ensure_covariance_stable(F, min_eigenval=1e-8, ensure_real=True)
            
            # Check for NaN/Inf before inversion
            if not _check_finite(F, f"F at t={t}"):
                # Fallback: use identity with large variance
                F = np.eye(F.shape[0]) * 1e6
                _logger.warning(f"skf: F matrix contains NaN/Inf at t={t}, using fallback")
            
            try:
                iF = inv(F)
            except (np.linalg.LinAlgError, ValueError) as e:
                # Matrix inversion failed - use pseudo-inverse with regularization
                F_reg = F + np.eye(F.shape[0]) * 1e-6
                iF = pinv(F_reg)
                _logger.warning(f"skf: F inversion failed at t={t}, using pinv: {type(e).__name__}")
            
            # Matrix of population regression coefficients (Kalman gain)
            VCF = VC @ iF
            
            # Difference between actual and predicted observation matrix values
            innov = Y_t - C_t @ Z
            
            # Check for NaN/Inf in innovation
            if not _check_finite(innov, f"innovation at t={t}"):
                _logger.warning(f"skf: Innovation contains NaN/Inf at t={t}, skipping update")
                Zu = Z
                Vu = V
            else:
                # Update estimate of factor values (posterior)
                Zu = Z + VCF @ innov
                
                # Clean NaN/Inf only (remove excessive clipping during iterations)
                if not _check_finite(Zu, f"Zu at t={t}"):
                    Zu = _clean_matrix(Zu, 'general', default_nan=0.0, default_inf=0.0)
                
                # Update covariance matrix (posterior) for time t
                Vu = V - VCF @ VC.T
                
                # Clean NaN/Inf before stabilization
                if not _check_finite(Vu, f"Vu at t={t}"):
                    Vu = _clean_matrix(Vu, 'general', default_nan=1e-8, default_inf=1e6)
                
                # Check for NaN/Inf after cleaning
                if not _check_finite(Vu, f"Vu at t={t}"):
                    _logger.warning(f"skf: Vu contains NaN/Inf at t={t}, using V as fallback")
                    Vu = V.copy()
                
                # Ensure Vu is real, symmetric, and positive semi-definite
                Vu = _ensure_covariance_stable(Vu, min_eigenval=1e-8, ensure_real=True)
                
                # Update log-likelihood (with safeguards)
                try:
                    det_iF = _safe_determinant(iF, use_logdet=True)
                    if det_iF > 0 and np.isfinite(det_iF):
                        log_det = np.log(det_iF)
                        innov_term = innov.T @ iF @ innov
                        if np.isfinite(innov_term):
                            loglik += 0.5 * (log_det - innov_term)
                        else:
                            _logger.debug(f"skf: innov_term not finite at t={t}, skipping loglik update")
                    else:
                        _logger.debug(f"skf: det(iF) <= 0 or not finite at t={t}, skipping loglik update")
                except (np.linalg.LinAlgError, ValueError, OverflowError):
                    _logger.debug(f"skf: Log-likelihood calculation failed at t={t}")
        
        # Store output
        # Store covariance and observation values for t (priors)
        # Ensure Z and V are real before storing
        Z = _ensure_real(Z)
        V = _ensure_real_and_symmetric(V)
        Zm[:, t] = Z
        Vm[:, :, t] = V
        
        # Store covariance and state values for t (posteriors)
        # i.e. Zu = Z_t|t   & Vu = V_t|t
        Zu = _ensure_real(Zu)
        Vu = _ensure_real_and_symmetric(Vu)
        ZmU[:, t + 1] = Zu
        VmU[:, :, t + 1] = Vu
    
    # Store Kalman gain k_t (from final iteration)
    # k_t should be m x n_obs where n_obs is number of observed series at final time
    # VCF is m x n_obs, C_t is n_obs x m, so VCF @ C_t gives m x m
    # However, if no observations at final time, use zeros
    if len(Y_t) == 0:
        k_t = np.zeros((m, m))
    else:
        # VCF is m x n_obs, C_t is n_obs x m, so k_t = VCF @ C_t is m x m
        k_t = VCF @ C_t
    
    return KalmanFilterState(Zm=Zm, Vm=Vm, ZmU=ZmU, VmU=VmU, loglik=loglik, k_t=k_t)


def fis(A: np.ndarray, S: KalmanFilterState) -> KalmanFilterState:
    """Apply fixed-interval smoother (backward pass).
    
    Parameters:
    -----------
    A : np.ndarray
        Transition matrix (m x m)
    S : KalmanFilterState
        State from Kalman filter (SKF)
        
    Returns:
    --------
    KalmanFilterState
        State with smoothed estimates added (ZmT, VmT, VmT_1)
    """
    m, nobs = S.Zm.shape
    
    # Initialize output matrices
    ZmT = np.zeros((m, nobs + 1))
    VmT = np.zeros((m, m, nobs + 1))
    
    # Fill the final period of ZmT, VmT with SKF posterior values
    ZmT[:, nobs] = S.ZmU[:, nobs]
    VmT[:, :, nobs] = S.VmU[:, :, nobs]
    
    # Initialize VmT_1 lag 1 covariance matrix for final period
    VmT_1 = np.zeros((m, m, nobs))
    VmT_1_temp = (np.eye(m) - S.k_t) @ A @ S.VmU[:, :, nobs - 1]
    VmT_1[:, :, nobs - 1] = _ensure_real_and_symmetric(VmT_1_temp)
    
    # Used for recursion process
    J_2 = S.VmU[:, :, nobs - 1] @ A.T @ pinv(S.Vm[:, :, nobs - 1])
    
    # Run smoothing algorithm
    # Loop through time reverse-chronologically (starting at final period nobs-1)
    for t in range(nobs - 1, -1, -1):
        # Store posterior and prior factor covariance values
        VmU = S.VmU[:, :, t]
        Vm1 = S.Vm[:, :, t]
        
        # Store previous period smoothed factor covariance and lag-1 covariance
        V_T = VmT[:, :, t + 1]
        V_T1 = VmT_1[:, :, t] if t < nobs - 1 else np.zeros((m, m))
        
        J_1 = J_2
        
        # Update smoothed factor estimate
        ZmT[:, t] = S.ZmU[:, t] + J_1 @ (ZmT[:, t + 1] - A @ S.ZmU[:, t])
        
        # Clean NaN/Inf only (remove excessive clipping)
        if not _check_finite(ZmT[:, t], f"ZmT[:, t] at t={t}"):
            ZmT[:, t] = _clean_matrix(ZmT[:, t], 'general', default_nan=0.0, default_inf=0.0)
        
        # Update smoothed factor covariance matrix
        VmT_temp = VmU + J_1 @ (V_T - Vm1) @ J_1.T
        VmT[:, :, t] = _ensure_real_and_symmetric(VmT_temp)
        
        # Clean NaN/Inf and ensure PSD (keep only critical regularization)
        if not _check_finite(VmT[:, :, t], f"VmT[:, :, t] at t={t}"):
            VmT[:, :, t] = _clean_matrix(VmT[:, :, t], 'general', default_nan=1e-8, default_inf=1e6)
        
        if t > 0:
            # Update weight
            J_2 = S.VmU[:, :, t - 1] @ A.T @ pinv(S.Vm[:, :, t - 1])
            
            # Update lag 1 factor covariance matrix 
            VmT_1_temp = VmU @ J_2.T + J_1 @ (V_T1 - A @ VmU) @ J_2.T
            VmT_1[:, :, t - 1] = _ensure_real_and_symmetric(VmT_1_temp)
    
    # Add smoothed estimates as attributes
    S.ZmT = ZmT
    S.VmT = VmT
    S.VmT_1 = VmT_1
    
    return S


def run_kf(Y: np.ndarray, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
           R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply Kalman filter and fixed-interval smoother.
    
    Parameters:
    -----------
    Y : np.ndarray
        Input data (k x nobs)
    A : np.ndarray
        Transition matrix (m x m)
    C : np.ndarray
        Observation matrix (k x m)
    Q : np.ndarray
        Covariance for transition residuals (m x m)
    R : np.ndarray
        Covariance for observation residuals (k x k)
    Z_0 : np.ndarray
        Initial state (m,)
    V_0 : np.ndarray
        Initial covariance (m x m)
        
    Returns:
    --------
    zsmooth : np.ndarray
        Smoothed factor estimates (m x (nobs+1)), zsmooth[:, t+1] = Z_t|T
    Vsmooth : np.ndarray
        Smoothed factor covariance (m x m x (nobs+1)), Vsmooth[:, :, t+1] = Cov(Z_t|T)
    VVsmooth : np.ndarray
        Lag 1 factor covariance (m x m x nobs), Cov(Z_t, Z_t-1|T)
    loglik : float
        Log-likelihood
    """
    # Kalman filter
    S = skf(Y, A, C, Q, R, Z_0, V_0)
    
    # Fixed-interval smoother
    S = fis(A, S)
    
    # Organize output
    zsmooth = S.ZmT
    Vsmooth = S.VmT
    VVsmooth = S.VmT_1
    loglik = S.loglik
    
    # Ensure loglik is real and finite
    loglik = _ensure_real(np.array([loglik]))[0] if np.iscomplexobj(loglik) else loglik
    if not np.isfinite(loglik):
        loglik = -np.inf
    
    return zsmooth, Vsmooth, VVsmooth, loglik


# ============================================================================
# Numeric Utilities
# ============================================================================

def _ensure_square_matrix(M: np.ndarray, method: str = 'diag') -> np.ndarray:
    """Ensure matrix is square by extracting diagonal if needed."""
    if M.size == 0:
        return M
    if M.shape[0] != M.shape[1]:
        if method == 'diag':
            return np.diag(np.diag(M))
        elif method == 'eye':
            size = max(M.shape[0], M.shape[1])
            return np.eye(size)
    return M


def _ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose."""
    return 0.5 * (M + M.T)


def _ensure_real(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real by extracting real part if complex."""
    if np.iscomplexobj(M):
        return np.real(M)
    return M


def _ensure_real_and_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real and symmetric."""
    M = _ensure_real(M)
    M = _ensure_symmetric(M)
    return M


def _ensure_covariance_stable(M: np.ndarray, min_eigenval: float = 1e-8,
                               ensure_real: bool = True) -> np.ndarray:
    """Ensure covariance matrix is real, symmetric, and positive semi-definite."""
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    # Step 1: Ensure real (if needed)
    if ensure_real:
        M = _ensure_real(M)
    
    # Step 2: Ensure symmetric and positive semi-definite
    M, _ = _ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
    
    return M


def _clean_matrix(M: np.ndarray, matrix_type: str = 'general', 
                  default_nan: float = 0.0, default_inf: Optional[float] = None) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability."""
    if matrix_type == 'covariance':
        M = np.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = _ensure_symmetric(M)
        try:
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < MIN_EIGENVAL_CLEAN:
                M = M + np.eye(M.shape[0]) * (MIN_EIGENVAL_CLEAN - min_eigenval)
                M = _ensure_symmetric(M)
        except (np.linalg.LinAlgError, ValueError):
            M = M + np.eye(M.shape[0]) * MIN_EIGENVAL_CLEAN
            M = _ensure_symmetric(M)
    elif matrix_type == 'diagonal':
        diag = np.diag(M)
        diag = np.nan_to_num(diag, nan=default_nan, 
                            posinf=default_inf if default_inf is not None else 1e4,
                            neginf=default_nan)
        diag = np.maximum(diag, MIN_DIAGONAL_VARIANCE)
        M = np.diag(diag)
    elif matrix_type == 'loading':
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M


def _ensure_positive_definite(M: np.ndarray, min_eigenval: float = 1e-8, 
                              warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ensure matrix is positive semi-definite by adding regularization if needed."""
    M = _ensure_symmetric(M)
    stats = {
        'regularized': False,
        'min_eigenval_before': None,
        'reg_amount': 0.0,
        'min_eigenval_after': None
    }
    if M.size == 0 or M.shape[0] == 0:
        return M, stats
    try:
        eigenvals = np.linalg.eigvalsh(M)
        min_eig = float(np.min(eigenvals))
        stats['min_eigenval_before'] = float(min_eig)
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0]) * reg_amount
            M = _ensure_symmetric(M)
            stats['regularized'] = True
            stats['reg_amount'] = float(reg_amount)
            eigenvals_after = np.linalg.eigvalsh(M)
            stats['min_eigenval_after'] = float(np.min(eigenvals_after))
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal. This biases the covariance matrix."
                )
        else:
            stats['min_eigenval_after'] = float(min_eig)
    except (np.linalg.LinAlgError, ValueError) as e:
        M = M + np.eye(M.shape[0]) * min_eigenval
        M = _ensure_symmetric(M)
        stats['regularized'] = True
        stats['reg_amount'] = float(min_eigenval)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed: {e}). "
                f"Added {min_eigenval:.2e} to diagonal. This biases the covariance matrix."
            )
    return M, stats


def _check_finite(array: np.ndarray, name: str = "array", raise_on_invalid: bool = False) -> bool:
    """Check if array contains only finite values."""
    has_nan = np.any(np.isnan(array))
    has_inf = np.any(np.isinf(array))
    
    if has_nan or has_inf:
        msg = f"{name} contains "
        issues = []
        if has_nan:
            issues.append(f"{np.sum(np.isnan(array))} NaN values")
        if has_inf:
            issues.append(f"{np.sum(np.isinf(array))} Inf values")
        msg += " and ".join(issues)
        
        if raise_on_invalid:
            raise ValueError(msg)
        else:
            _logger.warning(msg)
        return False
    return True


def _safe_determinant(M: np.ndarray, use_logdet: bool = True) -> float:
    """Compute determinant safely to avoid overflow warnings.
    
    Uses log-determinant computation for large matrices or matrices with high
    condition numbers to avoid numerical overflow. For positive semi-definite
    matrices, uses Cholesky decomposition which is more stable.
    
    Parameters
    ----------
    M : np.ndarray
        Square matrix for which to compute determinant
    use_logdet : bool
        Whether to use log-determinant computation (default: True)
        
    Returns
    -------
    det : float
        Determinant of M, or 0.0 if computation fails
    """
    if M.size == 0 or M.shape[0] == 0:
        return 0.0
    
    if M.shape[0] != M.shape[1]:
        _logger.debug("_safe_determinant: non-square matrix, returning 0.0")
        return 0.0
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(M)):
        _logger.debug("_safe_determinant: matrix contains NaN/Inf, returning 0.0")
        return 0.0
    
    # For small matrices (1x1 or 2x2), direct computation is safe
    if M.shape[0] <= 2:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                det = np.linalg.det(M)
                if np.isfinite(det):
                    return float(det)
        except (RuntimeWarning, OverflowError):
            pass
        # Fall through to log-determinant
    
    # Check condition number to decide on method
    try:
        eigenvals = np.linalg.eigvals(M)
        eigenvals = eigenvals[np.isfinite(eigenvals)]
        if len(eigenvals) > 0:
            max_eig = np.max(np.abs(eigenvals))
            min_eig = np.max(np.abs(eigenvals[eigenvals != 0])) if np.any(eigenvals != 0) else max_eig
            cond_num = max_eig / max(min_eig, 1e-12)
        else:
            cond_num = np.inf
    except (np.linalg.LinAlgError, ValueError):
        cond_num = np.inf
    
    # Use log-determinant for large condition numbers or if requested
    if use_logdet or cond_num > 1e10:
        try:
            # Try Cholesky decomposition first (more stable for PSD matrices)
            try:
                L = np.linalg.cholesky(M)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                # Check if log_det is too large to avoid overflow in exp
                if log_det > 700:  # exp(700) is near float64 max
                    _logger.debug("_safe_determinant: log_det too large, returning 0.0")
                    return 0.0
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    det = np.exp(log_det)
                if np.isfinite(det) and det > 0:
                    return float(det)
            except np.linalg.LinAlgError:
                # Not PSD: fall back to slogdet for general matrices
                try:
                    sign, log_det = np.linalg.slogdet(M)
                    # If determinant is non-positive or invalid, return 0.0
                    if not np.isfinite(log_det) or sign <= 0:
                        return 0.0
                    # Avoid overflow in exp
                    if log_det > 700:
                        _logger.debug("_safe_determinant: log_det too large, returning 0.0")
                        return 0.0
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        det = np.exp(log_det)
                    if np.isfinite(det):
                        return float(det)
                except (np.linalg.LinAlgError, ValueError, OverflowError):
                    pass
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            pass
    
    # Fallback: direct computation with exception handling
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            det = np.linalg.det(M)
            if np.isfinite(det):
                return float(det)
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        pass
    
    _logger.debug("_safe_determinant: all methods failed, returning 0.0")
    return 0.0


# Re-export for backward compatibility
# These functions are used by other modules
def _compute_principal_components(cov_matrix: np.ndarray, n_components: int,
                                   block_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition with fallbacks."""
    try:
        from scipy.sparse.linalg import eigs
        from scipy.sparse import csc_matrix
        SCIPY_SPARSE_AVAILABLE = True
    except ImportError:
        SCIPY_SPARSE_AVAILABLE = False
        eigs = None
        csc_matrix = None
    
    if cov_matrix.size == 1:
        eigenvector = np.array([[1.0]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else DEFAULT_VARIANCE_FALLBACK
        return np.array([eigenvalue]), eigenvector
    
    n_series = cov_matrix.shape[0]
    
    # Strategy 1: Sparse eigs when feasible
    if n_components < n_series - 1 and SCIPY_SPARSE_AVAILABLE:
        try:
            cov_sparse = csc_matrix(cov_matrix)
            eigenvalues, eigenvectors = eigs(cov_sparse, k=n_components, which='LM')
            eigenvectors = eigenvectors.real
            if np.any(~np.isfinite(eigenvalues)) or np.any(~np.isfinite(eigenvectors)):
                raise ValueError("Invalid eigenvalue results")
            return eigenvalues.real, eigenvectors
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if block_idx is not None:
                _logger.warning(
                    f"init_conditions: Sparse eigendecomposition failed for block {block_idx+1}, "
                    f"falling back to np.linalg.eig. Error: {type(e).__name__}"
                )
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
            return eigenvalues[sort_idx].real, eigenvectors[:, sort_idx].real
    
    # Strategy 2: Full eig
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        valid_mask = np.isfinite(eigenvalues)
        if np.sum(valid_mask) < n_components:
            raise ValueError("Not enough valid eigenvalues")
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask]
        sort_idx = np.argsort(np.abs(valid_eigenvalues))[::-1][:n_components]
        return valid_eigenvalues[sort_idx].real, valid_eigenvectors[:, sort_idx].real
    except (IndexError, ValueError, np.linalg.LinAlgError) as e:
        if block_idx is not None:
            _logger.warning(
                f"init_conditions: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = np.eye(n_series)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def _compute_covariance_safe(data: np.ndarray, rowvar: bool = True, 
                              pairwise_complete: bool = False,
                              min_eigenval: float = 1e-8,
                              fallback_to_identity: bool = True) -> np.ndarray:
    """Compute covariance matrix safely with robust error handling."""
    if data.size == 0:
        if fallback_to_identity:
            return np.eye(1) if data.ndim == 1 else np.eye(data.shape[1] if rowvar else data.shape[0])
        raise ValueError("Cannot compute covariance: data is empty")
    
    # Handle 1D case
    if data.ndim == 1:
        var_val = _compute_variance_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE, 
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = _compute_variance_safe(series_data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Compute covariance
    try:
        if pairwise_complete:
            # Pairwise complete covariance: compute covariance for each pair separately
            if rowvar:
                data_for_cov = data.T  # Transpose to (N, T) for np.cov
            else:
                data_for_cov = data
            
            # Compute pairwise complete covariance manually
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    var_i = data_for_cov[i, :]
                    var_j = data_for_cov[j, :]
                    complete_mask = np.isfinite(var_i) & np.isfinite(var_j)
                    if np.sum(complete_mask) < 2:
                        if i == j:
                            cov[i, j] = DEFAULT_VARIANCE_FALLBACK
                        else:
                            cov[i, j] = 0.0
                    else:
                        var_i_complete = var_i[complete_mask]
                        var_j_complete = var_j[complete_mask]
                        if i == j:
                            cov[i, j] = np.var(var_i_complete, ddof=0)
                        else:
                            mean_i = np.mean(var_i_complete)
                            mean_j = np.mean(var_j_complete)
                            cov[i, j] = np.mean((var_i_complete - mean_i) * (var_j_complete - mean_j))
                            cov[j, i] = cov[i, j]  # Symmetric
            
            # Ensure minimum variance
            np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE_COVARIANCE))
        else:
            # Standard covariance (listwise deletion)
            if rowvar:
                complete_rows = np.all(np.isfinite(data), axis=1)
                if np.sum(complete_rows) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[complete_rows, :]
                data_for_cov = data_clean.T  # (N, T)
                cov = np.cov(data_for_cov, rowvar=True)  # Returns (N, N)
            else:
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[:, complete_cols]
                data_for_cov = data_clean.T  # (T, N)
                cov = np.cov(data_for_cov, rowvar=False)  # Returns (N, N)
            
            # np.cov can sometimes return unexpected shapes, so verify
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                if len(cov) == n_vars:
                    cov = np.diag(cov)
                else:
                    raise ValueError(f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})")
        
        # Ensure correct shape
        if cov.shape != (n_vars, n_vars):
            raise ValueError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}"
            )
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise ValueError("Covariance contains non-finite values")
        
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            eye_matrix = np.eye(n_vars)
            cov = cov + eye_matrix * reg_amount
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return np.eye(n_vars)
        raise


def _compute_variance_safe(data: np.ndarray, ddof: int = 0, 
                           min_variance: float = MIN_VARIANCE_COVARIANCE,
                           default_variance: float = DEFAULT_VARIANCE_FALLBACK) -> float:
    """Compute variance safely with robust error handling."""
    if data.size == 0:
        return default_variance
    
    # Flatten if 2D
    if data.ndim > 1:
        data = data.flatten()
    
    # Compute variance with NaN handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_val = np.nanvar(data, ddof=ddof)
    
    # Validate and enforce minimum
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    
    return float(var_val)


def _estimate_ar_coefficient(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
                             vsmooth_sum: Optional[np.ndarray] = None,
                             T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR coefficients and innovation variances from expectations."""
    if np.isscalar(EZZ_FB):
        EZZ_FB = np.array([EZZ_FB])
        EZZ_BB = np.array([EZZ_BB])
    if EZZ_FB.ndim > 1:
        EZZ_FB_diag = np.diag(EZZ_FB).copy()
        EZZ_BB_diag = np.diag(EZZ_BB).copy()
    else:
        EZZ_FB_diag = EZZ_FB.copy()
        EZZ_BB_diag = EZZ_BB.copy()
    if vsmooth_sum is not None:
        if vsmooth_sum.ndim > 1:
            vsmooth_diag = np.diag(vsmooth_sum)
        else:
            vsmooth_diag = vsmooth_sum
        EZZ_BB_diag = EZZ_BB_diag + vsmooth_diag
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * MIN_DIAGONAL_VARIANCE, MIN_VARIANCE_COVARIANCE)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    # Use _clean_matrix for consistency
    if EZZ_FB_diag.ndim == 0:
        EZZ_FB_diag_clean = _clean_matrix(np.array([EZZ_FB_diag]), 'general', default_nan=0.0, default_inf=1e6)
        EZZ_FB_diag = EZZ_FB_diag_clean[0] if EZZ_FB_diag_clean.size > 0 else 0.0
    else:
        EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0, default_inf=1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    Q_diag = None
    return A_diag, Q_diag


def _clip_ar_coefficients(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
                         warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds."""
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def _apply_ar_clipping(A: np.ndarray, config: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration."""
    if config is None:
        return _clip_ar_coefficients(A, -0.99, 0.99, True)
    
    from .helpers import safe_get_attr
    
    clip_enabled = safe_get_attr(config, 'clip_ar_coefficients', True)
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = safe_get_attr(config, 'ar_clip_min', -0.99)
    max_val = safe_get_attr(config, 'ar_clip_max', 0.99)
    warn = safe_get_attr(config, 'warn_on_ar_clip', True)
    return _clip_ar_coefficients(A, min_val, max_val, warn)

