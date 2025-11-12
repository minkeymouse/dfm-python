"""Kalman filter and fixed-interval smoother implementation."""

import numpy as np
from scipy.linalg import inv, pinv
from typing import Tuple, Dict
from dataclasses import dataclass
import logging

_logger = logging.getLogger(__name__)


def _get_numeric_utils():
    """Lazy import to avoid circular dependency."""
    from .core.numeric import (
        _ensure_real_and_symmetric,
        _ensure_covariance_stable,
        _ensure_real,
        _clean_matrix,
        _check_finite
    )
    return _ensure_real_and_symmetric, _ensure_covariance_stable, _ensure_real, _clean_matrix, _check_finite


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
        _, _ensure_covariance_stable, _, _, _check_finite = _get_numeric_utils()
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
            _, _ensure_covariance_stable, _, _, _check_finite = _get_numeric_utils()
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
                _, _ensure_covariance_stable, _, _clean_matrix, _check_finite = _get_numeric_utils()
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
                    from .core.numeric import _safe_determinant
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
        _ensure_real_and_symmetric, _, _ensure_real, _, _ = _get_numeric_utils()
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
    _ensure_real_and_symmetric, _, _, _, _ = _get_numeric_utils()
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
        _ensure_real_and_symmetric, _, _, _clean_matrix, _check_finite = _get_numeric_utils()
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
            _ensure_real_and_symmetric, _, _, _, _ = _get_numeric_utils()
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
    _, _, _ensure_real, _, _ = _get_numeric_utils()
    loglik = _ensure_real(np.array([loglik]))[0] if np.iscomplexobj(loglik) else loglik
    if not np.isfinite(loglik):
        loglik = -np.inf
    
    return zsmooth, Vsmooth, VVsmooth, loglik

