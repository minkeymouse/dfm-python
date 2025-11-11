"""Estimation helpers for AR coefficients, innovation covariance, and sufficient statistics."""

from typing import Tuple
import numpy as np
import logging
from scipy.linalg import inv, pinv
from .utils import NUMERICAL_EXCEPTIONS
from ..numeric import _clean_matrix

_logger = logging.getLogger(__name__)


def estimate_ar_coefficients_ols(
    z: np.ndarray,
    Z_lag: np.ndarray,
    use_pinv: bool = False,
    pinv_cond: float = 1e-8
) -> Tuple[np.ndarray, bool]:
    """Estimate AR coefficients via ordinary least squares (OLS).
    
    Parameters
    ----------
    z : np.ndarray
        Dependent variable (T x r) - current factor values
    Z_lag : np.ndarray
        Independent variables (T x rp) - lagged factor values
    use_pinv : bool, default False
        If True, use pseudo-inverse. If False, use regular inverse with fallback.
    pinv_cond : float, default 1e-8
        Condition number threshold for pseudo-inverse
        
    Returns
    -------
    ar_coeffs : np.ndarray
        Estimated AR coefficients (r x rp)
    success : bool
        True if estimation succeeded, False if fallback was used
        
    Notes
    -----
    - Returns zero coefficients if OLS fails (MATLAB behavior)
    - Uses pseudo-inverse as fallback for numerical stability
    - Generic pattern used for both factor and idiosyncratic AR estimation
    """
    if z.size == 0 or Z_lag.size == 0 or Z_lag.shape[0] == 0 or Z_lag.shape[1] == 0:
        r = z.shape[1] if z.ndim > 1 else 1
        rp = Z_lag.shape[1] if Z_lag.ndim > 1 else 1
        return np.zeros((r, rp)), False
    
    try:
        if use_pinv:
            # Use rcond for scipy compatibility
            ar_coeffs = pinv(Z_lag, rcond=pinv_cond) @ z
        else:
            ZTZ = Z_lag.T @ Z_lag
            ar_coeffs = inv(ZTZ) @ Z_lag.T @ z
        return ar_coeffs.T if ar_coeffs.ndim > 1 else ar_coeffs.reshape(1, -1), True
    except NUMERICAL_EXCEPTIONS:
        r = z.shape[1] if z.ndim > 1 else 1
        rp = Z_lag.shape[1]
        return np.zeros((r, rp)), False


def compute_innovation_covariance(
    residuals: np.ndarray,
    default_variance: float = 0.1
) -> np.ndarray:
    """Compute innovation covariance from residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Innovation residuals (T x r) or (T,) for single series
    default_variance : float, default 0.1
        Default variance if computation fails
        
    Returns
    -------
    Q : np.ndarray
        Innovation covariance matrix (r x r) or (1 x 1) for single series
        
    Notes
    -----
    - Uses np.cov() for multiple series, np.var() for single series
    - Aligns with MATLAB: Q_i(1:r_i,1:r_i) = cov(e)
    """
    if residuals.size == 0:
        return np.array([[default_variance]])
    
    if residuals.ndim == 1 or residuals.shape[1] == 1:
        variance = np.var(residuals, ddof=0) if residuals.size > 0 else default_variance
        return np.array([[variance]])
    else:
        try:
            Q = np.cov(residuals.T)
            if np.any(~np.isfinite(Q)):
                return np.eye(residuals.shape[1]) * default_variance
            return Q
        except NUMERICAL_EXCEPTIONS:
            return np.eye(residuals.shape[1]) * default_variance


def compute_sufficient_stats(
    Zsmooth: np.ndarray,
    vsmooth: np.ndarray,
    vvsmooth: np.ndarray,
    block_indices: slice,
    T: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected sufficient statistics for EM updates.
    
    Computes E[Z_t Z_t'], E[Z_{t-1} Z_{t-1}'], and E[Z_t Z_{t-1}'] 
    from Kalman smoother output.
    
    Parameters
    ----------
    Zsmooth : np.ndarray
        Smoothed factor estimates (m x (T+1))
    vsmooth : np.ndarray
        Smoothed factor covariances (m x m x (T+1))
    vvsmooth : np.ndarray
        Smoothed lag-1 factor covariances (m x m x T)
    block_indices : slice
        Indices for the current block
    T : int
        Number of time periods
        
    Returns
    -------
    EZZ : np.ndarray
        E[Z_t Z_t' | Y] (r x r)
    EZZ_lag : np.ndarray
        E[Z_{t-1} Z_{t-1}' | Y] (rp x rp)
    EZZ_cross : np.ndarray
        E[Z_t Z_{t-1}' | Y] (r x rp)
    """
    # E[Z_t Z_t' | Y] = sum over t of (Z_t @ Z_t' + V_t)
    Z_block = Zsmooth[block_indices, 1:]
    V_block = vsmooth[block_indices, :, :][:, block_indices, 1:]
    EZZ = Z_block @ Z_block.T + np.sum(V_block, axis=2)
    
    # E[Z_{t-1} Z_{t-1}' | Y] = sum over t of (Z_{t-1} @ Z_{t-1}' + V_{t-1})
    Z_lag_block = Zsmooth[block_indices, :-1]
    V_lag_block = vsmooth[block_indices, :, :][:, block_indices, :-1]
    EZZ_lag = Z_lag_block @ Z_lag_block.T + np.sum(V_lag_block, axis=2)
    
    # E[Z_t Z_{t-1}' | Y] = sum over t of (Z_t @ Z_{t-1}' + VV_t)
    VV_block = vvsmooth[block_indices, :, :][:, block_indices, :]
    EZZ_cross = Z_block @ Z_lag_block.T + np.sum(VV_block, axis=2)
    
    return EZZ, EZZ_lag, EZZ_cross


def safe_mean_std(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation for each column, handling missing values.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (T x N)
        
    Returns
    -------
    means : np.ndarray
        Column means (N,)
    stds : np.ndarray
        Column standard deviations (N,)
        
    Notes
    -----
    - Handles missing values (NaN) by computing statistics only on finite values
    - Returns default values (mean=0, std=1) for columns with no valid data
    - Ensures std > 0 (minimum std = 1.0) to avoid division by zero in standardization
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    >>> means, stds = safe_mean_std(X)
    >>> means  # Column means
    array([3., 4., 6.])
    >>> stds  # Column standard deviations
    array([1.63299316, 1.63299316, 1.41421356])
    """
    n_series = matrix.shape[1]
    means = np.zeros(n_series)
    stds = np.ones(n_series)
    for j in range(n_series):
        col = matrix[:, j]
        mask = np.isfinite(col)
        if np.any(mask):
            means[j] = float(np.nanmean(col[mask]))
            std_val = float(np.nanstd(col[mask]))
            stds[j] = std_val if std_val > 0 else 1.0
        else:
            means[j] = 0.0
            stds[j] = 1.0
    return means, stds


def standardize_data(
    X: np.ndarray,
    clip_data_values: bool,
    data_clip_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize data and handle missing values.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (T x N)
    clip_data_values : bool
        Whether to clip extreme standardized values
    data_clip_threshold : float
        Threshold for clipping (in standard deviations)
    
    Returns
    -------
    x_standardized : np.ndarray
        Standardized data (T x N)
    Mx : np.ndarray
        Series means (N,)
    Wx : np.ndarray
        Series standard deviations (N,)
    """
    Mx, Wx = safe_mean_std(X)
    
    # Handle zero/near-zero standard deviations
    min_std = 1e-6
    Wx = np.maximum(Wx, min_std)
    
    # Handle NaN standard deviations
    nan_std_mask = np.isnan(Wx) | np.isnan(Mx)
    if np.any(nan_std_mask):
        _logger.warning(
            f"Series with NaN mean/std detected: {np.sum(nan_std_mask)}. "
            f"Setting Wx=1.0, Mx=0.0 for these series."
        )
        Wx[nan_std_mask] = 1.0
        Mx[nan_std_mask] = 0.0
    
    # Standardize
    x_standardized = (X - Mx) / Wx
    
    # Clip extreme values if enabled
    if clip_data_values:
        n_clipped_before = np.sum(np.abs(x_standardized) > data_clip_threshold)
        x_standardized = np.clip(x_standardized, -data_clip_threshold, data_clip_threshold)
        if n_clipped_before > 0:
            pct_clipped = 100.0 * n_clipped_before / x_standardized.size
            _logger.warning(
                f"Data value clipping applied: {n_clipped_before} values ({pct_clipped:.2f}%) "
                f"clipped beyond ±{data_clip_threshold} standard deviations."
            )
    
    # Replace any remaining NaN/Inf using consolidated utility
    default_inf_val = data_clip_threshold if clip_data_values else 100
    x_standardized = _clean_matrix(
        x_standardized,
        'general',
        default_nan=0.0,
        default_inf=default_inf_val
    )
    
    return x_standardized, Mx, Wx


# ============================================================================
# Parameter validation and covariance stabilization helpers
# ============================================================================

def validate_params(
    A: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    fallback_transition_coeff: float = 0.9,
    min_innovation_variance: float = 1e-8,
    default_observation_variance: float = 1e-4,
    default_idio_init_covariance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and clean DFM parameters for numerical stability.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix (m x m)
    Q : np.ndarray
        Innovation covariance (m x m)
    R : np.ndarray
        Observation covariance (n x n), typically diagonal
    C : np.ndarray
        Loading matrix (n x m)
    Z_0 : np.ndarray
        Initial state (m,)
    V_0 : np.ndarray
        Initial covariance (m x m)
    fallback_transition_coeff : float, default 0.9
        Fallback coefficient for transition matrix regularization
    min_innovation_variance : float, default 1e-8
        Minimum innovation variance for Q matrix
    default_observation_variance : float, default 1e-4
        Default observation variance for R matrix
    default_idio_init_covariance : float, default 0.1
        Default initial covariance for V_0 matrix
        
    Returns
    -------
    A_clean : np.ndarray
        Cleaned transition matrix
    Q_clean : np.ndarray
        Cleaned innovation covariance
    R_clean : np.ndarray
        Cleaned observation covariance
    C_clean : np.ndarray
        Cleaned loading matrix
    Z_0_clean : np.ndarray
        Cleaned initial state
    V_0_clean : np.ndarray
        Cleaned initial covariance
        
    Notes
    -----
    - Uses _check_finite and _clean_matrix for consistent cleaning
    - Applies parameter-specific fallback strategies
    - Logs warnings when cleaning is applied
    - Used in em_step for input validation
    """
    from ..numeric import _check_finite, _clean_matrix
    
    if not _check_finite(A, "A"):
        _logger.warning(
            f"validate_params: A contains NaN/Inf, "
            f"applying regularization ({fallback_transition_coeff}*I + {1-fallback_transition_coeff}*cleaned)"
        )
        A = (np.eye(A.shape[0]) * fallback_transition_coeff + 
             _clean_matrix(A, 'loading') * (1 - fallback_transition_coeff))
    
    if not _check_finite(Q, "Q"):
        _logger.warning(
            f"validate_params: Q contains NaN/Inf, "
            f"applying regularization (min={min_innovation_variance})"
        )
        Q = _clean_matrix(Q, 'covariance', default_nan=min_innovation_variance)
    
    if not _check_finite(R, "R"):
        _logger.warning(
            f"validate_params: R contains NaN/Inf, "
            f"applying regularization (min={default_observation_variance})"
        )
        R = _clean_matrix(R, 'diagonal', default_nan=default_observation_variance, default_inf=1e4)
    
    if not _check_finite(C, "C"):
        _logger.warning("validate_params: C contains NaN/Inf, cleaning matrix")
        C = _clean_matrix(C, 'loading')
    
    if not _check_finite(Z_0, "Z_0"):
        _logger.warning("validate_params: Z_0 contains NaN/Inf, resetting to zeros")
        Z_0 = np.zeros_like(Z_0)
    
    if not _check_finite(V_0, "V_0"):
        _logger.warning(
            f"validate_params: V_0 contains NaN/Inf, "
            f"using regularized identity ({default_idio_init_covariance}*I)"
        )
        V_0 = np.eye(V_0.shape[0]) * default_idio_init_covariance
    
    return A, Q, R, C, Z_0, V_0


def stabilize_cov(
    Q: np.ndarray,
    config: Optional['DFMConfig'],
    min_variance: float = 1e-8
) -> np.ndarray:
    """Apply standard stability operations to innovation covariance matrix.
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix (m x m)
    config : DFMConfig, optional
        Configuration object for stability parameters
    min_variance : float, default 1e-8
        Minimum variance to enforce on diagonal elements
        
    Returns
    -------
    Q_stable : np.ndarray
        Stabilized innovation covariance matrix
        
    Notes
    -----
    - Applies operations in order: clean -> PSD -> cap -> min variance
    - Re-applies min variance after each operation that might affect diagonal
    - Used in em_step for Q block updates
    """
    from ...config import DFMConfig
    from .utils import safe_get_attr
    from ..numeric import (
        _clean_matrix,
        _ensure_positive_definite,
        _cap_max_eigenvalue,
        _ensure_innovation_variance_minimum,
    )
    
    # Clean matrix first
    Q = _clean_matrix(Q, 'covariance', default_nan=0.0)
    
    # Get config parameters
    min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
    max_eigenval = safe_get_attr(config, "max_eigenvalue", 1e6)
    
    # Ensure minimum variance before PSD (critical for factor evolution)
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    # Apply positive definite enforcement
    Q, _ = _ensure_positive_definite(Q, min_eigenval, warn_reg)
    
    # Re-ensure minimum after PSD (it may have changed diagonal)
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    # Cap maximum eigenvalues
    Q = _cap_max_eigenvalue(Q, max_eigenval=max_eigenval)
    
    # Final minimum variance enforcement after capping
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    return Q

