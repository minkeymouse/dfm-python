"""Helper functions organized by domain.

This package provides utility functions for common code patterns, organized
into domain-specific modules for better maintainability.
"""

from typing import Optional, Tuple, Any, Dict
import numpy as np
import logging
from scipy.linalg import inv, pinv, block_diag

from ...config import DFMConfig
from ..numeric import (
    _clean_matrix,
    _check_finite,
    _compute_regularization_param,
    _ensure_positive_definite,
    _cap_max_eigenvalue,
    _ensure_innovation_variance_minimum,
)

_logger = logging.getLogger(__name__)

# Common exception types for numerical operations
NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)


# ============================================================================
# Estimation helpers
# ============================================================================

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
    config: Optional[DFMConfig],
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


# ============================================================================
# Matrix operation helpers
# ============================================================================

def reg_inv(
    denom: np.ndarray,
    nom: np.ndarray,
    config: Optional[DFMConfig],
    default_scale: float = 1e-5
) -> Tuple[np.ndarray, bool]:
    """Compute regularized matrix inverse for loading updates.
    
    Parameters
    ----------
    denom : np.ndarray
        Denominator matrix (typically sum of Z @ Z' terms)
    nom : np.ndarray
        Numerator matrix or vector (typically sum of y @ Z' terms)
    config : DFMConfig, optional
        Configuration object for regularization parameters
    default_scale : float, default 1e-5
        Default regularization scale if not in config
        
    Returns
    -------
    result : np.ndarray
        Computed result: inv(denom_reg) @ nom
    success : bool
        True if computation succeeded, False if exception occurred
        
    Notes
    -----
    - Aligns with MATLAB: vec_C = inv(denom)*nom(:)
    - Uses regularization for numerical stability (MATLAB uses direct inversion)
    - Handles both matrix and vector numerators
    """
    try:
        scale_factor = safe_get_attr(config, "regularization_scale", default_scale)
        warn_reg = safe_get_attr(config, "warn_on_regularization", True)
        reg_param, _ = _compute_regularization_param(denom, scale_factor, warn_reg)
        denom_reg = denom + np.eye(denom.shape[0]) * reg_param
        
        # Align with MATLAB: vec_C = inv(denom)*nom(:)
        # MATLAB uses direct inversion, we use regularized for numerical stability
        if nom.ndim == 1:
            result = inv(denom_reg) @ nom
        elif nom.ndim == 2 and nom.shape[0] == 1:
            result = inv(denom_reg) @ nom.T
        else:
            result = inv(denom_reg) @ nom.flatten()
        return result, True
    except NUMERICAL_EXCEPTIONS:
        return np.zeros(denom.shape[0]), False


def update_loadings(
    C_new: np.ndarray,
    C_update: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray
) -> None:
    """Update loading matrix from computed values.
    
    Parameters
    ----------
    C_new : np.ndarray
        Loading matrix to update (modified in-place)
    C_update : np.ndarray
        Computed loading values (n_rows x n_cols)
    row_indices : np.ndarray
        Row indices in C_new to update
    col_indices : np.ndarray
        Column indices in C_new to update
        
    Notes
    -----
    - Updates C_new in-place
    - Assumes C_update.shape == (len(row_indices), len(col_indices))
    - More efficient than nested loops for large matrices
    """
    if len(row_indices) == 0 or len(col_indices) == 0:
        return
    
    # Use vectorized assignment for efficiency
    if C_update.shape == (len(row_indices), len(col_indices)):
        C_new[np.ix_(row_indices, col_indices)] = C_update
    else:
        # Fallback to element-wise assignment if shapes don't match
        for ii, row_idx in enumerate(row_indices):
            for jj, col_idx in enumerate(col_indices):
                if ii < C_update.shape[0] and jj < C_update.shape[1]:
                    C_new[row_idx, col_idx] = C_update[ii, jj]


def extract_3d_matrix_slice(
    matrix_3d: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    time_idx: int
) -> np.ndarray:
    """Extract 2D slice from 3D matrix with safe indexing.
    
    Parameters
    ----------
    matrix_3d : np.ndarray
        3D matrix (m x m x T) to extract from
    row_indices : np.ndarray
        Row indices to extract
    col_indices : np.ndarray
        Column indices to extract
    time_idx : int
        Time index (third dimension)
        
    Returns
    -------
    slice_2d : np.ndarray
        2D slice (len(row_indices) x len(col_indices))
        Returns zeros if indices are invalid
        
    Notes
    -----
    - Handles dimension reduction (3D -> 2D)
    - Returns zeros if time_idx is out of bounds
    - Validates that extracted slice has expected shape
    """
    if time_idx >= matrix_3d.shape[2]:
        return np.zeros((len(row_indices), len(col_indices)))
    
    try:
        slice_3d = matrix_3d[np.ix_(row_indices, col_indices, [time_idx])]
        slice_2d = slice_3d[:, :, 0] if slice_3d.ndim == 3 else slice_3d
        # Ensure correct shape
        expected_shape = (len(row_indices), len(col_indices))
        if slice_2d.shape != expected_shape:
            return np.zeros(expected_shape)
        return slice_2d
    except (IndexError, ValueError):
        return np.zeros((len(row_indices), len(col_indices)))


def compute_obs_cov(
    y: np.ndarray,
    C: np.ndarray,
    Zsmooth: np.ndarray,
    vsmooth: np.ndarray,
    default_variance: float = 1e-4,
    min_variance: float = 1e-8,
    min_diagonal_variance_ratio: float = 1e-6
) -> np.ndarray:
    """Compute observation covariance diagonal (R_diag) from residuals and factor uncertainty.
    
    Computes: R[i,i] = mean_t((y[i,t] - C[i,:] @ Z[t])^2 + C[i,:] @ V[t] @ C[i,:]')
    
    Parameters
    ----------
    y : np.ndarray
        Observation matrix (n x T), may contain NaN
    C : np.ndarray
        Loading matrix (n x m)
    Zsmooth : np.ndarray
        Smoothed factor estimates (m x (T+1))
    vsmooth : np.ndarray
        Smoothed factor covariances (m x m x (T+1))
    default_variance : float, default 1e-4
        Default variance if computation fails
    min_variance : float, default 1e-8
        Minimum variance to enforce
    min_diagonal_variance_ratio : float, default 1e-6
        Minimum variance as ratio of mean variance
        
    Returns
    -------
    R_diag : np.ndarray
        Diagonal elements of R (n,)
        
    Notes
    -----
    - Handles missing data (NaN) by skipping those time points
    - Computes both residual variance and factor uncertainty contribution
    - Enforces minimum variance for numerical stability
    - Uses median imputation for invalid values
    """
    n, T = y.shape
    R_diag = np.zeros(n)
    n_obs_per_series = np.zeros(n, dtype=int)
    
    for t in range(T):
        if not safe_time_index(t, Zsmooth.shape[0], offset=1):
            continue
        Z_t = Zsmooth[t + 1, :].reshape(-1, 1)
        vsmooth_t = vsmooth[:, :, t + 1]
        y_pred = (C @ Z_t).flatten()
        
        for i in range(n):
            if np.isnan(y[i, t]):
                continue
            n_obs_per_series[i] += 1
            resid_sq = (y[i, t] - y_pred[i]) ** 2
            C_i = C[i, :].reshape(1, -1)
            var_factor = (C_i @ vsmooth_t @ C_i.T)[0, 0]
            R_diag[i] += resid_sq + var_factor
    
    # Normalize by number of observations per series
    n_obs_per_series = np.maximum(n_obs_per_series, 1)
    R_diag = R_diag / n_obs_per_series
    
    # Enforce minimum variance
    mean_var = np.mean(R_diag[R_diag > 0]) if np.any(R_diag > 0) else default_variance
    min_var = np.maximum(mean_var * min_diagonal_variance_ratio, min_variance)
    R_diag = np.maximum(R_diag, min_var)
    
    # Handle invalid values (NaN/Inf/negative)
    valid_mask = np.isfinite(R_diag) & (R_diag > 0)
    if np.any(valid_mask):
        median_var = np.median(R_diag[valid_mask])
        R_diag = np.where(valid_mask, R_diag, median_var)
    else:
        R_diag.fill(default_variance)
    
    return R_diag


def clean_variance_array(
    variance_array: np.ndarray,
    default_value: float = 1e-4,
    min_value: Optional[float] = None,
    replace_nan: bool = True,
    replace_inf: bool = True,
    replace_negative: bool = True
) -> np.ndarray:
    """Clean variance array by replacing invalid values.
    
    Parameters
    ----------
    variance_array : np.ndarray
        Array of variance values to clean
    default_value : float, default 1e-4
        Default value to use for invalid entries
    min_value : float, optional
        Minimum value to enforce. If None, uses default_value.
    replace_nan : bool, default True
        Replace NaN values
    replace_inf : bool, default True
        Replace Inf values
    replace_negative : bool, default True
        Replace negative values
        
    Returns
    -------
    cleaned_array : np.ndarray
        Cleaned variance array with all invalid values replaced
        
    Notes
    -----
    - Uses median imputation if any valid values exist
    - Falls back to default_value if no valid values
    - Enforces minimum value after cleaning
    """
    cleaned = variance_array.copy()
    
    # Build mask for invalid values
    invalid_mask = np.zeros(cleaned.shape, dtype=bool)
    if replace_nan:
        invalid_mask |= np.isnan(cleaned)
    if replace_inf:
        invalid_mask |= np.isinf(cleaned)
    if replace_negative:
        invalid_mask |= (cleaned < 0)
    
    # Replace invalid values
    if np.any(invalid_mask):
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            # Use median of valid values
            median_val = np.median(cleaned[valid_mask])
            cleaned = np.where(invalid_mask, median_val, cleaned)
        else:
            # All invalid - use default
            cleaned[invalid_mask] = default_value
    
    # Enforce minimum value
    min_val = min_value if min_value is not None else default_value
    cleaned = np.maximum(cleaned, min_val)
    
    return cleaned


# ============================================================================
# Block structure helpers
# ============================================================================

def update_block_diag(
    A: Optional[np.ndarray],
    Q: Optional[np.ndarray],
    V_0: Optional[np.ndarray],
    A_block: np.ndarray,
    Q_block: np.ndarray,
    V_0_block: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update block diagonal matrices by appending new blocks.
    
    Parameters
    ----------
    A : np.ndarray, optional
        Existing transition matrix (m x m) or None if first block
    Q : np.ndarray, optional
        Existing innovation covariance (m x m) or None if first block
    V_0 : np.ndarray, optional
        Existing initial covariance (m x m) or None if first block
    A_block : np.ndarray
        New transition matrix block to append (k x k)
    Q_block : np.ndarray
        New innovation covariance block to append (k x k)
    V_0_block : np.ndarray
        New initial covariance block to append (k x k)
        
    Returns
    -------
    A_new : np.ndarray
        Updated transition matrix with new block appended
    Q_new : np.ndarray
        Updated innovation covariance with new block appended
    V_0_new : np.ndarray
        Updated initial covariance with new block appended
    """
    if A is None:
        return A_block, Q_block, V_0_block
    else:
        return (
            block_diag(A, A_block),
            block_diag(Q, Q_block),
            block_diag(V_0, V_0_block)
        )


def get_block_indices(
    blocks: np.ndarray,
    block_idx: int
) -> np.ndarray:
    """Get series indices for a specific block.
    
    Parameters
    ----------
    blocks : np.ndarray
        Block structure matrix (n x n_blocks)
    block_idx : int
        Index of the block (0-based)
        
    Returns
    -------
    indices : np.ndarray
        Array of series indices belonging to the block
    """
    return np.where(blocks[:, block_idx] == 1)[0]


def compute_block_slice_indices(
    r: np.ndarray,
    block_idx: int,
    ppC: int
) -> Tuple[int, int]:
    """Compute start and end indices for a block in state space.
    
    Parameters
    ----------
    r : np.ndarray
        Number of factors per block (n_blocks,)
    block_idx : int
        Index of current block (0-based)
    ppC : int
        Maximum of p (AR lag) and pC (tent length)
        
    Returns
    -------
    t_start : int
        Start index for block in state space
    t_end : int
        End index for block in state space (exclusive)
    """
    factor_start_idx = int(np.sum(r[:block_idx]) * ppC)
    r_i_int = int(r[block_idx])
    t_start = factor_start_idx
    t_end = factor_start_idx + r_i_int * ppC
    return t_start, t_end


def extract_block_matrix(
    matrix: np.ndarray,
    t_start: int,
    t_end: int,
    copy: bool = True
) -> np.ndarray:
    """Extract block submatrix from a larger matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Full matrix to extract from (m x m)
    t_start : int
        Start index for block
    t_end : int
        End index for block (exclusive)
    copy : bool, default True
        Whether to copy the submatrix (recommended to avoid aliasing)
        
    Returns
    -------
    block_matrix : np.ndarray
        Extracted block submatrix ((t_end-t_start) x (t_end-t_start))
    """
    block = matrix[t_start:t_end, t_start:t_end]
    return block.copy() if copy else block


def update_block_in_matrix(
    matrix: np.ndarray,
    block_matrix: np.ndarray,
    t_start: int,
    t_end: int
) -> None:
    """Update a block submatrix in a larger matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to update (modified in-place)
    block_matrix : np.ndarray
        Block matrix to insert
    t_start : int
        Start index for block
    t_end : int
        End index for block (exclusive)
        
    Notes
    -----
    - Updates matrix in-place
    - Assumes block_matrix.shape == (t_end-t_start, t_end-t_start)
    """
    matrix[t_start:t_end, t_start:t_end] = block_matrix


# ============================================================================
# Configuration access helpers
# ============================================================================

def safe_get_method(config: Optional[DFMConfig], method_name: str, default: Any = None) -> Any:
    """Safely get a method from config if it exists and is callable.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    method_name : str
        Name of the method to retrieve
    default : Any
        Default value if method doesn't exist or isn't callable
        
    Returns
    -------
    Any
        Method result if callable, else default value
        
    Notes
    -----
    - Use this for config methods that need to be called (e.g., `get_series_ids()`)
    - Returns default if method doesn't exist or isn't callable
    - Used throughout package for safe config method access
    - Prefer `safe_get_attr()` for simple attribute access (no method call needed)
    """
    if config is None:
        return default
    method = getattr(config, method_name, None)
    if method is not None and callable(method):
        return method()
    return default


def safe_get_attr(config: Optional[DFMConfig], attr_name: str, default: Any = None) -> Any:
    """Safely get an attribute from config if it exists.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    attr_name : str
        Name of the attribute to retrieve
    default : Any
        Default value if attribute doesn't exist
        
    Returns
    -------
    Any
        Attribute value or default
        
    Notes
    -----
    - Use this for simple attribute access (e.g., `min_eigenvalue`, `ar_clip_max`)
    - Returns default if attribute doesn't exist
    - Used throughout package for safe config attribute access
    - Prefer `safe_get_method()` for methods that need to be called
    """
    if config is None:
        return default
    return getattr(config, attr_name, default)


# ============================================================================
# Frequency-related helpers
# ============================================================================

def get_tent_weights(
    freq: str,
    clock: str,
    tent_weights_dict: Optional[Dict[str, np.ndarray]],
    logger: Optional[Any] = None
) -> Optional[np.ndarray]:
    """Safely get tent weights for a frequency pair with fallback generation.
    
    Parameters
    ----------
    freq : str
        Target frequency ('q', 'sa', 'a', etc.)
    clock : str
        Clock frequency ('m', 'q', etc.)
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights
    logger : logging.Logger, optional
        Logger instance for warnings. If None, uses module logger.
        
    Returns
    -------
    tent_weights : np.ndarray, optional
        Tent weights array, or None if cannot be determined
        
    Notes
    -----
    - Checks dictionary first, then generates symmetric weights if needed
    - Uses FREQUENCY_HIERARCHY to determine number of periods
    - Raises ValueError if frequency pair cannot be handled
    """
    from ...utils import (
        FREQUENCY_HIERARCHY,
        get_tent_weights_for_pair,
        generate_tent_weights,
    )
    
    if logger is None:
        logger = _logger
    
    # Try dictionary first
    if tent_weights_dict and freq in tent_weights_dict:
        return tent_weights_dict[freq]
    
    # Try helper function
    tent_weights = get_tent_weights_for_pair(freq, clock)
    if tent_weights is not None:
        return tent_weights
    
    # Generate symmetric weights as fallback
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    freq_h = FREQUENCY_HIERARCHY.get(freq, 3)
    n_periods_est = freq_h - clock_h + 1
    
    if 0 < n_periods_est <= 12:
        tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
        logger.warning(f"get_tent_weights: generated symmetric tent weights for '{freq}'")
        return tent_weights
    else:
        raise ValueError(f"get_tent_weights: cannot determine tent weights for '{freq}'")


def infer_nQ(
    frequencies: Optional[np.ndarray],
    clock: str
) -> int:
    """Infer number of slower-frequency series from frequencies array.
    
    Parameters
    ----------
    frequencies : np.ndarray, optional
        Array of frequency strings (n,) for each series
    clock : str
        Clock frequency ('m', 'q', 'sa', 'a')
        
    Returns
    -------
    nQ : int
        Number of series with frequency slower than clock frequency
        
    Notes
    -----
    - Returns 0 if frequencies is None
    - Uses FREQUENCY_HIERARCHY to compare frequencies
    - Clock frequency is typically 'm' (monthly)
    """
    from ...utils import FREQUENCY_HIERARCHY
    
    if frequencies is None:
        return 0
    
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    return sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_h)


# ============================================================================
# Array utility helpers
# ============================================================================

def safe_time_index(
    t: int,
    max_index: int,
    offset: int = 1
) -> bool:
    """Check if time index is valid for array access.
    
    Parameters
    ----------
    t : int
        Current time index (0-based)
    max_index : int
        Maximum valid index (typically array.shape[0] or array.shape[2])
    offset : int, default 1
        Offset to add to t (typically 1 for t+1 access)
        
    Returns
    -------
    bool
        True if t + offset < max_index, False otherwise
    """
    return (t + offset) < max_index


def safe_array_operation(
    array: Optional[np.ndarray],
    operation: str = 'size',
    default: Any = None
) -> Any:
    """Safely perform operations on potentially None arrays.
    
    Parameters
    ----------
    array : np.ndarray, optional
        Array to operate on (may be None)
    operation : str, default 'size'
        Operation to perform:
        - 'size': Return array.size if not None, else 0
        - 'shape': Return array.shape if not None, else None
        - 'len': Return len(array) if not None, else 0
        - 'is_empty': Return True if None or size == 0
    default : Any, optional
        Default value if array is None
        
    Returns
    -------
    result : Any
        Result of operation or default value
    """
    if array is None:
        if default is not None:
            return default
        if operation == 'size':
            return 0
        elif operation == 'shape':
            return None
        elif operation == 'len':
            return 0
        elif operation == 'is_empty':
            return True
        return None
    
    if operation == 'size':
        return array.size
    elif operation == 'shape':
        return array.shape
    elif operation == 'len':
        return len(array)
    elif operation == 'is_empty':
        return array.size == 0
    else:
        return getattr(array, operation, default)


def get_matrix_shape(
    matrix: Optional[np.ndarray],
    dim: Optional[int] = None
) -> Any:
    """Safely get matrix shape or specific dimension.
    
    Parameters
    ----------
    matrix : np.ndarray, optional
        Matrix to get shape from (may be None)
    dim : int, optional
        Specific dimension to return (0=rows, 1=cols, 2=depth).
        If None, returns full shape tuple.
        
    Returns
    -------
    shape : tuple or int or None
        Matrix shape tuple, specific dimension, or None if matrix is None
    """
    if matrix is None:
        return None
    
    if dim is None:
        return matrix.shape
    elif 0 <= dim < len(matrix.shape):
        return matrix.shape[dim]
    else:
        return None


def has_valid_data(
    array: Optional[np.ndarray],
    min_size: int = 1
) -> bool:
    """Check if array has valid data (not None and meets size requirement).
    
    Parameters
    ----------
    array : np.ndarray, optional
        Array to check
    min_size : int, default 1
        Minimum size required (total elements for size check)
        
    Returns
    -------
    bool
        True if array is not None and has at least min_size elements
    """
    if array is None:
        return False
    return array.size >= min_size


def ensure_minimum_size(
    array: np.ndarray,
    min_size: int,
    pad_value: float = 0.0,
    pad_axis: int = 0
) -> np.ndarray:
    """Ensure array has minimum size by padding if needed.
    
    Parameters
    ----------
    array : np.ndarray
        Array to check and pad
    min_size : int
        Minimum size required
    pad_value : float, default 0.0
        Value to use for padding
    pad_axis : int, default 0
        Axis to pad along (0=rows, 1=cols)
        
    Returns
    -------
    array : np.ndarray
        Array with at least min_size elements along pad_axis
    """
    current_size = array.shape[pad_axis]
    if current_size >= min_size:
        return array
    
    # Pad along specified axis
    if pad_axis == 0:
        padding = np.full((min_size - current_size, array.shape[1]), pad_value)
        return np.vstack([array, padding])
    elif pad_axis == 1:
        padding = np.full((array.shape[0], min_size - current_size), pad_value)
        return np.hstack([array, padding])
    else:
        return array


def append_or_initialize(
    existing: Optional[np.ndarray],
    new_array: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """Append array to existing or initialize if None.
    
    Parameters
    ----------
    existing : np.ndarray, optional
        Existing array to append to, or None if first element
    new_array : np.ndarray
        New array to append
    axis : int, default 1
        Axis for concatenation:
        - 1 or 'h': Use np.hstack (horizontal, default)
        - 0 or 'v': Use np.vstack (vertical)
        
    Returns
    -------
    result : np.ndarray
        Concatenated array or new_array if existing is None
    """
    if existing is None:
        return new_array
    
    if axis == 1 or axis == 'h':
        return np.hstack([existing, new_array])
    elif axis == 0 or axis == 'v':
        return np.vstack([existing, new_array])
    else:
        raise ValueError(f"Invalid axis: {axis}. Use 0/'v' for vertical or 1/'h' for horizontal")


def create_empty_matrix(
    shape: Tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Create an empty matrix with specified shape.
    
    Parameters
    ----------
    shape : tuple of int, default (0, 0)
        Shape of the empty matrix (rows, cols)
        
    Returns
    -------
    empty_matrix : np.ndarray
        Empty matrix with specified shape
    """
    return np.zeros(shape)


def reshape_to_column_vector(
    array: np.ndarray
) -> np.ndarray:
    """Reshape array to column vector.
    
    Parameters
    ----------
    array : np.ndarray
        Array to reshape (1D or 2D)
        
    Returns
    -------
    column_vector : np.ndarray
        Column vector (n, 1)
    """
    return array.reshape(-1, 1)


def reshape_to_row_vector(
    array: np.ndarray
) -> np.ndarray:
    """Reshape array to row vector.
    
    Parameters
    ----------
    array : np.ndarray
        Array to reshape (1D or 2D)
        
    Returns
    -------
    row_vector : np.ndarray
        Row vector (1, n)
    """
    return array.reshape(1, -1)


def pad_matrix_to_shape(
    matrix: np.ndarray,
    target_shape: Tuple[int, int],
    pad_value: float = 0.0,
    pad_axis: int = 0
) -> np.ndarray:
    """Pad matrix to target shape with specified value.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to pad
    target_shape : tuple of int
        Target shape (rows, cols)
    pad_value : float, default 0.0
        Value to use for padding
    pad_axis : int, default 0
        Axis to pad along:
        - 0: Pad rows (bottom)
        - 1: Pad columns (right)
        
    Returns
    -------
    padded_matrix : np.ndarray
        Padded matrix with target_shape
        
    Notes
    -----
    - Only pads if target_shape is larger than matrix.shape
    - Truncates if target_shape is smaller (for axis=0, truncates bottom rows)
    """
    current_shape = matrix.shape
    
    if pad_axis == 0:
        # Pad rows
        if target_shape[0] > current_shape[0]:
            padding = np.full((target_shape[0] - current_shape[0], current_shape[1]), pad_value)
            return np.vstack([matrix, padding])
        elif target_shape[0] < current_shape[0]:
            return matrix[:target_shape[0], :]
    elif pad_axis == 1:
        # Pad columns
        if target_shape[1] > current_shape[1]:
            padding = np.full((current_shape[0], target_shape[1] - current_shape[1]), pad_value)
            return np.hstack([matrix, padding])
        elif target_shape[1] < current_shape[1]:
            return matrix[:, :target_shape[1]]
    
    return matrix


def resolve_param(override: Any, default: Any) -> Any:
    """Resolve parameter: use override if provided, else use default.
    
    Parameters
    ----------
    override : Any
        Override value (used if not None)
    default : Any
        Default value (used if override is None)
        
    Returns
    -------
    Any
        Override value if not None, else default value
        
    Examples
    --------
    >>> resolve_param(10, 5)
    10
    >>> resolve_param(None, 5)
    5
    """
    return override if override is not None else default


def safe_numerical_operation(
    operation: Any,
    *args,
    fallback_value: Any = None,
    fallback_factory: Optional[Any] = None,
    log_warning: bool = True,
    warning_msg: Optional[str] = None
) -> Tuple[Any, bool]:
    """Safely execute a numerical operation with fallback on exception.
    
    Parameters
    ----------
    operation : callable
        Function to execute (will be called as operation(*args))
    *args
        Arguments to pass to operation
    fallback_value : Any, optional
        Value to return if operation fails
    fallback_factory : callable, optional
        Function to call (with no args) to generate fallback value
    log_warning : bool, default True
        Whether to log a warning on failure
    warning_msg : str, optional
        Custom warning message. If None, uses default.
        
    Returns
    -------
    result : Any
        Result of operation or fallback value
    success : bool
        True if operation succeeded, False if exception occurred
    """
    try:
        result = operation(*args)
        return result, True
    except NUMERICAL_EXCEPTIONS as e:
        if log_warning:
            msg = warning_msg or f"Numerical operation failed: {type(e).__name__}"
            _logger.warning(msg)
        
        if fallback_factory is not None:
            return fallback_factory(), False
        elif fallback_value is not None:
            return fallback_value, False
        else:
            raise


__all__ = [
    # Config
    'safe_get_method',
    'safe_get_attr',
    # Estimation
    'estimate_ar_coefficients_ols',
    'compute_innovation_covariance',
    'compute_sufficient_stats',
    'safe_mean_std',
    'standardize_data',
    # Validation
    'validate_params',
    'stabilize_cov',
    # Matrix
    'reg_inv',
    'update_loadings',
    'extract_3d_matrix_slice',
    'compute_obs_cov',
    'clean_variance_array',
    # Block
    'update_block_diag',
    'get_block_indices',
    'compute_block_slice_indices',
    'extract_block_matrix',
    'update_block_in_matrix',
    # Frequency
    'get_tent_weights',
    'infer_nQ',
    # Array
    'safe_time_index',
    'safe_array_operation',
    'get_matrix_shape',
    'has_valid_data',
    'ensure_minimum_size',
    # Utils
    'append_or_initialize',
    'create_empty_matrix',
    'reshape_to_column_vector',
    'reshape_to_row_vector',
    'pad_matrix_to_shape',
    'safe_numerical_operation',
    'resolve_param',
]
