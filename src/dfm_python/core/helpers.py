"""Helper functions for DFM estimation."""
from typing import Optional, Tuple, Any, Dict
import numpy as np
import logging
from scipy.linalg import inv, pinv, block_diag

_logger = logging.getLogger(__name__)

# Common exception types for numerical operations
_NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)

def safe_get_method(config, method_name, default=None):
    if config is None:
        return default
    method = getattr(config, method_name, None)
    if method is not None and callable(method):
        return method()
    return default

def safe_get_attr(config, attr_name, default=None):
    if config is None:
        return default
    return getattr(config, attr_name, default)

def resolve_param(override, default):
    return override if override is not None else default

def safe_mean_std(X, clip_data_values=False, data_clip_threshold=10.0):
    """Compute mean and std with optional clipping."""
    if clip_data_values:
        X = np.clip(X, -data_clip_threshold, data_clip_threshold)
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0, ddof=0)
    Wx = np.where(Wx < 1e-8, 1.0, Wx)
    return Mx, Wx

def standardize_data(X, clip_data_values=False, data_clip_threshold=10.0):
    """Standardize data: x = (X - mean) / std."""
    Mx, Wx = safe_mean_std(X, clip_data_values, data_clip_threshold)
    x = (X - Mx) / Wx
    return x, Mx, Wx


# ============================================================================
# Block structure helpers
# ============================================================================

def get_block_indices(blocks: np.ndarray, block_idx: int) -> np.ndarray:
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


# ============================================================================
# Array utility helpers
# ============================================================================

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


def has_valid_data(array: Optional[np.ndarray], min_size: int = 1) -> bool:
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


def get_matrix_shape(matrix: Optional[np.ndarray], dim: Optional[int] = None) -> Any:
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


# ============================================================================
# Frequency-related helpers
# ============================================================================

def infer_nQ(frequencies: Optional[np.ndarray], clock: str) -> int:
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
    from ..utils import FREQUENCY_HIERARCHY
    
    if frequencies is None:
        return 0
    
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    return sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_h)


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
    from ..utils import FREQUENCY_HIERARCHY, get_tent_weights_for_pair, generate_tent_weights
    
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
    except _NUMERICAL_EXCEPTIONS:
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
        except _NUMERICAL_EXCEPTIONS:
            return np.eye(residuals.shape[1]) * default_variance


# ============================================================================
# Variance cleaning helpers
# ============================================================================

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
