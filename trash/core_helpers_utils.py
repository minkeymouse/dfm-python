"""General utility helpers for array manipulation and numerical operations."""

from typing import Optional, Tuple, Any, Dict
import numpy as np
import logging

_logger = logging.getLogger(__name__)

# Common exception types for numerical operations
NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)


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


# ============================================================================
# Configuration access helpers
# ============================================================================

def safe_get_method(config: Optional['DFMConfig'], method_name: str, default: Any = None) -> Any:
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


def safe_get_attr(config: Optional['DFMConfig'], attr_name: str, default: Any = None) -> Any:
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
    from ...utils.aggregation import (
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
    from ...utils.aggregation import FREQUENCY_HIERARCHY
    
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

