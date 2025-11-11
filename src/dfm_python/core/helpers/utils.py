"""General utility helpers for array manipulation and numerical operations."""

from typing import Optional, Tuple, Any
import numpy as np
import logging
from ._common import NUMERICAL_EXCEPTIONS

_logger = logging.getLogger(__name__)


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

