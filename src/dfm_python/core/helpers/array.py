"""Array utility helpers for safe indexing, shape checking, and validation."""

from typing import Optional, Any
import numpy as np


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

