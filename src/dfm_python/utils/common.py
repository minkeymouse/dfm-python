"""Common utilities for dfm-python.

This module provides shared utility functions used across multiple modules
for better code organization and reusability.
"""

from typing import Optional, Tuple, Union, List, Any, Dict, Sequence
import numpy as np
import torch
from torch import Tensor

from ..config.types import Device, ArrayLike
from .errors import NumericalError, DataValidationError
from ..logger import get_logger
from ..config.constants import DEFAULT_ZERO_VALUE, MAX_EIGENVALUE

_logger = get_logger(__name__)


def ensure_tensor(
    data: Union[np.ndarray, Tensor, List, float, int],
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False
) -> Tensor:
    """Convert input to torch Tensor with optional device/dtype conversion.
    
    This utility function provides a consistent way to convert various input
    types to torch Tensors, ensuring proper device placement and dtype.
    
    Parameters
    ----------
    data : array-like, Tensor, or scalar
        Input data to convert to Tensor
    device : Device, optional
        Target device (e.g., 'cpu', 'cuda:0'). If None, uses data's device
        or defaults to CPU.
    dtype : torch.dtype, optional
        Target dtype (e.g., torch.float32). If None, infers from data.
    requires_grad : bool, default=False
        Whether the tensor requires gradients
        
    Returns
    -------
    Tensor
        Converted tensor on specified device with specified dtype
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> tensor = ensure_tensor(data, device='cuda:0', dtype=torch.float32)
    >>> assert isinstance(tensor, Tensor)
    >>> assert tensor.device.type == 'cuda'
    """
    if isinstance(data, Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        tensor = torch.tensor(data)
    elif isinstance(data, (int, float)):
        tensor = torch.tensor([data])
    else:
        raise DataValidationError(
            f"Cannot convert {type(data).__name__} to Tensor. "
            f"Supported types: Tensor, np.ndarray, list, tuple, int, float.",
            details=f"Input type: {type(data).__name__}, value: {data}"
        )
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    # Convert dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype)
    
    # Set requires_grad
    if requires_grad:
        tensor = tensor.requires_grad_(True)
    
    return tensor


def ensure_numpy(
    data: Union[np.ndarray, Tensor, List, float, int, Any],
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Convert input to numpy array with optional dtype conversion.
    
    Supports multiple input types:
    - numpy.ndarray
    - torch.Tensor
    - pandas.DataFrame
    - polars.DataFrame
    - list, tuple, Sequence
    - int, float (scalars)
    
    Parameters
    ----------
    data : array-like, Tensor, DataFrame, or scalar
        Input data to convert to numpy array
    dtype : np.dtype, optional
        Target dtype (e.g., np.float32). If None, infers from data.
        
    Returns
    -------
    np.ndarray
        Converted numpy array
        
    Examples
    --------
    >>> import torch
    >>> tensor = torch.tensor([1.0, 2.0, 3.0])
    >>> array = ensure_numpy(tensor, dtype=np.float32)
    >>> assert isinstance(array, np.ndarray)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    >>> array = ensure_numpy(df)
    >>> assert isinstance(array, np.ndarray)
    """
    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, Tensor):
        array = data.detach().cpu().numpy()
    elif isinstance(data, str):
        # Strings are sequences but should not be converted to numpy arrays
        raise DataValidationError(
            f"Cannot convert {type(data).__name__} to numpy array. "
            f"Supported types: np.ndarray, Tensor, pandas.DataFrame, polars.DataFrame, list, tuple, int, float.",
            details=f"Input type: {type(data).__name__}, value: {data}"
        )
    elif hasattr(data, '__class__'):
        # Try pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                array = data.to_numpy()
            elif data.__class__.__module__ == 'pandas.core.frame':
                # Fallback for pandas DataFrame detection
                array = data.to_numpy()
            else:
                # Try polars DataFrame
                try:
                    import polars as pl
                    if isinstance(data, pl.DataFrame):
                        array = data.to_numpy()
                    else:
                        raise DataValidationError(
                            f"Cannot convert {type(data).__name__} to numpy array. "
                            f"Supported types: np.ndarray, Tensor, pandas.DataFrame, polars.DataFrame, list, tuple, int, float.",
                            details=f"Input type: {type(data).__name__}, module: {data.__class__.__module__}"
                        )
                except (ImportError, AttributeError):
                    # Not polars, try generic conversion
                    try:
                        array = np.array(data)
                    except Exception as e:
                        raise DataValidationError(
                            f"Cannot convert {type(data).__name__} to numpy array: {e}",
                            details=f"Input type: {type(data).__name__}, module: {data.__class__.__module__}"
                        ) from e
        except ImportError:
            # pandas not available, try polars or generic conversion
            try:
                import polars as pl
                if isinstance(data, pl.DataFrame):
                    array = data.to_numpy()
                else:
                    raise DataValidationError(
                        f"Cannot convert {type(data).__name__} to numpy array. "
                        f"pandas not available and input is not polars.DataFrame.",
                        details=f"Input type: {type(data).__name__}"
                    )
            except (ImportError, AttributeError):
                # Try generic conversion
                try:
                    array = np.array(data)
                except Exception as e:
                    raise DataValidationError(
                        f"Cannot convert {type(data).__name__} to numpy array: {e}",
                        details=f"Input type: {type(data).__name__}"
                    ) from e
    elif isinstance(data, Sequence):
        array = np.array(data)
    elif isinstance(data, (int, float)):
        array = np.array([data])
    else:
        raise DataValidationError(
            f"Cannot convert {type(data).__name__} to numpy array. "
            f"Supported types: np.ndarray, Tensor, pandas.DataFrame, polars.DataFrame, list, tuple, int, float.",
            details=f"Input type: {type(data).__name__}, value: {data}"
        )
    
    # Convert dtype if specified
    if dtype is not None:
        array = array.astype(dtype)
    
    return array


def sanitize_array(
    arr: np.ndarray,
    nan_value: float = DEFAULT_ZERO_VALUE,
    inf_value: float = MAX_EIGENVALUE,
    neginf_value: Optional[float] = None
) -> np.ndarray:
    """Sanitize array by replacing NaN/Inf with specified values.
    
    This helper consolidates the common pattern of using np.nan_to_num
    with DEFAULT_ZERO_VALUE for NaN and MAX_EIGENVALUE for Inf values.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to sanitize
    nan_value : float, default DEFAULT_ZERO_VALUE
        Value to replace NaN with
    inf_value : float, default MAX_EIGENVALUE
        Value to replace positive infinity with
    neginf_value : float, optional
        Value to replace negative infinity with. If None, uses -inf_value.
        
    Returns
    -------
    np.ndarray
        Sanitized array with NaN/Inf replaced
        
    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    >>> sanitized = sanitize_array(arr)
    >>> assert np.all(np.isfinite(sanitized))
    """
    if neginf_value is None:
        neginf_value = -inf_value
    return np.nan_to_num(arr, nan=nan_value, posinf=inf_value, neginf=neginf_value)




def validate_matrix_shape(
    matrix: Union[np.ndarray, Tensor],
    expected_shape: Tuple[int, ...],
    name: str = "matrix"
) -> None:
    """Validate matrix shape matches expected shape.
    
    Parameters
    ----------
    matrix : np.ndarray or Tensor
        Matrix to validate
    expected_shape : tuple
        Expected shape (can use -1 for "any" dimension)
    name : str, default="matrix"
        Name of matrix for error messages
        
    Raises
    ------
    ValueError
        If shape doesn't match expected shape
    """
    if isinstance(matrix, Tensor):
        actual_shape = tuple(matrix.shape)
    elif isinstance(matrix, np.ndarray):
        actual_shape = matrix.shape
    else:
        raise DataValidationError(
            f"{name} must be numpy array or torch Tensor, got {type(matrix).__name__}",
            details=f"Input type: {type(matrix).__name__}, value shape: {getattr(matrix, 'shape', 'N/A')}"
        )
    
    if len(actual_shape) != len(expected_shape):
        raise DataValidationError(
            f"{name} has {len(actual_shape)} dimensions, expected {len(expected_shape)}. "
            f"Shape: {actual_shape}, Expected: {expected_shape}",
            details=f"Dimension mismatch: actual={len(actual_shape)}, expected={len(expected_shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise DataValidationError(
                f"{name} dimension {i} is {actual}, expected {expected}. "
                f"Shape: {actual_shape}, Expected: {expected_shape}",
                details=f"Dimension {i} mismatch: actual={actual}, expected={expected}"
            )


def log_tensor_stats(
    tensor: Tensor,
    name: str,
    logger: Optional[Any] = None
) -> None:
    """Log tensor statistics for debugging.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to log statistics for
    name : str
        Name of tensor for log message
    logger : logger, optional
        Logger instance. If None, uses module logger.
    """
    if logger is None:
        logger = _logger
    
    stats = {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item()
    }
    
    logger.debug(
        f"{name} stats: shape={stats['shape']}, dtype={stats['dtype']}, "
        f"device={stats['device']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
        f"has_nan={stats['has_nan']}, has_inf={stats['has_inf']}"
    )


def create_default_standardization_arrays(
    n_series: int,
    dtype: Optional[np.dtype] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create default standardization arrays.
    
    Creates arrays representing standardized data (mean=0, std=1).
    Used internally when data is already standardized and no scaler is provided.
    
    Parameters
    ----------
    n_series : int
        Number of series
    dtype : np.dtype, optional
        Data type for arrays. If None, uses DEFAULT_DTYPE.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (mean_array, std_array) both of shape (n_series,)
        mean_array: zeros (standardized data has zero mean)
        std_array: ones (standardized data has unit variance)
    """
    from ..config.constants import DEFAULT_DTYPE
    
    if dtype is None:
        dtype = DEFAULT_DTYPE
    
    mean_array = np.zeros(n_series, dtype=dtype)
    std_array = np.ones(n_series, dtype=dtype)
    
    return mean_array, std_array


def select_columns_by_prefix(
    df: Any,
    prefixes: List[str],
    count_per_prefix: int = 2
) -> List[str]:
    """Select columns from DataFrame by prefix pattern.
    
    Selects columns matching pattern `{prefix}{i}` for each prefix,
    where i ranges from 1 to count_per_prefix. Useful for selecting
    balanced subsets of series from different categories.
    
    Parameters
    ----------
    df : DataFrame or object with .columns attribute
        DataFrame to select columns from
    prefixes : List[str]
        List of prefixes to match (e.g., ["D", "E", "I", "M", "P", "S", "V"])
    count_per_prefix : int, default=2
        Number of columns to select per prefix (i ranges from 1 to count_per_prefix)
        
    Returns
    -------
    List[str]
        List of selected column names that exist in df.columns
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"D1": [1, 2], "D2": [3, 4], "E1": [5, 6]})
    >>> select_columns_by_prefix(df, ["D", "E"], count_per_prefix=2)
    ['D1', 'D2', 'E1']
    """
    selected_cols = []
    for prefix in prefixes:
        for i in range(1, count_per_prefix + 1):
            col = f"{prefix}{i}"
            if hasattr(df, 'columns') and col in df.columns:
                selected_cols.append(col)
    return selected_cols
