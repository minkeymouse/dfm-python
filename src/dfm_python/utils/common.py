"""Common utilities for dfm-python.

This module provides shared utility functions used across multiple modules
for better code organization and reusability.
"""

from typing import Optional, Tuple, Union, List, Any, Dict, Sequence
import numpy as np
import torch
from torch import Tensor

from ..config.types import Device, ArrayLike
from .errors import DataValidationError
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
    neginf_value: Optional[float] = None,
    validate: bool = False,
    name: str = "array",
    context: Optional[str] = None
) -> np.ndarray:
    """Sanitize array by replacing NaN/Inf with specified values.
    
    This helper consolidates the common pattern of using np.nan_to_num
    with DEFAULT_ZERO_VALUE for NaN and MAX_EIGENVALUE for Inf values.
    
    Can optionally validate that the array contains finite values before
    sanitization, providing better error messages.
    
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
    validate : bool, default False
        If True, log a warning if non-finite values are found before sanitization
    name : str, default "array"
        Name for validation messages (only used if validate=True)
    context : str, optional
        Additional context for validation messages (only used if validate=True)
        
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
    
    >>> # With validation warning
    >>> sanitized = sanitize_array(arr, validate=True, name="factors", context="DDFM")
    """
    if validate and not np.all(np.isfinite(arr)):
        nan_count = np.sum(~np.isfinite(arr))
        context_str = f" in {context}" if context else ""
        _logger.warning(
            f"{name}{context_str} contains {nan_count} non-finite values. "
            f"Sanitizing with nan_value={nan_value}, inf_value={inf_value}"
        )
    
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


def compute_scale_stats(array: Union[np.ndarray, Tensor]) -> Tuple[float, float]:
    """Compute mean and standard deviation of an array.
    
    Parameters
    ----------
    array : np.ndarray or Tensor
        Array to compute statistics for
        
    Returns
    -------
    Tuple[float, float]
        (mean, std) tuple
    """
    if isinstance(array, Tensor):
        mean_val = array.mean().item()
        std_val = array.std().item()
    else:
        mean_val = float(np.mean(array))
        std_val = float(np.std(array))
    return (mean_val, std_val)


def standardize_with_ddof(
    data: Union[np.ndarray, Tensor],
    ddof: int = 1
) -> Union[np.ndarray, Tensor]:
    """Standardize data using sample standard deviation (ddof=1) to match pandas std() default.
    
    This function provides standardization with configurable ddof parameter, allowing
    exact match with original TensorFlow DDFM which uses pandas std() with ddof=1 (sample std).
    StandardScaler uses ddof=0 (population std), which creates a sqrt(n/(n-1)) scaling difference.
    
    Parameters
    ----------
    data : Union[np.ndarray, Tensor]
        Input data to standardize
    ddof : int, default 1
        Degrees of freedom for standard deviation calculation (1 = sample std, 0 = population std)
        
    Returns
    -------
    Union[np.ndarray, Tensor]
        Standardized data with mean≈0, std≈1 (using ddof for std calculation)
    """
    if isinstance(data, Tensor):
        data_np = ensure_numpy(data)
        mean = np.mean(data_np, axis=0, keepdims=True)
        std = np.std(data_np, axis=0, ddof=ddof, keepdims=True)
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)
        data_standardized_np = (data_np - mean) / std
        return ensure_tensor(data_standardized_np, dtype=data.dtype, device=data.device)
    else:
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, ddof=ddof, keepdims=True)
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)
        return (data - mean) / std


def check_and_standardize_data(
    data: Union[np.ndarray, Tensor],
    mean_threshold: float = 0.1,
    std_min: float = 0.1,
    std_max: float = 10.0,
    apply_standardization: bool = True,
    use_ddof_1: bool = False
) -> Tuple[Union[np.ndarray, Tensor], bool]:
    """Check if data is standardized and optionally apply standardization.
    
    Parameters
    ----------
    data : np.ndarray or Tensor
        Data to check/standardize
    mean_threshold : float, default 0.1
        Maximum acceptable absolute mean for standardized data
    std_min : float, default 0.1
        Minimum acceptable std for standardized data
    std_max : float, default 10.0
        Maximum acceptable std for standardized data
    apply_standardization : bool, default True
        If True and data is not standardized, apply standardization
    use_ddof_1 : bool, default False
        If True, use ddof=1 (sample std) for standardization to match pandas std() default.
        If False, use StandardScaler (ddof=0, population std). Set to True for DDFM to match original TensorFlow.
        
    Returns
    -------
    Tuple[Union[np.ndarray, Tensor], bool]
        (standardized_data, was_standardized) tuple
        was_standardized is True if standardization was applied, False if data was already standardized
    """
    data_mean, data_std = compute_scale_stats(data)
    needs_standardization = (
        abs(data_mean) > mean_threshold or
        data_std < std_min or
        data_std > std_max
    )
    
    if needs_standardization and apply_standardization:
        if use_ddof_1:
            # Use ddof=1 (sample std) to match original TensorFlow DDFM pandas std() default
            data_standardized = standardize_with_ddof(data, ddof=1)
        else:
            # Use StandardScaler (ddof=0, population std) for default behavior
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            if isinstance(data, Tensor):
                data_np = ensure_numpy(data)
                data_standardized_np = scaler.fit_transform(data_np)
                data_standardized = ensure_tensor(data_standardized_np, dtype=data.dtype, device=data.device)
            else:
                data_standardized = scaler.fit_transform(data)
        return (data_standardized, True)
    else:
        return (data, False)


def normalize_to_match_scale(
    prediction: Union[np.ndarray, Tensor],
    target: Union[np.ndarray, Tensor],
    min_std: float = 1e-10,
    raise_on_zero_std: bool = False
) -> Tuple[Union[np.ndarray, Tensor], float, bool]:
    """Normalize prediction to match target scale.
    
    This helper consolidates the common pattern of normalizing prediction arrays
    to match target data scale, used in DDFM scale alignment checks.
    
    Parameters
    ----------
    prediction : np.ndarray or Tensor
        Prediction array to normalize
    target : np.ndarray or Tensor
        Target array to match scale of
    min_std : float, default 1e-10
        Minimum standard deviation threshold for scale ratio computation
    raise_on_zero_std : bool, default False
        If True, raise DataError when prediction has zero std. If False, return prediction unchanged.
        
    Returns
    -------
    Tuple[Union[np.ndarray, Tensor], float, bool]
        (normalized_prediction, scale_ratio, was_normalized)
        - normalized_prediction: Prediction normalized to match target scale (or original if no normalization needed)
        - scale_ratio: Ratio of prediction std to target std
        - was_normalized: Whether normalization was applied
        
    Raises
    ------
    DataError
        If raise_on_zero_std=True and prediction has zero std
    """
    from ..config.constants import MIN_STD_FOR_SCALE_CHECK, DEFAULT_SCALE_RATIO_MAX, DEFAULT_SCALE_RATIO_MIN
    from .errors import DataError
    
    # Compute scale statistics
    target_mean, target_std = compute_scale_stats(target)
    pred_mean, pred_std = compute_scale_stats(prediction)
    
    # Compute scale ratio
    scale_ratio = pred_std / target_std if target_std > min_std else float('inf')
    
    # Check if normalization is needed
    if scale_ratio > DEFAULT_SCALE_RATIO_MAX or scale_ratio < DEFAULT_SCALE_RATIO_MIN:
        # Normalization needed
        if pred_std > min_std:
            # Normalize: (pred - pred_mean) / pred_std * target_std + target_mean
            if isinstance(prediction, Tensor):
                # Keep as Tensor
                normalized = (prediction - pred_mean) / pred_std * target_std + target_mean
            else:
                # Keep as numpy array
                normalized = (prediction - pred_mean) / pred_std * target_std + target_mean
            return (normalized, scale_ratio, True)
        else:
            # Prediction has zero std - cannot normalize
            if raise_on_zero_std:
                raise DataError(
                    f"Cannot normalize prediction - prediction has zero std (pred_std={pred_std:.6f}). "
                    f"This indicates prediction is constant or invalid.",
                    details=f"target: mean={target_mean:.6f}, std={target_std:.6f}, prediction: mean={pred_mean:.6f}, std={pred_std:.6f}"
                )
            # Return original prediction unchanged
            return (prediction, scale_ratio, False)
    else:
        # No normalization needed - scales already match
        return (prediction, scale_ratio, False)


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


def extract_tensor_value(tensor: Union[Tensor, np.ndarray, float, int]) -> Union[float, np.ndarray]:
    """Extract scalar or array value from tensor.
    
    For scalar tensors, returns Python float/int.
    For array tensors, returns NumPy array.
    
    This consolidates tensor value extraction logic, complementing
    ensure_numpy() which always returns arrays.
    
    Parameters
    ----------
    tensor : Tensor, np.ndarray, float, or int
        Input tensor, array, or scalar
        
    Returns
    -------
    float, int, or np.ndarray
        Extracted value (scalar for single-element tensors, array otherwise)
        
    Examples
    --------
    >>> t = torch.tensor(3.14)
    >>> val = extract_tensor_value(t)
    >>> assert isinstance(val, float)
    >>> assert val == 3.14
    
    >>> t = torch.tensor([1.0, 2.0, 3.0])
    >>> arr = extract_tensor_value(t)
    >>> assert isinstance(arr, np.ndarray)
    """
    if isinstance(tensor, (float, int)):
        return tensor
    elif isinstance(tensor, np.ndarray):
        if tensor.size == 1:
            return float(tensor.item()) if tensor.ndim == 0 else float(tensor.flat[0])
        return tensor
    elif isinstance(tensor, Tensor):
        if tensor.numel() == 1:
            return float(tensor.item())
        return ensure_numpy(tensor)
    else:
        raise DataValidationError(
            f"Expected Tensor, np.ndarray, float, or int, got {type(tensor).__name__}",
            details=f"Input type: {type(tensor).__name__}, value: {tensor}"
        )
