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
        raise TypeError(
            f"Cannot convert {type(data).__name__} to Tensor. "
            f"Supported types: Tensor, np.ndarray, list, tuple, int, float."
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
    data: Union[np.ndarray, Tensor, List, float, int],
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Convert input to numpy array with optional dtype conversion.
    
    Parameters
    ----------
    data : array-like, Tensor, or scalar
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
    """
    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, Tensor):
        array = data.detach().cpu().numpy()
    elif isinstance(data, Sequence) and not isinstance(data, str):
        array = np.array(data)
    elif isinstance(data, (int, float)):
        array = np.array([data])
    else:
        raise TypeError(
            f"Cannot convert {type(data).__name__} to numpy array. "
            f"Supported types: np.ndarray, Tensor, list, tuple, int, float."
        )
    
    # Convert dtype if specified
    if dtype is not None:
        array = array.astype(dtype)
    
    return array




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
        raise TypeError(
            f"{name} must be numpy array or torch Tensor, got {type(matrix).__name__}"
        )
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(actual_shape)} dimensions, expected {len(expected_shape)}. "
            f"Shape: {actual_shape}, Expected: {expected_shape}"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} dimension {i} is {actual}, expected {expected}. "
                f"Shape: {actual_shape}, Expected: {expected_shape}"
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
