"""Tensor conversion and manipulation utilities.

This module provides utilities for tensor operations that are not covered
by the common utilities in utils.common.

Note: Basic tensor conversion (tensor_to_numpy, numpy_to_tensor, ensure_tensor_on_device)
has been removed. Use ensure_numpy() and ensure_tensor() from utils.common instead.

Remaining utilities:
- extract_tensor_value: Extract scalar or array values
- normalize_tensor_shape: Normalize tensor dimensions
- validate_tensor_device: Validate device placement
- batch_tensor_operation: Apply operations to batch dimensions
"""

from typing import Union, Optional, Tuple, Any
import numpy as np
import torch
from torch import Tensor

from ..logger import get_logger

_logger = get_logger(__name__)


# Removed deprecated functions (use utils.common instead):
# - tensor_to_numpy() -> use ensure_numpy()
# - numpy_to_tensor() -> use ensure_tensor()
# - ensure_tensor_on_device() -> use ensure_tensor(..., device=...)


def extract_tensor_value(tensor: Union[Tensor, np.ndarray, float, int]) -> Union[float, np.ndarray]:
    """Extract scalar or array value from tensor.
    
    For scalar tensors, returns Python float/int.
    For array tensors, returns NumPy array.
    
    Parameters
    ----------
    tensor : Tensor, np.ndarray, float, or int
        Input tensor, array, or scalar
        
    Returns
    -------
    float, int, or np.ndarray
        Extracted value
        
    Examples
    --------
    >>> t = torch.tensor(3.14)
    >>> val = extract_tensor_value(t)
    >>> assert isinstance(val, float)
    >>> assert val == 3.14
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
        from .common import ensure_numpy
        return ensure_numpy(tensor)
    else:
        raise TypeError(
            f"Expected Tensor, np.ndarray, float, or int, got {type(tensor).__name__}"
        )


def normalize_tensor_shape(
    tensor: Union[Tensor, np.ndarray],
    expected_ndim: int,
    name: str = "tensor"
) -> Union[Tensor, np.ndarray]:
    """Normalize tensor shape by adding/removing dimensions.
    
    Parameters
    ----------
    tensor : Tensor or np.ndarray
        Input tensor
    expected_ndim : int
        Expected number of dimensions
    name : str, default="tensor"
        Name for error messages
        
    Returns
    -------
    Tensor or np.ndarray
        Tensor with normalized shape
        
    Raises
    ------
    ValueError
        If tensor has more dimensions than expected
        
    Examples
    --------
    >>> t = torch.randn(3, 4)  # 2D
    >>> t_norm = normalize_tensor_shape(t, expected_ndim=3)  # Add batch dimension
    >>> assert t_norm.shape == (1, 3, 4)
    """
    current_ndim = tensor.ndim
    
    if current_ndim == expected_ndim:
        return tensor
    elif current_ndim < expected_ndim:
        # Add dimensions at the beginning
        n_add = expected_ndim - current_ndim
        if isinstance(tensor, Tensor):
            for _ in range(n_add):
                tensor = tensor.unsqueeze(0)
        else:
            for _ in range(n_add):
                tensor = np.expand_dims(tensor, axis=0)
        return tensor
    else:
        raise ValueError(
            f"{name} has {current_ndim} dimensions, but expected at most {expected_ndim}. "
            f"Shape: {tensor.shape}"
        )


def validate_tensor_device(
    tensor: Tensor,
    expected_device: Optional[torch.device] = None,
    name: str = "tensor"
) -> None:
    """Validate that tensor is on expected device.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to validate
    expected_device : torch.device, optional
        Expected device. If None, only checks that device is valid.
    name : str, default="tensor"
        Name for error messages
        
    Raises
    ------
    ValueError
        If tensor is not on expected device
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a Tensor, got {type(tensor).__name__}")
    
    if expected_device is not None:
        if tensor.device != expected_device:
            raise ValueError(
                f"{name} is on device {tensor.device}, but expected {expected_device}"
            )


def batch_tensor_operation(
    tensor: Tensor,
    operation: callable,
    batch_dim: int = 0
) -> Tensor:
    """Apply operation to each item in batch dimension.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor with batch dimension
    operation : callable
        Operation to apply to each batch item
    batch_dim : int, default=0
        Batch dimension index
        
    Returns
    -------
    Tensor
        Result tensor with same batch structure
    """
    batch_size = tensor.shape[batch_dim]
    results = []
    
    for i in range(batch_size):
        if batch_dim == 0:
            batch_item = tensor[i]
        else:
            # More complex indexing for other batch dimensions
            indices = [slice(None)] * tensor.ndim
            indices[batch_dim] = i
            batch_item = tensor[tuple(indices)]
        
        result = operation(batch_item)
        results.append(result)
    
    return torch.stack(results, dim=batch_dim)
