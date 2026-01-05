"""Data preprocessing utilities for factor models.

This module provides reusable preprocessing functions for handling missing data,
mask shape adjustment, and tensor cleaning that can be used across different
factor models (DFM, DDFM, KDFM).
"""

import numpy as np
import torch
from typing import Tuple, Optional, Any
from torch import Tensor

from ..config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
    DEFAULT_ZERO_VALUE,
)
from ..numeric.stability import rem_nans_spline
from .common import ensure_numpy, ensure_tensor
from .misc import get_config_attr
from ..logger import get_logger

_logger = get_logger(__name__)


def preprocess_training_data(
    X_torch: Tensor,
    config: Optional[Any] = None,
    nan_method: Optional[int] = None,
    nan_k: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    replace_inf: bool = True
) -> Tuple[Tensor, np.ndarray]:
    """Preprocess training data: handle missing values and ensure finite values.
    
    This utility function handles missing data preprocessing for factor models.
    It can be parameterized for different models (DFM, DDFM, KDFM) by specifying
    config, nan_method, and nan_k.
    
    Parameters
    ----------
    X_torch : Tensor
        Raw training data (T x N) or any shape
    config : Any, optional
        Configuration object with nan_method and nan_k attributes.
        If provided, overrides nan_method and nan_k parameters.
    nan_method : int, optional
        Missing data handling method (see rem_nans_spline).
        If None, uses config.nan_method or DEFAULT_NAN_METHOD.
    nan_k : int, optional
        Spline interpolation order (see rem_nans_spline).
        If None, uses config.nan_k or DEFAULT_NAN_K.
    dtype : torch.dtype, optional
        Target dtype for output tensor. If None, uses DEFAULT_TORCH_DTYPE.
    replace_inf : bool, default True
        If True, replaces NaN/Inf with DEFAULT_ZERO_VALUE after imputation.
        
    Returns
    -------
    x_clean_torch : Tensor
        Cleaned data with missing values imputed
    missing_mask : np.ndarray
        Boolean mask indicating missing values (True = missing, False = observed)
        
    Examples
    --------
    >>> import torch
    >>> from dfm_python.utils.preprocessing import preprocess_training_data
    >>> 
    >>> # Example with missing data
    >>> X = torch.tensor([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]])
    >>> x_clean, mask = preprocess_training_data(X)
    >>> assert torch.all(torch.isfinite(x_clean))
    >>> assert mask.dtype == bool
    """
    if dtype is None:
        dtype = DEFAULT_TORCH_DTYPE
    
    # Resolve nan_method and nan_k from config or parameters
    if config is not None:
        nan_method = get_config_attr(config, 'nan_method', nan_method or DEFAULT_NAN_METHOD)
        nan_k = get_config_attr(config, 'nan_k', nan_k or DEFAULT_NAN_K)
    else:
        nan_method = nan_method if nan_method is not None else DEFAULT_NAN_METHOD
        nan_k = nan_k if nan_k is not None else DEFAULT_NAN_K
    
    X_np = ensure_numpy(X_torch)
    has_nan = np.any(np.isnan(X_np))
    
    if has_nan:
        x_clean_np, missing_mask = rem_nans_spline(X_np, method=nan_method, k=nan_k)
        x_clean_torch = ensure_tensor(x_clean_np, dtype=dtype, device=X_torch.device)
    else:
        device = X_torch.device if isinstance(X_torch, Tensor) else None
        x_clean_torch = ensure_tensor(X_torch, dtype=dtype, device=device)
        missing_mask = np.zeros(X_np.shape, dtype=bool)
    
    # Replace any remaining NaN/Inf with zeros if requested
    if replace_inf:
        device = x_clean_torch.device
        dtype_actual = x_clean_torch.dtype
        x_clean_torch = torch.where(
            torch.isfinite(x_clean_torch),
            x_clean_torch,
            torch.tensor(DEFAULT_ZERO_VALUE, device=device, dtype=dtype_actual)
        )
    
    return x_clean_torch, missing_mask


def create_mask_from_nan(tensor: Tensor) -> Tensor:
    """Create boolean mask from NaN values in tensor.
    
    Consolidates pattern: torch.where(torch.isnan(tensor), torch.zeros_like(tensor), torch.ones_like(tensor)).bool()
    Used when converting NaN values to a boolean mask for loss computation or data handling.
    
    Parameters
    ----------
    tensor : Tensor
        Input tensor that may contain NaN values
        
    Returns
    -------
    mask : Tensor
        Boolean mask where True indicates non-NaN values, False indicates NaN values.
        Same shape as input tensor.
        
    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from dfm_python.utils.preprocessing import create_mask_from_nan
    >>> 
    >>> x = torch.tensor([[1.0, 2.0], [np.nan, 4.0]])
    >>> mask = create_mask_from_nan(x)
    >>> assert mask.shape == x.shape
    >>> assert mask.dtype == torch.bool
    >>> assert mask[0, 0] == True  # Non-NaN
    >>> assert mask[1, 0] == False  # NaN
    """
    return torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor),
        torch.ones_like(tensor)
    ).bool()


def adjust_mask_shape(
    mask: np.ndarray,
    target_shape: Tuple[int, ...],
    time_dim: int = 0,
    variable_dim: int = 1,
    pad_value: bool = False,
    warn: bool = True
) -> np.ndarray:
    """Adjust missing mask shape to match target shape.
    
    This utility function handles mask shape adjustment for factor models.
    It can be parameterized for different models by specifying dimension indices
    and padding behavior.
    
    Parameters
    ----------
    mask : np.ndarray
        Missing mask (may have different shape)
    target_shape : Tuple[int, ...]
        Target shape (e.g., (T, N) for 2D, or (T, N, ...) for higher dimensions)
    time_dim : int, default 0
        Index of time dimension in mask
    variable_dim : int, default 1
        Index of variable dimension in mask
    pad_value : bool, default False
        Value to use for padding (False = missing, True = observed)
    warn : bool, default True
        If True, log a warning when shape adjustment is needed
        
    Returns
    -------
    mask : np.ndarray
        Adjusted mask with shape matching target_shape
        
    Examples
    --------
    >>> import numpy as np
    >>> from dfm_python.utils.preprocessing import adjust_mask_shape
    >>> 
    >>> # Example: adjust mask from (100, 10) to (50, 10)
    >>> mask = np.zeros((100, 10), dtype=bool)
    >>> adjusted = adjust_mask_shape(mask, (50, 10))
    >>> assert adjusted.shape == (50, 10)
    """
    if mask.shape == target_shape:
        return mask
    
    if warn:
        _logger.warning(
            f"Mask shape {mask.shape} != target shape {target_shape}, adjusting. "
            f"time_dim={time_dim}, variable_dim={variable_dim}"
        )
    
    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    # Handle 2D case (most common for factor models)
    if len(target_shape) == 2:
        T_target, N_target = target_shape
        
        # Adjust time dimension
        if mask.shape[time_dim] > T_target:
            mask = mask[:T_target, :] if time_dim == 0 else mask[:, :T_target]
        elif mask.shape[time_dim] < T_target:
            pad_size = T_target - mask.shape[time_dim]
            if time_dim == 0:
                pad = np.zeros((pad_size, mask.shape[variable_dim]), dtype=bool) if not pad_value else np.ones((pad_size, mask.shape[variable_dim]), dtype=bool)
                mask = np.vstack([mask, pad])
            else:
                pad = np.zeros((mask.shape[time_dim], pad_size), dtype=bool) if not pad_value else np.ones((mask.shape[time_dim], pad_size), dtype=bool)
                mask = np.hstack([mask, pad])
        
        # Adjust variable dimension
        if mask.shape[variable_dim] > N_target:
            mask = mask[:, :N_target] if variable_dim == 1 else mask[:N_target, :]
        elif mask.shape[variable_dim] < N_target:
            pad_size = N_target - mask.shape[variable_dim]
            if variable_dim == 1:
                pad = np.zeros((mask.shape[time_dim], pad_size), dtype=bool) if not pad_value else np.ones((mask.shape[time_dim], pad_size), dtype=bool)
                mask = np.hstack([mask, pad])
            else:
                pad = np.zeros((pad_size, mask.shape[time_dim]), dtype=bool) if not pad_value else np.ones((pad_size, mask.shape[time_dim]), dtype=bool)
                mask = np.vstack([mask, pad])
    else:
        # Generic case: adjust each dimension
        adjusted_mask = mask.copy()
        for dim_idx, target_size in enumerate(target_shape):
            if dim_idx >= adjusted_mask.ndim:
                # Need to add new dimension
                pad_shape = list(adjusted_mask.shape)
                pad_shape.insert(dim_idx, target_size)
                pad = np.zeros(pad_shape, dtype=bool) if not pad_value else np.ones(pad_shape, dtype=bool)
                adjusted_mask = np.concatenate([adjusted_mask, pad], axis=dim_idx)
            elif adjusted_mask.shape[dim_idx] > target_size:
                # Truncate
                slices = [slice(None)] * adjusted_mask.ndim
                slices[dim_idx] = slice(0, target_size)
                adjusted_mask = adjusted_mask[tuple(slices)]
            elif adjusted_mask.shape[dim_idx] < target_size:
                # Pad
                pad_size = target_size - adjusted_mask.shape[dim_idx]
                pad_shape = list(adjusted_mask.shape)
                pad_shape[dim_idx] = pad_size
                pad = np.zeros(pad_shape, dtype=bool) if not pad_value else np.ones(pad_shape, dtype=bool)
                adjusted_mask = np.concatenate([adjusted_mask, pad], axis=dim_idx)
        mask = adjusted_mask
    
    return mask


