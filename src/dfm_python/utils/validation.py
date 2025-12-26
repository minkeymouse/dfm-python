"""Consolidated validation utilities for dfm-python.

This module provides common validation functions used across the package,
replacing duplicate implementations in multiple modules.
"""

import numpy as np
import torch
from typing import Optional, Union, Any
from ..logger import get_logger

_logger = get_logger(__name__)


def check_finite_array(
    arr: np.ndarray,
    name: str = "array",
    context: Optional[str] = None,
    fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    """Check if numpy array contains only finite values, with fallback.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str, default "array"
        Name for error messages
    context : str, optional
        Additional context for error messages
    fallback : np.ndarray, optional
        Fallback array to use if check fails
        
    Returns
    -------
    np.ndarray
        Original array if finite, fallback if provided and check fails
        
    Raises
    ------
    ValueError
        If array contains non-finite values and no fallback provided
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(~np.isfinite(arr))
        context_str = f" in {context}" if context else ""
        msg = f"{name}{context_str} contains {nan_count} non-finite values"
        
        if fallback is not None:
            _logger.warning(f"{msg}. Using fallback array")
            return fallback
        else:
            _logger.error(msg)
            raise ValueError(msg)
    
    return arr


def check_finite_tensor(
    tensor: torch.Tensor,
    name: str = "tensor"
) -> bool:
    """Check if torch tensor contains only finite values.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    name : str, default "tensor"
        Name for error messages
        
    Returns
    -------
    bool
        True if tensor is finite, False otherwise
    """
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    
    if has_nan or has_inf:
        nan_count = torch.sum(torch.isnan(tensor)).item()
        inf_count = torch.sum(torch.isinf(tensor)).item()
        msg = f"{name} contains "
        issues = []
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values")
        if inf_count > 0:
            issues.append(f"{inf_count} Inf values")
        msg += " and ".join(issues)
        _logger.warning(msg)
        return False
    return True


def ensure_real(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is real by extracting real part if complex.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to ensure is real
        
    Returns
    -------
    torch.Tensor
        Real tensor (original if already real, real part if complex)
    """
    if tensor.is_complex():
        _logger.warning("Tensor is complex, extracting real part")
        return tensor.real
    return tensor


def validate_shape(
    arr: Union[np.ndarray, torch.Tensor],
    expected_shape: tuple,
    name: str = "array"
) -> None:
    """Validate array/tensor shape.
    
    Parameters
    ----------
    arr : Union[np.ndarray, torch.Tensor]
        Array or tensor to validate
    expected_shape : tuple
        Expected shape (can include None for variable dimensions)
    name : str, default "array"
        Name for error messages
        
    Raises
    ------
    ValueError
        If shape doesn't match expected shape
    """
    actual_shape = arr.shape
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} has wrong number of dimensions: "
            f"expected {len(expected_shape)}, got {len(actual_shape)}"
        )
    
    mismatches = []
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            mismatches.append(f"dim {i}: expected {expected}, got {actual}")
    
    if mismatches:
        raise ValueError(
            f"{name} shape mismatch: {', '.join(mismatches)}. "
            f"Expected {expected_shape}, got {actual_shape}"
        )

