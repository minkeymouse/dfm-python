"""Comprehensive type definitions for dfm-python.

This module provides unified type aliases and definitions used throughout
the codebase for better type hinting, code clarity, and consistency.

These types are designed to be compatible with both dfm-python and experiment code.
"""

from typing import Union, Tuple, Optional, List, Dict, Any, Sequence, Literal
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

# ============================================================================
# Array Types
# ============================================================================

ArrayLike = Union[np.ndarray, Tensor]
"""Union type for numpy arrays and torch tensors."""

FloatArray = Union[np.ndarray, Tensor]
"""Array containing floating-point values."""

IntArray = Union[np.ndarray, Tensor]
"""Array containing integer values."""

BoolArray = Union[np.ndarray, Tensor]
"""Array containing boolean values."""

OptionalArray = Optional[ArrayLike]
"""Optional array (numpy or torch)."""

OptionalTensor = Optional[Tensor]
"""Optional torch tensor."""

OptionalArrayLike = Optional[ArrayLike]
"""Optional array-like (numpy or torch)."""

# ============================================================================
# Model Types
# ============================================================================

FactorState = np.ndarray
"""Factor state array (m,) or (T, m)."""

ObservationState = np.ndarray
"""Observation array (N,) or (T, N)."""

ForecastResult = Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray]
]
"""Forecast result: either series only or (series, factors)."""

CoefficientMatrix = np.ndarray
"""Coefficient matrix (e.g., VAR coefficients). Shape: (K, K) or (p, K, K)."""

CovarianceMatrix = np.ndarray
"""Covariance matrix (e.g., Q, R). Shape: (K, K) or (m, m)."""

# IRFArray and CompanionMatrix removed - these were KDFM-specific types

# ============================================================================
# Configuration Types
# ============================================================================


SeriesID = str
"""Series identifier string."""

Frequency = Literal['d', 'w', 'm', 'q', 'sa', 'a']
"""Time series frequency code. Matches VALID_FREQUENCIES in constants.py."""

DatasetName = str
"""Dataset name identifier."""

ModelName = Literal['var', 'vecm', 'dfm', 'ddfm']
"""Model name identifier."""

# ============================================================================
# Training Types
# ============================================================================

Batch = Union[Tensor, Tuple[Tensor, Tensor]]
"""Training batch: data tensor or (data, target) tuple."""

Loss = Tensor
"""Training loss tensor."""

Optimizer = torch.optim.Optimizer
"""PyTorch optimizer type."""

# ============================================================================
# Result Types
# ============================================================================

ResultDict = Dict[str, Any]
"""Generic result dictionary."""

ForecastDict = Dict[str, Union[float, np.ndarray, Dict[str, Any]]]
"""Forecast result dictionary."""

MetricsDict = Dict[str, float]
"""Metrics dictionary (RMSE, MAE, R², etc.)."""

# IRFResultDict removed - this was KDFM-specific type

CheckpointDict = Dict[str, Any]
"""Model checkpoint dictionary."""

ConfigDict = Dict[str, Any]
"""Configuration dictionary."""

# ============================================================================
# Device Types
# ============================================================================

Device = Union[torch.device, str]
"""PyTorch device specification."""

# ============================================================================
# Shape Types
# ============================================================================

Shape = Tuple[int, ...]
"""Array shape tuple."""

Shape2D = Tuple[int, int]
"""2D shape tuple: (T, N) - time steps, variables."""

Shape3D = Tuple[int, int, int]
"""3D shape tuple: (T, N, N) or (H, N, N) - time/factor, variables, variables."""

Shape4D = Tuple[int, int, int, int]
"""4D shape tuple: (B, T, N, N) - batch, time, variables, variables."""

# ============================================================================
# Forecast and Analysis Types
# ============================================================================

ForecastHorizon = int
"""Forecast horizon. Typically 1, 4, or 8."""

IRFHorizon = int
"""IRF horizon. Typically 20."""

LagOrder = int
"""Lag order (p for AR, q for MA)."""

NumFactors = int
"""Number of factors (r or m)."""

NumVars = int
"""Number of variables (K or N)."""

# ============================================================================
# Path Types
# ============================================================================

PathLike = Union[str, Path]
"""Path-like object (string or Path)."""

# ============================================================================
# Validation Types
# ============================================================================

ValidationIssue = Dict[str, Any]
"""Validation issue dictionary with type, severity, description, etc."""

ValidationResult = Dict[str, Any]
"""Validation result dictionary with issues, summary, etc."""


# ============================================================================
# Type Guards and Utilities
# ============================================================================

def is_numpy_array(obj: Any) -> bool:
    """Check if object is a numpy array."""
    return isinstance(obj, np.ndarray)


def is_torch_tensor(obj: Any) -> bool:
    """Check if object is a torch tensor."""
    return isinstance(obj, Tensor)


def is_array_like(obj: Any) -> bool:
    """Check if object is array-like (numpy or torch)."""
    return is_numpy_array(obj) or is_torch_tensor(obj)


def get_array_shape(arr: ArrayLike) -> Shape:
    """Get shape of array (works for both numpy and torch)."""
    if is_numpy_array(arr):
        return arr.shape
    elif is_torch_tensor(arr):
        return tuple(arr.shape)
    else:
        raise DataValidationError(
            f"Expected numpy array or torch tensor, got {type(arr)}",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


def to_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert array-like to numpy array.
    
    Automatically detaches tensors that require grad to avoid RuntimeError.
    
    Parameters
    ----------
    arr : ArrayLike
        Array-like object to convert (numpy array or torch tensor)
    dtype : np.dtype, optional
        Target dtype for output array. If None, preserves original dtype.
        
    Returns
    -------
    np.ndarray
        Numpy array with specified dtype (or original dtype if dtype=None)
    """
    if is_numpy_array(arr):
        result = arr
    elif is_torch_tensor(arr):
        import torch
        if torch.is_tensor(arr):
            # Detach if tensor requires grad to avoid RuntimeError
            if arr.requires_grad:
                arr = arr.detach()
            result = arr.cpu().numpy()
        else:
            result = np.asarray(arr)
    else:
        result = np.asarray(arr)
    
    # Apply dtype conversion if specified
    if dtype is not None:
        result = result.astype(dtype)
    
    return result


def to_tensor(arr: ArrayLike, device: Optional[Device] = None, dtype: Optional['torch.dtype'] = None) -> Tensor:
    """Convert array-like to torch tensor.
    
    Parameters
    ----------
    arr : ArrayLike
        Array-like object to convert
    device : Optional[Device]
        Target device for tensor (default: None)
    dtype : Optional[torch.dtype]
        Target dtype for tensor (default: None)
        
    Returns
    -------
    Tensor
        PyTorch tensor with specified device and dtype
    """
    if is_torch_tensor(arr):
        result = arr
        if device is not None:
            result = result.to(device)
        if dtype is not None:
            result = result.to(dtype)
        return result
    elif is_numpy_array(arr):
        tensor = torch.from_numpy(arr)
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    else:
        # Use local import to avoid circular dependency
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Cannot convert {type(arr)} to torch tensor",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


__all__ = [
    # Array Types
    'ArrayLike',
    'FloatArray',
    'IntArray',
    'BoolArray',
    'OptionalArray',
    'OptionalTensor',
    'OptionalArrayLike',
    # Model Types
    'FactorState',
    'ObservationState',
    'ForecastResult',
    'CoefficientMatrix',
    'CovarianceMatrix',
    # Configuration Types
    'SeriesID',
    'Frequency',
    'DatasetName',
    'ModelName',
    # Training Types
    'Batch',
    'Loss',
    'Optimizer',
    # Result Types
    'ResultDict',
    'ForecastDict',
    'MetricsDict',
    'CheckpointDict',
    'ConfigDict',
    # Device Types
    'Device',
    # Shape Types
    'Shape',
    'Shape2D',
    'Shape3D',
    'Shape4D',
    # Forecast and Analysis Types
    'ForecastHorizon',
    'IRFHorizon',
    'LagOrder',
    'NumFactors',
    'NumVars',
    # Path Types
    'PathLike',
    # Validation Types
    'ValidationIssue',
    'ValidationResult',
    # Type Guards and Utilities
    'is_numpy_array',
    'is_torch_tensor',
    'is_array_like',
    'get_array_shape',
    'to_numpy',
    'to_tensor',
    # Re-export Tensor for convenience
    'Tensor',
]

