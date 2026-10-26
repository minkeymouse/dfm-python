"""Type aliases and the numpy/torch conversion helpers used across the package.

torch is optional (the ``deep`` extra), so the aliases name its types as
forward references and importing this module does not import torch. The
aliases of torch classes (``Tensor``, ``Loss``, ``Optimizer``) are resolved
on first access, which imports torch.
"""

import sys
from typing import TYPE_CHECKING, Union, Tuple, Optional, Dict, Any, Literal
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    import torch

# Array types
ArrayLike = Union[np.ndarray, "torch.Tensor"]
FloatArray = Union[np.ndarray, "torch.Tensor"]
IntArray = Union[np.ndarray, "torch.Tensor"]
BoolArray = Union[np.ndarray, "torch.Tensor"]
OptionalArray = Optional[ArrayLike]
OptionalTensor = Optional["torch.Tensor"]
OptionalArrayLike = Optional[ArrayLike]

# Model types
FactorState = np.ndarray
"""Factor state array (m,) or (T, m)."""

ObservationState = np.ndarray
"""Observation array (N,) or (T, N)."""

ForecastResult = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
"""Series only, or (series, factors)."""

CoefficientMatrix = np.ndarray
"""Coefficient matrix (K, K) or (p, K, K)."""

CovarianceMatrix = np.ndarray
"""Covariance matrix (K, K) or (m, m)."""

# Configuration types
SeriesID = str
Frequency = Literal['d', 'w', 'm', 'q', 'sa', 'a']
"""Frequency code; matches ``constants.VALID_FREQUENCIES``."""
DatasetName = str
ModelName = Literal['var', 'vecm', 'dfm', 'ddfm']

# Training types
Batch = Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]
"""Data tensor or (data, target) tuple."""

# Result types
ResultDict = Dict[str, Any]
ForecastDict = Dict[str, Union[float, np.ndarray, Dict[str, Any]]]
MetricsDict = Dict[str, float]
CheckpointDict = Dict[str, Any]
ConfigDict = Dict[str, Any]

Device = Union["torch.device", str]

# Shape types
Shape = Tuple[int, ...]
Shape2D = Tuple[int, int]
"""(T, N): time steps, variables."""
Shape3D = Tuple[int, int, int]
"""(T, N, N) or (H, N, N)."""
Shape4D = Tuple[int, int, int, int]
"""(B, T, N, N)."""

# Forecast and analysis types
ForecastHorizon = int
IRFHorizon = int
LagOrder = int
NumFactors = int
NumVars = int

PathLike = Union[str, Path]

# Validation types
ValidationIssue = Dict[str, Any]
ValidationResult = Dict[str, Any]


def is_numpy_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray)


def is_torch_tensor(obj: Any) -> bool:
    # A tensor exists only once torch is imported, so this never imports it.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


def is_array_like(obj: Any) -> bool:
    return is_numpy_array(obj) or is_torch_tensor(obj)


def get_array_shape(arr: ArrayLike) -> Shape:
    """Shape of a numpy array or torch tensor as a tuple; raises ``DataValidationError`` otherwise."""
    if is_numpy_array(arr):
        return arr.shape
    elif is_torch_tensor(arr):
        return tuple(arr.shape)
    else:
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Expected numpy array or torch tensor, got {type(arr)}",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


def to_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert to a numpy array (tensors are detached and moved to CPU), cast to ``dtype`` if given."""
    if is_numpy_array(arr):
        result = arr
    elif is_torch_tensor(arr):
        if arr.requires_grad:
            arr = arr.detach()
        result = arr.cpu().numpy()
    else:
        result = np.asarray(arr)

    if dtype is not None:
        result = result.astype(dtype)
    return result


def to_tensor(arr: ArrayLike, device: Optional[Device] = None, dtype: Optional['torch.dtype'] = None) -> "torch.Tensor":
    """Convert a numpy array or tensor to a tensor on ``device`` with ``dtype``; raises ``DataValidationError`` otherwise."""
    import torch

    if is_torch_tensor(arr):
        result = arr
        if device is not None:
            result = result.to(device)
        if dtype is not None:
            result = result.to(dtype)
        return result
    elif is_numpy_array(arr):
        # Hand torch a writable C-contiguous array: from_numpy warns on a
        # read-only array (pandas copy-on-write ``.values``) and keeps the
        # strides of a Fortran-ordered frame view, so the same data would reach
        # the kernels with a different layout depending on the pandas mode.
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        elif not arr.flags.writeable:
            arr = arr.copy()
        tensor = torch.from_numpy(arr)
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    else:
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Cannot convert {type(arr)} to torch tensor",
            details=f"Input type: {type(arr).__name__}, value shape: {getattr(arr, 'shape', 'N/A')}"
        )


def __getattr__(name: str) -> Any:
    # The aliases of torch classes; resolving one imports torch.
    if name not in ('Tensor', 'Loss', 'Optimizer'):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import torch

    return torch.optim.Optimizer if name == 'Optimizer' else torch.Tensor


__all__ = [
    'ArrayLike',
    'FloatArray',
    'IntArray',
    'BoolArray',
    'OptionalArray',
    'OptionalTensor',
    'OptionalArrayLike',
    'FactorState',
    'ObservationState',
    'ForecastResult',
    'CoefficientMatrix',
    'CovarianceMatrix',
    'SeriesID',
    'Frequency',
    'DatasetName',
    'ModelName',
    'Batch',
    'Loss',  # noqa: F822 - resolved by __getattr__
    'Optimizer',  # noqa: F822 - resolved by __getattr__
    'ResultDict',
    'ForecastDict',
    'MetricsDict',
    'CheckpointDict',
    'ConfigDict',
    'Device',
    'Shape',
    'Shape2D',
    'Shape3D',
    'Shape4D',
    'ForecastHorizon',
    'IRFHorizon',
    'LagOrder',
    'NumFactors',
    'NumVars',
    'PathLike',
    'ValidationIssue',
    'ValidationResult',
    'is_numpy_array',
    'is_torch_tensor',
    'is_array_like',
    'get_array_shape',
    'to_numpy',
    'to_tensor',
    'Tensor',  # noqa: F822 - resolved by __getattr__
]
