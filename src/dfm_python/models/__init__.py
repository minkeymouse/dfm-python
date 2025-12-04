"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFMLinear, DFM
from ..config.results import BaseResult, DFMResult, DDFMResult, FitParams

__all__ = [
    'BaseFactorModel', 'DFMLinear', 'DFM',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'FitParams',
]

# DDFM (PyTorch is mandatory)
from .ddfm import DDFM, DDFMModel
__all__.extend([
    'DDFM',  # High-level API
    'DDFMModel',  # Low-level implementation
])

