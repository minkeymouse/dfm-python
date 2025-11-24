"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFMLinear

__all__ = ['BaseFactorModel', 'DFMLinear']

# DDFM is optional (requires PyTorch)
try:
    from .ddfm import DDFM
    __all__.append('DDFM')
except ImportError:
    DDFM = None  # type: ignore

