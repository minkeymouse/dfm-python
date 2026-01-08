"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFM
from ..config import BaseResult, DFMResult, DDFMResult

__all__ = [
    'BaseFactorModel', 'DFM',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult',
]

# DDFM implementation
from .ddfm import DDFM
__all__.extend([
    'DDFM',  # High-level API
])

