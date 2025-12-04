"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFMLinear, DFM
# Note: Legacy module-level functions removed - use instance methods and trainer.fit() pattern
# Note: load_data removed - use DFMDataModule instead
from ..config.results import BaseResult, DFMResult, DDFMResult, FitParams, DFMParams

__all__ = [
    'BaseFactorModel', 'DFMLinear', 'DFM',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'DFMParams',
]

# DDFM (PyTorch is mandatory)
from .ddfm import DDFM, DDFMModel
# Note: load_data_ddfm removed - use DFMDataModule instead
__all__.extend([
    'DDFM',  # High-level API
    'DDFMModel',  # Low-level implementation
])

