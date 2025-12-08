"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFM
from ..config.results import BaseResult, DFMResult, DDFMResult, FitParams

__all__ = [
    'BaseFactorModel', 'DFM',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'FitParams',
]

# DDFM implementation
from .ddfm import DDFM, DDFMModel
from .mcmc import DDFMMCMCTrainer
__all__.extend([
    'DDFM',  # High-level API
    'DDFMModel',  # Low-level implementation
    'DDFMMCMCTrainer',  # MCMC training procedure
])

