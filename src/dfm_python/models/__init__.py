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
try:
    from .ddfm.ddfm import DDFM
    __all__.extend([
        'DDFM',  # High-level API
    ])
except ImportError:
    # DDFM not available (missing torch or other dependencies)
    pass

# AFM implementation (Attention Factor Model for statistical arbitrage)
try:
    from .afm.afm import AFM
    __all__.extend([
        'AFM',  # High-level API
    ])
except ImportError:
    # AFM not available (missing torch or other dependencies)
    pass

