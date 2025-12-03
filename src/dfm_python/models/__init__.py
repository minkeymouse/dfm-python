"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFMLinear, DFM
from .dfm import (
    from_yaml, from_spec, from_spec_df, from_dict,
    load_config, load_pickle, train, predict, plot, reset, create_model,
)
# Note: load_data has been removed - use DFMDataModule instead
from ..config.results import BaseResult, DFMResult, DDFMResult, DFMParams

__all__ = [
    'BaseFactorModel', 'DFMLinear', 'DFM',
    'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
    'load_config', 'load_pickle', 'train', 'predict', 'plot', 'reset', 'create_model',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'DFMParams',
]

# DDFM is optional (requires PyTorch)
try:
    from .ddfm import DDFM, DDFMModel
    # Note: load_data_ddfm has been removed - use DFMDataModule instead
    __all__.extend([
        'DDFM',  # High-level API
        'DDFMModel',  # Low-level implementation
    ])
except ImportError:
    DDFM = None  # type: ignore
    DDFMModel = None  # type: ignore

