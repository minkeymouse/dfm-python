"""Factor model implementations.

This package contains implementations of different factor models:
- DFM (Dynamic Factor Model): Linear factor model with EM estimation
- DDFM (Deep Dynamic Factor Model): Nonlinear encoder with PyTorch
"""

from .base import BaseFactorModel
from .dfm import DFMLinear, DFM
from .dfm import (
    from_yaml, from_spec, from_spec_df, from_dict,
    load_config, load_data, load_pickle, train, predict, plot, reset, create_model,
)
from .results import BaseResult, DFMResult, DDFMResult, DFMParams

__all__ = [
    'BaseFactorModel', 'DFMLinear', 'DFM',
    'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
    'load_config', 'load_data', 'load_pickle', 'train', 'predict', 'plot', 'reset', 'create_model',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'DFMParams',
]

# DDFM is optional (requires PyTorch)
try:
    from .ddfm import DDFM, DDFMModel
    from .ddfm import load_config_ddfm, load_data_ddfm, train_ddfm, predict_ddfm, plot_ddfm, reset_ddfm
    __all__.extend([
        'DDFM',  # High-level API
        'DDFMModel',  # Low-level implementation
        'load_config_ddfm', 'load_data_ddfm', 'train_ddfm', 'predict_ddfm', 'plot_ddfm', 'reset_ddfm',
    ])
except ImportError:
    DDFM = None  # type: ignore
    DDFMModel = None  # type: ignore

