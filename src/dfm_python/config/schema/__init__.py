"""Configuration schema package for DFM models.

This package contains:
- series.py: SeriesConfig - configuration for individual time series
- model.py: BaseModelConfig, DFMConfig, DDFMConfig, KDFMConfig - model configurations
- results.py: BaseResult, DFMResult, DDFMResult, KDFMResult - result structures
"""

from .series import SeriesConfig
from .model import BaseModelConfig, DFMConfig, DDFMConfig, KDFMConfig
from .results import BaseResult, DFMResult, DDFMResult, KDFMResult, FitParams

__all__ = [
    'SeriesConfig',
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'KDFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult', 'KDFMResult', 'FitParams',
]

