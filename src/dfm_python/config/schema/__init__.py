"""Configuration schema package for DFM models.

This package contains:
- model.py: BaseModelConfig, DFMConfig, DDFMConfig, KDFMConfig - model configurations
- results.py: BaseResult, DFMResult, DDFMResult, KDFMResult - result structures

Note: Series are specified via frequency dict mapping column names to frequencies.
"""

from .model import BaseModelConfig, DFMConfig, DDFMConfig, KDFMConfig
from .results import BaseResult, DFMResult, DDFMResult, KDFMResult, FitParams

__all__ = [
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'KDFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult', 'KDFMResult', 'FitParams',
]

