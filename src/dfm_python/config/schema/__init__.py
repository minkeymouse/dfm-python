"""Configuration schema package for DFM models.

This package contains:
- model.py: BaseModelConfig, DFMConfig, DDFMConfig - model configurations
- results.py: BaseResult, DFMResult, DDFMResult - result structures

Note: Series are specified via frequency dict mapping column names to frequencies.
"""

from .model import BaseModelConfig, DFMConfig, DDFMConfig
from .results import BaseResult, DFMResult, DDFMResult
from .params import DFMStateSpaceParams, DDFMStateSpaceParams

__all__ = [
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult',
    'DFMStateSpaceParams', 'DDFMStateSpaceParams',
]

