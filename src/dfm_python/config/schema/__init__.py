"""Configuration schema package for DFM models.

This package contains:
- model.py: BaseModelConfig, DFMConfig, DDFMConfig - model configurations
- results.py: BaseResult, DFMResult, DDFMResult - result structures

Note: Series are specified via frequency dict mapping column names to frequencies.
"""

from .model import BaseModelConfig, DFMConfig, DDFMConfig, iVDFMConfig
from .results import BaseResult, DFMResult, DDFMResult, iVDFMResult
from .params import DFMStateSpaceParams, DDFMStateSpaceParams, iVDFMModelState

__all__ = [
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'iVDFMConfig',
    'BaseResult', 'DFMResult', 'DDFMResult', 'iVDFMResult',
    'DFMStateSpaceParams', 'DDFMStateSpaceParams', 'iVDFMModelState',
]

