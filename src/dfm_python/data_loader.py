"""Data loading and transformation utilities for DFM estimation.

This module provides comprehensive data handling for Dynamic Factor Models.
For backward compatibility, functions are re-exported from the data package.

Note: This module is maintained for backward compatibility. New code should
import directly from dfm_python.data package.
"""

# Backward compatibility imports
from .data.loader import load_data, read_data, sort_data, load_config, _load_config_from_dataframe
from .data.utils import rem_nans_spline, summarize
from .data.transformer import transform_data

# Re-export for backward compatibility
__all__ = [
    'load_data',
    'read_data',
    'sort_data',
    'rem_nans_spline',
    'summarize',
    'transform_data',
    'load_config',
    '_load_config_from_dataframe',
]
