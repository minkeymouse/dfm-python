"""Data loading and preprocessing utilities for DFM estimation.

This package provides functions for reading, transforming, and preprocessing
time series data for Dynamic Factor Model estimation.

Modules:
    - loader: Main data loading and transformation functions
    - preprocess: Data preprocessing utilities (to be implemented)
"""

from .loader import (
    load_data,
    read_data,
    rem_nans_spline,
    calculate_release_date,
    create_data_view,
    DataView,
)
from .loader import transform_data

__all__ = [
    'transform_data',
    'load_data',
    'read_data',
    'rem_nans_spline',
    'calculate_release_date',
    'create_data_view',
    'DataView',
]

