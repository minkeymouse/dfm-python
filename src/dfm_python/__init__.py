"""Dynamic Factor Model (DFM) package for Python.

This package implements a comprehensive Dynamic Factor Model framework with support for:
- Mixed-frequency time series data (monthly, quarterly, semi-annual, annual)
- Clock-based synchronization of latent factors
- Tent kernel aggregation for low-to-high frequency mapping
- Expectation-Maximization (EM) algorithm for parameter estimation
- Kalman filtering and smoothing for factor extraction
- News decomposition for nowcasting

The package implements a clock-based approach to mixed-frequency DFMs, where all latent 
factors (global and block-level) are synchronized to a common "clock" frequency, typically 
monthly. Lower-frequency observed variables are mapped to higher-frequency latent states 
using deterministic tent kernels in the observation equation.

Note: Higher frequencies (daily, weekly) than the clock are not supported. If any series
has a frequency faster than the clock, a ValueError will be raised.

Key Features:
    - Unified configuration system (YAML with Hydra/OmegaConf, or direct DFMConfig objects)
    - Flexible block structure for factor modeling
    - Robust handling of missing data
    - Comprehensive transformation support
    - News decomposition for forecast updates

Example:
    >>> from dfm_python import load_config, load_data, dfm, DFMConfig, SeriesConfig
    >>> # Option 1: Load from YAML
    >>> config = load_config('config.yaml')
    >>> # Option 2: Create directly
    >>> config = DFMConfig(
    ...     series=[SeriesConfig(frequency='m', transformation='lin', blocks=[1], series_id='series1')],
    ...     block_names=['Global']
    ... )
    >>> X, Time, _ = load_data('data.csv', config)  # or use database adapter
    >>> result = dfm(X, config)
    >>> factors = result.Z  # Extract estimated factors

For detailed documentation, see the README.md file.
"""

__version__ = "0.1.6"

from .config import DFMConfig, SeriesConfig
from .data_loader import (
    load_config, load_config_from_yaml, load_data, transform_data
)
# Deprecated: load_config_from_csv (use YAML or create DFMConfig directly)
try:
    from .data_loader import load_config_from_csv
except ImportError:
    pass
from .dfm import DFMResult, dfm, calculate_rmse, diagnose_series, print_series_diagnosis
from .kalman import run_kf, skf, fis, miss_data
from .news import update_nowcast, news_dfm, para_const

# Backward compatibility aliases (deprecated - use DFMConfig and load_config)
ModelConfig = DFMConfig  # Deprecated: use DFMConfig

__all__ = [
    'DFMConfig', 'SeriesConfig',
    # Backward compatibility
    'ModelConfig',  # Deprecated alias for DFMConfig
    'load_config', 'load_config_from_yaml',
    'load_data', 'transform_data',
    'DFMResult', 'dfm', 'calculate_rmse', 'diagnose_series', 'print_series_diagnosis',
    'run_kf', 'skf', 'fis', 'miss_data',
    'update_nowcast', 'news_dfm', 'para_const',
]

