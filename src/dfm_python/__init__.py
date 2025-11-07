"""Dynamic Factor Model (DFM) package for Python.

This package implements a comprehensive Dynamic Factor Model framework with support for:
- Mixed-frequency time series data (daily, weekly, monthly, quarterly, semi-annual, annual)
- Clock-based synchronization of latent factors
- Tent kernel aggregation for low-to-high frequency mapping
- Expectation-Maximization (EM) algorithm for parameter estimation
- Kalman filtering and smoothing for factor extraction
- News decomposition for nowcasting

The package follows the FRBNY (Federal Reserve Bank of New York) approach to mixed-frequency
DFMs, where all latent factors (global and block-level) are synchronized to a common "clock"
frequency, typically monthly. Lower-frequency observed variables are mapped to higher-frequency
latent states using deterministic tent kernels in the observation equation.

Key Features:
    - Unified configuration system (YAML/CSV)
    - Flexible block structure for factor modeling
    - Robust handling of missing data
    - Comprehensive transformation support
    - News decomposition for forecast updates

Example:
    >>> from dfm_python import load_config, load_data, dfm
    >>> config = load_config('config.yaml')
    >>> X, Time, _ = load_data('data.csv', config)
    >>> result = dfm(X, config)
    >>> factors = result.Z  # Extract estimated factors

For detailed documentation, see the README.md file.
"""

__version__ = "0.1.2"

from .config import DFMConfig
from .data_loader import (
    load_config, load_config_from_yaml, load_config_from_csv, load_data, transform_data
)
from .dfm import DFMResult, dfm
from .kalman import run_kf, skf, fis, miss_data
from .news import update_nowcast, news_dfm, para_const

# Backward compatibility aliases
ModelConfig = DFMConfig  # Deprecated: ModelConfig merged into DFMConfig
ModelSpec = DFMConfig  # Deprecated: use DFMConfig
load_spec = load_config  # Deprecated: use load_config

__all__ = [
    'DFMConfig',
    # Backward compatibility
    'ModelConfig',  # Alias for DFMConfig
    'load_config', 'load_config_from_yaml', 'load_config_from_csv',
    'load_data', 'transform_data',
    'DFMResult', 'dfm',
    'run_kf', 'skf', 'fis', 'miss_data',
    'update_nowcast', 'news_dfm', 'para_const',
    # Deprecated aliases
    'ModelSpec', 'load_spec',
]

