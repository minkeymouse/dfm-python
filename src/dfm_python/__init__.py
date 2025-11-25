"""Dynamic Factor Model (DFM) package for Python.

This package implements a comprehensive Dynamic Factor Model framework with support for:
- Mixed-frequency time series data (monthly, quarterly, semi-annual, annual)
- Clock-based synchronization of latent factors
- Tent kernel aggregation for low-to-high frequency mapping
- Expectation-Maximization (EM) algorithm for parameter estimation
- Kalman filtering and smoothing for factor extraction
- News decomposition for nowcasting
- Deep Dynamic Factor Models (DDFM) with nonlinear encoders (optional, requires PyTorch)

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

Example (High-level API - Recommended):
    >>> import dfm_python as dfm
    >>> # Linear DFM
    >>> model = dfm.DFM()
    >>> model.load_config('config/default.yaml')
    >>> model.load_data('data/sample_data.csv')
    >>> model.train(max_iter=100)
    >>> Xf, Zf = model.predict(horizon=6)
    >>> 
    >>> # Or use DDFM (separate class)
    >>> ddfm_model = dfm.DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> ddfm_model.load_config('config/default.yaml')
    >>> ddfm_model.load_data('data/sample_data.csv')
    >>> ddfm_model.train(epochs=100)
    >>> Xf, Zf = ddfm_model.predict(horizon=6)
    
Example (Low-level API - For advanced usage):
    >>> from dfm_python import DFM, DFMConfig, SeriesConfig
    >>> from dfm_python.dataloader import load_data  # Preferred import
    >>> # Option 1: Load from YAML
    >>> config = load_config('config.yaml')
    >>> # Option 2: Create directly
    >>> config = DFMConfig(
    ...     series=[SeriesConfig(frequency='m', transformation='lin', blocks=[1], series_id='series1')],
    ...     block_names=['Global']
    ... )
    >>> X, Time, _ = load_data('data.csv', config)  # or use database adapter
    >>> model = DFM()
    >>> result = model.fit(X, config)
    >>> factors = result.Z  # Extract estimated factors

For detailed documentation, see the README.md file and the tutorial notebooks/scripts.
"""

__version__ = "0.3.0"

# ============================================================================
# PUBLIC API DEFINITION
# ============================================================================
# This __init__.py is the single source of truth for the public API.
# All symbols exported here are considered stable public API.
# Internal reorganization should not break these imports.
#
# Public API categories:
# 1. Configuration: DFMConfig, SeriesConfig, BlockConfig, Params, config sources
# 2. High-level API: DFM, DDFM, module-level convenience functions
# 3. Core utilities: DFMCore, run_kf, TimeIndex, diagnostics
# 4. Models: BaseFactorModel, DFMLinear, DDFM (low-level)
# 5. Nowcasting: Nowcast, result classes, para_const
# 6. Data & Results: DFMResult, transform_data
# ============================================================================

# Configuration (from config/ subpackage)
from .config import (
    DFMConfig, SeriesConfig, BlockConfig, Params, DEFAULT_GLOBAL_BLOCK_NAME,
    ConfigSource, YamlSource, DictSource, HydraSource,
    MergedConfigSource, make_config_source,
)

# Data utilities
from .dataloader import transform_data

# Results
from .core import DFMResult

# Core utilities (from core/ subpackage)
from .core import calculate_rmse, diagnose_series, print_series_diagnosis
from .core.state_space import run_kf, skf, fis, miss_data
# DFMCore is now an alias for DFMLinear (backward compatibility)
from .models.dfm import DFMLinear
DFMCore = DFMLinear

# Nowcasting (already in nowcasting/ subpackage)
from .nowcasting import (
    Nowcast,
    para_const,
    NowcastResult,
    NewsDecompResult,
    BacktestResult,
)

# High-level API (from api/ subpackage)
from .api import (
    DFM, _dfm_instance, from_yaml, from_spec, from_spec_df, from_dict,
    load_config, load_data, load_pickle, train, predict, plot, reset, create_model
)

# Model implementations
from .models.base import BaseFactorModel
from .models.dfm import DFMLinear

# DDFM high-level API and low-level model (both optional, requires PyTorch)
try:
    from .api import DDFM as DDFMAPI, _ddfm_instance
    from .api import load_config_ddfm, load_data_ddfm, train_ddfm, predict_ddfm, plot_ddfm, reset_ddfm
    from .models.ddfm import DDFM as DDFMModel
    _has_ddfm = True
    # Export high-level API as DDFM
    DDFM = DDFMAPI
except ImportError:
    _has_ddfm = False
    DDFM = None  # type: ignore
    DDFMModel = None  # type: ignore
    load_config_ddfm = None  # type: ignore
    load_data_ddfm = None  # type: ignore
    train_ddfm = None  # type: ignore
    predict_ddfm = None  # type: ignore
    plot_ddfm = None  # type: ignore
    reset_ddfm = None  # type: ignore

# Expose properties as module-level attributes
# Use property-like access via functions or direct attribute access
# Since 'config' conflicts with the config module, we'll use a different approach
# Users can access: dfm.get_config(), dfm.get_data(), dfm.get_result()
def get_config():
    """Get current configuration."""
    return _dfm_instance.config

def get_data():
    """Get current data matrix."""
    return _dfm_instance.data

def get_time():
    """Get current time index."""
    return _dfm_instance.time

def get_result():
    """Get training result."""
    return _dfm_instance.result

def get_original_data():
    """Get original (untransformed) data matrix."""
    return _dfm_instance.original_data

__all__ = [
    # Core classes
    'DFMConfig', 'SeriesConfig', 'BlockConfig', 'Params', 'DFM', 'DFMCore', 'Nowcast',
    # Model base and implementations
    'BaseFactorModel', 'DFMLinear',
    # Nowcast result classes
    'NowcastResult', 'NewsDecompResult', 'BacktestResult',
    # Constants
    'DEFAULT_GLOBAL_BLOCK_NAME',
    # Config sources
    'ConfigSource', 'YamlSource', 'DictSource', 'HydraSource',
    'MergedConfigSource', 'make_config_source',
    # High-level API (module-level - recommended)
    'load_config',
    'load_data', 'load_pickle', 'train', 'predict', 'plot', 'reset', 'create_model',
    'get_config', 'get_data', 'get_time', 'get_result', 'get_original_data',
    # Convenience constructors (cleaner API)
    'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
    # Low-level API (functional interface - advanced usage)
    'transform_data',
    'DFMResult', 'calculate_rmse', 'diagnose_series', 'print_series_diagnosis',
    'run_kf', 'skf', 'fis', 'miss_data',
    'para_const',  # Internal utility, kept for backward compatibility
]

# Add DDFM high-level API and convenience functions if available
if _has_ddfm:
    __all__.extend([
        'DDFM',  # High-level API class
        'load_config_ddfm', 'load_data_ddfm', 'train_ddfm', 
        'predict_ddfm', 'plot_ddfm', 'reset_ddfm'
    ])

