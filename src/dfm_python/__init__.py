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

Example (High-level API - Recommended):
    >>> import dfm_python as dfm
    >>> dfm.load_config('config/default.yaml')
    >>> dfm.load_data('data/sample_data.csv')
    >>> dfm.train(max_iter=1)
    >>> Xf, Zf = dfm.predict(horizon=6)
    >>> dfm.plot(kind='factor', factor_index=0, forecast_horizon=6, save_path='outputs/factor_forecast.png')
    >>> factors = dfm.get_result().Z  # Access factors via result
    
Example (Low-level API - For advanced usage):
    >>> from dfm_python import DFM, DFMConfig, SeriesConfig
    >>> from dfm_python.data import load_data  # Preferred import
    >>> # or for backward compatibility: from dfm_python.data_loader import load_data
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

__version__ = "0.2.4"

from .config import DFMConfig, SeriesConfig, BlockConfig, Params, DEFAULT_GLOBAL_BLOCK_NAME
from .config import (
    ConfigSource,
    YamlSource,
    DictSource,
    SpecCSVSource,
    HydraSource,
    MergedConfigSource,
    make_config_source,
)
from .data.transformer import transform_data
from .dfm import DFMResult
from .core.diagnostics import calculate_rmse, diagnose_series, print_series_diagnosis
from .dfm import DFM as _DFMCore  # Core DFM class from dfm.py
from .kalman import run_kf, skf, fis, miss_data
from .news import update_nowcast, news_dfm, para_const

# Import high-level API (extends core DFM with convenience methods)
from .api import DFM, _dfm_instance, from_yaml, from_spec, from_spec_df, from_dict

# Module-level API: expose singleton instance methods and properties
# This allows: import dfm_python as dfm; dfm.load_config(...); dfm.result
def load_config(
    source=None,
    *,
    yaml=None,
    mapping=None,
    spec=None,
    hydra=None,
    base=None,
    override=None,
    config=None,
):
    """Load configuration (module-level convenience function).
    
    See DFM.load_config() for parameter documentation.
    """
    return _dfm_instance.load_config(
        source=source,
        yaml=yaml,
        mapping=mapping,
        spec=spec,
        hydra=hydra,
        base=base,
        override=override,
        config=config,
    )

def load_data(data_path=None, data=None, **kwargs):
    """Load data (module-level convenience function)."""
    return _dfm_instance.load_data(data_path=data_path, data=data, **kwargs)

def train(threshold=None, max_iter=None, **kwargs):
    """Train the model (module-level convenience function)."""
    return _dfm_instance.train(threshold=threshold, max_iter=max_iter, **kwargs)

def predict(horizon=12, **kwargs):
    """Forecast using the trained model (module-level convenience function)."""
    return _dfm_instance.predict(horizon=horizon, **kwargs)

def plot(**kwargs):
    """Plot common visualizations (module-level convenience function)."""
    return _dfm_instance.plot(**kwargs)

def reset():
    """Reset state (module-level convenience function)."""
    return _dfm_instance.reset()

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

# For direct attribute access, we'll use __getattr__ but handle config module conflict

def __getattr__(name):
    """Allow access to DFM instance properties at module level."""
    # Handle special cases that conflict with submodules
    if name == 'config' and _dfm_instance.config is not None:
        return _dfm_instance.config
    elif name == 'data':
        return _dfm_instance.data
    elif name == 'time':
        return _dfm_instance.time
    elif name == 'result':
        return _dfm_instance.result
    elif name == 'original_data':
        return _dfm_instance.original_data
    # For other attributes, try normal import
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core classes
    'DFMConfig', 'SeriesConfig', 'BlockConfig', 'Params', 'DFM',
    # Constants
    'DEFAULT_GLOBAL_BLOCK_NAME',
    # Config sources
    'ConfigSource', 'YamlSource', 'DictSource', 'SpecCSVSource', 'HydraSource',
    'MergedConfigSource', 'make_config_source',
    # High-level API (module-level - recommended)
    'load_config',
    'load_data', 'train', 'predict', 'plot', 'reset',
    'get_config', 'get_data', 'get_time', 'get_result', 'get_original_data',
    # Convenience constructors (cleaner API)
    'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
    # Low-level API (functional interface - advanced usage)
    'transform_data',
    'DFMResult', 'calculate_rmse', 'diagnose_series', 'print_series_diagnosis',
    'run_kf', 'skf', 'fis', 'miss_data',
    'update_nowcast', 'news_dfm', 'para_const',
]

