"""Dynamic Factor Model (DFM) package for Python.

This package implements a comprehensive Dynamic Factor Model framework with support for:
- Mixed-frequency time series data (monthly, quarterly, semi-annual, annual)
- Clock-based synchronization of latent factors
- Tent kernel aggregation for low-to-high frequency mapping
- Expectation-Maximization (EM) algorithm for parameter estimation
- Kalman filtering and smoothing for factor extraction
- News decomposition for nowcasting
- Deep Dynamic Factor Models (DDFM) with nonlinear encoders (requires PyTorch)

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
    >>> from dfm_python.lightning import DFMDataModule
    >>> from sktime.transformations.compose import ColumnTransformer
    >>> 
    >>> # Linear DFM
    >>> model = dfm.DFM()
    >>> model.load_config('config/default.yaml')
    >>> 
    >>> # Create transformer (user must provide)
    >>> transformer = ColumnTransformer([...])  # User-defined sktime transformer
    >>> 
    >>> # Create DataModule
    >>> data_module = DFMDataModule(config=model.config, transformer=transformer, data_path='data/sample_data.csv')
    >>> data_module.setup()
    >>> 
    >>> # Train
    >>> model.train(data_module, max_iter=100)
    >>> Xf, Zf = model.predict(horizon=6)
    >>> 
    >>> # Or use DDFM (separate class)
    >>> ddfm_model = dfm.DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> ddfm_model.load_config('config/default.yaml')
    >>> 
    >>> # Create DataModule for DDFM
    >>> data_module_ddfm = DFMDataModule(config=ddfm_model.config, transformer=transformer, data_path='data/sample_data.csv')
    >>> data_module_ddfm.setup()
    >>> 
    >>> # Train
    >>> ddfm_model.train(data_module_ddfm, epochs=100)
    >>> Xf, Zf = ddfm_model.predict(horizon=6)
    
Example (Low-level API - For advanced usage):
    >>> from dfm_python import DFM, DFMConfig, SeriesConfig
    >>> from dfm_python.config.adapter import YamlSource
    >>> # Option 1: Load from YAML
    >>> config = YamlSource('config.yaml').load()
    >>> # Option 2: Create directly
    >>> config = DFMConfig(
    ...     series=[SeriesConfig(frequency='m', transformation='lin', blocks=[1], series_id='series1')],
    ...     block_names=['Global']
    ... )
    >>> from dfm_python.lightning import DFMDataModule
    >>> from sktime.transformations.compose import ColumnTransformer
    >>> 
    >>> # Create transformer (user must provide)
    >>> transformer = ColumnTransformer([...])  # User-defined
    >>> 
    >>> # Create DataModule
    >>> data_module = DFMDataModule(config=config, transformer=transformer, data_path='data.csv')
    >>> data_module.setup()
    >>> 
    >>> # Fit model
    >>> model = DFM()
    >>> result = model.fit(data_module, config)
    >>> factors = result.Z  # Extract estimated factors

For detailed documentation, see the README.md file and the tutorial notebooks/scripts.
"""

__version__ = "0.4.0"

# ============================================================================
# PUBLIC API DEFINITION
# ============================================================================
# This __init__.py is the single source of truth for the public API.
# All symbols exported here are considered stable public API.
# Internal reorganization should not break these imports.
#
# Public API categories:
# 1. Configuration: DFMConfig, SeriesConfig, config sources
# 2. High-level API: DFM, DDFM, module-level convenience functions
# 3. Core utilities: DFMLinear, TimeIndex, diagnostics
# 4. Models: BaseFactorModel, DFMLinear, DDFM (low-level)
# 5. Nowcasting: Nowcast, result classes, para_const
# 6. Data & Results: DFMResult
# ============================================================================

# Configuration (from config/ subpackage)
from .config import (
    DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME,
    ConfigSource, YamlSource, DictSource, HydraSource,
    MergedConfigSource, make_config_source,
)

# Data utilities
# Note: transform_data and DFMScaler have been removed - users must provide their own sktime transformers to DFMDataModule

# Results
from .config.results import DFMResult, DDFMResult, BaseResult

# Utilities (from utils/ subpackage)
from .utils.diagnostics import diagnose_series, print_series_diagnosis
from .utils.time import calculate_rmse

# PyTorch Lightning modules (mandatory dependency)
from .lightning import (
    DFMLightningModule,
    DDFMLightningModule,
    DFMDataModule,
    KalmanFilter,  # New module class
    EMAlgorithm,  # New module class
)

# Nowcasting (from nowcast/ subpackage)
from .nowcast import (
    Nowcast,
    NowcastResult,
)
from .nowcast.helpers import (
    para_const,
    NewsDecompResult,
    BacktestResult,
)

# Model implementations
from .models.base import BaseFactorModel
from .models.dfm import DFMLinear, DFM
from .models.dfm import (
    from_yaml, from_spec, from_spec_df, from_dict,
    load_config, load_pickle, train, predict, plot, reset, create_model
)
# Note: load_data has been removed - use DFMDataModule instead

# DDFM high-level API and low-level model (PyTorch is mandatory)
from .models.ddfm import DDFM, DDFMModel

__all__ = [
    # Core classes
    'DFMConfig', 'SeriesConfig', 'DFM', 'DFMLinear', 'Nowcast',
    # Model base and implementations
    'BaseFactorModel',
    # Nowcast result classes
    'NowcastResult', 'NewsDecompResult', 'BacktestResult',
    # Constants
    'DEFAULT_BLOCK_NAME',
    # Config sources
    'ConfigSource', 'YamlSource', 'DictSource', 'HydraSource',
    'MergedConfigSource', 'make_config_source',
    # High-level API (module-level - recommended)
    'load_config',
    'load_pickle', 'train', 'predict', 'plot', 'reset', 'create_model',
    # Convenience constructors (cleaner API)
    'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
    # Low-level API (functional interface - advanced usage)
    'BaseResult', 'DFMResult', 'DDFMResult', 'calculate_rmse', 'diagnose_series', 'print_series_diagnosis',
    'para_const',  # Internal utility for nowcasting
]

# DDFM high-level API (PyTorch is mandatory)
__all__.extend([
    'DDFM',  # High-level API class
    'DDFMModel',  # Low-level implementation
    # Note: Module-level convenience functions have been removed - 
    # use DDFM() instance methods directly (train, predict, etc.)
])

# Lightning modules (mandatory dependency)
__all__.extend([
    'DFMLightningModule',
    'DDFMLightningModule',
    'DFMDataModule',
    'KalmanFilter',  # New module class
    'EMAlgorithm',  # New module class
])

# Trainer classes (mandatory dependency)
from .trainer import DFMTrainer, DDFMTrainer

__all__.extend([
    'DFMTrainer',
    'DDFMTrainer',
])

