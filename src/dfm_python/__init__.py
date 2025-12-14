"""Dynamic Factor Model (DFM) package for Python.

This package implements a comprehensive Dynamic Factor Model framework with support for:
- Mixed-frequency time series data (monthly, quarterly, semi-annual, annual)
- Clock-based synchronization of latent factors
- Tent kernel aggregation for low-to-high frequency mapping
- Expectation-Maximization (EM) algorithm for parameter estimation
- Kalman filtering and smoothing for factor extraction
- Deep Dynamic Factor Models (DDFM) with nonlinear encoders (requires PyTorch)

The package implements a clock-based approach to mixed-frequency DFMs, where all latent 
factors (global and block-level) are synchronized to a common "clock" frequency, typically 
monthly. Lower-frequency observed variables are mapped to higher-frequency latent states 
using deterministic tent kernels in the observation equation.

Note: Higher frequencies (daily, weekly) than the clock are not supported. If any series
has a frequency faster than the clock, a ValueError will be raised.

Key Features:
    - Hydra-based configuration (YAML files) - primary configuration method
    - Expects preprocessed data from users - users handle all preprocessing, package extracts statistics from pipeline
    - Flexible block structure for factor modeling
    - Robust handling of missing data (internal spline interpolation)
    - Automatic standardization and data clipping

Example (Standard Lightning Pattern):
    >>> from dfm_python import DFM, DFMDataModule, DFMTrainer
    >>> import pandas as pd
    >>> 
    >>> # Step 1: Load and preprocess data
    >>> df = pd.read_csv('data/sample_data.csv')
    >>> df_processed = df[[col for col in df.columns if col != 'date']]
    >>> 
    >>> # Step 2: Create DataModule
    >>> dm = DFMDataModule(config_path='config/default.yaml', data=df_processed)
    >>> dm.setup()
    >>> 
    >>> # Step 3: Create model and load config
    >>> model = DFM()
    >>> model.load_config('config/default.yaml')
    >>> 
    >>> # Step 4: Create trainer and fit (standard Lightning pattern)
    >>> trainer = DFMTrainer(max_epochs=100)
    >>> trainer.fit(model, dm)
    >>> 
    >>> # Step 5: Predict
    >>> Xf, Zf = model.predict(horizon=6)
    >>> 
    >>> # Or use DDFM (same pattern)
    >>> from dfm_python import DDFM, DDFMTrainer
    >>> 
    >>> dm_ddfm = DFMDataModule(config_path='config/default.yaml', data=df_processed)
    >>> dm_ddfm.setup()
    >>> 
    >>> ddfm_model = DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> ddfm_model.load_config('config/default.yaml')
    >>> 
    >>> trainer_ddfm = DDFMTrainer(max_epochs=100)
    >>> trainer_ddfm.fit(ddfm_model, dm_ddfm)
    >>> Xf, Zf = ddfm_model.predict(horizon=6)
    
Note: DFMConfig and SeriesConfig are internal implementation details.
    Users should use Hydra YAML configuration files instead.

For detailed documentation, see the README.md file and the tutorial notebooks/scripts.
"""

__version__ = "0.5.5"

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
# 3. Core utilities: TimeIndex, diagnostics
# 4. Models: BaseFactorModel, DDFM (low-level)
# 5. Data & Results: DFMResult
# ============================================================================

# Configuration (from config/ subpackage)
from .config import (
    DEFAULT_BLOCK_NAME,
    ConfigSource, YamlSource, HydraSource,
    make_config_source,
)
# Internal imports (for backward compatibility, but not recommended)
from .config import DFMConfig, SeriesConfig  # Internal use only

# Results
from .config.results import DFMResult, DDFMResult, BaseResult

# Utilities (from utils/ subpackage)
from .utils.diagnostics import diagnose_series, print_series_diagnosis
from .utils.time import calculate_rmse

# PyTorch Lightning modules (mandatory dependency)
# Users can import these directly from dfm_python
from .lightning import (
    DFMDataModule,
    DDFMDataModule,  # DDFM-specific DataModule
    DFMDataset,  # Dataset class (usually not needed directly)
    DDFMDataset,  # DDFM Dataset class (usually not needed directly)
    KalmanFilter,  # Module class
    EMAlgorithm,  # Module class
)

# Model implementations
from .models.base import BaseFactorModel
from .models.dfm import DFM

# DDFM high-level API and low-level model (PyTorch is mandatory)
from .models.ddfm import DDFM, DDFMModel

__all__ = [
    # Core classes
    'DFM',
    # Model base and implementations
    'BaseFactorModel',
    # Constants
    'DEFAULT_BLOCK_NAME',
    # Config sources
    'ConfigSource', 'YamlSource', 'HydraSource',
    'make_config_source',
    # Low-level API (functional interface - advanced usage)
    'BaseResult', 'DFMResult', 'DDFMResult', 'calculate_rmse', 'diagnose_series', 'print_series_diagnosis',
]

# DDFM high-level API (PyTorch is mandatory)
__all__.extend([
    'DDFM',  # High-level API class
    'DDFMModel',  # Low-level implementation (used internally)
])

# Lightning modules (mandatory dependency)
__all__.extend([
    'DFMDataModule',
    'DDFMDataModule',  # DDFM-specific DataModule
    'DFMDataset',  # Dataset class (usually not needed directly)
    'DDFMDataset',  # DDFM Dataset class (usually not needed directly)
    'KalmanFilter',  # Module class
    'EMAlgorithm',  # Module class
])

# Trainer classes (mandatory dependency)
from .trainer import DFMTrainer, DDFMTrainer

__all__.extend([
    'DFMTrainer',
    'DDFMTrainer',
])


