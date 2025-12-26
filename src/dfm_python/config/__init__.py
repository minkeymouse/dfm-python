"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, SeriesConfig) in schema.py
- IO (ConfigSource, YamlSource, etc.) in adapter.py
"""

from .base import (
    BaseModelConfig, SeriesConfig, BaseResult,
    DEFAULT_BLOCK_NAME,
)
from .constants import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_TOLERANCE,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    MIN_EIGENVALUE,
    MIN_STD,
)
from .schema import (
    DFMConfig, DDFMConfig, KDFMConfig,
)
from .results import DFMResult, DDFMResult, KDFMResult, FitParams
from .utils import validate_frequency, validate_transformation
from .adapter import (
    ConfigSource,
    YamlSource,
    DictSource,
    HydraSource,
    MergedConfigSource,
    make_config_source,
)
from .utils import (
    FREQUENCY_HIERARCHY,
    PERIODS_PER_YEAR,
    get_periods_per_year,
    get_annual_factor,
    compute_idio_lengths,
    get_tent_weights,
    generate_tent_weights,
    generate_R_mat,
    get_agg_structure,
    group_by_freq,
)

__all__ = [
    # Base classes (from base.py)
    'BaseModelConfig', 'SeriesConfig', 'BaseResult',
    'DEFAULT_BLOCK_NAME',
    # Model-specific configs (from schema.py)
    'DFMConfig', 'DDFMConfig', 'KDFMConfig',
    # Parameter overrides
    'FitParams',
    # Model-specific results (from results.py)
    'DFMResult', 'DDFMResult', 'KDFMResult',
    # Utilities
    'validate_frequency', 'validate_transformation',
    # IO
    'ConfigSource', 'YamlSource', 'DictSource',
    'HydraSource', 'MergedConfigSource', 'make_config_source',
    # Frequency and aggregation utilities
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'get_periods_per_year',
    'get_annual_factor',
    'compute_idio_lengths',
    'get_tent_weights',
    'generate_tent_weights',
    'generate_R_mat',
    'get_agg_structure',
    'group_by_freq',
]

