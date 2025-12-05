"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, SeriesConfig) in schema.py
- IO (ConfigSource, YamlSource, etc.) in adapter.py
"""

from .schema import (
    BaseModelConfig, DFMConfig, DDFMConfig, SeriesConfig,
    DEFAULT_BLOCK_NAME,
)
from .results import BaseResult, DFMResult, DDFMResult, FitParams
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
    # Schema
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'SeriesConfig',
    'DEFAULT_BLOCK_NAME',
    # Parameter overrides
    'FitParams',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult',
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

