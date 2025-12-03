"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, SeriesConfig) in schema.py
- IO (ConfigSource, YamlSource, etc.) in adapter.py
"""

from .schema import (
    BaseModelConfig, DFMConfig, DDFMConfig, SeriesConfig,
    DEFAULT_BLOCK_NAME,
)
from .results import BaseResult, DFMResult, DDFMResult, FitParams, DFMParams
from .utils import validate_frequency, validate_transformation
from .adapter import (
    ConfigSource,
    YamlSource,
    DictSource,
    HydraSource,
    MergedConfigSource,
    make_config_source,
    # Note: from_spec is in models/dfm.py, not in adapter
    # Note: Private functions (_load_config_from_dataframe, _write_series_blocks_yaml)
    # are available from adapter module but not exported as public API
)
from .utils import (
    FREQUENCY_HIERARCHY,
    PERIODS_PER_YEAR,
    get_periods_per_year,
    get_annual_factor,
    compute_idio_chain_lengths,
    get_tent_weights_for_pair,
    generate_tent_weights,
    generate_R_mat,
    get_aggregation_structure,
    group_series_by_frequency,
)

__all__ = [
    # Schema
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'SeriesConfig',
    'DEFAULT_BLOCK_NAME',
    # Parameter overrides
    'FitParams', 'DFMParams',  # DFMParams is alias for FitParams (backward compatibility)
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
    'compute_idio_chain_lengths',
    'get_tent_weights_for_pair',
    'generate_tent_weights',
    'generate_R_mat',
    'get_aggregation_structure',
    'group_series_by_frequency',
    # Note: Private functions (_load_config_from_dataframe, _write_series_blocks_yaml)
    # are available from adapter module but not exported as public API
]

