"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, SeriesConfig, BlockConfig, Params) in schema.py
- IO (ConfigSource, YamlSource, etc.) in io.py
"""

from .schema import (
    BaseModelConfig, DFMConfig, DDFMConfig, SeriesConfig, BlockConfig,
    DEFAULT_GLOBAL_BLOCK_NAME,
)
from .params import Params, FitParams
from .results import BaseResult, DFMResult, DDFMResult, DFMParams
from .utils import validate_frequency, validate_transformation
from .io import (
    ConfigSource,
    YamlSource,
    DictSource,
    HydraSource,
    MergedConfigSource,
    make_config_source,
    from_spec,
    # Note: Private functions (_load_config_from_dataframe, _write_series_blocks_yaml)
    # are available from io module but not exported as public API
)
from .structure import (
    FREQUENCY_HIERARCHY,
    PERIODS_PER_YEAR,
    get_periods_per_year,
    compute_idio_chain_lengths,
    get_tent_weights_for_pair,
    generate_tent_weights,
    generate_R_mat,
    get_aggregation_structure,
    group_series_by_frequency,
)

__all__ = [
    # Schema
    'BaseModelConfig', 'DFMConfig', 'DDFMConfig', 'SeriesConfig', 'BlockConfig',
    'DEFAULT_GLOBAL_BLOCK_NAME',
    # Parameters
    'Params', 'FitParams',
    # Results
    'BaseResult', 'DFMResult', 'DDFMResult', 'DFMParams',
    # Utilities
    'validate_frequency', 'validate_transformation',
    # IO
    'ConfigSource', 'YamlSource', 'DictSource',
    'HydraSource', 'MergedConfigSource', 'make_config_source',
    'from_spec',
    # Structure utilities
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'get_periods_per_year',
    'compute_idio_chain_lengths',
    'get_tent_weights_for_pair',
    'generate_tent_weights',
    'generate_R_mat',
    'get_aggregation_structure',
    'group_series_by_frequency',
    # Note: Private functions (_load_config_from_dataframe, _write_series_blocks_yaml)
    # are available from io module but not exported as public API
]

