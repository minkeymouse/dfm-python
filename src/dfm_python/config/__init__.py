"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, SeriesConfig, BlockConfig, Params) in schema.py
- IO (ConfigSource, YamlSource, etc.) in io.py
"""

from .schema import (
    DFMConfig, SeriesConfig, BlockConfig, Params,
    DEFAULT_GLOBAL_BLOCK_NAME,
    validate_frequency, validate_transformation,
)
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

__all__ = [
    # Schema
    'DFMConfig', 'SeriesConfig', 'BlockConfig', 'Params',
    'DEFAULT_GLOBAL_BLOCK_NAME',
    'validate_frequency', 'validate_transformation',
    # IO
    'ConfigSource', 'YamlSource', 'DictSource',
    'HydraSource', 'MergedConfigSource', 'make_config_source',
    'from_spec',
    # Note: Private functions (_load_config_from_dataframe, _write_series_blocks_yaml)
    # are available from io module but not exported as public API
]

