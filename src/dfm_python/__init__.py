"""Core nowcasting modules for DFM estimation and forecasting."""

from .config import DFMConfig, AppConfig
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
    'DFMConfig', 'AppConfig',
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

