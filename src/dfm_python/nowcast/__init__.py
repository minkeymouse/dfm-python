"""Nowcasting, news decomposition, and backtesting for factor models."""

from .nowcast import Nowcast, NowcastResult
from .utils import (
    get_higher_frequency,
    calculate_backward_date,
    get_forecast_horizon_config,
    check_config_consistency,
    extract_news_summary,
)
from .helpers import NewsDecompResult, para_const, BacktestResult
from .dataview import DataView

__all__ = [
    # Main classes
    'Nowcast',
    'NowcastResult',
    'NewsDecompResult',
    'BacktestResult',
    # Core functions
    'para_const',
    # Utilities
    'get_higher_frequency',
    'calculate_backward_date',
    'get_forecast_horizon_config',
    'check_config_consistency',
    'extract_news_summary',
]
