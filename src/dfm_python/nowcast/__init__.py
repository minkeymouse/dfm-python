"""Nowcasting, news decomposition, and backtesting for factor models."""

from .nowcast import Nowcast, NowcastResult
from .utils import (
    get_higher_frequency,
    calc_backward_date,
    get_forecast_horizon,
    check_config,
    extract_news,
)
from .helpers import NewsDecompResult, para_const, BacktestResult
from .dataview import DataView

# Sktime integration
from .splitters import NowcastingSplitter, NowcastForecaster
from .transformers import PublicationLagMasker, NewsDecompositionTransformer

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
    'calc_backward_date',
    'get_forecast_horizon',
    'check_config',
    'extract_news',
    # Sktime integration (optional)
    'NowcastingSplitter',
    'NowcastForecaster',
    'PublicationLagMasker',
    'NewsDecompositionTransformer',
]
