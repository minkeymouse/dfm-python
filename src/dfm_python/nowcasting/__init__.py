"""Nowcasting and news decomposition for factor models.

This package provides nowcasting functionality and news decomposition framework
for understanding how new data releases affect nowcasts.
"""

from .nowcast import (
    Nowcast,
    para_const,
    NowcastResult,
    NewsDecompResult,
    BacktestResult,
)

__all__ = [
    'Nowcast',
    'para_const',
    'NowcastResult',
    'NewsDecompResult',
    'BacktestResult',
]

