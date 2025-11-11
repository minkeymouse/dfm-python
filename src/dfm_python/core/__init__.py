"""Core DFM functionality: diagnostics, numeric operations, EM algorithm, and helpers."""

from .diagnostics import (
    calculate_rmse,
    _display_dfm_tables,
    diagnose_series,
    print_series_diagnosis,
)

__all__ = [
    'calculate_rmse',
    '_display_dfm_tables',
    'diagnose_series',
    'print_series_diagnosis',
]
