"""Core DFM functionality (backward compatibility layer).

This module re-exports engine components for backward compatibility.
New code should use dfm_python.engine.* directly.
"""

# Re-export from engine for backward compatibility
from ..engine import (
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

# Synthetic DGP for testing (optional import)
try:
    from ..engine.synthetic_dgp import SyntheticDGP
    __all__.append('SyntheticDGP')
except ImportError:
    pass
