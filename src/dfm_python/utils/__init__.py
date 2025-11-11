"""Utility functions for data preprocessing and summary statistics."""

from .aggregation import (
    generate_tent_weights,
    generate_R_mat,
    get_tent_weights_for_pair,
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
    TENT_WEIGHTS_LOOKUP,
    MAX_TENT_SIZE,
)

# Lazy imports to avoid circular dependency with data module
def _get_rem_nans_spline():
    """Lazy import of rem_nans_spline to avoid circular dependency."""
    from ..data import rem_nans_spline
    return rem_nans_spline

def _get_summarize():
    """Lazy import of summarize to avoid circular dependency."""
    from ..data import summarize
    return summarize

# Re-export via __getattr__ for lazy loading
def __getattr__(name: str):
    if name == 'rem_nans_spline':
        return _get_rem_nans_spline()
    elif name == 'summarize':
        return _get_summarize()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'rem_nans_spline',
    'summarize',
    'generate_tent_weights',
    'generate_R_mat',
    'get_tent_weights_for_pair',
    'get_aggregation_structure',
    'FREQUENCY_HIERARCHY',
    'TENT_WEIGHTS_LOOKUP',
    'MAX_TENT_SIZE',
]

