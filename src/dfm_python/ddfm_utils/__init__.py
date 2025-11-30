"""Utility functions for dfm_python.

This package currently provides:
- Conversion utilities for DDFM models (convert.py)
- Loss functions for DDFM training (loss.py)
"""

# Conversion utilities
from .convert import (
    convert_decoder_to_numpy,
    estimate_state_space_params,
    estimate_idiosyncratic_params,
    # Backward compatibility aliases
    get_transition_params,
    get_idio,
)

# Loss functions
from .loss import (
    mse_missing,
    convergence_checker,
    mse_missing_numpy,
)

__all__ = [
    # Conversion utilities
    'convert_decoder_to_numpy',
    'estimate_state_space_params',
    'estimate_idiosyncratic_params',
    # Backward compatibility aliases
    'get_transition_params',
    'get_idio',
    # Loss functions
    'mse_missing',
    'convergence_checker',
    'mse_missing_numpy',
]

