"""DDFM-specific utility functions.

This package provides utilities specific to Deep Dynamic Factor Models (DDFM):
- Conversion utilities: Convert PyTorch models to NumPy state-space parameters (convert.py)
- Loss functions: Missing-aware loss functions for DDFM training (loss.py)

Note: These utilities are specific to DDFM. For general DFM utilities, see:
- core/ for core DFM algorithms (EM, Kalman filter, etc.)
- dataloader/ for data loading and preprocessing
- config/ for configuration management
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

