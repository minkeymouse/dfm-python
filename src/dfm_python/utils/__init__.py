"""Backward compatibility shims and utility functions for dfm_python.

This package contains:
- Backward compatibility shims for legacy imports (dfm_python.dfm)
- Conversion utilities for DDFM models (convert.py)
- Loss functions for DDFM training (loss.py)
"""

# Backward compatibility: re-export from dfm shim
from .dfm import (
    DFMCore,
    DFMResult,
    DFMParams,
    EMAlgorithmParams,
    _dfm_core,
    _prepare_data_and_params,
    _prepare_aggregation_structure,
    _run_em_algorithm,
)

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
    # Backward compatibility for dfm_python.dfm
    'DFMCore',
    'DFMResult',
    'DFMParams',
    'EMAlgorithmParams',
    '_dfm_core',
    '_prepare_data_and_params',
    '_prepare_aggregation_structure',
    '_run_em_algorithm',
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

