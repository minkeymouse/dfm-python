"""DFM model package.

This package contains the DFM model and its submodules:
- dfm: Main DFM class
- build: Model building functions
- initialization: Parameter initialization functions
- mixed_freq: Mixed-frequency parameter setup
- data: Data standardization and validation utilities
"""

from .dfm import DFM
from ...logger.dfm_logger import log_blocks_diagnostics
from ...numeric.builder import build_dfm_structure
from .initialization import (
    impute_for_init,
    initialize_clock_freq_idio,
    initialize_block_factors,
    add_idiosyncratic_observation_matrix,
    initialize_observation_noise,
    initialize_parameters,
)
from .mixed_freq import (
    setup_mixed_frequency_params,
    find_slower_frequency,
)

__all__ = [
    'DFM',
    'build_dfm_structure',
    'log_blocks_diagnostics',
    'impute_for_init',
    'initialize_clock_freq_idio',
    'initialize_block_factors',
    'add_idiosyncratic_observation_matrix',
    'initialize_observation_noise',
    'initialize_parameters',
    'setup_mixed_frequency_params',
    'find_slower_frequency',
]
