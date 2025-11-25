"""Dynamic Factor Model (DFM) module (backward compatibility re-export).

This module re-exports DFM components for backward compatibility.
New code should use dfm_python.estimation.* and dfm_python.results.* directly.
"""

# Re-export from new modules for backward compatibility
from ..core.estimation import (
    _dfm_core,
    _prepare_data_and_params,
    _prepare_aggregation_structure,
    _run_em_algorithm,
)
from ..core.results import (
    DFMResult,
    DFMParams,
    EMAlgorithmParams,
)
# DFMCore is now in models.dfm
from ..models.dfm import DFMCore

__all__ = [
    'DFMCore',
    'DFMResult',
    'DFMParams',
    'EMAlgorithmParams',
    '_dfm_core',
    '_prepare_data_and_params',
    '_prepare_aggregation_structure',
    '_run_em_algorithm',
]
