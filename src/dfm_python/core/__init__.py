"""Core layer: pure mathematical/algorithmic code shared by all factor models.

This package contains low-level core code that is model-agnostic:
- EM algorithm implementation
- State-space models (Kalman filter, smoother, numerical utilities)
- Structure utilities (frequency hierarchy, tent kernels, aggregation)
- Time series utilities
- Diagnostics and evaluation
"""

# Re-export key core components for convenience
from .em import (
    init_conditions,
    em_step,
    em_converged,
    EMStepParams,
    NaNHandlingOptions,
)
from .state_space import (
    run_kf,
    skf,
    fis,
    miss_data,
    KalmanFilterState,
    # Note: Private numeric utilities (_ensure_symmetric, etc.) are available
    # from state_space module but not exported as public API
)
from .structure import (
    group_series_by_frequency,
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
    get_periods_per_year,
    compute_idio_chain_lengths,
    get_tent_weights_for_pair,
    generate_tent_weights,
    generate_R_mat,
)
from .time import (
    TimeIndex,
    datetime_range,
    parse_timestamp,
    get_next_period_end,
    clock_to_datetime_freq,
    get_latest_time,
    convert_to_timestamp,
    find_time_index,
    extract_last_date,
    to_python_datetime,
    days_in_month,
    parse_period_string,
)
from .diagnostics import (
    calculate_rmse,
    diagnose_series,
    print_series_diagnosis,
    # Note: _display_dfm_tables is internal and not exported as public API
)
from .results import (
    DFMResult,
    DFMParams,
    EMAlgorithmParams,
)
# Note: Internal estimation functions (_dfm_core, _prepare_*, _run_em_algorithm)
# are available from estimation module but not exported as public API

__all__ = [
    # EM algorithm
    'init_conditions',
    'em_step',
    'em_converged',
    'EMStepParams',
    'NaNHandlingOptions',
    # State-space (Kalman filter)
    'run_kf',
    'skf',
    'fis',
    'miss_data',
    'KalmanFilterState',
    # Structure utilities
    'group_series_by_frequency',
    'get_aggregation_structure',
    'FREQUENCY_HIERARCHY',
    'get_periods_per_year',
    'compute_idio_chain_lengths',
    'get_tent_weights_for_pair',
    'generate_tent_weights',
    'generate_R_mat',
    # Time utilities
    'TimeIndex',
    'datetime_range',
    'parse_timestamp',
    'get_next_period_end',
    'clock_to_datetime_freq',
    'get_latest_time',
    'convert_to_timestamp',
    'find_time_index',
    'extract_last_date',
    'to_python_datetime',
    'days_in_month',
    'parse_period_string',
    # Diagnostics
    'calculate_rmse',
    'diagnose_series',
    'print_series_diagnosis',
    # Results
    'DFMResult',
    'DFMParams',
    'EMAlgorithmParams',
    # Note: Internal functions (_ensure_symmetric, _compute_principal_components,
    # _dfm_core, _prepare_*, _run_em_algorithm, _display_dfm_tables, etc.) are
    # available from their respective modules but not exported as public API
]

