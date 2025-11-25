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
    # Numeric utilities (re-exported from state_space)
    _ensure_symmetric,
    _compute_principal_components,
    _clean_matrix,
    _ensure_positive_definite,
    _compute_regularization_param,
    _apply_ar_clipping,
    _cap_max_eigenvalue,
    _estimate_ar_coefficient,
    _safe_divide,
    _check_finite,
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
    _display_dfm_tables,
    diagnose_series,
    print_series_diagnosis,
)
from .results import (
    DFMResult,
    DFMParams,
    EMAlgorithmParams,
)
from .estimation import (
    _dfm_core,
    _prepare_data_and_params,
    _prepare_aggregation_structure,
    _run_em_algorithm,
)

__all__ = [
    # EM algorithm
    'init_conditions',
    'em_step',
    'em_converged',
    'EMStepParams',
    'NaNHandlingOptions',
    # State-space (Kalman filter + numeric utilities)
    'run_kf',
    'skf',
    'fis',
    'miss_data',
    'KalmanFilterState',
    '_ensure_symmetric',
    '_compute_principal_components',
    '_clean_matrix',
    '_ensure_positive_definite',
    '_compute_regularization_param',
    '_apply_ar_clipping',
    '_cap_max_eigenvalue',
    '_estimate_ar_coefficient',
    '_safe_divide',
    '_check_finite',
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
    '_display_dfm_tables',
    'diagnose_series',
    'print_series_diagnosis',
    # Results
    'DFMResult',
    'DFMParams',
    'EMAlgorithmParams',
    # Estimation functions
    '_dfm_core',
    '_prepare_data_and_params',
    '_prepare_aggregation_structure',
    '_run_em_algorithm',
]

