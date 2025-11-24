"""Engine layer: pure mathematical/algorithmic code shared by all factor models.

This package contains low-level engine code that is model-agnostic:
- EM algorithm implementation
- Numerical utilities and stability helpers
- Kalman filter and smoother
- Time series utilities
- Diagnostics and evaluation
- Synthetic data generation for testing
"""

# Re-export key engine components for convenience
from .em import (
    init_conditions,
    em_step,
    em_converged,
    EMStepParams,
    NaNHandlingOptions,
)
from .numeric import (
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
from .kalman import (
    run_kf,
    skf,
    fis,
    miss_data,
    KalmanFilterState,
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
from .utils import (
    group_series_by_frequency,
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
    get_periods_per_year,
    compute_idio_chain_lengths,
)
from .diagnostics import (
    calculate_rmse,
    _display_dfm_tables,
    diagnose_series,
    print_series_diagnosis,
)

__all__ = [
    # EM algorithm
    'init_conditions',
    'em_step',
    'em_converged',
    'EMStepParams',
    'NaNHandlingOptions',
    # Numeric utilities
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
    # Kalman filter
    'run_kf',
    'skf',
    'fis',
    'miss_data',
    'KalmanFilterState',
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
    # Utils
    'group_series_by_frequency',
    'get_aggregation_structure',
    'FREQUENCY_HIERARCHY',
    'get_periods_per_year',
    'compute_idio_chain_lengths',
    # Diagnostics
    'calculate_rmse',
    '_display_dfm_tables',
    'diagnose_series',
    'print_series_diagnosis',
]

# Synthetic DGP for testing (optional import)
try:
    from .synthetic_dgp import SyntheticDGP
    __all__.append('SyntheticDGP')
except ImportError:
    pass

