"""Utility functions for dfm-python.

This package contains utility functions organized in modules:
- data.py: Data loading and transformation utilities
- helpers.py: Helper functions for configuration, validation, and data operations
- diagnostics.py: Diagnostic functions for model evaluation
- statespace.py: State-space utilities (numerical stability, AR estimation, DDFM utilities)
- time.py: Time utilities (datetime operations, frequency mapping) and metrics (RMSE, MAE, MAPE, R²)

Nowcasting utilities are in the nowcast/ package.
"""

from .statespace import (
    estimate_var1,
    estimate_var2,
    estimate_idiosyncratic_dynamics,
    build_observation_matrix,
    build_state_space,
    estimate_state_space_params,
    # Private functions are not exported - they're internal utilities
)

from .time import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
    TimeIndex,
    parse_timestamp,
    datetime_range,
    days_in_month,
    clock_to_datetime_freq,
    get_next_period_end,
    find_time_index,
    parse_period_string,
    get_latest_time,
    convert_to_timestamp,
    to_python_datetime,
)

# Frequency utilities from config.utils
from ..config.utils import get_periods_per_year, FREQUENCY_HIERARCHY

# Re-export nowcast utilities from nowcast package
from ..nowcast.helpers import (
    NewsDecompResult,
    BacktestResult,
    para_const,
)
from ..nowcast.utils import (
    get_higher_frequency,
    calculate_backward_date,
    get_forecast_horizon_config,
    check_config_consistency,
    extract_news_summary,
)

from ..nowcast.dataview import DataView

# Data loading utilities (from core.loader)
from .data import (
    sort_data,
    rem_nans_spline,
    calculate_release_date,
    create_data_view,
)

# Helper utilities (from core.helpers)
from .helpers import (
    safe_get_attr,
    safe_get_method,
    resolve_param,
    get_clock_frequency,
    get_series_ids,
    get_series_names,
    get_frequencies_from_config,
    find_series_index,
    get_series_id_by_index,
    ParameterResolver,
    DFMError,
    DFMConfigError,
    DFMDataError,
    DFMEstimationError,
    DFMValidationError,
    DFMImportError,
)

# Diagnostic utilities (from core.diagnostics)
from .diagnostics import (
    diagnose_series,
    print_series_diagnosis,
    evaluate_factor_estimation,
    evaluate_loading_estimation,
)

# VAE functions are now in encoder.vae
from ..encoder.vae import (
    extract_decoder_params,
    convert_decoder_to_numpy,
)

__all__ = [
    # VAE functions (from encoder.vae)
    'extract_decoder_params',
    'convert_decoder_to_numpy',
    # State-space utilities (includes DDFM utilities)
    'estimate_var1',
    'estimate_var2',
    'estimate_idiosyncratic_dynamics',
    'build_observation_matrix',
    'build_state_space',
    'estimate_state_space_params',
    # Private functions (_safe_determinant, etc.) are internal and not exported
    # Time utilities (includes metrics)
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
    'TimeIndex',
    'parse_timestamp',
    'datetime_range',
    'days_in_month',
    'clock_to_datetime_freq',
    'get_next_period_end',
    'find_time_index',
    'parse_period_string',
    'get_latest_time',
    'convert_to_timestamp',
    'to_python_datetime',
    'get_periods_per_year',
    'FREQUENCY_HIERARCHY',
    # Nowcasting utilities
    'NewsDecompResult',
    'BacktestResult',
    'para_const',
    'get_higher_frequency',
    'calculate_backward_date',
    'get_forecast_horizon_config',
    'check_config_consistency',
    'extract_news_summary',
    # DataView
    'DataView',
    # Data loading utilities (from core.loader)
    'sort_data',
    'rem_nans_spline',
    'calculate_release_date',
    'create_data_view',
    # Helper utilities (from core.helpers)
    'safe_get_attr',
    'safe_get_method',
    'resolve_param',
    'get_clock_frequency',
    'get_series_ids',
    'get_series_names',
    'get_frequencies_from_config',
    'find_series_index',
    'get_series_id_by_index',
    'ParameterResolver',
    # Note: Private validation functions (_validate_*) are internal and not exported
    'DFMError',
    'DFMConfigError',
    'DFMDataError',
    'DFMEstimationError',
    'DFMValidationError',
    'DFMImportError',
    # Diagnostic utilities (from core.diagnostics)
    'diagnose_series',
    'print_series_diagnosis',
    'evaluate_factor_estimation',
    'evaluate_loading_estimation',
]
