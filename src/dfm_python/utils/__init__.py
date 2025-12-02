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
    _safe_determinant,
    _compute_principal_components,  # From encoder.pca (backward compatibility)
    _estimate_ar_coefficient,
    _clip_ar_coefficients,
    _compute_covariance_safe,
    _compute_variance_safe,
    _apply_ar_clipping,
    _compute_regularization_param,
    _cap_max_eigenvalue,
    _ensure_innovation_variance_minimum,
    _safe_divide,
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

# These are in config.structure, not utils.time
from ..config.structure import get_periods_per_year, FREQUENCY_HIERARCHY

# Backward compatibility: Re-export nowcast utilities from nowcast package
from ..nowcast.helpers import (
    NewsDecompResult,
    BacktestResult,
    para_const,
    _get_higher_frequency,
    _calculate_backward_date,
    _get_forecast_horizon_config,
    _check_config_consistency,
    _transform_series,
    transform_data,
    _extract_news_summary_impl,
)

from ..nowcast.dataview import DataView

# Data loading utilities (from core.loader)
from .data import (
    read_data,
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
    safe_mean_std,
    get_clock_frequency,
    get_series_ids,
    get_series_names,
    get_frequencies_from_config,
    get_units_from_config,
    find_series_index,
    get_series_id_by_index,
    ParameterResolver,
    _validate_config_loaded,
    _validate_data_loaded,
    _validate_result_loaded,
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
    '_safe_determinant',
    '_compute_principal_components',
    '_estimate_ar_coefficient',
    '_clip_ar_coefficients',
    '_compute_covariance_safe',
    '_compute_variance_safe',
    '_apply_ar_clipping',
    '_compute_regularization_param',
    '_cap_max_eigenvalue',
    '_ensure_innovation_variance_minimum',
    '_safe_divide',
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
    # Nowcasting utilities (backward compatibility)
    'NewsDecompResult',
    'BacktestResult',
    'para_const',
    '_get_higher_frequency',
    '_calculate_backward_date',
    '_get_forecast_horizon_config',
    '_check_config_consistency',
    '_transform_series',
    'transform_data',
    '_extract_news_summary_impl',
    # DataView
    'DataView',
    # Data loading utilities (from core.loader)
    'read_data',
    'sort_data',
    'rem_nans_spline',
    'calculate_release_date',
    'create_data_view',
    # Helper utilities (from core.helpers)
    'safe_get_attr',
    'safe_get_method',
    'resolve_param',
    'safe_mean_std',
    'get_clock_frequency',
    'get_series_ids',
    'get_series_names',
    'get_frequencies_from_config',
    'get_units_from_config',
    'find_series_index',
    'get_series_id_by_index',
    'ParameterResolver',
    '_validate_config_loaded',
    '_validate_data_loaded',
    '_validate_result_loaded',
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
