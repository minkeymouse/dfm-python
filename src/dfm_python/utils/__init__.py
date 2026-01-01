"""Utility functions for dfm-python.

This package contains utility functions organized in modules:
- errors.py: Custom exception classes
- common.py: Common tensor/numpy utilities
- misc.py: Miscellaneous utilities (helpers, diagnostics, scaling)
- metric.py: Metric calculation utilities (RMSE, MAE, MAPE, R2)
- tensor_utils.py: Tensor conversion and manipulation utilities

Validation functions are in numeric.validator.
Helper functions are inlined into models or moved to numeric/functional.
"""

# State-space utilities (imported from numeric.builder)
from ..numeric.builder import (
    build_observation_matrix,
    build_state_space,
)
from ..numeric.estimator import (
    estimate_var,
    estimate_idio_dynamics,
)

# Time utilities
from ..dataset.process import TimeIndex, parse_timestamp

# Metric utilities
from .metric import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
)

# Config utilities - re-exported from their new locations
from ..config import (
    get_periods_per_year,
    validate_frequency,
    get_tent_weights,
    get_agg_structure,
    group_by_freq,
    compute_idio_lengths,
    detect_config_type,
)

# Import constants from config.constants
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    TENT_WEIGHTS_LOOKUP,
    MAX_TENT_SIZE,
    PERIODS_PER_YEAR,
    DEFAULT_BLOCK_NAME,
)

# Tent kernel matrix functions (from numeric module)
from ..numeric.tent import generate_tent_weights, generate_R_mat

# Scaling utilities (from misc)
from .misc import (
    _check_sklearn,
    _get_scaler,
)

# Validation utilities (from numeric.validator)
from ..numeric.validator import (
    validate_ar_order,
    validate_ma_order,
    validate_learning_rate,
    validate_batch_size,
    validate_data_shape,
    validate_no_nan_inf,
    validate_eigenvalue_bounds,
    validate_matrix_condition,
    validate_horizon,
    validate_irf_horizon,
)

# Exception classes (from errors.py)
from .errors import (
    DFMError,
    ModelNotInitializedError,
    ModelNotTrainedError,
    ConfigurationError,
    DataError,
    NumericalError,
    PredictionError,
    DataValidationError,
    NumericalStabilityError,
    ConfigValidationError,
)

# Common utilities
from .common import (
    ensure_tensor,
    ensure_numpy,
    validate_matrix_shape,
    log_tensor_stats,
    select_columns_by_prefix,
)

# Analytics utilities (from numeric.stability)
from ..numeric.stability import (
    safe_matrix_power,
    extract_matrix_block,
    compute_forecast_metrics,
)

# Model validation utilities (from numeric.validator)
from ..numeric.validator import (
    validate_companion_stability,
    validate_companion_matrix,
    validate_model_initialized,
    validate_prediction_inputs,
    validate_model_components,
    validate_forecast_inputs,
    validate_result_structure,
    validate_parameter_shapes,
)

# Helper utilities (from misc)
from .misc import (
    resolve_param,
    get_clock_frequency,
)

# Tensor conversion utilities
from .tensor_utils import (
    extract_tensor_value,
    normalize_tensor_shape,
    validate_tensor_device,
    batch_tensor_operation,
)


# Autoencoder functions are now in encoder.simple_encoder
from ..encoder.simple_encoder import (
    extract_decoder_params,
    convert_decoder_to_numpy,
)

__all__ = [
    # Autoencoder functions (from encoder.simple_encoder)
    'extract_decoder_params',
    'convert_decoder_to_numpy',
    # State-space utilities
    'estimate_var',
    'estimate_idio_dynamics',
    'build_observation_matrix',
    'build_state_space',
    # Time utilities (includes metrics)
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
    'TimeIndex',
    'parse_timestamp',
    # Config utilities (frequency and parsing)
    'get_periods_per_year',
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'TENT_WEIGHTS_LOOKUP',
    'MAX_TENT_SIZE',
    'DEFAULT_BLOCK_NAME',
    'validate_frequency',
    'generate_tent_weights',
    'generate_R_mat',
    'get_tent_weights',
    'get_agg_structure',
    'group_by_freq',
    'compute_idio_lengths',
    'detect_config_type',
    # Helper utilities
    'resolve_param',
    'get_clock_frequency',
    # Scaling utilities
    '_check_sklearn',
    '_get_scaler',
    # Validation utilities (from numeric.validator)
    'validate_ar_order',
    'validate_ma_order',
    'validate_learning_rate',
    'validate_batch_size',
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_eigenvalue_bounds',
    'validate_matrix_condition',
    'validate_horizon',
    'validate_irf_horizon',
    # Exception classes (from errors.py)
    'ModelNotInitializedError',
    'ModelNotTrainedError',
    'ConfigurationError',
    'DataError',
    'NumericalError',
    'PredictionError',
    'DataValidationError',
    'NumericalStabilityError',
    'ConfigValidationError',
    # Common utilities (new in Iteration 11)
    'ensure_tensor',
    'ensure_numpy',
    'safe_matrix_power',
    'extract_matrix_block',
    'validate_matrix_shape',
    'log_tensor_stats',
    'select_columns_by_prefix',
    # Model validation utilities (from numeric.validator)
    'validate_companion_stability',
    'validate_companion_matrix',
    'validate_model_initialized',
    'validate_prediction_inputs',
    'validate_model_components',
    'validate_forecast_inputs',
    'validate_result_structure',
    'validate_parameter_shapes',
    # Tensor conversion utilities
    'extract_tensor_value',
    'normalize_tensor_shape',
    'validate_tensor_device',
    'batch_tensor_operation',
]
