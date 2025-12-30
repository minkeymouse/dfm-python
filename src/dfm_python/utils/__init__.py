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
try:
    from ..numeric.builder import (
        build_observation_matrix,
        build_state_space,
    )
    from ..numeric.estimator import (
        estimate_var1,
        estimate_var2,
        estimate_idio_dynamics,
        estimate_state_space_params,
    )
except ImportError:
    # If import fails, set to None (should not happen in normal usage)
    estimate_var1 = None
    estimate_var2 = None
    estimate_idio_dynamics = None
    build_observation_matrix = None
    build_state_space = None
    estimate_state_space_params = None

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
    validate_transformation,
    get_tent_weights,
    get_agg_structure,
    group_by_freq,
    compute_idio_lengths,
    parse_series_list,
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
# Import lazily to avoid circular imports
try:
    from ..numeric.tent import generate_tent_weights, generate_R_mat
except ImportError:
    generate_tent_weights = None
    generate_R_mat = None

# Block parsing utilities (moved to functional/dfm_block.py)
# Import lazily to avoid circular imports - these are only used in config parsing
try:
    from ..functional.dfm_block import parse_blocks_dict, infer_blocks
except ImportError:
    parse_blocks_dict = None
    infer_blocks = None

# Scaling utilities (from misc)
# Note: _get_mean and _get_scale are internal utilities, not re-exported
from .misc import (
    _check_sklearn,
    _get_scaler,
)

# DFM utilities - sort_data moved to dataset.data_utils, rem_nans_spline in numeric.missing
# Import lazily to avoid circular imports - DO NOT import at module level
# rem_nans_spline should be imported directly from numeric.missing when needed
# This avoids circular dependency: utils -> config -> schema -> utils.errors -> utils
# Setting to None here to maintain __all__ compatibility
def _get_rem_nans_spline():
    """Lazy import of rem_nans_spline to avoid circular dependencies."""
    try:
        from ..numeric.missing import rem_nans_spline
        return rem_nans_spline
    except ImportError:
        return None

rem_nans_spline = None  # Will be set on-demand via _get_rem_nans_spline() if needed

# Validation utilities (moved to numeric.validator)
try:
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
except ImportError:
    # Validation may not be available in all versions
    validate_ar_order = None
    validate_ma_order = None
    validate_learning_rate = None
    validate_batch_size = None
    validate_data_shape = None
    validate_no_nan_inf = None
    validate_eigenvalue_bounds = None
    validate_matrix_condition = None
    validate_horizon = None
    validate_irf_horizon = None

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

# Common utilities (new in Iteration 11)
try:
    from .common import (
        ensure_tensor,
        ensure_numpy,
        validate_matrix_shape,
        log_tensor_stats,
    )
except ImportError:
    # Common utilities may not be available in all versions
    ensure_tensor = None
    ensure_numpy = None
    validate_matrix_shape = None
    log_tensor_stats = None

# Analytics utilities (moved from common.py and model_helpers.py)
try:
    from ..numeric.analytic import (
        safe_matrix_power,
        extract_matrix_block,
        compute_forecast_metrics,
    )
except ImportError:
    # Analytics may not be available in all versions
    safe_matrix_power = None
    extract_matrix_block = None
    compute_forecast_metrics = None

# Model validation utilities (moved to numeric.validator)
try:
    from ..numeric.validator import (
        validate_companion_stability,
        validate_companion_matrix,  # Alias for backward compatibility
        validate_model_initialized,
        validate_prediction_inputs,
        validate_model_components,
        validate_forecast_inputs,
        validate_result_structure,
        validate_parameter_shapes,
    )
except ImportError:
    # Model validation may not be available in all versions
    validate_companion_stability = None
    validate_companion_matrix = None
    validate_model_initialized = None
    validate_prediction_inputs = None
    validate_model_components = None
    validate_forecast_inputs = None
    validate_result_structure = None
    validate_parameter_shapes = None

# Common validation utilities removed - not used anywhere

# Helper utilities (from misc)
from .misc import (
    resolve_param,
    check_finite_array,
    get_clock_frequency,
    DFMError,
    # Removed exception aliases - use DFMError or proper exceptions from errors.py
)

# Model helper utilities removed - functions inlined into models or moved to numeric/functional

# Tensor conversion utilities
# Note: tensor_to_numpy, numpy_to_tensor, ensure_tensor_on_device removed - use ensure_numpy/ensure_tensor from common instead
try:
    from .tensor_utils import (
        extract_tensor_value,
        normalize_tensor_shape,
        validate_tensor_device,
        batch_tensor_operation,
    )
except ImportError:
    # Tensor utilities may not be available in all versions
    extract_tensor_value = None
    normalize_tensor_shape = None
    validate_tensor_device = None
    batch_tensor_operation = None


# Autoencoder functions are now in encoder.autoencoder
from ..encoder.autoencoder import (
    extract_decoder_params,
    convert_decoder_to_numpy,
)

__all__ = [
    # Autoencoder functions (from encoder.autoencoder)
    'extract_decoder_params',
    'convert_decoder_to_numpy',
    # State-space utilities
    'estimate_var1',
    'estimate_var2',
    'estimate_idio_dynamics',
    'build_observation_matrix',
    'build_state_space',
    'estimate_state_space_params',
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
    'validate_transformation',
    'generate_tent_weights',
    'generate_R_mat',
    'get_tent_weights',
    'get_agg_structure',
    'group_by_freq',
    'compute_idio_lengths',
    'parse_series_list',
    'parse_blocks_dict',
    'infer_blocks',
    'detect_config_type',
    # Helper utilities
    'resolve_param',
    'check_finite_array',
    'get_clock_frequency',
    # Exception classes
    'DFMError',
    # Removed exception aliases - use DFMError or proper exceptions from errors.py
    # Scaling utilities
    '_check_sklearn',
    '_get_scaler',
    # DFM utilities
    'rem_nans_spline',
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
    # Model validation utilities (from numeric.validator)
    'validate_companion_stability',
    'validate_companion_matrix',
    'validate_model_initialized',
    'validate_prediction_inputs',
    'validate_model_components',
    'validate_forecast_inputs',
    'validate_result_structure',
    'validate_parameter_shapes',
    # Tensor conversion utilities (new in Iteration 13)
    # Removed: tensor_to_numpy, numpy_to_tensor, ensure_tensor_on_device - use ensure_numpy/ensure_tensor from common
    'extract_tensor_value',
    'normalize_tensor_shape',
    'validate_tensor_device',
    'batch_tensor_operation',
]
