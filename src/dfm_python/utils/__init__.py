"""Utility functions for dfm-python.

This package contains utility functions organized in modules:
- errors.py: Custom exception classes
- common.py: Common tensor/numpy utilities
- misc.py: Miscellaneous utilities (helpers, diagnostics, scaling)
- metric.py: Metric calculation utilities (RMSE, MAE, MAPE, R2)

Validation functions are in numeric.validator.
Helper functions are inlined into models or moved to numeric/functional.
"""

# State-space utilities (imported from numeric.builder)
# Note: build_observation_matrix and build_state_space are DDFM methods, not standalone functions
from ..numeric.estimator import (
    estimate_var,
    estimate_idio_dynamics,
)

# Time utilities
from ..dataset.time import TimeIndex

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
    get_agg_structure,
    group_by_freq,
    detect_config_type,
)

# Import constants from config.constants
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    MAX_TENT_SIZE,
    PERIODS_PER_YEAR,
    DEFAULT_BLOCK_NAME,
)

# Tent kernel matrix functions (from models/dfm module)
from ..models.dfm.tent import generate_tent_weights, generate_R_mat

# Scaling utilities (from misc)
from .misc import (
    _check_sklearn,
    get_target_scaler,
    select_columns_by_prefix,
)

# Validation utilities (from numeric.validator)
from ..numeric.validator import (
    validate_data_shape,
    validate_no_nan_inf,
    validate_horizon,
    validate_update_data_shape,
    validate_ndarray_ndim,
    validate_parameters_initialized,
    validate_dfm_initialization,
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

# Common utilities (moved from common.py - now using direct numpy/torch operations)
# Note: ensure_numpy and sanitize_array removed - use np.asarray() or .cpu().numpy() directly

# Preprocessing utilities (removed - module doesn't exist)

# Analytics utilities (from numeric.stability)
from ..numeric.stability import (
    create_scaled_identity,
    ensure_symmetric,
    cap_max_eigenval,
    ensure_positive_definite,
    ensure_covariance_stable,
    compute_var_safe,
    compute_cov_safe,
    convergence_checker,
    solve_regularized_ols,
)

# Model validation utilities (from numeric.validator)
# Note: Most model validation functions were removed - only basic validators remain

# Helper utilities (from misc)
from .misc import (
    resolve_param,
    resolve_named_param,
    get_config_attr,
    get_clock_frequency,
)

# Autoencoder functions should be imported directly from encoder.simple_autoencoder
# Removed re-exports to avoid confusion - import from encoder module instead

__all__ = [
    # State-space utilities
    'estimate_var',
    'estimate_idio_dynamics',
    # Time utilities (includes metrics)
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
    'TimeIndex',
    # Config utilities (frequency and parsing)
    'get_periods_per_year',
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'MAX_TENT_SIZE',
    'DEFAULT_BLOCK_NAME',
    'validate_frequency',
    'generate_tent_weights',
    'generate_R_mat',
    'get_agg_structure',
    'group_by_freq',
    'detect_config_type',
    # Helper utilities
    'resolve_param',
    'resolve_named_param',
    'get_config_attr',
    'get_clock_frequency',
    # Scaling utilities
    '_check_sklearn',
    'get_target_scaler',
    # Validation utilities (from numeric.validator)
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_horizon',
    'validate_update_data_shape',
    'validate_ndarray_ndim',
    'validate_parameters_initialized',
    'validate_dfm_initialization',
    # Exception classes (from errors.py)
    'DFMError',  # Base exception class
    'ModelNotInitializedError',
    'ModelNotTrainedError',
    'ConfigurationError',
    'DataError',
    'NumericalError',
    'PredictionError',
    'DataValidationError',
    'NumericalStabilityError',
    'ConfigValidationError',
    # Stability utilities (from numeric.stability)
    'create_scaled_identity',
    'ensure_symmetric',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_var_safe',
    'compute_cov_safe',
    'convergence_checker',
    'solve_regularized_ols',
    # Misc utilities
    'select_columns_by_prefix',  # From misc
]
