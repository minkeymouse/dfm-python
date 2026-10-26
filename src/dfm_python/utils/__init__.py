"""Utility functions for dfm-python.

Modules: ``errors`` (exception classes), ``misc`` (parameter and config
helpers, scaler lookup), ``metric`` (RMSE, MAE, MAPE, R2), ``helper``
(linear-algebra fallback, interpolation), ``validation`` (condition checks),
``optional`` (the ``deep`` extra check), ``run_control`` (the training time
budget and per-iteration callback), and the torch-only ``checkpoint``
(input-dimension inference), ``loss`` (masked and ELBO losses) and ``seed``
(global RNG seeding). Names from ``numeric``, ``config`` and ``dataset`` are
re-exported here for backward compatibility.
"""

from ..numeric.estimator import (
    estimate_var,
    estimate_idio_dynamics,
)
from ..dataset.time import TimeIndex
from .metric import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
)
from ..config import (
    get_periods_per_year,
    validate_frequency,
)
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    MAX_TENT_SIZE,
    PERIODS_PER_YEAR,
    DEFAULT_BLOCK_NAME,
)
from ..models.dfm.tent import generate_tent_weights, generate_R_mat, get_agg_structure, group_by_freq
from .misc import (
    _check_sklearn,
    get_target_scaler,
    select_columns_by_prefix,
    resolve_param,
    resolve_named_param,
    get_config_attr,
    get_clock_frequency,
)
from ..numeric.validator import (
    validate_data_shape,
    validate_no_nan_inf,
    validate_horizon,
    validate_update_data_shape,
    validate_ndarray_ndim,
    validate_parameters_initialized,
    validate_dfm_initialization,
)
from .errors import (
    DFMError,
    ModelNotInitializedError,
    ModelNotTrainedError,
    NotFittedError,
    ConfigurationError,
    DataError,
    NumericalError,
    PredictionError,
    DataValidationError,
    UnsupportedInputError,
    NumericalStabilityError,
    ConfigValidationError,
)
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

__all__ = [
    'estimate_var',
    'estimate_idio_dynamics',
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
    'TimeIndex',
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
    'resolve_param',
    'resolve_named_param',
    'get_config_attr',
    'get_clock_frequency',
    '_check_sklearn',
    'get_target_scaler',
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_horizon',
    'validate_update_data_shape',
    'validate_ndarray_ndim',
    'validate_parameters_initialized',
    'validate_dfm_initialization',
    'DFMError',
    'ModelNotInitializedError',
    'ModelNotTrainedError',
    'NotFittedError',
    'ConfigurationError',
    'DataError',
    'NumericalError',
    'PredictionError',
    'DataValidationError',
    'UnsupportedInputError',
    'NumericalStabilityError',
    'ConfigValidationError',
    'create_scaled_identity',
    'ensure_symmetric',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_var_safe',
    'compute_cov_safe',
    'convergence_checker',
    'solve_regularized_ols',
    'select_columns_by_prefix',
]
