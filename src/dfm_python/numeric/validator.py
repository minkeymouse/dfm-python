"""Model validation utilities for comprehensive error checking.

This module provides validation utilities for model components, ensuring
consistent error handling and validation across all models (KDFM, DFM, DDFM).

Common validation patterns:
- Model initialization checks
- Component existence validation
- Parameter shape validation
- Numerical stability checks
- Companion matrix validation
- Forecast/prediction input validation
"""

from typing import Optional, Union, List, Tuple, Any, Dict
import numpy as np
import torch
from torch import Tensor

from ..utils.errors import (
    ModelNotInitializedError,
    ModelNotTrainedError,
    NumericalError,
    NumericalStabilityError,
    DataValidationError,
    PredictionError,
    ConfigurationError
)
from ..config.types import ArrayLike, to_numpy
from ..config.constants import DEFAULT_MIN_DELTA
from ..logger import get_logger

_logger = get_logger(__name__)


def validate_model_components(
    companion_ar: Optional[Any] = None,
    companion_ma: Optional[Any] = None,
    structural_id: Optional[Any] = None,
    model_name: str = "model"
) -> None:
    """Validate that required model components are initialized.
    
    This function checks that at least the AR companion matrix is initialized,
    which is required for all model operations. Optionally checks for structural
    identification component if provided.
    
    Parameters
    ----------
    companion_ar : object, optional
        AR companion SSM component
    companion_ma : object, optional
        MA companion SSM component (optional, only needed if ma_order > 0)
    structural_id : object, optional
        Structural identification SSM component (optional, only required for some models)
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    ModelNotInitializedError
        If companion_ar is None (required component)
    """
    if companion_ar is None:
        raise ModelNotInitializedError(
            f"{model_name} requires initialized AR companion matrix. "
            f"Call initialize_from_data() or train the model before using it.",
            details="companion_ar is None"
        )
    
    # Optional: check structural_id if provided (some models require it)
    if structural_id is None and hasattr(companion_ar, 'structural_id'):
        # Only warn if structural_id is expected but not found
        _logger.debug(
            f"{model_name} structural identification component not found. "
            f"This may be optional depending on the model."
        )


def validate_companion_stability(
    companion_matrix: Union[np.ndarray, Tensor],
    threshold: float = 1.0,
    warn_threshold: float = 0.99,
    model_name: str = "model",
    name: Optional[str] = None
) -> Tuple[bool, float]:
    """Validate companion matrix stability by checking eigenvalues.
    
    This function checks if a companion matrix is stable by computing its
    eigenvalues and verifying they are within the unit circle (magnitude < 1.0).
    This is critical for IRF computation and forecast generation.
    
    Parameters
    ----------
    companion_matrix : np.ndarray or Tensor
        Companion matrix to validate (shape: (n, n) or (..., n, n))
    threshold : float, default=1.0
        Maximum allowed eigenvalue magnitude (strictly < threshold for stability)
    warn_threshold : float, default=0.99
        Threshold for warning (magnitudes > this trigger warning)
    model_name : str, default="model"
        Name of model for error messages
    name : str, optional
        Name for error messages (alternative to model_name, for backward compatibility)
        
    Returns
    -------
    tuple
        (is_stable, max_eigenvalue) where:
        - is_stable: bool, True if all eigenvalues are within unit circle
        - max_eigenvalue: float, maximum eigenvalue magnitude
        
    Raises
    ------
    NumericalStabilityError
        If maximum eigenvalue magnitude >= threshold (model is unstable)
    NumericalError
        If matrix contains NaN/Inf or eigenvalue computation fails
        
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0.5, 0.1], [0.0, 0.8]])  # Stable matrix
    >>> is_stable, max_eig = validate_companion_stability(A)
    >>> assert is_stable is True
    >>> assert max_eig < 1.0
    """
    # Use name if provided, otherwise use model_name
    display_name = name if name is not None else f"{model_name} companion matrix"
    
    # Convert to numpy if needed
    matrix_np = to_numpy(companion_matrix)
    
    # Check for NaN/Inf
    if np.any(np.isnan(matrix_np)) or np.any(np.isinf(matrix_np)):
        raise NumericalError(
            f"{display_name} contains NaN/Inf values. Model may not be properly trained.",
            details=f"Matrix shape: {matrix_np.shape}, NaN count: {np.isnan(matrix_np).sum()}, Inf count: {np.isinf(matrix_np).sum()}"
        )
    
    # Handle multi-dimensional arrays (take last two dimensions)
    if matrix_np.ndim > 2:
        # For kernel dimension, take first kernel
        if matrix_np.ndim == 3:
            matrix_np = matrix_np[0]
        else:
            raise DataValidationError(
                f"{display_name} must be 2D or 3D (with kernel dimension), "
                f"got shape {matrix_np.shape}",
                details="Companion matrix validation requires 2D or 3D array (3D for batched validation)"
            )
    
    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(matrix_np)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise NumericalError(
            f"Cannot compute eigenvalues for {display_name}: {e}. "
            f"Matrix may be singular or have invalid shape.",
            details=f"Matrix shape: {matrix_np.shape}"
        ) from e
    
    # Compute maximum magnitude
    magnitudes = np.abs(eigenvalues)
    max_magnitude = float(np.max(magnitudes))
    
    # Check stability
    is_stable = max_magnitude < threshold
    
    if not is_stable:
        raise NumericalStabilityError(
            f"{display_name} is unstable: maximum eigenvalue magnitude "
            f"{max_magnitude:.6f} >= {threshold}. "
            f"Model may produce unreliable forecasts or IRFs. "
            f"Consider: (1) Regularization, (2) Differencing, (3) Lower lag order.",
            details=f"Max eigenvalue magnitude: {max_magnitude:.6f}, Threshold: {threshold}"
        )
    
    # Warn if near-unstable
    if max_magnitude > warn_threshold:
        _logger.warning(
            f"{display_name} is near-unstable: maximum eigenvalue magnitude "
            f"{max_magnitude:.6f} > {warn_threshold}. "
            f"Consider regularization or differencing."
        )
    
    return is_stable, max_magnitude


def validate_model_initialized(
    companion_ar: Optional[Any],
    companion_ma: Optional[Any] = None,
    structural_id: Optional[Any] = None,
    model_name: str = "model"
) -> None:
    """Validate that model components are initialized.
    
    Parameters
    ----------
    companion_ar : object, optional
        AR companion SSM (should not be None)
    companion_ma : object, optional
        MA companion SSM (can be None if q=0)
    structural_id : object, optional
        Structural identification SSM (should not be None for models that require it)
    model_name : str, default="model"
        Name of model for error messages
        
    Raises
    ------
    ModelNotInitializedError
        If required components are not initialized
    """
    if companion_ar is None:
        raise ModelNotInitializedError(
            f"{model_name} AR companion SSM is not initialized. "
            f"Call initialize_from_data() before using the model.",
            details="companion_ar is None"
        )
    
    if structural_id is None:
        raise ModelNotInitializedError(
            f"{model_name} structural identification SSM is not initialized. "
            f"Call initialize_from_data() before using the model.",
            details="structural_id is None"
        )


def validate_prediction_inputs(
    horizon: Optional[int],
    last_observation: Optional[Union[np.ndarray, Tensor]],
    expected_n_vars: int,
    model_name: str = "model"
) -> Tuple[int, Optional[np.ndarray]]:
    """Validate inputs for prediction method.
    
    This is the comprehensive version that validates and returns normalized inputs.
    For simpler validation that just raises errors, use validate_forecast_inputs.
    
    Parameters
    ----------
    horizon : int, optional
        Forecast horizon (must be > 0)
    last_observation : np.ndarray or Tensor, optional
        Last observation for initialization (shape: (K,) or (1, K))
    expected_n_vars : int
        Expected number of variables (K)
    model_name : str, default="model"
        Name of model for error messages
        
    Returns
    -------
    tuple
        (validated_horizon, validated_last_obs) where:
        - validated_horizon: int, validated horizon
        - validated_last_obs: np.ndarray or None, validated last observation
        
    Raises
    ------
    PredictionError
        If inputs are invalid
    """
    # Validate horizon inline
    if horizon is None:
        horizon = 1  # Default horizon
    if not isinstance(horizon, int):
        raise PredictionError(
            f"{model_name} prediction: horizon must be an integer, got {type(horizon).__name__}",
            details=f"horizon={horizon}"
        )
    if horizon < 1:
        raise PredictionError(
            f"{model_name} prediction: horizon must be >= 1, got {horizon}",
            details=f"horizon={horizon}"
        )
    if horizon > 100:
        _logger.warning(
            f"{model_name} prediction: horizon {horizon} is very large (> 100). "
            f"Forecast accuracy may degrade significantly."
        )
    validated_horizon = horizon
    
    # Validate last_observation if provided
    validated_last_obs = None
    if last_observation is not None:
        # Convert to numpy
        last_obs_np = to_numpy(last_observation)
        
        # Handle shape: (K,) or (1, K) -> (K,)
        if last_obs_np.ndim == 2:
            if last_obs_np.shape[0] == 1:
                last_obs_np = last_obs_np[0]
            else:
                raise PredictionError(
                    f"{model_name} prediction: last_observation must have shape (K,) or (1, K), "
                    f"got {last_observation.shape}",
                    details=f"Expected {expected_n_vars} variables"
                )
        
        # Validate number of variables
        if last_obs_np.shape[0] != expected_n_vars:
            raise PredictionError(
                f"{model_name} prediction: last_observation has {last_obs_np.shape[0]} variables, "
                f"expected {expected_n_vars}",
                details=f"Shape: {last_obs_np.shape}, Expected: ({expected_n_vars},)"
            )
        
        validated_last_obs = last_obs_np
    
    return validated_horizon, validated_last_obs


def validate_forecast_inputs(
    horizon: int,
    last_observation: Optional[Union[Tensor, np.ndarray]] = None,
    n_vars: Optional[int] = None,
    model_name: str = "model"
) -> None:
    """Validate forecast input parameters (simpler version that just validates).
    
    This is a simpler validation function that only checks and raises errors.
    For validation that also normalizes and returns inputs, use validate_prediction_inputs.
    
    Parameters
    ----------
    horizon : int
        Forecast horizon (must be >= 1)
    last_observation : Tensor or np.ndarray, optional
        Last observation for initialization (shape must match n_vars if provided)
    n_vars : int, optional
        Number of variables (required if last_observation is provided)
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    DataValidationError
        If horizon < 1 or last_observation shape doesn't match n_vars
    """
    if horizon < 1:
        raise DataValidationError(
            f"{model_name} forecast horizon must be >= 1, got {horizon}.",
            details=f"horizon={horizon}"
        )
    
    if last_observation is not None and n_vars is not None:
        # Normalize to numpy for shape checking
        from ..config.types import to_numpy
        last_obs_np = to_numpy(last_observation)
        
        # Check shape: should be (1, n_vars) or (n_vars,)
        if last_obs_np.ndim == 1:
            if last_obs_np.shape[0] != n_vars:
                raise DataValidationError(
                    f"{model_name} last_observation shape mismatch: expected ({n_vars},), got {last_obs_np.shape}.",
                    details=f"last_observation.shape={last_obs_np.shape}, n_vars={n_vars}"
                )
        elif last_obs_np.ndim == 2:
            if last_obs_np.shape[1] != n_vars or last_obs_np.shape[0] != 1:
                raise DataValidationError(
                    f"{model_name} last_observation shape mismatch: expected (1, {n_vars}), got {last_obs_np.shape}.",
                    details=f"last_observation.shape={last_obs_np.shape}, n_vars={n_vars}"
                )
        else:
            raise DataValidationError(
                f"{model_name} last_observation must be 1D or 2D, got {last_obs_np.ndim}D.",
                details=f"last_observation.shape={last_obs_np.shape}"
            )


def validate_result_structure(
    result: Any,
    required_fields: List[str],
    model_name: str = "model"
) -> None:
    """Validate that result object has required fields.
    
    Parameters
    ----------
    result : object
        Result object to validate
    required_fields : list of str
        List of required field names
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    ModelNotTrainedError
        If result is None or missing required fields
    """
    if result is None:
        raise ModelNotTrainedError(
            f"{model_name} result is None. Model must be trained before extracting results.",
            details="result is None"
        )
    
    missing_fields = []
    for field in required_fields:
        if not hasattr(result, field):
            missing_fields.append(field)
    
    if missing_fields:
        raise ModelNotTrainedError(
            f"{model_name} result is missing required fields: {', '.join(missing_fields)}.",
            details=f"Missing fields: {missing_fields}, Available fields: {[f for f in dir(result) if not f.startswith('_')]}"
        )


def validate_parameter_shapes(
    parameters: Dict[str, Union[np.ndarray, Tensor]],
    expected_shapes: Dict[str, Tuple[int, ...]],
    model_name: str = "model"
) -> None:
    """Validate that parameters have expected shapes.
    
    Parameters
    ----------
    parameters : dict
        Dictionary of parameter names to arrays/tensors
    expected_shapes : dict
        Dictionary of parameter names to expected shape tuples
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    DataValidationError
        If any parameter has unexpected shape
    """
    for param_name, expected_shape in expected_shapes.items():
        if param_name not in parameters:
            raise DataValidationError(
                f"{model_name} missing parameter: {param_name}.",
                details=f"Expected parameters: {list(expected_shapes.keys())}"
            )
        
        param = parameters[param_name]
        if param is None:
            continue  # None is allowed (optional parameters)
        
        # Get shape
        if isinstance(param, Tensor):
            actual_shape = tuple(param.shape)
        elif isinstance(param, np.ndarray):
            actual_shape = param.shape
        else:
            raise DataValidationError(
                f"{model_name} parameter {param_name} must be Tensor or np.ndarray, got {type(param).__name__}.",
                details=f"param_name={param_name}, type={type(param).__name__}"
            )
        
        # Check shape
        if actual_shape != expected_shape:
            raise DataValidationError(
                f"{model_name} parameter {param_name} shape mismatch: expected {expected_shape}, got {actual_shape}.",
                details=f"param_name={param_name}, expected_shape={expected_shape}, actual_shape={actual_shape}"
            )


# Alias for compatibility
validate_companion_matrix = validate_companion_stability


# ============================================================================
# Simple validation functions (moved from utils.validation)
# ============================================================================

def validate_ar_order(ar_order: int, min_order: int = 1, max_order: int = 20) -> int:
    """Validate AR order (VAR lag order)."""
    if not isinstance(ar_order, int):
        raise ConfigurationError(f"ar_order must be an integer, got {type(ar_order).__name__}")
    if ar_order < min_order:
        raise ConfigurationError(f"ar_order must be >= {min_order}, got {ar_order}")
    if ar_order > max_order:
        raise ConfigurationError(f"ar_order must be <= {max_order}, got {ar_order}. Very high orders may cause numerical instability.")
    return ar_order


def validate_ma_order(ma_order: int, min_order: int = 0, max_order: int = 10) -> int:
    """Validate MA order."""
    if not isinstance(ma_order, int):
        raise ConfigurationError(f"ma_order must be an integer, got {type(ma_order).__name__}")
    if ma_order < min_order:
        raise ConfigurationError(f"ma_order must be >= {min_order}, got {ma_order}")
    if ma_order > max_order:
        raise ConfigurationError(f"ma_order must be <= {max_order}, got {ma_order}. Very high orders may cause numerical instability.")
    return ma_order


def validate_learning_rate(learning_rate: float, min_lr: float = DEFAULT_MIN_DELTA, max_lr: float = 1.0) -> float:
    """Validate learning rate.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate to validate
    min_lr : float, default DEFAULT_MIN_DELTA
        Minimum learning rate threshold (uses DEFAULT_MIN_DELTA constant)
    max_lr : float, default 1.0
        Maximum learning rate threshold
        
    Returns
    -------
    float
        Validated learning rate
    """
    if not isinstance(learning_rate, (int, float)):
        raise ConfigurationError(f"learning_rate must be a number, got {type(learning_rate).__name__}")
    learning_rate = float(learning_rate)
    if learning_rate <= 0:
        raise ConfigurationError(f"learning_rate must be > 0, got {learning_rate}")
    if learning_rate < min_lr:
        _logger.warning(f"learning_rate {learning_rate} is very small (< {min_lr}). Training may be very slow.")
    if learning_rate > max_lr:
        raise ConfigurationError(f"learning_rate {learning_rate} is very large (> {max_lr}). Training may be unstable. Consider reducing it.")
    return learning_rate


def validate_batch_size(batch_size: int, min_size: int = 1) -> int:
    """Validate batch size."""
    if not isinstance(batch_size, int):
        raise ConfigurationError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < min_size:
        raise ConfigurationError(f"batch_size must be >= {min_size}, got {batch_size}")
    return batch_size


def validate_data_shape(
    data: Union[np.ndarray, Tensor],
    min_dims: int = 2,
    max_dims: int = 3,
    min_size: int = 1
) -> Tuple[int, ...]:
    """Validate data shape."""
    if isinstance(data, Tensor):
        shape = tuple(data.shape)
    elif isinstance(data, np.ndarray):
        shape = data.shape
    else:
        raise DataValidationError(f"data must be numpy array or torch Tensor, got {type(data).__name__}")
    
    if len(shape) < min_dims:
        raise DataValidationError(f"data must have at least {min_dims} dimensions, got {len(shape)}")
    if len(shape) > max_dims:
        raise DataValidationError(f"data must have at most {max_dims} dimensions, got {len(shape)}")
    
    if any(s < min_size for s in shape):
        raise DataValidationError(f"All dimensions must be >= {min_size}, got shape {shape}")
    
    return shape


def validate_no_nan_inf(data: Union[np.ndarray, Tensor], name: str = "data") -> None:
    """Check for NaN and Inf values in data."""
    if isinstance(data, Tensor):
        has_nan = torch.isnan(data).any().item()
        has_inf = torch.isinf(data).any().item()
    elif isinstance(data, np.ndarray):
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
    else:
        return  # Skip validation for other types
    
    if has_nan:
        raise DataValidationError(f"{name} contains NaN values. Please handle missing data before training.")
    if has_inf:
        raise DataValidationError(f"{name} contains Inf values. Please check data preprocessing.")


def _validate_integer_range(
    value: int,
    min_val: int,
    max_val: int,
    name: str,
    warning_msg: str
) -> int:
    """Helper function to validate integer range with consistent error handling.
    
    Parameters
    ----------
    value : int
        Value to validate
    min_val : int
        Minimum allowed value
    max_val : int
        Maximum allowed value (warning issued if exceeded)
    name : str
        Name of the parameter for error messages
    warning_msg : str
        Warning message to log if value exceeds max_val
        
    Returns
    -------
    int
        Validated value
        
    Raises
    ------
    ConfigurationError
        If value is not an integer or is less than min_val
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")
    if value < min_val:
        raise ConfigurationError(f"{name} must be >= {min_val}, got {value}")
    if value > max_val:
        _logger.warning(warning_msg)
    return value


def validate_horizon(horizon: int, min_horizon: int = 1, max_horizon: int = 100) -> int:
    """Validate forecast horizon."""
    return _validate_integer_range(
        horizon,
        min_horizon,
        max_horizon,
        "horizon",
        f"horizon {horizon} is very large (> {max_horizon}). Forecast accuracy may degrade significantly."
    )


def validate_irf_horizon(horizon: int, min_horizon: int = 1, max_horizon: int = 200) -> int:
    """Validate IRF computation horizon."""
    return _validate_integer_range(
        horizon,
        min_horizon,
        max_horizon,
        "IRF horizon",
        f"IRF horizon {horizon} is very large (> {max_horizon}). Computation may be slow and IRF magnitudes may decay to near-zero."
    )


def validate_eigenvalue_bounds(
    eigenvalues: np.ndarray,
    max_magnitude: float = 1.0,
    warn_threshold: float = 0.99
) -> None:
    """Validate eigenvalue magnitudes for stability."""
    magnitudes = np.abs(eigenvalues)
    max_mag = np.max(magnitudes)
    
    if max_mag >= max_magnitude:
        raise NumericalStabilityError(
            f"Maximum eigenvalue magnitude {max_mag:.6f} >= {max_magnitude}. Model may be unstable. Consider regularization or differencing."
        )
    
    if max_mag > warn_threshold:
        _logger.warning(
            f"Maximum eigenvalue magnitude {max_mag:.6f} > {warn_threshold}. Model may be near-unstable. Consider regularization."
        )


def validate_matrix_condition(
    matrix: Union[np.ndarray, Tensor],
    max_condition: float = 1e12,
    name: str = "matrix"
) -> None:
    """Validate matrix condition number."""
    matrix_np = to_numpy(matrix)
    
    if matrix_np.size == 0:
        return
    
    try:
        condition = np.linalg.cond(matrix_np)
        if condition > max_condition:
            raise NumericalStabilityError(
                f"{name} is ill-conditioned (condition number {condition:.2e} > {max_condition:.2e}). Numerical errors may occur. Consider regularization."
            )
        if condition > max_condition / 100:
            _logger.warning(f"{name} has high condition number {condition:.2e}. Consider regularization.")
    except (np.linalg.LinAlgError, ValueError):
        # Matrix may be singular or not square - skip condition check
        pass


def validate_update_data_shape(
    data: np.ndarray,
    training_data: Optional[np.ndarray],
    model_name: str = "model"
) -> None:
    """Validate that new data shape matches training data for update() or predict().
    
    This function validates that:
    1. Model has been trained (training_data is not None)
    2. Data is 2D array with shape (T_new x N) where:
       - T_new: Number of new time steps (can be any positive integer)
       - N: Number of series (must match training data)
    3. Number of series (N) matches training data
    
    Parameters
    ----------
    data : np.ndarray
        New data to validate (must be 2D: T_new x N)
        - T_new: Number of new time steps (any positive integer)
        - N: Number of series (must match training data)
    training_data : np.ndarray, optional
        Training data array (T_train x N) for shape comparison.
        If None, raises ModelNotTrainedError.
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    ModelNotTrainedError
        If model has not been trained yet (training_data is None)
    DataValidationError
        If data shape doesn't match training data (N must match)
    """
    # Validate model is trained
    if training_data is None:
        raise ModelNotTrainedError(
            f"{model_name} must be trained before validating data shape",
            details="Please call fit() method first"
        )
    
    # Validate data is 2D
    if data.ndim != 2:
        raise DataValidationError(
            f"{model_name} data must be 2D array (T_new x N), got {data.ndim}D array",
            details=f"Shape: {data.shape}. Expected 2D array with shape (T_new, N) where T_new is number of time steps and N is number of series."
        )
    
    # Validate number of series (N) matches training data
    expected_N = training_data.shape[1]
    actual_N = data.shape[1]
    
    if actual_N != expected_N:
        raise DataValidationError(
            f"{model_name} new data has {actual_N} series but training data has {expected_N} series. "
            f"Number of series (N) must match.",
            details=f"Expected shape: (T_new, {expected_N}), got: {data.shape}. "
                    f"Note: T_new can be any positive integer, but N must match training data."
        )


def validate_ndarray_ndim(
    arr: Any,
    name: str,
    expected_ndim: int
) -> None:
    """Validate a numpy array has expected number of dimensions.
    
    Parameters
    ----------
    arr : Any
        Array to validate
    name : str
        Name of the array for error messages
    expected_ndim : int
        Expected number of dimensions
        
    Raises
    ------
    DataValidationError
        If array is not a numpy array or has wrong number of dimensions
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != expected_ndim:
        raise DataValidationError(
            f"{name} must be {expected_ndim}D numpy array, got shape {arr.shape if isinstance(arr, np.ndarray) else 'not array'}"
        )


def validate_parameters_initialized(
    parameters: Dict[str, Optional[Any]],
    model_name: str = "model"
) -> None:
    """Validate that model parameters are initialized.
    
    Parameters
    ----------
    parameters : dict
        Dictionary mapping parameter names to values (None indicates uninitialized)
    model_name : str, default="model"
        Model name for error messages
        
    Raises
    ------
    ModelNotInitializedError
        If any required parameter is None
    """
    missing_params = [name for name, value in parameters.items() if value is None]
    if missing_params:
        raise ModelNotInitializedError(
            f"{model_name}: Model parameters not initialized",
            details=f"Parameters {missing_params} are required but are None. Please call fit() first to initialize parameters"
        )


def validate_and_convert_update_data(
    data: Union[np.ndarray, Any],
    training_data: Optional[np.ndarray],
    dtype: type = np.float64,
    model_name: str = "model"
) -> np.ndarray:
    """Validate and convert data for update() or predict() methods.
    
    Users must preprocess data themselves (same preprocessing as training).
    This function only validates shape and converts to numpy.
    
    **Data Shape**: The input data must be 2D with shape (T_new x N) where:
    - T_new: Number of new time steps (can be any positive integer)
    - N: Number of series (must match training data)
    
    **Supported Types**:
    - numpy.ndarray: (T_new x N) array
    - pandas.DataFrame: DataFrame with N columns, T_new rows
    - polars.DataFrame: DataFrame with N columns, T_new rows
    
    Parameters
    ----------
    data : np.ndarray, pandas.DataFrame, or polars.DataFrame
        Preprocessed observations with shape (T_new x N) where:
        - T_new: Number of new time steps (any positive integer)
        - N: Number of series (must match training data)
    training_data : np.ndarray, optional
        Training data array (T_train x N) for shape comparison.
        If None, raises ModelNotTrainedError.
    dtype : type, default=np.float64
        Data type for converted array
    model_name : str, default="model"
        Model name for error messages
        
    Returns
    -------
    np.ndarray
        Data as numpy array with shape (T_new x N)
        
    Raises
    ------
    ModelNotTrainedError
        If model has not been trained yet
    DataValidationError
        If data shape doesn't match training data (N must match)
    """
    # Convert to NumPy (handles pandas, polars, torch, numpy)
    data_np = to_numpy(data).astype(dtype)
    
    # Validate shape matches training data
    validate_update_data_shape(data_np, training_data, model_name=model_name)
    
    return data_np


__all__ = [
    'validate_model_components',
    'validate_companion_stability',
    'validate_companion_matrix',  # Alias for backward compatibility
    'validate_model_initialized',
    'validate_prediction_inputs',
    'validate_forecast_inputs',
    'validate_result_structure',
    'validate_parameter_shapes',
    # Simple validation functions
    'validate_ar_order',
    'validate_ma_order',
    'validate_learning_rate',
    'validate_batch_size',
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_horizon',
    'validate_irf_horizon',
    'validate_eigenvalue_bounds',
    'validate_matrix_condition',
    # Update/predict data validation
    'validate_update_data_shape',
    'validate_and_convert_update_data',
    # Array validation
    'validate_ndarray_ndim',
    # Parameter validation
    'validate_parameters_initialized',
    # DDFM-specific validators
    'validate_factors',
    'validate_ddfm_training_data',
]


def validate_factors(
    factors: Union[np.ndarray, Tensor],
    num_factors: int,
    operation: str = "operation"
) -> np.ndarray:
    """Validate and normalize factors shape and content quality.
    
    Parameters
    ----------
    factors : np.ndarray or Tensor
        Factors to validate
    num_factors : int
        Expected number of factors (for reshaping 1D arrays)
    operation : str, default "operation"
        Operation name for error messages
        
    Returns
    -------
    np.ndarray
        Validated and normalized factors (2D array)
        
    Raises
    ------
    DataError
        If factors are empty, invalid shape, or contain NaN/Inf
    """
    from ..utils.errors import DataError
    
    factors = to_numpy(factors)
    
    if factors.ndim == 0 or factors.size == 0:
        raise DataError(
            f"Factors validation failed: factors is empty or invalid (shape: {factors.shape})",
            details="This indicates training did not complete properly"
        )
    
    # Reshape 1D factors to 2D
    if factors.ndim == 1:
        factors = factors.reshape(-1, num_factors) if factors.size > 0 else factors.reshape(0, num_factors)
    
    if factors.ndim != 2:
        raise DataError(
            f"Factors validation failed: factors must be 2D array (T x m), got shape {factors.shape}",
            details="Factors should be a 2D array with shape (T, m) where T is time steps and m is number of factors"
        )
    
    # Validate factors are finite
    validate_no_nan_inf(factors, name=f"factors ({operation})")
    
    return factors


def validate_ddfm_training_data(
    X_torch: Tensor,
    num_factors: int,
    encoder_layers: Optional[List[int]] = None,
    encoder: Optional[Any] = None,
    operation: str = "training setup"
) -> Tuple[int, int]:
    """Validate data dimensions and model configuration before training starts.
    
    Parameters
    ----------
    X_torch : torch.Tensor
        Training data tensor
    num_factors : int
        Number of factors
    encoder_layers : List[int], optional
        Encoder layer dimensions
    encoder : object, optional
        Encoder instance (for input_dim validation)
    operation : str, default "training setup"
        Operation name for error messages
        
    Returns
    -------
    T : int
        Number of time steps
    N : int
        Number of variables
        
    Raises
    ------
    DataError
        If data is None, invalid type, or invalid shape
    ConfigurationError
        If num_factors is invalid or encoder dimensions don't match
    """
    from ..config.constants import MIN_VARIABLES, MIN_DDFM_TIME_STEPS
    from ..utils.errors import DataError, ConfigurationError
    from ..utils.validation import check_condition
    
    check_condition(
        X_torch is not None,
        DataError,
        f"DDFM {operation} failed: X_torch is None",
        details="Please provide training data"
    )
    
    check_condition(
        isinstance(X_torch, Tensor),
        DataError,
        f"DDFM {operation} failed: X_torch must be torch.Tensor, got {type(X_torch)}",
        details="Training data must be a torch.Tensor. Convert numpy arrays using torch.from_numpy()"
    )
    
    # Validate shape using existing utility
    validate_data_shape(X_torch, min_dims=2, max_dims=2, min_size=MIN_DDFM_TIME_STEPS)
    T, N = X_torch.shape
    
    check_condition(
        N >= MIN_VARIABLES,
        DataError,
        f"DDFM {operation} failed: Need at least {MIN_VARIABLES} series, got N={N}",
        details="DDFM requires at least 1 series (variable) in the data"
    )
    
    check_condition(
        num_factors is not None and num_factors >= 1,
        ConfigurationError,
        f"DDFM {operation} failed: num_factors must be >= 1, got {num_factors}",
        details="Number of factors must be a positive integer"
    )
    
    check_condition(
        num_factors <= N,
        ConfigurationError,
        f"DDFM {operation} failed: num_factors ({num_factors}) cannot exceed number of series (N={N})",
        details="Number of factors cannot exceed the number of input series"
    )
    
    if encoder_layers is not None and len(encoder_layers) > 0:
        if encoder_layers[0] != N:
            _logger.warning(
                f"DDFM {operation}: encoder_layers[0] ({encoder_layers[0]}) does not match input dimension (N={N}). "
                "Encoder will be reinitialized with correct input dimension."
            )
    
    if encoder is not None:
        if hasattr(encoder, 'input_dim') and encoder.input_dim != N:
            raise ConfigurationError(
                f"DDFM {operation} failed: encoder.input_dim ({encoder.input_dim}) must match input dimension (N={N})",
                details="Encoder input dimension must match the number of series in the data"
            )
    
    return T, N

