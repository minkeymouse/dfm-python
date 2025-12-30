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
from ..config.types import ArrayLike
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
    if isinstance(companion_matrix, Tensor):
        matrix_np = companion_matrix.detach().cpu().numpy()
    else:
        matrix_np = np.asarray(companion_matrix)
    
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
            raise ValueError(
                f"{display_name} must be 2D or 3D (with kernel dimension), "
                f"got shape {matrix_np.shape}"
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
        if isinstance(last_observation, Tensor):
            last_obs_np = last_observation.detach().cpu().numpy()
        else:
            last_obs_np = last_observation
        
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
        if isinstance(last_observation, Tensor):
            last_obs_np = last_observation.detach().cpu().numpy()
        else:
            last_obs_np = np.asarray(last_observation)
        
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


# Backward compatibility aliases
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


def validate_learning_rate(learning_rate: float, min_lr: float = 1e-6, max_lr: float = 1.0) -> float:
    """Validate learning rate."""
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


def validate_horizon(horizon: int, min_horizon: int = 1, max_horizon: int = 100) -> int:
    """Validate forecast horizon."""
    if not isinstance(horizon, int):
        raise ConfigurationError(f"horizon must be an integer, got {type(horizon).__name__}")
    if horizon < min_horizon:
        raise ConfigurationError(f"horizon must be >= {min_horizon}, got {horizon}")
    if horizon > max_horizon:
        _logger.warning(f"horizon {horizon} is very large (> {max_horizon}). Forecast accuracy may degrade significantly.")
    return horizon


def validate_irf_horizon(horizon: int, min_horizon: int = 1, max_horizon: int = 200) -> int:
    """Validate IRF computation horizon."""
    if not isinstance(horizon, int):
        raise ConfigurationError(f"IRF horizon must be an integer, got {type(horizon).__name__}")
    if horizon < min_horizon:
        raise ConfigurationError(f"IRF horizon must be >= {min_horizon}, got {horizon}")
    if horizon > max_horizon:
        _logger.warning(f"IRF horizon {horizon} is very large (> {max_horizon}). Computation may be slow and IRF magnitudes may decay to near-zero.")
    return horizon


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
    if isinstance(matrix, Tensor):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = matrix
    
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
]

