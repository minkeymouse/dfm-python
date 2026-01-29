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

from typing import Optional, Union, List, Tuple, Any, Dict, TYPE_CHECKING
import numpy as np

# Tensor type for DDFM compatibility
# Note: DFM only uses numpy arrays, but DDFM uses torch.Tensor
# This shared validator module supports both for validate_no_nan_inf and validate_data_shape
if TYPE_CHECKING:
    try:
        import torch
        Tensor = torch.Tensor
    except ImportError:
        Tensor = Any
else:
    try:
        import torch
        Tensor = torch.Tensor
    except ImportError:
        Tensor = Any

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


# ============================================================================
# Simple validation functions (moved from utils.validation)
# ============================================================================


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
    # This warning is often noisy for long-horizon benchmarking (e.g., ETT 192/336/720).
    # Allow silencing via env var without changing call sites.
    import os
    if os.environ.get("DFM_SUPPRESS_HORIZON_WARNING", "").strip().lower() in ("1", "true", "yes"):
        return int(horizon)
    return _validate_integer_range(
        horizon,
        min_horizon,
        max_horizon,
        "horizon",
        f"horizon {horizon} is very large (> {max_horizon}). Forecast accuracy may degrade significantly."
    )


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


def validate_column_order(
    data: Any,
    scaler: Optional[Any],
    expected_features: Optional[List[str]] = None
) -> None:
    """Validate DataFrame column order matches training data.
    
    Only validates pandas DataFrames. If expected_features is not provided,
    attempts to extract from scaler.feature_names_in_.
    
    Parameters
    ----------
    data : Any
        Data to validate (only pandas.DataFrame is validated)
    scaler : Any, optional
        Scaler instance with feature_names_in_ attribute
    expected_features : List[str], optional
        Expected column names in order. If None, extracted from scaler.
        
    Raises
    ------
    ValueError
        If column count or order doesn't match expected features
    """
    try:
        import pandas as pd
    except ImportError:
        return  # Cannot validate without pandas
    
    if not isinstance(data, pd.DataFrame):
        return  # Only validate DataFrames
    
    if expected_features is None:
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            return  # Cannot validate without expected features
    
    # Extract numeric columns (exclude date columns)
    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    data_features = [col for col in data.columns if col not in date_cols]
    
    # Validate column count
    if len(data_features) != len(expected_features):
        raise ValueError(
            f"DFM: Column count mismatch!\n"
            f"Expected {len(expected_features)} features from training data.\n"
            f"Got {len(data_features)} features in data.\n"
            f"Expected columns (first 5): {expected_features[:5]}{'...' if len(expected_features) > 5 else ''}\n"
            f"Got columns (first 5): {data_features[:5]}{'...' if len(data_features) > 5 else ''}\n"
            f"Data must have exactly the same columns as training data."
        )
    
    # Validate column order
    if list(data_features) != expected_features:
        # Find first mismatch for error message
        mismatch_info = ""
        if len(expected_features) > 0 and len(data_features) > 0:
            first_mismatch = next(
                (i for i, (e, d) in enumerate(zip(expected_features, data_features)) if e != d),
                None
            )
            if first_mismatch is not None:
                mismatch_info = (
                    f"\nFirst mismatch at index {first_mismatch}: "
                    f"expected '{expected_features[first_mismatch]}', got '{data_features[first_mismatch]}'"
                )
        
        raise ValueError(
            f"DFM: Column order mismatch!\n"
            f"Expected {len(expected_features)} features in order: {expected_features[:5]}{'...' if len(expected_features) > 5 else ''}\n"
            f"Got {len(data_features)} features in order: {data_features[:5]}{'...' if len(data_features) > 5 else ''}"
            f"{mismatch_info}\n"
            f"Data columns must be in the EXACT same order as training data.\n"
            f"Reorder columns to match training order."
        )
    
    # Validate all columns are numeric
    non_numeric_cols = [
        col for col in data_features
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"DFM: Non-numeric columns found: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}.\n"
            f"All columns must be numeric to match training data format."
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
    # Simple validation functions
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_horizon',
    # Update/predict data validation
    'validate_update_data_shape',
    'validate_and_convert_update_data',
    # Array validation
    'validate_ndarray_ndim',
    # Parameter validation
    'validate_parameters_initialized',
    'validate_dfm_initialization',
    # DDFM-specific validators
    'validate_factors',
    'validate_ddfm_training_data',
    # Block index validation
    'validate_block_index_dimensions',
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


def validate_block_index_dimensions(
    bl_idxQ_i: np.ndarray,
    expected_rps: int,
    block_pattern_idx: int,
    block_pattern: np.ndarray,
    n_blocks: int,
    r: np.ndarray,
    p_plus_one: int,
    operation: str = "slower-frequency series update"
) -> None:
    """Validate that block index array has correct dimensions for slower-frequency series update.
    
    This validator checks that `bl_idxQ_i` (indices for slower-frequency factor states)
    has the expected length `rps = rs * p_plus_one` where `rs` is the sum of factors
    in active blocks for the given block pattern.
    
    **Purpose**: Catches dimension mismatches that cause broadcasting errors when
    updating slower-frequency series loadings in the EM algorithm M-step.
    
    Parameters
    ----------
    bl_idxQ_i : np.ndarray
        Boolean mask indices for slower-frequency factor states (from np.where(bl_idxQ[i])[0])
    expected_rps : int
        Expected number of slower-frequency factor states: `rs * p_plus_one`
        where `rs = sum(r[bl_pattern > 0])`
    block_pattern_idx : int
        Index of the block pattern being processed (for error messages)
    block_pattern : np.ndarray
        Block pattern array indicating which blocks are active (for error messages)
    n_blocks : int
        Total number of blocks (for error messages)
    r : np.ndarray
        Number of factors per block (for computing expected rs)
    p_plus_one : int
        State dimension per factor (for computing expected rps)
    operation : str, default "slower-frequency series update"
        Operation name for error messages
        
    Raises
    ------
    DataValidationError
        If `len(bl_idxQ_i) != expected_rps`, indicating a bug in block index construction
        or block pattern matching.
        
    Examples
    --------
    >>> # Valid case: bl_idxQ_i has 9 indices, expected_rps=9
    >>> validate_block_index_dimensions(bl_idxQ_i, 9, 0, bl_pattern, 7, r, 3)
    >>> # No error raised
    >>> 
    >>> # Invalid case: bl_idxQ_i has 18 indices, expected_rps=9
    >>> validate_block_index_dimensions(bl_idxQ_i, 9, 0, bl_pattern, 7, r, 3)
    >>> # Raises DataValidationError with diagnostic information
    """
    actual_rps = len(bl_idxQ_i)
    
    if actual_rps != expected_rps:
        # Compute diagnostic information
        rs_expected = expected_rps // p_plus_one
        active_blocks = [j for j in range(n_blocks) if block_pattern[j] > 0]
        r_active = r[active_blocks] if len(active_blocks) > 0 else np.array([])
        rs_computed = int(np.sum(r_active))
        
        raise DataValidationError(
            f"{operation} failed: Block index dimension mismatch for pattern {block_pattern_idx}. "
            f"Expected {expected_rps} indices (rps = {rs_expected} factors × {p_plus_one}), "
            f"but got {actual_rps} indices from bl_idxQ[{block_pattern_idx}]. "
            f"This indicates a bug in block index construction: "
            f"bl_idxQ[{block_pattern_idx}] has {actual_rps // p_plus_one} factor states selected, "
            f"but block pattern indicates only {rs_computed} factors should be active.",
            details=(
                f"Pattern {block_pattern_idx}: active blocks {active_blocks}, "
                f"r[active] = {r_active.tolist() if len(r_active) > 0 else '[]'}, "
                f"rs_expected = {rs_expected}, rps_expected = {expected_rps}, "
                f"actual_rps = {actual_rps}, p_plus_one = {p_plus_one}, "
                f"block_pattern = {block_pattern.tolist()}, "
                f"bl_idxQ_i length = {actual_rps}, "
                f"bl_idxQ_i indices (first 10): {bl_idxQ_i[:10].tolist() if len(bl_idxQ_i) > 0 else '[]'}"
            )
        )


def validate_dfm_initialization(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray
) -> None:
    """Validate numerical stability of DFM initialized parameters.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix
    C : np.ndarray
        Observation matrix
    Q : np.ndarray
        Process noise covariance
    R : np.ndarray
        Observation noise covariance
    Z_0 : np.ndarray
        Initial state mean
    V_0 : np.ndarray
        Initial state covariance
        
    Raises
    ------
    NumericalError
        If any parameter contains non-finite values
    """
    from ..config.constants import MAX_EIGENVALUE
    
    # Check for NaN/Inf
    for name, param in [('A', A), ('C', C), ('Q', Q), ('R', R), ('Z_0', Z_0), ('V_0', V_0)]:
        try:
            validate_no_nan_inf(param, name=name)
        except DataValidationError as e:
            raise NumericalError(
                f"Initialization contains non-finite values in {name}",
                details=str(e)
            ) from e
    
    # Check for extreme values
    max_abs_values = {
        'A': np.abs(A).max(),
        'C': np.abs(C).max(),
        'Q': np.abs(Q).max(),
        'R': np.abs(R).max(),
        'V_0': np.abs(V_0).max()
    }
    
    extreme_threshold = MAX_EIGENVALUE * 1e3
    for name, max_val in max_abs_values.items():
        if max_val > extreme_threshold:
            _logger.warning(
                f"Large values detected in {name}: max(abs)={max_val:.2e}. "
                f"This may indicate scaling issues or numerical instability."
            )
    
    # Check covariance matrices are positive definite
    for name, cov in [('Q', Q), ('R', R), ('V_0', V_0)]:
        try:
            eigenvals = np.linalg.eigvals(cov)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < 0:
                _logger.warning(
                    f"{name} has negative eigenvalues (min={min_eigenval:.2e}). "
                    f"Matrix may not be positive definite."
                )
            if min_eigenval < 1e-8:
                _logger.warning(
                    f"{name} has very small eigenvalues (min={min_eigenval:.2e}). "
                    f"This may cause numerical instability."
                )
        except Exception as e:
            _logger.warning(f"Could not check eigenvalues for {name}: {e}")
    
    # Check data scaling (via observation matrix)
    C_scale = np.abs(C).mean()
    if C_scale > 100 or C_scale < 0.01:
        _logger.warning(
            f"Observation matrix C has unusual scale (mean(abs)={C_scale:.2e}). "
            f"This may indicate data scaling issues."
        )

