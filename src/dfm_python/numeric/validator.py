"""Input and parameter validators shared by the DFM, DDFM, iVDFM and AFM engines."""

import os
from typing import Optional, Union, List, Tuple, Any, Dict
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..utils.errors import (
    ModelNotInitializedError,
    ModelNotTrainedError,
    NumericalError,
    DataValidationError,
    DataError,
    ConfigurationError
)
from ..utils.validation import check_condition
from ..config.types import to_numpy
from ..config.constants import MAX_EIGENVALUE, MIN_VARIABLES, MIN_DDFM_TIME_STEPS
from ..logger import get_logger

_logger = get_logger(__name__)


def validate_data_shape(
    data: Union[np.ndarray, Tensor],
    min_dims: int = 2,
    max_dims: int = 3,
    min_size: int = 1
) -> Tuple[int, ...]:
    """Return ``data.shape`` after checking its ndim and that every dimension is >= ``min_size``."""
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
    """Raise ``DataValidationError`` if an array or tensor holds NaN or Inf; other types pass."""
    if isinstance(data, Tensor):
        has_nan = torch.isnan(data).any().item()
        has_inf = torch.isinf(data).any().item()
    elif isinstance(data, np.ndarray):
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
    else:
        return

    if has_nan:
        raise DataValidationError(f"{name} contains NaN values. Please handle missing data before training.")
    if has_inf:
        raise DataValidationError(f"{name} contains Inf values. Please check data preprocessing.")


def validate_horizon(horizon: int, min_horizon: int = 1, max_horizon: int = 100) -> int:
    """Return ``horizon`` after checking it is an int >= ``min_horizon``; warn above ``max_horizon``.

    Setting ``DFM_SUPPRESS_HORIZON_WARNING=1`` silences the warning only
    (long-horizon benchmarks such as ETT 192/336/720 would otherwise warn on
    every call); the checks still apply.
    """
    if not isinstance(horizon, int):
        raise ConfigurationError(f"horizon must be an integer, got {type(horizon).__name__}")
    if horizon < min_horizon:
        raise ConfigurationError(f"horizon must be >= {min_horizon}, got {horizon}")
    suppressed = os.environ.get("DFM_SUPPRESS_HORIZON_WARNING", "").strip().lower() in ("1", "true", "yes")
    if horizon > max_horizon and not suppressed:
        _logger.warning(
            f"horizon {horizon} is very large (> {max_horizon}). Forecast accuracy may degrade significantly."
        )
    return horizon


def validate_update_data_shape(
    data: np.ndarray,
    training_data: Optional[np.ndarray],
    model_name: str = "model"
) -> None:
    """Check that ``data`` is 2-D with the same number of series as ``training_data`` (T_train x N).

    Raises
    ------
    ModelNotTrainedError
        If ``training_data`` is None.
    DataValidationError
        If ``data`` is not 2-D or its column count differs.
    """
    if training_data is None:
        raise ModelNotTrainedError(
            f"{model_name} must be trained before validating data shape",
            details="Please call fit() method first"
        )
    if data.ndim != 2:
        raise DataValidationError(
            f"{model_name} data must be 2D array (T_new x N), got {data.ndim}D array",
            details=f"Shape: {data.shape}. Expected 2D array with shape (T_new, N) where T_new is number of time steps and N is number of series."
        )

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
    """Check a DataFrame's non-date columns against the training feature order.

    Only ``pandas.DataFrame`` input is checked; ``expected_features`` defaults
    to ``scaler.feature_names_in_`` and the check is skipped when neither is
    available. Columns named date, date_w, year, month or day are ignored.

    Raises
    ------
    ValueError
        On a column count or order mismatch, or a non-numeric column.
    """
    if not isinstance(data, pd.DataFrame):
        return

    if expected_features is None:
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            return

    date_cols = {'date', 'date_w', 'year', 'month', 'day'}
    data_features = [col for col in data.columns if col not in date_cols]

    if len(data_features) != len(expected_features):
        raise ValueError(
            f"DFM: Column count mismatch!\n"
            f"Expected {len(expected_features)} features from training data.\n"
            f"Got {len(data_features)} features in data.\n"
            f"Expected columns (first 5): {expected_features[:5]}{'...' if len(expected_features) > 5 else ''}\n"
            f"Got columns (first 5): {data_features[:5]}{'...' if len(data_features) > 5 else ''}\n"
            f"Data must have exactly the same columns as training data."
        )

    if list(data_features) != expected_features:
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
    """Raise ``DataValidationError`` unless ``arr`` is an ``expected_ndim``-D numpy array."""
    if not isinstance(arr, np.ndarray) or arr.ndim != expected_ndim:
        raise DataValidationError(
            f"{name} must be {expected_ndim}D numpy array, got shape {arr.shape if isinstance(arr, np.ndarray) else 'not array'}"
        )


def validate_parameters_initialized(
    parameters: Dict[str, Optional[Any]],
    model_name: str = "model"
) -> None:
    """Raise ``ModelNotInitializedError`` naming every parameter whose value is None."""
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
    """Convert ``data`` (ndarray, pandas or polars DataFrame, tensor) to a ``dtype`` array (T_new x N).

    No preprocessing is applied; the caller must have scaled the data as at
    training time. Shape is checked with :func:`validate_update_data_shape`.
    """
    data_np = to_numpy(data).astype(dtype)
    validate_update_data_shape(data_np, training_data, model_name=model_name)
    return data_np


def validate_factors(
    factors: Union[np.ndarray, Tensor],
    num_factors: int,
    operation: str = "operation"
) -> np.ndarray:
    """Return ``factors`` as a finite 2-D (T x num_factors) array; 1-D input is reshaped.

    Raises
    ------
    DataError
        If ``factors`` is empty or has more than two dimensions.
    DataValidationError
        If ``factors`` contains NaN or Inf.
    """
    factors = to_numpy(factors)

    if factors.ndim == 0 or factors.size == 0:
        raise DataError(
            f"Factors validation failed: factors is empty or invalid (shape: {factors.shape})",
            details="This indicates training did not complete properly"
        )
    if factors.ndim == 1:
        factors = factors.reshape(-1, num_factors)
    if factors.ndim != 2:
        raise DataError(
            f"Factors validation failed: factors must be 2D array (T x m), got shape {factors.shape}",
            details="Factors should be a 2D array with shape (T, m) where T is time steps and m is number of factors"
        )

    validate_no_nan_inf(factors, name=f"factors ({operation})")
    return factors


def validate_ddfm_training_data(
    X_torch: Tensor,
    num_factors: int,
    encoder_layers: Optional[List[int]] = None,
    encoder: Optional[Any] = None,
    operation: str = "training setup"
) -> Tuple[int, int]:
    """Check DDFM training data and return ``(T, N)``.

    ``X_torch`` must be a 2-D tensor with every dimension >=
    ``MIN_DDFM_TIME_STEPS``, ``N >= MIN_VARIABLES`` and
    ``1 <= num_factors <= N``. A mismatching ``encoder_layers[0]`` only warns
    (the encoder is rebuilt); a mismatching ``encoder.input_dim`` raises.

    Raises
    ------
    DataError
        If ``X_torch`` is None or not a tensor, or has too few series.
    DataValidationError
        If the shape is invalid.
    ConfigurationError
        If ``num_factors`` or ``encoder.input_dim`` is incompatible.
    """
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
    """Raise ``DataValidationError`` unless ``len(bl_idxQ_i) == expected_rps``.

    ``expected_rps = rs * p_plus_one`` with ``rs = sum(r[block_pattern > 0])``;
    a mismatch means the block index construction in the EM M-step is wrong.
    The remaining arguments only feed the diagnostic message.
    """
    actual_rps = len(bl_idxQ_i)

    if actual_rps != expected_rps:
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
    """Check the initial DFM state-space parameters.

    Raises ``NumericalError`` on NaN/Inf; only warns on extreme magnitudes,
    negative or tiny covariance eigenvalues and an unusual scale of ``C``.
    """
    for name, param in [('A', A), ('C', C), ('Q', Q), ('R', R), ('Z_0', Z_0), ('V_0', V_0)]:
        try:
            validate_no_nan_inf(param, name=name)
        except DataValidationError as e:
            raise NumericalError(
                f"Initialization contains non-finite values in {name}",
                details=str(e)
            ) from e

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

    # Q is singular by construction (zero process noise on the companion lag states
    # and the idiosyncratic chain shift states), so only a negative eigenvalue is
    # reported for it.
    for name, cov in [('Q', Q), ('R', R), ('V_0', V_0)]:
        try:
            eigenvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
            min_eigenval = np.min(eigenvals)
            scale = max(1.0, float(np.max(np.abs(eigenvals))))
            if min_eigenval < -1e-12 * scale:
                _logger.warning(
                    f"{name} has negative eigenvalues (min={min_eigenval:.2e}). "
                    f"Matrix may not be positive definite."
                )
            if name != 'Q' and min_eigenval < 1e-8:
                _logger.warning(
                    f"{name} has very small eigenvalues (min={min_eigenval:.2e}). "
                    f"This may cause numerical instability."
                )
        except (np.linalg.LinAlgError, ValueError) as e:  # not square, empty, or eigvalsh failed
            _logger.warning(f"Could not check eigenvalues for {name}: {e}")

    C_scale = np.abs(C).mean()
    if C_scale > 100 or C_scale < 0.01:
        _logger.warning(
            f"Observation matrix C has unusual scale (mean(abs)={C_scale:.2e}). "
            f"This may indicate data scaling issues."
        )


__all__ = [
    'validate_data_shape',
    'validate_no_nan_inf',
    'validate_horizon',
    'validate_update_data_shape',
    'validate_and_convert_update_data',
    'validate_ndarray_ndim',
    'validate_parameters_initialized',
    'validate_dfm_initialization',
    'validate_factors',
    'validate_ddfm_training_data',
    'validate_block_index_dimensions',
]
