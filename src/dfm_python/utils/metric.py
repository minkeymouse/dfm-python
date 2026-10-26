"""Forecast evaluation metrics: RMSE, MAE (sktime), MAPE and R2 (scikit-learn).

Each function takes ``actual`` and ``predicted`` of the same shape, (T,) or
(T x N), an optional boolean ``mask`` of that shape, and returns
``(overall, per_series)``: the metric over all masked entries pooled and
over each series separately, ``per_series`` being (N,) or a length-1 array
for 1-D input. Entries that are NaN or Inf in either array are always
excluded; a series (or the pooled set) with no valid entry gives NaN.

sktime is imported when RMSE or MAE is first computed: importing it takes
seconds and imports torch when that is installed.
"""

from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from ..utils.errors import DataValidationError


def _as_float(result) -> float:
    """sktime metrics may return a Series (multioutput); reduce to a float."""
    if hasattr(result, 'iloc'):
        return float(result.iloc[0] if len(result) > 0 else result)
    return float(result)


def _masked_metric(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray],
    score: Callable[[np.ndarray, np.ndarray], float],
) -> Tuple[float, np.ndarray]:
    """Apply ``score`` to the masked entries overall and per series (see module docstring)."""
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}",
            details=f"Shape mismatch: actual.shape={actual.shape}, predicted.shape={predicted.shape}"
        )

    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)

    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        value = score(actual[mask], predicted[mask])
        return value, np.array([value])

    N = actual.shape[1]
    per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            per_series[i] = score(actual[series_mask, i], predicted[series_mask, i])
        else:
            per_series[i] = np.nan

    if np.any(mask):
        overall = score(actual[mask], predicted[mask])
    else:
        overall = np.nan
    return overall, per_series


def _require_sktime(name: str) -> Any:
    """``sktime.performance_metrics.forecasting``, or ``ImportError`` naming the ``metrics`` extra."""
    try:
        from sktime.performance_metrics import forecasting
    except ImportError as err:
        raise ImportError(
            f"sktime is required for {name}. "
            "Install it with: pip install 'dfm-python[metrics]'"
        ) from err
    return forecasting


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Root mean squared error via sktime ``MeanSquaredError(square_root=True)``; needs the ``metrics`` extra."""
    metrics = _require_sktime("calculate_rmse")

    def score(a: np.ndarray, p: np.ndarray) -> float:
        return _as_float(metrics.MeanSquaredError(square_root=True)(pd.Series(a), pd.Series(p)))

    return _masked_metric(actual, predicted, mask, score)


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Mean absolute error via sktime ``MeanAbsoluteError``; needs the ``metrics`` extra."""
    metrics = _require_sktime("calculate_mae")

    def score(a: np.ndarray, p: np.ndarray) -> float:
        return _as_float(metrics.MeanAbsoluteError()(pd.Series(a), pd.Series(p)))

    return _masked_metric(actual, predicted, mask, score)


def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Mean absolute percentage error via ``sklearn.metrics.mean_absolute_percentage_error``."""
    return _masked_metric(actual, predicted, mask, mean_absolute_percentage_error)


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Coefficient of determination via ``sklearn.metrics.r2_score``."""
    return _masked_metric(actual, predicted, mask, r2_score)


__all__ = [
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
]
