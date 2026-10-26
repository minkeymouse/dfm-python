"""Linear-algebra fallback wrapper and missing-value interpolation."""

from typing import Any, Optional, Callable
import numpy as np
import pandas as pd

from ..logger import get_logger

_logger = get_logger(__name__)


def handle_linear_algebra_error(
    operation: Callable,
    operation_name: str,
    fallback_value: Optional[Any] = None,
    fallback_func: Optional[Callable] = None,
    *args,
    **kwargs
) -> Any:
    """Call ``operation(*args, **kwargs)``; on ``LinAlgError``/``ValueError`` warn and fall back.

    ``fallback_func`` (called with the same arguments) takes precedence over
    ``fallback_value``; with neither the error is re-raised.
    """
    try:
        return operation(*args, **kwargs)
    except (np.linalg.LinAlgError, ValueError) as e:
        _logger.warning(
            f"{operation_name} failed ({type(e).__name__}): {e}. Using fallback."
        )
        if fallback_func is not None:
            return fallback_func(*args, **kwargs)
        if fallback_value is not None:
            return fallback_value
        raise


def interpolate_array(
    arr: np.ndarray,
    axis: int = 0,
    kind: str = 'cubic',
    fill_value: str = 'extrapolate'
) -> np.ndarray:
    """Interpolate NaNs along ``axis`` with ``scipy.interpolate.interp1d``.

    A series with a single valid point is filled with that value; one with
    none is left as is. For ``axis != 0`` the array is transposed, filled
    column by column and transposed back.
    """
    from scipy.interpolate import interp1d

    arr = np.asarray(arr)
    if arr.ndim == 1:
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            if mask.sum() == 1:
                arr = np.full_like(arr, arr[mask][0])
            return arr

        x_valid = np.where(mask)[0]
        y_valid = arr[mask]
        f = interp1d(x_valid, y_valid, kind=kind, fill_value=fill_value, bounds_error=False)
        x_all = np.arange(len(arr))
        return f(x_all)

    result = arr.copy()
    if axis == 0:
        for i in range(arr.shape[1]):
            col_data = arr[:, i]
            mask = ~np.isnan(col_data)
            if mask.sum() >= 2:
                x_valid = np.where(mask)[0]
                y_valid = col_data[mask]
                f = interp1d(x_valid, y_valid, kind=kind, fill_value=fill_value, bounds_error=False)
                x_all = np.arange(len(col_data))
                result[:, i] = f(x_all)
            elif mask.sum() == 1:
                result[:, i] = col_data[mask][0]
    else:
        arr_t = np.moveaxis(arr, axis, 0)
        result_t = interpolate_array(arr_t, axis=0, kind=kind, fill_value=fill_value)
        result = np.moveaxis(result_t, 0, axis)
    return result


def interpolate_dataframe(
    df: pd.DataFrame,
    method: str = 'linear',
    limit: Optional[int] = 10,
    limit_direction: str = 'both'
) -> pd.DataFrame:
    """Fill missing values with ``DataFrame.interpolate``; the result has no NaNs.

    Gaps the interpolation leaves are forward/backward filled, then set to
    the column mean (0 for all-NaN columns).

    Parameters
    ----------
    df : pd.DataFrame
    method : str, default 'linear'
        Any pandas method. 'spline' (order 3) and 'cubic' fall back to
        'linear' when they fail (too few points); 'linear' is the stable
        choice under heavy missingness.
    limit : int, optional, default 10
        Maximum consecutive NaNs to fill, which bounds extrapolation; None is unlimited.
    limit_direction : str, default 'both'
        'forward', 'backward' or 'both'.
    """
    df_interpolated = df.copy()

    if df_interpolated.isna().sum().sum() == 0:
        return df_interpolated

    if method in ['spline', 'cubic']:
        try:
            df_interpolated = df_interpolated.interpolate(
                method=method,
                limit_direction=limit_direction,
                order=3 if method == 'spline' else None,
                limit=limit
            )
        except Exception:  # documented fallback; scipy's FITPACK raises a private error class, not a ValueError
            _logger.warning(
                "Spline interpolation failed (possibly not enough points), falling back to linear"
            )
            df_interpolated = df_interpolated.interpolate(
                method='linear',
                limit_direction=limit_direction,
                limit=limit
            )
    else:
        df_interpolated = df_interpolated.interpolate(
            method=method,
            limit_direction=limit_direction,
            limit=limit
        )

    if df_interpolated.isna().sum().sum() > 0:
        df_interpolated = df_interpolated.ffill().bfill()

    if df_interpolated.isna().sum().sum() > 0:
        df_interpolated = df_interpolated.fillna(df_interpolated.mean().fillna(0))

    return df_interpolated
