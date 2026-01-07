"""Common helper functions for error handling, config access, and validation.

This module provides reusable helpers to reduce code duplication and improve
consistency across the codebase.
"""

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
    """Handle linear algebra errors with fallback."""
    try:
        return operation(*args, **kwargs)
    except (np.linalg.LinAlgError, ValueError) as e:
        _logger.warning(
            f"{operation_name} failed ({type(e).__name__}): {e}. Using fallback."
        )
        if fallback_func is not None:
            return fallback_func(*args, **kwargs)
        elif fallback_value is not None:
            return fallback_value
        else:
            raise


# get_config_attr moved to utils.misc for consolidation with other config utilities

def interpolate_array(
    arr: np.ndarray,
    axis: int = 0,
    kind: str = 'cubic',
    fill_value: str = 'extrapolate'
) -> np.ndarray:
    """Interpolate missing values in a numpy array along specified axis."""
    from scipy.interpolate import interp1d
    
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # 1D array
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            # Not enough points to interpolate, return as-is or fill with mean
            if mask.sum() == 1:
                arr = np.full_like(arr, arr[mask][0])
            return arr
        
        x_valid = np.where(mask)[0]
        y_valid = arr[mask]
        f = interp1d(x_valid, y_valid, kind=kind, fill_value=fill_value, bounds_error=False)
        x_all = np.arange(len(arr))
        return f(x_all)
    else:
        # Multi-dimensional array - interpolate along specified axis
        result = arr.copy()
        if axis == 0:
            # Interpolate along first dimension (time)
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
            # For other axes, transpose, interpolate, then transpose back
            arr_t = np.moveaxis(arr, axis, 0)
            result_t = interpolate_array(arr_t, axis=0, kind=kind, fill_value=fill_value)
            result = np.moveaxis(result_t, 0, axis)
        return result


def interpolate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate DataFrame values using array interpolation.
    
    This function interpolates missing values in a DataFrame by converting
    to numpy array, applying interpolate_array, and returning a new DataFrame
    with the same index and columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially missing values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values, same index and columns as input
    """
    df_interpolated = df.copy()
    df_interpolated.values[:] = interpolate_array(df_interpolated.values)
    return df_interpolated


