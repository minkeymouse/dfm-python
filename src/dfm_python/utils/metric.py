"""Evaluation metrics for Dynamic Factor Models.

This module provides metric calculation functions for evaluating model performance:
- calculate_rmse: Root Mean Squared Error
- calculate_mae: Mean Absolute Error
- calculate_mape: Mean Absolute Percentage Error
- calculate_r2: R² (coefficient of determination)

All functions support:
- 1D and 2D arrays (single and multiple series)
- Masking for selective calculation
- Per-series and overall metrics
- Automatic handling of missing data (NaN values)
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.errors import DataValidationError

try:
    import sktime
    from sktime.performance_metrics.forecasting import (
        MeanSquaredError,
        MeanAbsoluteError,
    )
    _has_sktime = True
except ImportError:
    _has_sktime = False
    MeanSquaredError = None
    MeanAbsoluteError = None

try:
    import sklearn
    from sklearn.metrics import (
        mean_absolute_percentage_error,
        r2_score,
    )
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    mean_absolute_percentage_error = None
    r2_score = None

from ..logger import get_logger

_logger = get_logger(__name__)


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Uses sktime.performance_metrics.forecasting.MeanSquaredError with square_root=True.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    rmse_overall : float
        Overall RMSE averaged across all series and time periods
    rmse_per_series : np.ndarray
        RMSE for each series (N,) or scalar if 1D input
        
    Notes
    -----
    - Returns NaN for overall RMSE if no valid observations exist
    - Returns NaN for individual series if that series has no valid observations
    - Mask parameter allows selective calculation (e.g., exclude certain time periods)
    - Automatically handles missing data by excluding NaN values
    - Requires sktime to be installed
    """
    if not _has_sktime:
        raise ImportError(
            "sktime is required for calculate_rmse. "
            "Please install: pip install sktime"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}",
            details=f"Shape mismatch: actual.shape={actual.shape}, predicted.shape={predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        # Use non-NaN values in both actual and predicted
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        # Combine user mask with finite check
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case (single series)
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        # Convert to pandas Series for sktime
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        # Use sktime metric
        mse_metric = MeanSquaredError(square_root=True)
        rmse_result = mse_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(rmse_result, 'iloc'):
            rmse_series = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
        else:
            rmse_series = float(rmse_result)
        return rmse_series, np.array([rmse_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    rmse_per_series = np.zeros(N)
    
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            y_true = pd.Series(actual_series)
            y_pred = pd.Series(predicted_series)
            
            mse_metric = MeanSquaredError(square_root=True)
            rmse_result = mse_metric(y_true, y_pred)
            # Handle both scalar and Series returns
            if hasattr(rmse_result, 'iloc'):
                rmse_per_series[i] = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
            else:
                rmse_per_series[i] = float(rmse_result)
        else:
            rmse_per_series[i] = np.nan
    
    # Calculate overall RMSE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        mse_metric = MeanSquaredError(square_root=True)
        rmse_result = mse_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(rmse_result, 'iloc'):
            rmse_overall = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
        else:
            rmse_overall = float(rmse_result)
    else:
        rmse_overall = np.nan
    
    return rmse_overall, rmse_per_series


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Mean Absolute Error (MAE) between actual and predicted values.
    
    Uses sktime.performance_metrics.forecasting.MeanAbsoluteError.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    mae_overall : float
        Overall MAE averaged across all series and time periods
    mae_per_series : np.ndarray
        MAE for each series (N,) or scalar if 1D input
        
    Notes
    -----
    - Requires sktime to be installed
    """
    if not _has_sktime:
        raise ImportError(
            "sktime is required for calculate_mae. "
            "Please install: pip install sktime"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}",
            details=f"Shape mismatch: actual.shape={actual.shape}, predicted.shape={predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        # Convert to pandas Series for sktime
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        # Use sktime metric
        mae_metric = MeanAbsoluteError()
        mae_result = mae_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(mae_result, 'iloc'):
            mae_series = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
        else:
            mae_series = float(mae_result)
        return mae_series, np.array([mae_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    mae_per_series = np.zeros(N)
    
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            y_true = pd.Series(actual_series)
            y_pred = pd.Series(predicted_series)
            
            mae_metric = MeanAbsoluteError()
            mae_result = mae_metric(y_true, y_pred)
            # Handle both scalar and Series returns
            if hasattr(mae_result, 'iloc'):
                mae_per_series[i] = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
            else:
                mae_per_series[i] = float(mae_result)
        else:
            mae_per_series[i] = np.nan
    
    # Calculate overall MAE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        mae_metric = MeanAbsoluteError()
        mae_result = mae_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(mae_result, 'iloc'):
            mae_overall = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
        else:
            mae_overall = float(mae_result)
    else:
        mae_overall = np.nan
    
    return mae_overall, mae_per_series


def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Mean Absolute Percentage Error (MAPE) between actual and predicted values.
    
    Uses sklearn.metrics.mean_absolute_percentage_error for MAPE calculation.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    mape_overall : float
        Overall MAPE averaged across all series and time periods
    mape_per_series : np.ndarray
        MAPE for each series (N,) or scalar if 1D input
    """
    if not _has_sklearn:
        raise ImportError(
            "sklearn is required for calculate_mape. "
            "Please install: pip install scikit-learn"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}",
            details=f"Shape mismatch: actual.shape={actual.shape}, predicted.shape={predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        mape_series = mean_absolute_percentage_error(actual_masked, predicted_masked)
        return mape_series, np.array([mape_series])
    
    # Handle 2D case
    T, N = actual.shape
    
    mape_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            mape_per_series[i] = mean_absolute_percentage_error(
                actual_series, predicted_series
            )
        else:
            mape_per_series[i] = np.nan
    
    # Calculate overall MAPE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        mape_overall = mean_absolute_percentage_error(actual_masked, predicted_masked)
    else:
        mape_overall = np.nan
    
    return mape_overall, mape_per_series


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate R² (coefficient of determination) between actual and predicted values.
    
    Uses sklearn.metrics.r2_score for R² calculation.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    r2_overall : float
        Overall R² averaged across all series and time periods
    r2_per_series : np.ndarray
        R² for each series (N,) or scalar if 1D input
    """
    if not _has_sklearn:
        raise ImportError(
            "sklearn is required for calculate_r2. "
            "Please install: pip install scikit-learn"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise DataValidationError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}",
            details=f"Shape mismatch: actual.shape={actual.shape}, predicted.shape={predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        r2_series = r2_score(actual_masked, predicted_masked)
        return r2_series, np.array([r2_series])
    
    # Handle 2D case
    T, N = actual.shape
    
    r2_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            r2_per_series[i] = r2_score(actual_series, predicted_series)
        else:
            r2_per_series[i] = np.nan
    
    # Calculate overall R²
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        r2_overall = r2_score(actual_masked, predicted_masked)
    else:
        r2_overall = np.nan
    
    return r2_overall, r2_per_series


__all__ = [
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_r2',
]

