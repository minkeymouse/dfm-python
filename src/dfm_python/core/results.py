from typing import Optional, Tuple
import numpy as np


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    RMSE is calculated as: sqrt(mean((actual - predicted)^2))
    
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
    - Used in diagnostics and model evaluation
    - Automatically handles missing data by excluding NaN values
    """
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(f"actual and predicted must have same shape, got {actual.shape} and {predicted.shape}")
    
    # Create mask for valid values
    if mask is None:
        # Use non-NaN values in both actual and predicted
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        # Combine user mask with finite check
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Calculate squared errors
    errors_sq = (actual - predicted) ** 2
    
    # Handle 1D case (single series)
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        rmse_series = np.sqrt(np.mean(errors_sq[mask]))
        return rmse_series, np.array([rmse_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    
    # Calculate RMSE per series
    rmse_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            rmse_per_series[i] = np.sqrt(np.mean(errors_sq[series_mask, i]))
        else:
            rmse_per_series[i] = np.nan
    
    # Calculate overall RMSE (average across all valid observations)
    if np.any(mask):
        rmse_overall = np.sqrt(np.mean(errors_sq[mask]))
    else:
        rmse_overall = np.nan
    
    return rmse_overall, rmse_per_series
