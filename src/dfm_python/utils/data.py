"""Data loading and transformation utilities for DFM estimation.

This module provides functions for reading, sorting, transforming, and loading time series data
for Dynamic Factor Model estimation.
"""

from typing import List, Optional, Tuple, Union, Any, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
from datetime import datetime

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch.nn.functional as F
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    F = None

from ..logger import get_logger
from ..config.schema import DFMConfig, SeriesConfig
from ..utils.time import TimeIndex, parse_timestamp, to_python_datetime

_logger = get_logger(__name__)


def sort_data(Z: np.ndarray, Mnem: List[str], config: DFMConfig) -> Tuple[np.ndarray, List[str]]:
    """Sort data columns to match configuration order.
    
    Parameters
    ----------
    Z : np.ndarray
        Data matrix (T x N)
    Mnem : List[str]
        Series identifiers (mnemonics) from data file
    config : DFMConfig
        Model configuration with series order
        
    Returns
    -------
    Z_sorted : np.ndarray
        Sorted data matrix (T x N)
    Mnem_sorted : List[str]
        Sorted series identifiers
    """
    from ..utils.helpers import get_series_ids
    series_ids = get_series_ids(config)
    
    # Create mapping from series_id to index in data
    mnem_to_idx = {m: i for i, m in enumerate(Mnem)}
    
    # Find permutation
    perm = []
    Mnem_filt = []
    for sid in series_ids:
        if sid in mnem_to_idx:
            perm.append(mnem_to_idx[sid])
            Mnem_filt.append(sid)
        else:
            _logger.warning(f"Series '{sid}' from config not found in data")
    
    if len(perm) == 0:
        raise ValueError("No matching series found between config and data")
    
    # Apply permutation
    Z_filt = Z[:, perm]
    
    return Z_filt, Mnem_filt


def rem_nans_spline(X: np.ndarray, method: int = 2, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Treat NaNs in dataset for DFM estimation using standard interpolation methods.
    
    This function implements standard econometric practice for handling missing data
    in time series, following the approach used in FRBNY Nowcasting Model and similar
    DFM implementations. The Kalman Filter in the DFM will handle remaining missing
    values during estimation (see miss_data function in kalman.py).
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (T x N)
    method : int
        Missing data handling method:
        - 1: Replace all missing values using spline interpolation
        - 2: Remove >80% NaN rows, then fill (default, recommended)
        - 3: Only remove all-NaN rows
        - 4: Remove all-NaN rows, then fill
        - 5: Fill missing values
    k : int
        Spline interpolation order (default: 3 for cubic spline)
        
    Returns
    -------
    X : np.ndarray
        Data with NaNs treated
    indNaN : np.ndarray
        Boolean mask indicating original NaN positions
        
    Notes
    -----
    This preprocessing step is followed by Kalman Filter-based missing data handling
    during DFM estimation, which is the standard approach in state-space models.
    See Mariano & Murasawa (2003) and Harvey (1989) for theoretical background.
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import lfilter
    
    # Ensure X is a numeric numpy array
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        # Convert non-numeric types to numeric, handling errors
        try:
            X = X.astype(np.float64)
        except (ValueError, TypeError):
            # If conversion fails, try using pandas for better type handling
            try:
                import pandas as pd
                X_df = pd.DataFrame(X)
                X = X_df.select_dtypes(include=[np.number]).to_numpy()
                if X.size == 0:
                    raise ValueError("Input data contains no numeric columns")
                # If shape changed, we need to handle it
                if X.shape != X_df.shape:
                    _logger.warning(f"Non-numeric columns removed. Shape changed from {X_df.shape} to {X.shape}")
            except ImportError:
                raise TypeError(f"Cannot convert input data to numeric. dtype: {X.dtype}")
    
    T, N = X.shape
    indNaN = np.isnan(X)
    
    def _remove_leading_trailing(threshold: float):
        """Remove rows with NaN count above threshold."""
        rem = np.sum(indNaN, axis=1) > (N * threshold if threshold < 1 else threshold)
        nan_lead = np.cumsum(rem) == np.arange(1, T + 1)
        nan_end = np.cumsum(rem[::-1]) == np.arange(1, T + 1)[::-1]
        return ~(nan_lead | nan_end)
    
    def _fill_missing(x: np.ndarray, mask: np.ndarray):
        """Fill missing values using spline interpolation and moving average."""
        if len(mask) != len(x):
            mask = mask[:len(x)]
        
        non_nan = np.where(~mask)[0]
        if len(non_nan) < 2:
            return x
        
        x_filled = x.copy()
        if non_nan[-1] >= len(x):
            non_nan = non_nan[non_nan < len(x)]
        if len(non_nan) < 2:
            return x
        
        x_filled[non_nan[0]:non_nan[-1]+1] = CubicSpline(non_nan, x[non_nan])(np.arange(non_nan[0], min(non_nan[-1]+1, len(x))))
        x_filled[mask[:len(x_filled)]] = np.nanmedian(x_filled)
        
        # Moving average filter
        pad = np.concatenate([np.full(k, x_filled[0]), x_filled, np.full(k, x_filled[-1])])
        ma = lfilter(np.ones(2*k+1)/(2*k+1), 1, pad)[2*k+1:]
        if len(ma) == len(x_filled):
            x_filled[mask[:len(x_filled)]] = ma[mask[:len(x_filled)]]
        return x_filled
    
    if method == 1:
        # Replace all missing values
        for i in range(N):
            mask = indNaN[:, i]
            x = X[:, i].copy()
            x[mask] = np.nanmedian(x)
            pad = np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])])
            ma = lfilter(np.ones(2*k+1)/(2*k+1), 1, pad)[2*k+1:]
            x[mask] = ma[mask]
            X[:, i] = x
    
    elif method == 2:
        # Remove >80% NaN rows, then fill
        mask = _remove_leading_trailing(0.8)
        X = X[mask]
        indNaN = np.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 3:
        # Only remove all-NaN rows
        mask = _remove_leading_trailing(N)
        X = X[mask]
        indNaN = np.isnan(X)
    
    elif method == 4:
        # Remove all-NaN rows, then fill
        mask = _remove_leading_trailing(N)
        X = X[mask]
        indNaN = np.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 5:
        # Fill missing values
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    return X, indNaN


def rem_nans_spline_torch(X: "torch.Tensor", method: int = 2, k: int = 3) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """PyTorch version of rem_nans_spline for GPU acceleration.
    
    Treat NaNs in dataset for DFM estimation using standard interpolation methods.
    This is a GPU-accelerated version that stays entirely on GPU, avoiding CPU transfers.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data matrix (T x N) on GPU or CPU
    method : int
        Missing data handling method:
        - 1: Replace all missing values using spline interpolation
        - 2: Remove >80% NaN rows, then fill (default, recommended)
        - 3: Only remove all-NaN rows
        - 4: Remove all-NaN rows, then fill
        - 5: Fill missing values
    k : int
        Spline interpolation order (default: 3 for cubic spline)
        
    Returns
    -------
    X : torch.Tensor
        Data with NaNs treated (same device and dtype as input)
    indNaN : torch.Tensor
        Boolean mask indicating original NaN positions (same device as input)
        
    Notes
    -----
    This function implements the same logic as rem_nans_spline() but uses PyTorch
    operations to stay on GPU. All operations preserve the input device and dtype.
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for rem_nans_spline_torch")
    
    device = X.device
    dtype = X.dtype
    T, N = X.shape
    indNaN = torch.isnan(X)
    
    def _remove_leading_trailing(threshold: float):
        """Remove rows with NaN count above threshold."""
        nan_count = torch.sum(indNaN.float(), dim=1)  # (T,)
        if threshold < 1:
            threshold_count = N * threshold
        else:
            threshold_count = threshold
        
        rem = nan_count > threshold_count
        # Leading NaNs: cumulative sum equals position
        nan_lead = torch.cumsum(rem.float(), dim=0) == torch.arange(1, T + 1, device=device, dtype=dtype)
        # Trailing NaNs: reverse cumulative sum
        nan_end = torch.flip(torch.cumsum(torch.flip(rem.float(), dims=[0]), dim=0), dims=[0]) == torch.arange(1, T + 1, device=device, dtype=dtype)
        return ~(nan_lead | nan_end)
    
    def _linear_interpolate(x: torch.Tensor, non_nan_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """Linear interpolation using PyTorch.
        
        This approximates spline interpolation using piecewise linear interpolation.
        """
        if len(non_nan_idx) < 2:
            return x
        
        # Get non-NaN values and their indices
        x_vals = x[non_nan_idx]
        x_idx = non_nan_idx.float()
        
        # Create result tensor
        result = x.clone()
        
        # For indices before first non-NaN, use first value
        mask_before = target_idx < x_idx[0]
        if mask_before.any():
            result[target_idx[mask_before].long()] = x_vals[0]
        
        # For indices after last non-NaN, use last value
        mask_after = target_idx > x_idx[-1]
        if mask_after.any():
            result[target_idx[mask_after].long()] = x_vals[-1]
        
        # For indices in between, use linear interpolation
        mask_middle = (target_idx >= x_idx[0]) & (target_idx <= x_idx[-1])
        if mask_middle.any():
            target_positions = target_idx[mask_middle]
            
            # Find the indices to interpolate between
            # For each target position, find the two surrounding non-NaN indices
            interpolated_vals = torch.zeros_like(target_positions)
            
            for i, pos in enumerate(target_positions):
                # Find the two surrounding indices
                # Find the largest index <= pos
                lower_idx = torch.where(x_idx <= pos)[0]
                if len(lower_idx) > 0:
                    lower_idx = lower_idx[-1]
                    upper_idx = lower_idx + 1 if lower_idx + 1 < len(x_idx) else lower_idx
                else:
                    lower_idx = 0
                    upper_idx = 0
                
                if lower_idx == upper_idx:
                    interpolated_vals[i] = x_vals[lower_idx]
                else:
                    # Linear interpolation
                    x_lower = x_idx[lower_idx]
                    x_upper = x_idx[upper_idx]
                    y_lower = x_vals[lower_idx]
                    y_upper = x_vals[upper_idx]
                    
                    if abs(x_upper - x_lower) < 1e-8:
                        interpolated_vals[i] = y_lower
                    else:
                        alpha = (pos - x_lower) / (x_upper - x_lower)
                        interpolated_vals[i] = y_lower + alpha * (y_upper - y_lower)
            
            result[target_idx[mask_middle].long()] = interpolated_vals
        
        return result
    
    def _fill_missing(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fill missing values using spline interpolation and moving average."""
        if len(mask) != len(x):
            mask = mask[:len(x)]
        
        non_nan = torch.where(~mask)[0]
        if len(non_nan) < 2:
            # If too few non-NaN values, fill with median
            median_val = torch.nanmedian(x)
            return torch.where(mask, median_val, x)
        
        x_filled = x.clone()
        
        # Get target indices for interpolation
        target_start = max(0, int(non_nan[0].item()))
        target_end = min(len(x), int(non_nan[-1].item()) + 1)
        target_idx = torch.arange(target_start, target_end, device=device, dtype=dtype)
        
        if len(target_idx) > 0:
            # Interpolate missing values
            interpolated = _linear_interpolate(x, non_nan, target_idx)
            x_filled[target_idx.long()] = interpolated[target_idx.long()]
        
        # Fill remaining NaNs with median
        remaining_nan = torch.isnan(x_filled) & mask
        if remaining_nan.any():
            median_val = torch.nanmedian(x_filled)
            x_filled[remaining_nan] = median_val
        
        # Moving average filter using conv1d
        # Pad the signal
        pad_val = x_filled[0] if len(x_filled) > 0 else torch.tensor(0.0, device=device, dtype=dtype)
        pad_start = torch.full((k,), pad_val, device=device, dtype=dtype)
        pad_end = torch.full((k,), x_filled[-1] if len(x_filled) > 0 else pad_val, device=device, dtype=dtype)
        padded = torch.cat([pad_start, x_filled, pad_end])
        
        # Create moving average kernel
        kernel_size = 2 * k + 1
        kernel = torch.ones(1, 1, kernel_size, device=device, dtype=dtype) / kernel_size
        
        # Apply conv1d (need to add batch and channel dimensions)
        padded_4d = padded.unsqueeze(0).unsqueeze(0)  # (1, 1, T+2k)
        ma_4d = F.conv1d(padded_4d, kernel, padding=0)  # (1, 1, T)
        ma = ma_4d.squeeze(0).squeeze(0)  # (T,)
        
        # Apply moving average to originally missing positions
        if len(ma) == len(x_filled):
            x_filled[mask[:len(x_filled)]] = ma[mask[:len(x_filled)]]
        
        return x_filled
    
    if method == 1:
        # Replace all missing values
        for i in range(N):
            mask = indNaN[:, i]
            x = X[:, i].clone()
            median_val = torch.nanmedian(x)
            x[mask] = median_val
            
            # Moving average
            pad_val = x[0] if len(x) > 0 else torch.tensor(0.0, device=device, dtype=dtype)
            pad_start = torch.full((k,), pad_val, device=device, dtype=dtype)
            pad_end = torch.full((k,), x[-1] if len(x) > 0 else pad_val, device=device, dtype=dtype)
            padded = torch.cat([pad_start, x, pad_end])
            
            kernel_size = 2 * k + 1
            kernel = torch.ones(1, 1, kernel_size, device=device, dtype=dtype) / kernel_size
            padded_4d = padded.unsqueeze(0).unsqueeze(0)
            ma_4d = F.conv1d(padded_4d, kernel, padding=0)
            ma = ma_4d.squeeze(0).squeeze(0)
            
            if len(ma) == len(x):
                x[mask] = ma[mask]
            X[:, i] = x
    
    elif method == 2:
        # Remove >80% NaN rows, then fill
        mask = _remove_leading_trailing(0.8)
        X = X[mask]
        indNaN = torch.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 3:
        # Only remove all-NaN rows
        mask = _remove_leading_trailing(float(N))
        X = X[mask]
        indNaN = torch.isnan(X)
    
    elif method == 4:
        # Remove all-NaN rows, then fill
        mask = _remove_leading_trailing(float(N))
        X = X[mask]
        indNaN = torch.isnan(X)
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    elif method == 5:
        # Fill missing values
        for i in range(N):
            X[:, i] = _fill_missing(X[:, i], indNaN[:, i])
    
    return X, indNaN


def calculate_release_date(release_date: int, period: datetime) -> datetime:
    """Calculate release date relative to the period."""
    from calendar import monthrange
    
    if release_date is None:
        return period
    
    if release_date >= 1:
        # Day of current month
        last_day = monthrange(period.year, period.month)[1]
        day = min(release_date, last_day)
        return datetime(period.year, period.month, day)
    
    # release_date < 0 => days before end of previous month
    if period.month == 1:
        prev_year = period.year - 1
        prev_month = 12
    else:
        prev_year = period.year
        prev_month = period.month - 1
    last_day_prev_month = monthrange(prev_year, prev_month)[1]
    day = last_day_prev_month + release_date + 1
    day = max(1, day)
    return datetime(prev_year, prev_month, day)


def validate_no_nan(X: Union[np.ndarray, "torch.Tensor", pd.DataFrame], name: str = "data") -> None:
    """Validate that data contains no NaN or Inf values.
    
    This function ensures that dfm-python receives only fully preprocessed data
    without any missing values. NaN values must be imputed before passing data
    to dfm-python.
    
    Parameters
    ----------
    X : np.ndarray, torch.Tensor, or pd.DataFrame
        Data to validate
    name : str, default "data"
        Name of the data for error messages
        
    Raises
    ------
    ValueError
        If data contains NaN or Inf values
        
    Notes
    -----
    dfm-python requires fully preprocessed data with no missing values.
    Users must impute missing values before passing data to dfm-python.
    This validation ensures data quality and prevents numerical issues during training.
    """
    import numpy as np
    
    # Convert to numpy for validation
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    elif hasattr(X, 'cpu'):  # torch.Tensor
        X_array = X.cpu().numpy()
    else:
        X_array = np.asarray(X)
    
    # Check for NaN
    nan_count = np.sum(np.isnan(X_array))
    if nan_count > 0:
        nan_ratio = nan_count / X_array.size
        raise ValueError(
            f"{name} contains {nan_count} NaN values ({nan_ratio:.2%} of all values). "
            f"dfm-python requires fully preprocessed data with no missing values. "
            f"Please impute missing values before passing data to dfm-python. "
            f"You can use rem_nans_spline() or other imputation methods."
        )
    
    # Check for Inf
    inf_count = np.sum(np.isinf(X_array))
    if inf_count > 0:
        inf_ratio = inf_count / X_array.size
        raise ValueError(
            f"{name} contains {inf_count} Inf values ({inf_ratio:.2%} of all values). "
            f"dfm-python requires fully preprocessed data with no infinite values. "
            f"Please handle infinite values before passing data to dfm-python."
        )


def create_data_view(
    X: np.ndarray,
    Time: Union[TimeIndex, Any],
    Z: Optional[np.ndarray] = None,
    config: Optional[DFMConfig] = None,
    view_date: Union[datetime, str, None] = None,
    *,
    X_frame: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
    """Create data view at a specific view date."""
    from ..utils.time import get_latest_time
    from ..utils.helpers import get_series_ids
    
    if Time is None:
        raise ValueError("Time index is required for create_data_view")
    
    if isinstance(view_date, str):
        view_date = parse_timestamp(view_date)
    elif view_date is None:
        view_date = get_latest_time(Time)
    
    if not isinstance(view_date, datetime):
        view_date = parse_timestamp(view_date)
    
    if config is None or not hasattr(config, 'series') or not config.series:
        return X.copy(), Time, Z.copy() if Z is not None else None
    
    # Prepare time list
    if isinstance(Time, TimeIndex):
        time_list = [to_python_datetime(t) for t in Time]
    elif Time is None:
        raise ValueError("Time index is required for create_data_view")
    else:
        time_list = []
        for t in Time:
            if isinstance(t, datetime):
                time_list.append(t)
            elif isinstance(t, pd.Timestamp):
                time_list.append(t.to_pydatetime())
            elif hasattr(t, 'to_pydatetime'):
                time_list.append(t.to_pydatetime())
            elif hasattr(t, 'to_python'):
                time_list.append(t.to_python())
            else:
                time_list.append(parse_timestamp(t))
    
    # Build pandas DataFrame reference
    try:
        series_ids = get_series_ids(config)
    except ValueError:
        series_ids = [f'series_{i}' for i in range(X.shape[1])]
    
    if X_frame is not None:
        df = X_frame.copy()
    else:
        df = pd.DataFrame(X, columns=series_ids[:X.shape[1]])
    df['_view_time'] = time_list
    
    # Track masks for applying to numpy fallbacks
    series_masks: Dict[int, np.ndarray] = {}
    
    for i, series_cfg in enumerate(config.series):
        if i >= len(df.columns) - 1:  # exclude time column
            continue
        release_offset = getattr(series_cfg, 'release_date', None)
        if release_offset is None:
            continue
        
        release_dates = [calculate_release_date(release_offset, t) for t in time_list]
        mask = np.array([view_date >= rd for rd in release_dates], dtype=bool)
        series_masks[i] = mask
        
        # Apply mask using pandas where
        df[series_ids[i]] = df[series_ids[i]].where(mask, None)
    
    df_view = df.drop(columns=['_view_time'])
    X_view = df_view.to_numpy()
    
    if Z is not None:
        Z_view = Z.copy()
        for i, mask in series_masks.items():
            Z_view[~mask, i] = np.nan
    else:
        Z_view = None
    
    return X_view, Time, Z_view


