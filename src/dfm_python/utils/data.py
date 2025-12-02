"""Data loading and transformation utilities for DFM estimation.

This module provides functions for reading, sorting, transforming, and loading time series data
for Dynamic Factor Model estimation.
"""

import logging
from ..logger import get_logger

_logger = get_logger(__name__)
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict

import numpy as np
import polars as pl
from datetime import datetime
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config.schema import DFMConfig, SeriesConfig, BlockConfig
from ..utils.time import TimeIndex, parse_timestamp, to_python_datetime

logger = logging.getLogger(__name__)


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, TimeIndex, List[str]]:
    """Read time series data from file.
    
    Supports tabular data formats with dates and series values.
    Automatically detects date column and handles various data layouts.
    
    Expected format:
    - First column: Date (YYYY-MM-DD format or datetime-parseable)
    - Subsequent columns: Series data (one column per series)
    - Header row: Series IDs
    
    Alternative format (long format):
    - Metadata columns: series_id, series_name, etc.
    - Date columns: Starting from first date column
    - One row per series, dates as columns
    
    Parameters
    ----------
    datafile : str or Path
        Path to data file
        
    Returns
    -------
    Z : np.ndarray
        Data matrix (T x N) with T time periods and N series
    Time : TimeIndex
        Time index for the data
    mnemonics : List[str]
        Series identifiers (column names)
    """
    datafile = Path(datafile)
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read data file
    try:
        # Use infer_schema_length=None to infer all rows, and try_parse_dates=False
        # to avoid parsing issues with mixed numeric/string columns
        df = pl.read_csv(datafile, infer_schema_length=None, try_parse_dates=False)
    except Exception as e:
        raise ValueError(f"Failed to read data file {datafile}: {e}")
    
    # Check if first column is a date column or metadata
    first_col = df.columns[0]
    
    # Try to parse first column as date
    try:
        first_val = df[first_col][0]
        if first_val is None:
            is_date_first = False
        else:
            parse_timestamp(str(first_val))
            is_date_first = True
    except (ValueError, TypeError, IndexError):
        is_date_first = False
    
    # If first column is not a date, check if data is in "long" format (one row per series)
    # Skip this check if first column is integer (likely date_id) - treat as standard format
    if not is_date_first:
        first_col_type = df[first_col].dtype
        is_integer_id = first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
        
        # Only check for long format if first column is not an integer ID
        if not is_integer_id:
            # Look for date columns (starting from a certain column)
            date_cols = []
            for col in df.columns:
                try:
                    parse_timestamp(str(df[col][0]))
                    date_cols.append(col)
                except (ValueError, TypeError):
                    pass
            
            if len(date_cols) > 0:
                # Long format: transpose and use first date column as index
                first_date_col = date_cols[0]
                date_col_idx = df.columns.index(first_date_col)
                date_cols_all = df.columns[date_col_idx:]
                
                # Extract dates from column names (they are dates in long format)
                dates = []
                for col in date_cols_all:
                    try:
                        dates.append(parse_timestamp(col))
                    except (ValueError, TypeError):
                        # Skip invalid date columns
                        pass
                
                # Transpose: rows become series, columns become time
                # Select date columns and transpose
                date_data = df.select(date_cols_all)
                Z = date_data.to_numpy().T.astype(float)
                Time = TimeIndex(dates)
                mnemonics = df[first_col].to_list() if first_col in df.columns else [f"series_{i}" for i in range(len(df))]
                
                return Z, Time, mnemonics
    
    # Standard format: first column is date, rest are series
    # Handle integer date_id columns (treat as sequential time index)
    try:
        # Check if first column is integer (date_id format)
        first_col_type = df[first_col].dtype
        if first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
            # Integer date_id: use as sequential index, generate synthetic dates
            n_periods = len(df)
            from datetime import datetime, timedelta
            # Start from a default date and increment by day
            start_date = datetime(2000, 1, 1)
            dates = [start_date + timedelta(days=int(df[first_col][i])) for i in range(n_periods)]
            Time = TimeIndex(dates)
        else:
            # Try to parse as date
            time_series = df[first_col].cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
            # If that fails, try other formats
            if time_series.null_count() > 0:
                # Try parsing as string first
                time_series = df[first_col].str.strptime(pl.Datetime, strict=False)
            Time = TimeIndex(time_series)
    except (ValueError, TypeError) as e:
        # If date parsing fails, treat first column as integer date_id
        try:
            first_col_type = df[first_col].dtype
            if first_col_type in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                n_periods = len(df)
                from datetime import datetime, timedelta
                start_date = datetime(2000, 1, 1)
                dates = [start_date + timedelta(days=int(df[first_col][i])) for i in range(n_periods)]
                Time = TimeIndex(dates)
            else:
                raise ValueError(f"Failed to parse date column '{first_col}': {e}")
        except Exception:
            raise ValueError(f"Failed to parse date column '{first_col}': {e}")
    
    # Extract series data (all columns except first)
    series_cols = [col for col in df.columns if col != first_col]
    series_data = df.select(series_cols)
    Z = series_data.to_numpy().astype(float)
    mnemonics = series_cols
    
    return Z, Time, mnemonics


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


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[datetime, str]] = None,
              sample_end: Optional[Union[datetime, str]] = None) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
    """Load time series data for DFM estimation.
    
    This function reads time series data and aligns it with the model configuration.
    The data is sorted to match the configuration order and validated against frequency constraints.
    
    Note: This function returns raw (untransformed) data. To apply transformations and
    standardization, use DFMScaler after loading the data.
    
    Data Format:
        - File-based: CSV format supported for convenience
        - Database-backed: Implement adapters that return (X, Time, Z) arrays
        
    Frequency Constraints:
        - Frequencies faster than the clock frequency are not supported
        - If any series violates this constraint, a ValueError is raised
        
    Parameters
    ----------
    datafile : str or Path
        Path to data file (CSV format supported)
    config : DFMConfig
        Model configuration object
    sample_start : datetime or str, optional
        Start date for sample (YYYY-MM-DD). If None, uses beginning of data.
        Data before this date will be dropped.
    sample_end : datetime or str, optional
        End date for sample (YYYY-MM-DD). If None, uses end of data.
        Data after this date will be dropped.
        
    Returns
    -------
    X : np.ndarray
        Raw data matrix (T x N), not transformed. Use DFMScaler to apply transformations.
    Time : TimeIndex
        Time index for the data (aligned to clock frequency)
    Z : np.ndarray
        Original untransformed data (T x N), same as X (for backward compatibility)
        
    Raises
    ------
    ValueError
        If any series has frequency faster than clock, or data format is invalid
    FileNotFoundError
        If datafile does not exist
    """
    from ..config.structure import FREQUENCY_HIERARCHY
    
    _logger.info('Loading data...')
    
    datafile_path = Path(datafile)
    if datafile_path.suffix.lower() != '.csv':
        _logger.warning(f"Data file extension is not .csv: {datafile_path.suffix}. Assuming CSV format.")
    
    # Read raw data
    Z, Time, Mnem = read_data(datafile_path)
    _logger.info(f"Read {Z.shape[0]} time periods, {Z.shape[1]} series from {datafile_path}")
    
    # Sort data to match config order
    Z, Mnem = sort_data(Z, Mnem, config)
    _logger.info(f"Sorted data to match configuration order")
    
    # Apply sample date filters
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = parse_timestamp(sample_start)
        mask = Time >= sample_start
        if isinstance(mask, pl.Series):
            mask = mask.to_numpy()
        Z = Z[mask]
        Time = Time.filter(mask) if hasattr(Time, 'filter') else Time[mask]
        _logger.info(f"Filtered to start date: {sample_start}")
    
    if sample_end is not None:
        if isinstance(sample_end, str):
            sample_end = parse_timestamp(sample_end)
        mask = Time <= sample_end
        if isinstance(mask, pl.Series):
            mask = mask.to_numpy()
        Z = Z[mask]
        Time = Time.filter(mask) if hasattr(Time, 'filter') else Time[mask]
        _logger.info(f"Filtered to end date: {sample_end}")
    
    # Return raw data (transformations should be applied using DFMScaler)
    X = Z
    _logger.info(f"Loaded data: {X.shape[0]} time periods, {X.shape[1]} series (raw, not transformed)")
    
    # Validate data quality
    # Note: DFMConfig always has 'clock' attribute, but use safe_get_attr for consistency
    from ..utils.helpers import safe_get_attr, get_frequencies_from_config, get_series_ids
    clock = safe_get_attr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    frequencies = get_frequencies_from_config(config)
    series_ids = get_series_ids(config)
    warnings_list = []
    
    for i, freq in enumerate(frequencies):
        if i >= X.shape[1]:
            continue
        
        series_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
        if series_hierarchy < clock_hierarchy:
            raise ValueError(
                f"Series '{series_ids[i]}' has frequency '{freq}' which is faster than clock '{clock}'. "
                f"Higher frequencies (daily, weekly) are not supported."
            )
        
        # Check for T < N condition (may cause numerical issues)
        valid_obs = np.sum(~np.isnan(X[:, i]))
        if valid_obs < X.shape[1]:
            warnings_list.append((series_ids[i], valid_obs, X.shape[1]))
    
    if len(warnings_list) > 0:
        for series_id, T_obs, N_total in warnings_list[:5]:
            _logger.warning(
                f"Series '{series_id}': T={T_obs} < N={N_total} (may cause numerical issues). "
                f"Suggested fix: increase sample size or reduce number of series."
            )
        if len(warnings_list) > 5:
            _logger.warning(f"... and {len(warnings_list) - 5} more series with T < N")
        
        warnings.warn(
            f"Insufficient data: {len(warnings_list)} series have T < N (time periods < number of series). "
            f"This may cause numerical issues. Suggested fix: increase sample size or reduce number of series. "
            f"See log for details.",
            UserWarning,
            stacklevel=2
        )
    
    # Validate extreme missing data (>90% missing per series)
    missing_ratios = np.sum(np.isnan(X), axis=0) / X.shape[0]
    extreme_missing_series = []
    for i, ratio in enumerate(missing_ratios):
        if ratio > 0.9:
            from ..utils.helpers import get_series_id_by_index
            series_id = get_series_id_by_index(config, i)
            extreme_missing_series.append((series_id, ratio))
    
    if len(extreme_missing_series) > 0:
        for series_id, ratio in extreme_missing_series[:5]:
            _logger.warning(
                f"Series '{series_id}' has {ratio:.1%} missing data (>90%). "
                f"This may cause estimation issues. Consider removing this series or increasing data coverage."
            )
        if len(extreme_missing_series) > 5:
            _logger.warning(f"... and {len(extreme_missing_series) - 5} more series with >90% missing data")
        
        warnings.warn(
            f"Extreme missing data detected: {len(extreme_missing_series)} series have >90% missing values. "
            f"Estimation may be unreliable. Consider removing these series or increasing data coverage. "
            f"See log for details.",
            UserWarning,
            stacklevel=2
        )
    
    return X, Time, Z


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


def rem_nans_spline_torch(X: torch.Tensor, method: int = 2, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if not TORCH_AVAILABLE:
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


def create_data_view(
    X: np.ndarray,
    Time: Union[TimeIndex, Any],
    Z: Optional[np.ndarray] = None,
    config: Optional[DFMConfig] = None,
    view_date: Union[datetime, str, None] = None,
    *,
    X_frame: Optional[pl.DataFrame] = None
) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
    """Create data view at a specific view date."""
    from ..utils.time import get_latest_time
    from ..utils.helpers import get_series_ids
    
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
    else:
        time_list = []
        for t in Time:
            if isinstance(t, datetime):
                time_list.append(t)
            elif hasattr(t, 'to_python'):
                time_list.append(t.to_python())
            else:
                time_list.append(parse_timestamp(t))
    
    # Build polars DataFrame reference
    try:
        series_ids = get_series_ids(config)
    except ValueError:
        series_ids = [f'series_{i}' for i in range(X.shape[1])]
    
    if X_frame is not None:
        df = X_frame.clone()
    else:
        df = pl.DataFrame(X, schema=series_ids[:X.shape[1]])
    df = df.with_columns(pl.Series('_view_time', time_list))
    
    # Track masks for applying to numpy fallbacks
    series_masks: Dict[int, np.ndarray] = {}
    
    for i, series_cfg in enumerate(config.series):
        if i >= df.width - 1:  # exclude time column
            continue
        release_offset = getattr(series_cfg, 'release_date', None)
        if release_offset is None:
            continue
        
        release_dates = [calculate_release_date(release_offset, t) for t in time_list]
        mask = np.array([view_date >= rd for rd in release_dates], dtype=bool)
        series_masks[i] = mask
        
        mask_col = pl.Series('_mask', mask)
        df = df.with_columns(mask_col)
        df = df.with_columns(
            pl.when(pl.col('_mask'))
            .then(pl.col(series_ids[i]))
            .otherwise(pl.lit(None))
            .alias(series_ids[i])
        ).drop('_mask')
    
    df_view = df.drop('_view_time')
    X_view = df_view.to_numpy()
    
    if Z is not None:
        Z_view = Z.copy()
        for i, mask in series_masks.items():
            Z_view[~mask, i] = np.nan
    else:
        Z_view = None
    
    return X_view, Time, Z_view


# DataView class has been moved to nowcast/dataview.py
# Import it from there: from ..nowcast.dataview import DataView

# ============================================================================


# ============================================================================
# DataLoader class (merged from data_loader.py)
# ============================================================================


class DataLoader:
    """Encapsulates data loading, transformation, and preprocessing operations.
    
    This class provides a unified interface for loading time series data,
    applying transformations, and preparing data for DFM estimation.
    
    Parameters
    ----------
    config : DFMConfig
        Model configuration object
    """
    
    def __init__(self, config: DFMConfig):
        """Initialize DataLoader with configuration.
        
        Parameters
        ----------
        config : DFMConfig
            Model configuration object
        """
        self.config = config
        self._data: Optional[np.ndarray] = None
        self._time: Optional[TimeIndex] = None
        self._original_data: Optional[np.ndarray] = None
    
    def load(
        self,
        datafile: Union[str, Path],
        sample_start: Optional[Union[datetime, str]] = None,
        sample_end: Optional[Union[datetime, str]] = None
    ) -> 'DataLoader':
        """Load and transform time series data.
        
        Parameters
        ----------
        datafile : str or Path
            Path to data file (CSV format supported)
        sample_start : datetime or str, optional
            Start date for sample (YYYY-MM-DD). If None, uses beginning of data.
        sample_end : datetime or str, optional
            End date for sample (YYYY-MM-DD). If None, uses end of data.
            
        Returns
        -------
        DataLoader
            Self for method chaining
        """
        from .loader import load_data
        
        self._data, self._time, self._original_data = load_data(
            datafile, self.config, sample_start, sample_end
        )
        return self
    
    def load_from_array(
        self,
        data: np.ndarray,
        time: Optional[TimeIndex] = None
    ) -> 'DataLoader':
        """Load data from numpy array.
        
        Parameters
        ----------
        data : np.ndarray
            Raw data matrix (T x N)
        time : TimeIndex, optional
            Time index. If None, generates default time index.
            
        Returns
        -------
        DataLoader
            Self for method chaining
        """
        from ..utils.time import datetime_range, clock_to_datetime_freq
        from ..utils.helpers import get_clock_frequency
        from datetime import datetime
        
        self._original_data = data.copy()
        
        # Generate time index if not provided
        if time is None:
            clock = get_clock_frequency(self.config, 'm')
            datetime_freq = clock_to_datetime_freq(clock)
            start_date = datetime(2000, 1, 1)
            self._time = TimeIndex(datetime_range(start=start_date, periods=len(data), freq=datetime_freq))
        else:
            self._time = time
        
        # Use raw data (transformations should be applied using DFMScaler)
        self._data = data
        self._original_data = data
        
        return self
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get transformed data matrix (T x N)."""
        return self._data
    
    @property
    def time(self) -> Optional[TimeIndex]:
        """Get time index."""
        return self._time
    
    @property
    def original_data(self) -> Optional[np.ndarray]:
        """Get original (untransformed) data matrix."""
        return self._original_data
    
    def get_data_tuple(self) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
        """Get data as tuple (X, Time, Z).
        
        Returns
        -------
        Tuple[np.ndarray, TimeIndex, np.ndarray]
            (transformed_data, time_index, original_data)
            
        Raises
        ------
        ValueError
            If data has not been loaded
        """
        from ..utils.helpers import DFMDataError
        
        if self._data is None:
            raise DFMDataError("Data not loaded. Call load() or load_from_array() first.")
        
        return self._data, self._time, self._original_data
