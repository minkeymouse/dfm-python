"""Utility functions for data preprocessing and summary statistics.

This module provides essential data handling utilities for Dynamic Factor Models:
- Missing value treatment via spline interpolation
- Data summary and visualization
- Robust handling of edge cases (all-NaN series, constant series, etc.)

The spline interpolation methods follow standard econometric practice for handling
missing data in time series, with options for different treatment strategies
depending on data characteristics.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter
from typing import Tuple, Optional
import logging

# Set up logger
_logger = logging.getLogger(__name__)


def rem_nans_spline(X: np.ndarray, method: int = 2, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Treat NaNs in dataset for DFM estimation."""
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
        # Ensure mask matches x length
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


def summarize(X: np.ndarray, Time, config, vintage: Optional[str] = None) -> None:
    """Display data summary table."""
    T, N = X.shape
    
    print("\n\n\nTable 2: Data Summary")
    print(f"N = {N:4d} data series")
    
    try:
        time_start = Time[0].strftime('%Y-%m-%d') if hasattr(Time[0], 'strftime') else str(Time[0])
        time_end = Time[-1].strftime('%Y-%m-%d') if hasattr(Time[-1], 'strftime') else str(Time[-1])
    except (AttributeError, ValueError, TypeError) as e:
        _logger.debug(f"summarize: Failed to format time range: {type(e).__name__}: {str(e)}")
        time_start = time_end = "N/A"
    
    print(f"T = {T:4d} observations from {time_start:10s} to {time_end:10s}\n")
    print(f"{'Data Series':30s} | {'Observations':17s}    {'Units':12s}    "
          f"{'Frequency':10s}    {'Mean':8s}    {'Std. Dev.':8s}    {'Min':8s}    {'Max':8s}")
    print("-" * 130)
    
    for i in range(N):
        t_obs = ~np.isnan(X[:, i])
        obs_idx = np.where(t_obs)[0]
        
        name = config.SeriesName[i][:27] + "..." if len(config.SeriesName[i]) > 30 else config.SeriesName[i]
        sid = f"[{config.SeriesID[i][:25]}...]" if len(config.SeriesID[i]) > 28 else f"[{config.SeriesID[i]}]"
        
        freq = 'Monthly' if config.Frequency[i] == 'm' else 'Quarterly'
        trans = config.Transformation[i]
        units_t = 'MoM%' if trans == 'pch' and config.Frequency[i] == 'm' else \
                  'QoQ% AR' if trans == 'pca' and config.Frequency[i] == 'q' else \
                  config.UnitsTransformed[i]
        
        date_range = "N/A"
        if len(obs_idx) > 0:
            try:
                fmt = '%b %Y' if config.Frequency[i] == 'm' else '%Y-%m'
                # Use iloc for positional indexing
                if hasattr(Time, 'iloc'):
                    date_range = f"{Time.iloc[obs_idx[0]].strftime(fmt)}-{Time.iloc[obs_idx[-1]].strftime(fmt)}"
                else:
                    date_range = f"{Time[obs_idx[0]].strftime(fmt)}-{Time[obs_idx[-1]].strftime(fmt)}"
            except (AttributeError, ValueError, TypeError) as e:
                # Fallback to positional indexing
                _logger.debug(f"summarize: Failed to format date range with strftime: {type(e).__name__}: {str(e)}")
                try:
                    if hasattr(Time, 'iloc'):
                        date_range = f"{Time.iloc[obs_idx[0]]}-{Time.iloc[obs_idx[-1]]}"
                    else:
                        date_range = f"{Time[obs_idx[0]]}-{Time[obs_idx[-1]]}"
                except (AttributeError, ValueError, TypeError, IndexError) as e2:
                    _logger.debug(f"summarize: Failed to format date range fallback: {type(e2).__name__}: {str(e2)}")
                    date_range = "N/A"
        
        y = X[t_obs, i]
        if len(y) == 0:
            stats = (np.nan, np.nan, np.nan, np.nan)
        else:
            stats = np.nanmean(y), np.nanstd(y), np.nanmin(y) if len(y) > 0 else np.nan, np.nanmax(y) if len(y) > 0 else np.nan
        
        print(f"{name:30s} | {len(obs_idx):17d}    {units_t:12s}    {freq:10s}    "
              f"{stats[0]:8.1f}    {stats[1]:8.1f}    {stats[2]:8.1f}    {stats[3]:8.1f}")
        print(f"{sid:30s} | {date_range:17s}\n")
    
    print("\n\n\n")
