"""Data transformation utilities for DFM estimation.

This module provides functions for transforming time series data according to
configuration specifications, including differences, percent changes, and log
transformations.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from ..config import DFMConfig
from ..utils.aggregation import FREQUENCY_HIERARCHY


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Transform a single time series according to formula.
    
    Parameters
    ----------
    Z : np.ndarray
        Raw time series (T,)
    formula : str
        Transformation code (lin, chg, pch, etc.)
    freq : str
        Frequency code (m, q, sa, a)
    step : int
        Number of base periods per observation (1 for monthly, 3 for quarterly, etc.)
        
    Returns
    -------
    X : np.ndarray
        Transformed series (may be shorter than Z due to differencing)
    """
    T = len(Z)
    X = np.full(T, np.nan)
    
    if formula == 'lin':
        X[:] = Z
    elif formula == 'chg':
        # First difference
        if T > step:
            X[step:] = Z[step:] - Z[:-step]
    elif formula == 'ch1':
        # Year-over-year difference (12 for monthly, 4 for quarterly)
        year_step = 12 if freq == 'm' else (4 if freq == 'q' else (2 if freq == 'sa' else 1))
        if T > year_step:
            X[year_step:] = Z[year_step:] - Z[:-year_step]
    elif formula == 'pch':
        # Percent change
        if T > step:
            X[step:] = 100.0 * (Z[step:] - Z[:-step]) / np.abs(Z[:-step] + 1e-10)
    elif formula == 'pc1':
        # Year-over-year percent change
        year_step = 12 if freq == 'm' else (4 if freq == 'q' else (2 if freq == 'sa' else 1))
        if T > year_step:
            X[year_step:] = 100.0 * (Z[year_step:] - Z[:-year_step]) / np.abs(Z[:-year_step] + 1e-10)
    elif formula == 'pca':
        # Percent change annualized
        if T > step:
            annual_factor = 12.0 / step if freq == 'm' else (4.0 / step if freq == 'q' else 1.0)
            X[step:] = annual_factor * 100.0 * (Z[step:] - Z[:-step]) / np.abs(Z[:-step] + 1e-10)
    elif formula == 'cch':
        # Continuously compounded rate of change
        if T > step:
            X[step:] = 100.0 * (np.log(np.abs(Z[step:]) + 1e-10) - np.log(np.abs(Z[:-step]) + 1e-10))
    elif formula == 'cca':
        # Continuously compounded annual rate of change
        if T > step:
            annual_factor = 12.0 / step if freq == 'm' else (4.0 / step if freq == 'q' else 1.0)
            X[step:] = annual_factor * 100.0 * (np.log(np.abs(Z[step:]) + 1e-10) - np.log(np.abs(Z[:-step]) + 1e-10))
    elif formula == 'log':
        # Natural log
        X[:] = np.log(np.abs(Z) + 1e-10)
    else:
        X[:] = Z
    
    return X


def transform_data(Z: np.ndarray, Time: pd.DatetimeIndex, config: DFMConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Transform each data series according to configuration.
    
    Applies the specified transformation formula to each series based on its
    frequency and transformation type. Handles mixed-frequency data by
    applying transformations at the appropriate observation intervals.
    
    Supported frequencies: monthly (m), quarterly (q), semi-annual (sa), annual (a).
    Frequencies faster than the clock frequency are not supported.
    
    Parameters
    ----------
    Z : np.ndarray
        Raw data matrix (T x N)
    Time : pd.DatetimeIndex
        Time index for the data
    config : DFMConfig
        Model configuration with transformation specifications
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : pd.DatetimeIndex
        Time index (may be truncated after transformation)
    Z : np.ndarray
        Original data (may be truncated to match X)
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Validate frequencies - reject higher frequencies than clock
    clock = getattr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    frequencies = config.get_frequencies()
    series_ids = config.get_series_ids()
    for i, freq in enumerate(frequencies):
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
        if freq_hierarchy < clock_hierarchy:
            raise ValueError(
                f"Series '{series_ids[i]}' has frequency '{freq}' which is faster than clock '{clock}'. "
                f"Higher frequencies (daily, weekly) are not supported. "
                f"Please use monthly, quarterly, semi-annual, or annual frequencies only."
            )
    
    # Frequency to step mapping (step = number of base periods per observation)
    freq_to_step = {'m': 1, 'q': 3, 'sa': 6, 'a': 12}
    
    transformations = [s.transformation for s in config.series] if hasattr(config, 'series') else ['lin'] * N
    
    for i in range(N):
        freq = frequencies[i] if i < len(frequencies) else clock
        step = freq_to_step.get(freq, 1)
        formula = transformations[i] if i < len(transformations) else 'lin'
        X[:, i] = _transform_series(Z[:, i], formula, freq, step)
    
    # Remove leading NaN rows (from differencing)
    drop = 0
    for t in range(T):
        if np.all(np.isnan(X[t, :])):
            drop += 1
        else:
            break
    
    if T > drop:
        return X[drop:], Time[drop:], Z[drop:]
    return X, Time, Z
