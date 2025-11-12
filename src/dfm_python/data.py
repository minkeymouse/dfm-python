"""Data loading and transformation utilities for DFM estimation.

This module provides functions for reading, sorting, transforming, and loading time series data
for Dynamic Factor Model estimation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DFMConfig, SeriesConfig, BlockConfig

logger = logging.getLogger(__name__)


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Read time series data from file.
    
    Supports tabular data formats with dates and series values.
    Automatically detects date column and handles various data layouts.
    
    Expected format:
    - First column: Date (YYYY-MM-DD format or pandas-parseable)
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
    Time : pd.DatetimeIndex
        Time index for the data
    mnemonics : List[str]
        Series identifiers (column names)
    """
    datafile = Path(datafile)
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read data file
    try:
        df = pd.read_csv(datafile)
    except Exception as e:
        raise ValueError(f"Failed to read data file {datafile}: {e}")
    
    # Check if first column is a date column or metadata
    first_col = df.columns[0]
    
    # Try to parse first column as date
    try:
        pd.to_datetime(df[first_col].iloc[0])
        is_date_first = True
    except (ValueError, TypeError):
        is_date_first = False
    
    # If first column is not a date, check if data is in "long" format (one row per series)
    if not is_date_first:
        # Look for date columns (starting from a certain column)
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                date_cols.append(col)
            except (ValueError, TypeError):
                pass
        
        if len(date_cols) > 0:
            # Long format: transpose and use first date column as index
            first_date_col = date_cols[0]
            date_col_idx = df.columns.get_loc(first_date_col)
            date_cols_all = df.columns[date_col_idx:]
            
            # Extract dates from first row
            dates = pd.to_datetime(df[first_date_col].iloc[0])
            for col in date_cols_all[1:]:
                try:
                    dates = pd.to_datetime([dates] + [df[col].iloc[0]])
                except:
                    pass
            
            # Transpose: rows become series, columns become time
            Z = df[date_cols_all].T.values.astype(float)
            Time = pd.to_datetime(date_cols_all)
            mnemonics = df[first_col].tolist() if first_col in df.columns else [f"series_{i}" for i in range(len(df))]
            
            return Z, Time, mnemonics
    
    # Standard format: first column is date, rest are series
    try:
        Time = pd.to_datetime(df[first_col])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse date column '{first_col}': {e}")
    
    # Extract series data (all columns except first)
    series_cols = [col for col in df.columns if col != first_col]
    Z = df[series_cols].values.astype(float)
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
    series_ids = config.get_series_ids()
    
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
            logger.warning(f"Series '{sid}' from config not found in data")
    
    if len(perm) == 0:
        raise ValueError("No matching series found between config and data")
    
    # Apply permutation
    Z_filt = Z[:, perm]
    
    return Z_filt, Mnem_filt


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
    from .utils import FREQUENCY_HIERARCHY
    
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


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None,
              sample_end: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load and transform time series data for DFM estimation.
    
    This function reads time series data, aligns it with the model configuration,
    and applies the specified transformations. The data is sorted to match the
    configuration order and validated against frequency constraints.
    
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
    sample_start : pd.Timestamp or str, optional
        Start date for sample (YYYY-MM-DD). If None, uses beginning of data.
        Data before this date will be dropped.
    sample_end : pd.Timestamp or str, optional
        End date for sample (YYYY-MM-DD). If None, uses end of data.
        Data after this date will be dropped.
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N), ready for DFM estimation
    Time : pd.DatetimeIndex
        Time index for the data (aligned to clock frequency)
    Z : np.ndarray
        Original untransformed data (T x N), for reference
        
    Raises
    ------
    ValueError
        If any series has frequency faster than clock, or data format is invalid
    FileNotFoundError
        If datafile does not exist
    """
    from .utils import FREQUENCY_HIERARCHY
    
    logger.info('Loading data...')
    
    datafile_path = Path(datafile)
    if datafile_path.suffix.lower() != '.csv':
        logger.warning(f"Data file extension is not .csv: {datafile_path.suffix}. Assuming CSV format.")
    
    # Read raw data
    Z, Time, Mnem = read_data(datafile_path)
    logger.info(f"Read {Z.shape[0]} time periods, {Z.shape[1]} series from {datafile_path}")
    
    # Sort data to match config order
    Z, Mnem = sort_data(Z, Mnem, config)
    logger.info(f"Sorted data to match configuration order")
    
    # Apply sample date filters
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.Timestamp(sample_start)
        mask = Time >= sample_start
        Z = Z[mask]
        Time = Time[mask]
        logger.info(f"Filtered to start date: {sample_start}")
    
    if sample_end is not None:
        if isinstance(sample_end, str):
            sample_end = pd.Timestamp(sample_end)
        mask = Time <= sample_end
        Z = Z[mask]
        Time = Time[mask]
        logger.info(f"Filtered to end date: {sample_end}")
    
    # Transform data
    X, Time, Z = transform_data(Z, Time, config)
    logger.info(f"Transformed data: {X.shape[0]} time periods, {X.shape[1]} series")
    
    # Validate data quality
    clock = getattr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    frequencies = config.get_frequencies()
    series_ids = config.get_series_ids()
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
            logger.warning(
                f"Series '{series_id}': T={T_obs} < N={N_total} (may cause numerical issues). "
                f"Suggested fix: increase sample size or reduce number of series."
            )
        if len(warnings_list) > 5:
            logger.warning(f"... and {len(warnings_list) - 5} more series with T < N")
        
        import warnings
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
            series_id = series_ids[i] if i < len(series_ids) else f"series_{i}"
            extreme_missing_series.append((series_id, ratio))
    
    if len(extreme_missing_series) > 0:
        for series_id, ratio in extreme_missing_series[:5]:
            logger.warning(
                f"Series '{series_id}' has {ratio:.1%} missing data (>90%). "
                f"This may cause estimation issues. Consider removing this series or increasing data coverage."
            )
        if len(extreme_missing_series) > 5:
            logger.warning(f"... and {len(extreme_missing_series) - 5} more series with >90% missing data")
        
        import warnings
        warnings.warn(
            f"Extreme missing data detected: {len(extreme_missing_series)} series have >90% missing values. "
            f"Estimation may be unreliable. Consider removing these series or increasing data coverage. "
            f"See log for details.",
            UserWarning,
            stacklevel=2
        )
    
    return X, Time, Z


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load configuration from DataFrame (internal helper).
    
    This function converts a DataFrame with series specifications into a DFMConfig.
    It's used internally by SpecCSVSource and other config loaders.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: series_id, series_name, frequency, transformation, blocks
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    series_list = []
    block_names_set = set()
    
    for _, row in df.iterrows():
        # Parse blocks (can be comma-separated string or list)
        blocks_str = row.get('blocks', 'Block_Global')
        if isinstance(blocks_str, str):
            blocks = [b.strip() for b in blocks_str.split(',')]
        else:
            blocks = blocks_str if isinstance(blocks_str, list) else ['Block_Global']
        
        # Track block names
        for block in blocks:
            block_names_set.add(block)
        
        series_list.append(SeriesConfig(
            series_id=row.get('series_id', f"series_{len(series_list)}"),
            series_name=row.get('series_name', row.get('series_id', f"Series {len(series_list)}")),
            frequency=row.get('frequency', 'm'),
            transformation=row.get('transformation', 'lin'),
            blocks=blocks
        ))
    
    # Create default blocks if none specified
    block_names = sorted(block_names_set) if block_names_set else ['Block_Global']
    blocks = {}
    for block_name in block_names:
        blocks[block_name] = BlockConfig(factors=1, ar_lag=1, clock='m')
    
    # block_names is derived automatically in DFMConfig.__post_init__ from blocks dict
    return DFMConfig(series=series_list, blocks=blocks)


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
