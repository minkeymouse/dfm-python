"""Data loading and transformation utilities for DFM estimation.

This module provides functions for reading, sorting, transforming, and loading time series data
for Dynamic Factor Model estimation.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict

import numpy as np
import polars as pl
from datetime import datetime
from dataclasses import dataclass

from .config import DFMConfig, SeriesConfig, BlockConfig
from .core.time import TimeIndex, parse_timestamp, to_python_datetime

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
    if not is_date_first:
        # Look for date columns (starting from a certain column)
        date_cols = []
        for col in df.columns:
            try:
                parse_timestamp(df[col][0])
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
    try:
        # Parse date column
        time_series = df[first_col].cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
        # If that fails, try other formats
        if time_series.null_count() > 0:
            # Try parsing as string first
            time_series = df[first_col].str.strptime(pl.Datetime, strict=False)
        Time = TimeIndex(time_series)
    except (ValueError, TypeError) as e:
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
    from .core.helpers import get_series_ids
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
        # Year-over-year difference (generic based on frequency)
        from .core.utils import get_periods_per_year
        year_step = get_periods_per_year(freq)
        if T > year_step:
            X[year_step:] = Z[year_step:] - Z[:-year_step]
    elif formula == 'pch':
        # Percent change
        if T > step:
            X[step:] = 100.0 * (Z[step:] - Z[:-step]) / np.abs(Z[:-step] + 1e-10)
    elif formula == 'pc1':
        # Year-over-year percent change (generic based on frequency)
        from .core.utils import get_periods_per_year
        year_step = get_periods_per_year(freq)
        if T > year_step:
            X[year_step:] = 100.0 * (Z[year_step:] - Z[:-year_step]) / np.abs(Z[:-year_step] + 1e-10)
    elif formula == 'pca':
        # Percent change annualized (generic based on frequency)
        if T > step:
            from .core.utils import get_annual_factor
            annual_factor = get_annual_factor(freq, step)
            X[step:] = annual_factor * 100.0 * (Z[step:] - Z[:-step]) / np.abs(Z[:-step] + 1e-10)
    elif formula == 'cch':
        # Continuously compounded rate of change
        if T > step:
            X[step:] = 100.0 * (np.log(np.abs(Z[step:]) + 1e-10) - np.log(np.abs(Z[:-step]) + 1e-10))
    elif formula == 'cca':
        # Continuously compounded annual rate of change (generic based on frequency)
        if T > step:
            from .core.utils import get_annual_factor
            annual_factor = get_annual_factor(freq, step)
            X[step:] = annual_factor * 100.0 * (np.log(np.abs(Z[step:]) + 1e-10) - np.log(np.abs(Z[:-step]) + 1e-10))
    elif formula == 'log':
        # Natural log
        X[:] = np.log(np.abs(Z) + 1e-10)
    else:
        X[:] = Z
    
    return X


def transform_data(Z: np.ndarray, Time: TimeIndex, config: DFMConfig) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
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
    Time : TimeIndex
        Time index for the data
    config : DFMConfig
        Model configuration with transformation specifications
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : TimeIndex
        Time index (may be truncated after transformation)
    Z : np.ndarray
        Original data (may be truncated to match X)
    """
    from .core.utils import FREQUENCY_HIERARCHY
    
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Validate frequencies - reject higher frequencies than clock
    from .core.helpers import safe_get_attr
    clock = safe_get_attr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    from .core.helpers import get_frequencies_from_config
    frequencies = get_frequencies_from_config(config)
    from .core.helpers import get_series_ids
    series_ids = get_series_ids(config)
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
    
    # DFMConfig always has series attribute, but check for safety
    transformations = [s.transformation for s in config.series] if hasattr(config, 'series') and config.series else ['lin'] * N
    
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
              sample_start: Optional[Union[datetime, str]] = None,
              sample_end: Optional[Union[datetime, str]] = None) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
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
    sample_start : datetime or str, optional
        Start date for sample (YYYY-MM-DD). If None, uses beginning of data.
        Data before this date will be dropped.
    sample_end : datetime or str, optional
        End date for sample (YYYY-MM-DD). If None, uses end of data.
        Data after this date will be dropped.
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N), ready for DFM estimation
    Time : TimeIndex
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
    from .core.utils import FREQUENCY_HIERARCHY
    
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
            sample_start = parse_timestamp(sample_start)
        mask = Time >= sample_start
        if isinstance(mask, pl.Series):
            mask = mask.to_numpy()
        Z = Z[mask]
        Time = Time.filter(mask) if hasattr(Time, 'filter') else Time[mask]
        logger.info(f"Filtered to start date: {sample_start}")
    
    if sample_end is not None:
        if isinstance(sample_end, str):
            sample_end = parse_timestamp(sample_end)
        mask = Time <= sample_end
        if isinstance(mask, pl.Series):
            mask = mask.to_numpy()
        Z = Z[mask]
        Time = Time.filter(mask) if hasattr(Time, 'filter') else Time[mask]
        logger.info(f"Filtered to end date: {sample_end}")
    
    # Transform data
    X, Time, Z = transform_data(Z, Time, config)
    logger.info(f"Transformed data: {X.shape[0]} time periods, {X.shape[1]} series")
    
    # Validate data quality
    # Note: DFMConfig always has 'clock' attribute, but use safe_get_attr for consistency
    from .core.helpers import safe_get_attr
    clock = safe_get_attr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    from .core.helpers import get_frequencies_from_config
    frequencies = get_frequencies_from_config(config)
    from .core.helpers import get_series_ids
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
            logger.warning(
                f"Series '{series_id}': T={T_obs} < N={N_total} (may cause numerical issues). "
                f"Suggested fix: increase sample size or reduce number of series."
            )
        if len(warnings_list) > 5:
            logger.warning(f"... and {len(warnings_list) - 5} more series with T < N")
        
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
            from .core.helpers import get_series_id_by_index
            series_id = get_series_id_by_index(config, i)
            extreme_missing_series.append((series_id, ratio))
    
    if len(extreme_missing_series) > 0:
        for series_id, ratio in extreme_missing_series[:5]:
            logger.warning(
                f"Series '{series_id}' has {ratio:.1%} missing data (>90%). "
                f"This may cause estimation issues. Consider removing this series or increasing data coverage."
            )
        if len(extreme_missing_series) > 5:
            logger.warning(f"... and {len(extreme_missing_series) - 5} more series with >90% missing data")
        
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
    from .core.time import get_latest_time
    from .core.helpers import get_series_ids
    
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


@dataclass
class DataView:
    """Lightweight descriptor for pseudo real-time data slices."""
    X: np.ndarray
    Time: Union[TimeIndex, Any]
    Z: Optional[np.ndarray]
    config: Optional[DFMConfig]
    view_date: Optional[Union[datetime, str]] = None
    description: Optional[str] = None
    X_frame: Optional[pl.DataFrame] = None
    
    def materialize(self) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
        """Return the masked arrays for this data view."""
        return create_data_view(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=self.view_date,
            X_frame=self.X_frame
        )
    
    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        Time: Union[TimeIndex, Any],
        Z: Optional[np.ndarray],
        config: Optional[DFMConfig],
        view_date: Optional[Union[datetime, str]] = None,
        description: Optional[str] = None,
        X_frame: Optional[pl.DataFrame] = None
    ) -> 'DataView':
        return cls(
            X=X,
            Time=Time,
            Z=Z,
            config=config,
            view_date=view_date,
            description=description,
            X_frame=X_frame
        )
    
    def with_view_date(self, view_date: Union[datetime, str]) -> 'DataView':
        """Return a shallow copy with a different view date."""
        return DataView(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=view_date,
            description=self.description,
            X_frame=self.X_frame
        )
