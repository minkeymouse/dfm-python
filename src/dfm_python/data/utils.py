"""Data reading utilities for DFM.

This module provides functions for reading time series data from files.
For transformations, users should provide their own sktime transformers
to DFMDataModule.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import DFMConfig
from ..config.utils import FREQUENCY_HIERARCHY
from ..utils.time import TimeIndex, parse_timestamp
from ..utils.helpers import (
    get_frequencies,
    get_series_ids,
)
from ..config.utils import get_periods_per_year
from ..logger import get_logger

_logger = get_logger(__name__)


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
        # Use pandas read_csv with low_memory=False to infer all columns properly
        df = pd.read_csv(datafile, low_memory=False)
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
        is_integer_id = pd.api.types.is_integer_dtype(first_col_type)
        
        # Only check for long format if first column is not an integer ID
        if not is_integer_id:
            # Look for date columns (starting from a certain column)
            date_cols = []
            for col in df.columns:
                try:
                    parse_timestamp(str(df[col].iloc[0]))
                    date_cols.append(col)
                except (ValueError, TypeError, IndexError):
                    pass
            
            if len(date_cols) > 0:
                # Long format: transpose and use first date column as index
                first_date_col = date_cols[0]
                date_col_idx = df.columns.get_loc(first_date_col)
                date_cols_all = df.columns[date_col_idx:].tolist()
                
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
                date_data = df[date_cols_all]
                Z = date_data.to_numpy().T.astype(float)
                Time = TimeIndex(dates)
                mnemonics = df[first_col].tolist() if first_col in df.columns else [f"series_{i}" for i in range(len(df))]
                
                return Z, Time, mnemonics
    
    # Standard format: first column is date, rest are series
    # Handle integer date_id columns (treat as sequential time index)
    try:
        # Check if first column is integer (date_id format)
        first_col_type = df[first_col].dtype
        if pd.api.types.is_integer_dtype(first_col_type):
            # Integer date_id: use as sequential index, generate synthetic dates
            n_periods = len(df)
            from datetime import timedelta
            # Start from a default date and increment by day
            start_date = datetime(2000, 1, 1)
            dates = [start_date + timedelta(days=int(df[first_col].iloc[i])) for i in range(n_periods)]
            Time = TimeIndex(dates)
        else:
            # Try to parse as date using pandas
            time_series = pd.to_datetime(df[first_col], errors='coerce', format='%Y-%m-%d')
            # If that fails, try without format
            if time_series.isna().any():
                time_series = pd.to_datetime(df[first_col], errors='coerce')
            Time = TimeIndex(time_series)
    except (ValueError, TypeError) as e:
        # If date parsing fails, treat first column as integer date_id
        try:
            first_col_type = df[first_col].dtype
            if pd.api.types.is_integer_dtype(first_col_type):
                n_periods = len(df)
                from datetime import timedelta
                start_date = datetime(2000, 1, 1)
                dates = [start_date + timedelta(days=int(df[first_col].iloc[i])) for i in range(n_periods)]
                Time = TimeIndex(dates)
            else:
                raise ValueError(f"Failed to parse date column '{first_col}': {e}")
        except Exception:
            raise ValueError(f"Failed to parse date column '{first_col}': {e}")
    
    # Extract series data (all columns except first)
    series_cols = [col for col in df.columns if col != first_col]
    series_data = df[series_cols]
    Z = series_data.to_numpy().astype(float)
    mnemonics = series_cols
    
    return Z, Time, mnemonics


def load_data(
    datafile: Union[str, Path],
    config: DFMConfig,
    sample_start: Optional[Union[datetime, str]] = None,
    sample_end: Optional[Union[datetime, str]] = None
) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
    """Load time series data for DFM estimation.
    
    This function reads time series data and aligns it with the model configuration.
    The data is sorted to match the configuration order and validated against frequency constraints.
    
    Note: This function returns raw (untransformed) data. To apply transformations and
    standardization, provide a custom sktime transformer to DFMDataModule.
    
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
        Raw data matrix (T x N), not transformed. Provide a custom sktime transformer to DFMDataModule.
    Time : TimeIndex
        Time index for the data (aligned to clock frequency)
    Z : np.ndarray
        Original untransformed data (T x N), same as X
        
    Raises
    ------
    ValueError
        If any series has frequency faster than clock, or data format is invalid
    FileNotFoundError
        If datafile does not exist
    """
    _logger.info('Loading data...')
    
    datafile_path = Path(datafile)
    if datafile_path.suffix.lower() != '.csv':
        _logger.warning(f"Data file extension is not .csv: {datafile_path.suffix}. Assuming CSV format.")
    
    # Read raw data
    Z, Time, Mnem = read_data(datafile_path)
    _logger.info(f"Read {Z.shape[0]} time periods, {Z.shape[1]} series from {datafile_path}")
    
    # Sort data to match config order (import from utils.data)
    from ..utils.data import sort_data
    Z, Mnem = sort_data(Z, Mnem, config)
    _logger.info(f"Sorted data to match configuration order")
    
    # Apply sample date filters
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = parse_timestamp(sample_start)
        mask = Time >= sample_start
        if isinstance(mask, pd.Series):
            mask = mask.values
        # Ensure mask is boolean numpy array
        mask = np.asarray(mask, dtype=bool)
        Z = Z.iloc[mask] if isinstance(Z, pd.DataFrame) else Z[mask]
        # TimeIndex has filter() method - convert mask to list for filter()
        Time = Time.filter(mask.tolist())
        _logger.info(f"Filtered to start date: {sample_start}")
    
    if sample_end is not None:
        if isinstance(sample_end, str):
            sample_end = parse_timestamp(sample_end)
        mask = Time <= sample_end
        if isinstance(mask, pd.Series):
            mask = mask.values
        # Ensure mask is boolean numpy array
        mask = np.asarray(mask, dtype=bool)
        Z = Z.iloc[mask] if isinstance(Z, pd.DataFrame) else Z[mask]
        # TimeIndex has filter() method - convert mask to list for filter()
        Time = Time.filter(mask.tolist())
        _logger.info(f"Filtered to end date: {sample_end}")
    
    # Return raw data (transformations should be applied via custom sktime transformer in DFMDataModule)
    X = Z
    _logger.info(f"Loaded data: {X.shape[0]} time periods, {X.shape[1]} series (raw, not transformed)")
    
    # Validate data quality
    clock = getattr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    frequencies = get_frequencies(config)
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
            from ..utils.helpers import get_series_id
            series_id = get_series_id(config, i)
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

