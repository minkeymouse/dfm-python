"""Data loading and transformation utilities for DFM estimation.

This module provides comprehensive data handling for Dynamic Factor Models:
- Configuration loading from YAML files or direct DFMConfig objects
- Time series data loading with automatic date parsing
- Time series transformations (differences, percent changes, etc.)
- Data sorting and alignment with configuration
- Missing value treatment via spline interpolation
- Data summary and visualization

The module supports flexible configuration formats and handles common data
issues such as missing dates, inconsistent formats, and transformation errors.

Configuration:
    - YAML files via Hydra/OmegaConf
    - Direct DFMConfig object creation
    - Application-specific adapters for custom formats

Data Loading:
    - File-based data loading (CSV format supported for convenience)
    - Database-backed applications should implement adapters that return
      the same interface: (X, Time, Z) arrays
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DFMConfig
from .data.utils import rem_nans_spline  # Backward compatibility

logger = logging.getLogger(__name__)


def load_config_from_yaml(configfile: Union[str, Path]) -> DFMConfig:
    """Load model configuration from YAML file.
    
    This function loads the config structure:
    - Main settings (estimation parameters) from config/default.yaml
    - Series definitions from config/series/default.yaml
    - Block definitions from config/blocks/default.yaml
    
    Parameters
    ----------
    configfile : str or Path
        Path to main YAML configuration file (e.g., config/default.yaml)
        
    Returns
    -------
    DFMConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile or referenced files do not exist
    ImportError
        If omegaconf is not available
    ValueError
        If configuration is invalid
    """
    from .config import YamlSource
    return YamlSource(configfile).load()


# Note: Historical CSV config loading has been removed for simplicity and to keep
# the codebase focused. Use YAML or direct DFMConfig creation instead.
# If you need CSV config loading, implement it via a custom ConfigSource adapter.


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
    from .config import DFMConfig, SeriesConfig, BlockConfig
    
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
    
    return DFMConfig(series=series_list, blocks=blocks, block_names=block_names)


def load_config_from_spec(specfile: Union[str, Path]) -> DFMConfig:
    """Load configuration from spec CSV file.
    
    Parameters
    ----------
    specfile : str or Path
        Path to spec CSV file
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    specfile = Path(specfile)
    if not specfile.exists():
        raise FileNotFoundError(f"Spec file not found: {specfile}")
    
    df = pd.read_csv(specfile)
    return _load_config_from_dataframe(df)


def load_config(configfile: Union[str, Path, DFMConfig]) -> DFMConfig:
    """Load configuration from file or return DFMConfig directly.
    
    This is a convenience function that handles both file paths and DFMConfig objects.
    For more control, use the specific load_config_from_* functions or ConfigSource classes.
    
    Parameters
    ----------
    configfile : str, Path, or DFMConfig
        Path to config file (YAML or spec CSV) or DFMConfig object
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    if isinstance(configfile, DFMConfig):
        return configfile
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Config file not found: {configfile}")
    
    # Try YAML first, then spec CSV
    if configfile.suffix.lower() in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile)
    elif configfile.suffix.lower() == '.csv':
        return load_config_from_spec(configfile)
    else:
        # Default to YAML
        return load_config_from_yaml(configfile)


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
    from .utils.aggregation import FREQUENCY_HIERARCHY
    
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
    
    return Z_filt[:, perm], [Mnem_filt[i] for i in perm]


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
    from .utils.aggregation import FREQUENCY_HIERARCHY
    
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
            logger.warning(f"Series '{series_id}': T={T_obs} < N={N_total} (may cause numerical issues)")
        if len(warnings_list) > 5:
            logger.warning(f"... and {len(warnings_list) - 5} more series with T < N")
        
        import warnings
        warnings.warn(
            f"Insufficient data: {len(warnings_list)} block(s) have T < N. "
            f"See log for details.",
            UserWarning,
            stacklevel=2
        )
    
    return X, Time, Z


def summarize(X: np.ndarray, Time, config: DFMConfig, vintage: Optional[str] = None) -> None:
    """Display data summary table."""
    T, N = X.shape
    
    print("\n\n\nTable 2: Data Summary")
    print(f"N = {N:4d} data series")
    
    try:
        time_start = Time[0].strftime('%Y-%m-%d') if hasattr(Time[0], 'strftime') else str(Time[0])
        time_end = Time[-1].strftime('%Y-%m-%d') if hasattr(Time[-1], 'strftime') else str(Time[-1])
    except (AttributeError, ValueError, TypeError) as e:
        logger.debug(f"summarize: Failed to format time range: {type(e).__name__}: {str(e)}")
        time_start = time_end = "N/A"
    
    print(f"T = {T:4d} observations from {time_start:10s} to {time_end:10s}\n")
    print(f"{'Data Series':30s} | {'Observations':17s}    {'Units':12s}    "
          f"{'Frequency':10s}    {'Mean':8s}    {'Std. Dev.':8s}    {'Min':8s}    {'Max':8s}")
    print("-" * 130)
    
    for i in range(N):
        t_obs = ~np.isnan(X[:, i])
        obs_idx = np.where(t_obs)[0]
        
        if hasattr(config, 'get_series_names'):
            series_names = config.get_series_names()
        elif hasattr(config, 'SeriesName'):
            series_names = config.SeriesName
        else:
            series_names = []
        
        from .core.helpers import safe_get_method
        series_ids = safe_get_method(config, 'get_series_ids', [])
        if not series_ids and hasattr(config, 'SeriesID'):
            series_ids = config.SeriesID
        
        if hasattr(config, 'get_frequencies'):
            frequencies = config.get_frequencies()
        elif hasattr(config, 'Frequency'):
            frequencies = config.Frequency
        else:
            frequencies = []
        
        if hasattr(config, 'series'):
            transformations = [s.transformation for s in config.series]
        elif hasattr(config, 'Transformation'):
            transformations = config.Transformation
        else:
            transformations = []
        
        name = series_names[i][:27] + "..." if i < len(series_names) and len(series_names[i]) > 30 else (series_names[i] if i < len(series_names) else f"series_{i}")
        sid = f"[{series_ids[i][:25]}...]" if i < len(series_ids) and len(series_ids[i]) > 28 else f"[{series_ids[i] if i < len(series_ids) else f'series_{i}'}]"
        
        # Map frequency codes to readable names
        freq_map = {'m': 'Monthly', 'q': 'Quarterly', 'sa': 'Semi-annual', 'a': 'Annual'}
        series_freq = frequencies[i] if i < len(frequencies) else 'm'
        freq = freq_map.get(series_freq, series_freq.upper())
        trans = transformations[i] if i < len(transformations) else 'lin'
        
        # Get transformation units
        from .config import _TRANSFORM_UNITS_MAP
        units_t = 'MoM%' if trans == 'pch' and series_freq == 'm' else \
                  'QoQ% AR' if trans == 'pca' and series_freq == 'q' else \
                  _TRANSFORM_UNITS_MAP.get(trans, trans)
        
        date_range = "N/A"
        if len(obs_idx) > 0:
            try:
                fmt = '%b %Y' if series_freq == 'm' else '%Y-%m'
                if hasattr(Time, 'iloc'):
                    date_range = f"{Time.iloc[obs_idx[0]].strftime(fmt)}-{Time.iloc[obs_idx[-1]].strftime(fmt)}"
                else:
                    date_range = f"{Time[obs_idx[0]].strftime(fmt)}-{Time[obs_idx[-1]].strftime(fmt)}"
            except (AttributeError, ValueError, TypeError) as e:
                logger.debug(f"summarize: Failed to format date range with strftime: {type(e).__name__}: {str(e)}")
                try:
                    if hasattr(Time, 'iloc'):
                        date_range = f"{Time.iloc[obs_idx[0]]}-{Time.iloc[obs_idx[-1]]}"
                    else:
                        date_range = f"{Time[obs_idx[0]]}-{Time[obs_idx[-1]]}"
                except (AttributeError, ValueError, TypeError, IndexError) as e2:
                    logger.debug(f"summarize: Failed to format date range fallback: {type(e2).__name__}: {str(e2)}")
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
