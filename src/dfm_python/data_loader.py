"""Data loading and transformation utilities for DFM estimation.

This module provides comprehensive data handling for Dynamic Factor Models:
- Configuration loading from YAML files or direct DFMConfig objects
- Time series data loading with automatic date parsing
- Time series transformations (differences, percent changes, etc.)
- Data sorting and alignment with configuration

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

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DFMConfig

ModelConfig = DFMConfig

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


def load_config_from_yaml(configfile: Union[str, Path]) -> DFMConfig:
    """Load model configuration from YAML file.
    
    Parameters
    ----------
    configfile : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    DFMConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist
    ImportError
        If omegaconf is not available
    ValueError
        If configuration is invalid
    """
    if not OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    cfg = OmegaConf.load(configfile)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract model config and dfm parameters
    model_dict = None
    dfm_params = {}
    
    # Handle nested structure (if config has @package model: directive)
    if 'series' in cfg_dict and 'block_names' in cfg_dict:
        # Direct model config structure
        model_dict = cfg_dict
    elif 'model' in cfg_dict:
        # Nested under 'model' key (from @package model:)
        model_dict = cfg_dict['model'].copy() if isinstance(cfg_dict['model'], dict) else {}
        # If series is at top level, merge it into model_dict
        if 'series' in cfg_dict and 'series' not in model_dict:
            model_dict['series'] = cfg_dict['series']
    elif 'series' in cfg_dict:
        # Series at top level, construct model_dict from top-level keys
        model_dict = {k: v for k, v in cfg_dict.items() if k not in ['dfm', 'data', 'experiment_name', 'output_dir', 'defaults']}
    else:
        # Try to construct from top-level keys
        model_dict = cfg_dict
    
    # Extract dfm parameters (estimation parameters like clock)
    if 'dfm' in cfg_dict:
        dfm_params = cfg_dict['dfm']
    
    # Merge dfm parameters into model config
    if dfm_params:
        if isinstance(model_dict, dict):
            model_dict = {**model_dict, **dfm_params}
        else:
            # If model_dict is not a dict, create new dict
            model_dict = {'series': model_dict, **dfm_params} if model_dict else dfm_params
    
    return DFMConfig.from_dict(model_dict)


def load_config_from_csv(configfile: Union[str, Path]) -> DFMConfig:
    """DEPRECATED: Load model configuration from CSV file.
    
    This function is deprecated. CSV config loading has been removed to make
    the package more generic. Users should:
    - Use YAML config files with Hydra/OmegaConf
    - Create DFMConfig objects directly in code
    - Use application-specific adapters for custom formats
    
    This function is kept for backward compatibility but will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "load_config_from_csv is deprecated. Use YAML config files or create DFMConfig objects directly. "
        "For CSV configs, implement an application-specific adapter.",
        DeprecationWarning,
        stacklevel=2
    )
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    try:
        df = pd.read_csv(configfile)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {configfile}: {e}")
    
    return _load_config_from_dataframe(df)


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load DFMConfig from a pandas DataFrame.
    
    This is a helper function for deprecated CSV config loading.
    Converts tabular configuration data into DFMConfig format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with configuration columns (series metadata and block loadings)
        
    Returns
    -------
    DFMConfig
        Model configuration object
        
    Note
    ----
    This function is used internally by deprecated load_config_from_csv.
    For new code, use YAML configs or create DFMConfig objects directly.
    """
    # Handle series_id generation from various column combinations
    # Otherwise, use 'id' as fallback
    if 'id' in df.columns and 'series_id' not in df.columns:
        # Try to generate series_id from data_code, item_id, api_source if available
        if all(col in df.columns for col in ['data_code', 'item_id', 'api_source']):
            # Generate series_id: {api_source}_{data_code}_{item_id}
            df['series_id'] = df.apply(
                lambda row: f"{row.get('api_source', '')}_{row.get('data_code', '')}_{row.get('item_id', '')}",
                axis=1
            )
        else:
            # Fallback: use 'id' as series_id if series_id column doesn't exist
            df['series_id'] = df['id'].astype(str)
    elif 'id' in df.columns and 'series_id' in df.columns:
        # Both exist - prefer series_id, but if it's empty, use id
        df['series_id'] = df['series_id'].fillna(df['id'].astype(str))
    
    # Required fields for DFM configuration
    required_fields = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
    
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Detect block columns (all columns not in required fields or excluded metadata)
    # Exclude application-specific metadata fields that are not part of core DFM config
    excluded_fields = set(required_fields) | {
        'id', 'country', 'data_code', 'item_id', 'api_source', 'api_code', 
        'api_group_id', 'is_kpi', 'description', 'priority', 'is_active', 'metadata'
    }
    # Preserve original column order from DataFrame
    block_columns = [col for col in df.columns if col not in excluded_fields]
    
    if not block_columns:
        raise ValueError("No block columns found. Expected columns like 'Block1', 'Block2', etc.")
    
    # Validate block columns contain only 0 or 1
    for block_col in block_columns:
        if not df[block_col].isin([0, 1]).all():
            raise ValueError(f"Block column '{block_col}' must contain only 0 or 1")
    
    # Ensure all series load on at least one block (first block should always be 1)
    if block_columns[0] not in df.columns:
        raise ValueError(f"First block column '{block_columns[0]}' is required")
    
    if not (df[block_columns[0]] == 1).all():
        raise ValueError(f"All series must load on the first block '{block_columns[0]}'")
    
    # Build blocks array (N x n_blocks)
    blocks_data = df[block_columns].values.astype(int)
    
    # Convert to DFMConfig format
    from .config import SeriesConfig
    
    series_list = []
    for idx, row in df.iterrows():
        # Build block array from block columns
        blocks = [int(row[col]) for col in block_columns]
        
        # Create SeriesConfig with core DFM fields only
        # Application-specific metadata fields are ignored
        series_list.append(SeriesConfig(
            series_id=str(row['series_id']),  # Ensure string type
            series_name=row['series_name'],
            frequency=row['frequency'],
            transformation=row['transformation'],
            category=row['category'],
            units=row['units'],
            blocks=blocks
        ))
    
    return DFMConfig(series=series_list, block_names=block_columns)




def load_config(configfile: Union[str, Path, DFMConfig]) -> DFMConfig:
    """Load model configuration from file or return existing DFMConfig object.
    
    This function supports:
    - YAML files (using Hydra/OmegaConf)
    - Direct DFMConfig objects (pass through)
    
    For CSV configs, use application-specific adapters or create DFMConfig objects directly.
    
    Parameters
    ----------
    configfile : str, Path, or DFMConfig
        - Path to YAML configuration file (.yaml, .yml)
        - Or existing DFMConfig object (returned as-is)
        
    Returns
    -------
    DFMConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist (for file paths)
    ValueError
        If file format is not supported or configuration is invalid
    TypeError
        If configfile is not a valid type
    """
    # If already a DFMConfig object, return as-is
    if isinstance(configfile, DFMConfig):
        return configfile
    
    # Handle file paths
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    suffix = configfile.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .yaml or .yml files, "
            f"or pass a DFMConfig object directly. "
            f"For CSV configs, implement an application-specific adapter."
        )


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Apply transformation formula to a single time series.
    
    Transforms raw series data according to the specified formula and frequency.
    Handles various transformation types including differences, percent changes,
    and logarithms.
    
    Parameters
    ----------
    Z : np.ndarray
        Raw series data (1D array)
    formula : str
        Transformation formula: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'log'
    freq : str
        Series frequency (used for step calculation)
    step : int
        Number of base periods per observation (e.g., 3 for quarterly)
        
    Returns
    -------
    X : np.ndarray
        Transformed series data (1D array, same length as Z)
    """
    T = Z.shape[0]
    X = np.full(T, np.nan)
    t1 = step
    n = step / 12
    
    if formula == 'lin':
        X[:] = Z
    elif formula == 'chg':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = Z[idx[1:]] - Z[idx[:-1]]
    elif formula == 'ch1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            X[idx] = Z[idx] - Z[idx - 12]
    elif formula == 'pch':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = 100 * (Z[idx[1:]] / Z[idx[:-1]] - 1)
    elif formula == 'pc1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx] = 100 * (Z[idx] / Z[idx - 12] - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'pca':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx[1:]] = 100 * ((Z[idx[1:]] / Z[idx[:-1]]) ** (1/n) - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'log':
        with np.errstate(invalid='ignore'):
            X[:] = np.log(Z)
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
    
    for i, freq in enumerate(config.Frequency):
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
        if freq_hierarchy < clock_hierarchy:
            raise ValueError(
                f"Series '{config.SeriesID[i]}' has frequency '{freq}' which is faster than clock '{clock}'. "
                f"Higher frequencies (daily, weekly) are not supported. "
                f"Please use monthly, quarterly, semi-annual, or annual frequencies only."
            )
    
    # Frequency to step mapping (step = number of base periods per observation)
    # Base frequency is monthly, so step is months per observation
    freq_to_step = {
        'm': 1,   # Monthly: 1 month per observation
        'q': 3,   # Quarterly: 3 months per observation
        'sa': 6,  # Semi-annual: 6 months per observation
        'a': 12,  # Annual: 12 months per observation
    }
    
    for i in range(N):
        freq = config.Frequency[i]
        step = freq_to_step.get(freq, 1)  # Default to 1 if unknown frequency
        X[:, i] = _transform_series(Z[:, i], config.Transformation[i], freq, step)
    
    # Drop initial observations to handle transformation edge effects
    # Use maximum step (longest observation period) to determine drop period
    max_step = max([freq_to_step.get(f, 1) for f in config.Frequency])
    # Drop period ensures sufficient history for transformations
    drop = max(4, max_step + 1)
    
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
        # Check if first column contains series IDs (string values)
        first_col_values = df[first_col].astype(str)
        # If first column looks like series IDs and we have date columns, transpose
        if 'series_id' in first_col.lower() or first_col_values.str.match(r'^[A-Z0-9_]+$').any():
            # Long format: transpose so series are columns and dates are rows
            # Find first date column
            date_col_idx = None
            for i, col in enumerate(df.columns):
                try:
                    # Try to parse first value as date
                    pd.to_datetime(df[col].iloc[0])
                    date_col_idx = i
                    break
                except (ValueError, TypeError):
                    continue
            
            if date_col_idx is None:
                raise ValueError(f"Could not find date column in data file {datafile}")
            
            # Set series_id as index, then transpose
            series_id_col = df.columns[0]
            df = df.set_index(series_id_col)
            
            # Get date columns (from date_col_idx onwards)
            date_cols = df.columns[date_col_idx:]
            df_data = df[date_cols].T  # Transpose: dates become rows, series become columns
            
            # Convert date column names to datetime index
            df_data.index = pd.to_datetime(df_data.index)
            
            mnemonics = df_data.columns.tolist()
            Time = df_data.index
            Z = df_data.apply(pd.to_numeric, errors='coerce').values.astype(float)
            
            return Z, Time, mnemonics
        else:
            # Find first column that looks like a date
            date_col_idx = None
            for i, col in enumerate(df.columns):
                try:
                    # Try to parse first value as date
                    pd.to_datetime(df[col].iloc[0])
                    date_col_idx = i
                    break
                except (ValueError, TypeError):
                    continue
            
            if date_col_idx is None:
                raise ValueError(f"Could not find date column in data file {datafile}")
            
            # Use first date column as index, drop metadata columns
            date_col = df.columns[date_col_idx]
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)
            # Drop any remaining non-numeric columns (metadata)
            numeric_cols = []
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue
            df = df[numeric_cols]
    else:
        # Standard format: first column is date
        df = df.set_index(first_col)
        df.index = pd.to_datetime(df.index)
    
    mnemonics = df.columns.tolist()
    Time = df.index
    # Convert to float, handling any remaining non-numeric values
    Z = df.apply(pd.to_numeric, errors='coerce').values.astype(float)
    
    return Z, Time, mnemonics


def sort_data(Z: np.ndarray, Mnem: List[str], config: DFMConfig) -> Tuple[np.ndarray, List[str]]:
    """Sort data series to match configuration order.
    
    Filters and reorders series to match the order specified in the configuration.
    Only series present in both data and configuration are retained.
    
    Parameters
    ----------
    Z : np.ndarray
        Data matrix (T x N) with N series
    Mnem : List[str]
        Series identifiers (mnemonics) corresponding to columns of Z
    config : DFMConfig
        Model configuration with series order specification
        
    Returns
    -------
    Z_sorted : np.ndarray
        Filtered and sorted data matrix (T x M) where M <= N
    Mnem_sorted : List[str]
        Sorted series identifiers matching config.SeriesID order
    """
    in_config = [m in config.SeriesID for m in Mnem]
    Mnem_filt = [m for m, in_c in zip(Mnem, in_config) if in_c]
    Z_filt = Z[:, in_config]
    
    perm = [Mnem_filt.index(sid) for sid in config.SeriesID]
    return Z_filt[:, perm], [Mnem_filt[i] for i in perm]


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
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
        Start date for sample (YYYY-MM-DD format). If None, uses all available data.
        Data before this date will be dropped.
        
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
    print('Loading data...')
    
    datafile = Path(datafile)
    if datafile.suffix.lower() != '.csv':
        raise ValueError(
            'File-based data loading currently supports CSV format. '
            'For other formats or database-backed applications, implement '
            'an adapter that returns (X, Time, Z) arrays.'
        )
    
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read data from file
    Z, Time, Mnem = read_data(datafile)
    
    # Process data: sort to match config order
    Z, _ = sort_data(Z, Mnem, config)
    
    # Get clock frequency (default to monthly)
    clock = getattr(config, 'clock', 'm')
    
    # Import frequency hierarchy for resampling logic
    from .utils.aggregation import FREQUENCY_HIERARCHY
    
    # Validate frequency constraints: no series can be faster than clock
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    faster_series = []
    for i, series_id in enumerate(config.SeriesID):
        freq = config.Frequency[i] if i < len(config.Frequency) else clock
        series_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
        if series_hierarchy < clock_hierarchy:
            faster_series.append((series_id, freq))
    
    if faster_series:
        faster_list = ', '.join([f"{sid} ({freq})" for sid, freq in faster_series])
        raise ValueError(
            f"Frequency constraint violation: The following series have frequencies "
            f"faster than clock '{clock}': {faster_list}. "
            f"All series must have frequency equal to or slower than the clock frequency."
        )
    
    # Apply transformations (lower frequency series handled via tent kernels in dfm.py)
    X, Time, Z = transform_data(Z, Time, config)
    
    # Apply sample_start filtering
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.to_datetime(sample_start)
        mask = Time >= sample_start
        Time, X, Z = Time[mask], X[mask], Z[mask]
    
    return X, Time, Z


