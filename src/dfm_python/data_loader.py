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

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DFMConfig

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
# the package generic. Use YAML configs, Hydra DictConfigs, dictionaries, or
# the Spec CSV adapter (load_config_from_spec) for series definitions.


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load DFMConfig from a pandas DataFrame (used by spec CSV adapter).
    
    Converts tabular configuration data (series metadata + block loadings)
    into a validated DFMConfig object.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with configuration columns (series metadata and block loadings)
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    # Handle series_id generation: use 'id' as fallback if 'series_id' not present
    if 'series_id' not in df.columns:
        if 'id' in df.columns:
            df['series_id'] = df['id'].astype(str)
        else:
            # Generate sequential IDs if neither exists
            df['series_id'] = [f'series_{i}' for i in range(len(df))]
    elif 'id' in df.columns and 'series_id' in df.columns:
        # Both exist - prefer series_id, but if it's empty, use id
        df['series_id'] = df['series_id'].fillna(df['id'].astype(str))
    
    # Required fields for DFM configuration
    required_fields = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
    
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Detect block columns (all columns not in required fields)
    # Exclude common metadata fields that are not part of core DFM config
    excluded_fields = set(required_fields) | {
        'id',  # Generic ID field (used as fallback for series_id)
        'Release', 'release',  # Release date/metadata column
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
    from .config import SeriesConfig, BlockConfig
    
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
    
    # Create blocks dictionary from block column names
    # Each block gets default BlockConfig (1 factor, monthly clock)
    blocks_dict = {}
    for block_name in block_columns:
        blocks_dict[block_name] = BlockConfig(
            factors=1,
            clock='m'  # Default to monthly, can be overridden in config
        )
    
    return DFMConfig(series=series_list, blocks=blocks_dict)




def load_config_from_spec(specfile: Union[str, Path]) -> DFMConfig:
    """Load DFMConfig from a spec CSV file.
    
    This function reads a spec file (CSV format) that contains series definitions
    with their configurations and block memberships. The spec file should have:
    
    Required columns:
    - series_id: Unique identifier for the series
    - series_name: Human-readable name
    - frequency: Frequency code ('m', 'q', 'sa', 'a')
    - transformation: Transformation code ('lin', 'pch', 'pca', etc.)
    - category: Category/group name
    - units: Units of measurement
    
    Block columns (one per block, containing 0 or 1):
    - Block_Global (or first block), Block_Consumption, Block_Investment, etc.
    - All series must load on the first block (global block) = 1
    
    Only these columns are read - all other columns (e.g., Release, metadata) are ignored.
    
    Parameters
    ----------
    specfile : str or Path
        Path to spec CSV file
        
    Returns
    -------
    DFMConfig
        Model configuration object
        
    Raises
    ------
    FileNotFoundError
        If specfile does not exist
    ValueError
        If required columns are missing or data is invalid
        
    Examples
    --------
    >>> from dfm_python.data_loader import load_config_from_spec
    >>> from dfm_python import DFM
    >>> config = load_config_from_spec('data/sample_spec.csv')
    >>> # Use config for DFM estimation
    >>> model = DFM()
    >>> result = model.fit(X, config)
    """
    from .config import SpecCSVSource
    return SpecCSVSource(specfile).load()


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
        - Relative paths are resolved relative to current working directory
        - If not found, tries to resolve relative to package installation directory
        
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
    configfile_path = Path(configfile)
    
    # If path is relative and doesn't exist, try to find it relative to common locations
    if not configfile_path.is_absolute() and not configfile_path.exists():
        # Try current working directory first (already checked above)
        # Try relative to package directory
        import os
        package_dir = Path(__file__).parent.parent.parent  # Go up from src/dfm_python/data_loader.py
        potential_paths = [
            configfile_path,  # Original path (relative to cwd)
            package_dir / configfile_path,  # Relative to package root
            Path.cwd() / configfile_path,  # Relative to current working directory
        ]
        
        for path in potential_paths:
            if path.exists():
                configfile_path = path
                break
        else:
            # If still not found, raise error with helpful message
            raise FileNotFoundError(
                f"Configuration file not found: {configfile}\n"
                f"Tried paths:\n"
                f"  - {configfile_path}\n"
                f"  - {package_dir / configfile_path}\n"
                f"  - {Path.cwd() / configfile_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please provide an absolute path or ensure the file exists relative to the current working directory."
            )
    elif not configfile_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile_path}")
    
    suffix = configfile_path.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile_path)
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
    # Base frequency is monthly, so step is months per observation
    freq_to_step = {
        'm': 1,   # Monthly: 1 month per observation
        'q': 3,   # Quarterly: 3 months per observation
        'sa': 6,  # Semi-annual: 6 months per observation
        'a': 12,  # Annual: 12 months per observation
    }
    
    for i in range(N):
        freq = frequencies[i]
        step = freq_to_step.get(freq, 1)  # Default to 1 if unknown frequency
        transformations = [s.transformation for s in config.series]
        trans = transformations[i] if i < len(transformations) else 'lin'
        X[:, i] = _transform_series(Z[:, i], trans, freq, step)
    
    # Drop initial observations to handle transformation edge effects
    # Use maximum step (longest observation period) to determine drop period
    max_step = max([freq_to_step.get(f, 1) for f in frequencies])
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
        Sorted series identifiers matching config.get_series_ids() order
    """
    series_ids = config.get_series_ids()
    in_config = [m in series_ids for m in Mnem]
    Mnem_filt = [m for m, in_c in zip(Mnem, in_config) if in_c]
    Z_filt = Z[:, in_config]
    
    perm = [Mnem_filt.index(sid) for sid in series_ids]
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
        raise ValueError(
            'File-based data loading currently supports CSV format. '
            'For other formats or database-backed applications, implement '
            'an adapter that returns (X, Time, Z) arrays.'
        )
    
    # If path is relative and doesn't exist, try to find it relative to common locations
    if not datafile_path.is_absolute() and not datafile_path.exists():
        # Try current working directory first (already checked above)
        # Try relative to package directory
        package_dir = Path(__file__).parent.parent.parent  # Go up from src/dfm_python/data_loader.py
        potential_paths = [
            datafile_path,  # Original path (relative to cwd)
            package_dir / datafile_path,  # Relative to package root
            Path.cwd() / datafile_path,  # Relative to current working directory
        ]
        
        for path in potential_paths:
            if path.exists():
                datafile_path = path
                break
        else:
            # If still not found, raise error with helpful message
            raise FileNotFoundError(
                f"Data file not found: {datafile}\n"
                f"Tried paths:\n"
                f"  - {datafile_path}\n"
                f"  - {package_dir / datafile_path}\n"
                f"  - {Path.cwd() / datafile_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please provide an absolute path or ensure the file exists relative to the current working directory."
            )
    elif not datafile_path.exists():
        raise FileNotFoundError(f"Data file not found: {datafile_path}")
    
    datafile = datafile_path
    
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
    
    series_ids = config.get_series_ids()
    frequencies = config.get_frequencies()
    faster_series = []
    for i, series_id in enumerate(series_ids):
        freq = frequencies[i] if i < len(frequencies) else clock
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
    
    # Apply sample window filtering
    start_ts = None
    end_ts = None
    if sample_start is not None:
        start_ts = pd.to_datetime(sample_start) if isinstance(sample_start, str) else sample_start
    if sample_end is not None:
        end_ts = pd.to_datetime(sample_end) if isinstance(sample_end, str) else sample_end
    if start_ts is not None or end_ts is not None:
        mask = np.ones(len(Time), dtype=bool)
        if start_ts is not None:
            mask &= (Time >= start_ts)
        if end_ts is not None:
            mask &= (Time <= end_ts)
        Time, X, Z = Time[mask], X[mask], Z[mask]
    
    # Check T >= N per block after windowing
    T, N = X.shape
    blocks_array = config.get_blocks_array()  # N x n_blocks
    block_names = config.block_names
    
    warnings_list = []
    for block_idx, block_name in enumerate(block_names):
        # Count series in this block (non-zero loadings)
        series_in_block = blocks_array[:, block_idx] == 1
        n_series_in_block = series_in_block.sum()
        
        if n_series_in_block == 0:
            continue  # Skip empty blocks
        
        # Check if T >= N for this block
        if T < n_series_in_block:
            warnings_list.append({
                'block_name': block_name,
                'block_idx': block_idx,
                'T': T,
                'N': n_series_in_block,
                'series_ids': [series_ids[i] for i in range(N) if series_in_block[i]],
            })
    
    # Issue warnings with actionable suggestions
    if warnings_list:
        warning_msg = (
            f"\n{'='*70}\n"
            f"WARNING: Insufficient data for covariance calculation\n"
            f"{'='*70}\n"
            f"After applying date windowing (sample_start={sample_start}, sample_end={sample_end}),\n"
            f"some blocks have fewer time periods (T) than series (N).\n"
            f"This will cause singular covariance matrices during estimation.\n\n"
        )
        
        for w in warnings_list:
            warning_msg += (
                f"Block '{w['block_name']}' (index {w['block_idx']}):\n"
                f"  - Time periods (T): {w['T']}\n"
                f"  - Series count (N): {w['N']}\n"
                f"  - Series IDs: {', '.join(w['series_ids'][:5])}"
                + (f" ... ({len(w['series_ids'])-5} more)" if len(w['series_ids']) > 5 else "")
                + "\n\n"
            )
        
        warning_msg += (
            f"Suggested fixes:\n"
            f"  1. Extend sample_start to include more historical data\n"
            f"  2. Reduce the number of series in affected blocks\n"
            f"  3. Use a different nan_method (e.g., nan_method=1 for listwise deletion)\n"
            f"  4. Increase regularization_scale in config for numerical stability\n"
            f"{'='*70}\n"
        )
        
        logger.warning(warning_msg)
        # Also issue a Python warning for visibility
        import warnings
        warnings.warn(
            f"Insufficient data: {len(warnings_list)} block(s) have T < N. "
            f"See log for details.",
            UserWarning,
            stacklevel=2
        )
    
    return X, Time, Z


