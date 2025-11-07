"""Data loading and transformation utilities for DFM estimation.

This module provides comprehensive data handling for Dynamic Factor Models:
- Configuration loading from YAML and CSV files
- Data loading from CSV with automatic date parsing
- Time series transformations (differences, percent changes, etc.)
- Data sorting and alignment with configuration

The module supports flexible configuration formats and handles common data
issues such as missing dates, inconsistent formats, and transformation errors.
"""

import numpy as np
import pandas as pd
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings
import logging

from .config import DFMConfig

# Backward compatibility alias (deprecated - use DFMConfig)
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
    ModelConfig
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
    """Load model configuration from CSV file.
    
    CSV format should have columns:
    - series_id (or 'id' as alias), series_name, frequency, transformation, category, units
    - Block columns (named after block names, e.g., Global, Consumption, Investment, External, or Block_Global, Block_Consumption, etc.)
    - Block columns contain 0 or 1 (1 = series loads on that block)
    
    Note: API/database-specific columns (data_code, item_id, api_source, country, is_kpi, etc.)
    are ignored by the generic DFM module. They should be handled by application-specific adapters.
    
    Example:
        id,series_name,frequency,transformation,category,units,Block_Global,Block_Consumption,Block_Invest,Block_Extern
        0,Real GDP (Quarterly),q,pca,GDP,Billion Won,1,1,0,0
        1,Nominal GDP (Quarterly),q,pca,GDP,Billion Won,1,1,0,0
    
    Parameters
    ----------
    configfile : str or Path
        Path to CSV specification file
        
    Returns
    -------
    ModelConfig
        Model configuration object
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist
    ValueError
        If required columns are missing or data is invalid
    """
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    try:
        df = pd.read_csv(configfile)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {configfile}: {e}")
    
    return _load_config_from_dataframe(df)


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load DFMConfig from a pandas DataFrame (shared by CSV and BytesIO loading).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CSV columns
        
    Returns
    -------
    DFMConfig
        Model configuration
    """
    # Handle 'id' as alias for 'series_id' (for backward compatibility)
    # If CSV has data_code, item_id, api_source, generate series_id from them
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
    
    # Required fields (DFM-relevant only - no API/database fields)
    required_fields = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
    optional_fields = []  # No optional fields in generic DFM module
    
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Detect block columns (all columns that are not in required_fields or optional_fields)
    # Also exclude 'id' (if it was used as alias, we already created series_id from it)
    # and other non-DFM metadata columns (API/database fields are ignored by generic DFM module)
    # Preserve original column order from DataFrame - first block should be Global
    excluded_fields = set(required_fields) | set(optional_fields) | {
        'id', 'country', 'data_code', 'item_id', 'api_source', 'api_code', 
        'api_group_id', 'is_kpi', 'description', 'priority', 'is_active', 'metadata'
    }
    # Use DataFrame column order (not sorted) to preserve CSV structure
    block_columns = [col for col in df.columns if col not in excluded_fields]
    
    if not block_columns:
        raise ValueError("No block columns found. Expected columns like 'Global', 'Consumption', etc.")
    
    # Validate block columns contain only 0 or 1
    for block_col in block_columns:
        if not df[block_col].isin([0, 1]).all():
            raise ValueError(f"Block column '{block_col}' must contain only 0 or 1")
    
    # Ensure all series load on at least one block (first block should always be 1)
    if block_columns[0] not in df.columns:
        raise ValueError(f"First block column '{block_columns[0]}' is required")
    
    if not (df[block_columns[0]] == 1).all():
        raise ValueError(f"All series must load on the first block '{block_columns[0]}' (Global)")
    
    # Build blocks array (N x n_blocks)
    blocks_data = df[block_columns].values.astype(int)
    
    # Convert to ModelConfig format
    from .config import SeriesConfig
    
    series_list = []
    for idx, row in df.iterrows():
        # Build block array from block columns
        blocks = [int(row[col]) for col in block_columns]
        
        # Create SeriesConfig with only DFM-relevant fields
        # API/database fields (data_code, item_id, api_source, etc.) are ignored by generic DFM module
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


# Note: Excel support removed - use CSV or YAML configs instead


def load_config(configfile: Union[str, Path, io.BytesIO]) -> DFMConfig:
    """Load model configuration from file (auto-detects YAML or CSV).
    
    Parameters
    ----------
    configfile : str, Path, or BytesIO
        Path to configuration file (.yaml, .yml, or .csv), or BytesIO object with CSV content
        
    Returns
    -------
    ModelConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile does not exist (for file paths)
    ValueError
        If file format is not supported or configuration is invalid
    """
    # Handle BytesIO (for database storage downloads)
    if isinstance(configfile, io.BytesIO):
        # Read CSV from BytesIO
        configfile.seek(0)  # Reset to beginning
        df = pd.read_csv(configfile)
        # Use the same logic as load_config_from_csv but with DataFrame
        return _load_config_from_dataframe(df)
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Configuration file not found: {configfile}")
    
    suffix = configfile.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile)
    elif suffix == '.csv':
        return load_config_from_csv(configfile)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .csv")


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Apply transformation to a single series."""
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
    """Transform each data series based on specification.
    
    Handles all frequencies: daily (d), weekly (w), monthly (m), 
    quarterly (q), semi-annual (sa), annual (a).
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Frequency to step mapping (step = number of base periods per observation)
    # Base frequency is monthly, so step is months per observation
    freq_to_step = {
        'd': 1,   # Daily: 1 month per observation (aggregated to monthly)
        'w': 1,   # Weekly: 1 month per observation (aggregated to monthly)
        'm': 1,   # Monthly: 1 month per observation
        'q': 3,   # Quarterly: 3 months per observation
        'sa': 6,  # Semi-annual: 6 months per observation
        'a': 12,  # Annual: 12 months per observation
    }
    
    for i in range(N):
        freq = config.Frequency[i]
        step = freq_to_step.get(freq, 1)  # Default to 1 if unknown frequency
        X[:, i] = _transform_series(Z[:, i], config.Transformation[i], freq, step)
    
    # Drop initial observations based on highest frequency in data
    # Find maximum step (longest period) to determine drop period
    max_step = max([freq_to_step.get(f, 1) for f in config.Frequency])
    # Drop period should be at least max_step + 1 to handle transformations
    drop = max(4, max_step + 1)  # At least 4 for quarterly, more for annual
    
    if T > drop:
        return X[drop:], Time[drop:], Z[drop:]
    return X, Time, Z


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Read data from CSV file.
    
    CSV file should have:
    - First column: Date (YYYY-MM-DD format or pandas parseable)
    - Subsequent columns: Series data (one column per series)
    - Header row: Series IDs (matching config.SeriesID)
    
    Alternatively, CSV can have metadata columns first:
    - Metadata columns: series_id, series_name, frequency, transformation, category, units
    - Then date columns starting from the first date column
    """
    datafile = Path(datafile)
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read CSV file
    try:
        df = pd.read_csv(datafile)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {datafile}: {e}")
    
    # Check if first column is a date column or metadata
    first_col = df.columns[0]
    
    # Try to parse first column as date
    try:
        pd.to_datetime(df[first_col].iloc[0])
        is_date_first = True
    except (ValueError, TypeError):
        is_date_first = False
    
    # If first column is not a date, check if CSV is in "long" format (one row per series)
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
                raise ValueError(f"Could not find date column in CSV file {datafile}")
            
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
                raise ValueError(f"Could not find date column in CSV file {datafile}")
            
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
    """Sort series to match configuration order."""
    in_config = [m in config.SeriesID for m in Mnem]
    Mnem_filt = [m for m, in_c in zip(Mnem, in_config) if in_c]
    Z_filt = Z[:, in_config]
    
    perm = [Mnem_filt.index(sid) for sid in config.SeriesID]
    return Z_filt[:, perm], [Mnem_filt[i] for i in perm]


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load and transform data from CSV file.
    
    This function reads data from a CSV file, sorts it to match the model
    configuration (from CSV or YAML), and applies the specified transformations.
    
    Note: For database-backed applications, create adapters that implement
    the same interface (return X, Time, Z arrays).
    
    Parameters
    ----------
    datafile : str or Path
        Path to CSV data file (.csv)
    config : ModelConfig
        Model configuration object (from CSV or YAML)
    sample_start : pd.Timestamp or str, optional
        Start date for sample (YYYY-MM-DD). If None, uses all available data.
        Data before this date will be dropped.
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N), ready for DFM estimation
    Time : pd.DatetimeIndex
        Time index for the data
    Z : np.ndarray
        Original untransformed data (T x N), for reference
    """
    print('Loading data...')
    
    datafile = Path(datafile)
    if datafile.suffix.lower() != '.csv':
        raise ValueError('Only CSV files supported. Use database for production data loading.')
    
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")
    
    # Read data from CSV file
    Z, Time, Mnem = read_data(datafile)
    
    # Process data: sort to match config order, then transform
    Z, _ = sort_data(Z, Mnem, config)
    X, Time, Z = transform_data(Z, Time, config)
    
    # Apply sample_start filtering
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start = pd.to_datetime(sample_start)
        mask = Time >= sample_start
        Time, X, Z = Time[mask], X[mask], Z[mask]
    
    return X, Time, Z


# Database adapters are application-specific and should be implemented separately.
# This keeps the DFM module generic and database-agnostic.
