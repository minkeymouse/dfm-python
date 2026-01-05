"""Common utilities for dataset initialization and preprocessing.

This module provides shared functionality for all dataset classes to avoid duplication.
"""

from typing import Optional, Union, List
from pathlib import Path
import pandas as pd
import numpy as np

from ..config import DFMConfig, YamlSource
from ..dataset.process import TimeIndex
from ..utils.errors import ConfigurationError
from ..utils.misc import get_config_attr


def load_config(config: Optional[DFMConfig], config_path: Optional[Union[str, Path]]) -> DFMConfig:
    """Load configuration from config object or config_path.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    config_path : str or Path, optional
        Path to configuration file
        
    Returns
    -------
    DFMConfig
        Loaded configuration object
        
    Raises
    ------
    ConfigurationError
        If both config and config_path are None
    """
    if config is None and config_path is not None:
        source = YamlSource(config_path)
        config = source.load()
    
    if config is None:
        raise ConfigurationError(
            "Dataset initialization failed: either config or config_path must be provided. "
            "Please provide a DFMConfig object or a path to a configuration file.",
            details="Both config and config_path are None. One must be provided."
        )
    
    return config


def normalize_target_series(target_series: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize target_series to a list.
    
    Parameters
    ----------
    target_series : str, List[str], or None
        Target series specification
        
    Returns
    -------
    List[str]
        Normalized list of target series (empty list if None)
    """
    if target_series is None:
        return []
    elif isinstance(target_series, str):
        return [target_series]
    else:
        return list(target_series)


def setup_time_index(
    time_index: Optional[Union[str, List[str], TimeIndex]],
    time_index_column: Optional[Union[str, List[str]]] = None
) -> tuple[Optional[TimeIndex], Optional[Union[str, List[str]]]]:
    """Setup time_index and time_index_column attributes.
    
    Parameters
    ----------
    time_index : str, List[str], TimeIndex, or None
        Time index specification
    time_index_column : str, List[str], or None
        Legacy time_index_column parameter (for backward compatibility)
        
    Returns
    -------
    tuple[Optional[TimeIndex], Optional[Union[str, List[str]]]]
        (time_index, time_index_column) tuple
    """
    if time_index is None and time_index_column is not None:
        time_index = time_index_column
    
    if isinstance(time_index, TimeIndex):
        return time_index, None
    elif isinstance(time_index, (str, list)):
        return None, time_index
    else:
        return None, None


def convert_to_dataframe(data: Union[np.ndarray, pd.DataFrame], config: DFMConfig) -> pd.DataFrame:
    """Convert data to pandas DataFrame.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    config : DFMConfig
        Configuration object for series ID generation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with proper column names
    """
    if isinstance(data, np.ndarray):
        columns = [f"series_{i}" for i in range(data.shape[1])]
        series_ids = config.get_series_ids(columns)
        return pd.DataFrame(data, columns=pd.Index(series_ids))
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        from ..utils.errors import DataValidationError
        raise DataValidationError(
            f"Unsupported data type {type(data)}. "
            f"Please provide data as numpy.ndarray or pandas.DataFrame.",
            details=f"Received type: {type(data).__name__}. Expected: numpy.ndarray or pandas.DataFrame."
        )


def filter_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with only numeric columns
    """
    return df.select_dtypes(include=[np.number])


def extract_time_index_from_dataframe(
    df: pd.DataFrame,
    time_index_column: Union[str, List[str]]
) -> TimeIndex:
    """Extract time index from DataFrame using time_index_column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract time index from
    time_index_column : str or List[str]
        Column name(s) containing time index
        
    Returns
    -------
    TimeIndex
        Extracted time index
        
    Raises
    ------
    ConfigurationError
        If time_index_column is None
    DataValidationError
        If time_index_column(s) not found in DataFrame
    """
    from ..dataset.process import parse_timestamp
    from ..utils.errors import ConfigurationError, DataValidationError
    
    if time_index_column is None:
        raise ConfigurationError(
            "time_index_column must be set to extract time index from DataFrame",
            details="time_index_column attribute is None. Set it before calling extract_time_index()."
        )
    
    time_cols = [time_index_column] if isinstance(time_index_column, str) else time_index_column
    
    missing_cols = [col for col in time_cols if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"time_index_column(s) {missing_cols} not found in DataFrame. "
            f"Available columns: {list(df.columns)}",
            details=f"Requested columns: {missing_cols}. DataFrame has {len(df.columns)} columns."
        )
    
    time_data = df[time_cols]
    
    if len(time_cols) == 1:
        time_list = [parse_timestamp(str(val)) for val in time_data.iloc[:, 0]]
    else:
        time_list = [parse_timestamp(' '.join(str(val) for val in row)) for row in time_data.values]
    
    return TimeIndex(time_list)


def extract_time_index_if_needed(
    df: pd.DataFrame,
    time_index: Optional[TimeIndex],
    time_index_column: Optional[Union[str, List[str]]]
) -> tuple[pd.DataFrame, Optional[TimeIndex]]:
    """Extract time index from DataFrame if needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract time index from
    time_index : TimeIndex or None
        Existing time index (if already extracted)
    time_index_column : str, List[str], or None
        Column name(s) containing time index
        
    Returns
    -------
    tuple[pd.DataFrame, Optional[TimeIndex]]
        (DataFrame with time columns removed, extracted TimeIndex or None)
    """
    from ..utils.errors import DataValidationError
    
    if time_index is None and time_index_column is not None:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                "time_index_column can only be used with DataFrame input. "
                "Please provide data as pandas.DataFrame.",
                details=f"time_index_column is set but data is {type(df).__name__}, not DataFrame."
            )
        
        extracted_time_index = extract_time_index_from_dataframe(df, time_index_column)
        time_cols = [time_index_column] if isinstance(time_index_column, str) else time_index_column
        df = df.drop(columns=time_cols)
        from ..logger import get_logger
        _logger = get_logger(__name__)
        _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        return df, extracted_time_index
    
    return df, time_index

