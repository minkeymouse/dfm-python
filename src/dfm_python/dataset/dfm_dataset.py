"""PyTorch Dataset class for Dynamic Factor Model (DFM).

This module provides dataset implementation for DFM training.
Handles data loading, preprocessing, and mixed-frequency parameter setup.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..config import DFMConfig

from ..logger import get_logger
from ..config import DFMConfig
from ..numeric.tent import get_tent_weights, generate_R_mat
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    TENT_WEIGHTS_LOOKUP,
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
)
from ..utils.misc import get_clock_frequency
from ..dataset.time import TimeIndex
from ..utils.errors import DataValidationError, ConfigurationError

_logger = get_logger(__name__)


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
        time_list = pd.to_datetime(time_data.iloc[:, 0]).tolist()
    else:
        time_list = [pd.to_datetime(' '.join(str(val) for val in row)) for row in time_data.values]
    
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
        _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        return df, extracted_time_index
    
    return df, time_index


class DFMDataset:
    """Dataset for DFM training.
    
    This dataset handles data loading, preprocessing, and mixed-frequency parameter setup
    for DFM models. Unlike DDFM and KDFM, DFM doesn't use PyTorch Dataset for training
    (it uses NumPy arrays), but this class provides the same interface for consistency.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    config_path : str or Path, optional
        Path to configuration file
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed (imputation, scaling, etc.)
        before passing to this Dataset.
    target_series : str or List[str], optional
        Target series column names. Can be a single string or list of strings.
    time_index : str, List[str], or TimeIndex, optional
        Time index for the data. Can be TimeIndex object, column name(s), or None.
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target_series: Optional[Union[str, List[str]]] = None,
        time_index: Optional[Union[str, List[str], TimeIndex]] = None,
        **kwargs
    ):
        """Initialize DFM dataset with data loading and preprocessing."""
        # Store attributes (DFMDataset doesn't inherit from a class that accepts these)
        self.config = config
        self.config_path = config_path
        self.target_series = target_series
        
        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data
        
        # Setup time_index
        time_index_column = kwargs.pop('time_index_column', None)
        self.time_index, self.time_index_column = setup_time_index(time_index, time_index_column)
        
        # Will be set in _setup()
        self.data_processed: Optional[np.ndarray] = None
        
        # Mixed frequency detection (internal property, set during setup)
        self._is_mixed_freq: bool = False
        
        # Mixed frequency parameters (set during setup if mixed frequency detected)
        self._constraint_matrix: Optional[np.ndarray] = None
        self._constraint_vector: Optional[np.ndarray] = None
        self._n_slower_freq: int = 0
        self._tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
        self._frequencies: Optional[np.ndarray] = None
        self._idio_indicator: Optional[np.ndarray] = None
        self._idio_chain_lengths: Optional[np.ndarray] = None
        
        # Setup data
        self._setup()
    
    def _setup(self) -> None:
        """Load and prepare data, setup mixed-frequency parameters."""
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise DataValidationError(
                    "DFMDataset setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame.",
                    details="Both data and data_path are None. One must be provided."
                )
            
            # Load data from file using pandas
            data = pd.read_csv(self.data_path)
            X = data.drop(columns=[data.columns[0]]).values  # Assume first column is time
            Time = pd.to_datetime(data.iloc[:, 0])
            Z = None  # Z not used in current implementation
            self.data = X
            self.time_index = Time
        
        # Convert to pandas DataFrame if needed
        X_df = convert_to_dataframe(self.data, self.config)
        
        # Extract time index from column if specified
        X_df, self.time_index = extract_time_index_if_needed(X_df, self.time_index, self.time_index_column)
        
        # Separate target and feature columns
        all_columns = list(X_df.columns)
        # If target_series is None or empty, use all columns as targets
        if self.target_series is None:
            target_cols = all_columns
        elif isinstance(self.target_series, str):
            target_cols = [self.target_series] if self.target_series in all_columns else []
        elif len(self.target_series) == 0:
            target_cols = all_columns
        else:
            target_cols = [col for col in self.target_series if col in all_columns]
        
        # Convert to numpy - filter numeric columns
        X_transformed = filter_numeric_columns(X_df)
        
        X_processed_np = X_transformed.to_numpy().astype(np.float32)
        self.data_processed = X_processed_np
        
        # Store processed columns for use in _setup_mixed_frequency_params
        self._processed_columns = all_columns
        
        # Detect and cache mixed-frequency status
        self._is_mixed_freq = self._is_mixed_frequency(all_columns)
        
        # Setup mixed-frequency parameters if detected
        if self._is_mixed_freq:
            self._setup_mixed_frequency_params()
        else:
            # Initialize unified frequency parameters
            n_features = self.data_processed.shape[1] if self.data_processed is not None else 0
            self._constraint_matrix = None
            self._constraint_vector = None
            self._n_slower_freq = 0
            self._tent_weights_dict = None
            self._frequencies = None
            self._idio_indicator = np.ones(n_features, dtype=np.float32)
            self._idio_chain_lengths = np.zeros(n_features, dtype=np.int32)
    
    
    
    def _is_mixed_frequency(self, columns: Optional[List[str]] = None) -> bool:
        """Check if data has mixed frequencies (some series slower than clock)."""
        clock = get_clock_frequency(self.config)
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        
        frequencies = self.config.get_frequencies(columns)
        if not frequencies:
            return False
        
        for freq in frequencies:
            freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
            if freq_hierarchy > clock_hierarchy:
                return True
        
        return False
    
    def _setup_mixed_frequency_params(self) -> None:
        """Setup mixed-frequency parameters from config and data."""
        if self.data_processed is None:
            raise ConfigurationError(
                "DFMDataset: data_processed must be set before calling _setup_mixed_frequency_params()",
                details="Call _setup() first to load and process data."
            )
        
        clock = get_clock_frequency(self.config)
        # Use processed columns to ensure frequencies_list matches data_processed.shape[1]
        if hasattr(self, '_processed_columns') and self._processed_columns is not None:
            all_columns = self._processed_columns
        else:
            # Fallback: try to get columns from data, excluding time_index_column
            if isinstance(self.data, pd.DataFrame):
                time_cols = [self.time_index_column] if isinstance(self.time_index_column, str) else (self.time_index_column if self.time_index_column else [])
                all_columns = [col for col in self.data.columns if col not in time_cols]
            else:
                all_columns = None
        
        # Get frequencies using new API
        frequencies_list = self.config.get_frequencies(all_columns)
        frequencies_set = set(frequencies_list)
        
        # Update idio flags based on detected frequencies
        if hasattr(self.config, '_update_idio_flags_from_frequencies'):
            self.config._update_idio_flags_from_frequencies(frequencies_list)
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        
        # Validate frequency pairs
        missing_pairs = [
            (freq, clock) for freq in frequencies_set
                    if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy and get_tent_weights(freq, clock) is None
        ]
        if missing_pairs:
            raise DataValidationError(
                f"Mixed-frequency data detected but the following frequency pairs are not in TENT_WEIGHTS_LOOKUP: {missing_pairs}. "
                f"Available pairs: {list(TENT_WEIGHTS_LOOKUP.keys())}. "
                f"Either add the missing pairs to TENT_WEIGHTS_LOOKUP or ensure all series use clock frequency.",
                details=f"Frequency pairs without tent weights: {missing_pairs}"
            )
        
        # Get aggregation structure
        tent_weights_dict = {}
        for freq in frequencies_set:
            if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy:
                tent_w = get_tent_weights(freq, clock)
                if tent_w is not None:
                    tent_weights_dict[freq] = np.array(tent_w, dtype=np.float32)
        
        # Validate: DFM supports only clock + one slower frequency
        if len(tent_weights_dict) > 1:
            slower_freqs = list(tent_weights_dict.keys())
            raise DataValidationError(
                f"DFM supports only one slower frequency, but found {len(tent_weights_dict)} slower frequencies: {slower_freqs}. "
                f"Please ensure all slower-frequency series use the same frequency, or use a different clock frequency.",
                details=f"Slower frequencies detected: {slower_freqs}, clock: {clock}"
            )
        
        # Generate constraint matrices if needed
        R_mat = None
        q = None
        if tent_weights_dict:
            # Use the single tent weight to generate constraint matrix
            first_tent_weights = list(tent_weights_dict.values())[0]
            R_mat, q = generate_R_mat(first_tent_weights)
            R_mat = np.array(R_mat, dtype=np.float32)
            q = np.array(q, dtype=np.float32)
        
        n_slower_freq = sum(1 for freq in frequencies_list if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy)
        idio_indicator = np.array([1 if freq == clock else 0 for freq in frequencies_list], dtype=np.float32)
        # Map frequencies to hierarchy values
        frequencies_np = np.array([
            FREQUENCY_HIERARCHY.get(f, FREQUENCY_HIERARCHY.get(DEFAULT_CLOCK_FREQUENCY, DEFAULT_HIERARCHY_VALUE))
            for f in frequencies_list
        ], dtype=np.int32)
        
        self._constraint_matrix = R_mat
        self._constraint_vector = q
        self._n_slower_freq = n_slower_freq
        self._tent_weights_dict = tent_weights_dict
        self._frequencies = frequencies_np
        self._idio_indicator = idio_indicator
        n_features = self.data_processed.shape[1] if self.data_processed is not None else len(idio_indicator)
        self._idio_chain_lengths = np.zeros(n_features, dtype=np.int32)
    
    def get_processed_data(self) -> np.ndarray:
        """Get processed data array."""
        if self.data_processed is None:
            raise ConfigurationError(
                "DFMDataset: data_processed not available. "
                "Please ensure dataset was initialized with data.",
                details="get_processed_data() requires data_processed attribute."
            )
        return self.data_processed
    
    def get_initialization_params(self) -> Dict[str, Any]:
        """Get parameters needed for DFM initialization.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'X': processed data array (all series, including features and targets)
            - 'target_scaler': sklearn scaler for target series inverse transformation (if available)
            - 'R_mat': constraint matrix (if mixed-frequency detected)
            - 'q': constraint vector (if mixed-frequency detected)
            - 'n_slower_freq': number of slower frequency series
            - 'tent_weights_dict': tent weights dictionary
            - 'frequencies': frequency array
            - 'idio_indicator': idiosyncratic indicator array
            - 'idio_chain_lengths': idiosyncratic chain lengths
            - 'opt_nan': missing data handling options
            - 'clock': clock frequency
            - 'is_mixed_freq': whether data has mixed frequencies (internal property)
        """
        if self.data_processed is None:
            raise ConfigurationError(
                "DFMDataset: data_processed not available. "
                "Please ensure dataset was initialized with data.",
                details="get_initialization_params() requires data_processed attribute."
            )
        
        # Get target_scaler from dataset attribute or config
        target_scaler = getattr(self, 'target_scaler', None)
        if target_scaler is None and self.config is not None:
            target_scaler = getattr(self.config, 'target_scaler', None)
        
        return {
            'X': self.data_processed,
            'target_scaler': target_scaler,
            'R_mat': self._constraint_matrix,
            'q': self._constraint_vector,
            'n_slower_freq': self._n_slower_freq,
            'tent_weights_dict': self._tent_weights_dict,
            'frequencies': self._frequencies,
            'idio_indicator': self._idio_indicator,
            'idio_chain_lengths': self._idio_chain_lengths,
            'opt_nan': {'method': DEFAULT_NAN_METHOD, 'k': DEFAULT_NAN_K},
            'clock': get_clock_frequency(self.config),
            'is_mixed_freq': self._is_mixed_freq
        }

