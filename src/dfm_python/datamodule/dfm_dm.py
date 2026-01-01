"""Custom DFM DataModule for initialization and data handling.

This module provides a custom DFMDataModule that doesn't inherit from PyTorch Lightning.
It handles data loading, preprocessing, and parameter initialization for DFM models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any, List
from pathlib import Path

from .base import BaseDataModule
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
from ..dataset.process import TimeIndex
from ..logger import get_logger
from ..utils.errors import DataValidationError, ConfigurationError

_logger = get_logger(__name__)


class DFMDataModule(BaseDataModule):
    """Custom DataModule for DFM (not inheriting from PyTorch Lightning).
    
    This DataModule handles:
    - Data loading (assumes data is preprocessed)
    - Mixed-frequency parameter setup
    - Parameter initialization preparation
    
    **Important**: 
    - Data must be **preprocessed** before passing to this DataModule (imputation, scaling, etc.)
    - Feature series (non-target) are assumed to be manually preprocessed
    - Target series scaling: If you want to scale target series, provide a **fitted** sklearn scaler 
      (e.g., StandardScaler, RobustScaler) in config.target_scaler. The scaler must be fitted on 
      target data before passing to the config. Use scaler.inverse_transform() for unstandardization.
    - If no target_scaler provided, target series are assumed to be already in the desired scale
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    data_path : str or Path, optional
        Path to data file (CSV)
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed (imputation, scaling, etc.)
        before passing to this DataModule.
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
        # Get target_scaler from config if available, otherwise None
        target_scaler = config.target_scaler if config is not None else None
        # Initialize base class (handles target_series, target_scaler, time_index)
        super().__init__(
            config=config,
            config_path=config_path,
            data_path=data_path,
            data=data,
            target_series=target_series,
            target_scaler=target_scaler,
            time_index=time_index,
            **kwargs
        )
        
        # Will be set in setup()
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
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data, setup mixed-frequency parameters."""
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise DataValidationError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame.",
                    details="Both data and data_path are None. One must be provided."
                )
            
            # Load data from file using base class method
            X, Time, Z = self.load_data(self.data_path)
            self.data = X
            self.time_index = Time
        
        # Convert to pandas DataFrame if needed
        if isinstance(self.data, np.ndarray):
            # Auto-create frequency dict from data shape
            columns = [f"series_{i}" for i in range(self.data.shape[1])]
            series_ids = self.config.get_series_ids(columns)
            X_df = pd.DataFrame(self.data, columns=pd.Index(series_ids))
        elif isinstance(self.data, pd.DataFrame):
            X_df = self.data.copy()
        else:
            raise DataValidationError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or pandas.DataFrame.",
                details=f"Received type: {type(self.data).__name__}. Expected: numpy.ndarray or pandas.DataFrame."
            )
        
        # Extract time index from column if specified
        if self.time_index is None and self.time_index_column is not None:
            if not isinstance(X_df, pd.DataFrame):
                raise DataValidationError(
                    "time_index_column can only be used with DataFrame input. "
                    "Please provide data as pandas.DataFrame.",
                    details=f"time_index_column is set but data is {type(X_df).__name__}, not DataFrame."
                )
            
            self.time_index = self._extract_time_index_from_dataframe(X_df)
            time_cols = [self.time_index_column] if isinstance(self.time_index_column, str) else self.time_index_column
            X_df = X_df.drop(columns=time_cols)
            _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        
        # Data is assumed to be preprocessed - use as-is
        X_transformed = X_df.copy()
        
        # Separate target and feature columns
        all_columns = list(X_df.columns)
        target_cols = [col for col in self.target_series if col in all_columns]
        
        # Convert to numpy - filter numeric columns
        X_transformed = self._filter_numeric_columns(X_transformed)
        
        X_processed_np = X_transformed.to_numpy().astype(np.float32)
        self.data_processed = X_processed_np
        
        # Store processed columns for use in _setup_mixed_frequency_params
        # This ensures frequencies_list matches data_processed.shape[1]
        self._processed_columns = all_columns
        
        # Target scaler is stored in self.target_scaler (from base class)
        # No need to extract Mx/Wx - use scaler.inverse_transform() directly
        
        # Detect and cache mixed-frequency status (internal property)
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
    
    def _setup_mixed_frequency_params(self) -> None:
        """Setup mixed-frequency parameters from config and data."""
        self._check_setup('_setup_mixed_frequency_params')
        
        clock = get_clock_frequency(self.config)
        # Use processed columns (after time_index extraction) to ensure frequencies_list matches data_processed.shape[1]
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
        
        # Get aggregation structure (note: get_agg_structure may need updating for new API)
        # For now, we'll compute tent weights directly
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
        self._check_setup('get_initialization_params')
        
        return {
            'X': self.data_processed,
            'target_scaler': self.target_scaler,
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
    
    
    def get_processed_data(self) -> np.ndarray:
        """Get processed data array."""
        self._check_setup('get_processed_data')
        if self.data_processed is None:
            raise ConfigurationError(
                "DataModule setup() must be called before get_processed_data()",
                details="get_processed_data() requires data_processed attribute which is set by setup()."
            )
        return self.data_processed

