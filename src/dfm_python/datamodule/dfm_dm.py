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
from ..numeric.tent import get_agg_structure, get_tent_weights
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    TENT_WEIGHTS_LOOKUP,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_CLOCK_FREQUENCY,
)
from ..utils.misc import get_clock_frequency
from ..dataset.process import TimeIndex
from ..logger import get_logger

_logger = get_logger(__name__)

# Import frequency hierarchy from constants
from ..config.constants import FREQUENCY_HIERARCHY


class DFMDataModule(BaseDataModule):
    """Custom DataModule for DFM (not inheriting from PyTorch Lightning).
    
    This DataModule handles:
    - Data loading (assumes data is preprocessed)
    - Mixed-frequency parameter setup
    - Parameter initialization preparation
    
    **Important**: 
    - Data must be **preprocessed** before passing to this DataModule (imputation, scaling, etc.)
    - Data is assumed to be standardized (mean=0, std=1) for all series
    - Optional `scaler` parameter can be used to extract Mx/Wx for inverse transformation
    - If no scaler provided, defaults to Mx=0, Wx=1 (assuming standardized data)
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    data_path : str or Path, optional
        Path to data file (CSV)
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed (imputation, scaling, etc.)
        before passing to this DataModule.
    scaler : Any, optional
        Optional scaler for extracting Mx/Wx statistics. Can be:
        - `None` (default): Assumes data is standardized (Mx=0, Wx=1)
        - Scaler instance: Fitted scaler object (e.g., StandardScaler, RobustScaler)
        - The scaler is only used to extract mean/scale statistics, not to transform data
    time_index : TimeIndex, optional
        Time index for the data
    time_index_column : str or list of str, optional
        Column name(s) in DataFrame to use as time index
    mixed_freq : bool, default False
        Whether to use mixed-frequency handling
    nan_method : int, default 2
        Missing data handling method (for internal use)
    nan_k : int, default 3
        Spline interpolation order (for internal use)
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        scaler: Optional[Any] = None,
        time_index: Optional[TimeIndex] = None,
        time_index_column: Optional[Union[str, List[str]]] = None,
        mixed_freq: bool = False,
        nan_method: Optional[int] = None,
        nan_k: Optional[int] = None,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            config=config,
            config_path=config_path,
            data_path=data_path,
            data=data,
            time_index=time_index,
            time_index_column=time_index_column,
            **kwargs
        )
        
        self.scaler = scaler
        self.mixed_freq = mixed_freq
        self.nan_method = nan_method if nan_method is not None else DEFAULT_NAN_METHOD
        self.nan_k = nan_k if nan_k is not None else DEFAULT_NAN_K
        
        # Will be set in setup()
        self.data_processed: Optional[np.ndarray] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        
        # Mixed frequency parameters (set during setup)
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
                raise ValueError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame."
                )
            
            # Load data from file using base class method
            X, Time, Z = self.load_data(self.data_path)
            self.data = X
            self.time_index = Time
        
        # Convert to pandas DataFrame if needed
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pd.DataFrame(self.data, columns=pd.Index(series_ids))
        elif isinstance(self.data, pd.DataFrame):
            X_df = self.data.copy()
        else:
            raise TypeError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or pandas.DataFrame."
            )
        
        # Extract time index from column if specified
        if self.time_index is None and self.time_index_column is not None:
            if not isinstance(X_df, pd.DataFrame):
                raise ValueError(
                    "time_index_column can only be used with DataFrame input. "
                    "Please provide data as pandas.DataFrame."
                )
            
            time_cols = [self.time_index_column] if isinstance(self.time_index_column, str) else self.time_index_column
            
            missing_cols = [col for col in time_cols if col not in X_df.columns]
            if missing_cols:
                raise ValueError(
                    f"time_index_column(s) {missing_cols} not found in DataFrame. "
                    f"Available columns: {list(X_df.columns)}"
                )
            
            time_data = X_df[time_cols]
            
            from ..dataset.process import parse_timestamp
            if len(time_cols) == 1:
                time_list = [parse_timestamp(str(val)) for val in time_data.iloc[:, 0]]
            else:
                time_list = [parse_timestamp(' '.join(str(val) for val in row)) for row in time_data.values]
            
            self.time_index = TimeIndex(time_list)
            X_df = X_df.drop(columns=time_cols)
            _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        
        # Data is assumed to be preprocessed - use as-is
        X_transformed = X_df.copy()
        
        # Extract Mx/Wx from scaler if provided
        from ..dataset.process import _extract_mx_wx
        X_processed_np = X_transformed.to_numpy()
        try:
            self.Mx, self.Wx = _extract_mx_wx(self.scaler, X_processed_np)
        except (AttributeError, ImportError) as e:
            _logger.warning(
                f"Could not extract Mx/Wx from scaler: {e}. "
                f"Assuming data is standardized (Mx=0, Wx=1)."
            )
            self.Mx = None
            self.Wx = None
        
        # Convert to numpy - filter numeric columns
        from ..datamodule.base import _filter_numeric_columns
        X_transformed = _filter_numeric_columns(X_transformed, _logger)
        
        X_processed_np = X_transformed.to_numpy().astype(np.float32)
        self.data_processed = X_processed_np
        
        # Set defaults if Mx/Wx not extracted from scaler
        from ..datamodule.base import _set_default_mx_wx
        self.Mx, self.Wx = _set_default_mx_wx(self.Mx, self.Wx, X_processed_np.shape[1], self.scaler, _logger)
        
        # Setup mixed-frequency parameters
        if self.mixed_freq:
            self._setup_mixed_frequency_params()
    
    def _setup_mixed_frequency_params(self) -> None:
        """Setup mixed-frequency parameters from config and data."""
        self._check_setup('_setup_mixed_frequency_params')
        
        clock = get_clock_frequency(self.config)
        
        if not self.mixed_freq:
            self._constraint_matrix = None
            self._constraint_vector = None
            self._n_slower_freq = 0
            self._tent_weights_dict = None
            self._frequencies = None
            n_features = self.data_processed.shape[1] if self.data_processed is not None else 0
            self._idio_indicator = np.ones(n_features, dtype=np.float32)
            self._idio_chain_lengths = np.zeros(n_features, dtype=np.int32)
            return
        
        agg_structure = get_agg_structure(self.config, clock=clock)
        frequencies_list = [s.frequency for s in self.config.series]
        frequencies_set = set(frequencies_list)
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        
        # Validate frequency pairs
        missing_pairs = [
            (freq, clock) for freq in frequencies_set
                    if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy and get_tent_weights(freq, clock) is None
        ]
        if missing_pairs:
            raise ValueError(
                f"mixed_freq=True but the following frequency pairs are not in TENT_WEIGHTS_LOOKUP: {missing_pairs}. "
                f"Available pairs: {list(TENT_WEIGHTS_LOOKUP.keys())}. "
                f"Either add the missing pairs to TENT_WEIGHTS_LOOKUP or set mixed_freq=False."
            )
        
        tent_weights_dict = {k: np.array(v, dtype=np.float32) for k, v in agg_structure['tent_weights'].items()}
        
        R_mat = None
        q = None
        if agg_structure['structures']:
            first_structure = list(agg_structure['structures'].values())[0]
            R_mat = np.array(first_structure[0], dtype=np.float32)
            q = np.array(first_structure[1], dtype=np.float32)
        
        n_slower_freq = sum(1 for freq in frequencies_list if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy)
        idio_indicator = np.array([1 if freq == clock else 0 for freq in frequencies_list], dtype=np.float32)
        # Map frequencies to hierarchy values (default to monthly=3 if not found)
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
            - 'X': processed data array
            - 'Mx': mean values for unstandardization
            - 'Wx': standard deviation values for unstandardization
            - 'R_mat': constraint matrix (if mixed_freq)
            - 'q': constraint vector (if mixed_freq)
            - 'n_slower_freq': number of slower frequency series
            - 'tent_weights_dict': tent weights dictionary
            - 'frequencies': frequency array
            - 'idio_indicator': idiosyncratic indicator array
            - 'idio_chain_lengths': idiosyncratic chain lengths
            - 'opt_nan': missing data handling options
            - 'clock': clock frequency
        """
        self._check_setup('get_initialization_params')
        
        return {
            'X': self.data_processed,
            'Mx': self.Mx,
            'Wx': self.Wx,
            'R_mat': self._constraint_matrix,
            'q': self._constraint_vector,
            'n_slower_freq': self._n_slower_freq,
            'tent_weights_dict': self._tent_weights_dict,
            'frequencies': self._frequencies,
            'idio_indicator': self._idio_indicator,
            'idio_chain_lengths': self._idio_chain_lengths,
            'opt_nan': {'method': self.nan_method, 'k': self.nan_k},
            'clock': get_clock_frequency(self.config)
        }
    
    def get_std_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get standardization parameters (Mx, Wx)."""
        self._check_setup('get_std_params')
        return self.Mx, self.Wx
    
    def get_processed_data(self) -> np.ndarray:
        """Get processed data array."""
        self._check_setup('get_processed_data')
        if self.data_processed is None:
            raise RuntimeError("DataModule setup() must be called before get_processed_data()")
        return self.data_processed

