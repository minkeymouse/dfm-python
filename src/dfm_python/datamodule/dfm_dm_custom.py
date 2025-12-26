"""Custom DFM DataModule for initialization and data handling.

This module provides a custom DFMDataModule that doesn't inherit from PyTorch Lightning.
It handles data loading, preprocessing, and parameter initialization for DFM models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any, List
from pathlib import Path

from ..config import DFMConfig
from ..config.utils import get_agg_structure, get_tent_weights, FREQUENCY_HIERARCHY, TENT_WEIGHTS_LOOKUP
from ..data.utils import load_data as _load_data
from ..utils.time import TimeIndex
from ..logger import get_logger
from .utils import (
    _check_sktime,
    _get_scaler,
    _get_mean,
    _get_scale,
    _compute_mx_wx,
    create_passthrough_transformer,
)

_logger = get_logger(__name__)

# Constants
_FREQ_TO_INT = {'d': 1, 'w': 2, 'm': 3, 'q': 4, 'sa': 5, 'a': 6}


class DFMDataModule:
    """Custom DataModule for DFM (not inheriting from PyTorch Lightning).
    
    This DataModule handles:
    - Data loading and preprocessing
    - Mixed-frequency parameter setup
    - Parameter initialization preparation
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    pipeline : Any, optional
        sktime-compatible preprocessing pipeline for statistics extraction
    data_path : str or Path, optional
        Path to data file (CSV)
    data : np.ndarray or pd.DataFrame, optional
        Data array or DataFrame
    preprocessed : bool, default False
        Whether data is already preprocessed
    time_index : TimeIndex, optional
        Time index for the data
    time_index_column : str or list of str, optional
        Column name(s) in DataFrame to use as time index
    mixed_freq : bool, default False
        Whether to use mixed-frequency handling
    nan_method : int, default 2
        Missing data handling method
    nan_k : int, default 3
        Spline interpolation order
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        pipeline: Optional[Any] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        preprocessed: bool = False,
        time_index: Optional[TimeIndex] = None,
        time_index_column: Optional[Union[str, List[str]]] = None,
        mixed_freq: bool = False,
        nan_method: int = 2,
        nan_k: int = 3,
        **kwargs
    ):
        _check_sktime()
        
        # Load config if config_path provided
        if config is None and config_path is not None:
            from ..config import YamlSource
            source = YamlSource(config_path)
            config = source.load()
        
        if config is None:
            raise ValueError(
                "DataModule initialization failed: either config or config_path must be provided. "
                "Please provide a DFMConfig object or a path to a configuration file."
            )
        
        self.config = config
        self.pipeline = pipeline
        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data
        self.preprocessed = preprocessed
        self.time_index = time_index
        self.time_index_column = time_index_column
        self.mixed_freq = mixed_freq
        self.nan_method = nan_method
        self.nan_k = nan_k
        
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
        self._i_idio: Optional[np.ndarray] = None
        self._idio_chain_lengths: Optional[np.ndarray] = None
    
    def setup(self) -> None:
        """Load and prepare data, setup mixed-frequency parameters."""
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise ValueError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame."
                )
            
            # Load data from file
            X, Time, Z = _load_data(
                self.data_path,
                self.config,
            )
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
            
            from ..utils.time import parse_timestamp
            if len(time_cols) == 1:
                time_list = [parse_timestamp(str(val)) for val in time_data.iloc[:, 0]]
            else:
                time_list = [parse_timestamp(' '.join(str(val) for val in row)) for row in time_data.values]
            
            self.time_index = TimeIndex(time_list)
            X_df = X_df.drop(columns=time_cols)
            _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        
        # Determine pipeline to use
        if self.pipeline is None:
            pipeline_to_use = create_passthrough_transformer()
        else:
            pipeline_to_use = self.pipeline
        
        # Set pandas output for sktime pipelines
        try:
            if hasattr(pipeline_to_use, 'set_output'):
                pipeline_to_use.set_output(transform="pandas")
        except (AttributeError, ValueError) as e:
            _logger.debug(f"Could not set pandas output on pipeline: {e}")
        
        # Apply pipeline based on preprocessed flag
        if self.preprocessed:
            X_transformed = X_df.copy()
            if pipeline_to_use is not None:
                try:
                    scaler = _get_scaler(pipeline_to_use)
                    if scaler is not None:
                        X_processed_np = X_transformed.to_numpy()
                        self.Mx = _get_mean(scaler, X_processed_np)
                        self.Wx = _get_scale(scaler, X_processed_np)
                except (AttributeError, ImportError, Exception) as e:
                    _logger.debug(f"Could not extract scaler from pipeline: {e}")
        else:
            try:
                X_transformed = pipeline_to_use.fit_transform(X_df)
            except Exception as e:
                raise ValueError(
                    f"DataModule setup failed: pipeline fit_transform error: {e}. "
                    f"Ensure pipeline is sktime-compatible and supports pandas DataFrames."
                ) from e
        
        # Ensure output is pandas DataFrame
        if not isinstance(X_transformed, pd.DataFrame):
            if isinstance(X_transformed, np.ndarray):
                n_cols = X_transformed.shape[1] if len(X_transformed.shape) > 1 else 1
                if n_cols == len(X_df.columns):
                    X_transformed = pd.DataFrame(X_transformed, columns=pd.Index(X_df.columns))
                else:
                    X_transformed = pd.DataFrame(X_transformed, 
                        columns=pd.Index([f'feature_{i}' for i in range(n_cols)]))
                if len(X_transformed) == len(X_df):
                    X_transformed.index = X_df.index
            else:
                raise TypeError(
                    f"DataModule setup failed: pipeline returned unsupported type {type(X_transformed)}. "
                    f"Expected pandas.DataFrame or numpy.ndarray."
                )
        
        # Convert to numpy
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X_transformed.columns):
            non_numeric = [col for col in X_transformed.columns if col not in numeric_cols]
            _logger.warning(f"Excluding non-numeric columns from data: {non_numeric}")
            X_transformed = X_transformed[numeric_cols]
        
        X_processed_np = X_transformed.to_numpy().astype(np.float32)
        self.data_processed = X_processed_np
        
        # Extract standardization parameters if not already extracted
        if not self.preprocessed:
            try:
                scaler = _get_scaler(pipeline_to_use)
                if scaler is not None:
                    self.Mx = _get_mean(scaler, X_processed_np)
                    self.Wx = _get_scale(scaler, X_processed_np)
            except (AttributeError, ImportError, Exception) as e:
                _logger.debug(f"Could not extract scaler from pipeline: {e}")
        
        # Fallback: compute Mx/Wx from data if not available
        if self.Mx is None or self.Wx is None:
            mx_fallback, wx_fallback = _compute_mx_wx(X_processed_np)
            if self.Mx is None:
                self.Mx = mx_fallback
            if self.Wx is None:
                self.Wx = wx_fallback
        
        # Setup mixed-frequency parameters
        if self.mixed_freq:
            self._setup_mixed_frequency_params()
    
    def _setup_mixed_frequency_params(self) -> None:
        """Setup mixed-frequency parameters from config and data."""
        if self.data_processed is None:
            raise RuntimeError("DataModule setup() must be called before _setup_mixed_frequency_params()")
        
        clock = getattr(self.config, 'clock', 'm')
        
        if not self.mixed_freq:
            self._constraint_matrix = None
            self._constraint_vector = None
            self._n_slower_freq = 0
            self._tent_weights_dict = None
            self._frequencies = None
            self._i_idio = np.ones(self.data_processed.shape[1], dtype=np.float32)
            self._idio_chain_lengths = np.zeros(self.data_processed.shape[1], dtype=np.int32)
            return
        
        agg_structure = get_agg_structure(self.config, clock=clock)
        frequencies_list = [s.frequency for s in self.config.series]
        frequencies_set = set(frequencies_list)
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        
        # Validate frequency pairs
        missing_pairs = [
            (freq, clock) for freq in frequencies_set
            if FREQUENCY_HIERARCHY.get(freq, 3) > clock_hierarchy and get_tent_weights(freq, clock) is None
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
        
        n_slower_freq = sum(1 for freq in frequencies_list if FREQUENCY_HIERARCHY.get(freq, 3) > clock_hierarchy)
        i_idio = np.array([1 if freq == clock else 0 for freq in frequencies_list], dtype=np.float32)
        frequencies_np = np.array([_FREQ_TO_INT.get(f, 3) for f in frequencies_list], dtype=np.int32)
        
        self._constraint_matrix = R_mat
        self._constraint_vector = q
        self._n_slower_freq = n_slower_freq
        self._tent_weights_dict = tent_weights_dict
        self._frequencies = frequencies_np
        self._i_idio = i_idio
        self._idio_chain_lengths = np.zeros(self.data_processed.shape[1], dtype=np.int32)
    
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
            - 'i_idio': idiosyncratic indicator array
            - 'idio_chain_lengths': idiosyncratic chain lengths
            - 'opt_nan': missing data handling options
            - 'clock': clock frequency
        """
        if self.data_processed is None:
            raise RuntimeError("DataModule setup() must be called before get_initialization_params()")
        
        return {
            'X': self.data_processed,
            'Mx': self.Mx,
            'Wx': self.Wx,
            'R_mat': self._constraint_matrix,
            'q': self._constraint_vector,
            'n_slower_freq': self._n_slower_freq,
            'tent_weights_dict': self._tent_weights_dict,
            'frequencies': self._frequencies,
            'i_idio': self._i_idio,
            'idio_chain_lengths': self._idio_chain_lengths,
            'opt_nan': {'method': self.nan_method, 'k': self.nan_k},
            'clock': getattr(self.config, 'clock', 'm')
        }
    
    def get_std_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get standardization parameters (Mx, Wx)."""
        if self.data_processed is None:
            raise RuntimeError("DataModule setup() must be called before get_std_params()")
        return self.Mx, self.Wx
    
    def get_processed_data(self) -> np.ndarray:
        """Get processed data array."""
        if self.data_processed is None:
            raise RuntimeError("DataModule setup() must be called before get_processed_data()")
        return self.data_processed

