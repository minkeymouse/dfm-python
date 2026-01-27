"""Dataset class for Dynamic Factor Model (DFM).

This module provides dataset implementation for DFM training.
Handles data loading and basic data preparation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, TYPE_CHECKING
from pathlib import Path
try:
    import polars as pl
    _has_polars = True
    PolarsDataFrame = pl.DataFrame
except ImportError:
    _has_polars = False
    pl = None  # For type hints
    PolarsDataFrame = type(None)  # Dummy type for type hints when polars not available

if TYPE_CHECKING:
    from ..config import DFMConfig

from ..logger import get_logger
from ..config import DFMConfig
from ..config.constants import FREQUENCY_HIERARCHY, DEFAULT_HIERARCHY_VALUE
from ..dataset.time import TimeIndex
from ..utils.errors import DataValidationError, ConfigurationError

_logger = get_logger(__name__)


class DFMDataset:
    """Dataset for DFM training.
    
    This dataset handles data loading and basic data preparation for DFM models.
    Unlike DDFM and KDFM, DFM doesn't use PyTorch Dataset for training
    (it uses NumPy arrays), but this class provides the same interface for consistency.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    data : pd.DataFrame, PolarsDataFrame, str, or Path
        Preprocessed data DataFrame or path to CSV file. Data must be preprocessed (imputation, scaling, etc.)
        before passing to this Dataset. If path is provided, CSV will be read with pandas.
    time_index : str
        Time index column name to extract from DataFrame (required).
    """
    
    def __init__(
        self,
        config: DFMConfig,
        data: Union[pd.DataFrame, PolarsDataFrame, str, Path],
        time_index: str
    ):
        """Initialize DFM dataset with data loading and preprocessing.
        
        Parameters
        ----------
        config : DFMConfig
            DFM configuration object (required)
        data : pd.DataFrame, PolarsDataFrame, str, or Path
            Preprocessed data DataFrame or path to CSV file (required). 
            If path is provided, CSV will be read with pandas.
            Data must be preprocessed (imputation, scaling, etc.) before passing to this Dataset.
        time_index : str
            Time index column name to extract from DataFrame (required).
        """
        if config is None:
            raise ConfigurationError(
                "DFMDataset: config is required",
                details="Please provide a DFMConfig object"
            )
        if data is None:
            raise DataValidationError(
                "DFMDataset: data is required",
                details="Please provide data as pandas.DataFrame, polars.DataFrame, or path to CSV file"
            )
        
        # Store attributes
        self.config = config
        self.data = data
        
        # Will be set in _setup()
        self._time_index: Optional[TimeIndex] = None
        self.data_df: Optional[pd.DataFrame] = None
        self._columns: Optional[List[str]] = None
        
        # Setup data
        self._setup(time_index)
        
        # Validate dataset
        self._validate()
    
    def _setup(self, time_index: str) -> None:
        """Load and prepare data.
        
        Parameters
        ----------
        time_index : str
            Time index column name to extract from DataFrame (required).
        """
        # Load data: handle pandas DataFrame, polars DataFrame, or file path
        if isinstance(self.data, pd.DataFrame):
            X_df = self.data.copy()
        elif _has_polars and isinstance(self.data, pl.DataFrame):
            # Convert polars DataFrame to pandas
            X_df = self.data.to_pandas()
        elif isinstance(self.data, (str, Path)):
            # Read CSV file with pandas
            path = Path(self.data) if isinstance(self.data, str) else self.data
            if not path.exists():
                raise DataValidationError(
                    f"Data file not found: {path}",
                    details=f"Please provide a valid path to a CSV file. Path provided: {path}"
                )
            X_df = pd.read_csv(path)
        else:
            raise DataValidationError(
                f"Unsupported data type {type(self.data)}. "
                f"Please provide data as pandas.DataFrame, polars.DataFrame (if available), or path to CSV file.",
                details=f"Received type: {type(self.data).__name__}. Expected: pandas.DataFrame, polars.DataFrame (if available), str, or Path."
            )
        
        # Extract time index from column and set as DataFrame index
        try:
            # Extract time data and convert to TimeIndex
            time_list = pd.to_datetime(X_df[time_index]).tolist()
            self._time_index = TimeIndex(time_list)
            
            # Set time column as DataFrame index and remove from columns
            X_df = X_df.set_index(time_index)
            _logger.info(f"Extracted time index from column: {time_index}, set as DataFrame index")
        except KeyError:
            raise DataValidationError(
                f"time_index column '{time_index}' not found in DataFrame. "
                f"Available columns: {list(X_df.columns)}",
                details=f"Requested column: {time_index}. DataFrame has {len(X_df.columns)} columns."
            )
        
        # Get all available columns (numeric only)
        numeric_columns = list(X_df.select_dtypes(include=[np.number]).columns)
        
        # Reorder columns for mixed-frequency models: clock-frequency first, then slower-frequency
        numeric_columns = self._reorder_columns_for_mixed_freq(numeric_columns, X_df)
        
        self._columns = numeric_columns
        
        # Store DataFrame with time index (numeric columns only, properly ordered)
        self.data_df = X_df[numeric_columns]
    
    @property
    def variables(self) -> pd.DataFrame:
        """Get variables DataFrame."""
        return self.data_df
    
    @property
    def time_index(self) -> Optional[TimeIndex]:
        """Get time index."""
        return self._time_index
    
    def _reorder_columns_for_mixed_freq(
        self, 
        columns: List[str], 
        data_df: pd.DataFrame
    ) -> List[str]:
        """Reorder columns for mixed-frequency models.
        
        Ensures columns are ordered as [clock-frequency series | slower-frequency series]
        as required by the mixed-frequency model implementation.
        
        Parameters
        ----------
        columns : List[str]
            Current column list
        data_df : pd.DataFrame
            DataFrame with data (for validation)
            
        Returns
        -------
        List[str]
            Reordered column list (clock-frequency first, then slower-frequency)
        """
        # Check if config has frequency mapping
        if not self.config.frequency:
            return columns  # No frequency mapping, return as-is
        
        # Get clock frequency
        clock = self.config.clock
        
        # Build frequency dict from config
        # Handle both grouped format {'w': [series1, ...], 'm': [series2, ...]}
        # and individual format {'series1': 'w', 'series2': 'm', ...}
        frequency_dict = {}
        if isinstance(self.config.frequency, dict):
            for key, value in self.config.frequency.items():
                if isinstance(value, list):
                    # Grouped format: {'w': [series1, ...], 'm': [series2, ...]}
                    for series in value:
                        if series in columns:
                            frequency_dict[series] = key
                else:
                    # Individual format: {'series1': 'w', 'series2': 'm', ...}
                    if key in columns:
                        frequency_dict[key] = value
        
        # If no frequency mapping found for any columns, return as-is
        if not frequency_dict:
            return columns
        
        # Get clock hierarchy value
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        
        # Separate columns by frequency
        clock_cols = []
        slower_cols = []
        
        for col in columns:
            freq = frequency_dict.get(col)
            if freq is None:
                # Column not in frequency mapping - assume clock frequency
                clock_cols.append(col)
            else:
                freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
                if freq_hierarchy <= clock_hierarchy:
                    # Clock frequency or same as clock
                    clock_cols.append(col)
                else:
                    # Slower frequency
                    slower_cols.append(col)
        
        # Reorder: clock-frequency first, then slower-frequency
        reordered = clock_cols + slower_cols
        
        # Log if reordering occurred
        if reordered != columns:
            _logger.info(
                f"Reordered columns for mixed-frequency model: "
                f"{len(clock_cols)} clock-frequency, {len(slower_cols)} slower-frequency"
            )
        
        return reordered
    
    def _validate(self) -> None:
        """Validate dataset state and data integrity.
        
        Raises
        ------
        ConfigurationError
            If dataset is not properly initialized or data is missing.
        DataValidationError
            If data integrity checks fail.
        """
        if self.data_df is None:
            raise ConfigurationError(
                "DFMDataset: data_df not available. "
                "Please ensure dataset was initialized with data.",
                details="Dataset validation failed: data_df is None."
            )
        
        if self._columns is None or len(self._columns) == 0:
            raise DataValidationError(
                "DFMDataset: columns not available. "
                "Please ensure dataset was initialized with valid data.",
                details="Dataset validation failed: _columns is None or empty."
            )
        
        if self._time_index is None:
            raise DataValidationError(
                "DFMDataset: time_index not available. "
                "Please ensure dataset was initialized with time_index parameter.",
                details="Dataset validation failed: time_index is None."
            )

