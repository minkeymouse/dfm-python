"""DataView class for pseudo real-time data slices.

This module provides the DataView class for creating time-specific views
of data for nowcasting and forecasting purposes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
from datetime import datetime
import numpy as np
import pandas as pd

from ..config import DFMConfig
from ..utils.time import TimeIndex
from ..utils.data import create_data_view


@dataclass
class DataView:
    """Lightweight descriptor for pseudo real-time data slices.
    
    This class provides a convenient way to create and manage time-specific
    views of data for nowcasting and forecasting. It encapsulates the data
    and configuration needed to create a view at a specific point in time.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    Time : TimeIndex or array-like
        Time index for the data
    Z : np.ndarray, optional
        Original untransformed data (T x N)
    config : DFMConfig, optional
        Model configuration with series release date information
    view_date : datetime or str, optional
        Date for which to create the view. If None, uses latest available date.
    description : str, optional
        Optional description of this data view
    X_frame : pd.DataFrame, optional
        Pandas DataFrame representation of X (for efficient masking)
    """
    X: np.ndarray
    Time: Union[TimeIndex, Any]
    Z: Optional[np.ndarray]
    config: Optional[DFMConfig]
    view_date: Optional[Union[datetime, str]] = None
    description: Optional[str] = None
    X_frame: Optional[pd.DataFrame] = None
    
    def materialize(self) -> Tuple[np.ndarray, Union[TimeIndex, Any], Optional[np.ndarray]]:
        """Return the masked arrays for this data view.
        
        This method creates a time-specific view of the data by masking
        observations that are not yet available at the view_date based on
        release date information in the config.
        
        Returns
        -------
        X_view : np.ndarray
            Masked data matrix (T x N) with NaN for unavailable observations
        Time : TimeIndex or array-like
            Time index (unchanged)
        Z_view : np.ndarray or None
            Masked original data (T x N) or None if Z was not provided
        """
        return create_data_view(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=self.view_date,
            X_frame=self.X_frame
        )
    
    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        Time: Union[TimeIndex, Any],
        Z: Optional[np.ndarray],
        config: Optional[DFMConfig],
        view_date: Optional[Union[datetime, str]] = None,
        description: Optional[str] = None,
        X_frame: Optional[pd.DataFrame] = None
    ) -> 'DataView':
        """Create DataView from arrays.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N)
        Time : TimeIndex or array-like
            Time index for the data
        Z : np.ndarray, optional
            Original untransformed data
        config : DFMConfig, optional
            Model configuration
        view_date : datetime or str, optional
            View date
        description : str, optional
            Description of this view
        X_frame : pd.DataFrame, optional
            Pandas DataFrame representation
        
        Returns
        -------
        DataView
            New DataView instance
        """
        return cls(
            X=X,
            Time=Time,
            Z=Z,
            config=config,
            view_date=view_date,
            description=description,
            X_frame=X_frame
        )
    
    def with_view_date(self, view_date: Union[datetime, str]) -> 'DataView':
        """Return a shallow copy with a different view date.
        
        Parameters
        ----------
        view_date : datetime or str
            New view date
        
        Returns
        -------
        DataView
            New DataView instance with updated view_date
        """
        return DataView(
            X=self.X,
            Time=self.Time,
            Z=self.Z,
            config=self.config,
            view_date=view_date,
            description=self.description,
            X_frame=self.X_frame
        )

