"""Data view management for DFM nowcasting.

This module provides adapters for managing data views (time-point specific
data views) for pseudo real-time nowcasting evaluation.
"""

from typing import Protocol, Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import numpy as np

# Type hints - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfm_python.config import DFMConfig
else:
    # Runtime imports
    try:
        from dfm_python.config import DFMConfig
    except ImportError:
        # Fallback for when dfm_python is not installed
        from typing import Any as DFMConfig


class DataViewManager(Protocol):
    """Protocol for data view managers.
    
    Any implementation can manage data views for pseudo real-time
    nowcasting evaluation.
    """
    
    def get_data_view(
        self,
        view_date: str,
        config: DFMConfig
    ) -> tuple:
        """Get data as it was available at a specific view date.
        
        Parameters
        ----------
        view_date : str
            View date (YYYY-MM-DD format)
        config : DFMConfig
            Model configuration
            
        Returns
        -------
        tuple
            (X, Time, Z) - data matrix, time index, original data
        """
        ...
    
    def list_data_views(self) -> List[str]:
        """List all available view dates.
        
        Returns
        -------
        list
            List of view dates (sorted)
        """
        ...
    
    def get_latest_data_view(self) -> Optional[str]:
        """Get the latest available view date.
        
        Returns
        -------
        str or None
            Latest view date, or None if no views available
        """
        ...


class BasicDataViewManager:
    """Basic data view manager.
    
    File-based implementation for data view management using in-memory data source.
    Uses create_data_view() from dfm_python.data to filter data based on release_date.
    
    Examples
    --------
    >>> from dfm_python.adapters import BasicDataViewManager
    >>> # Initialize with data tuple (X, Time, Z)
    >>> manager = BasicDataViewManager(data_source=(X, Time, Z))
    >>> X_view, Time_view, Z_view = manager.get_data_view('2024-01-01', config)
    """
    
    def __init__(self, data_source: Optional[Union[Tuple[np.ndarray, Any, Optional[np.ndarray]], str, Path]] = None):
        """Initialize data view manager.
        
        Parameters
        ----------
        data_source : tuple, str, Path, or None
            Data source can be:
            - Tuple (X, Time, Z): In-memory data (X: transformed, Time: time index, Z: original)
            - str or Path: File path to data (not yet implemented)
            - None: Must be set later or passed to get_data_view()
        """
        self.data_source = data_source
        self._views: List[str] = []
        self._cache: Dict[str, Tuple[np.ndarray, Any, Optional[np.ndarray]]] = {}
    
    def get_data_view(
        self,
        view_date: str,
        config: DFMConfig,
        data_source: Optional[Union[Tuple[np.ndarray, Any, Optional[np.ndarray]], str, Path]] = None
    ) -> Tuple[np.ndarray, Any, Optional[np.ndarray]]:
        """Get data as it was available at a specific view date.
        
        Parameters
        ----------
        view_date : str
            View date (YYYY-MM-DD format)
        config : DFMConfig
            Model configuration
        data_source : tuple, str, Path, or None, optional
            Override data source for this call. If None, uses self.data_source
            
        Returns
        -------
        tuple
            (X, Time, Z) - data matrix, time index, original data
            
        Notes
        -----
        - Uses create_data_view() from dfm_python.data to filter data based on release_date
        - Caches results in memory for performance
        - If data_source is a tuple, uses it directly
        - If data_source is a file path, loads data (not yet implemented)
        """
        # Use provided data_source or fall back to instance data_source
        source = data_source if data_source is not None else self.data_source
        
        if source is None:
            raise ValueError(
                "data_source must be provided either in __init__ or get_data_view()"
            )
        
        # Check cache
        if view_date in self._cache:
            return self._cache[view_date]
        
        # Import create_data_view from data module
        try:
            from dfm_python.data import create_data_view
        except ImportError:
            raise ImportError(
                "Cannot import create_data_view from dfm_python.data. "
                "Make sure dfm_python is properly installed."
            )
        
        # Handle different data_source types
        if isinstance(source, tuple):
            # In-memory data: (X, Time, Z)
            if len(source) == 2:
                X, Time = source
                Z = None
            elif len(source) == 3:
                X, Time, Z = source
            else:
                raise ValueError(
                    f"data_source tuple must have 2 or 3 elements, got {len(source)}"
                )
            
            # Validate types
            if not isinstance(X, np.ndarray):
                raise ValueError(f"X must be numpy array, got {type(X)}")
            
            # Create data view using create_data_view()
            X_view, Time_view, Z_view = create_data_view(
                X=X,
                Time=Time,
                Z=Z,
                config=config,
                view_date=view_date
            )
            
            # Cache result
            self._cache[view_date] = (X_view, Time_view, Z_view)
            if view_date not in self._views:
                self._views.append(view_date)
            
            return X_view, Time_view, Z_view
        
        elif isinstance(source, (str, Path)):
            # File path (not yet implemented)
            raise NotImplementedError(
                "File-based data source not yet implemented. "
                "Please provide data as tuple (X, Time, Z)."
            )
        
        else:
            raise ValueError(
                f"Unsupported data_source type: {type(source)}. "
                "Expected tuple (X, Time, Z) or file path."
            )
    
    def list_data_views(self) -> List[str]:
        """List all available view dates.
        
        Returns
        -------
        list
            List of view dates (sorted, newest first)
            
        Notes
        -----
        For in-memory data sources (tuple), returns cached view dates
        that have been accessed via get_data_view().
        
        For file-based sources (not yet implemented), would query
        the file system for available view files.
        """
        # For in-memory data sources, return cached views
        # (views are added when get_data_view() is called)
        return sorted(self._views, reverse=True)
    
    def get_latest_data_view(self) -> Optional[str]:
        """Get the latest available view date.
        
        Returns
        -------
        str or None
            Latest view date, or None if no views available
        """
        views = self.list_data_views()
        return views[0] if views else None

