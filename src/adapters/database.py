"""Database adapters for DFM data and results.

This module provides adapters for integrating with various databases
(SQLite, PostgreSQL, etc.) for data storage and retrieval.
"""

from typing import Protocol, Optional, Any, Dict
from pathlib import Path
import sqlite3

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


class DatabaseAdapter(Protocol):
    """Protocol for database adapters.
    
    Any implementation can connect to a database and provide
    data access methods for DFM operations.
    """
    
    def connect(self) -> Any:
        """Connect to database.
        
        Returns
        -------
        connection
            Database connection object
        """
        ...
    
    def close(self) -> None:
        """Close database connection."""
        ...
    
    def load_data(
        self,
        config: DFMConfig,
        view_date: Optional[str] = None
    ) -> tuple:
        """Load data for DFM estimation.
        
        Parameters
        ----------
        config : DFMConfig
            Model configuration
        view_date : str, optional
            View date (for time-point specific data)
            
        Returns
        -------
        tuple
            (X, Time, Z) - data matrix, time index, original data
        """
        ...


class SQLiteAdapter:
    """SQLite database adapter.
    
    Basic implementation for SQLite database integration.
    
    Examples
    --------
    >>> from dfm_python.adapters import SQLiteAdapter
    >>> adapter = SQLiteAdapter(db_path='data.db')
    >>> X, Time, Z = adapter.load_data(config, view_date='2024-01-01')
    """
    
    def __init__(self, db_path: str):
        """Initialize SQLite adapter.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> sqlite3.Connection:
        """Connect to SQLite database.
        
        Returns
        -------
        sqlite3.Connection
            Database connection
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            # Enable row factory for dict-like access
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def load_data(
        self,
        config: DFMConfig,
        view_date: Optional[str] = None
    ) -> tuple:
        """Load data from SQLite database.
        
        Parameters
        ----------
        config : DFMConfig
            Model configuration
        view_date : str, optional
            View date (for time-point specific data)
            
        Returns
        -------
        tuple
            (X, Time, Z) - data matrix, time index, original data
            
        Notes
        -----
        This is an experimental feature. Database integration requires:
        1. Database schema with time series data tables
        2. View date tracking for pseudo real-time nowcasting
        3. Proper indexing for efficient queries
        
        For production use, implement:
        - Query data based on config.series (series_ids)
        - Apply view_date filtering based on release_date
        - Return (X, Time, Z) tuple compatible with DFM estimation
        
        Currently, use file-based data loading or BasicDataViewManager
        for in-memory data views.
        """
        conn = self.connect()
        
        raise NotImplementedError(
            "SQLiteAdapter.load_data() is experimental and not yet implemented. "
            "Use file-based data loading (load_data() from dfm_python.data) "
            "or BasicDataViewManager for in-memory data views."
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

