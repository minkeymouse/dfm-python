"""Time index utilities for Dynamic Factor Models.

This module provides a simplified TimeIndex class for datetime handling.
"""

from typing import Union, List, Any
from datetime import datetime
import numpy as np
import pandas as pd
from ..utils.errors import DataValidationError


class TimeIndex:
    """Time index abstraction wrapping pandas Series with datetime dtype.
    
    This class provides a simple datetime index interface using pandas Series internally.
    
    Parameters
    ----------
    data : pd.Series, list, np.ndarray, or datetime-like
        Time index data. If pd.Series, must have datetime dtype.
        If list/array, will be converted to datetime.
    """
    
    def __init__(self, data: Union[pd.Series, List, np.ndarray, Any]):
        """Initialize TimeIndex from various input types."""
        if isinstance(data, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(data):
                try:
                    data = pd.to_datetime(data)
                except (TypeError, ValueError) as e:
                    raise DataValidationError(
                        f"Cannot convert Series with dtype {data.dtype} to datetime: {e}",
                        details="TimeIndex requires datetime-compatible data types"
                    )
            self._series = data
        elif isinstance(data, TimeIndex):
            self._series = data._series.copy()
        else:
            try:
                self._series = pd.Series(pd.to_datetime(data), name="time")
            except (TypeError, ValueError) as e:
                raise DataValidationError(
                    f"Cannot create TimeIndex from {type(data)}: {e}",
                    details="TimeIndex requires datetime-compatible input"
                )
    
    @property
    def series(self) -> pd.Series:
        """Get underlying pandas Series."""
        return self._series
    
    def __len__(self) -> int:
        """Return length of time index."""
        return len(self._series)
    
    def __getitem__(self, key: Union[int, slice, np.ndarray, pd.Series]) -> Union[datetime, 'TimeIndex']:
        """Get item or slice from time index."""
        if isinstance(key, (int, np.integer)):
            val = self._series.iloc[key]
            if isinstance(val, pd.Timestamp):
                return val.to_pydatetime()
            return val if isinstance(val, datetime) else datetime.fromisoformat(str(val))
        elif isinstance(key, slice):
            return TimeIndex(self._series.iloc[key])
        elif isinstance(key, (np.ndarray, pd.Series)):
            if isinstance(key, np.ndarray):
                key = pd.Series(key, index=self._series.index)
            return TimeIndex(self._series[key])
        else:
            raise DataValidationError(
                f"Unsupported index type: {type(key)}",
                details="TimeIndex indexing supports int, slice, array, or Series"
            )
    
    def __iter__(self):
        """Iterate over time index."""
        for val in self._series:
            if isinstance(val, pd.Timestamp):
                yield val.to_pydatetime()
            else:
                yield val
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TimeIndex({len(self)} periods)"
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array of datetime objects."""
        return np.array([dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt 
                        for dt in self._series], dtype=object)
    
    def to_list(self) -> List[datetime]:
        """Convert to list of datetime objects."""
        return [dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt 
                for dt in self._series]

