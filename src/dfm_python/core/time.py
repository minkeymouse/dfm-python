"""Time index and timestamp utilities for DFM operations.

This module provides:
- TimeIndex class: Polars-based time index abstraction
- Timestamp utilities: datetime parsing, conversion, and range generation
- Time helpers: time index operations and period parsing

This module consolidates time_index.py, timestamp.py, and time-related functions
from helpers.py for better organization and reduced file count.
"""

from typing import Union, Optional, Any, List, Dict
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import calendar


# ============================================================================
# TimeIndex Class
# ============================================================================

class TimeIndex:
    """Time index abstraction wrapping polars Series with datetime dtype.
    
    This class provides a datetime index interface while using
    polars Series internally for better performance.
    
    Parameters
    ----------
    data : pl.Series, list, np.ndarray, or datetime-like
        Time index data. If pl.Series, must have datetime dtype.
        If list/array, will be converted to datetime.
    """
    
    def __init__(self, data: Union[pl.Series, List, np.ndarray, Any]):
        """Initialize TimeIndex from various input types."""
        if isinstance(data, pl.Series):
            if data.dtype != pl.Datetime:
                # Try to convert to datetime
                try:
                    data = data.cast(pl.Datetime)
                except (pl.exceptions.ComputeError, TypeError, ValueError) as e:
                    raise ValueError(f"Cannot convert Series with dtype {data.dtype} to datetime: {e}")
            self._series = data
        elif isinstance(data, TimeIndex):
            self._series = data._series.clone()
        else:
            # Convert list/array to polars Series
            try:
                self._series = pl.Series("time", data, dtype=pl.Datetime)
            except (pl.exceptions.ComputeError, TypeError, ValueError):
                # Try parsing as strings
                try:
                    self._series = pl.Series("time", data).str.strptime(pl.Datetime)
                except Exception as e:
                    raise ValueError(f"Cannot create TimeIndex from {type(data)}: {e}")
    
    @property
    def series(self) -> pl.Series:
        """Get underlying polars Series."""
        return self._series
    
    def __len__(self) -> int:
        """Return length of time index."""
        return len(self._series)
    
    def __getitem__(self, key: Union[int, slice, np.ndarray, pl.Series]) -> Union[datetime, 'TimeIndex']:
        """Get item or slice from time index."""
        if isinstance(key, (int, np.integer)):
            # Return single datetime
            val = self._series[key]
            if isinstance(val, datetime):
                return val
            # Convert polars datetime to Python datetime
            return val.to_python() if hasattr(val, 'to_python') else datetime.fromisoformat(str(val))
        elif isinstance(key, slice):
            # Return TimeIndex slice
            return TimeIndex(self._series[key])
        elif isinstance(key, (np.ndarray, pl.Series)):
            # Boolean indexing
            if isinstance(key, np.ndarray):
                key = pl.Series(key)
            return TimeIndex(self._series.filter(key))
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")
    
    def __iter__(self):
        """Iterate over time index."""
        for val in self._series:
            if hasattr(val, 'to_python'):
                yield val.to_python()
            else:
                yield datetime.fromisoformat(str(val)) if isinstance(str(val), str) else val
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TimeIndex({len(self)} periods, dtype=datetime)"
    
    def iloc(self, key: Union[int, slice]) -> Union[datetime, 'TimeIndex']:
        """Integer location-based indexing (pandas-like)."""
        return self[key]
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array of datetime objects."""
        return np.array([dt.to_python() if hasattr(dt, 'to_python') else dt 
                        for dt in self._series], dtype=object)
    
    def to_list(self) -> List[datetime]:
        """Convert to list of datetime objects."""
        return [dt.to_python() if hasattr(dt, 'to_python') else dt 
                for dt in self._series]
    
    def filter(self, mask: Union[np.ndarray, pl.Series, List[bool]]) -> 'TimeIndex':
        """Filter time index using boolean mask."""
        if isinstance(mask, np.ndarray):
            mask = pl.Series(mask)
        elif isinstance(mask, list):
            mask = pl.Series(mask)
        return TimeIndex(self._series.filter(mask))
    
    def __ge__(self, other: Union[datetime, 'TimeIndex']) -> pl.Series:
        """Greater than or equal comparison."""
        if isinstance(other, datetime):
            return self._series >= other
        elif isinstance(other, TimeIndex):
            return self._series >= other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __le__(self, other: Union[datetime, 'TimeIndex']) -> pl.Series:
        """Less than or equal comparison."""
        if isinstance(other, datetime):
            return self._series <= other
        elif isinstance(other, TimeIndex):
            return self._series <= other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __gt__(self, other: Union[datetime, 'TimeIndex']) -> pl.Series:
        """Greater than comparison."""
        if isinstance(other, datetime):
            return self._series > other
        elif isinstance(other, TimeIndex):
            return self._series > other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __lt__(self, other: Union[datetime, 'TimeIndex']) -> pl.Series:
        """Less than comparison."""
        if isinstance(other, datetime):
            return self._series < other
        elif isinstance(other, TimeIndex):
            return self._series < other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __eq__(self, other: Any) -> Union[pl.Series, bool]:
        """Equality comparison."""
        if isinstance(other, datetime):
            return self._series == other
        elif isinstance(other, TimeIndex):
            return self._series == other._series
        else:
            return False


# ============================================================================
# Timestamp Utilities
# ============================================================================

# Type alias for Timestamp (replaces pd.Timestamp)
Timestamp = datetime


def create_timestamp(year: int, month: int, day: int = 1, 
                    hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """Create a datetime object (replaces pd.Timestamp).
    
    Parameters
    ----------
    year : int
        Year
    month : int
        Month (1-12)
    day : int, default 1
        Day of month
    hour : int, default 0
        Hour
    minute : int, default 0
        Minute
    second : int, default 0
        Second
        
    Returns
    -------
    datetime
        Datetime object
    """
    return datetime(year, month, day, hour, minute, second)


def days_in_month(year: int, month: int) -> int:
    """Get number of days in a month.
    
    Parameters
    ----------
    year : int
        Year
    month : int
        Month (1-12)
        
    Returns
    -------
    int
        Number of days in the month
    """
    return calendar.monthrange(year, month)[1]


def parse_timestamp(value: Union[str, datetime, int, float]) -> datetime:
    """Parse value to datetime (replaces pd.Timestamp).
    
    Parameters
    ----------
    value : str, datetime, int, or float
        Value to parse. If int/float, treated as Unix timestamp.
        
    Returns
    -------
    datetime
        Parsed datetime object
        
    Raises
    ------
    ValueError
        If parsing fails
    """
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        # Try ISO format first
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            # Try common formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d', '%m/%d/%Y']:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse datetime string: {value}")
    elif isinstance(value, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(value)
    else:
        raise ValueError(f"Cannot parse {type(value)} to datetime")


def to_timestamp(value: Union[str, datetime, int, float]) -> datetime:
    """Convert value to datetime (alias for parse_timestamp)."""
    return parse_timestamp(value)


def datetime_range(start: datetime, end: Optional[datetime] = None, 
                  periods: Optional[int] = None, freq: str = 'D') -> list:
    """Generate datetime range (replaces pd.date_range).
    
    Parameters
    ----------
    start : datetime
        Start date
    end : datetime, optional
        End date (required if periods not specified)
    periods : int, optional
        Number of periods (required if end not specified)
    freq : str, default 'D'
        Frequency string:
        - 'D': daily
        - 'W': weekly
        - 'ME': month end
        - 'MS': month start
        - 'QE': quarter end
        - 'QS': quarter start
        - 'YE': year end
        - 'YS': year start
        
    Returns
    -------
    list
        List of datetime objects
    """
    # Map frequency to polars duration
    freq_map = {
        'D': '1d',
        'W': '1w',
        'ME': '1mo',
        'MS': '1mo',  # Polars doesn't distinguish start/end, use month
        'QE': '3mo',
        'QS': '3mo',
        'YE': '1y',
        'YS': '1y',
    }
    
    if freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    polars_freq = freq_map[freq]
    
    if end is not None:
        # Generate range from start to end
        result = pl.datetime_range(start, end, polars_freq, eager=True)
    elif periods is not None:
        # Generate range with specified number of periods
        # Calculate end date based on frequency
        if freq in ['ME', 'MS']:
            # For monthly, add months manually
            end = start
            for _ in range(periods):
                # Get next month
                if end.month == 12:
                    end = datetime(end.year + 1, 1, 1)
                else:
                    end = datetime(end.year, end.month + 1, 1)
                # For ME, go to end of month
                if freq == 'ME':
                    end = datetime(end.year, end.month, days_in_month(end.year, end.month))
        else:
            # For other frequencies, use timedelta
            delta_map = {
                'D': timedelta(days=periods),
                'W': timedelta(weeks=periods),
                'QE': timedelta(days=periods * 90),  # Approximate
                'QS': timedelta(days=periods * 90),
                'YE': timedelta(days=periods * 365),
                'YS': timedelta(days=periods * 365),
            }
            if freq in delta_map:
                end = start + delta_map[freq]
            else:
                end = start + timedelta(days=periods)
        
        result = pl.datetime_range(start, end, polars_freq, eager=True)
        # Trim to exact number of periods
        if len(result) > periods:
            result = result[:periods]
    else:
        raise ValueError("Either 'end' or 'periods' must be specified")
    
    # Convert to list of datetime objects
    return [to_python_datetime(dt) for dt in result]


# Clock frequency to datetime frequency mapping (used across modules)
CLOCK_TO_DATETIME_FREQ: Dict[str, str] = {
    'd': 'D',      # Daily
    'w': 'W',      # Weekly
    'm': 'ME',     # Monthly end
    'q': 'QE',     # Quarterly end
    'sa': 'ME',    # Semi-annual (use monthly, 6 periods = 6 months)
    'a': 'YE',     # Annual end
}


def clock_to_datetime_freq(clock: str) -> str:
    """Convert clock frequency code to datetime frequency string.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    str
        Datetime frequency string for datetime_range(): 'D', 'W', 'ME', 'QE', 'YE'
        
    Examples
    --------
    >>> clock_to_datetime_freq('m')
    'ME'
    >>> clock_to_datetime_freq('q')
    'QE'
    """
    return CLOCK_TO_DATETIME_FREQ.get(clock, 'ME')  # Default to monthly if unknown


def get_next_period_end(last_date: datetime, frequency: str) -> datetime:
    """Get the next period end date based on frequency.
    
    Parameters
    ----------
    last_date : datetime
        Last date in the time series
    frequency : str
        Frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    datetime
        Next period end date
        
    Examples
    --------
    >>> from datetime import datetime
    >>> get_next_period_end(datetime(2024, 3, 31), 'q')
    datetime(2024, 6, 30, 0, 0)
    >>> get_next_period_end(datetime(2024, 12, 31), 'a')
    datetime(2025, 12, 31, 0, 0)
    """
    if frequency == 'd':
        return last_date + timedelta(days=1)
    elif frequency == 'w':
        return last_date + timedelta(weeks=1)
    elif frequency == 'm':
        # Next month end
        if last_date.month == 12:
            next_year = last_date.year + 1
            next_month = 1
        else:
            next_year = last_date.year
            next_month = last_date.month + 1
        return datetime(next_year, next_month, days_in_month(next_year, next_month))
    elif frequency == 'q':
        # Next quarter end
        if last_date.month in [1, 2, 3]:
            return datetime(last_date.year, 3, 31)
        elif last_date.month in [4, 5, 6]:
            return datetime(last_date.year, 6, 30)
        elif last_date.month in [7, 8, 9]:
            return datetime(last_date.year, 9, 30)
        else:
            return datetime(last_date.year + 1, 12, 31)
    elif frequency == 'sa':
        # Next semi-annual end (6 months)
        if last_date.month in [1, 2, 3, 4, 5, 6]:
            return datetime(last_date.year, 6, 30)
        else:
            return datetime(last_date.year + 1, 12, 31)
    elif frequency == 'a':
        # Next year end
        return datetime(last_date.year + 1, 12, 31)
    else:
        # Default to monthly
        if last_date.month == 12:
            next_year = last_date.year + 1
            next_month = 1
        else:
            next_year = last_date.year
            next_month = last_date.month + 1
        return datetime(next_year, next_month, days_in_month(next_year, next_month))


def format_timestamp(dt: datetime, fmt: str = '%Y-%m-%d') -> str:
    """Format datetime as string.
    
    Parameters
    ----------
    dt : datetime
        Datetime to format
    fmt : str, default '%Y-%m-%d'
        Format string
        
    Returns
    -------
    str
        Formatted string
    """
    return dt.strftime(fmt)


# ============================================================================
# Time Helper Functions (from helpers.py)
# ============================================================================

def to_python_datetime(value: Any) -> datetime:
    """Convert value to Python datetime (handles polars datetime, strings, etc.).
    
    Parameters
    ----------
    value : Any
        Value to convert (polars datetime, string, datetime, etc.)
        
    Returns
    -------
    datetime
        Python datetime object
        
    Raises
    ------
    ValueError
        If value cannot be converted to datetime
    """
    if isinstance(value, datetime):
        return value
    
    # Handle polars datetime
    if hasattr(value, 'to_python'):
        try:
            return value.to_python()
        except (AttributeError, TypeError):
            pass
    
    # Try parsing as string
    return parse_timestamp(value)


def extract_last_date(time_index: Any) -> datetime:
    """Extract last date from time index (handles TimeIndex, list, array, etc.).
    
    Parameters
    ----------
    time_index : Any
        Time index (TimeIndex, list, array, etc.)
        
    Returns
    -------
    datetime
        Last date in the time index
        
    Raises
    ------
    ValueError
        If time_index is empty or cannot be indexed
    IndexError
        If time_index has no elements
    """
    # Handle TimeIndex
    if hasattr(time_index, '__getitem__'):
        try:
            last_val = time_index[-1]
            return to_python_datetime(last_val)
        except (IndexError, KeyError):
            pass
    
    # Handle list
    if isinstance(time_index, (list, tuple)):
        if len(time_index) == 0:
            raise IndexError("Time index is empty")
        return to_python_datetime(time_index[-1])
    
    # Try to convert to list and get last element
    try:
        time_list = list(time_index)
        if len(time_list) == 0:
            raise IndexError("Time index is empty")
        return to_python_datetime(time_list[-1])
    except (TypeError, ValueError):
        raise ValueError(f"Cannot extract last date from {type(time_index)}")


def find_time_index(
    time_index: Union[TimeIndex, np.ndarray, Any],
    target_period: datetime
) -> Optional[int]:
    """Find time index for a target period.
    
    Parameters
    ----------
    time_index : TimeIndex, np.ndarray, or compatible
        Time index to search
    target_period : datetime
        Target period to find
        
    Returns
    -------
    int or None
        Index of matching time period, or None if not found
    """
    for i, t in enumerate(time_index):
        if not isinstance(t, datetime):
            try:
                if hasattr(t, 'to_python'):
                    t = t.to_python()
                else:
                    t = parse_timestamp(t)
            except (ValueError, TypeError, AttributeError):
                # Skip invalid time values
                continue
        
        if isinstance(t, datetime) and t.year == target_period.year and t.month == target_period.month:
            return i
    
    return None


def convert_to_timestamp(
    value: Union[int, str, datetime],
    time_index: Optional[Union[TimeIndex, np.ndarray, Any]] = None,
    frequency: Optional[str] = None
) -> datetime:
    """Convert value to datetime.
    
    Parameters
    ----------
    value : int, str, or datetime
        Value to convert:
        - int: Index into time_index
        - str: Period string (e.g., "2024q1", "2024m3") or date string
        - datetime: Returned as-is
    time_index : TimeIndex, np.ndarray, or compatible, optional
        Time index for integer conversion
    frequency : str, optional
        Frequency code ('m', 'q') for parsing period strings
        
    Returns
    -------
    datetime
        Converted timestamp
        
    Raises
    ------
    ValueError
        If conversion fails
    """
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, int):
        if time_index is None:
            raise ValueError("time_index required for integer conversion")
        
        # TimeIndex and compatible types support direct indexing
        if hasattr(time_index, '__getitem__'):
            result = time_index[value]
            if isinstance(result, datetime):
                return result
            elif isinstance(result, TimeIndex):
                # If it's a TimeIndex, get the first element
                return to_python_datetime(result[0])
            elif hasattr(result, 'to_python'):
                return result.to_python()
            else:
                return to_python_datetime(result)
        else:
            raise ValueError("Cannot access time_index with integer: must support indexing")
    
    if isinstance(value, str):
        # Try parsing as period string if frequency provided
        if frequency is not None:
            try:
                return parse_period_string(value, frequency)
            except ValueError:
                # If period parsing fails, try parsing as date string
                pass
        # Try parsing as date string
        return parse_timestamp(value)
    
    # Fallback: try direct conversion
    return parse_timestamp(value)


def get_latest_time(
    time_index: Union[TimeIndex, np.ndarray, Any]
) -> datetime:
    """Get latest time from time index.
    
    Parameters
    ----------
    time_index : TimeIndex, np.ndarray, or compatible
        Time index to extract latest time from
        
    Returns
    -------
    datetime
        Latest time in the index
        
    Raises
    ------
    ValueError
        If time_index is empty or cannot determine latest time
    """
    if hasattr(time_index, '__getitem__'):
        try:
            # Try negative indexing
            latest = time_index[-1]
            if isinstance(latest, datetime):
                return latest
            elif isinstance(latest, TimeIndex):
                # If it's a TimeIndex, get the first element
                return to_python_datetime(latest[0])
            elif hasattr(latest, 'to_python'):
                return latest.to_python()
            else:
                return to_python_datetime(latest)
        except (IndexError, KeyError, TypeError):
            pass
    
    # Fallback: convert to list and get last element
    try:
        time_list = list(time_index)
        if len(time_list) == 0:
            raise ValueError("Time index is empty")
        return to_python_datetime(time_list[-1])
    except (TypeError, ValueError) as e:
        raise ValueError("Cannot determine latest time from time_index: must support indexing")


def parse_period_string(period: str, frequency: str) -> datetime:
    """Parse period string to datetime (generic for all frequencies).
    
    Parameters
    ----------
    period : str
        Period string. Format depends on frequency:
        - 'm': "YYYYmMM" (e.g., "2024m3" for March 2024)
        - 'q': "YYYYqQ" (e.g., "2024q1" for Q1 2024)
        - 'sa': "YYYYsS" (e.g., "2024s1" for first half 2024)
        - 'a': "YYYY" (e.g., "2024" for year 2024)
        - 'd': "YYYY-MM-DD" or "YYYYmMMdDD"
        - 'w': "YYYY-Www" or "YYYYmMMdDD" (week start)
    frequency : str
        Frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    datetime
        Parsed timestamp (first day of period)
        
    Raises
    ------
    ValueError
        If period format is invalid or frequency is unsupported
    """
    if frequency == 'm':
        if 'm' not in period:
            raise ValueError(f"Period '{period}' must contain 'm' for monthly frequency (format: YYYYmMM)")
        year_str, month_str = period.split('m')
        year = int(year_str)
        month = int(month_str)
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be between 1 and 12, got {month}")
        return datetime(year, month, 1)
    elif frequency == 'q':
        if 'q' not in period:
            raise ValueError(f"Period '{period}' must contain 'q' for quarterly frequency (format: YYYYqQ)")
        year_str, q_str = period.split('q')
        year = int(year_str)
        q = int(q_str)
        if not (1 <= q <= 4):
            raise ValueError(f"Quarter must be between 1 and 4, got {q}")
        month = 3 * q
        return datetime(year, month, 1)
    elif frequency == 'sa':
        if 's' not in period:
            raise ValueError(f"Period '{period}' must contain 's' for semi-annual frequency (format: YYYYsS)")
        year_str, s_str = period.split('s')
        year = int(year_str)
        s = int(s_str)
        if not (1 <= s <= 2):
            raise ValueError(f"Semi-annual period must be 1 or 2, got {s}")
        month = 1 if s == 1 else 7
        return datetime(year, month, 1)
    elif frequency == 'a':
        # Annual: just year
        year = int(period)
        return datetime(year, 1, 1)
    elif frequency in ['d', 'w']:
        # Daily/weekly: try standard date format first
        try:
            return parse_timestamp(period)
        except (ValueError, TypeError):
            # Try alternative format YYYYmMMdDD
            if 'm' in period and 'd' in period:
                parts = period.split('m')
                if len(parts) == 2 and 'd' in parts[1]:
                    year = int(parts[0])
                    month_day = parts[1].split('d')
                    month = int(month_day[0])
                    day = int(month_day[1])
                    return datetime(year, month, day)
            raise ValueError(f"Period '{period}' format not recognized for frequency '{frequency}'")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

