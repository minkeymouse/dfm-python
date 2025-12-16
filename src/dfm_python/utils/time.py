"""Time index and timestamp utilities for DFM operations.

This module provides:
- TimeIndex class: Pandas-based time index abstraction
- Timestamp utilities: datetime parsing, conversion, and range generation
- Time helpers: time index operations and period parsing

This module consolidates time_index.py, timestamp.py, and time-related functions
from helpers.py for better organization and reduced file count.
"""

from typing import Union, Optional, Any, List, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar


# ============================================================================
# TimeIndex Class
# ============================================================================

class TimeIndex:
    """Time index abstraction wrapping pandas Series with datetime dtype.
    
    This class provides a datetime index interface while using
    pandas Series internally for compatibility with sktime and PyTorch Forecasting.
    
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
                # Try to convert to datetime
                try:
                    data = pd.to_datetime(data)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Cannot convert Series with dtype {data.dtype} to datetime: {e}")
            self._series = data
        elif isinstance(data, TimeIndex):
            self._series = data._series.copy()
        else:
            # Convert list/array to pandas Series
            try:
                self._series = pd.Series(pd.to_datetime(data), name="time")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot create TimeIndex from {type(data)}: {e}")
    
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
            # Return single datetime
            val = self._series.iloc[key]
            if isinstance(val, datetime):
                return val
            # Convert pandas Timestamp to Python datetime
            if isinstance(val, pd.Timestamp):
                return val.to_pydatetime()
            return datetime.fromisoformat(str(val)) if isinstance(str(val), str) else val
        elif isinstance(key, slice):
            # Return TimeIndex slice
            return TimeIndex(self._series.iloc[key])
        elif isinstance(key, (np.ndarray, pd.Series)):
            # Boolean indexing
            if isinstance(key, np.ndarray):
                key = pd.Series(key, index=self._series.index)
            return TimeIndex(self._series[key])
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")
    
    def __iter__(self):
        """Iterate over time index."""
        for val in self._series:
            if isinstance(val, pd.Timestamp):
                yield val.to_pydatetime()
            elif isinstance(val, datetime):
                yield val
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
        return np.array([dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt 
                        for dt in self._series], dtype=object)
    
    def to_list(self) -> List[datetime]:
        """Convert to list of datetime objects."""
        return [dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt 
                for dt in self._series]
    
    def filter(self, mask: Union[np.ndarray, pd.Series, List[bool]]) -> 'TimeIndex':
        """Filter time index using boolean mask."""
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, index=self._series.index)
        return TimeIndex(self._series[mask])
    
    def __ge__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Greater than or equal comparison."""
        if isinstance(other, datetime):
            return self._series >= other
        elif isinstance(other, TimeIndex):
            return self._series >= other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __le__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Less than or equal comparison."""
        if isinstance(other, datetime):
            return self._series <= other
        elif isinstance(other, TimeIndex):
            return self._series <= other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __gt__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Greater than comparison."""
        if isinstance(other, datetime):
            return self._series > other
        elif isinstance(other, TimeIndex):
            return self._series > other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __lt__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Less than comparison."""
        if isinstance(other, datetime):
            return self._series < other
        elif isinstance(other, TimeIndex):
            return self._series < other._series
        else:
            raise TypeError(f"Cannot compare TimeIndex with {type(other)}")
    
    def __eq__(self, other: Any) -> Union[pd.Series, bool]:
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


def datetime_range(start: datetime, end: Optional[datetime] = None, 
                  periods: Optional[int] = None, freq: str = 'D') -> list:
    """Generate datetime range (uses pd.date_range).
    
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
    # Use pandas date_range directly
    if end is not None:
        result = pd.date_range(start=start, end=end, freq=freq)
    elif periods is not None:
        result = pd.date_range(start=start, periods=periods, freq=freq)
    else:
        raise ValueError("Either 'end' or 'periods' must be specified")
    
    # Convert to list of datetime objects
    return [dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt for dt in result]


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


# ============================================================================
# Time Helper Functions (from helpers.py)
# ============================================================================

def to_python_datetime(value: Any) -> datetime:
    """Convert value to Python datetime (handles pandas Timestamp, strings, etc.).
    
    Parameters
    ----------
    value : Any
        Value to convert (pandas Timestamp, string, datetime, etc.)
        
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
    
    # Handle pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    
    # Try parsing as string
    return parse_timestamp(value)


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
                if isinstance(t, pd.Timestamp):
                    t = t.to_pydatetime()
                elif hasattr(t, 'to_pydatetime'):
                    t = t.to_pydatetime()
                elif hasattr(t, 'to_python'):
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
            elif isinstance(result, pd.Timestamp):
                return result.to_pydatetime()
            elif hasattr(result, 'to_pydatetime'):
                return result.to_pydatetime()
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
            elif isinstance(latest, pd.Timestamp):
                return latest.to_pydatetime()
            elif hasattr(latest, 'to_pydatetime'):
                return latest.to_pydatetime()
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

"""Metrics for evaluating model performance.

This module provides metric functions for evaluating DFM model performance,
using sklearn and sktime where available.
"""

from typing import Optional, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    mean_squared_error = None
    mean_absolute_error = None
    mean_absolute_percentage_error = None
    r2_score = None

from sktime.performance_metrics.forecasting import (
    MeanSquaredError,
    MeanAbsoluteError,
)


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Uses sktime.performance_metrics.forecasting.MeanSquaredError with square_root=True.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    rmse_overall : float
        Overall RMSE averaged across all series and time periods
    rmse_per_series : np.ndarray
        RMSE for each series (N,) or scalar if 1D input
        
    Notes
    -----
    - Returns NaN for overall RMSE if no valid observations exist
    - Returns NaN for individual series if that series has no valid observations
    - Mask parameter allows selective calculation (e.g., exclude certain time periods)
    - Automatically handles missing data by excluding NaN values
    - Requires sktime to be installed
    """
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        # Use non-NaN values in both actual and predicted
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        # Combine user mask with finite check
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case (single series)
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        # Convert to pandas Series for sktime
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        # Use sktime metric
        mse_metric = MeanSquaredError(square_root=True)
        rmse_result = mse_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(rmse_result, 'iloc'):
            rmse_series = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
        else:
            rmse_series = float(rmse_result)
        return rmse_series, np.array([rmse_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    rmse_per_series = np.zeros(N)
    
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            y_true = pd.Series(actual_series)
            y_pred = pd.Series(predicted_series)
            
            mse_metric = MeanSquaredError(square_root=True)
            rmse_result = mse_metric(y_true, y_pred)
            # Handle both scalar and Series returns
            if hasattr(rmse_result, 'iloc'):
                rmse_per_series[i] = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
            else:
                rmse_per_series[i] = float(rmse_result)
        else:
            rmse_per_series[i] = np.nan
    
    # Calculate overall RMSE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        mse_metric = MeanSquaredError(square_root=True)
        rmse_result = mse_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(rmse_result, 'iloc'):
            rmse_overall = float(rmse_result.iloc[0] if len(rmse_result) > 0 else rmse_result)
        else:
            rmse_overall = float(rmse_result)
    else:
        rmse_overall = np.nan
    
    return rmse_overall, rmse_per_series


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Mean Absolute Error (MAE) between actual and predicted values.
    
    Uses sktime.performance_metrics.forecasting.MeanAbsoluteError.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    mae_overall : float
        Overall MAE averaged across all series and time periods
    mae_per_series : np.ndarray
        MAE for each series (N,) or scalar if 1D input
        
    Notes
    -----
    - Requires sktime to be installed
    """
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        # Convert to pandas Series for sktime
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        # Use sktime metric
        mae_metric = MeanAbsoluteError()
        mae_result = mae_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(mae_result, 'iloc'):
            mae_series = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
        else:
            mae_series = float(mae_result)
        return mae_series, np.array([mae_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    mae_per_series = np.zeros(N)
    
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            y_true = pd.Series(actual_series)
            y_pred = pd.Series(predicted_series)
            
            mae_metric = MeanAbsoluteError()
            mae_result = mae_metric(y_true, y_pred)
            # Handle both scalar and Series returns
            if hasattr(mae_result, 'iloc'):
                mae_per_series[i] = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
            else:
                mae_per_series[i] = float(mae_result)
        else:
            mae_per_series[i] = np.nan
    
    # Calculate overall MAE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        y_true = pd.Series(actual_masked)
        y_pred = pd.Series(predicted_masked)
        
        mae_metric = MeanAbsoluteError()
        mae_result = mae_metric(y_true, y_pred)
        # Handle both scalar and Series returns
        if hasattr(mae_result, 'iloc'):
            mae_overall = float(mae_result.iloc[0] if len(mae_result) > 0 else mae_result)
        else:
            mae_overall = float(mae_result)
    else:
        mae_overall = np.nan
    
    return mae_overall, mae_per_series


def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate Mean Absolute Percentage Error (MAPE) between actual and predicted values.
    
    Uses sklearn.metrics.mean_absolute_percentage_error for MAPE calculation.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    mape_overall : float
        Overall MAPE averaged across all series and time periods
    mape_per_series : np.ndarray
        MAPE for each series (N,) or scalar if 1D input
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn is required for calculate_mape. "
            "Please install: pip install scikit-learn"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        mape_series = mean_absolute_percentage_error(actual_masked, predicted_masked)
        return mape_series, np.array([mape_series])
    
    # Handle 2D case
    T, N = actual.shape
    
    mape_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            mape_per_series[i] = mean_absolute_percentage_error(
                actual_series, predicted_series
            )
        else:
            mape_per_series[i] = np.nan
    
    # Calculate overall MAPE
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        mape_overall = mean_absolute_percentage_error(actual_masked, predicted_masked)
    else:
        mape_overall = np.nan
    
    return mape_overall, mape_per_series


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Calculate R² (coefficient of determination) between actual and predicted values.
    
    Uses sklearn.metrics.r2_score for R² calculation.
    Supports masking and per-series calculation for multivariate time series.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    r2_overall : float
        Overall R² averaged across all series and time periods
    r2_per_series : np.ndarray
        R² for each series (N,) or scalar if 1D input
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn is required for calculate_r2. "
            "Please install: pip install scikit-learn"
        )
    
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have same shape, "
            f"got {actual.shape} and {predicted.shape}"
        )
    
    # Create mask for valid values
    if mask is None:
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Handle 1D case
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        r2_series = r2_score(actual_masked, predicted_masked)
        return r2_series, np.array([r2_series])
    
    # Handle 2D case
    T, N = actual.shape
    
    r2_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            actual_series = actual[series_mask, i]
            predicted_series = predicted[series_mask, i]
            r2_per_series[i] = r2_score(actual_series, predicted_series)
        else:
            r2_per_series[i] = np.nan
    
    # Calculate overall R²
    if np.any(mask):
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        r2_overall = r2_score(actual_masked, predicted_masked)
    else:
        r2_overall = np.nan
    
    return r2_overall, r2_per_series

