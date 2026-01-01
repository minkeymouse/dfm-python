"""Data preprocessing utilities for Dynamic Factor Models.

This module provides utilities for working with sklearn scalers for inverse 
transformation of forecasts. Use scaler.inverse_transform() directly instead 
of extracting Mx/Wx arrays.

**Purpose**: Helper functions for finding and working with sklearn scalers in
transformer pipelines. Use the scaler object's inverse_transform() method
directly for unstandardization.

**Key Functions**:
- `_get_scaler()`: Recursively find scaler in transformer pipelines
- `_get_scaler_attr()`: Extract specific attributes from scalers (for validation)
- `TimeIndex`: Time index abstraction for datetime handling

**Note**: sklearn is required for these functions to work with StandardScaler,
RobustScaler, and other sklearn scalers.
"""

import warnings
from typing import Optional, Any, List, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from ..utils.common import ensure_numpy
from ..utils.errors import DataValidationError, ConfigurationError
from ..config.constants import DEFAULT_IDENTITY_SCALE, DEFAULT_ZERO_VALUE

try:
    import sklearn
    _has_sklearn = True
except ImportError:
    _has_sklearn = False

try:
    import sktime
    _has_sktime = True
except ImportError:
    _has_sktime = False

from ..logger import get_logger
from typing import Union

_logger = get_logger(__name__)


# ============================================================================
# Scaler Extraction and Attribute Access
# ============================================================================

def _check_sklearn():
    """Check if sklearn is available and raise ImportError if not."""
    if not _has_sklearn:
        raise ImportError(
            "Scaling utilities require scikit-learn. Install with: pip install scikit-learn"
        )


def _get_scaler(transformer: Any) -> Optional[Any]:
    """Extract scaler from transformer, handling wrappers and pipelines.
    
    Recursively searches through transformer wrappers and pipelines
    to find any scaler instance (StandardScaler, MinMaxScaler, RobustScaler, etc.)
    that has mean/center and scale attributes for unstandardization.
    
    Parameters
    ----------
    transformer : Any
        Transformer to search (StandardScaler, TransformerPipeline, sklearn Pipeline, etc.)
        
    Returns
    -------
    Optional[Any]
        Scaler instance if found (any scaler with mean/center and scale attributes), 
        None otherwise
    """
    if transformer is None:
        return None
    
    # Check if transformer is a scaler (has mean/center and scale attributes)
    has_mean_attr = hasattr(transformer, 'mean_') or hasattr(transformer, 'center_')
    has_scale_attr = hasattr(transformer, 'scale_')
    if has_mean_attr and has_scale_attr:
        return transformer
    
    # Check if transformer is wrapped (TabularToSeriesAdaptor or similar)
    if hasattr(transformer, 'transformer'):
        return _get_scaler(getattr(transformer, 'pipeline', None) or getattr(transformer, 'transformer', None))
    
    # Check if transformer is a pipeline (sktime TransformerPipeline or sklearn Pipeline)
    if hasattr(transformer, 'steps'):
        for _, step in transformer.steps:
            scaler = _get_scaler(step)
            if scaler is not None:
                return scaler
    
    # Check if transformer is ColumnEnsembleTransformer (or ColumnTransformer)
    if hasattr(transformer, 'transformers'):
        for _, trans, _ in transformer.transformers:
            scaler = _get_scaler(trans)
            if scaler is not None:
                return scaler
    
    return None


def _normalize_wx(wx: np.ndarray) -> np.ndarray:
    """Normalize Wx to avoid division by zero.
    
    This function replaces zero or NaN values in Wx with 1.0 to prevent
    division by zero during standardization/unstandardization.
    
    Parameters
    ----------
    wx : np.ndarray
        Scale values (N,), may contain zeros or NaN
        
    Returns
    -------
    np.ndarray
        Normalized scale values with zeros and NaN replaced by DEFAULT_IDENTITY_SCALE
    """
    # Replace both zero and NaN with DEFAULT_IDENTITY_SCALE
    return np.where((wx == 0) | np.isnan(wx), DEFAULT_IDENTITY_SCALE, wx)


def _get_scaler_attr(scaler: Any, attr_name: str, data: np.ndarray, default_value: Optional[float] = None, normalize: bool = False) -> Optional[np.ndarray]:
    """Extract attribute from any scaler with fallbacks.
    
    Supports multiple scaler types (StandardScaler, MinMaxScaler, RobustScaler, etc.)
    by checking for common attribute names and enable flags.
    
    Parameters
    ----------
    scaler : Any
        Scaler instance (StandardScaler, MinMaxScaler, RobustScaler, or any scaler
        with mean/center and scale attributes)
    attr_name : str
        Attribute name to extract ('mean_', 'center_', or 'scale_')
    data : np.ndarray
        Processed data array (T x N) for fallback computation
    default_value : float, optional
        Default value if attribute is disabled (DEFAULT_ZERO_VALUE for mean, DEFAULT_IDENTITY_SCALE for scale)
    normalize : bool, default False
        Whether to normalize the result (for scale, replaces zeros with DEFAULT_IDENTITY_SCALE)
        
    Returns
    -------
    Optional[np.ndarray]
        Attribute values (N,) if extracted, None if fallback needed
    """
    # Map attribute names to their enable flags (for StandardScaler)
    # Other scalers may not have these flags, so we'll try direct access
    enable_flag_map = {
        'mean_': 'with_mean',
        'center_': 'with_mean',  # Some scalers use 'center_' instead
        'scale_': 'with_std'
    }
    enable_flag = enable_flag_map.get(attr_name)
    
    # Try to get attribute directly first (works for most scalers)
    # Check for both 'mean_' and 'center_' for mean extraction
    attr_names_to_try = [attr_name]
    if attr_name == 'mean_':
        attr_names_to_try = ['mean_', 'center_']  # Try both
    
    for try_attr_name in attr_names_to_try:
        if hasattr(scaler, try_attr_name):
            try:
                attr_val = getattr(scaler, try_attr_name)
                if attr_val is not None:
                    if not isinstance(attr_val, np.ndarray):
                        attr_val = ensure_numpy(attr_val)
                    if normalize:
                        attr_val = _normalize_wx(attr_val)
                    return attr_val
            except (AttributeError, TypeError):
                continue
    
    # If direct access failed, check enable flags (for StandardScaler)
    if enable_flag and hasattr(scaler, enable_flag):
        enabled = getattr(scaler, enable_flag)
        if not enabled:
            # If disabled, return default value
            if default_value is not None:
                return np.full(data.shape[1], default_value, dtype=float)
            return None
    
    # No attribute found
    return None


# Legacy Mx/Wx extraction functions removed - use scaler.inverse_transform() directly
# The following functions were removed as they extract Mx/Wx arrays:
# - _extract_mx_wx()
# - _get_mean()
# - _get_scale()
# 
# Instead, use the scaler object directly:
#   if target_scaler is not None and hasattr(target_scaler, 'inverse_transform'):
#       X_original = target_scaler.inverse_transform(X_standardized)


# create_passthrough_transformer removed - no longer needed
# create_default_preprocessing_pipeline removed - tutorials now create sklearn Pipeline explicitly


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
                    raise DataValidationError(
                        f"Cannot convert Series with dtype {data.dtype} to datetime: {e}",
                        details="TimeIndex requires datetime-compatible data types"
                    )
            self._series = data
        elif isinstance(data, TimeIndex):
            self._series = data._series.copy()
        else:
            # Convert list/array to pandas Series
            try:
                self._series = pd.Series(pd.to_datetime(data), name="time")
            except (TypeError, ValueError) as e:
                raise DataValidationError(
                    f"Cannot create TimeIndex from {type(data)}: {e}",
                    details="TimeIndex requires datetime-compatible input (Series, list, array, or TimeIndex)"
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
            raise DataValidationError(
                f"Unsupported index type: {type(key)}",
                details="TimeIndex indexing supports int, slice, list, array, or Series types"
            )
    
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
            raise DataValidationError(
                f"Cannot compare TimeIndex with {type(other)}",
                details="TimeIndex comparison supports datetime or TimeIndex types"
            )
    
    def __le__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Less than or equal comparison."""
        if isinstance(other, datetime):
            return self._series <= other
        elif isinstance(other, TimeIndex):
            return self._series >= other._series
        else:
            raise DataValidationError(
                f"Cannot compare TimeIndex with {type(other)}",
                details="TimeIndex comparison supports datetime or TimeIndex types"
            )
    
    def __gt__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Greater than comparison."""
        if isinstance(other, datetime):
            return self._series > other
        elif isinstance(other, TimeIndex):
            return self._series > other._series
        else:
            raise DataValidationError(
                f"Cannot compare TimeIndex with {type(other)}",
                details="TimeIndex comparison supports datetime or TimeIndex types"
            )
    
    def __lt__(self, other: Union[datetime, 'TimeIndex']) -> pd.Series:
        """Less than comparison."""
        if isinstance(other, datetime):
            return self._series < other
        elif isinstance(other, TimeIndex):
            return self._series < other._series
        else:
            raise DataValidationError(
                f"Cannot compare TimeIndex with {type(other)}",
                details="TimeIndex comparison supports datetime or TimeIndex types"
            )
    
    def __eq__(self, other: Any) -> Union[pd.Series, bool]:
        """Equality comparison."""
        if isinstance(other, datetime):
            return self._series == other
        elif isinstance(other, TimeIndex):
            return self._series == other._series
        else:
            return False


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
            raise DataValidationError(
                f"Cannot parse datetime string: {value}",
                details="String must match one of the supported datetime formats: %Y-%m-%d, %Y-%m-%d %H:%M:%S, %Y/%m/%d, %m/%d/%Y"
            )
    elif isinstance(value, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(value)
    else:
        raise DataValidationError(
            f"Cannot parse {type(value)} to datetime",
            details="parse_timestamp supports datetime, str, int, float, or None types"
        )


__all__ = [
    # Scaler utilities
    '_get_scaler',
    '_get_scaler_attr',
    '_get_mean',
    '_get_scale',
    '_normalize_wx',
    '_check_sklearn',
    # Time index
    'TimeIndex',
    # Time utilities
    'parse_timestamp',
]

