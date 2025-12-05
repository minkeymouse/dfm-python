"""Custom sktime splitters for nowcasting backtesting.

This module provides custom splitter implementations for nowcasting evaluation
that respect publication lags and backward-looking nature of nowcasting.
"""

from typing import Iterator, Tuple, List, Optional, Union, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.forecasting.base import BaseForecaster

from ..config import DFMConfig
from ..utils.time import TimeIndex, parse_timestamp, to_python_datetime
from ..utils.data import create_data_view, calculate_release_date
from ..utils.helpers import get_series_ids, get_clock_frequency
from .utils import calc_backward_date, get_higher_frequency


class NowcastingSplitter(BaseSplitter):
    """Custom splitter for nowcasting backtesting with publication lags.
    
    This splitter creates train/test splits for nowcasting evaluation where:
    - Training: Data available at view_date (respects publication lags via SeriesConfig.release_date)
    - Test: Target period to nowcast (backward-looking, current period estimation)
    - Iterates through backward_steps to simulate real-time data releases
    
    The splitter respects:
    - Publication lags from SeriesConfig.release_date
    - Mixed frequencies via tent kernel (no frequencies higher than clock)
    - Backward-looking nature of nowcasting (estimating current period, not future)
    
    Parameters
    ----------
    target_periods : List[datetime]
        List of target periods to nowcast
    backward_steps : int
        Number of backward steps to simulate (e.g., 20 means 20 data releases before target)
    config : DFMConfig
        Model configuration containing series release date information
    time_index : TimeIndex or array-like
        Time index for the data
    higher_freq : bool, default False
        If True, use frequency one step faster than clock for snapshots
    clock : str, optional
        Clock frequency. If None, inferred from config
    
    Examples
    --------
    >>> from datetime import datetime
    >>> from dfm_python.nowcast.splitters import NowcastingSplitter
    >>> 
    >>> target_periods = [datetime(2024, 3, 31), datetime(2024, 6, 30)]
    >>> splitter = NowcastingSplitter(
    ...     target_periods=target_periods,
    ...     backward_steps=20,
    ...     config=model.config,
    ...     time_index=data_module.time_index
    ... )
    >>> 
    >>> for train_idx, test_idx in splitter.split(y):
    ...     # train_idx: indices of data available at view_date
    ...     # test_idx: index of target period to nowcast
    ...     pass
    """
    
    def __init__(
        self,
        target_periods: List[datetime],
        backward_steps: int,
        config: DFMConfig,
        time_index: Union[TimeIndex, List, np.ndarray],
        higher_freq: bool = False,
        clock: Optional[str] = None
    ):
        super().__init__()
        
        self.target_periods = target_periods
        self.backward_steps = backward_steps
        self.config = config
        self.higher_freq = higher_freq
        
        # Convert time_index to list of datetimes
        if isinstance(time_index, TimeIndex):
            self.time_list = [to_python_datetime(t) for t in time_index]
        else:
            self.time_list = []
            for t in time_index:
                if isinstance(t, datetime):
                    self.time_list.append(t)
                elif isinstance(t, pd.Timestamp):
                    self.time_list.append(t.to_pydatetime())
                else:
                    self.time_list.append(parse_timestamp(t))
        
        # Get clock frequency
        if clock is None:
            self.clock = get_clock_frequency(config, 'm')
        else:
            self.clock = clock
        
        # Get backward frequency
        if higher_freq:
            self.backward_freq = get_higher_frequency(self.clock)
            if self.backward_freq is None:
                # No higher frequency available, use clock
                self.backward_freq = self.clock
        else:
            self.backward_freq = self.clock
        
        # Pre-compute all splits
        self._splits: List[Tuple[np.ndarray, np.ndarray]] = []
        self._view_dates: List[datetime] = []
        self._target_dates: List[datetime] = []
        self._compute_splits()
    
    def _compute_splits(self):
        """Pre-compute all train/test splits."""
        self._splits = []
        self._view_dates = []
        self._target_dates = []
        
        for target_period in self.target_periods:
            # Iterate through backward steps
            for step in range(self.backward_steps, -1, -1):
                # Calculate view date (when data snapshot is taken)
                view_date = calc_backward_date(
                    target_period,
                    step,
                    self.backward_freq
                )
                
                # Get available data mask at view_date
                # We need to create a dummy data array to use create_data_view
                # The actual data will be provided in split()
                # For now, we just compute which time indices would be available
                available_mask = self._get_available_mask(view_date)
                
                # Train indices: all available data up to view_date
                train_idx = np.where(available_mask)[0]
                
                # Test index: target period (backward-looking nowcast)
                # Find the index of target_period in time_list
                test_idx = self._find_time_index(target_period)
                
                if test_idx is not None and len(train_idx) > 0:
                    self._splits.append((train_idx, np.array([test_idx])))
                    self._view_dates.append(view_date)
                    self._target_dates.append(target_period)
    
    def _get_available_mask(self, view_date: datetime) -> np.ndarray:
        """Get mask for data available at view_date based on publication lags.
        
        Parameters
        ----------
        view_date : datetime
            Date when data snapshot is taken
            
        Returns
        -------
        np.ndarray
            Boolean mask (T,) indicating which time periods have data available
        """
        if self.config is None or not hasattr(self.config, 'series') or not self.config.series:
            # No config, assume all data available
            return np.ones(len(self.time_list), dtype=bool)
        
        # For each time period, check if data would be available at view_date
        # This is based on release_date from SeriesConfig
        available_mask = np.zeros(len(self.time_list), dtype=bool)
        
        for t_idx, period in enumerate(self.time_list):
            # Check each series to see if it would be available
            # We need at least one series available for this period to count
            period_available = False
            
            for series_cfg in self.config.series:
                release_offset = getattr(series_cfg, 'release_date', None)
                if release_offset is None:
                    # No release date specified, assume available
                    period_available = True
                    break
                
                # Calculate when this series would be released for this period
                release_date = calculate_release_date(release_offset, period)
                
                # Data is available if view_date >= release_date
                if view_date >= release_date:
                    period_available = True
                    break
            
            available_mask[t_idx] = period_available
        
        return available_mask
    
    def _find_time_index(self, target_date: datetime) -> Optional[int]:
        """Find index of target_date in time_list.
        
        Parameters
        ----------
        target_date : datetime
            Target date to find
            
        Returns
        -------
        int or None
            Index of target_date, or None if not found
        """
        # Try exact match first
        if target_date in self.time_list:
            return self.time_list.index(target_date)
        
        # Try to find closest match (within same period)
        # For nowcasting, we're looking for the period containing target_date
        for i, t in enumerate(self.time_list):
            # Check if target_date is in the same period as t
            # This depends on frequency, but for simplicity, check if same month/quarter
            if (t.year == target_date.year and 
                t.month == target_date.month):
                return i
        
        return None
    
    def split(self, y: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for nowcasting backtest.
        
        Parameters
        ----------
        y : pd.DataFrame
            Time series data (T × N) with datetime index
            
        Yields
        ------
        train_idx : np.ndarray
            Training indices (data available at view_date)
        test_idx : np.ndarray
            Test indices (target period to nowcast)
        """
        # Return pre-computed splits
        for train_idx, test_idx in self._splits:
            yield train_idx, test_idx
    
    def get_n_splits(self, y: Optional[pd.DataFrame] = None) -> int:
        """Return the number of splits.
        
        Parameters
        ----------
        y : pd.DataFrame, optional
            Time series data (not used, kept for API compatibility)
            
        Returns
        -------
        int
            Number of splits
        """
        return len(self._splits)
    
    def get_split_params(self, split_idx: int) -> dict:
        """Get parameters for a specific split.
        
        Parameters
        ----------
        split_idx : int
            Index of the split
            
        Returns
        -------
        dict
            Dictionary with 'view_date' and 'target_date'
        """
        if split_idx >= len(self._splits):
            raise IndexError(f"Split index {split_idx} out of range [0, {len(self._splits)})")
        
        return {
            'view_date': self._view_dates[split_idx],
            'target_date': self._target_dates[split_idx]
        }


class NowcastForecaster(BaseForecaster):
    """sktime-compatible forecaster wrapper for nowcasting.
    
    This class wraps the Nowcast manager to work with sktime's forecasting API,
    enabling use with splitters and evaluation functions for nowcasting backtesting.
    
    Parameters
    ----------
    nowcast_manager : Nowcast
        Nowcast manager instance (from model.nowcast property)
    target_series : str
        Target series ID to nowcast
    target_period : datetime or str
        Target period for nowcast
    
    Examples
    --------
    >>> from dfm_python.nowcast.splitters import NowcastForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> 
    >>> forecaster = NowcastForecaster(
    ...     nowcast_manager=model.nowcast,
    ...     target_series='gdp',
    ...     target_period='2024Q1'
    ... )
    >>> 
    >>> # Fit (stores view_date from metadata)
    >>> forecaster.fit(y, fh=None)
    >>> 
    >>> # Predict (uses stored view_date and target_period)
    >>> y_pred = forecaster.predict(fh=None)
    """
    
    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
    }
    
    def __init__(
        self,
        nowcast_manager: Any,
        target_series: str,
        target_period: Union[datetime, str]
    ):
        super().__init__()
        
        self.nowcast_manager = nowcast_manager
        self.target_series = target_series
        self.target_period = target_period
        self._view_date = None
        self._is_fitted = False
    
    def _fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None, fh=None):
        """Fit the forecaster (stores view_date from metadata).
        
        Parameters
        ----------
        y : pd.DataFrame
            Time series data (T × N) with datetime index
        X : pd.DataFrame, optional
            Exogenous variables (not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon (not used for nowcasting)
            
        Returns
        -------
        self
        """
        # Extract view_date from y metadata or index
        # For nowcasting, view_date is typically the latest date in the index
        # or can be passed via metadata
        if hasattr(y, 'attrs') and 'view_date' in y.attrs:
            self._view_date = y.attrs['view_date']
        elif hasattr(y, 'view_date'):
            self._view_date = getattr(y, 'view_date')
        else:
            # Use latest date in index as view_date
            if isinstance(y.index, pd.DatetimeIndex):
                self._view_date = y.index[-1].to_pydatetime()
            else:
                # Try to parse
                from ..utils.time import parse_timestamp
                self._view_date = parse_timestamp(str(y.index[-1]))
        
        self._is_fitted = True
        return self
    
    def _predict(self, fh, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """Generate nowcast prediction.
        
        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon (not used for nowcasting, kept for API compatibility)
        X : pd.DataFrame, optional
            Exogenous variables (not used, but can contain target_period in metadata)
            
        Returns
        -------
        pd.Series
            Nowcast value as Series with target_period as index
        """
        if not self._is_fitted or self._view_date is None:
            raise ValueError(
                "Forecaster must be fitted before prediction. Call fit() first."
            )
        
        # Get target_period from X metadata if provided, otherwise use stored
        target_period = self.target_period
        if X is not None:
            if hasattr(X, 'attrs') and 'target_period' in X.attrs:
                target_period = X.attrs['target_period']
            elif hasattr(X, 'target_period'):
                target_period = getattr(X, 'target_period')
        
        # Use nowcast manager to compute nowcast
        nowcast_value = self.nowcast_manager(
            self.target_series,
            view_date=self._view_date,
            target_period=target_period
        )
        
        # Convert target_period to datetime if string
        if isinstance(target_period, str):
            from ..utils.time import parse_period_string, get_clock_frequency
            clock = get_clock_frequency(self.nowcast_manager.model.config, 'm')
            target_date = parse_period_string(target_period, clock)
        else:
            target_date = target_period
        
        # Return as Series with target_period as index
        # Convert target_date to datetime if needed
        if isinstance(target_date, str):
            from ..utils.time import parse_timestamp
            target_date = parse_timestamp(target_date)
        # Ensure target_date is datetime
        if not isinstance(target_date, datetime):
            from ..utils.time import to_python_datetime
            target_date = to_python_datetime(target_date)
        # Create DatetimeIndex and return Series
        index = pd.DatetimeIndex([target_date])
        return pd.Series([nowcast_value], index=index, name=self.target_series)
    
    def set_view_date(self, view_date: Union[datetime, str]):
        """Set view date for prediction.
        
        Parameters
        ----------
        view_date : datetime or str
            View date to use for nowcast calculation
        """
        if isinstance(view_date, str):
            from ..utils.time import parse_timestamp
            self._view_date = parse_timestamp(view_date)
        else:
            self._view_date = view_date

