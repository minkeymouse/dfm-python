"""Nowcasting and news decomposition for DFM models.

This module implements nowcasting functionality and news decomposition framework for 
understanding how new data releases affect nowcasts. The "news" is defined as the 
difference between the new data release and the model's previous forecast, decomposed
into contributions from each data series.

The module provides:
- Nowcasting: Generate current-period estimates before official data release
- News decomposition: Forecast updates when new data arrives
- Attribution: Forecast changes attributed to specific data series
- Understanding: Which data releases are most informative

This is essential for nowcasting applications where policymakers need to
understand the drivers of forecast revisions.
"""

import numpy as np
from scipy.linalg import pinv, inv
from typing import Tuple, Optional, Dict, Union, List, Any, Callable
from datetime import datetime, timedelta
import warnings
import logging
import polars as pl
import time

from .kalman import skf, fis, miss_data
from .config import DFMConfig
from .dfm import DFMResult
from .core.time import (
    TimeIndex,
    parse_timestamp,
    datetime_range,
    days_in_month,
    clock_to_datetime_freq,
    get_next_period_end,
    find_time_index,
    parse_period_string,
    get_latest_time,
    convert_to_timestamp,
    to_python_datetime,
)
from .core.helpers import (
    safe_get_attr,
    get_series_ids,
    get_series_names,
    find_series_index,
    get_series_id_by_index,
    get_frequencies_from_config,
    get_units_from_config,
    get_clock_frequency,
)
from .data import calculate_release_date, create_data_view, DataView
from dataclasses import dataclass

# Set up logger
_logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_DATE: str = '2017-01-01'


# ============================================================================
# Data Classes for Backtest Results
# ============================================================================

@dataclass
class NowcastResult:
    """Result from a single nowcast calculation."""
    target_series: str
    target_period: datetime
    view_date: datetime
    nowcast_value: float
    confidence_interval: Optional[Tuple[float, float]] = None  # (lower, upper)
    factors_at_view: Optional[np.ndarray] = None  # Factor values at view_date
    dfm_result: Optional[DFMResult] = None  # Full DFM result for this view
    data_availability: Optional[Dict[str, int]] = None  # n_available, n_missing


@dataclass
class NewsDecompResult:
    """Result from news decomposition calculation."""
    y_old: float
    y_new: float
    change: float
    singlenews: np.ndarray
    top_contributors: List[Tuple[str, float]]
    actual: np.ndarray
    forecast: np.ndarray
    weight: np.ndarray
    t_miss: np.ndarray
    v_miss: np.ndarray
    innov: np.ndarray


@dataclass
class BacktestResult:
    """Result from pseudo real-time backtest."""
    target_series: str
    target_date: datetime
    backward_steps: int
    higher_freq: bool
    backward_freq: str  # Actual frequency used
    
    # List of data views and results
    view_list: List[DataView]  # Each view with view_date set
    nowcast_results: List[NowcastResult]  # One per view
    news_results: List[Optional[NewsDecompResult]]  # None for first view
    
    # Point-wise metrics (one per backward step)
    actual_values: np.ndarray  # (backward_steps,)
    errors: np.ndarray  # (backward_steps,) - nowcast - actual
    mae_per_step: np.ndarray  # (backward_steps,)
    mse_per_step: np.ndarray  # (backward_steps,)
    rmse_per_step: np.ndarray  # (backward_steps,)
    
    # Overall metrics
    overall_mae: Optional[float] = None
    overall_rmse: Optional[float] = None
    overall_mse: Optional[float] = None
    
    # Error tracking
    failed_steps: List[int] = None  # type: ignore  # Indices where nowcast failed
    
    def __post_init__(self):
        """Initialize failed_steps if None."""
        if self.failed_steps is None:
            object.__setattr__(self, 'failed_steps', [])
    
    def plot(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """Plot point-wise metrics (RMSE, MSE, MAE) over backward steps.
        
        Parameters
        ----------
        save_path : str or Path, optional
            Path to save the plot. If None, plot is not saved.
        show : bool, default False
            Whether to display the plot.
            
        Raises
        ------
        ImportError
            If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        steps = np.arange(self.backward_steps)
        
        # Filter out NaN values for plotting
        valid_rmse = ~np.isnan(self.rmse_per_step)
        valid_mse = ~np.isnan(self.mse_per_step)
        valid_mae = ~np.isnan(self.mae_per_step)
        
        plt.figure(figsize=(12, 6))
        
        # Plot only valid (non-NaN) points
        if np.any(valid_rmse):
            plt.plot(steps[valid_rmse], self.rmse_per_step[valid_rmse], 'b-', 
                    label='RMSE', linewidth=2, marker='o', markersize=4)
        if np.any(valid_mse):
            plt.plot(steps[valid_mse], self.mse_per_step[valid_mse], 'g--', 
                    label='MSE', linewidth=2, marker='s', markersize=4)
        if np.any(valid_mae):
            plt.plot(steps[valid_mae], self.mae_per_step[valid_mae], 'r:', 
                    label='MAE', linewidth=2, marker='^', markersize=4)
        
        plt.xlabel('Backward Step (0 = oldest, N-1 = target)', fontsize=12)
        plt.ylabel('Error Metric', fontsize=12)
        plt.title(f'Point-wise Backtest Metrics: {self.target_series} at {self.target_date}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for failed steps if any
        if len(self.failed_steps) > 0:
            plt.text(0.02, 0.98, f'Failed steps: {len(self.failed_steps)}', 
                    transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            from pathlib import Path
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path_obj), dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


# ============================================================================
# Helper Functions
# ============================================================================

def _get_higher_frequency(clock: str) -> Optional[str]:
    """Get frequency one step faster than clock.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    str or None
        Frequency one step faster than clock, or None if no higher frequency available
    """
    from .core.utils import FREQUENCY_HIERARCHY
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    target_h = clock_h - 1
    
    if target_h < 1:
        return None  # No higher frequency available (clock is already fastest)
    
    # Find frequency with target hierarchy
    for freq, h in FREQUENCY_HIERARCHY.items():
        if h == target_h:
            return freq
    
    return None  # No higher frequency found


def _calculate_backward_date(
    target_date: datetime,
    step: int,
    freq: str
) -> datetime:
    """Calculate backward date with accurate calendar handling.
    
    Parameters
    ----------
    target_date : datetime
        Target date to go backward from
    step : int
        Number of steps to go backward
    freq : str
        Frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    datetime
        Calculated backward date
    """
    try:
        from dateutil.relativedelta import relativedelta
        use_relativedelta = True
    except ImportError:
        use_relativedelta = False
        relativedelta = None  # type: ignore
        _logger.debug("dateutil.relativedelta not available, using timedelta approximation")
    
    if freq == 'd':
        return target_date - timedelta(days=step)
    elif freq == 'w':
        return target_date - timedelta(weeks=step)
    elif freq == 'm':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step)
        else:
            # Approximate: 30 days per month
            return target_date - timedelta(days=step * 30)
    elif freq == 'q':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 3)
        else:
            # Approximate: 90 days per quarter
            return target_date - timedelta(days=step * 90)
    elif freq == 'sa':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 6)
        else:
            # Approximate: 180 days per semi-annual
            return target_date - timedelta(days=step * 180)
    elif freq == 'a':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(years=step)
        else:
            # Approximate: 365 days per year
            return target_date - timedelta(days=step * 365)
    else:
        # Fallback for unknown frequencies
        _logger.warning(f"Unknown frequency '{freq}', using 30-day approximation")
        return target_date - timedelta(days=step * 30)


def _get_forecast_horizon_config(clock: str, horizon: Optional[int] = None) -> Tuple[int, str]:
    """Get forecast horizon configuration based on clock frequency.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
    horizon : int, optional
        Number of periods for forecast horizon. If None, defaults to 1 timestep.
        
    Returns
    -------
    Tuple[int, str]
        (horizon_periods, datetime_freq) where:
        - horizon_periods: Number of periods to forecast
        - datetime_freq: Frequency string for datetime_range() ('D', 'W', 'ME', 'QE', 'YE')
        
    Notes
    -----
    - Default horizon is 1 timestep based on clock frequency (generic)
    - For semi-annual ('sa'), uses 6-month periods
    """
    if horizon is None:
        horizon = 1  # Default: 1 timestep based on clock frequency
    
    # Map clock frequency to datetime frequency string (use shared mapping)
    datetime_freq = clock_to_datetime_freq(clock)
    
    # For semi-annual, we need 6 months per period
    if clock == 'sa' and horizon > 0:
        horizon = horizon * 6  # Convert to months
    
    return horizon, datetime_freq


def _check_config_consistency(saved_config: Any, current_config: DFMConfig) -> None:
    """Check if saved config is consistent with current config.
    
    Parameters
    ----------
    saved_config : Any
        Saved configuration object (may be DFMConfig or dict-like)
    current_config : DFMConfig
        Current configuration object
        
    Notes
    -----
    - Issues a warning if configs differ significantly
    - Does not raise exceptions (allows computation to continue)
    """
    try:
        # Basic checks
        if hasattr(saved_config, 'series') and hasattr(current_config, 'series'):
            if len(saved_config.series) != len(current_config.series):
                _logger.warning(
                    f"Config mismatch: saved config has {len(saved_config.series)} series, "
                    f"current config has {len(current_config.series)} series"
                )
        
        if hasattr(saved_config, 'block_names') and hasattr(current_config, 'block_names'):
            if saved_config.block_names != current_config.block_names:
                _logger.warning(
                    f"Config mismatch: block names differ. "
                    f"Saved: {saved_config.block_names}, Current: {current_config.block_names}"
                )
    except Exception as e:
        _logger.debug(f"Config consistency check failed (non-critical): {str(e)}")
        # If comparison fails, continue anyway


def para_const(X: np.ndarray, P: Union[DFMResult, Dict[str, Any]], lag: int) -> Dict[str, Any]:
    """Implement Kalman filter for news calculation with fixed parameters.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix (T x N)
    P : DFMResult
        DFM parameters
    lag : int
        Number of lags
        
    Returns:
    --------
    Dict with keys: Plag, P, X_sm, F
    """
    # Set model parameters
    Z_0 = P.Z_0
    V_0 = P.V_0
    A = P.A
    C = P.C
    Q = P.Q
    R = P.R
    Mx = P.Mx
    Wx = P.Wx
    
    # Prepare data
    T, _ = X.shape
    
    # Standardize
    Y = ((X - Mx) / Wx).T  # n x T
    
    # Apply Kalman filter and smoother
    Sf = skf(Y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    
    # Calculate parameter output
    Vs = Ss.VmT[:, :, 1:]  # Smoothed factor covariance
    Vf = Sf.VmU[:, :, 1:]  # Filtered factor posterior covariance
    Zsmooth = Ss.ZmT
    Vsmooth = Ss.VmT
    
    Plag = [Vs]
    
    for jk in range(1, lag + 1):
        Plag_jk = np.zeros_like(Plag[0])
        for jt in range(Plag[0].shape[2] - 1, lag, -1):  # Backward iteration to match MATLAB
            As = Vf[:, :, jt - jk] @ A.T @ pinv(A @ Vf[:, :, jt - jk] @ A.T + Q)
            Plag_jk[:, :, jt] = As @ Plag[jk - 1][:, :, jt]
        Plag.append(Plag_jk)
    
    # Prepare data for output
    Zsmooth = Zsmooth.T  # T x m
    x_sm = Zsmooth[1:, :] @ C.T  # T x N
    X_sm = Wx * x_sm + Mx  # Unstandardized
    
    return {
        'Plag': Plag,
        'P': Vsmooth,
        'X_sm': X_sm,
        'F': Zsmooth[1:, :]
    }


class Nowcast:
    """Nowcasting and news decomposition manager.
    
    This class provides a unified interface for nowcasting operations,
    news decomposition, and forecast updates. It takes a DFM model instance
    and provides methods for calculating nowcasts and decomposing forecast
    updates into news contributions.
    
    Parameters
    ----------
    model : DFM
        Trained DFM model instance. Must have:
        - result: DFMResult (from training)
        - config: DFMConfig
        - data: np.ndarray (T x N)
        - time: TimeIndex
        - original_data: np.ndarray (optional)
    
    Examples
    --------
    >>> from dfm_python import DFM, Nowcast
    >>> 
    >>> # Train model
    >>> model = DFM()
    >>> model.load_config('config/default.yaml')
    >>> model.load_data('data/sample_data.csv')
    >>> model.train()
    >>> 
    >>> # Create Nowcast instance
    >>> nowcast = Nowcast(model)
    >>> 
    >>> # Calculate nowcast (callable interface)
    >>> value = nowcast('gdp', view_date='2024-01-15')
    >>> 
    >>> # News decomposition
    >>> news = nowcast.decompose(
    ...     target_series='gdp',
    ...     target_period='2024Q1',
    ...     view_date_old='2024-01-15',
    ...     view_date_new='2024-02-15'
    ... )
    >>> print(f"Change: {news.change:.2f}")
    >>> print(f"Top contributors: {news.top_contributors}")
    """
    
    def __init__(self, model: Any):  # type: ignore
        """Initialize Nowcast manager.
        
        Parameters
        ----------
        model : DFM
            Trained DFM model instance (validation is done in DFM.nowcast property)
        
        Note
        ----
        Validation is performed in DFM.nowcast property before creating instance.
        This class directly references model.data, model.time, etc. for efficiency.
        """
        self.model = model
        # Direct references (no copying) - always use latest data
        # Caching for performance
        self._para_const_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._data_view_cache: Dict[str, Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]] = {}
    
    def get_data_view(self, view_date: Union[datetime, str]) -> Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]:
        """Get data view at specific date (with caching).
        
        Parameters
        ----------
        view_date : datetime or str
            View date for data availability
            
        Returns
        -------
        Tuple[np.ndarray, TimeIndex, Optional[np.ndarray]]
            (X_view, Time_view, Z_view) - data available at view_date
        """
        view_date_str = str(view_date)
        if view_date_str in self._data_view_cache:
            return self._data_view_cache[view_date_str]
        
        start = time.perf_counter()
        X_view, Time_view, Z_view = create_data_view(
            X=self.model.data,
            Time=self.model.time,
            Z=self.model.original_data,
            config=self.model.config,
            view_date=view_date,
            X_frame=getattr(self.model, 'data_frame', None)
        )
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "create_data_view[%s] completed in %.3fs",
                view_date_str,
                time.perf_counter() - start
            )
        self._data_view_cache[view_date_str] = (X_view, Time_view, Z_view)
        return X_view, Time_view, Z_view
    
    def _get_kalman_result(self, cache_key: str, X_view: np.ndarray) -> Dict[str, Any]:
        """Cache-aware wrapper around para_const for profiling."""
        key = (cache_key, X_view.shape[0])
        if key in self._para_const_cache:
            return self._para_const_cache[key]
        start = time.perf_counter()
        self._para_const_cache[key] = para_const(X_view, self.model.result, 0)
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "para_const[%s] completed in %.3fs",
                cache_key,
                time.perf_counter() - start
            )
        return self._para_const_cache[key]
    
    def _parse_target_date(
        self,
        target_date: Union[datetime, str],
        target_series: Optional[str] = None
    ) -> datetime:
        """Parse target date from string or datetime.
        
        Parameters
        ----------
        target_date : datetime or str
            Target date to parse
        target_series : str, optional
            Target series ID (used to determine frequency for string parsing)
            
        Returns
        -------
        datetime
            Parsed target date
        """
        if isinstance(target_date, datetime):
            return target_date
        elif isinstance(target_date, str):
            clock = get_clock_frequency(self.model.config, 'm')
            if target_series is not None:
                frequencies = get_frequencies_from_config(self.model.config)
                i_series = find_series_index(self.model.config, target_series)
                freq = frequencies[i_series] if i_series < len(frequencies) else clock
            else:
                freq = clock
            return parse_period_string(target_date, freq)
        else:
            return parse_timestamp(target_date)
    
    def _create_nowcast_result_with_metadata(
        self,
        target_series: str,
        target_period: datetime,
        view_date: datetime,
        nowcast_value: float,
        X_view: np.ndarray,
        Time_view: TimeIndex
    ) -> NowcastResult:
        """Create NowcastResult with metadata (data availability, factors, etc.).
        
        This is a consolidated helper method to create NowcastResult objects
        with consistent metadata extraction.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        target_period : datetime
            Target period for nowcast
        view_date : datetime
            View date for data availability
        nowcast_value : float
            Calculated nowcast value
        X_view : np.ndarray
            Data view matrix (T x N)
        Time_view : TimeIndex
            Time index for the view
            
        Returns
        -------
        NowcastResult
            NowcastResult with all metadata populated
        """
        # Calculate data availability
        n_total = X_view.size
        n_missing = int(np.sum(np.isnan(X_view)))
        n_available = n_total - n_missing
        data_availability = {
            'n_available': n_available,
            'n_missing': n_missing
        }
        
        # Extract factors and DFM result
        factors_at_view = None
        dfm_result = None
        if self.model.result is not None and hasattr(self.model.result, 'Z'):
            t_view = find_time_index(Time_view, view_date)
            if t_view is not None and t_view < self.model.result.Z.shape[0]:
                factors_at_view = self.model.result.Z[t_view, :].copy()
            dfm_result = self.model.result
        
        return NowcastResult(
            target_series=target_series,
            target_period=target_period,
            view_date=view_date,
            nowcast_value=nowcast_value,
            confidence_interval=None,  # Could be calculated from Kalman filter covariance (C @ Var(Z) @ C.T + R)
            factors_at_view=factors_at_view,
            dfm_result=dfm_result,
            data_availability=data_availability
        )
    
    def _news_dfm(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        t_fcst: int,
        v_news: Union[int, np.ndarray, List[int]],
        *,
        Res_old_cache: Optional[Dict[str, Any]] = None,
        Res_new_cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray], np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Calculate changes in news from data releases (internal method).
        
        This is the core news decomposition algorithm. See news_dfm() documentation
        for detailed explanation of the algorithm.
        
        Parameters
        ----------
        X_old : np.ndarray
            Old data matrix (T x N)
        X_new : np.ndarray
            New data matrix (T x N)
        t_fcst : int
            Target time index for forecast
        v_news : int, np.ndarray, or List[int]
            Target variable index(s)
            
        Returns
        -------
        Tuple containing 9 elements:
        y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov
        """
        # Input validation
        if not isinstance(X_old, np.ndarray) or X_old.ndim != 2:
            raise ValueError(f"X_old must be a 2D numpy array, got {type(X_old)}")
        if not isinstance(X_new, np.ndarray) or X_new.ndim != 2:
            raise ValueError(f"X_new must be a 2D numpy array, got {type(X_new)}")
        if X_old.shape[1] != X_new.shape[1]:
            raise ValueError(f"X_old and X_new must have same number of series. Got {X_old.shape[1]} and {X_new.shape[1]}")
        if not isinstance(t_fcst, (int, np.integer)) or t_fcst < 0:
            raise ValueError(f"t_fcst must be a non-negative integer, got {t_fcst}")
        if t_fcst >= X_new.shape[0]:
            raise ValueError(f"t_fcst ({t_fcst}) must be less than number of time periods ({X_new.shape[0]})")
        
        # Normalize v_news to array
        v_news_arr = np.atleast_1d(v_news)
        is_scalar = isinstance(v_news, (int, np.integer)) or (isinstance(v_news, np.ndarray) and v_news.ndim == 0)
        n_targets = len(v_news_arr)
        
        # Validate v_news indices
        if np.any(v_news_arr < 0) or np.any(v_news_arr >= X_new.shape[1]):
            raise ValueError(f"v_news indices must be in range [0, {X_new.shape[1]}), got {v_news_arr}")
        
        r = self.model.result.C.shape[1]
        _, N = X_new.shape
        
        def _resolve_res(cache: Optional[Dict[str, Any]], X_mat: np.ndarray, lag: int = 0) -> Dict[str, Any]:
            if cache is not None and lag == 0:
                return cache
            return para_const(X_mat, self.model.result, lag)
        
        # Check if targets are already observed
        targets_observed = np.array([not np.isnan(X_new[t_fcst, v]) for v in v_news_arr])
        
        if np.all(targets_observed):
            # NO FORECAST CASE: Already values for all target variables at time t_fcst
            Res_old = _resolve_res(Res_old_cache, X_old)
            
            # Initialize output arrays
            if is_scalar:
                singlenews = np.zeros(N)
                singlenews[v_news_arr[0]] = X_new[t_fcst, v_news_arr[0]] - Res_old['X_sm'][t_fcst, v_news_arr[0]]
                y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                y_new = X_new[t_fcst, v_news_arr[0]]
            else:
                singlenews = np.zeros((N, n_targets))
                for i, v in enumerate(v_news_arr):
                    singlenews[v, i] = X_new[t_fcst, v] - Res_old['X_sm'][t_fcst, v]
                y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                y_new = np.array([X_new[t_fcst, v] for v in v_news_arr])
            
            actual = np.array([])
            forecast = np.array([])
            weight = np.array([])
            t_miss = np.array([])
            v_miss = np.array([])
            innov = np.array([])
        else:
            # FORECAST CASE
            Mx = self.model.result.Mx
            Wx = self.model.result.Wx
            
            # Calculate indicators for missing values
            miss_old = np.isnan(X_old)
            miss_new = np.isnan(X_new)
            
            # Indicator for missing: 1 = new data available, -1 = old data available but not new
            i_miss = miss_old.astype(int) - miss_new.astype(int)
            
            # Time/variable indices where new data is available
            t_miss, v_miss = np.where(i_miss == 1)
            t_miss = t_miss.flatten()
            v_miss = v_miss.flatten()
            
            if len(v_miss) == 0:
                # NO NEW INFORMATION
                Res_old = _resolve_res(Res_old_cache, X_old)
                Res_new = _resolve_res(Res_new_cache, X_new)
                
                if is_scalar:
                    y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                    y_new = y_old
                    singlenews = np.array([])
                else:
                    y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                    y_new = y_old
                    singlenews = np.array([]).reshape(0, n_targets)
                
                actual = np.array([])
                forecast = np.array([])
                weight = np.array([])
                t_miss = np.array([])
                v_miss = np.array([])
                innov = np.array([])
            else:
                # NEW INFORMATION
                # Difference between forecast time and new data time
                lag = t_fcst - t_miss
                
                # Biggest time interval
                lag_abs_max = float(np.max(np.abs(lag)))
                lag_range = float(np.max(lag) - np.min(lag))
                k = int(max(lag_abs_max, lag_range))
                
                C = self.model.result.C
                R_cov = self.model.result.R.T
                
                n_news = len(lag)
                
                # Smooth old dataset
                Res_old = _resolve_res(Res_old_cache if k == 0 else None, X_old, k)
                Plag = Res_old['Plag']
                
                # Smooth new dataset
                Res_new = _resolve_res(Res_new_cache, X_new, 0)
                
                # Get nowcasts for all target variables
                if is_scalar:
                    y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                    y_new = Res_new['X_sm'][t_fcst, v_news_arr[0]]
                else:
                    y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                    y_new = np.array([Res_new['X_sm'][t_fcst, v] for v in v_news_arr])
                
                # Calculate projection matrices
                P1 = []
                for i in range(n_news):
                    h = abs(t_fcst - t_miss[i])
                    m = max(t_miss[i], t_fcst)
                    
                    if t_miss[i] > t_fcst:
                        Pp = Plag[h + 1][:, :, m] if h + 1 < len(Plag) else Plag[-1][:, :, m]
                    else:
                        Pp = Plag[h + 1][:, :, m].T if h + 1 < len(Plag) else Plag[-1][:, :, m].T
                    
                    P1.append(Pp @ C[v_miss[i], :r].T)
                
                P1 = np.hstack(P1) if len(P1) > 0 else np.zeros((r, 1))
                
                # Calculate innovations
                innov = np.zeros(n_news)
                for i in range(n_news):
                    X_new_norm = (X_new[t_miss[i], v_miss[i]] - Mx[v_miss[i]]) / Wx[v_miss[i]]
                    X_sm_norm = (Res_old['X_sm'][t_miss[i], v_miss[i]] - Mx[v_miss[i]]) / Wx[v_miss[i]]
                    innov[i] = X_new_norm - X_sm_norm
                
                # Calculate P2 (covariance of innovations)
                P2 = np.zeros((n_news, n_news))
                for i in range(n_news):
                    for j in range(n_news):
                        h = abs(lag[i] - lag[j])
                        m = max(t_miss[i], t_miss[j])
                        
                        if t_miss[j] > t_miss[i]:
                            Pp = Plag[h + 1][:, :, m] if h + 1 < len(Plag) else Plag[-1][:, :, m]
                        else:
                            Pp = Plag[h + 1][:, :, m].T if h + 1 < len(Plag) else Plag[-1][:, :, m].T
                        
                        if v_miss[i] == v_miss[j] and t_miss[i] != t_miss[j]:
                            WW = 0
                        else:
                            WW = R_cov[v_miss[i], v_miss[j]]
                        
                        P2[i, j] = C[v_miss[i], :r] @ Pp @ C[v_miss[j], :r].T + WW
                
                # Calculate weights and news for each target variable
                if n_news > 0 and P2.size > 0:
                    try:
                        P2_inv = inv(P2)
                        # Calculate gain for each target variable
                        if is_scalar:
                            v_idx = v_news_arr[0]
                            gain = (Wx[v_idx] * C[v_idx, :r] @ P1 @ P2_inv).reshape(-1)
                            totnews = Wx[v_idx] * C[v_idx, :r] @ P1 @ P2_inv @ innov
                        else:
                            gain = np.zeros((n_targets, n_news))
                            totnews = np.zeros(n_targets)
                            for idx, v in enumerate(v_news_arr):
                                gain[idx, :] = Wx[v] * C[v, :r] @ P1 @ P2_inv
                                totnews[idx] = Wx[v] * C[v, :r] @ P1 @ P2_inv @ innov
                    except (np.linalg.LinAlgError, ValueError) as e:
                        # If inversion fails, use simpler approach
                        _logger.warning(
                            f"Matrix inversion failed for P2, using fallback values. "
                            f"Error: {type(e).__name__}: {str(e)}"
                        )
                        if is_scalar:
                            gain = np.ones(n_news) * 0.1
                            totnews = np.sum(innov) * 0.1
                        else:
                            gain = np.ones((n_targets, n_news)) * 0.1
                            totnews = np.ones(n_targets) * np.sum(innov) * 0.1
                else:
                    if is_scalar:
                        gain = np.zeros(n_news)
                        totnews = 0
                    else:
                        gain = np.zeros((n_targets, n_news))
                        totnews = np.zeros(n_targets)
                
                # Organize output
                if is_scalar:
                    singlenews = np.full(N, np.nan)
                    actual = np.full(N, np.nan)
                    forecast = np.full(N, np.nan)
                    weight = np.full(N, np.nan)
                    
                    for i in range(n_news):
                        actual[v_miss[i]] = X_new[t_miss[i], v_miss[i]]
                        forecast[v_miss[i]] = Res_old['X_sm'][t_miss[i], v_miss[i]]
                        if i < len(gain):
                            singlenews[v_miss[i]] = gain[i] * innov[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                            weight[v_miss[i]] = gain[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                else:
                    # Multiple targets: singlenews is (N, n_targets)
                    singlenews = np.full((N, n_targets), np.nan)
                    actual = np.full(N, np.nan)
                    forecast = np.full(N, np.nan)
                    weight = np.full((N, n_targets), np.nan)
                    
                    for i in range(n_news):
                        actual[v_miss[i]] = X_new[t_miss[i], v_miss[i]]
                        forecast[v_miss[i]] = Res_old['X_sm'][t_miss[i], v_miss[i]]
                        for idx in range(n_targets):
                            if i < len(gain[idx]):
                                singlenews[v_miss[i], idx] = gain[idx, i] * innov[i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                                weight[v_miss[i], idx] = gain[idx, i] / Wx[v_miss[i]] if Wx[v_miss[i]] != 0 else 0
                
                # Remove duplicates from v_miss
                v_miss = np.unique(v_miss)
        
        return y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov
    
    def __call__(
        self,
        target_series: str,
        view_date: Optional[Union[datetime, str]] = None,
        target_period: Optional[Union[datetime, str]] = None,
        return_result: bool = False
    ) -> Union[float, NowcastResult]:
        """Calculate nowcast for target series (callable interface).
        
        Parameters
        ----------
        target_series : str
            Target series ID
        view_date : datetime or str, optional
            Data view date. If None, uses latest available.
        target_period : datetime or str, optional
            Target period for nowcast. If None, uses latest.
        return_result : bool, default False
            If True, returns NowcastResult with additional information.
            If False, returns only the nowcast value (float).
            
        Returns
        -------
        float or NowcastResult
            Nowcast value if return_result=False, or NowcastResult if return_result=True
            
        Examples
        --------
        >>> nowcast = Nowcast(model)
        >>> value = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
        >>> # Or get full result
        >>> result = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1', return_result=True)
        >>> print(f"Nowcast: {result.nowcast_value}, Factors: {result.factors_at_view}")
        """
        if view_date is None:
            view_date = get_latest_time(self.model.time)
        elif isinstance(view_date, str):
            view_date = parse_timestamp(view_date)
        
        X_view, Time_view, _ = self.get_data_view(view_date)
        
        # Get series index
        i_series = find_series_index(self.model.config, target_series)
        
        # Get frequency
        frequencies = get_frequencies_from_config(self.model.config)
        freq = frequencies[i_series] if i_series < len(frequencies) else 'm'
        
        # Determine target period
        if target_period is None:
            target_period = get_latest_time(Time_view)
        else:
            target_period = self._parse_target_date(target_period, target_series)
        
        # Find time index for target period
        t_nowcast = find_time_index(Time_view, target_period)
        if t_nowcast is None:
            raise ValueError(f"Target period {target_period} not found in Time index")
        
        # Get forecast horizon based on clock frequency
        clock = get_clock_frequency(self.model.config, 'm')
        forecast_horizon, _ = _get_forecast_horizon_config(clock, horizon=None)
        
        # Extend data with forecast horizon if needed
        T, N = X_view.shape
        if t_nowcast >= T - forecast_horizon:
            # Need to extend data
            X_extended = np.vstack([X_view, np.full((forecast_horizon, N), np.nan)])
        else:
            X_extended = X_view
        
        # Use _news_dfm to calculate nowcast
        # For nowcast calculation, we use the same data for old and new (no update)
        try:
            nowcast_cache_key = f"nowcast:{view_date}"
            res_cache = self._get_kalman_result(nowcast_cache_key, X_extended)
            y_new = self._news_dfm(
                X_extended,
                X_extended,
                t_nowcast,
                i_series,
                Res_old_cache=res_cache,
                Res_new_cache=res_cache
            )[1]
            
            # Handle both scalar and array returns
            if isinstance(y_new, np.ndarray):
                if y_new.size == 1:
                    nowcast_value = float(y_new.item())
                else:
                    nowcast_value = float(y_new[0])
            else:
                nowcast_value = float(y_new)
            
            # Return simple float if not requested
            if not return_result:
                return nowcast_value
            
            # Return full NowcastResult
            # Use helper method pattern for consistency
            return self._create_nowcast_result_with_metadata(
                target_series=target_series,
                target_period=target_period,
                view_date=view_date,
                nowcast_value=nowcast_value,
                X_view=X_view,
                Time_view=Time_view
            )
            
        except Exception as e:
            _logger.error(f"Nowcast calculation failed: {e}")
            raise
    
    def decompose(
        self,
        target_series: str,
        target_period: Union[datetime, str],
        view_date_old: Union[datetime, str],
        view_date_new: Union[datetime, str],
        return_dict: bool = False
    ) -> Union[NewsDecompResult, Dict[str, Any]]:
        """Decompose forecast update into news contributions.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        target_period : datetime or str
            Target period (e.g., '2024Q1')
        view_date_old : datetime or str
            Older data view date
        view_date_new : datetime or str
            Newer data view date
        return_dict : bool, default False
            If True, returns dictionary (for backward compatibility).
            If False, returns NewsDecompResult dataclass.
            
        Returns
        -------
        NewsDecompResult or Dict[str, Any]
            News decomposition result. If return_dict=True, returns dictionary with keys:
            - 'y_old': float - old forecast
            - 'y_new': float - new forecast
            - 'change': float - forecast update
            - 'singlenews': np.ndarray - news contributions per series
            - 'top_contributors': List[Tuple[str, float]] - top contributors
            - 'actual': np.ndarray - actual values of new data
            - 'forecast': np.ndarray - forecasted values
            - 'weight': np.ndarray - weights
            - 't_miss': np.ndarray - time indices of new data
            - 'v_miss': np.ndarray - variable indices of new data
            - 'innov': np.ndarray - innovation terms
        """
        # Get data views
        X_old, Time_old, _ = self.get_data_view(view_date_old)
        X_new, Time_new, _ = self.get_data_view(view_date_new)
        
        # Find target period index
        i_series = find_series_index(self.model.config, target_series)
        target_date = self._parse_target_date(target_period, target_series)
        t_fcst = find_time_index(Time_new, target_date)
        
        if t_fcst is None:
            raise ValueError(f"Target period {target_period} not found in time index")
        
        # Ensure same time dimension
        T_old = X_old.shape[0]
        T_new = X_new.shape[0]
        if T_new > T_old:
            X_old = np.vstack([X_old, np.full((T_new - T_old, X_old.shape[1]), np.nan)])
        
        cache_key_old = f"{view_date_old}"
        cache_key_new = f"{view_date_new}"
        Res_old_cache = self._get_kalman_result(cache_key_old, X_old)
        Res_new_cache = self._get_kalman_result(cache_key_new, X_new)
        
        # Call _news_dfm
        y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = \
            self._news_dfm(
                X_old,
                X_new,
                t_fcst,
                i_series,
                Res_old_cache=Res_old_cache,
                Res_new_cache=Res_new_cache
            )
        
        # Extract summary
        series_ids = get_series_ids(self.model.config)
        summary = self._extract_news_summary(singlenews, weight, series_ids, top_n=5)
        
        # Create NewsDecompResult
        news_result = NewsDecompResult(
            y_old=float(y_old),
            y_new=float(y_new),
            change=float(y_new - y_old),
            singlenews=singlenews,
            top_contributors=summary['top_contributors'],
            actual=actual,
            forecast=forecast,
            weight=weight,
            t_miss=t_miss,
            v_miss=v_miss,
            innov=innov
        )
        
        if return_dict:
            # Return dictionary for backward compatibility
            return {
                'y_old': news_result.y_old,
                'y_new': news_result.y_new,
                'change': news_result.change,
                'singlenews': news_result.singlenews,
                'top_contributors': news_result.top_contributors,
                'actual': news_result.actual,
                'forecast': news_result.forecast,
                'weight': news_result.weight,
                't_miss': news_result.t_miss,
                'v_miss': news_result.v_miss,
                'innov': news_result.innov
            }
        
        return news_result
    
    def update(
        self,
        target_series: str,
        target_period: Union[datetime, str],
        view_date_old: Union[datetime, str],
        view_date_new: Union[datetime, str],
        save_callback: Optional[Callable] = None
    ) -> NewsDecompResult:
        """Update nowcast and decompose changes.
        
        This method combines nowcast calculation and news decomposition,
        and optionally saves results via callback.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        target_period : datetime or str
            Target period (e.g., '2024Q1')
        view_date_old : datetime or str
            Older data view date
        view_date_new : datetime or str
            Newer data view date
        save_callback : Callable, optional
            Optional callback function to save nowcast results.
            Called with: (target_series, target_period, view_date_old, view_date_new, news_result)
            
        Returns
        -------
        NewsDecompResult
            News decomposition result
        """
        # Get news decomposition (always return NewsDecompResult)
        news_result = self.decompose(target_series, target_period, view_date_old, view_date_new, return_dict=False)
        if not isinstance(news_result, NewsDecompResult):
            # This should not happen, but handle for type safety
            raise TypeError("Expected NewsDecompResult from decompose()")
        news = news_result
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Nowcast Update: {target_series} at {target_period}")
        print(f"{'='*70}")
        print(f"Data view: {view_date_old} → {view_date_new}")
        print(f"Old forecast: {news.y_old:.4f}")
        print(f"New forecast: {news.y_new:.4f}")
        print(f"Change: {news.change:.4f}")
        
        if len(news.top_contributors) > 0:
            print(f"\nTop 5 Contributors:")
            for series_id, impact in news.top_contributors:
                print(f"  {series_id}: {impact:.4f}")
        
        # Save via callback if provided
        if save_callback is not None:
            try:
                save_callback(
                    target_series=target_series,
                    target_period=target_period,
                    view_date_old=view_date_old,
                    view_date_new=view_date_new,
                    news=news
                )
            except Exception as e:
                _logger.warning(f"Save callback failed: {e}")
        
        return news
    
    def _extract_news_summary(
        self,
        singlenews: np.ndarray,
        weight: np.ndarray,
        series_ids: List[str],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Extract news summary from news decomposition results.
        
        Parameters
        ----------
        singlenews : np.ndarray
            Individual news contributions (N,) or (N, n_targets)
        weight : np.ndarray
            Weights (N,) or (N, n_targets)
        series_ids : List[str]
            Series IDs
        top_n : int, default 5
            Number of top contributors to include
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'total_impact', 'top_contributors', etc.
        """
        # Handle both 1D and 2D arrays
        if singlenews.ndim == 1:
            news_contributions = singlenews
            weights = weight
        else:
            # If 2D, use first target (column 0)
            news_contributions = singlenews[:, 0]
            weights = weight[:, 0] if weight.ndim > 1 else weight
        
        # Calculate total impact
        total_impact = np.nansum(news_contributions)
        
        # Get top contributors
        abs_contributions = np.abs(news_contributions)
        top_indices = np.argsort(abs_contributions)[::-1][:top_n]
        
        # Build list of top contributors
        top_contributors = []
        for idx in top_indices:
            if not np.isnan(news_contributions[idx]):
                if idx < len(series_ids):
                    series_id = series_ids[idx]
                else:
                    series_id = get_series_id_by_index(self.model.config, idx)
                impact = float(news_contributions[idx])
                top_contributors.append((series_id, impact))
        
        return {
            'total_impact': float(total_impact),
            'top_contributors': top_contributors,
            'revision_impact': 0.0,  # Placeholder
            'release_impact': float(total_impact)
        }
    
    def backtest(
        self,
        target_series: str,
        target_date: Union[datetime, str],
        backward_steps: int,
        higher_freq: bool = False,
        include_actual: bool = True
    ) -> BacktestResult:
        """Perform pseudo real-time backtest for target series.
        
        This method generates backward data views at regular intervals and calculates
        nowcasts for each view, allowing evaluation of model performance in a pseudo
        real-time setting.
        
        Parameters
        ----------
        target_series : str
            Target series ID to nowcast
        target_date : datetime or str
            Target date to nowcast (e.g., '2024Q4' or datetime(2024, 12, 31))
        backward_steps : int
            Number of backward steps to test
        higher_freq : bool, default False
            If True, use frequency one step faster than clock frequency for backward steps.
            If False, use clock frequency.
        include_actual : bool, default True
            Whether to compare nowcasts with actual values (if available)
            
        Returns
        -------
        BacktestResult
            Backtest results with nowcasts, news decomposition, and metrics
            
        Examples
        --------
        >>> nowcast = Nowcast(model)
        >>> result = nowcast.backtest(
        ...     target_series='gdp',
        ...     target_date='2024Q4',
        ...     backward_steps=20,
        ...     higher_freq=True
        ... )
        >>> print(f"Overall RMSE: {result.overall_rmse:.4f}")
        >>> result.plot(save_path='backtest_results.png')
        """
        # Get clock frequency
        clock = get_clock_frequency(self.model.config, 'm')
        
        # Determine backward frequency
        if higher_freq:
            backward_freq = _get_higher_frequency(clock)
            if backward_freq is None:
                _logger.warning(
                    f"No higher frequency available for clock '{clock}'. "
                    f"Using clock frequency instead."
                )
                backward_freq = clock
        else:
            backward_freq = clock
        
        # Parse target_date
        target_date = self._parse_target_date(target_date, target_series)
        
        # Get target series index
        i_series = find_series_index(self.model.config, target_series)
        
        # Generate backward view dates (from oldest to newest)
        # Step 0 = target_date, step N-1 = oldest date
        view_dates = []
        for step in range(backward_steps):
            view_date = _calculate_backward_date(target_date, step, backward_freq)
            view_dates.append(view_date)
        
        # Reverse to get from oldest (step 0) to newest (step N-1 = target_date)
        # This ensures step 0 is the oldest view and step N-1 is closest to target
        view_dates = list(reversed(view_dates))
        
        # Validate that we have valid dates
        if len(view_dates) != backward_steps:
            raise ValueError(f"Failed to generate {backward_steps} view dates. Got {len(view_dates)}")
        
        # Initialize result lists
        view_list: List[DataView] = []
        nowcast_results: List[NowcastResult] = []
        news_results: List[Optional[NewsDecompResult]] = []
        actual_values = []
        failed_steps: List[int] = []
        
        # Create base DataView factory
        base_view = DataView.from_arrays(
            X=self.model.data,
            Time=self.model.time,
            Z=self.model.original_data,
            config=self.model.config,
            X_frame=getattr(self.model, 'data_frame', None)
        )
        
        # Helper function to create placeholder NowcastResult
        def _create_placeholder_nowcast(view_date: datetime) -> NowcastResult:
            """Create placeholder NowcastResult with NaN values."""
            return NowcastResult(
                target_series=target_series,
                target_period=target_date,
                view_date=view_date,
                nowcast_value=np.nan,
                factors_at_view=None,
                dfm_result=None,
                data_availability=None
            )
        
        # Helper function to get actual value
        def _get_actual_value() -> float:
            """Get actual value for target series at target date."""
            if not include_actual:
                return np.nan
            t_idx = find_time_index(self.model.time, target_date)
            if t_idx is not None and t_idx < self.model.data.shape[0] and i_series < self.model.data.shape[1]:
                return self.model.data[t_idx, i_series]
            return np.nan
        
        # Helper function to create NowcastResult from successful calculation
        # (Used as fallback if __call__ returns float instead of NowcastResult)
        def _create_nowcast_result(view_date: datetime, nowcast_value: float) -> NowcastResult:
            """Create NowcastResult from successful nowcast calculation.
            
            This is a fallback helper used when __call__ returns float instead of NowcastResult.
            Uses the consolidated _create_nowcast_result_with_metadata method.
            """
            # Get data view to extract additional information
            X_view, Time_view, _ = self.get_data_view(view_date)
            
            # Use consolidated method
            return self._create_nowcast_result_with_metadata(
                target_series=target_series,
                target_period=target_date,
                view_date=view_date,
                nowcast_value=nowcast_value,
                X_view=X_view,
                Time_view=Time_view
            )
        
        # Process each backward step
        for step_idx, view_date in enumerate(view_dates):
            try:
                # Create data view for this date
                view = base_view.with_view_date(view_date)
                view_list.append(view)
                
                # Calculate nowcast
                try:
                    # Use return_result=True to get full NowcastResult
                    nowcast_result_obj = self(
                        target_series=target_series,
                        view_date=view_date,
                        target_period=target_date,
                        return_result=True
                    )
                    # Type check: should be NowcastResult when return_result=True
                    if isinstance(nowcast_result_obj, NowcastResult):
                        nowcast_result = nowcast_result_obj
                    else:
                        # Fallback: create result manually (should not happen)
                        _logger.warning(
                            f"Expected NowcastResult but got {type(nowcast_result_obj)}. "
                            f"Creating manually."
                        )
                        if isinstance(nowcast_result_obj, (int, float)):
                            nowcast_result = _create_nowcast_result(view_date, float(nowcast_result_obj))
                        else:
                            # Last resort: create with NaN
                            nowcast_result = _create_placeholder_nowcast(view_date)
                    nowcast_results.append(nowcast_result)
                    
                except Exception as e:
                    _logger.warning(
                        f"Nowcast calculation failed at step {step_idx} "
                        f"(view_date={view_date}): {e}"
                    )
                    failed_steps.append(step_idx)
                    nowcast_results.append(_create_placeholder_nowcast(view_date))
                
                # Calculate news decomposition (except for first step)
                if step_idx == 0:
                    # First step: no previous view, set to None (no change)
                    news_results.append(None)
                else:
                    try:
                        # Get previous view date
                        prev_view_date = view_dates[step_idx - 1]
                        
                        # Calculate news decomposition (returns NewsDecompResult)
                        news_result = self.decompose(
                            target_series=target_series,
                            target_period=target_date,
                            view_date_old=prev_view_date,
                            view_date_new=view_date,
                            return_dict=False
                        )
                        # Type check for safety
                        if isinstance(news_result, NewsDecompResult):
                            news_results.append(news_result)
                        else:
                            _logger.warning(f"Unexpected return type from decompose(): {type(news_result)}")
                            news_results.append(None)
                        
                    except Exception as e:
                        _logger.warning(
                            f"News decomposition failed at step {step_idx} "
                            f"(view_date={view_date}): {e}"
                        )
                        news_results.append(None)
                
                # Get actual value (same for all steps)
                actual_values.append(_get_actual_value())
                    
            except Exception as e:
                _logger.error(
                    f"Unexpected error at step {step_idx} (view_date={view_date}): {e}"
                )
                failed_steps.append(step_idx)
                # Create placeholder entries
                view_list.append(base_view.with_view_date(view_date))
                nowcast_results.append(_create_placeholder_nowcast(view_date))
                news_results.append(None)
                actual_values.append(np.nan)
        
        # Convert to arrays
        actual_values = np.array(actual_values)
        nowcast_values = np.array([r.nowcast_value for r in nowcast_results])
        
        # Calculate point-wise metrics
        errors = nowcast_values - actual_values
        mae_per_step = np.abs(errors)
        mse_per_step = np.where(np.isnan(errors), np.nan, errors ** 2)  # Preserve NaN
        rmse_per_step = np.sqrt(np.where(np.isnan(mse_per_step), np.nan, mse_per_step))
        
        # Calculate overall metrics (excluding NaN values and failed steps)
        # Exclude both NaN and failed steps from metric calculation
        valid_mask = ~(np.isnan(mae_per_step) | np.isnan(mse_per_step))
        if np.any(valid_mask):
            overall_mae = float(np.mean(mae_per_step[valid_mask]))
            overall_mse = float(np.mean(mse_per_step[valid_mask]))
            overall_rmse = float(np.sqrt(overall_mse))
        else:
            overall_mae = None
            overall_mse = None
            overall_rmse = None
            _logger.warning(
                f"No valid metrics calculated for backtest. "
                f"All {backward_steps} steps had NaN values or failed."
            )
        
        # Create and return BacktestResult
        return BacktestResult(
            target_series=target_series,
            target_date=target_date,
            backward_steps=backward_steps,
            higher_freq=higher_freq,
            backward_freq=backward_freq,
            view_list=view_list,
            nowcast_results=nowcast_results,
            news_results=news_results,
            actual_values=actual_values,
            errors=errors,
            mae_per_step=mae_per_step,
            mse_per_step=mse_per_step,
            rmse_per_step=rmse_per_step,
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            overall_mse=overall_mse,
            failed_steps=failed_steps
        )
