"""Nowcasting utility functions, news decomposition, and backtest classes.

This module contains helper functions and result classes extracted from nowcast.py
to keep the main file under 1000 lines.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from ..core.results import DFMResult
from ..core.state_space import skf, fis
from ..config import DFMConfig
from ..core.time import TimeIndex, parse_timestamp
from ..core.helpers import get_logger, get_series_id_by_index, get_periods_per_year
from ..dataloader.loader import read_data, sort_data, rem_nans_spline, calculate_release_date, transform_data

_logger = get_logger(__name__)


@dataclass
class NewsDecompResult:
    """Result from news decomposition calculation.
    
    This dataclass contains all information about how new data releases
    affect the nowcast, including the forecast update and contributions
    from each data series.
    
    Attributes
    ----------
    y_old : float
        Nowcast value using old data view
    y_new : float
        Nowcast value using new data view
    change : float
        Forecast update (y_new - y_old)
    singlenews : np.ndarray
        News contributions per series (N,) or (N, n_targets)
    top_contributors : List[Tuple[str, float]]
        Top contributors to the forecast update, sorted by absolute impact
    actual : np.ndarray
        Actual values of newly released data
    forecast : np.ndarray
        Forecasted values for new data (from old view)
    weight : np.ndarray
        Weights for news contributions (N,) or (N, n_targets)
    t_miss : np.ndarray
        Time indices of new data releases
    v_miss : np.ndarray
        Variable indices of new data releases
    innov : np.ndarray
        Innovation terms (standardized differences between actual and forecast)
    """
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


def para_const(X: np.ndarray, result: DFMResult, lag: int = 0) -> Dict[str, Any]:
    """Implement Kalman filter for news calculation with fixed parameters.
    
    This function applies the Kalman filter and smoother to a data matrix X
    using pre-estimated model parameters from a DFMResult. It is used in
    news decomposition when model parameters are already known.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N) with potentially missing values (NaN)
    result : DFMResult
        DFM result containing estimated parameters (A, C, Q, R, Mx, Wx, Z_0, V_0)
    lag : int, default 0
        Maximum lag for calculating Plag (smoothed factor covariances)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'Plag': List of smoothed factor covariances for different lags
        - 'P': Smoothed factor covariance matrix
        - 'X_sm': Smoothed data matrix (T x N)
        - 'F': Smoothed factors (T x r)
        - 'Z': Smoothed factors (T+1 x r, includes initial state)
        - 'V': Smoothed factor covariances (T+1 x r x r)
    
    Notes
    -----
    This function is based on the MATLAB para_const() function from the
    FRBNY Nowcasting Model. It implements Kalman filtering with fixed
    parameters for use in news decomposition calculations.
    
    The function standardizes the input data using Mx and Wx from the
    result, applies the Kalman filter and smoother, then transforms
    the smoothed factors back to observation space.
    """
    # Extract parameters from result
    Z_0 = result.Z_0
    V_0 = result.V_0
    A = result.A
    C = result.C
    Q = result.Q
    R = result.R
    Mx = result.Mx
    Wx = result.Wx
    
    T, N = X.shape
    r = A.shape[0]  # Number of factors
    
    # Standardize data: Y = (X - Mx) / Wx
    # Handle division by zero
    Wx_safe = np.where(Wx == 0, 1.0, Wx)
    Y = ((X - Mx) / Wx_safe).T  # Transpose to (N x T) for Kalman filter
    
    # Apply Kalman filter
    Sf = skf(Y, A, C, Q, R, Z_0, V_0)
    
    # Apply smoother
    Ss = fis(A, Sf)
    
    # Extract smoothed results
    Zsmooth = Ss.ZmT  # (T+1) x r (includes initial state)
    Vsmooth = Ss.VmT  # (T+1) x r x r
    
    # Smoothed factor covariances for transition matrix
    # Vs is V_{t|T} for t = 1, ..., T (skip initial state)
    Vs = Ss.VmT[1:, :, :]  # T x r x r
    Vf = Sf.VmU[:, :, 1:]  # r x r x T (filtered posterior covariance, skip initial)
    
    # Calculate Plag (smoothed factor covariances for different lags)
    Plag = [Vs]  # Plag[0] = Vs (lag 0)
    
    if lag > 0:
        for jk in range(1, lag + 1):
            Plag_jk = np.zeros_like(Vs)
            for jt in range(lag, T):
                # Calculate smoothed covariance for lag jk at time jt
                # As = V_{t-jk|t} * A' * (A * V_{t-jk|t} * A' + Q)^{-1}
                V_t_jk = Vf[:, :, jt - jk] if jt - jk >= 0 else Vs[0]
                try:
                    As = V_t_jk @ A.T @ np.linalg.pinv(A @ V_t_jk @ A.T + Q)
                    Plag_jk[jt] = As @ Plag[jk - 1][jt]
                except (np.linalg.LinAlgError, ValueError):
                    # Fallback if inversion fails
                    Plag_jk[jt] = Plag[jk - 1][jt]
            Plag.append(Plag_jk)
    
    # Transform factors to observation space
    # x_sm = Z * C' (standardized)
    x_sm = Zsmooth[1:, :] @ C.T  # T x N (skip initial state)
    
    # Unstandardize: X_sm = x_sm * Wx + Mx
    X_sm = x_sm * Wx + Mx  # T x N
    
    return {
        'Plag': Plag,
        'P': Vsmooth[1:, :, :],  # T x r x r (skip initial state)
        'X_sm': X_sm,  # T x N
        'F': Zsmooth[1:, :],  # T x r (smoothed factors, skip initial state)
        'Z': Zsmooth,  # (T+1) x r (includes initial state)
        'V': Vsmooth,  # (T+1) x r x r (includes initial state)
    }

# ============================================================================
# Nowcasting helper functions (merged from nowcast_helpers.py)
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
    from ..core.structure import FREQUENCY_HIERARCHY
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


# ============================================================================
# Backtest result classes (merged from backtest.py)
# ============================================================================




@dataclass
class BacktestResult:
    """Result from backtest evaluation of nowcasting model.
    
    This dataclass contains all information from a pseudo real-time backtest,
    including nowcasts at different view dates, news decomposition between steps,
    and evaluation metrics.
    """
    target_series: str
    target_date: datetime
    backward_steps: int
    higher_freq: bool
    backward_freq: str
    view_list: List  # List[DataView] - avoiding circular import
    nowcast_results: List["NowcastResult"]
    news_results: List[Optional["NewsDecompResult"]]
    actual_values: np.ndarray
    errors: np.ndarray
    mae_per_step: np.ndarray
    mse_per_step: np.ndarray
    rmse_per_step: np.ndarray
    overall_mae: Optional[float]
    overall_rmse: Optional[float]
    overall_mse: Optional[float]
    failed_steps: List[int]
    
    def plot(self, save_path: Optional[str] = None, show: bool = True):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Nowcast values vs actual
            ax1 = axes[0]
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax1.plot(view_dates, nowcast_values, 'o-', label='Nowcast', color='blue')
            if not np.all(np.isnan(self.actual_values)):
                ax1.axhline(y=self.actual_values[0], color='red', linestyle='--', label='Actual')
            ax1.set_xlabel('View Date')
            ax1.set_ylabel('Value')
            ax1.set_title(f'Backtest Results: {self.target_series} at {self.target_date}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Error metrics
            ax2 = axes[1]
            steps = range(self.backward_steps)
            ax2.plot(steps, self.rmse_per_step, 'o-', label='RMSE', color='green')
            ax2.set_xlabel('Backward Step')
            ax2.set_ylabel('Error')
            ax2.set_title('Error Metrics per Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    def plot_trajectory(self, save_path: Optional[str] = None, show: bool = True):
        """Plot nowcast trajectory over backward steps."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax.plot(view_dates, nowcast_values, 'o-', label='Nowcast Trajectory', color='blue', linewidth=2, markersize=8)
            
            if not np.all(np.isnan(self.actual_values)):
                ax.axhline(y=self.actual_values[0], color='red', linestyle='--', linewidth=2, label='Actual')
            
            ax.set_xlabel('View Date', fontsize=12)
            ax.set_ylabel('Nowcast Value', fontsize=12)
            ax.set_title(f'Nowcast Trajectory: {self.target_series} at {self.target_date}', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

# Additional helper functions from nowcast.py
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
        # Optimization: First step uses nowcast, subsequent steps use decompose results
        # to avoid redundant Kalman filter calculations
        for step_idx, view_date in enumerate(view_dates):
            try:
                # Create data view for this date
                view = base_view.with_view_date(view_date)
                view_list.append(view)
                
                if step_idx == 0:
                    # First step: calculate nowcast (no previous view for comparison)
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
                        news_results.append(None)  # No previous view for comparison
                        
                    except Exception as e:
                        _logger.warning(
                            f"Nowcast calculation failed at step {step_idx} "
                            f"(view_date={view_date}): {e}"
                        )
                        failed_steps.append(step_idx)
                        nowcast_results.append(_create_placeholder_nowcast(view_date))
                        news_results.append(None)
                else:
                    # Subsequent steps: use decompose to get both news and nowcast
                    try:
                        # Get previous view date
                        prev_view_date = view_dates[step_idx - 1]
                        
                        # Calculate news decomposition (returns NewsDecompResult)
                        # This already computes y_new (the nowcast value) efficiently
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
                            # Extract nowcast from decompose result (avoids redundant calculation)
                            nowcast_result = self._decomp_to_nowcast_result(
                                news_result,
                                target_series=target_series,
                                target_period=target_date,
                                view_date=view_date
                            )
                            nowcast_results.append(nowcast_result)
                        else:
                            _logger.warning(f"Unexpected return type from decompose(): {type(news_result)}")
                            news_results.append(None)
                            nowcast_results.append(_create_placeholder_nowcast(view_date))
                            failed_steps.append(step_idx)
                        
                    except Exception as e:
                        _logger.warning(
                            f"News decomposition failed at step {step_idx} "
                            f"(view_date={view_date}): {e}"
                        )
                        news_results.append(None)
                        nowcast_results.append(_create_placeholder_nowcast(view_date))
                        failed_steps.append(step_idx)
                
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


# Additional loader functions (moved to keep loader.py under 1000 lines)


# ============================================================================
# Data preprocessing functions (merged from preprocess.py)
# ============================================================================


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Transform a single time series according to formula.
    
    Parameters
    ----------
    Z : np.ndarray
        Raw time series (1D array)
    formula : str
        Transformation formula: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'
    freq : str
        Frequency code: 'm', 'q', 'sa', 'a'
    step : int
        Number of base periods per observation
        
    Returns
    -------
    np.ndarray
        Transformed series
    """
    T = len(Z)
    X = np.full(T, np.nan)
    
    if formula == 'lin':
        X[:] = Z
    elif formula == 'chg':
        # First difference
        if T > step:
            X[step:] = Z[step:] - Z[:-step]
    elif formula == 'ch1':
        # Year-over-year difference (generic based on frequency)
        year_step = get_periods_per_year(freq)
        if T > year_step:
            X[year_step:] = Z[year_step:] - Z[:-year_step]
    elif formula == 'pch':
        # Percent change
        if T > step:
            X[step:] = 100.0 * (Z[step:] - Z[:-step]) / np.abs(Z[:-step] + 1e-10)
    elif formula == 'pc1':
        # Year-over-year percent change (generic based on frequency)
        year_step = get_periods_per_year(freq)
        if T > year_step:
            X[year_step:] = 100.0 * (Z[year_step:] - Z[:-year_step]) / np.abs(Z[:-year_step] + 1e-10)

def transform_data(Z: np.ndarray, Time: TimeIndex, config: DFMConfig) -> Tuple[np.ndarray, TimeIndex, np.ndarray]:
    """Transform each data series according to configuration.
    
    Applies the specified transformation formula to each series based on its
    frequency and transformation type. Handles mixed-frequency data by
    applying transformations at the appropriate observation intervals.
    
    Supported frequencies: monthly (m), quarterly (q), semi-annual (sa), annual (a).
    Frequencies faster than the clock frequency are not supported.
    
    Parameters
    ----------
    Z : np.ndarray
        Raw data matrix (T x N)
    Time : TimeIndex
        Time index for the data
    config : DFMConfig
        Model configuration with transformation specifications
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : TimeIndex
        Time index (may be truncated after transformation)
    Z : np.ndarray
        Original data (may be truncated to match X)
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Validate frequencies - reject higher frequencies than clock
    clock = safe_get_attr(config, 'clock', 'm')
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
    
    frequencies = get_frequencies_from_config(config)
    series_ids = get_series_ids(config)
    for i, freq in enumerate(frequencies):
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
        if freq_hierarchy < clock_hierarchy:
            raise ValueError(
                f"Series '{series_ids[i]}' has frequency '{freq}' which is faster than clock '{clock}'. "
                f"Higher frequencies (daily, weekly) are not supported. "
                f"Please use monthly, quarterly, semi-annual, or annual frequencies only."
            )
    
    # Frequency to step mapping (step = number of base periods per observation)
    freq_to_step = {'m': 1, 'q': 3, 'sa': 6, 'a': 12}
    
    # DFMConfig always has series attribute, but check for safety
    transformations = [s.transformation for s in config.series] if hasattr(config, 'series') and config.series else ['lin'] * N
    
    for i in range(N):
        freq = frequencies[i] if i < len(frequencies) else clock
        step = freq_to_step.get(freq, 1)
        formula = transformations[i] if i < len(transformations) else 'lin'
        X[:, i] = _transform_series(Z[:, i], formula, freq, step)
    
    # Remove leading NaN rows (from differencing)
    drop = 0
    for t in range(T):
        if np.all(np.isnan(X[t, :])):
            drop += 1
        else:
            break
    
    if T > drop:
        return X[drop:], Time[drop:], Z[drop:]
    return X, Time, Z



# ============================================================================
# News summary extraction (from nowcast.py)
# ============================================================================


def _extract_news_summary_impl(
    singlenews: np.ndarray,
    weight: np.ndarray,
    series_ids: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """Extract summary statistics from news decomposition (implementation).
    
    Parameters
    ----------
    singlenews : np.ndarray
        News contributions (N,) or (N, n_targets)
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
        if idx < len(series_ids):
            top_contributors.append({
                'series_id': series_ids[idx],
                'contribution': float(news_contributions[idx]),
                'weight': float(weights[idx]) if idx < len(weights) else 0.0
            })
    
    return {
        'total_impact': float(total_impact),
        'top_contributors': top_contributors,
        'revision_impact': float(total_impact),  # Placeholder
        'release_impact': 0.0  # Placeholder
    }


