"""News decomposition for nowcasting updates.

This module implements the news decomposition framework for understanding how
new data releases affect nowcasts. The "news" is defined as the difference
between the new data release and the model's previous forecast, decomposed
into contributions from each data series.

The news decomposition provides:
- Forecast updates when new data arrives
- Attribution of forecast changes to specific data series
- Understanding of which data releases are most informative

This is essential for nowcasting applications where policymakers need to
understand the drivers of forecast revisions.
"""

import numpy as np
from scipy.linalg import pinv, inv
from typing import Tuple, Optional, Dict, Union, List, Any, Callable
import pandas as pd
import logging

from .kalman import skf, fis, miss_data
from .config import DFMConfig
from .dfm import DFMResult

# Set up logger
_logger = logging.getLogger(__name__)

FORECAST_HORIZON_MONTHS: int = 12
DEFAULT_FALLBACK_DATE: str = '2017-01-01'


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
    Issues a UserWarning if configs differ. Logs warnings if comparison fails.
    """
    try:
        # Compare series IDs using new API
        if hasattr(saved_config, 'get_series_ids'):
            saved_ids = saved_config.get_series_ids()
        elif hasattr(saved_config, 'SeriesID'):
            saved_ids = saved_config.SeriesID
        else:
            saved_ids = []
        
        if hasattr(current_config, 'get_series_ids'):
            current_ids = current_config.get_series_ids()
        elif hasattr(current_config, 'SeriesID'):
            current_ids = current_config.SeriesID
        else:
            current_ids = []
        if saved_ids != current_ids:
                import warnings
                warnings.warn(
                    "Config used in estimation differs from current config. "
                    "Results may be inconsistent. Consider re-estimating.",
                    UserWarning
                )
    except Exception as e:
        _logger.warning(f"update_nowcast: Config consistency check failed: {type(e).__name__}: {str(e)}")
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


def news_dfm(X_old: np.ndarray, X_new: np.ndarray, Res: Union[DFMResult, Dict[str, Any]], 
            t_fcst: int, v_news: Union[int, np.ndarray, List[int]]) -> Tuple[
                Union[float, np.ndarray], Union[float, np.ndarray], np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate changes in news from data releases.
    
    This function implements the news decomposition algorithm for dynamic factor models.
    It computes how new data releases affect forecasts of target variables by decomposing
    the forecast update into contributions from each new data point (news).
    
    The algorithm:
    1. Identifies new data points (available in X_new but not in X_old)
    2. Runs Kalman filter/smoother on both old and new datasets
    3. Computes forecast updates (y_new - y_old) for target variables
    4. Decomposes the update into individual news contributions from each new data point
    
    Parameters:
    -----------
    X_old : np.ndarray
        Old data matrix (T x N), where T is time periods and N is number of series.
        Contains data available before the new release.
    X_new : np.ndarray
        New data matrix (T x N), same dimensions as X_old.
        Contains data after the new release. Must have same number of series as X_old.
    Res : Union[DFMResult, Dict[str, Any]]
        DFM estimation results. Can be:
        - DFMResult: Direct result object from dfm() function
        - Dict: Dictionary containing 'Res' key with DFMResult, and optionally 'Config'
        Must have attributes: C, Mx, Wx, X_sm (or equivalent keys if dict)
    t_fcst : int
        Target time index (0-based) for the forecast period.
        Must satisfy 0 <= t_fcst < T (number of time periods in X_new).
    v_news : Union[int, np.ndarray, List[int]]
        Target variable index(s) (0-based). Specifies which series to forecast.
        Can be:
        - int: Single target variable
        - np.ndarray or List[int]: Multiple target variables (MATLAB-compatible)
        All indices must be in range [0, N) where N is number of series.
        
    Returns:
    --------
    Tuple containing 9 elements:
    
    y_old : Union[float, np.ndarray]
        Forecast of target variable(s) using old data.
        - If v_news is scalar: float
        - If v_news is array: (n_targets,) array
    y_new : Union[float, np.ndarray]
        Forecast of target variable(s) using new data.
        - If v_news is scalar: float
        - If v_news is array: (n_targets,) array
    singlenews : np.ndarray
        Individual news contributions from each new data point.
        - If v_news is scalar: (N,) array, where singlenews[i] is contribution from series i
        - If v_news is array: (N, n_targets) array, where each column is news for one target
    actual : np.ndarray
        Actual values of new data points (N,). NaN for series with no new data.
    forecast : np.ndarray
        Forecasted values (before new data) for new data points (N,). NaN for series with no new data.
    weight : np.ndarray
        Weights for news contributions (N,). NaN for series with no new data.
        - If v_news is scalar: (N,) array
        - If v_news is array: (N, n_targets) array
    t_miss : np.ndarray
        Time indices where new data is available (n_news,). Integer array.
    v_miss : np.ndarray
        Variable indices where new data is available (n_news,). Integer array.
    innov : np.ndarray
        Innovation terms (n_news,). Forecast errors for new data points.
    
    Raises:
    ------
    ValueError
        If inputs are invalid:
        - X_old or X_new are not 2D numpy arrays
        - X_old and X_new have different number of series
        - t_fcst is out of bounds
        - v_news indices are out of range
        - Res does not have required attributes (C, Mx, Wx, X_sm)
    
    Notes:
    -----
    - Now supports both single and multiple target variables, matching MATLAB functionality.
    - The function handles two cases:
      1. NO FORECAST CASE: Target variable(s) already observed at t_fcst in X_new
      2. FORECAST CASE: Target variable(s) not yet observed, requires forecast update
    - News decomposition follows the methodology in Banbura et al. (2015).
    - If matrix inversion fails during gain calculation, fallback values are used and logged.
    
    Examples:
    --------
    >>> from dfm_python import dfm, news_dfm
    >>> # Estimate DFM model
    >>> Res = dfm(X, config)
    >>> # Calculate news for single target variable
    >>> y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = \\
    ...     news_dfm(X_old, X_new, Res, t_fcst=100, v_news=0)
    >>> # Calculate news for multiple target variables
    >>> targets = [0, 1, 2]
    >>> y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = \\
    ...     news_dfm(X_old, X_new, Res, t_fcst=100, v_news=targets)
    """
    # Input validation
    if not isinstance(X_old, np.ndarray) or X_old.ndim != 2:
        raise ValueError(f"news_dfm: X_old must be a 2D numpy array, got {type(X_old)} with shape {getattr(X_old, 'shape', 'N/A')}")
    if not isinstance(X_new, np.ndarray) or X_new.ndim != 2:
        raise ValueError(f"news_dfm: X_new must be a 2D numpy array, got {type(X_new)} with shape {getattr(X_new, 'shape', 'N/A')}")
    if X_old.shape[1] != X_new.shape[1]:
        raise ValueError(f"news_dfm: X_old and X_new must have same number of series. Got {X_old.shape[1]} and {X_new.shape[1]}")
    if not isinstance(t_fcst, (int, np.integer)) or t_fcst < 0:
        raise ValueError(f"news_dfm: t_fcst must be a non-negative integer, got {t_fcst}")
    if t_fcst >= X_new.shape[0]:
        raise ValueError(f"news_dfm: t_fcst ({t_fcst}) must be less than number of time periods ({X_new.shape[0]})")
    
    # Validate Res structure
    if isinstance(Res, dict):
        if 'Res' not in Res:
            raise ValueError("news_dfm: Res dict must contain 'Res' key")
        Res_actual = Res['Res']
    else:
        Res_actual = Res
    
    # Check required attributes
    required_attrs = ['C', 'Mx', 'Wx', 'X_sm']
    for attr in required_attrs:
        if not hasattr(Res_actual, attr):
            raise ValueError(f"news_dfm: Res must have '{attr}' attribute. Got {type(Res_actual)}")
    
    # Normalize v_news to array for consistent handling
    v_news_arr = np.atleast_1d(v_news)
    is_scalar = isinstance(v_news, (int, np.integer)) or (isinstance(v_news, np.ndarray) and v_news.ndim == 0)
    n_targets = len(v_news_arr)
    
    # Validate v_news indices
    if np.any(v_news_arr < 0) or np.any(v_news_arr >= X_new.shape[1]):
        raise ValueError(f"news_dfm: v_news indices must be in range [0, {X_new.shape[1]}), got {v_news_arr}")
    
    r = Res_actual.C.shape[1]
    _, N = X_new.shape
    
    # Check if targets are already observed
    targets_observed = np.array([not np.isnan(X_new[t_fcst, v]) for v in v_news_arr])
    
    if np.all(targets_observed):
        # NO FORECAST CASE: Already values for all target variables at time t_fcst
        Res_old = para_const(X_old, Res_actual, 0)
        
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
        Mx = Res_actual.Mx
        Wx = Res_actual.Wx
        
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
            Res_old = para_const(X_old, Res_actual, 0)
            Res_new = para_const(X_new, Res_actual, 0)
            
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
            k = max([np.max(np.abs(lag)), np.max(lag) - np.min(lag)])
            k = int(k)
            
            C = Res_actual.C
            R_cov = Res_actual.R.T
            
            n_news = len(lag)
            
            # Smooth old dataset
            Res_old = para_const(X_old, Res_actual, k)
            Plag = Res_old['Plag']
            
            # Smooth new dataset
            Res_new = para_const(X_new, Res_actual, 0)
            
            # Get nowcasts for all target variables
            if is_scalar:
                y_old = Res_old['X_sm'][t_fcst, v_news_arr[0]]
                y_new = Res_new['X_sm'][t_fcst, v_news_arr[0]]
            else:
                y_old = np.array([Res_old['X_sm'][t_fcst, v] for v in v_news_arr])
                y_new = np.array([Res_new['X_sm'][t_fcst, v] for v in v_news_arr])
            
            # Calculate projection matrices (simplified)
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
            
            # Calculate P2 (covariance of innovations) - simplified
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
                        f"news_dfm: Matrix inversion failed for P2, using fallback values. "
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


# Database saving is application-specific and should be implemented in adapters.
# This keeps the DFM module generic and database-agnostic.
def update_nowcast(X_old: np.ndarray, X_new: np.ndarray, Time: pd.DatetimeIndex,
                  config: DFMConfig, Res: Union[DFMResult, Dict[str, Any]], 
                  series: str, period: str,
                  vintage_old: str, vintage_new: str,
                  model_id: Optional[int] = None,
                  save_callback: Optional[Callable] = None) -> None:
    """Update nowcast and decompose changes into news.
    
    Parameters:
    -----------
    X_old : np.ndarray
        Old vintage data (T_old x N)
    X_new : np.ndarray
        New vintage data (T_new x N)
    Time : pd.DatetimeIndex
        Time index
    config : DFMConfig
        Model configuration
    Res : DFMResult or dict
        DFM estimation results. If dict, must contain 'Res' and optionally 'Config'.
        If DFMResult and has .Config attribute, it will be checked for consistency.
    series : str
        Target series ID
    period : str
        Target period (e.g., '2016q4')
    vintage_old : str
        Old vintage date
    vintage_new : str
        New vintage date
    model_id : int, optional
        Model ID for saving (passed to save_callback if provided)
    save_callback : Callable, optional
        Optional callback function to save nowcast results.
        If provided, will be called with nowcast parameters.
        Example: Implement save_callback in your application adapter
                 save_callback=lambda **kwargs: save_nowcast_to_db(**kwargs)
        
    Notes:
    ------
    If Res contains a saved Config and it differs from the current config, a warning
    is issued. This matches MATLAB behavior where Res.Config is checked for consistency.
    
    Database saving is robust with error handling - failures will log warnings
    but not interrupt the nowcast computation.
    """
    # Input validation
    if not isinstance(X_old, np.ndarray) or X_old.ndim != 2:
        raise ValueError(f"update_nowcast: X_old must be a 2D numpy array, got {type(X_old)}")
    if not isinstance(X_new, np.ndarray) or X_new.ndim != 2:
        raise ValueError(f"update_nowcast: X_new must be a 2D numpy array, got {type(X_new)}")
    if X_old.shape[1] != X_new.shape[1]:
        raise ValueError(f"update_nowcast: X_old and X_new must have same number of series. Got {X_old.shape[1]} and {X_new.shape[1]}")
    if not isinstance(series, str):
        raise ValueError(f"update_nowcast: series must be a string, got {type(series)}")
    if not isinstance(period, str):
        raise ValueError(f"update_nowcast: period must be a string, got {type(period)}")
    # Get series IDs and find index
    if hasattr(config, 'get_series_ids'):
        series_ids = config.get_series_ids()
    elif hasattr(config, 'SeriesID'):
        series_ids = config.SeriesID
    else:
        series_ids = []
    
    if not series_ids:
        raise ValueError(f"update_nowcast: config must have series definitions")
    if series not in series_ids:
        raise ValueError(f"update_nowcast: series '{series}' not found in config")
    
    # Validate period format
    try:
        i_series = series_ids.index(series)
        if hasattr(config, 'get_frequencies'):
            frequencies = config.get_frequencies()
        elif hasattr(config, 'Frequency'):
            frequencies = config.Frequency
        else:
            frequencies = []
        freq = frequencies[i_series] if i_series < len(frequencies) else 'm'
        if freq == 'm':
            # Format: YYYYmMM (e.g., "2016m12")
            parts = period.split('m')
            if len(parts) != 2:
                raise ValueError(f"update_nowcast: period '{period}' must be in format 'YYYYmMM' for monthly frequency")
            year, month = int(parts[0]), int(parts[1])
            if not (1 <= month <= 12):
                raise ValueError(f"update_nowcast: month must be between 1 and 12, got {month}")
        elif freq == 'q':
            # Format: YYYYqQ (e.g., "2016q4")
            parts = period.split('q')
            if len(parts) != 2:
                raise ValueError(f"update_nowcast: period '{period}' must be in format 'YYYYqQ' for quarterly frequency")
            year, q = int(parts[0]), int(parts[1])
            if not (1 <= q <= 4):
                raise ValueError(f"update_nowcast: quarter must be between 1 and 4, got {q}")
        else:
            raise ValueError(f"update_nowcast: Unsupported frequency '{freq}' for period validation")
    except (ValueError, IndexError) as e:
        if isinstance(e, ValueError) and "not found" not in str(e):
            raise ValueError(f"update_nowcast: Invalid period format '{period}': {str(e)}")
        raise
    
    # Handle dict format (from saved results) - extract Res and check config consistency
    if isinstance(Res, dict):
        if 'Config' in Res and 'Res' in Res:
            saved_config = Res['Config']
            Res = Res['Res']
            _check_config_consistency(saved_config, config)
        elif 'Res' in Res:
            Res = Res['Res']
    
    # Check config consistency if Res has Config attribute
    if hasattr(Res, 'Config'):
        _check_config_consistency(Res.Config, config)
    
    # Make sure datasets are same size
    N = X_new.shape[1]
    T_old = X_old.shape[0]
    T_new = X_new.shape[0]
    
    if T_new > T_old:
        X_old = np.vstack([X_old, np.full((T_new - T_old, N), np.nan)])
    
    # Append 1 year (12 months) for forecasting horizon
    X_old = np.vstack([X_old, np.full((FORECAST_HORIZON_MONTHS, N), np.nan)])
    X_new = np.vstack([X_new, np.full((FORECAST_HORIZON_MONTHS, N), np.nan)])
    
    # Extend time index
    last_date = Time.iloc[-1] if hasattr(Time, 'iloc') else Time[-1]
    if hasattr(last_date, 'month'):
        new_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_HORIZON_MONTHS,
            freq='MS'
        )
    else:
        # Fallback if not pandas datetime
        new_dates = pd.date_range(start=DEFAULT_FALLBACK_DATE, periods=FORECAST_HORIZON_MONTHS, freq='MS')
    
    # Handle both Index and Series types
    if isinstance(Time, pd.Series):
        Time = pd.concat([Time, pd.Series(new_dates)])
    elif hasattr(Time, 'append'):
        try:
            Time = Time.append(pd.Index(new_dates))
        except AttributeError:
            Time = pd.concat([Time, pd.Index(new_dates)])
    else:
        Time = pd.concat([Time, pd.Index(new_dates)])
    
    # Find series index
    if hasattr(config, 'get_series_ids'):
        series_ids = config.get_series_ids()
    elif hasattr(config, 'SeriesID'):
        series_ids = config.SeriesID
    else:
        series_ids = []
    
    try:
        i_series = series_ids.index(series)
    except ValueError:
        raise ValueError(f"Series {series} not found in configuration")
    
    if hasattr(config, 'get_series_names'):
        series_names = config.get_series_names()
    elif hasattr(config, 'SeriesName'):
        series_names = config.SeriesName
    else:
        series_names = []
    
    if hasattr(config, 'get_frequencies'):
        frequencies = config.get_frequencies()
    elif hasattr(config, 'Frequency'):
        frequencies = config.Frequency
    else:
        frequencies = []
    series_name = series_names[i_series] if i_series < len(series_names) else series
    freq = frequencies[i_series] if i_series < len(frequencies) else 'm'
    
    # Parse period
    if freq == 'm':
        year_str, month_str = period.split('m')
        year = int(year_str)
        month = int(month_str)
        target_date = pd.Timestamp(year, month, 1)
    elif freq == 'q':
        year_str, q_str = period.split('q')
        year = int(year_str)
        q = int(q_str)
        month = 3 * q
        target_date = pd.Timestamp(year, month, 1)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    # Find time index
    t_nowcast = None
    for i, t in enumerate(Time):
        if hasattr(t, 'year') and hasattr(t, 'month'):
            if t.year == target_date.year and t.month == target_date.month:
                t_nowcast = i
                break
    
    if t_nowcast is None:
        raise ValueError('Period is out of nowcasting horizon (up to one year ahead).')
    
    # Separate revisions from new releases
    X_rev = X_new.copy()
    X_rev[np.isnan(X_old)] = np.nan
    
    # Compute news
    # Impact from data revisions
    y_old = news_dfm(X_old, X_rev, Res, t_nowcast, i_series)[0]
    
    # Impact from data releases
    y_rev, y_new, _, actual, forecast, weight = news_dfm(X_rev, X_new, Res, t_nowcast, i_series)[:6]
    
    # Display output
    print('\n\n\n')
    print(f'Nowcast Update: {pd.to_datetime(vintage_new).strftime("%B %d, %Y")}')
    
    if freq == 'q':
        period_str = pd.Timestamp(Time[t_nowcast]).strftime('%Y:Q%q')
    else:
        period_str = pd.Timestamp(Time[t_nowcast]).strftime('%Y-%m')
    
    # Get units
    if hasattr(config, 'series') and i_series < len(config.series):
        units = config.series[i_series].units
    elif hasattr(config, 'UnitsTransformed') and i_series < len(config.UnitsTransformed):
        units = config.UnitsTransformed[i_series]
    else:
        units = ""
    print(f'Nowcast for {series_name} ({units}), {period_str}')
    
    if len(forecast) == 0 or np.all(np.isnan(forecast)):
        print('\n  No forecast was made.\n')
    else:
        impact_revisions = y_rev - y_old
        news = actual - forecast
        impact_releases = weight * news
        
        # Create summary table
        data_released = np.any(np.isnan(X_old) & ~np.isnan(X_new), axis=0)
        
        print('\n  Nowcast Impact Decomposition')
        print('  Note: The displayed output is subject to rounding error\n')
        print(f'                  {pd.to_datetime(vintage_old).strftime("%b %d")} nowcast:                  {y_old:5.2f}')
        print(f'      Impact from data revisions:      {impact_revisions:5.2f}')
        print(f'       Impact from data releases:      {np.nansum(impact_releases):5.2f}')
        print('                                     +_________')
        print(f'                    Total impact:      {impact_revisions + np.nansum(impact_releases):5.2f}')
        print(f'                  {pd.to_datetime(vintage_new).strftime("%b %d")} nowcast:                  {y_new:5.2f}\n')
        
        # Display table for series with updates
        print('\n  Nowcast Detail Table \n')
        for i in range(N):
            if data_released[i] and not np.isnan(forecast[i]):
                if hasattr(config, 'get_series_ids'):
                    series_ids = config.get_series_ids()
                elif hasattr(config, 'SeriesID'):
                    series_ids = config.SeriesID
                else:
                    series_ids = []
                series_id = series_ids[i] if i < len(series_ids) else f"series_{i}"
                print(f'{series_id:20s}  Forecast: {forecast[i]:8.2f}  Actual: {actual[i]:8.2f}  '
                      f'Weight: {weight[i]:8.4f}  Impact: {impact_releases[i]:8.2f}')
        
        # Calculate and display RMSE for nowcast
        # Compare actual values to forecasts where both are available
        valid_mask = np.isfinite(actual) & np.isfinite(forecast)
        if np.any(valid_mask):
            actual_valid = actual[valid_mask]
            forecast_valid = forecast[valid_mask]
            if len(actual_valid) > 0:
                nowcast_rmse = np.sqrt(np.mean((actual_valid - forecast_valid) ** 2))
                print(f'\n  Nowcast RMSE (across all series with updates): {nowcast_rmse:.6f}')
        
        # Save nowcast using callback if provided (generic - not database-specific)
        if save_callback is not None:
            try:
                save_callback(
                    model_id=model_id,
                    series=series,
                    forecast_date=target_date,
                    forecast_value=y_new,
                    old_forecast_value=y_old,
                    impact_revisions=impact_revisions,
                    impact_releases=np.nansum(impact_releases),
                    total_impact=impact_revisions + np.nansum(impact_releases),
                    vintage_old=vintage_old,
                    vintage_new=vintage_new,
                    Res=Res
                )
            except Exception as e:
                import warnings
                warnings.warn(
                    f"save_callback failed: {e}. Continuing without saving.",
                    UserWarning
                )

