"""Nowcasting utility functions, news decomposition, and backtest classes.

This module contains helper functions and result classes extracted from nowcast.py
to keep the main file under 1000 lines.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from ..config.results import DFMResult
from ..config import DFMConfig
from ..utils.time import TimeIndex, parse_timestamp
from ..logger import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..nowcast.nowcast import NowcastResult
from ..utils.helpers import (
    get_series_id_by_index,
    get_periods_per_year,
    get_frequencies_from_config,
    get_series_ids,
    safe_get_attr,
)
from ..config.structure import FREQUENCY_HIERARCHY
from ..utils.time import clock_to_datetime_freq
from ..transformations.utils import read_data
# transform_data and _transform_series removed - use DataModule with custom transformers instead
from ..utils.data import sort_data, rem_nans_spline, calculate_release_date

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
    
    # Use PyTorch Lightning KalmanFilter
    try:
        import torch
        from ..ssm.kalman import KalmanFilter
        
        # Convert to torch tensors
        device = torch.device('cpu')  # Use CPU for nowcasting
        Y_torch = torch.tensor(Y, dtype=torch.float64, device=device)
        A_torch = torch.tensor(A, dtype=torch.float64, device=device)
        C_torch = torch.tensor(C, dtype=torch.float64, device=device)
        Q_torch = torch.tensor(Q, dtype=torch.float64, device=device)
        R_torch = torch.tensor(R, dtype=torch.float64, device=device)
        Z_0_torch = torch.tensor(Z_0, dtype=torch.float64, device=device)
        V_0_torch = torch.tensor(V_0, dtype=torch.float64, device=device)
        
        # Apply Kalman filter and smoother
        kalman = KalmanFilter()
        zsmooth, Vsmooth, VVsmooth, _ = kalman(
            Y_torch, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch
        )
        
        # Convert back to numpy
        Zsmooth = zsmooth.T.cpu().numpy()  # (T+1) x r
        Vsmooth = Vsmooth.cpu().numpy()  # r x r x (T+1)
        VVsmooth = VVsmooth.cpu().numpy()  # r x r x T
        
        # Get filtered state for Vf calculation
        Sf = kalman.filter_forward(Y_torch, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch)
        Vf = Sf.VmU.cpu().numpy()  # r x r x (T+1)
        
        # Smoothed factor covariances for transition matrix
        # Vs is V_{t|T} for t = 1, ..., T (skip initial state)
        Vs = Vsmooth[:, :, 1:].transpose(2, 0, 1)  # T x r x r
        Vf = Vf[:, :, 1:]  # r x r x T (filtered posterior covariance, skip initial)
        
    except ImportError:
        raise ImportError(
            "PyTorch is required for para_const. Install with: pip install torch"
        )
    
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
# Note: These functions are re-exported from nowcast.utils for backward compatibility.
# The canonical implementations are in nowcast.utils (without underscore prefix).

from .utils import (
    get_higher_frequency,
    calculate_backward_date,
    get_forecast_horizon_config,
    check_config_consistency,
    extract_news_summary,
)

# Backward compatibility aliases (with underscore prefix)
_get_higher_frequency = get_higher_frequency
_calculate_backward_date = calculate_backward_date
_get_forecast_horizon_config = get_forecast_horizon_config
_check_config_consistency = check_config_consistency
_extract_news_summary_impl = extract_news_summary

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




# transform_data and _transform_series removed - use DataModule with custom transformers instead


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
        'revision_impact': float(total_impact),  # TODO: Implement revision impact calculation
        'release_impact': 0.0  # TODO: Implement release impact calculation
    }

