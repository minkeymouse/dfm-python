"""News decomposition result classes and para_const function.

This module provides the NewsDecompResult dataclass and the para_const function
for Kalman filtering with fixed parameters (used in news decomposition).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from ..core.results import DFMResult
from ..core.state_space import skf, fis


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
