"""Estimation helpers for AR coefficients, innovation covariance, and sufficient statistics."""

from typing import Tuple
import numpy as np
from scipy.linalg import inv, pinv
from ._common import NUMERICAL_EXCEPTIONS


def estimate_ar_coefficients_ols(
    z: np.ndarray,
    Z_lag: np.ndarray,
    use_pinv: bool = False,
    pinv_cond: float = 1e-8
) -> Tuple[np.ndarray, bool]:
    """Estimate AR coefficients via ordinary least squares (OLS).
    
    Parameters
    ----------
    z : np.ndarray
        Dependent variable (T x r) - current factor values
    Z_lag : np.ndarray
        Independent variables (T x rp) - lagged factor values
    use_pinv : bool, default False
        If True, use pseudo-inverse. If False, use regular inverse with fallback.
    pinv_cond : float, default 1e-8
        Condition number threshold for pseudo-inverse
        
    Returns
    -------
    ar_coeffs : np.ndarray
        Estimated AR coefficients (r x rp)
    success : bool
        True if estimation succeeded, False if fallback was used
        
    Notes
    -----
    - Returns zero coefficients if OLS fails (MATLAB behavior)
    - Uses pseudo-inverse as fallback for numerical stability
    - Generic pattern used for both factor and idiosyncratic AR estimation
    """
    if z.size == 0 or Z_lag.size == 0 or Z_lag.shape[0] == 0 or Z_lag.shape[1] == 0:
        r = z.shape[1] if z.ndim > 1 else 1
        rp = Z_lag.shape[1] if Z_lag.ndim > 1 else 1
        return np.zeros((r, rp)), False
    
    try:
        if use_pinv:
            # Use rcond for scipy compatibility
            ar_coeffs = pinv(Z_lag, rcond=pinv_cond) @ z
        else:
            ZTZ = Z_lag.T @ Z_lag
            ar_coeffs = inv(ZTZ) @ Z_lag.T @ z
        return ar_coeffs.T if ar_coeffs.ndim > 1 else ar_coeffs.reshape(1, -1), True
    except NUMERICAL_EXCEPTIONS:
        r = z.shape[1] if z.ndim > 1 else 1
        rp = Z_lag.shape[1]
        return np.zeros((r, rp)), False


def compute_innovation_covariance(
    residuals: np.ndarray,
    default_variance: float = 0.1
) -> np.ndarray:
    """Compute innovation covariance from residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Innovation residuals (T x r) or (T,) for single series
    default_variance : float, default 0.1
        Default variance if computation fails
        
    Returns
    -------
    Q : np.ndarray
        Innovation covariance matrix (r x r) or (1 x 1) for single series
        
    Notes
    -----
    - Uses np.cov() for multiple series, np.var() for single series
    - Aligns with MATLAB: Q_i(1:r_i,1:r_i) = cov(e)
    """
    if residuals.size == 0:
        return np.array([[default_variance]])
    
    if residuals.ndim == 1 or residuals.shape[1] == 1:
        variance = np.var(residuals, ddof=0) if residuals.size > 0 else default_variance
        return np.array([[variance]])
    else:
        try:
            Q = np.cov(residuals.T)
            if np.any(~np.isfinite(Q)):
                return np.eye(residuals.shape[1]) * default_variance
            return Q
        except NUMERICAL_EXCEPTIONS:
            return np.eye(residuals.shape[1]) * default_variance


def compute_sufficient_stats(
    Zsmooth: np.ndarray,
    vsmooth: np.ndarray,
    vvsmooth: np.ndarray,
    block_indices: slice,
    T: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected sufficient statistics for EM updates.
    
    Computes E[Z_t Z_t'], E[Z_{t-1} Z_{t-1}'], and E[Z_t Z_{t-1}'] 
    from Kalman smoother output.
    
    Parameters
    ----------
    Zsmooth : np.ndarray
        Smoothed factor estimates (m x (T+1))
    vsmooth : np.ndarray
        Smoothed factor covariances (m x m x (T+1))
    vvsmooth : np.ndarray
        Smoothed lag-1 factor covariances (m x m x T)
    block_indices : slice
        Indices for the current block
    T : int
        Number of time periods
        
    Returns
    -------
    EZZ : np.ndarray
        E[Z_t Z_t' | Y] (r x r)
    EZZ_lag : np.ndarray
        E[Z_{t-1} Z_{t-1}' | Y] (rp x rp)
    EZZ_cross : np.ndarray
        E[Z_t Z_{t-1}' | Y] (r x rp)
    """
    # E[Z_t Z_t' | Y] = sum over t of (Z_t @ Z_t' + V_t)
    Z_block = Zsmooth[block_indices, 1:]
    V_block = vsmooth[block_indices, :, :][:, block_indices, 1:]
    EZZ = Z_block @ Z_block.T + np.sum(V_block, axis=2)
    
    # E[Z_{t-1} Z_{t-1}' | Y] = sum over t of (Z_{t-1} @ Z_{t-1}' + V_{t-1})
    Z_lag_block = Zsmooth[block_indices, :-1]
    V_lag_block = vsmooth[block_indices, :, :][:, block_indices, :-1]
    EZZ_lag = Z_lag_block @ Z_lag_block.T + np.sum(V_lag_block, axis=2)
    
    # E[Z_t Z_{t-1}' | Y] = sum over t of (Z_t @ Z_{t-1}' + VV_t)
    VV_block = vvsmooth[block_indices, :, :][:, block_indices, :]
    EZZ_cross = Z_block @ Z_lag_block.T + np.sum(VV_block, axis=2)
    
    return EZZ, EZZ_lag, EZZ_cross


def safe_mean_std(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation for each column, handling missing values.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (T x N)
        
    Returns
    -------
    means : np.ndarray
        Column means (N,)
    stds : np.ndarray
        Column standard deviations (N,)
        
    Notes
    -----
    - Handles missing values (NaN) by computing statistics only on finite values
    - Returns default values (mean=0, std=1) for columns with no valid data
    - Ensures std > 0 (minimum std = 1.0) to avoid division by zero in standardization
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    >>> means, stds = safe_mean_std(X)
    >>> means  # Column means
    array([3., 4., 6.])
    >>> stds  # Column standard deviations
    array([1.63299316, 1.63299316, 1.41421356])
    """
    n_series = matrix.shape[1]
    means = np.zeros(n_series)
    stds = np.ones(n_series)
    for j in range(n_series):
        col = matrix[:, j]
        mask = np.isfinite(col)
        if np.any(mask):
            means[j] = float(np.nanmean(col[mask]))
            std_val = float(np.nanstd(col[mask]))
            stds[j] = std_val if std_val > 0 else 1.0
        else:
            means[j] = 0.0
            stds[j] = 1.0
    return means, stds

