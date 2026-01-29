"""Estimation functions for state-space model parameters.

This module provides functions for estimating VAR dynamics, AR coefficients,
and idiosyncratic component parameters from data. Also includes AR coefficient
clipping utilities for numerical stability.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from scipy.linalg import solve

if TYPE_CHECKING:
    from ..functional.em import EMConfig

from ..logger import get_logger
from ..utils.errors import ConfigurationError, DataValidationError
from ..config.constants import (
    MIN_DIAGONAL_VARIANCE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    MIN_EIGENVALUE,
    MAX_EIGENVALUE,
    MIN_STD,
    VAR_STABILITY_THRESHOLD,
    AR_CLIP_MIN,
    AR_CLIP_MAX,
    MIN_Q_FLOOR,
    DEFAULT_CLEAN_NAN,
    DEFAULT_CLEAN_INF,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_PROCESS_NOISE,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_VARIANCE_FALLBACK,
    DEFAULT_MIN_OBS,
)
from .stability import (
    ensure_process_noise_stable,
    cap_max_eigenval,
    compute_var_safe,
    compute_cov_safe,
    ensure_covariance_stable,
    solve_regularized_ols,
    stabilize_innovation_covariance,
    create_scaled_identity,
    ensure_symmetric,
    cap_smoothed_states,
)
from ..utils.helper import handle_linear_algebra_error
from ..utils.misc import get_config_attr
from ..config.constants import DEFAULT_DTYPE
from .validator import validate_ndarray_ndim, validate_no_nan_inf
from ..utils.errors import DataValidationError, NumericalError

_logger = get_logger(__name__)

# Note: MIN_VARIANCE_COVARIANCE is defined in stability.py to avoid duplication


# ============================================================================
# AR Coefficient Clipping
# ============================================================================

def clip_ar(
    A: np.ndarray,
    min_val: float = AR_CLIP_MIN,
    max_val: float = AR_CLIP_MAX,
    warn: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients to clip
    min_val : float, default AR_CLIP_MIN
        Minimum allowed value
    max_val : float, default AR_CLIP_MAX
        Maximum allowed value
    warn : bool, default True
        Whether to log warnings
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients
    stats : dict
        Statistics about clipping
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else DEFAULT_ZERO_VALUE
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def apply_ar_clipping(
    A: np.ndarray,
    config: Optional["EMConfig"] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients
    config : object, optional
        Configuration object with clipping parameters
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients
    stats : dict
        Statistics about clipping
    """
    if config is None:
        return clip_ar(A, AR_CLIP_MIN, AR_CLIP_MAX, True)
    
    # Check new consolidated ar_clip parameter
    ar_clip = get_config_attr(config, 'ar_clip', None)
    if ar_clip is None:
        # No clipping if ar_clip is None
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    # Extract min/max from dict
    if not isinstance(ar_clip, dict):
        raise ValueError(f"ar_clip must be a dict with 'min' and 'max' keys, got {type(ar_clip)}")
    
    min_val = ar_clip.get('min', AR_CLIP_MIN)
    max_val = ar_clip.get('max', AR_CLIP_MAX)
    
    # Always warn when clipping is enabled (ar_clip is not None)
    return clip_ar(A, min_val, max_val, warn=True)


# Removed estimate_ar - unused function

def estimate_var(factors: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR(1) dynamics for factors.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
    order : int, default 1
        AR order (always 1)
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x m) for AR(1)
    Q : np.ndarray
        Innovation covariance (m x m)
    """
    T, m = factors.shape
    
    if T < 2:
        # Not enough data, use identity
        A = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE)
        return A, Q
    
    # Prepare data for OLS: f_t = A @ f_{t-1}
    Y = factors[1:, :]  # T-1 x m (dependent)
    X = factors[:-1, :]  # T-1 x m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y
    A = solve_regularized_ols(X, Y, regularization=DEFAULT_REGULARIZATION).T
    
    # Ensure stability: clip eigenvalues
    eigenvals = np.linalg.eigvals(A)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= VAR_STABILITY_THRESHOLD:
        A = A * (VAR_STABILITY_THRESHOLD / max_eigenval)
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = compute_cov_safe(residuals.T, rowvar=True, pairwise_complete=False)
    
    # Stabilize Q: symmetrize, ensure positive definite, apply floor
    Q = stabilize_innovation_covariance(Q, min_eigenval=MIN_EIGENVALUE, min_floor=MIN_Q_FLOOR, dtype=np.float32)
    
    return A, Q


def get_idio(eps: np.ndarray, idx_no_missings: np.ndarray, min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get AR(1) statistics from eps (matching original TensorFlow get_idio).
    
    Returns Phi (AR(1) coefficient matrix), mu_eps (mean), and std_eps (std dev).
    
    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals (T x N)
    idx_no_missings : np.ndarray
        Boolean mask (T x N) indicating non-missing values
    min_obs : int, default 5
        Minimum number of observations required
        
    Returns
    -------
    Phi : np.ndarray
        AR(1) coefficient matrix (N x N), diagonal
    mu_eps : np.ndarray
        Mean of idiosyncratic components (N,)
    std_eps : np.ndarray
        Standard deviation of idiosyncratic components (N,)
    """
    Phi = np.zeros((eps.shape[1], eps.shape[1]))
    mu_eps = np.zeros(eps.shape[1])
    std_eps = np.zeros(eps.shape[1])
    for j in range(eps.shape[1]):
        to_select = idx_no_missings[:, j]
        to_select = np.hstack((np.array([False]), to_select[:-1] & to_select[1:]))
        if np.sum(to_select) >= min_obs:
            this_eps = eps[to_select, j]
        else:
            # Insufficient observations: use zero AR(1) coefficient
            Phi[j, j] = 0.0
            mu_eps[j] = 0.0
            std_eps[j] = 1.0
            continue
        mu_eps[j] = np.mean(this_eps)
        std_eps[j] = np.std(this_eps)
        cov1_eps = np.cov(this_eps[1:], this_eps[:-1])[0][1]
        Phi[j, j] = cov1_eps / (std_eps[j] ** 2)
    return Phi, mu_eps, std_eps


def get_transition_params(f_t: np.ndarray, eps_t: np.ndarray, bool_no_miss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate transition parameters (matching original TensorFlow get_transition_params).
    
    Factor order is fixed to 1 (VAR(1) only).
    
    Parameters
    ----------
    f_t : np.ndarray
        Common factors (T x m)
    eps_t : np.ndarray
        Idiosyncratic residuals (T x N)
    bool_no_miss : np.ndarray
        Boolean mask (T x N) indicating non-missing values
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (combines VAR(1) for factors and AR(1) for idiosyncratic)
    Q : np.ndarray
        Process noise covariance (W in original)
    mu_0 : np.ndarray
        Initial state mean
    Sigma_0 : np.ndarray
        Initial state covariance
    x_t : np.ndarray
        Latent states [f_t, eps_t]
    """
    # Factor order is fixed to 1 (VAR(1) only)
    f_past = f_t[:-1, :]
    # Robust VAR(1) estimation for factors.
    #
    # In practice, upstream training can occasionally produce NaNs/Infs in factors.
    # Those propagate into (f_past.T @ f_past) and cause SVD non-convergence inside pinv().
    # To keep the pipeline running (so we can still forecast/evaluate), we:
    # - replace non-finite values with 0
    # - add a small ridge term to stabilize the inversion
    # - fall back to least squares if pinv still fails
    # - validate results to ensure numerical stability
    f_past_safe = np.nan_to_num(f_past, nan=0.0, posinf=0.0, neginf=0.0)
    f_next_safe = np.nan_to_num(f_t[1:, :], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for constant factors (zero variance) which can cause issues
    f_past_std = np.std(f_past_safe, axis=0)
    if np.any(f_past_std < 1e-10):
        from ..logger import get_logger
        _logger = get_logger(__name__)
        _logger.warning(
            f"get_transition_params: Some factors have near-zero variance (min_std={f_past_std.min():.2e}). "
            "This may cause numerical instability in transition matrix estimation."
        )
    
    XtX = f_past_safe.T @ f_past_safe
    Xty = f_past_safe.T @ f_next_safe
    
    # Adaptive ridge regularization based on matrix condition
    # Use larger ridge for ill-conditioned matrices
    try:
        cond_num = np.linalg.cond(XtX)
        # Use larger ridge if condition number is very high
        ridge_scale = max(1e-6, min(1e-4, 1e-6 * (1 + np.log10(max(cond_num, 1)))))
    except (np.linalg.LinAlgError, ValueError):
        # If condition number computation fails, use default
        ridge_scale = 1e-6
    
    ridge = ridge_scale * np.eye(XtX.shape[0], dtype=XtX.dtype)
    
    try:
        A_f = (np.linalg.pinv(XtX + ridge) @ Xty).T
        # Validate result
        if not np.all(np.isfinite(A_f)):
            raise ValueError("pinv produced non-finite values")
    except (np.linalg.LinAlgError, ValueError) as e:
        # Solve (XtX + ridge) A^T = Xty via least squares as a fallback
        from ..logger import get_logger
        _logger = get_logger(__name__)
        _logger.warning(
            f"get_transition_params: pinv failed ({type(e).__name__}), using lstsq fallback. "
            "This may indicate numerical instability in factor estimation."
        )
        try:
            A_t, residuals, rank, s = np.linalg.lstsq(XtX + ridge, Xty, rcond=None)
            A_f = A_t.T
            # Check if solution is valid
            if rank < XtX.shape[0]:
                _logger.warning(
                    f"get_transition_params: lstsq solution has rank {rank} < {XtX.shape[0]}. "
                    "Matrix may be rank-deficient."
                )
        except np.linalg.LinAlgError as e2:
            # If both methods fail, use identity as last resort
            _logger.error(
                f"get_transition_params: Both pinv and lstsq failed. Using identity matrix as fallback. "
                f"Errors: {type(e).__name__}, {type(e2).__name__}"
            )
            A_f = np.eye(f_past_safe.shape[1], dtype=f_past_safe.dtype)
    
    Phi, _, _ = get_idio(eps_t, bool_no_miss)
    
    x_t = np.vstack((f_t.T, eps_t.T))
    A = np.vstack((
        np.hstack((A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))),  # VAR factors
        np.hstack((np.zeros((eps_t.shape[1], A_f.shape[1])), Phi))  # AR(1) idio
    ))
    
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    Q = np.diag(np.diag(np.cov(w_t)))
    mu_0 = np.mean(x_t, axis=1)
    Sigma_0 = np.cov(x_t)
    Sigma_0[:A_f.shape[1], A_f.shape[1]:] = 0
    Sigma_0[A_f.shape[1]:, :A_f.shape[1]] = 0
    Sigma_0[A_f.shape[1]:, A_f.shape[1]:] = np.diag(np.diag(Sigma_0[A_f.shape[1]:, A_f.shape[1]:]))
    return A, Q, mu_0, Sigma_0, x_t


def forecast_ar1_factors(
    Z_last: np.ndarray,
    A: np.ndarray,
    horizon: int,
    dtype: type = DEFAULT_DTYPE
) -> np.ndarray:
    """Forecast factors using AR(1) dynamics.
    
    Uses iterative matrix multiplication for efficient computation.
    
    **AR(1) Dynamics**: f_t = A @ f_{t-1}
    
    This function is used by DFM and DDFM for factor forecasting, ensuring
    consistent AR dynamics computation across models.
    
    Parameters
    ----------
    Z_last : np.ndarray
        Last factor state of shape (m,) where m is number of factors.
        Must be 1D array with m >= 1.
    A : np.ndarray
        Transition matrix of shape (m, m) for AR(1).
        Must be 2D array with compatible dimensions.
    horizon : int
        Number of periods to forecast. Must be >= 1.
    dtype : type, default=DEFAULT_DTYPE
        Data type for output array
        
    Returns
    -------
    np.ndarray
        Forecasted factors of shape (horizon, m) where:
        - horizon: Number of forecast periods
        - m: Number of factors (matches Z_last.shape[0])
        
    Raises
    ------
    DataValidationError
        If input shapes are incompatible or invalid
    NumericalError
        If matrix operations produce NaN/Inf values
        If forecast computation produces NaN/Inf values
    """
    # Validate inputs
    validate_ndarray_ndim(Z_last, "Z_last", 1)
    validate_ndarray_ndim(A, "A", 2)
    
    m = Z_last.shape[0]
    if m < 1:
        raise DataValidationError(f"Z_last must have at least 1 factor, got m={m}")
    
    if horizon < 1:
        raise DataValidationError(f"horizon must be >= 1, got {horizon}")
    
    validate_no_nan_inf(Z_last, name="Z_last")
    validate_no_nan_inf(A, name="transition matrix A")
    
    # Validate A shape
    if A.shape != (m, m):
        raise DataValidationError(
            f"A (transition matrix) must have shape ({m}, {m}), got {A.shape}"
        )
    
    try:
        # AR(1): f_t = A @ f_{t-1}
        Z_forecast = np.zeros((horizon, m), dtype=dtype)
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Validate output
        validate_no_nan_inf(Z_forecast, name="forecasted factors")
        
        if Z_forecast.shape != (horizon, m):
            raise DataValidationError(
                f"Forecast output must have shape ({horizon}, {m}), got {Z_forecast.shape}"
            )
        
        return Z_forecast
        
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
        # Re-raise configuration/validation errors as-is
        if isinstance(e, (ConfigurationError, DataValidationError)):
            raise
        # Wrap other errors
        raise NumericalError(
            f"Forecast computation failed: {e}",
            details=(
                f"Z_last shape: {Z_last.shape}, A shape: {A.shape}, "
                f"horizon={horizon}. Check: (1) Matrix dimensions, "
                f"(2) Numerical stability, (3) Input validity."
            )
        ) from e


def estimate_idio_dynamics(
    residuals: np.ndarray,
    missing_mask: np.ndarray,
    min_obs: int = DEFAULT_MIN_OBS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR(1) dynamics for idiosyncratic components.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from observation equation (T x N)
    missing_mask : np.ndarray
        Missing data mask (T x N), True where data is missing
    min_obs : int, default DEFAULT_MIN_OBS
        Minimum number of observations required for estimation
        
    Returns
    -------
    A_eps : np.ndarray
        AR(1) coefficients (N x N), diagonal matrix
    Q_eps : np.ndarray
        Innovation covariance (N x N), diagonal matrix
    """
    T, N = residuals.shape
    A_eps = np.zeros((N, N))
    Q_eps = np.zeros((N, N))
    
    for j in range(N):
        # Match original get_idio selection logic:
        # Original: to_select = np.hstack((np.array([False]), to_select[:-1] * to_select[1:]))
        # This selects eps[t] where both t-1 and t are non-missing (for t>=1)
        # Then uses np.cov(this_eps[1:], this_eps[:-1]) to get AR(1) coefficient
        valid = ~missing_mask[:, j]  # True where non-missing
        # Select indices where both t-1 and t are non-missing (for t>=1)
        # First element is False (can't use t=0 without t-1)
        to_select = np.hstack((np.array([False]), valid[:-1] & valid[1:]))
        
        if np.sum(to_select) < min_obs:
            # Insufficient data: use zero AR(1) coefficient
            _logger.warning(
                f"Insufficient observations ({np.sum(to_select)}) for idio AR(1) "
                f"estimation for series {j}. Using zero AR(1) coefficient."
            )
            A_eps[j, j] = DEFAULT_ZERO_VALUE
            # Use variance of available residuals
            if np.sum(valid) > 0:
                Q_eps[j, j] = compute_var_safe(residuals[valid, j], ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)
            else:
                Q_eps[j, j] = MIN_DIAGONAL_VARIANCE
        else:
            # Extract selected eps values (matches original: this_eps = eps[to_select, j])
            this_eps = residuals[to_select, j]
            
            # Compute AR(1) coefficient using original method:
            # cov1_eps = np.cov(this_eps[1:], this_eps[:-1])[0][1]
            # phi = cov1_eps / (std_eps[j] ** 2)
            # This computes covariance between consecutive pairs in selected sequence
            if len(this_eps) >= 2:
                cov_matrix = compute_cov_safe(np.vstack([this_eps[1:], this_eps[:-1]]), rowvar=True, pairwise_complete=False)
                cov_eps = cov_matrix[0, 1]
                var_eps = compute_var_safe(this_eps[:-1], ddof=0, min_variance=MIN_FACTOR_VARIANCE)
                if var_eps > MIN_FACTOR_VARIANCE:
                    A_eps[j, j] = cov_eps / var_eps
                else:
                    A_eps[j, j] = DEFAULT_ZERO_VALUE
            else:
                A_eps[j, j] = DEFAULT_ZERO_VALUE
            
            # Ensure stability: clip AR(1) coefficient
            if abs(A_eps[j, j]) >= VAR_STABILITY_THRESHOLD:
                sign = np.sign(A_eps[j, j])
                A_eps[j, j] = sign * VAR_STABILITY_THRESHOLD
                _logger.debug(
                    f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                )
            
            # Estimate innovation covariance from AR(1) residuals
            # Use the same selection as for AR coefficient
            if len(this_eps) >= 2:
                eps_t = this_eps[1:]
                eps_t_1 = this_eps[:-1]
                residuals_ar = eps_t - A_eps[j, j] * eps_t_1
                Q_eps[j, j] = compute_var_safe(residuals_ar, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)
            else:
                # Fallback: use variance of selected eps
                Q_eps[j, j] = compute_var_safe(this_eps, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)
    
    return A_eps, Q_eps


# Removed estimate_idio_params - unused (estimate_idio_dynamics is used instead)
# Removed estimate_state_space_params - unused function

# ============================================================================
# Unified Estimation Functions (work with raw data or smoothed expectations)
# ============================================================================

def estimate_var(
    y: np.ndarray,
    x: np.ndarray,
    V_smooth: Optional[np.ndarray] = None,
    V_smooth_lag: Optional[np.ndarray] = None,
    VVsmooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    min_variance: float = MIN_EIGENVALUE,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """VAR estimation that works with raw data or smoothed expectations.
    
    Parameters
    ----------
    y : np.ndarray
        Current state (T x m) or (T-1 x m) for raw data, or smoothed expectations E[z_t]
    x : np.ndarray
        Lagged state (T-1 x p) for raw data, or smoothed expectations E[z_{t-1}]
    V_smooth : np.ndarray, optional
        Smoothed state covariances for current state (T x m x m) or (T-1 x m x m). 
        Required for smoothed expectations mode.
    V_smooth_lag : np.ndarray, optional
        Smoothed state covariances for lagged state (T-1 x p x p). 
        If None and V_smooth is provided, will attempt to derive from V_smooth.
        Required when lagged state dimension (p) differs from current state dimension (m).
    VVsmooth : np.ndarray, optional
        Lag-1 cross-covariances (T x m x p). Required for smoothed expectations.
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization parameter for OLS
    min_variance : float, default MIN_EIGENVALUE
        Minimum variance floor
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x p)
    Q : np.ndarray
        Process noise covariance (m x m)
    """
    if V_smooth is not None:
        # Smoothed expectations mode
        # E[z_t z_t'] = EZ @ EZ' + V_smooth
        # E[z_{t-1} z_{t-1}'] = EZ_lag @ EZ_lag' + V_smooth_lag
        # E[z_t z_{t-1}'] = EZ @ EZ_lag' + VVsmooth
        
        T = y.shape[0] if y is not None and len(y.shape) > 0 else 1
        m = y.shape[1]
        p = x.shape[1]
        
        # CRITICAL FIX: Cap V_smooth eigenvalues before computing EZZ to prevent explosion
        # This addresses root cause of Q explosions: V_smooth explosion → EZZ explosion → Q explosion
        # 
        if V_smooth.ndim == 3:
            # PERFORMANCE FIX: Validate shapes BEFORE the loop to avoid exceptions in hot path
            # V_smooth should be (T, m, m) where each V_smooth[t] is square
            T_v, m_v, n_v = V_smooth.shape
            if m_v != n_v:
                raise ValueError(
                    f"V_smooth must contain square matrices, got shape={V_smooth.shape}. "
                    f"This indicates a bug in state structure or slicing."
                )
            
            # CRITICAL FIX: Sum in float64 first to prevent overflow, then cap, then cast
            # The sum of 2538 large covariance matrices can exceed float32 max (~3.4e38)
            # Sum in higher precision, cap eigenvalues, then cast to target dtype
            V_smooth_sum = np.sum(V_smooth.astype(np.float64), axis=0)
            # CRITICAL: Cap the sum after accumulation to prevent explosion
            V_smooth_sum = cap_max_eigenval(
                V_smooth_sum,
                max_eigenval=MAX_EIGENVALUE,
                symmetric=True,
                warn=False
            ).astype(dtype)
        elif V_smooth.ndim == 2:
            V_smooth_sum = cap_max_eigenval(
                V_smooth, 
                max_eigenval=MAX_EIGENVALUE, 
                symmetric=True, 
                warn=False
            ).astype(dtype)
        else:
            V_smooth_sum = V_smooth.astype(dtype) if V_smooth is not None else None
        
        # Compute expectations
        # De-duplicate: use shared helper for state capping (also used in EM).
        y = cap_smoothed_states(y, max_eigenval=MAX_EIGENVALUE, warn=True)
        
        EZZ = y.T @ y
        if V_smooth_sum is not None:
            EZZ = EZZ + V_smooth_sum
        
        EZZ_BB = x.T @ x
        # Use V_smooth_lag if provided, otherwise try to derive from V_smooth
        if V_smooth_lag is not None:
            # Use explicitly provided lagged state covariance
            if V_smooth_lag.ndim == 3:
                # PERFORMANCE FIX: Validate shapes BEFORE processing
                T_lag, m_lag, n_lag = V_smooth_lag.shape
                if m_lag != n_lag:
                    raise ValueError(
                        f"V_smooth_lag must contain square matrices, got shape={V_smooth_lag.shape}. "
                        f"This indicates a bug in state structure or slicing."
                    )
                
                # CRITICAL FIX: Sum in float64 first to prevent overflow, then cap, then cast
                V_smooth_lag_sum = np.sum(V_smooth_lag.astype(np.float64), axis=0)
                V_smooth_lag_sum = cap_max_eigenval(
                    V_smooth_lag_sum,
                    max_eigenval=MAX_EIGENVALUE,
                    symmetric=True,
                    warn=False
                ).astype(dtype)
            elif V_smooth_lag.ndim == 2:
                V_smooth_lag_sum = cap_max_eigenval(
                    V_smooth_lag,
                    max_eigenval=MAX_EIGENVALUE,
                    symmetric=True,
                    warn=False
                ).astype(dtype)
            else:
                V_smooth_lag_sum = V_smooth_lag.astype(dtype)
            EZZ_BB = EZZ_BB + V_smooth_lag_sum
        elif V_smooth is not None:
            # Fallback: derive from V_smooth (only works if m == p)
            if V_smooth.ndim == 3:
                # Use original V_smooth, not V_smooth_capped (which no longer exists after optimization)
                V_smooth_lag_derived = V_smooth[:-1] if V_smooth.shape[0] == T + 1 else V_smooth
                # CRITICAL FIX: Sum in float64 first to prevent overflow, then cap, then cast
                V_smooth_lag_sum = np.sum(V_smooth_lag_derived.astype(np.float64), axis=0)
                # Validate shape matches x dimension
                if V_smooth_lag_sum.shape[0] != p or V_smooth_lag_sum.shape[1] != p:
                    _logger.warning(
                        f"Shape mismatch: V_smooth_lag derived from V_smooth has shape {V_smooth_lag_sum.shape}, "
                        f"but x has {p} columns. This may cause errors. Provide V_smooth_lag explicitly."
                    )
                # Cap the lag sum as well
                V_smooth_lag_sum = cap_max_eigenval(
                    V_smooth_lag_sum,
                    max_eigenval=MAX_EIGENVALUE,
                    symmetric=True,
                    warn=False
                ).astype(dtype)
                EZZ_BB = EZZ_BB + V_smooth_lag_sum
            elif V_smooth.ndim == 2:
                if V_smooth_sum.shape[0] == p and V_smooth_sum.shape[1] == p:
                    EZZ_BB = EZZ_BB + V_smooth_sum
                else:
                    _logger.warning(
                        f"Shape mismatch: V_smooth has shape {V_smooth_sum.shape}, but x has {p} columns. "
                        f"Provide V_smooth_lag explicitly."
                    )
        
        # De-duplicate: reuse shared helper for lagged state capping.
        x = cap_smoothed_states(x, max_eigenval=MAX_EIGENVALUE, warn=True)
        
        EZZ_FB = y[1:].T @ x if y.shape[0] > x.shape[0] else y.T @ x
        # CRITICAL FIX: Cap VVsmooth eigenvalues as well to prevent explosion
        if VVsmooth is not None:
            if VVsmooth.ndim == 3:
                # PERFORMANCE FIX: Validate shapes BEFORE processing
                T_vv, m_vv, n_vv = VVsmooth.shape
                # NOTE: VVsmooth can be rectangular (m_vv × n_vv) when used in block updates
                # For VAR(p) estimation, VVsmooth_block has shape (T-1, r_i, rp) where:
                # - r_i = number of factors in current state
                # - rp = r_i * (p+1) = number of factors in all state (current + lags)
                # This is correct: it's cross-covariance between current state and lagged state
                # Only validate that dimensions are positive, not that they're square
                if m_vv <= 0 or n_vv <= 0:
                    raise ValueError(
                        f"VVsmooth has invalid dimensions: shape={VVsmooth.shape}. "
                        f"Both m_vv={m_vv} and n_vv={n_vv} must be positive."
                    )
                
                # CRITICAL FIX: Sum in float64 first to prevent overflow, then cap/clip, then cast
                # VVsmooth is cross-covariance, so we sum over time axis
                # NOTE: VVsmooth can be rectangular (m_vv × n_vv) for block updates
                # For VAR(p): VVsmooth_block is (T-1, r_i, rp) where r_i < rp
                # This is cross-covariance between current state (r_i) and lagged state (rp)
                VVsmooth_to_sum = VVsmooth[1:] if VVsmooth.shape[0] == T + 1 else VVsmooth
                VVsmooth_sum = np.sum(VVsmooth_to_sum.astype(np.float64), axis=0)
                # CRITICAL: Cap the sum after accumulation
                # For rectangular matrices, only cap if square (symmetric case)
                # For rectangular cross-covariance, just add directly (no eigenvalue capping)
                if m_vv == n_vv:
                    # Square case: symmetric cross-covariance, can cap eigenvalues
                    VVsmooth_sum = cap_max_eigenval(
                        VVsmooth_sum,
                        max_eigenval=MAX_EIGENVALUE,
                        symmetric=True,
                        warn=False
                    ).astype(dtype)
                else:
                    # Rectangular case: cross-covariance between different state dimensions
                    # Clip values to prevent numerical issues, but don't cap eigenvalues
                    VVsmooth_sum = np.clip(VVsmooth_sum, -MAX_EIGENVALUE, MAX_EIGENVALUE).astype(dtype)
                EZZ_FB = EZZ_FB + VVsmooth_sum
            elif VVsmooth.ndim == 2:
                VVsmooth_capped = cap_max_eigenval(
                    VVsmooth, 
                    max_eigenval=MAX_EIGENVALUE, 
                    symmetric=True, 
                    warn=False
                ).astype(dtype)
                EZZ_FB = EZZ_FB + VVsmooth_capped
        
        # Check for extreme values in EZZ and EZZ_FB before Q computation (prevent overflow)
        # EZZ = y.T @ y + V_smooth_sum can be huge if y has extreme values
        max_EZZ = np.max(np.abs(EZZ)) if EZZ.size > 0 else 0.0
        max_EZZ_FB = np.max(np.abs(EZZ_FB)) if EZZ_FB.size > 0 else 0.0
        safe_max = MAX_EIGENVALUE * T  # Safe maximum for Q computation
        
        if max_EZZ > safe_max or max_EZZ_FB > safe_max:
            _logger.warning(
                f"EZZ or EZZ_FB has extreme values (max_EZZ={max_EZZ:.2e}, max_EZZ_FB={max_EZZ_FB:.2e}, "
                f"safe_max={safe_max:.2e}). This will cause Q to overflow. "
                f"Scaling moments to safe range to prevent overflow..."
            )
            # IMPORTANT:
            # - EZZ is (m x m)
            # - EZZ_FB is (m x p) and MUST remain rectangular when p != m (companion form)
            #
            # FRBNY-style safe fallback: shrink (scale) the moments rather than reshaping them.
            # This preserves the cross-cov structure and avoids shape-mismatch errors (e.g., 5 vs 15).

            # Stabilize EZZ: keep it PSD-ish by using a scaled identity at the correct (m x m) shape.
            EZZ = create_scaled_identity(EZZ.shape[0], MAX_EIGENVALUE, dtype=dtype) * T

            # Stabilize EZZ_FB: keep its (m x p) shape by scaling existing values down to safe_max.
            EZZ_FB = np.where(np.isfinite(EZZ_FB), EZZ_FB, 0.0)
            max_abs_fb = np.max(np.abs(EZZ_FB)) if EZZ_FB.size > 0 else 0.0
            if max_abs_fb > safe_max and max_abs_fb > 0:
                scale = safe_max / max_abs_fb
                EZZ_FB = (EZZ_FB * scale).astype(dtype, copy=False)
            else:
                EZZ_FB = EZZ_FB.astype(dtype, copy=False)

            _logger.warning(
                f"Stabilized moments for Q computation: EZZ set to scaled identity, "
                f"EZZ_FB scaled to preserve shape={EZZ_FB.shape} (safe_max={safe_max:.2e})."
            )
        
        # Regularize
        EZZ_BB_reg = EZZ_BB + create_scaled_identity(p, regularization, dtype=dtype)
        
        def _compute_A_Q():
            # OLS: A = (EZZ_BB)^(-1) @ EZZ_FB'
            # Note: EZZ_BB_reg is already regularized, so use use_XTX=False
            # EZZ_BB_reg is (p x p), EZZ_FB.T is (p x m)
            # solve_regularized_ols with use_XTX=False returns (p x m), so we transpose to get (m x p)
            A = solve_regularized_ols(EZZ_BB_reg, EZZ_FB.T, regularization=DEFAULT_ZERO_VALUE, use_XTX=False, dtype=dtype).T  # (m x p)
            
            
            # Q = (EZZ - A @ EZZ_FB') / T
            # CRITICAL: Check EZZ and EZZ_FB for non-finite values before computation
            if np.any(~np.isfinite(EZZ)) or np.any(~np.isfinite(EZZ_FB)) or np.any(~np.isfinite(A)):
                _logger.error(
                    f"EZZ, EZZ_FB, or A contains non-finite values before Q computation. "
                    f"EZZ: Inf={np.sum(np.isinf(EZZ))}, NaN={np.sum(np.isnan(EZZ))}. "
                    f"EZZ_FB: Inf={np.sum(np.isinf(EZZ_FB))}, NaN={np.sum(np.isnan(EZZ_FB))}. "
                    f"A: Inf={np.sum(np.isinf(A))}, NaN={np.sum(np.isnan(A))}. "
                    f"Using fallback Q."
                )
                Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
            else:
                # Check for extreme values that will cause overflow
                max_EZZ_val = np.max(np.abs(EZZ))
                max_EZZ_FB_val = np.max(np.abs(EZZ_FB))
                max_A_val = np.max(np.abs(A))
                safe_max = MAX_EIGENVALUE * T  # Safe maximum for Q computation
                
                if max_EZZ_val > safe_max or max_EZZ_FB_val > safe_max or max_A_val > 10:
                    _logger.error(
                        f"Extreme values detected before Q computation: "
                        f"max_EZZ={max_EZZ_val:.2e}, max_EZZ_FB={max_EZZ_FB_val:.2e}, max_A={max_A_val:.2e}. "
                        f"Safe max={safe_max:.2e}. Using fallback Q."
                    )
                    Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
                else:
                    try:
                        # Compute A @ EZZ_FB.T first to check for overflow
                        A_EZZ_FB_T = A @ EZZ_FB.T
                        if np.any(~np.isfinite(A_EZZ_FB_T)):
                            _logger.error("A @ EZZ_FB.T produced non-finite values. Using fallback Q.")
                            Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
                        else:
                            Q = (EZZ - A_EZZ_FB_T) / T
                    except (OverflowError, FloatingPointError) as e:
                        _logger.error(
                            f"Overflow in Q computation: {e}. Using fallback Q."
                        )
                        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
                    else:
                        # Check for non-finite values after computation (overflow recovery)
                        if np.any(~np.isfinite(Q)):
                            _logger.warning(
                                f"Q computation produced non-finite values (Inf: {np.sum(np.isinf(Q))}, NaN: {np.sum(np.isnan(Q))}). "
                                f"This indicates overflow. Using fallback Q."
                            )
                            Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
            
            Q = ensure_process_noise_stable(Q, min_eigenval=min_variance, warn=True, dtype=dtype)
            
            # Final check: ensure Q is finite and bounded before return (prevent overflow in downstream code)
            if np.any(~np.isfinite(Q)) or np.max(np.abs(Q)) > MAX_EIGENVALUE * 10:
                _logger.warning(
                    f"Q still has extreme values after stabilization (max={np.max(np.abs(Q)):.2e}). "
                    f"Applying additional clipping..."
                )
                Q = np.clip(Q, -MAX_EIGENVALUE, MAX_EIGENVALUE)
                Q = np.where(np.isfinite(Q), Q, DEFAULT_PROCESS_NOISE)
                Q = ensure_symmetric(Q)
                # Re-stabilize after clipping
                Q = ensure_process_noise_stable(Q, min_eigenval=min_variance, warn=False, dtype=dtype)
            
            return A, Q
        
        def _fallback_A_Q():
            # Fallback: use identity
            A = create_scaled_identity(m, DEFAULT_TRANSITION_COEF, dtype)
            if p > m:
                # Pad A to match p columns
                A = np.hstack([A, np.zeros((m, p - m), dtype=dtype)])
            Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype)
            return A, Q
        
        A, Q = handle_linear_algebra_error(
            _compute_A_Q, "VAR estimation",
            fallback_func=_fallback_A_Q
        )
    else:
        # Raw data mode

        T = y.shape[0]
        m = y.shape[1]
        p = x.shape[1]
        
        if T < 2:
            # For rectangular matrix (m x p), use np.eye directly (create_scaled_identity only supports square)
            # Note: A is rectangular (m x p), so cannot use create_scaled_identity
            A = np.eye(m, p, dtype=dtype) * DEFAULT_TRANSITION_COEF
            Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
            return A, Q
        
        # OLS: A = (X'X + reg*I)^(-1) X'Y
        A = solve_regularized_ols(x, y, regularization=regularization, dtype=dtype).T  # (m x p)
        
        # Estimate Q from residuals
        residuals = y - x @ A.T
        if m == 1:
            var_val = compute_var_safe(residuals.flatten(), ddof=0, min_variance=min_variance)
            Q = np.atleast_2d(var_val)
        else:
            # Use pairwise_complete=True during initialization to handle NaNs better
            # This allows covariance computation even when there aren't enough complete rows
            Q = compute_cov_safe(residuals.T, rowvar=True, pairwise_complete=True, min_eigenval=min_variance)
        
        Q = stabilize_innovation_covariance(Q, min_eigenval=min_variance, min_floor=MIN_Q_FLOOR, dtype=dtype)
    
    return A.astype(dtype), Q.astype(dtype)


def estimate_ar1(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    V_smooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    min_variance: float = MIN_EIGENVALUE,
    default_ar_coef: float = DEFAULT_TRANSITION_COEF,
    default_noise: float = DEFAULT_PROCESS_NOISE,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """AR(1) estimation that works with raw data or smoothed expectations.
    
    Parameters
    ----------
    y : np.ndarray
        Current state (T x n) or (T-1 x n) for raw data, or smoothed expectations E[z_t]
    x : np.ndarray, optional
        Lagged state (T-1 x n) for raw data, or smoothed expectations E[z_{t-1}].
        If None, uses y[:-1] for raw data mode.
    V_smooth : np.ndarray, optional
        Smoothed state covariances. Required for smoothed expectations mode.
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization parameter
    min_variance : float, default MIN_EIGENVALUE
        Minimum variance floor
    default_ar_coef : float, default DEFAULT_TRANSITION_COEF
        Default AR coefficient if estimation fails
    default_noise : float, default DEFAULT_PROCESS_NOISE
        Default noise variance if estimation fails
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    A_diag : np.ndarray
        AR(1) coefficients (n,) - diagonal
    Q_diag : np.ndarray
        Innovation variances (n,) - diagonal
    """
    if V_smooth is not None:
        # Smoothed expectations mode
        T = y.shape[0]
        n = y.shape[1]
        
        if x is None:
            x = y[:-1]
            y = y[1:]
            T = T - 1
        
        # Compute diagonal expectations
        EZZ = np.diag(y.T @ y)
        if V_smooth.ndim == 3:
            EZZ = EZZ + np.diag(np.sum(V_smooth[1:], axis=0))
        elif V_smooth.ndim == 2:
            EZZ = EZZ + np.diag(V_smooth)
        
        EZZ_BB = np.diag(x.T @ x)
        if V_smooth.ndim == 3:
            V_smooth_lag = V_smooth[:-1] if V_smooth.shape[0] == T + 1 else V_smooth
            EZZ_BB = EZZ_BB + np.diag(np.sum(V_smooth_lag, axis=0))
        elif V_smooth.ndim == 2:
            EZZ_BB = EZZ_BB + np.diag(V_smooth)
        
        EZZ_FB = np.diag(y.T @ x)
        # Note: VVsmooth handling would go here if needed
        
        # Regularize
        EZZ_BB_reg = EZZ_BB + regularization
        
        # AR(1) coefficients: A = EZZ_FB / EZZ_BB
        A_diag = EZZ_FB / np.maximum(EZZ_BB_reg, min_variance)
        A_diag = np.where(np.isfinite(A_diag), A_diag, default_ar_coef)
        
        # Q = (EZZ - A * EZZ_FB) / T
        Q_diag = (EZZ - A_diag * EZZ_FB) / T
        Q_diag = np.maximum(Q_diag, min_variance)
        Q_diag = np.where(np.isfinite(Q_diag), Q_diag, default_noise)
    else:
        # Raw data mode
        if x is None:
            x = y[:-1]
            y = y[1:]
        
        T, n = y.shape
        
        if T < 2:
            A_diag = np.full(n, default_ar_coef, dtype=dtype)
            Q_diag = np.full(n, default_noise, dtype=dtype)
            return A_diag, Q_diag
        
        A_diag = np.zeros(n, dtype=dtype)
        Q_diag = np.zeros(n, dtype=dtype)
        
        for i in range(n):
            y_i = y[:, i]
            x_i = x[:, i].reshape(-1, 1)
            
            # Skip if insufficient data
            valid = np.isfinite(y_i) & np.isfinite(x_i.squeeze())
            if np.sum(valid) < 2:
                A_diag[i] = default_ar_coef
                Q_diag[i] = default_noise
                continue
            
            y_i_clean = y_i[valid]
            x_i_clean = x_i[valid]
            
            try:
                # OLS: A = (x'x + reg)^(-1) x'y
                XTX = x_i_clean.T @ x_i_clean
                XTX_reg = XTX + create_scaled_identity(1, regularization, dtype=dtype)
                A_i = solve(XTX_reg, x_i_clean.T @ y_i_clean, overwrite_a=False, overwrite_b=False, check_finite=False).item()
                
                # Estimate Q from residuals
                residuals = y_i_clean - x_i_clean.squeeze() * A_i
                Q_i = compute_var_safe(residuals, ddof=0, min_variance=min_variance, default_variance=default_noise) if len(residuals) > 1 else default_noise
                
                A_diag[i] = A_i if np.isfinite(A_i) else default_ar_coef
                Q_diag[i] = max(Q_i, min_variance) if np.isfinite(Q_i) else default_noise
            except (np.linalg.LinAlgError, ValueError):
                A_diag[i] = default_ar_coef
                Q_diag[i] = default_noise
        
        # Ensure all values are finite and bounded before cast (prevent overflow)
        A_diag = np.where(np.isfinite(A_diag), A_diag, default_ar_coef)
        Q_diag = np.where(np.isfinite(Q_diag), Q_diag, default_noise)
        # Cap extreme values to prevent overflow in cast
        A_diag = np.clip(A_diag, -VAR_STABILITY_THRESHOLD, VAR_STABILITY_THRESHOLD)
        Q_diag = np.clip(Q_diag, min_variance, MAX_EIGENVALUE)
    
    return A_diag, Q_diag


def estimate_arp_ols(
    factors: np.ndarray,
    order: int,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR(p) coefficients using OLS for each factor independently.
    
    For each factor i, estimates: f_i[t] = a_i[1]*f_i[t-1] + ... + a_i[p]*f_i[t-p] + e_i[t]
    Returns diagonal AR coefficients (one AR(p) per factor).
    
    Parameters
    ----------
    factors : np.ndarray
        Factor time series (T x r)
    order : int
        AR order (p)
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization parameter for OLS
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    ar_coeffs : np.ndarray
        AR coefficients (r x p) - one AR(p) per factor
    B_diag : np.ndarray
        Innovation standard deviations (r,) - diagonal elements of B
    """
    T, r = factors.shape
    
    if T < order + 1:
        # Not enough data, return zeros
        ar_coeffs = np.zeros((r, order), dtype=dtype)
        B_diag = np.ones(r, dtype=dtype) * DEFAULT_PROCESS_NOISE
        return ar_coeffs, B_diag
    
    ar_coeffs = np.zeros((r, order), dtype=dtype)
    B_diag = np.zeros(r, dtype=dtype)
    
    for i in range(r):
        f_i = factors[:, i]  # (T,)
        
        # Build design matrix: X[t] = [f_i[t-1], f_i[t-2], ..., f_i[t-p]]
        # Target: y[t] = f_i[t]
        X = np.zeros((T - order, order), dtype=dtype)
        y = f_i[order:]  # (T-order,)
        
        for lag in range(order):
            X[:, lag] = f_i[order - lag - 1:T - lag - 1]
        
        # OLS: ar_coeffs = (X'X + reg*I)^(-1) X'y
        try:
            ar_i = solve_regularized_ols(X, y, regularization=regularization, dtype=dtype)
            
            # Ensure stability: check if AR polynomial is stable
            # For AR(p), roots of 1 - a[1]*z - ... - a[p]*z^p should be outside unit circle
            # Simplified: check if sum of coefficients < 1 (rough stability check)
            if np.sum(np.abs(ar_i)) >= VAR_STABILITY_THRESHOLD:
                # Scale down to ensure stability
                ar_i = ar_i * (VAR_STABILITY_THRESHOLD * 0.9 / np.sum(np.abs(ar_i)))
            
            ar_coeffs[i, :] = ar_i
            
            # Estimate innovation variance from residuals
            residuals = y - X @ ar_i
            B_diag[i] = np.maximum(
                compute_var_safe(residuals, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE),
                MIN_DIAGONAL_VARIANCE
            )
        except Exception as e:
            _logger.warning(f"AR({order}) estimation failed for factor {i}: {e}. Using zeros.")
            ar_coeffs[i, :] = 0.0
            B_diag[i] = DEFAULT_PROCESS_NOISE
    
    return ar_coeffs, B_diag


def estimate_constrained_ols(
    y: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    q: np.ndarray,
    V_smooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> np.ndarray:
    """Constrained OLS estimation that works with raw data or smoothed expectations.
    
    Solves: min ||y - X*beta||^2 subject to R @ beta = q
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable (T x n) for raw data, or E[y_t * f_t'] for smoothed expectations
    X : np.ndarray
        Independent variables (T x p) for raw data, or E[f_t * f_t'] for smoothed expectations
    R : np.ndarray
        Constraint matrix (n_constraints x p)
    q : np.ndarray
        Constraint vector (n_constraints,)
    V_smooth : np.ndarray, optional
        Smoothed state covariances. Required for smoothed expectations mode.
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization parameter
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    beta : np.ndarray
        Constrained OLS coefficients (p,)
    """
    if V_smooth is not None:
        # Smoothed expectations mode
        # y is E[y_t * f_t'], X is E[f_t * f_t']
        # Need to handle V_smooth if it affects the computation
        
        # Unconstrained OLS: beta = X^(-1) @ y
        # X is already a covariance matrix, so use use_XTX=False
        beta_unconstrained = solve_regularized_ols(X, y, regularization=regularization, use_XTX=False, dtype=dtype)
        
        # Apply constraints: beta_constrained = beta_unconstrained - X^(-1) @ R' @ (R @ X^(-1) @ R')^(-1) @ (R @ beta_unconstrained - q)
        # Optimized: avoid computing X^(-1) explicitly, solve linear systems instead
        try:
            X_reg = X + create_scaled_identity(X.shape[0], regularization, dtype=dtype)
            constraint_term = R @ beta_unconstrained - q
            
            # Solve: X_reg @ temp1 = R.T  (instead of computing X_inv @ R.T)
            # Then: (R @ X_inv @ R.T) = R @ temp1 = R @ solve(X_reg, R.T)
            R_X_reg_inv = solve(X_reg, R.T, overwrite_a=False, overwrite_b=False, check_finite=False)
            R_X_inv_RT = R @ R_X_reg_inv
            R_X_inv_RT_reg = R_X_inv_RT + create_scaled_identity(R_X_inv_RT.shape[0], regularization, dtype=dtype)
            
            # Solve: (R_X_inv_RT_reg) @ temp2 = constraint_term
            temp2 = solve(R_X_inv_RT_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
            
            # beta_constrained = beta_unconstrained - R_X_reg_inv @ temp2
            beta_constrained = beta_unconstrained - R_X_reg_inv @ temp2
        except (np.linalg.LinAlgError, ValueError):
            beta_constrained = beta_unconstrained
    else:
        # Raw data mode
        # Unconstrained OLS: beta = (X'X)^(-1) X'y
        beta_unconstrained = solve_regularized_ols(X, y, regularization=regularization, dtype=dtype)
        
        # Apply constraints
        # Optimized: avoid computing (X'X)^(-1) explicitly, solve linear systems instead
        try:
            XTX = X.T @ X
            # Check condition number and increase regularization if ill-conditioned
            try:
                eigenvals_XTX = np.linalg.eigvalsh(XTX)
                max_eig = np.max(np.abs(eigenvals_XTX))
                min_eig = np.max(np.abs(eigenvals_XTX[eigenvals_XTX != 0]))
                cond_num = max_eig / max(min_eig, 1e-12)
                # Increase regularization for ill-conditioned matrices
                if cond_num > 1e10:
                    adaptive_reg = regularization * (cond_num / 1e10)
                    XTX_reg = XTX + create_scaled_identity(XTX.shape[0], adaptive_reg, dtype=dtype)
                else:
                    XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype=dtype)
            except:
                # Fallback to default regularization if eigendecomposition fails
                XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype=dtype)
            
            constraint_term = R @ beta_unconstrained - q
            
            # Solve: XTX_reg @ temp1 = R.T  (instead of computing XTX_inv @ R.T)
            # Use lstsq for better handling of ill-conditioned matrices
            try:
                R_XTX_inv = solve(XTX_reg, R.T, overwrite_a=False, overwrite_b=False, check_finite=False)
            except np.linalg.LinAlgError:
                # Fallback to least squares for very ill-conditioned matrices
                from scipy.linalg import lstsq
                R_XTX_inv, _, _, _ = lstsq(XTX_reg, R.T, lapack_driver='gelsy', cond=1e-10)
            
            R_XTX_inv_RT = R @ R_XTX_inv
            R_XTX_inv_RT_reg = R_XTX_inv_RT + create_scaled_identity(R_XTX_inv_RT.shape[0], regularization, dtype=dtype)
            
            # Solve: (R_XTX_inv_RT_reg) @ temp2 = constraint_term
            try:
                temp2 = solve(R_XTX_inv_RT_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
            except np.linalg.LinAlgError:
                # Fallback to least squares if still ill-conditioned
                from scipy.linalg import lstsq
                temp2, _, _, _ = lstsq(R_XTX_inv_RT_reg, constraint_term, lapack_driver='gelsy', cond=1e-10)
            
            # beta_constrained = beta_unconstrained - R_XTX_inv @ temp2
            beta_constrained = beta_unconstrained - R_XTX_inv @ temp2
        except (np.linalg.LinAlgError, ValueError) as e:
            _logger.debug(f"Constrained OLS failed, using unconstrained solution: {e}")
            beta_constrained = beta_unconstrained
    
    return beta_constrained.astype(dtype)


def estimate_variance(
    residuals: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    EZ: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    V_smooth: Optional[np.ndarray] = None,
    min_variance: float = MIN_EIGENVALUE,
    default_variance: float = DEFAULT_PROCESS_NOISE,
    dtype: type = np.float32
) -> np.ndarray:
    """Variance estimation that works with raw residuals or smoothed expectations.
    
    Parameters
    ----------
    residuals : np.ndarray, optional
        Raw residuals (T x N). Required for raw data mode.
    X : np.ndarray, optional
        Data array (T x N). Required for smoothed expectations mode.
    EZ : np.ndarray, optional
        Smoothed state means (T+1 x m). Required for smoothed expectations mode.
    C : np.ndarray, optional
        Observation matrix (N x m). Required for smoothed expectations mode.
    V_smooth : np.ndarray, optional
        Smoothed state covariances (T+1 x m x m). Required for smoothed expectations mode.
    min_variance : float, default MIN_EIGENVALUE
        Minimum variance floor
    default_variance : float, default DEFAULT_PROCESS_NOISE
        Default variance if estimation fails
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    R : np.ndarray
        Variance/covariance matrix (N x N), diagonal
    """
    if residuals is not None:
        # Raw data mode
        T, N = residuals.shape
        
        if T <= 1:
            R = create_scaled_identity(N, default_variance, dtype=dtype)
            return R
        
        # Compute variance for each series
        var_res = np.array([
            compute_var_safe(
                residuals[:, i][np.isfinite(residuals[:, i])],
                ddof=0,
                min_variance=min_variance,
                default_variance=default_variance
            )
            if np.sum(np.isfinite(residuals[:, i])) > 1 else default_variance
            for i in range(N)
        ], dtype=dtype)
        var_res = np.where(np.isfinite(var_res), var_res, default_variance)
        var_res = np.maximum(var_res, min_variance)
        
        R = np.diag(var_res)
    else:
        # Smoothed expectations mode
        if X is None or EZ is None or C is None:
            raise ConfigurationError(
                "X, EZ, and C are required for smoothed expectations mode",
                details=f"Missing required parameters: X={X is None}, EZ={EZ is None}, C={C is None}"
            )
        
        T, N = X.shape
        m = EZ.shape[1]
        
        # Compute residuals from smoothed expectations
        # R = E[(y_t - C @ z_t) (y_t - C @ z_t)'] = E[y_t y_t'] - C @ E[z_t y_t'] - E[y_t z_t'] @ C' + C @ E[z_t z_t'] @ C'
        # For diagonal R, we only need diagonal elements
        
        R = np.zeros((N, N), dtype=dtype)
        
        for t in range(T):
            # Residual: y_t - C @ z_{t+1}
            z_t = EZ[t+1, :]  # (m,)
            residual = X[t, :] - C @ z_t  # (N,)
            
            # Add contribution: residual @ residual' + C @ V_{t+1} @ C'
            R += np.outer(residual, residual)
            if V_smooth is not None:
                V_t = V_smooth[t+1]  # (m x m)
                R += C @ V_t @ C.T
        
        R = R / T
        
        # Extract diagonal and apply floors
        R_diag = np.diag(R)
        R_diag = np.maximum(R_diag, min_variance)
        R_diag = np.where(np.isfinite(R_diag), R_diag, default_variance)
        R = np.diag(R_diag)
    
    return R.astype(dtype)


def compute_initial_covariance_from_transition(
    A: np.ndarray,
    Q: np.ndarray,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> np.ndarray:
    """Compute initial covariance V_0 from transition matrix A and process noise Q.
    
    Solves the Lyapunov equation: (I - A ⊗ A) vec(V_0) = vec(Q)
    This is used to compute the steady-state covariance for the initial state.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix (n x n)
    Q : np.ndarray
        Process noise covariance (n x n)
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization parameter for numerical stability
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    V_0 : np.ndarray
        Initial covariance matrix (n x n)
    """
    n = A.shape[0]
    try:
        kron_AA = np.kron(A, A)
        eye_kron = create_scaled_identity(n ** 2, DEFAULT_IDENTITY_SCALE, dtype=dtype)
        V_0_flat = solve(
            eye_kron - kron_AA + create_scaled_identity(n ** 2, regularization, dtype=dtype),
            Q.flatten(),
            overwrite_a=False,
            overwrite_b=False,
            check_finite=False
        )
        V_0 = V_0_flat.reshape(n, n).astype(dtype)
        return V_0
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use Q as initial covariance
        _logger.warning(
            f"Initial covariance computation failed for transition matrix of size {n}x{n}. "
            f"Using process noise Q as fallback."
        )
        return Q.copy().astype(dtype)


__all__ = [
    # AR clipping
    'clip_ar',
    'apply_ar_clipping',
    # Estimation functions
    'estimate_idio_dynamics',
    # DDFM-specific functions
    'get_idio',
    'get_transition_params',
    # Estimation functions
    'estimate_var',
    'estimate_ar1',
    'estimate_constrained_ols',
    'estimate_variance',
    'compute_initial_covariance_from_transition',
    'stabilize_innovation_covariance',
    # Forecast functions
    'forecast_ar1_factors',
]

