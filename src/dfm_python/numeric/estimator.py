"""Estimation functions for state-space model parameters.

This module provides functions for estimating VAR dynamics, AR coefficients,
and idiosyncratic component parameters from data.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..logger import get_logger
from ..config.constants import (
    MIN_DIAGONAL_VARIANCE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    MIN_EIGENVALUE,
    VAR_STABILITY_THRESHOLD,
    AR_CLIP_MIN,
    AR_CLIP_MAX,
    MIN_Q_FLOOR,
    DEFAULT_PROCESS_NOISE,
)
from .stability import clean_matrix, cap_max_eigenval
from .analytic import compute_var_safe

_logger = get_logger(__name__)

# Numerical stability constants
MIN_VARIANCE_COVARIANCE = MIN_FACTOR_VARIANCE
DEFAULT_VARIANCE_FALLBACK = 1.0


def estimate_ar(
    EZZ_FB: np.ndarray,
    EZZ_BB: np.ndarray,
    vsmooth_sum: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Estimate AR coefficients and innovation variances from expectations.
    
    Parameters
    ----------
    EZZ_FB : np.ndarray
        Forward-backward expectation E[z_t z_{t-1}']
    EZZ_BB : np.ndarray
        Backward-backward expectation E[z_{t-1} z_{t-1}']
    vsmooth_sum : np.ndarray, optional
        Sum of smoothing variances
        
    Returns
    -------
    A_diag : np.ndarray
        AR coefficients (diagonal)
    Q_diag : np.ndarray or None
        Innovation variances (diagonal, currently None)
    """
    if np.isscalar(EZZ_FB):
        EZZ_FB = np.array([EZZ_FB])
        EZZ_BB = np.array([EZZ_BB])
    if EZZ_FB.ndim > 1:
        EZZ_FB_diag = np.diag(EZZ_FB).copy()
        EZZ_BB_diag = np.diag(EZZ_BB).copy()
    else:
        EZZ_FB_diag = EZZ_FB.copy()
        EZZ_BB_diag = EZZ_BB.copy()
    if vsmooth_sum is not None:
        if vsmooth_sum.ndim > 1:
            vsmooth_diag = np.diag(vsmooth_sum)
        else:
            vsmooth_diag = vsmooth_sum
        EZZ_BB_diag = EZZ_BB_diag + vsmooth_diag
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * MIN_DIAGONAL_VARIANCE, MIN_VARIANCE_COVARIANCE)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    # Use clean_matrix for consistency
    if EZZ_FB_diag.ndim == 0:
        EZZ_FB_diag_clean = clean_matrix(np.array([EZZ_FB_diag]), 'general', default_nan=0.0, default_inf=1e6)
        EZZ_FB_diag = EZZ_FB_diag_clean[0] if EZZ_FB_diag_clean.size > 0 else 0.0
    else:
        EZZ_FB_diag = clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0, default_inf=1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    # Q_diag is not computed here (returns None for compatibility)
    Q_diag: Optional[np.ndarray] = None
    return A_diag, Q_diag


def estimate_var1(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(1) dynamics for factors.
    
    Note: Maximum supported VAR order is VAR(2). Use estimate_var2() for VAR(2) estimation.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x m)
    Q : np.ndarray
        Innovation covariance (m x m)
    """
    T, m = factors.shape
    
    if T < 2:
        # Not enough data, use identity
        A = np.eye(m)
        Q = np.eye(m) * DEFAULT_PROCESS_NOISE
        return A, Q
    
    # Prepare data for OLS: f_t = A @ f_{t-1}
    Y = factors[1:, :]  # T-1 x m (dependent)
    X = factors[:-1, :]  # T-1 x m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y
    try:
        A = np.linalg.solve(X.T @ X + np.eye(m) * DEFAULT_REGULARIZATION, X.T @ Y).T
    except np.linalg.LinAlgError:
        # Fallback to pinv
        A = np.linalg.pinv(X) @ Y
    
    # Ensure stability: clip eigenvalues
    eigenvals = np.linalg.eigvals(A)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= VAR_STABILITY_THRESHOLD:
        A = A * (VAR_STABILITY_THRESHOLD / max_eigenval)
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = np.cov(residuals.T)
    
    # Ensure Q is positive definite
    Q = (Q + Q.T) / 2  # Symmetrize
    eigenvals_Q = np.linalg.eigvals(Q)
    min_eigenval = np.min(eigenvals_Q)
    if min_eigenval < MIN_EIGENVALUE:
        Q = Q + np.eye(m) * (MIN_EIGENVALUE - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * MIN_Q_FLOOR)
    
    return A, Q


def estimate_var2(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(2) dynamics for factors.
    
    Note: VAR(2) is the maximum supported VAR order in this implementation.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x 2m) = [A1, A2]
    Q : np.ndarray
        Innovation covariance (m x m)
    """
    T, m = factors.shape
    
    if T < 3:
        # Not enough data, use VAR(1) fallback
        _logger.warning(
            f"Insufficient data (T={T}) for VAR(2). Falling back to VAR(1)."
        )
        A1, Q = estimate_var1(factors)
        # Pad A to VAR(2) format: [A1, A2] where A2 = 0
        A = np.hstack([A1, np.zeros((A1.shape[0], A1.shape[1]))])
        return A, Q
    
    # Prepare data for VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
    Y = factors[2:, :]  # T-2 x m (dependent)
    X = np.hstack((factors[1:-1, :], factors[:-2, :]))  # T-2 x 2m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y, where A = [A1, A2]
    try:
        A = np.linalg.solve(X.T @ X + np.eye(2 * m) * DEFAULT_REGULARIZATION, X.T @ Y).T
    except np.linalg.LinAlgError:
        # Fallback to pinv
        A = np.linalg.pinv(X) @ Y
    
    # Split into A1 and A2
    A1 = A[:, :m]
    A2 = A[:, m:]
    
    # Ensure stability: check eigenvalues of companion form
    companion = np.block([
        [A1, A2],
        [np.eye(m), np.zeros((m, m))]
    ])
    eigenvals = np.linalg.eigvals(companion)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= VAR_STABILITY_THRESHOLD:
        scale = VAR_STABILITY_THRESHOLD / max_eigenval
        A1 = A1 * scale
        A2 = A2 * scale
        A = np.hstack((A1, A2))
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = np.cov(residuals.T)
    
    # Ensure Q is positive definite
    Q = (Q + Q.T) / 2  # Symmetrize
    eigenvals_Q = np.linalg.eigvals(Q)
    min_eigenval = np.min(eigenvals_Q)
    if min_eigenval < MIN_EIGENVALUE:
        Q = Q + np.eye(m) * (MIN_EIGENVALUE - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * MIN_Q_FLOOR)
    
    return A, Q


def estimate_idio_dynamics(
    residuals: np.ndarray,
    missing_mask: np.ndarray,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR(1) dynamics for idiosyncratic components.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from observation equation (T x N)
    missing_mask : np.ndarray
        Missing data mask (T x N), True where data is missing
    min_obs : int, default 5
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
        # Find valid consecutive pairs (both t-1 and t must be non-missing)
        valid = ~missing_mask[:, j]
        valid_pairs = valid[:-1] & valid[1:]
        
        if np.sum(valid_pairs) < min_obs:
            # Insufficient data: use zero AR(1) coefficient
            _logger.warning(
                f"Insufficient observations ({np.sum(valid_pairs)}) for idio AR(1) "
                f"estimation for series {j}. Using zero AR(1) coefficient."
            )
            A_eps[j, j] = 0.0
            # Use variance of available residuals
            if np.sum(valid) > 0:
                Q_eps[j, j] = np.var(residuals[valid, j])
            else:
                Q_eps[j, j] = MIN_DIAGONAL_VARIANCE
        else:
            # Extract valid consecutive pairs
            eps_t = residuals[1:, j][valid_pairs]
            eps_t_1 = residuals[:-1, j][valid_pairs]
            
            # Estimate AR(1) coefficient using covariance
            var_eps_t_1 = np.var(eps_t_1)
            if var_eps_t_1 > MIN_FACTOR_VARIANCE:
                cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
                A_eps[j, j] = cov_eps / var_eps_t_1
                
                # Ensure stability: clip AR(1) coefficient
                if abs(A_eps[j, j]) >= VAR_STABILITY_THRESHOLD:
                    sign = np.sign(A_eps[j, j])
                    A_eps[j, j] = sign * VAR_STABILITY_THRESHOLD
                    _logger.debug(
                        f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                    )
            else:
                A_eps[j, j] = 0.0
            
            # Estimate innovation covariance
            residuals_ar = eps_t - A_eps[j, j] * eps_t_1
            Q_eps[j, j] = np.var(residuals_ar)
            Q_eps[j, j] = max(Q_eps[j, j], MIN_DIAGONAL_VARIANCE)  # Floor
    
    return A_eps, Q_eps


def estimate_idio_params(
    eps: np.ndarray,
    idx_no_missings: Optional[np.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate AR(1) parameters for idiosyncratic components.
    
    Falls back to zero-coefficient models when insufficient observations are
    available instead of raising errors, ensuring downstream pipelines remain
    robust.
    
    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals (T x N)
    idx_no_missings : np.ndarray, optional
        Boolean mask (T x N) indicating non-missing values
    min_obs : int, default 5
        Minimum number of observations required
        
    Returns
    -------
    phi : np.ndarray
        AR(1) coefficients (N x N), diagonal
    mu_eps : np.ndarray
        Mean of idiosyncratic components (N,)
    std_eps : np.ndarray
        Standard deviation of idiosyncratic components (N,)
    """
    T, N = eps.shape
    phi = np.zeros((N, N))
    mu_eps = np.zeros(N)
    std_eps = np.zeros(N)
    
    if idx_no_missings is None:
        idx_no_missings = np.ones((T, N), dtype=bool)
    
    insufficient_series = []
    
    for j in range(N):
        mask = idx_no_missings[:, j]
        observed = eps[mask, j]
        
        if observed.size == 0:
            mu_eps[j] = 0.0
            std_eps[j] = MIN_DIAGONAL_VARIANCE
            insufficient_series.append((j, 0))
            continue
        
        mu_eps[j] = float(np.mean(observed))
        std_eps_j = float(np.std(observed))
        std_eps[j] = max(std_eps_j, 1e-8)
        
        valid_pairs = mask[:-1] & mask[1:]
        pair_count = int(np.sum(valid_pairs))
        
        if pair_count < max(min_obs, 1):
            insufficient_series.append((j, pair_count))
            continue
        
        eps_t = eps[1:, j][valid_pairs]
        eps_t_1 = eps[:-1, j][valid_pairs]
        var_prev = np.var(eps_t_1)
        
        if var_prev < MIN_FACTOR_VARIANCE:
            insufficient_series.append((j, pair_count))
            continue
        
        cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
        coeff = cov_eps / var_prev
        phi[j, j] = float(np.clip(coeff, AR_CLIP_MIN, AR_CLIP_MAX))
    
    if insufficient_series:
        from ..config.constants import MAX_WARNING_ITEMS
        preview = ", ".join(f"{idx}:{cnt}" for idx, cnt in insufficient_series[:MAX_WARNING_ITEMS])
        more = ""
        if len(insufficient_series) > 5:
            more = f", ... (+{len(insufficient_series) - 5} more)"
        _logger.warning(
            "Falling back to zero AR coefficients for %d series (insufficient observations). "
            "Series indices and available pairs: %s%s",
            len(insufficient_series),
            preview,
            more,
        )
    
    return phi, mu_eps, std_eps


def estimate_state_space_params(
    f_t: np.ndarray,
    eps_t: np.ndarray,
    factor_order: int,
    bool_no_miss: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate state-space transition parameters from factors and residuals.
    
    Estimates the transition matrix A, innovation covariance W, initial mean mu_0,
    initial covariance Σ_0, and latent states x_t for the companion form state-space
    representation.
    
    Parameters
    ----------
    f_t : np.ndarray
        Common factors (T x m)
    eps_t : np.ndarray
        Idiosyncratic terms (T x N)
    factor_order : int
        Lag order for common factors. Only VAR(1) and VAR(2) are supported.
    bool_no_miss : np.ndarray, optional
        Boolean array (T x N) indicating non-missing values.
        If None, assumes no missing values.
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (state_dim x state_dim) in companion form
    W : np.ndarray
        Innovation covariance matrix (state_dim x state_dim), diagonal
    mu_0 : np.ndarray
        Unconditional mean of initial state (state_dim,)
    Σ_0 : np.ndarray
        Unconditional covariance of initial state (state_dim x state_dim)
    x_t : np.ndarray
        Latent states (state_dim x T) in companion form
    """
    T, m = f_t.shape
    T_eps, N = eps_t.shape
    
    if T != T_eps:
        raise ValueError(f"Time dimension mismatch: f_t has {T} timesteps, eps_t has {T_eps}")
    
    # Estimate factor dynamics (VAR)
    if factor_order == 2:
        if T < 3:
            raise ValueError("Insufficient data for VAR(2). Need at least 3 timesteps.")
        f_past = np.hstack((f_t[1:-1, :], f_t[:-2, :]))  # (T-2) x 2m
        f_future = f_t[2:, :]  # (T-2) x m
        # OLS: A_f = (f_past' @ f_past)^{-1} @ f_past' @ f_future
        try:
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(2*m) * DEFAULT_REGULARIZATION, f_past.T @ f_future).T
        except np.linalg.LinAlgError:
            A_f = (np.linalg.pinv(f_past) @ f_future).T
        # Split into A1 and A2
        A1 = A_f[:, :m]  # m x m
        A2 = A_f[:, m:]  # m x m
    elif factor_order == 1:
        if T < 2:
            raise ValueError("Insufficient data for VAR(1). Need at least 2 timesteps.")
        f_past = f_t[:-1, :]  # (T-1) x m
        f_future = f_t[1:, :]  # (T-1) x m
        # OLS: A_f = (f_past' @ f_past)^{-1} @ f_past' @ f_future
        try:
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(m) * DEFAULT_REGULARIZATION, f_past.T @ f_future).T
        except np.linalg.LinAlgError:
            A_f = (np.linalg.pinv(f_past) @ f_future).T
        A1 = A_f
        A2 = np.zeros((m, m))  # VAR(1) doesn't use A2, but set to zeros for consistency
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported (maximum supported order is VAR(2)). "
            f"Got factor_order={factor_order}. Please use factor_order=1 (VAR(1)) or factor_order=2 (VAR(2))"
        )
    
    # Estimate idiosyncratic AR(1) dynamics
    A_eps, _, _ = estimate_idio_params(eps_t, bool_no_miss, min_obs=5)
    
    # Construct companion form state vector and transition matrix
    if factor_order == 2:
        # x_t = [f_t, f_{t-1}, eps_t]
        x_t = np.vstack([
            f_t[1:, :].T,  # m x (T-1)
            f_t[:-1, :].T,  # m x (T-1)
            eps_t[1:, :].T  # N x (T-1)
        ])  # (2m + N) x (T-1)
        
        # Transition matrix in companion form
        A = np.vstack([
            np.hstack([A1, A2, np.zeros((m, N))]),  # f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            np.hstack([np.eye(m), np.zeros((m, m)), np.zeros((m, N))]),  # f_{t-1} = f_{t-1}
            np.hstack([np.zeros((N, m)), np.zeros((N, m)), A_eps])  # eps_t = A_eps @ eps_{t-1}
        ])
    else:  # factor_order == 1
        # x_t = [f_t, eps_t]
        x_t = np.vstack([
            f_t.T,  # m x T
            eps_t.T  # N x T
        ])  # (m + N) x T
        
        # Transition matrix
        A = np.vstack([
            np.hstack([A1, np.zeros((m, N))]),  # f_t = A1 @ f_{t-1}
            np.hstack([np.zeros((N, m)), A_eps])  # eps_t = A_eps @ eps_{t-1}
        ])
    
    # Estimate innovation covariance (diagonal)
    # w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    W = np.diag(np.diag(np.cov(w_t)))
    # Ensure positive diagonal
    W = np.maximum(W, np.eye(W.shape[0]) * MIN_DIAGONAL_VARIANCE)
    
    # Unconditional moments of initial state
    mu_0 = np.mean(x_t, axis=1)
    Σ_0 = np.cov(x_t)
    
    # Enforce zero correlation between factors and idiosyncratic components
    if factor_order == 2:
        factor_dim = 2 * m
    else:
        factor_dim = m
    
    Σ_0[:factor_dim, factor_dim:] = 0
    Σ_0[factor_dim:, :factor_dim] = 0
    # Ensure diagonal covariance for idiosyncratic components
    Σ_0[factor_dim:, factor_dim:] = np.diag(np.diag(Σ_0[factor_dim:, factor_dim:]))
    
    # Ensure positive semidefinite
    from .stability import ensure_positive_definite
    eigenvals = np.linalg.eigvals(Σ_0)
    if np.any(eigenvals < 0):
        Σ_0 = Σ_0 + np.eye(Σ_0.shape[0]) * (MIN_DIAGONAL_VARIANCE - np.min(eigenvals))
    
    return A, W, mu_0, Σ_0, x_t


__all__ = [
    'estimate_ar',
    'estimate_var1',
    'estimate_var2',
    'estimate_idio_dynamics',
    'estimate_idio_params',
    'estimate_state_space_params',
]

