"""Conversion utilities for DDFM models.

This module provides utilities to convert PyTorch decoder models to NumPy arrays
and build state-space matrices for Kalman filtering.
"""

from typing import Tuple, Optional
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None


_logger = logging.getLogger(__name__)


def convert_decoder_to_numpy(
    decoder: nn.Module,
    has_bias: bool = True,
    factor_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PyTorch decoder to NumPy arrays for state-space model.
    
    Extracts weights and biases from a PyTorch decoder (typically nn.Linear)
    and constructs the observation matrix (emission matrix) for the state-space
    representation. Supports VAR(1) and VAR(2) factor dynamics.
    
    Parameters
    ----------
    decoder : nn.Module
        PyTorch decoder model (typically a single Linear layer or a model with
        a final Linear layer accessible via `.decoder` attribute)
    has_bias : bool
        Whether the decoder has a bias term
    factor_order : int
        Lag order for common factors (1 for VAR(1), 2 for VAR(2))
        
    Returns
    -------
    bias : np.ndarray
        Bias terms (N,) where N is the number of series
    emission : np.ndarray
        Emission matrix (N x state_dim) for state-space model.
        For VAR(1): [C, I] where C is loading matrix and I is identity for idio
        For VAR(2): [C, zeros, I] where zeros are for lagged factors
        
    Notes
    -----
    The emission matrix structure depends on the state vector:
    - VAR(1): x_t = [f_t, eps_t], emission = [C, I]
    - VAR(2): x_t = [f_t, f_{t-1}, eps_t], emission = [C, zeros, I]
    
    Examples
    --------
    >>> import torch.nn as nn
    >>> decoder = nn.Linear(3, 10)  # 3 factors, 10 series
    >>> bias, emission = convert_decoder_to_numpy(decoder, has_bias=True, factor_order=1)
    >>> emission.shape  # (10, 3 + 10) = (10, 13)
    (10, 13)
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for decoder conversion")
    
    # Extract the actual Linear layer
    if hasattr(decoder, 'decoder'):
        # If decoder is wrapped in a class (e.g., Decoder class)
        linear_layer = decoder.decoder
    elif isinstance(decoder, nn.Linear):
        # If decoder is directly a Linear layer
        linear_layer = decoder
    else:
        # Try to find the last Linear layer
        linear_layers = [m for m in decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("No Linear layer found in decoder")
        linear_layer = linear_layers[-1]
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = linear_layer.weight.data.cpu().numpy()  # N x m
    
    # Extract bias if present
    if has_bias and linear_layer.bias is not None:
        bias = linear_layer.bias.data.cpu().numpy()  # N,
    else:
        bias = np.zeros(weight.shape[0])  # N,
    
    # Construct emission matrix for state-space model
    N, m = weight.shape
    
    if factor_order == 2:
        # VAR(2): x_t = [f_t, f_{t-1}, eps_t]
        # emission = [C, zeros, I]
        # where C is the loading matrix (N x m)
        C = weight.T  # m x N, but we need N x m for emission
        # Actually, emission should be N x (m + m + N) = N x (2m + N)
        emission = np.hstack([
            weight,  # N x m (current factors)
            np.zeros((N, m)),  # N x m (lagged factors, zero contribution)
            np.eye(N)  # N x N (idiosyncratic components)
        ])
    elif factor_order == 1:
        # VAR(1): x_t = [f_t, eps_t]
        # emission = [C, I]
        emission = np.hstack([
            weight,  # N x m (factors)
            np.eye(N)  # N x N (idiosyncratic components)
        ])
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported. "
            f"Got factor_order={factor_order}"
        )
    
    return bias, emission


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
        Lag order for common factors (1 for VAR(1), 2 for VAR(2))
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
        
    Notes
    -----
    The companion form depends on factor_order:
    - VAR(1): x_t = [f_t, eps_t], A = [[A_f, 0], [0, A_eps]]
    - VAR(2): x_t = [f_t, f_{t-1}, eps_t], A = [[A_f, 0, 0], [I, 0, 0], [0, 0, A_eps]]
    
    The innovation covariance W is diagonal, and Σ_0 enforces zero correlation
    between factors and idiosyncratic components.
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
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(2*m) * 1e-6, f_past.T @ f_future).T
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
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(m) * 1e-6, f_past.T @ f_future).T
        except np.linalg.LinAlgError:
            A_f = (np.linalg.pinv(f_past) @ f_future).T
        A1 = A_f
        A2 = None
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported. "
            f"Got factor_order={factor_order}"
        )
    
    # Estimate idiosyncratic AR(1) dynamics
    A_eps, _, _ = estimate_idiosyncratic_params(eps_t, bool_no_miss, min_obs=5)
    
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
    W = np.maximum(W, np.eye(W.shape[0]) * 1e-8)
    
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
    eigenvals = np.linalg.eigvals(Σ_0)
    if np.any(eigenvals < 0):
        Σ_0 = Σ_0 + np.eye(Σ_0.shape[0]) * (1e-8 - np.min(eigenvals))
    
    return A, W, mu_0, Σ_0, x_t


def estimate_idiosyncratic_params(
    eps: np.ndarray,
    idx_no_missings: Optional[np.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate AR(1) parameters for idiosyncratic components.
    
    Falls back to zero-coefficient models when insufficient observations are
    available instead of raising errors, ensuring downstream pipelines remain
    robust.
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
            std_eps[j] = 1e-8
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
        
        if var_prev < 1e-10:
            insufficient_series.append((j, pair_count))
            continue
        
        cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
        coeff = cov_eps / var_prev
        phi[j, j] = float(np.clip(coeff, -0.99, 0.99))
    
    if insufficient_series:
        preview = ", ".join(f"{idx}:{cnt}" for idx, cnt in insufficient_series[:5])
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


# Backward compatibility aliases
get_transition_params = estimate_state_space_params
get_idio = estimate_idiosyncratic_params

