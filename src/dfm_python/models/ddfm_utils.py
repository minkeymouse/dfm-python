"""DDFM utility functions (training, state-space, conversion, loss).

This module contains utility functions for DDFM extracted from ddfm.py
to keep the main file under 1000 lines.
"""

import numpy as np
from typing import Tuple, Optional
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None
    optim = None

from ..core.helpers import get_logger

_logger = get_logger(__name__)


if _has_torch:
    def train_autoencoder(
        encoder: nn.Module,
        decoder: nn.Module,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        verbose: bool = True,
    ) -> None:
        """Train the autoencoder using PyTorch.
        
        Parameters
        ----------
        encoder : nn.Module
            Encoder network
        decoder : nn.Module
            Decoder network
        X : np.ndarray
            Standardized data (T x N)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate for Adam optimizer
        device : torch.device
            Device to run training on
        verbose : bool
            Whether to log training progress. Default: True
        """
        T, N = X.shape
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate,
        )
        
        # Loss function (MSE)
        criterion = nn.MSELoss()
        
        # Training loop
        encoder.train()
        decoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_target in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                factors = encoder(batch_X)
                reconstructed = decoder(factors)
                
                # Compute loss
                loss = criterion(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                _logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
        
        encoder.eval()
        decoder.eval()
else:
    def train_autoencoder(*args, **kwargs):
        """Placeholder when PyTorch is not available."""
        raise ImportError("PyTorch is required for DDFM training")


# State-space model building utilities




def estimate_var1(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(1) dynamics for factors.
    
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
        Q = np.eye(m) * 0.1
        return A, Q
    
    # Prepare data for OLS: f_t = A @ f_{t-1}
    Y = factors[1:, :]  # T-1 x m (dependent)
    X = factors[:-1, :]  # T-1 x m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y
    try:
        A = np.linalg.solve(X.T @ X + np.eye(m) * 1e-6, X.T @ Y).T
    except np.linalg.LinAlgError:
        # Fallback to pinv
        A = np.linalg.pinv(X) @ Y
    
    # Ensure stability: clip eigenvalues
    eigenvals = np.linalg.eigvals(A)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= 0.99:
        A = A * (0.99 / max_eigenval)
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = np.cov(residuals.T)
    
    # Ensure Q is positive definite
    Q = (Q + Q.T) / 2  # Symmetrize
    eigenvals_Q = np.linalg.eigvals(Q)
    min_eigenval = np.min(eigenvals_Q)
    if min_eigenval < 1e-8:
        Q = Q + np.eye(m) * (1e-8 - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * 0.01)
    
    return A, Q


def estimate_var2(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(2) dynamics for factors.
    
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
        A = np.linalg.solve(X.T @ X + np.eye(2 * m) * 1e-6, X.T @ Y).T
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
    if max_eigenval >= 0.99:
        scale = 0.99 / max_eigenval
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
    if min_eigenval < 1e-8:
        Q = Q + np.eye(m) * (1e-8 - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * 0.01)
    
    return A, Q


def estimate_idiosyncratic_dynamics(
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
    min_obs : int
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
                Q_eps[j, j] = 1e-8
        else:
            # Extract valid consecutive pairs
            eps_t = residuals[1:, j][valid_pairs]
            eps_t_1 = residuals[:-1, j][valid_pairs]
            
            # Estimate AR(1) coefficient using covariance
            var_eps_t_1 = np.var(eps_t_1)
            if var_eps_t_1 > 1e-10:
                cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
                A_eps[j, j] = cov_eps / var_eps_t_1
                
                # Ensure stability: clip AR(1) coefficient
                if abs(A_eps[j, j]) >= 0.99:
                    sign = np.sign(A_eps[j, j])
                    A_eps[j, j] = sign * 0.99
                    _logger.debug(
                        f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                    )
            else:
                A_eps[j, j] = 0.0
            
            # Estimate innovation covariance
            residuals_ar = eps_t - A_eps[j, j] * eps_t_1
            Q_eps[j, j] = np.var(residuals_ar)
            Q_eps[j, j] = max(Q_eps[j, j], 1e-8)  # Floor
    
    return A_eps, Q_eps


def build_observation_matrix(C: np.ndarray, factor_order: int, N: int) -> np.ndarray:
    """Build observation matrix H including idiosyncratic components.
    
    Constructs the observation matrix H = [C, I] for VAR(1) or
    H = [C, 0, I] for VAR(2), where C loads on factors and I on idio.
    
    Parameters
    ----------
    C : np.ndarray
        Loading matrix (N x m) from decoder
    factor_order : int
        VAR lag order (1 or 2)
    N : int
        Number of series
        
    Returns
    -------
    H : np.ndarray
        Observation matrix (N x state_dim)
    """
    N_series, m = C.shape
    
    if factor_order == 1:
        # H = [C, I] where C loads on f_t, I loads on eps_t
        H = np.hstack([C, np.eye(N_series)])
    elif factor_order == 2:
        # H = [C, 0, I] where C loads on f_t, 0 on f_{t-1}, I on eps_t
        H = np.hstack([C, np.zeros((N_series, m)), np.eye(N_series)])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
    return H


def build_state_space(
    factors: np.ndarray,
    A_f: np.ndarray,
    Q_f: np.ndarray,
    A_eps: np.ndarray,
    Q_eps: np.ndarray,
    factor_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space model with companion form.
    
    Constructs the complete state-space model including both factors
    and idiosyncratic components in the state vector.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
    A_f : np.ndarray
        Factor transition matrix (m x m) for VAR(1) or (m x 2m) for VAR(2)
    Q_f : np.ndarray
        Factor innovation covariance (m x m)
    A_eps : np.ndarray
        Idiosyncratic AR(1) coefficients (N x N), diagonal
    Q_eps : np.ndarray
        Idiosyncratic innovation covariance (N x N), diagonal
    factor_order : int
        VAR lag order (1 or 2)
        
    Returns
    -------
    A : np.ndarray
        Complete transition matrix (state_dim x state_dim)
    Q : np.ndarray
        Complete innovation covariance (state_dim x state_dim)
    Z_0 : np.ndarray
        Initial state vector
    V_0 : np.ndarray
        Initial state covariance
    """
    T, m = factors.shape
    N = A_eps.shape[0]
    
    if factor_order == 1:
        # State: [f_t, eps_t]
        # Transition: f_t = A_f @ f_{t-1} + v_f, eps_t = A_eps @ eps_{t-1} + v_eps
        # Block diagonal structure
        A = np.block([
            [A_f, np.zeros((m, N))],
            [np.zeros((N, m)), A_eps]
        ])
        
        Q = np.block([
            [Q_f, np.zeros((m, N))],
            [np.zeros((N, m)), Q_eps]
        ])
        
        # Initial state: [f_0, eps_0]
        Z_0 = np.concatenate([factors[0, :], np.zeros(N)])
        
        # Initial covariance: block diagonal
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))  # Use Q_eps as initial idio covariance
        V_0 = np.block([
            [V_f, np.zeros((m, N))],
            [np.zeros((N, m)), V_eps]
        ])
        
    elif factor_order == 2:
        # State: [f_t, f_{t-1}, eps_t]
        # Transition: f_t = A1 @ f_{t-1} + A2 @ f_{t-2} + v_f
        #            f_{t-1} = f_{t-1} (identity)
        #            eps_t = A_eps @ eps_{t-1} + v_eps
        A1 = A_f[:, :m]
        A2 = A_f[:, m:]
        
        A = np.block([
            [A1, A2, np.zeros((m, N))],
            [np.eye(m), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), A_eps]
        ])
        
        Q = np.block([
            [Q_f, np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), Q_eps]
        ])
        
        # Initial state: [f_0, f_{-1}, eps_0]
        # Use f_0 for both f_0 and f_{-1} (or use first two if available)
        if T >= 2:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        else:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        
        # Initial covariance: block diagonal
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))
        V_0 = np.block([
            [V_f, V_f, np.zeros((m, N))],
            [V_f, V_f, np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), V_eps]
        ])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
    return A, Q, Z_0, V_0


def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation matrix C and bias from trained decoder.
    
    Parameters
    ----------
    decoder
        Trained PyTorch decoder module
        
    Returns
    -------
    C : np.ndarray
        Loading matrix (N x m) from decoder weights
    bias : np.ndarray
        Bias terms (N,)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for DDFM")
    
    decoder_layer = decoder.decoder
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = decoder_layer.weight.data.cpu().numpy()
    
    # Extract bias if present
    if decoder_layer.bias is not None:
        bias = decoder_layer.bias.data.cpu().numpy()
    else:
        bias = np.zeros(weight.shape[0])
    
    # C = weight.T (m x N) -> (N x m) for consistency with DFMResult
    C = weight.T
    
    return C, bias




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
    
    --------
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


def mse_missing(
    y_actual: torch.Tensor,
    y_predicted: torch.Tensor,
) -> torch.Tensor:
    """Mean Squared Error loss function that handles missing data.
    
    Computes MSE only on non-missing values. Missing values in y_actual
    (represented as NaN) are masked out from the loss computation.
    
    Parameters
    ----------
    y_actual : torch.Tensor
        Actual values (batch_size x N) with NaN for missing values
    y_predicted : torch.Tensor
        Predicted values (batch_size x N)
        
    Returns
    -------
    torch.Tensor
        Scalar MSE loss computed only on non-missing values
        
    --------
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for mse_missing")
    
    # Create mask: 1 for non-missing, 0 for missing
    mask = torch.where(
        torch.isnan(y_actual),
        torch.zeros_like(y_actual),
        torch.ones_like(y_actual)
    )
    
    # Replace NaN with 0 for computation
    y_actual_clean = torch.where(
        torch.isnan(y_actual),
        torch.zeros_like(y_actual),
        y_actual
    )
    
    # Apply mask to predictions
    y_predicted_masked = y_predicted * mask
    
    # Compute MSE (automatically ignores masked values)
    loss = nn.functional.mse_loss(y_actual_clean, y_predicted_masked, reduction='mean')
    
    return loss


def convergence_checker(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
) -> Tuple[float, float]:
    """Check convergence of reconstruction error (matches original TensorFlow implementation).
    
    Returns only delta and loss_now (no converged flag), matching original code.
    
    Parameters
    ----------
    y_prev : np.ndarray
        Previous reconstruction (T x N)
    y_now : np.ndarray
        Current reconstruction (T x N)
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
        
    Returns
    -------
    delta : float
        Relative change in loss: |loss_now - loss_prev| / loss_prev
    loss_now : float
        Current MSE loss (on non-missing values)
    """
    # Mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    # Compute MSE on non-missing values (matches original implementation)
    y_prev_valid = y_prev[mask]
    y_now_valid = y_now[mask]
    y_actual_valid = y_actual[mask]
    
    loss_prev = np.mean((y_actual_valid - y_prev_valid) ** 2)
    loss_now = np.mean((y_actual_valid - y_now_valid) ** 2)
    
    # Relative change
    if loss_prev < 1e-10:
        delta = abs(loss_now - loss_prev)
    else:
        delta = abs(loss_now - loss_prev) / loss_prev
    
    return delta, loss_now


def check_convergence(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
    threshold: float = 1e-6,
) -> Tuple[float, float, bool]:
    """Check convergence of reconstruction error.
    
    Computes the relative change in MSE between two iterations and checks
    if convergence has been reached.
    
    Parameters
    ----------
    y_prev : np.ndarray
        Previous reconstruction (T x N)
    y_now : np.ndarray
        Current reconstruction (T x N)
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
    threshold : float
        Convergence threshold for relative change in loss
        
    Returns
    -------
    relative_change : float
        Relative change in loss: |loss_now - loss_prev| / loss_prev
    loss_now : float
        Current MSE loss (on non-missing values)
    converged : bool
        True if relative change is below threshold
        
    --------
    True
    """
    # Mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    # Compute MSE on non-missing values (NumPy implementation, no sklearn dependency)
    # Previous loss
    y_prev_valid = y_prev[mask]
    y_actual_valid = y_actual[mask]
    loss_prev = np.mean((y_actual_valid - y_prev_valid) ** 2)
    
    # Current loss
    y_now_valid = y_now[mask]
    loss_now = np.mean((y_actual_valid - y_now_valid) ** 2)
    
    # Relative change
    if loss_prev < 1e-10:
        # Near-zero loss, consider converged
        relative_change = 0.0
    else:
        relative_change = abs(loss_now - loss_prev) / loss_prev
    
    converged = relative_change < threshold
    
    return relative_change, loss_now, converged


def mse_missing_numpy(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
) -> float:
    """NumPy version of missing-aware MSE loss.
    
    Computes MSE only on non-missing values. Missing values in y_actual
    (represented as NaN) are masked out from the loss computation.
    
    Parameters
    ----------
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
    y_predicted : np.ndarray
        Predicted values (T x N)
        
    Returns
    -------
    float
        MSE loss computed only on non-missing values
        
    """
    # Create mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    if np.sum(mask) == 0:
        # All values are missing
        return 0.0
    
    # Compute MSE only on non-missing values
    y_actual_valid = y_actual[mask]
    y_predicted_valid = y_predicted[mask]
    
    mse = np.mean((y_actual_valid - y_predicted_valid) ** 2)
    
    return mse


if _has_torch:
    def fit_ddfm_mcmc(
        encoder: nn.Module,
        decoder: nn.Module,
        x_standardized: np.ndarray,
        x_clean: np.ndarray,
        missing_mask: np.ndarray,
        epochs_per_iter: int,
        batch_size: int,
        learning_rate: float,
        max_iter: int,
        tolerance: float,
        disp: int,
        device: torch.device,
        rng: np.random.RandomState,
        use_idiosyncratic: bool,
        min_obs_idio: int,
        lags_input: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool, int]:
        """MCMC iterative training procedure (matches original TensorFlow DDFM).
        
        Parameters
        ----------
        encoder : nn.Module
            Encoder network
        decoder : nn.Module
            Decoder network
        x_standardized : np.ndarray
            Standardized data with missing values (T x N)
        x_clean : np.ndarray
            Clean data (interpolated, T x N)
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
        epochs_per_iter : int
            Number of epochs per MCMC iteration (MC samples)
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for Adam optimizer
        max_iter : int
            Maximum number of MCMC iterations
        tolerance : float
            Convergence tolerance
        disp : int
            Display progress every 'disp' iterations
        device : torch.device
            Device to run training on
        rng : np.random.RandomState
            Random number generator for MC sampling
        use_idiosyncratic : bool
            Whether to model idiosyncratic components
        min_obs_idio : int
            Minimum observations for idio estimation
        lags_input : int
            Number of input lags (currently not used, for future extension)
            
        Returns
        -------
        factors : np.ndarray
            Final extracted factors (T x num_factors)
        prediction_iter : np.ndarray
            Final reconstruction (T x N)
        converged : bool
            Whether convergence was achieved
        num_iter : int
            Number of iterations completed
        """
        T, N = x_clean.shape
        bool_no_miss = ~missing_mask
        
        # Initialize data structures (matching original TensorFlow code)
        data_mod_only_miss = x_standardized.copy()  # Original with missing values
        data_mod = x_clean.copy()  # Clean data (will be modified during MCMC)
        
        # Initial prediction
        x_tensor = torch.FloatTensor(data_mod).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            factors_init = encoder(x_tensor).cpu().numpy()
            prediction_iter = decoder(torch.FloatTensor(factors_init).to(device)).cpu().numpy()
        
        # Update missing values with initial prediction
        bool_miss = missing_mask
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Initial residuals
        eps = data_mod_only_miss - prediction_iter
        
        # MCMC loop
        iter_count = 0
        not_converged = True
        prediction_prev_iter = None
        delta = float('inf')  # Initialize to large value
        
        _logger.info(f"Starting MCMC training: max_iter={max_iter}, tolerance={tolerance}")
        
        while not_converged and iter_count < max_iter:
            iter_count += 1
            
            # Get idio distribution
            if use_idiosyncratic:
                phi, mu_eps, std_eps = estimate_idiosyncratic_params(eps, bool_no_miss, min_obs_idio)
            else:
                # No idio modeling: use zero mean, small std
                phi = np.zeros((N, N))
                mu_eps = np.zeros(N)
                std_eps = np.ones(N) * 1e-8
            
            # Subtract conditional AR-idio mean from x (matching original code)
            if use_idiosyncratic and eps.shape[0] > 1:
                # data_mod[lags_input + 1:] = data_mod_only_miss[lags_input + 1:] - eps[:-1, :] @ phi
                data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
                # For first observations set to 0 the idio
                data_mod[:1] = data_mod_only_miss[:1]
            else:
                data_mod = data_mod_only_miss.copy()
            
            # Generate MC samples for idio (dims = epochs x T x N)
            # Original: eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (epochs, T))
            # But multivariate_normal expects 2D, so we sample per time step
            eps_draws = np.zeros((epochs_per_iter, T, N))
            for t in range(T):
                # Sample for each time step independently
                eps_draws[:, t, :] = rng.multivariate_normal(
                    mu_eps, np.diag(std_eps), size=epochs_per_iter
                )
            
            # Initialize noisy inputs (dims = epochs x T x N)
            x_sim_den = np.zeros((epochs_per_iter, T, N))
            
            # Loop over MC samples (matching original code)
            factors_samples = []
            for i in range(epochs_per_iter):
                x_sim_den[i, :, :] = data_mod.copy()
                # Corrupt input data, only current observations
                x_sim_den[i, :, :] = x_sim_den[i, :, :] - eps_draws[i, :, :]
                
                # Train autoencoder on corrupted sample (1 epoch)
                train_autoencoder(
                    encoder, decoder, x_sim_den[i, :, :],
                    epochs=1, batch_size=batch_size,
                    learning_rate=learning_rate, device=device,
                    verbose=False
                )
                
                # Extract factors from this sample
                x_sample_tensor = torch.FloatTensor(x_sim_den[i, :, :]).to(device)
                encoder.eval()
                with torch.no_grad():
                    factors_sample = encoder(x_sample_tensor).cpu().numpy()
                factors_samples.append(factors_sample)
            
            # Update factors: average over all MC samples
            factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
            
            # Check convergence
            decoder.eval()
            with torch.no_grad():
                factors_tensor = torch.FloatTensor(factors).to(device)
                prediction_iter = decoder(factors_tensor).cpu().numpy()
            
            if iter_count > 1 and prediction_prev_iter is not None:
                delta, loss_now = convergence_checker(
                    prediction_prev_iter, prediction_iter, data_mod_only_miss
                )
                
                if iter_count % disp == 0:
                    _logger.info(
                        f"Iteration {iter_count}/{max_iter}: loss={loss_now:.6f}, delta={delta:.6f}"
                    )
                
                if delta < tolerance:
                    not_converged = False
                    _logger.info(
                        f"Convergence achieved in {iter_count} iterations: "
                        f"loss={loss_now:.6f}, delta={delta:.6f} < {tolerance}"
                    )
            
            # Store previous prediction for convergence checking
            prediction_prev_iter = prediction_iter.copy()
            
            # Update missing values
            if bool_miss.any():
                data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
            
            # Update residuals
            eps = data_mod_only_miss - prediction_iter
        
        if not_converged:
            if iter_count > 1 and delta != float('inf'):
                delta_str = f"{delta:.6f}"
            else:
                delta_str = 'N/A'
            _logger.warning(
                f"Convergence not achieved within {max_iter} iterations. "
                f"Final delta: {delta_str}"
            )
        
        converged = not not_converged
        return factors, prediction_iter, converged, iter_count
else:
    def fit_ddfm_mcmc(*args, **kwargs):
        """Placeholder when PyTorch is not available."""
        raise ImportError("PyTorch is required for DDFM MCMC training")
