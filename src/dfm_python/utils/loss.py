"""Loss functions and convergence checking utilities for DDFM training.

This module provides missing-aware loss functions and convergence checking
utilities for training Deep Dynamic Factor Models.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None


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
        
    Examples
    --------
    >>> import torch
    >>> y_actual = torch.tensor([[1.0, 2.0, float('nan')], [3.0, 4.0, 5.0]])
    >>> y_predicted = torch.tensor([[1.1, 2.1, 0.0], [3.1, 4.1, 5.1]])
    >>> loss = mse_missing(y_actual, y_predicted)
    >>> loss.item()  # Only computes MSE on non-NaN values
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
        
    Examples
    --------
    >>> import numpy as np
    >>> y_prev = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y_now = np.array([[1.01, 2.01], [3.01, 4.01]])
    >>> y_actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> rel_change, loss, converged = convergence_checker(y_prev, y_now, y_actual)
    >>> converged
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
        
    Examples
    --------
    >>> import numpy as np
    >>> y_actual = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
    >>> y_predicted = np.array([[1.1, 2.1, 0.0], [3.1, 4.1, 5.1]])
    >>> loss = mse_missing_numpy(y_actual, y_predicted)
    >>> loss  # Only computes MSE on non-NaN values
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

