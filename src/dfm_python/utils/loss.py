"""Loss computation utilities for factor models.

This module provides reusable loss functions for training factor models,
including support for missing data masking and robust loss functions.
"""

import torch
from typing import Literal
from ..config.constants import DEFAULT_EPSILON, HUBER_QUADRATIC_COEFF


def compute_masked_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_function: Literal['mse', 'huber'] = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """Compute loss with missing data masking.
    
    This function computes reconstruction loss (MSE or Huber) while properly
    handling missing data through masking. Only observed values (where mask=True)
    contribute to the loss.
    
    Parameters
    ----------
    reconstructed : torch.Tensor
        Model reconstruction, shape (n_mc_samples, T, N) or (T, N) or any shape
        matching target
    target : torch.Tensor
        Target values, same shape as reconstructed
    mask : torch.Tensor
        Missing data mask, same shape as target. True where data is observed,
        False where data is missing. Must be boolean dtype.
    loss_function : {'mse', 'huber'}, default 'mse'
        Loss function to use:
        - 'mse': Mean squared error (default)
        - 'huber': Huber loss (more robust to outliers)
    huber_delta : float, default 1.0
        Delta parameter for Huber loss. Controls the transition point between
        quadratic and linear regions. Only used if loss_function='huber'.
        
    Returns
    -------
    loss : torch.Tensor
        Scalar loss value. Loss is computed over masked (observed) elements only,
        but normalized by total number of elements (target.numel()) to match
        original TensorFlow MeanSquaredError behavior.
        
    Examples
    --------
    >>> import torch
    >>> from dfm_python.utils.loss import compute_masked_loss
    >>> 
    >>> # Example with MSE loss
    >>> reconstructed = torch.randn(10, 5)  # (T, N)
    >>> target = torch.randn(10, 5)
    >>> mask = torch.ones(10, 5, dtype=torch.bool)  # All observed
    >>> loss = compute_masked_loss(reconstructed, target, mask, loss_function='mse')
    >>> 
    >>> # Example with Huber loss and missing data
    >>> mask[0, 0] = False  # First element is missing
    >>> loss = compute_masked_loss(
    ...     reconstructed, target, mask, 
    ...     loss_function='huber', huber_delta=1.0
    ... )
    """
    # Ensure mask is boolean
    if mask.dtype != torch.bool:
        mask = mask.bool()
    
    # Apply mask to match original TensorFlow pattern:
    # 1. Zero out missing values in target (target_clean)
    # 2. Multiply prediction by mask (reconstructed_masked)
    # This order matches the original TensorFlow implementation's mask application
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    reconstructed_masked = reconstructed * mask
    diff = target_clean - reconstructed_masked
    
    if loss_function == 'huber':
        # Huber loss: more robust to outliers
        # For |diff| <= delta: 0.5 * diff^2 (quadratic)
        # For |diff| > delta: delta * (|diff| - 0.5 * delta) (linear)
        abs_diff = torch.abs(diff)
        loss_values = torch.where(
            abs_diff <= huber_delta,
            HUBER_QUADRATIC_COEFF * diff ** 2,
            huber_delta * (abs_diff - HUBER_QUADRATIC_COEFF * huber_delta)
        )
    else:
        # MSE loss (default)
        loss_values = diff ** 2
    
    # Normalize by total elements (target.numel()) to match TensorFlow MeanSquaredError behavior.
    # TensorFlow's MeanSquaredError divides by total elements, not just observed elements.
    # This ensures consistent scaling regardless of missing data pattern.
    loss = torch.sum(loss_values * mask) / (target.numel() + DEFAULT_EPSILON)
    
    return loss

