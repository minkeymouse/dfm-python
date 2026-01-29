"""Loss computation utilities for factor models.

This module provides reusable loss functions for training factor models,
including support for missing data masking, robust loss functions, and
variational inference losses (ELBO).
"""

import torch
import numpy as np
from typing import Literal, Dict, Tuple, Optional, Any
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


# KL divergence functions moved to models.ivdfm.prior module
# Import them lazily to avoid circular imports
def _get_kl_functions():
    """Lazy import of KL functions to avoid circular imports."""
    from ..models.ivdfm.prior import (
        compute_kl_gaussian_laplace,
        compute_kl_gaussian_gaussian,
        compute_kl_gaussian_gamma,
        compute_kl_gaussian_beta,
        compute_kl_gaussian_exponential,
    )
    return {
        'compute_kl_gaussian_laplace': compute_kl_gaussian_laplace,
        'compute_kl_gaussian_gaussian': compute_kl_gaussian_gaussian,
        'compute_kl_gaussian_gamma': compute_kl_gaussian_gamma,
        'compute_kl_gaussian_beta': compute_kl_gaussian_beta,
        'compute_kl_gaussian_exponential': compute_kl_gaussian_exponential,
    }

# Create module-level aliases that will be populated on first use
_kl_functions = None

def _ensure_kl_functions():
    """Ensure KL functions are loaded."""
    global _kl_functions
    if _kl_functions is None:
        _kl_functions = _get_kl_functions()
    return _kl_functions

# Create function aliases that lazily load
def compute_kl_gaussian_laplace(*args, **kwargs):
    return _ensure_kl_functions()['compute_kl_gaussian_laplace'](*args, **kwargs)

def compute_kl_gaussian_gaussian(*args, **kwargs):
    return _ensure_kl_functions()['compute_kl_gaussian_gaussian'](*args, **kwargs)

def compute_kl_gaussian_gamma(*args, **kwargs):
    return _ensure_kl_functions()['compute_kl_gaussian_gamma'](*args, **kwargs)

def compute_kl_gaussian_beta(*args, **kwargs):
    return _ensure_kl_functions()['compute_kl_gaussian_beta'](*args, **kwargs)

def compute_kl_gaussian_exponential(*args, **kwargs):
    return _ensure_kl_functions()['compute_kl_gaussian_exponential'](*args, **kwargs)


def compute_reconstruction_loss_gaussian(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: float = 1.0
) -> torch.Tensor:
    """Compute Gaussian reconstruction loss: -log p(y | f).
    
    For Gaussian observation model: p(y | f) = N(y; g(f), σ²)
    Negative log-likelihood: -log p(y | f) = 0.5 * log(2πσ²) + 0.5 * (y - g(f))²/σ²
    
    Parameters
    ----------
    y_true : torch.Tensor
        True observations, shape (batch, T, N) or (batch, N)
    y_pred : torch.Tensor
        Predicted observations, shape (batch, T, N) or (batch, N)
    variance : float, default 1.0
        Observation variance σ²
        
    Returns
    -------
    loss : torch.Tensor
        Scalar reconstruction loss (negative log-likelihood)
    """
    # Negative log-likelihood
    recon_loss = 0.5 * (
        np.log(2 * np.pi * variance) +
        ((y_true - y_pred) ** 2) / variance
    )
    
    # Sum over data dimensions, mean over batch and time
    if y_true.dim() == 3:
        # (batch, T, N) -> sum over N, mean over batch and T
        recon_loss = recon_loss.sum(dim=-1).mean()
    else:
        # (batch, N) -> sum over N, mean over batch
        recon_loss = recon_loss.sum(dim=-1).mean()
    
    return recon_loss


def compute_elbo_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    encoder_params: list,
    prior_params: list,
    innovation_distribution: str = 'laplace',
    decoder_variance: float = 1.0,
    beta_kl: float = 1.0,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Evidence Lower Bound (ELBO) loss for variational inference.
    
    ELBO = E[log p(y | f)] - β * Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
    
    The ELBO is maximized, so we return the negative ELBO for minimization.
    β (beta_kl) controls the weight of the KL term. β < 1 reduces KL pressure.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True observations, shape (batch, T, N)
    y_pred : torch.Tensor
        Predicted observations, shape (batch, T, N)
    encoder_params : list
        List of dictionaries with encoder parameters for each time step.
        Each dict should contain 'mu' and 'logvar' for variational posterior q.
    prior_params : list
        List of dictionaries with prior parameters for each time step.
        Format depends on innovation_distribution:
        - 'laplace': {'location', 'log_scale'}
        - 'gaussian': {'mu', 'logvar'}
        - 'student_t': {'location', 'log_scale', 'log_df'}
        - 'gamma': {'shape', 'log_rate'}
        - 'beta': {'log_alpha', 'log_beta'}
        - 'exponential': {'log_rate'}
    innovation_distribution : str, default 'laplace'
        Distribution type for innovations: 'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'
    decoder_variance : float, default 1.0
        Variance of Gaussian observation model
    beta_kl : float, default 1.0
        Weight for KL term in ELBO. β < 1 reduces KL pressure, allowing better reconstruction.
    reduction : str, default 'mean'
        Reduction method: 'mean' or 'sum'
        
    Returns
    -------
    Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        (elbo_loss, loss_dict) where:
        - elbo_loss: Negative ELBO (to minimize)
        - loss_dict: Dictionary with component losses:
            - 'elbo': total ELBO loss
            - 'reconstruction': reconstruction term
            - 'kl': KL divergence term
    """
    batch_size, T, _ = y_true.shape
    
    # Reconstruction term: E[log p(y_t | f_t)]
    recon_loss = compute_reconstruction_loss_gaussian(
        y_true, y_pred, variance=decoder_variance
    )
    
    # KL divergence terms: Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
    # Vectorize: stack all time steps and compute KL for all at once
    mu_q_all = torch.stack([encoder_params[t]['mu'] for t in range(T)], dim=1)  # (batch, T, dim)
    logvar_q_all = torch.stack([encoder_params[t]['logvar'] for t in range(T)], dim=1)  # (batch, T, dim)
    
    # Stack prior parameters
    if innovation_distribution == 'laplace':
        location_p_all = torch.stack([prior_params[t]['location'] for t in range(T)], dim=1)  # (batch, T, dim)
        log_scale_p_all = torch.stack([prior_params[t]['log_scale'] for t in range(T)], dim=1)  # (batch, T, dim)
        # Reshape for vectorized KL: (batch*T, dim)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        location_p_flat = location_p_all.view(-1, location_p_all.shape[-1])
        log_scale_p_flat = log_scale_p_all.view(-1, log_scale_p_all.shape[-1])
        kl_all = compute_kl_gaussian_laplace(mu_q_flat, logvar_q_flat, location_p_flat, log_scale_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    elif innovation_distribution == 'gaussian':
        mu_p_all = torch.stack([prior_params[t]['mu'] for t in range(T)], dim=1)  # (batch, T, dim)
        logvar_p_all = torch.stack([prior_params[t]['logvar'] for t in range(T)], dim=1)  # (batch, T, dim)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        mu_p_flat = mu_p_all.view(-1, mu_p_all.shape[-1])
        logvar_p_flat = logvar_p_all.view(-1, logvar_p_all.shape[-1])
        kl_all = compute_kl_gaussian_gaussian(mu_q_flat, logvar_q_flat, mu_p_flat, logvar_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    elif innovation_distribution == 'student_t':
        location_p_all = torch.stack([prior_params[t]['location'] for t in range(T)], dim=1)
        log_scale_p_all = torch.stack([prior_params[t]['log_scale'] for t in range(T)], dim=1)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        location_p_flat = location_p_all.view(-1, location_p_all.shape[-1])
        log_scale_p_flat = log_scale_p_all.view(-1, log_scale_p_all.shape[-1])
        kl_all = compute_kl_gaussian_laplace(mu_q_flat, logvar_q_flat, location_p_flat, log_scale_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    elif innovation_distribution == 'gamma':
        shape_p_all = torch.stack([prior_params[t]['shape'] for t in range(T)], dim=1)
        log_rate_p_all = torch.stack([prior_params[t]['log_rate'] for t in range(T)], dim=1)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        shape_p_flat = shape_p_all.view(-1, shape_p_all.shape[-1])
        log_rate_p_flat = log_rate_p_all.view(-1, log_rate_p_all.shape[-1])
        kl_all = compute_kl_gaussian_gamma(mu_q_flat, logvar_q_flat, shape_p_flat, log_rate_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    elif innovation_distribution == 'beta':
        log_alpha_p_all = torch.stack([prior_params[t]['log_alpha'] for t in range(T)], dim=1)
        log_beta_p_all = torch.stack([prior_params[t]['log_beta'] for t in range(T)], dim=1)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        log_alpha_p_flat = log_alpha_p_all.view(-1, log_alpha_p_all.shape[-1])
        log_beta_p_flat = log_beta_p_all.view(-1, log_beta_p_all.shape[-1])
        kl_all = compute_kl_gaussian_beta(mu_q_flat, logvar_q_flat, log_alpha_p_flat, log_beta_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    elif innovation_distribution == 'exponential':
        log_rate_p_all = torch.stack([prior_params[t]['log_rate'] for t in range(T)], dim=1)
        mu_q_flat = mu_q_all.view(-1, mu_q_all.shape[-1])
        logvar_q_flat = logvar_q_all.view(-1, logvar_q_all.shape[-1])
        log_rate_p_flat = log_rate_p_all.view(-1, log_rate_p_all.shape[-1])
        kl_all = compute_kl_gaussian_exponential(mu_q_flat, logvar_q_flat, log_rate_p_flat)
        kl_loss = kl_all.view(batch_size, T)  # (batch, T)
    else:
        raise NotImplementedError(
            f"KL divergence for {innovation_distribution} not implemented"
        )
    
    if reduction == 'mean':
        kl_loss = kl_loss.mean()  # Mean over batch and time
    else:
        kl_loss = kl_loss.sum()  # Sum over batch and time
    
    # ELBO = E[log p(y|f)] - β * KL
    # recon_loss = -E[log p(y|f)] (negative log-likelihood, positive value)
    # So: ELBO = -recon_loss - β * KL
    # For minimization, return -ELBO = recon_loss + β * KL
    elbo = recon_loss + beta_kl * kl_loss
    
    loss_dict = {
        'elbo': elbo,
        'reconstruction': recon_loss,
        'kl': kl_loss,
    }
    
    return elbo, loss_dict

# Note: compute_ivdfm_elbo was removed as redundant.
# The iVDFM model's elbo() method directly calls compute_elbo_loss() after forward pass.

