"""Loss computation utilities for factor models.

This module provides reusable loss functions for training factor models,
including support for missing data masking, robust loss functions, and
variational inference losses (ELBO).
"""

import torch
import numpy as np
from typing import Literal, Dict, Tuple, Optional
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


def compute_kl_gaussian_laplace(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    location_p: torch.Tensor,
    log_scale_p: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between Gaussian q and Laplace p.
    
    KL(q || p) where:
    - q ~ N(mu_q, exp(logvar_q)) is Gaussian variational posterior
    - p ~ Laplace(location_p, exp(log_scale_p)) is Laplace prior
    
    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of Gaussian q, shape (batch, dim)
    logvar_q : torch.Tensor
        Log-variance of Gaussian q, shape (batch, dim)
    location_p : torch.Tensor
        Location parameter of Laplace p, shape (batch, dim)
    log_scale_p : torch.Tensor
        Log-scale parameter of Laplace p, shape (batch, dim)
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence, shape (batch,)
    """
    var_q = torch.exp(logvar_q)
    scale_p = torch.exp(log_scale_p)
    
    # KL(q || p) = E_q[log q] - E_q[log p]
    # For Gaussian q and Laplace p:
    # KL = 0.5 * (log(2*pi*var_q) + 1) - log(2*scale_p) - |mu_q - location_p|/scale_p
    #     + 0.5 * var_q / scale_p^2
    
    # Simplified version (more stable):
    # KL ≈ 0.5 * (logvar_q - 2*log_scale_p + var_q/scale_p^2 + (mu_q - location_p)^2/scale_p^2 - 1)
    kl = 0.5 * (
        logvar_q - 2 * log_scale_p +
        (var_q + (mu_q - location_p) ** 2) / (scale_p ** 2) - 1
    ).sum(dim=-1)
    
    return kl


def compute_kl_gaussian_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between two Gaussians.
    
    KL(q || p) where:
    - q ~ N(mu_q, exp(logvar_q))
    - p ~ N(mu_p, exp(logvar_p))
    
    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of Gaussian q, shape (batch, dim)
    logvar_q : torch.Tensor
        Log-variance of Gaussian q, shape (batch, dim)
    mu_p : torch.Tensor
        Mean of Gaussian p, shape (batch, dim)
    logvar_p : torch.Tensor
        Log-variance of Gaussian p, shape (batch, dim)
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence, shape (batch,)
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    # KL(q || p) = 0.5 * (logvar_p - logvar_q + var_q/var_p + (mu_q - mu_p)^2/var_p - 1)
    kl = 0.5 * (
        logvar_p - logvar_q +
        var_q / var_p +
        (mu_q - mu_p) ** 2 / var_p - 1
    ).sum(dim=-1)
    
    return kl


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
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Evidence Lower Bound (ELBO) loss for variational inference.
    
    ELBO = E[log p(y | f)] - Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
    
    The ELBO is maximized, so we return the negative ELBO for minimization.
    
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
    innovation_distribution : str, default 'laplace'
        Distribution type for innovations: 'laplace' or 'gaussian'
    decoder_variance : float, default 1.0
        Variance of Gaussian observation model
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
    kl_losses = []
    for t in range(T):
        mu_q = encoder_params[t]['mu']  # (batch, dim)
        logvar_q = encoder_params[t]['logvar']  # (batch, dim)
        prior_p = prior_params[t]
        
        if innovation_distribution == 'laplace':
            location_p = prior_p['location']
            log_scale_p = prior_p['log_scale']
            kl_t = compute_kl_gaussian_laplace(
                mu_q, logvar_q, location_p, log_scale_p
            )
        elif innovation_distribution == 'gaussian':
            mu_p = prior_p['mu']
            logvar_p = prior_p['logvar']
            kl_t = compute_kl_gaussian_gaussian(
                mu_q, logvar_q, mu_p, logvar_p
            )
        else:
            raise NotImplementedError(
                f"KL divergence for {innovation_distribution} not implemented"
            )
        
        kl_losses.append(kl_t)
    
    # Stack and reduce over time
    kl_loss = torch.stack(kl_losses, dim=1)  # (batch, T)
    
    if reduction == 'mean':
        kl_loss = kl_loss.mean()  # Mean over batch and time
    else:
        kl_loss = kl_loss.sum()  # Sum over batch and time
    
    # ELBO = recon_loss - kl_loss (we want to maximize ELBO)
    # Return negative ELBO for minimization
    elbo = -(recon_loss - kl_loss)
    
    loss_dict = {
        'elbo': elbo,
        'reconstruction': recon_loss,
        'kl': kl_loss,
    }
    
    return elbo, loss_dict

