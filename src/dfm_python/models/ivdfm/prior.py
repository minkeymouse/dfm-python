"""Prior network for iVDFM.

This module provides the prior network that maps auxiliary variables u_t
to prior parameters for innovations p(η_t | u_t), and KL divergence
computations between variational posteriors and priors.
"""

from typing import Dict, Union, List, Optional
import torch
import torch.nn as nn
import numpy as np

from ...layer.mlp import MLP
from ...utils.errors import ConfigurationError
from ...logger import get_logger

_logger = get_logger(__name__)


class iVDFMPriorNetwork(nn.Module):
    """Prior network for iVDFM: p(η_t | u_t).
    
    Maps auxiliary variables to prior parameters for innovations.
    Supports different distributions (Laplace, Gaussian, etc.) by outputting
    appropriate natural parameters for the exponential family.
    """
    
    def __init__(
        self,
        aux_dim: int,
        latent_dim: int,
        hidden_dim: Union[int, List[int]] = 100,
        n_hidden_layers: int = 1,
        activation: str = 'lrelu',
        slope: float = 0.1,
        innovation_distribution: str = 'laplace',
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize prior network.
        
        Parameters
        ----------
        aux_dim : int
            Dimension of auxiliary variable u_t
        latent_dim : int
            Dimension of latent factors/innovations (r)
        hidden_dim : Union[int, List[int]]
            Hidden layer dimension(s) for MLP network
        n_hidden_layers : int
            Number of hidden layers in MLP network
        activation : str
            Activation function ('lrelu', 'relu', 'tanh', 'sigmoid')
        slope : float
            Slope for leaky ReLU
        innovation_distribution : str
            Distribution for innovations ('laplace', 'gaussian', etc.)
        device : Optional[Union[str, torch.device]]
            Device to move model to
        seed : Optional[int]
            Random seed for weight initialization
        """
        super().__init__()
        
        self.aux_dim = aux_dim
        self.latent_dim = latent_dim
        self.innovation_distribution = innovation_distribution
        
        # Determine output dimension based on distribution
        if innovation_distribution == 'laplace':
            # Laplace: location and log-scale (2 * latent_dim)
            output_dim = latent_dim * 2
        elif innovation_distribution == 'gaussian':
            # Gaussian: mean and log-variance (2 * latent_dim)
            output_dim = latent_dim * 2
        elif innovation_distribution == 'student_t':
            # Student-t: location, log-scale, and log-df (3 * latent_dim)
            output_dim = latent_dim * 3
        elif innovation_distribution == 'gamma':
            # Gamma: shape and log-rate (2 * latent_dim)
            output_dim = latent_dim * 2
        elif innovation_distribution == 'beta':
            # Beta: log-concentration parameters α and β (2 * latent_dim)
            output_dim = latent_dim * 2
        elif innovation_distribution == 'exponential':
            # Exponential: log-rate (1 * latent_dim)
            output_dim = latent_dim * 1
        else:
            raise ConfigurationError(
                f"Unsupported innovation_distribution: {innovation_distribution}. "
                f"Options: 'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'"
            )
        
        # Prior network: maps auxiliary variable to prior parameters
        self.prior_network = MLP(
            input_dim=aux_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            slope=slope,
            device=device,
            seed=seed,
        )
        
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)
    
    def forward(self, u_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through prior network.
        
        Parameters
        ----------
        u_t : torch.Tensor
            Auxiliary variable.
            - (batch, aux_dim) or (aux_dim,) for a single time step
            - (batch, T, aux_dim) for a full sequence (vectorized over time)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing prior parameters (distribution-dependent):
            - For 'laplace': {'location': (batch, r), 'log_scale': (batch, r)}
            - For 'gaussian': {'mu': (batch, r), 'logvar': (batch, r)}
            - For 'student_t': {'location': (batch, r), 'log_scale': (batch, r), 'log_df': (batch, r)}
            - For 'gamma': {'shape': (batch, r), 'log_rate': (batch, r)}
            - For 'beta': {'log_alpha': (batch, r), 'log_beta': (batch, r)}
            - For 'exponential': {'log_rate': (batch, r)}
        """
        # Handle input shape
        # - 1D: (aux_dim,) -> (1, aux_dim)
        # - 2D: (batch, aux_dim)
        # - 3D: (batch, T, aux_dim) -> flatten to (batch*T, aux_dim) and reshape outputs back
        squeeze_output = False
        needs_reshape = False
        batch_size = None
        T = None
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(0)
            squeeze_output = True
        elif u_t.dim() == 3:
            batch_size, T, aux_dim = u_t.shape
            u_t = u_t.reshape(-1, aux_dim)  # (batch*T, aux_dim)
            needs_reshape = True

        # Get raw output from network
        params = self.prior_network(u_t)
        
        # Split and format based on distribution
        if self.innovation_distribution == 'laplace':
            location = params[:, :self.latent_dim]
            log_scale = params[:, self.latent_dim:]
            # Clamp to prevent exp overflow/underflow in downstream KL
            log_scale = torch.clamp(log_scale, min=-10.0, max=10.0)
            
            if needs_reshape:
                location = location.reshape(batch_size, T, self.latent_dim)
                log_scale = log_scale.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                location = location.squeeze(0)
                log_scale = log_scale.squeeze(0)
            
            return {'location': location, 'log_scale': log_scale}
        
        elif self.innovation_distribution == 'gaussian':
            mu = params[:, :self.latent_dim]
            logvar = params[:, self.latent_dim:]
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            
            if needs_reshape:
                mu = mu.reshape(batch_size, T, self.latent_dim)
                logvar = logvar.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                mu = mu.squeeze(0)
                logvar = logvar.squeeze(0)
            
            return {'mu': mu, 'logvar': logvar}
        
        elif self.innovation_distribution == 'student_t':
            location = params[:, :self.latent_dim]
            log_scale = params[:, self.latent_dim:2*self.latent_dim]
            log_df = params[:, 2*self.latent_dim:]
            log_scale = torch.clamp(log_scale, min=-10.0, max=10.0)
            # df must be positive; clamp log_df and let downstream exp/log1p handle
            log_df = torch.clamp(log_df, min=-2.0, max=6.0)
            
            if needs_reshape:
                location = location.reshape(batch_size, T, self.latent_dim)
                log_scale = log_scale.reshape(batch_size, T, self.latent_dim)
                log_df = log_df.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                location = location.squeeze(0)
                log_scale = log_scale.squeeze(0)
                log_df = log_df.squeeze(0)
            
            return {'location': location, 'log_scale': log_scale, 'log_df': log_df}
        
        elif self.innovation_distribution == 'gamma':
            # Gamma: shape (α) and log-rate (β)
            shape = torch.clamp(params[:, :self.latent_dim], min=1e-6)  # Ensure positive
            log_rate = params[:, self.latent_dim:]
            log_rate = torch.clamp(log_rate, min=-10.0, max=10.0)
            
            if needs_reshape:
                shape = shape.reshape(batch_size, T, self.latent_dim)
                log_rate = log_rate.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                shape = shape.squeeze(0)
                log_rate = log_rate.squeeze(0)
            
            return {'shape': shape, 'log_rate': log_rate}
        
        elif self.innovation_distribution == 'beta':
            # Beta: log-concentration parameters α and β
            log_alpha = params[:, :self.latent_dim]
            log_beta = params[:, self.latent_dim:]
            log_alpha = torch.clamp(log_alpha, min=-10.0, max=10.0)
            log_beta = torch.clamp(log_beta, min=-10.0, max=10.0)
            
            if needs_reshape:
                log_alpha = log_alpha.reshape(batch_size, T, self.latent_dim)
                log_beta = log_beta.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                log_alpha = log_alpha.squeeze(0)
                log_beta = log_beta.squeeze(0)
            
            return {'log_alpha': log_alpha, 'log_beta': log_beta}
        
        elif self.innovation_distribution == 'exponential':
            # Exponential: log-rate (λ)
            log_rate = params[:, :self.latent_dim]
            log_rate = torch.clamp(log_rate, min=-10.0, max=10.0)
            
            if needs_reshape:
                log_rate = log_rate.reshape(batch_size, T, self.latent_dim)
            elif squeeze_output:
                log_rate = log_rate.squeeze(0)
            
            return {'log_rate': log_rate}
        
        else:
            raise ConfigurationError(
                f"Unsupported innovation_distribution: {self.innovation_distribution}"
            )
    
    def get_prior_mean(self, u_t: torch.Tensor) -> torch.Tensor:
        """Get prior mean/location parameter.
        
        Parameters
        ----------
        u_t : torch.Tensor
            Auxiliary variable, shape (batch, aux_dim) or (aux_dim,)
        
        Returns
        -------
        torch.Tensor
            Prior mean/location, shape (batch, r) or (r,)
        """
        params = self.forward(u_t)
        
        if self.innovation_distribution == 'laplace':
            return params['location']
        elif self.innovation_distribution == 'gaussian':
            return params['mu']
        elif self.innovation_distribution == 'student_t':
            return params['location']
        elif self.innovation_distribution == 'gamma':
            # Gamma mean = shape / rate
            rate = torch.exp(params['log_rate'])
            return params['shape'] / rate
        elif self.innovation_distribution == 'beta':
            # Beta mean = α / (α + β)
            alpha = torch.exp(params['log_alpha'])
            beta = torch.exp(params['log_beta'])
            return alpha / (alpha + beta + 1e-8)
        elif self.innovation_distribution == 'exponential':
            # Exponential mean = 1 / rate
            rate = torch.exp(params['log_rate'])
            return 1.0 / (rate + 1e-8)
        else:
            raise ConfigurationError(
                f"Unsupported innovation_distribution: {self.innovation_distribution}"
            )


# ============================================================================
# KL Divergence Computations
# ============================================================================

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
    
    Correct formula:
    KL(q || p) = E_q[log q] - E_q[log p]
               = -0.5*log(2π*var_q) - 0.5 - log(2*scale_p) - E_q[|x - location_p|]/scale_p
    
    Where E_q[|x - location_p|] for x ~ N(mu_q, var_q) is computed using:
    E[|x - μ|] = σ * sqrt(2/π) * exp(-(μ - location)²/(2σ²)) 
                + (μ - location) * (2*Φ((μ - location)/σ) - 1)
    
    For numerical stability, we use an approximation when |μ - location| is large.
    
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
    std_q = torch.sqrt(var_q + 1e-8)  # Add small epsilon for numerical stability
    scale_p = torch.exp(log_scale_p)
    
    # Compute E_q[|x - location_p|] where x ~ N(mu_q, var_q)
    # Formula: E[|x - location|] = σ * sqrt(2/π) * exp(-(μ - location)²/(2σ²)) 
    #                             + (μ - location) * (2*Φ((μ - location)/σ) - 1)
    delta = mu_q - location_p
    delta_normalized = delta / (std_q + 1e-8)
    
    # Use PyTorch's erf for the cumulative distribution function
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # 2*Φ(z) - 1 = erf(z / sqrt(2))
    sqrt_2 = np.sqrt(2.0)
    erf_term = torch.erf(delta_normalized / sqrt_2)
    
    # First term: σ * sqrt(2/π) * exp(-delta²/(2σ²))
    exp_term = torch.exp(-0.5 * delta_normalized ** 2)
    first_term = std_q * np.sqrt(2 / np.pi) * exp_term
    
    # Second term: delta * erf(delta / (σ * sqrt(2)))
    second_term = delta * erf_term
    
    # E_q[|x - location_p|]
    expected_abs = first_term + second_term
    
    # KL(q || p) = E_q[log q] - E_q[log p]
    # E_q[log q] = -0.5*log(2π*var_q) - 0.5
    # E_q[log p] = -log(2*scale_p) - E_q[|x - location_p|]/scale_p
    # So: KL = -0.5*log(2π*var_q) - 0.5 + log(2*scale_p) + E_q[|x - location_p|]/scale_p
    kl = (
        -0.5 * (logvar_q + np.log(2 * np.pi)) - 0.5  # E_q[log q]
        + log_scale_p + np.log(2)  # +log(2*scale_p) (note: positive sign)
        + expected_abs / scale_p  # +E_q[|x - location_p|]/scale_p (note: positive sign)
    ).sum(dim=-1)
    
    # Ensure non-negative (should be, but clamp for numerical stability)
    kl = torch.clamp(kl, min=0.0)
    
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


def compute_kl_gaussian_gamma(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    shape_p: torch.Tensor,
    log_rate_p: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between Gaussian q and Gamma p.
    
    KL(q || p) where:
    - q ~ N(mu_q, exp(logvar_q)) is Gaussian variational posterior
    - p ~ Gamma(shape_p, exp(log_rate_p)) is Gamma prior
    
    Note: This KL divergence uses a softplus transformation to map Gaussian
    to positive domain: η = softplus(z) where z ~ q.
    
    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of Gaussian q, shape (batch, dim)
    logvar_q : torch.Tensor
        Log-variance of Gaussian q, shape (batch, dim)
    shape_p : torch.Tensor
        Shape parameter of Gamma p, shape (batch, dim)
    log_rate_p : torch.Tensor
        Log-rate parameter of Gamma p, shape (batch, dim)
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence, shape (batch,)
    """
    var_q = torch.exp(logvar_q)
    rate_p = torch.exp(log_rate_p)
    
    # Use softplus transformation: η = softplus(z) where z ~ N(mu_q, var_q)
    # Approximate E[softplus(z)] ≈ softplus(mu_q) for small var_q
    softplus_mu = torch.nn.functional.softplus(mu_q)
    
    # KL(q || p) = E_q[log q(z)] - E_q[log p(softplus(z))]
    # Gamma log-pdf: (shape-1)*log(η) - rate*η - lgamma(shape) + shape*log(rate)
    kl = (
        0.5 * (logvar_q + np.log(2 * np.pi) + 1) +  # E_q[log q(z)]
        0.5 * (mu_q ** 2 + var_q) / var_q -  # Normalization
        (shape_p - 1) * torch.log(softplus_mu + 1e-8) +  # E_q[log p(softplus(z))]
        rate_p * softplus_mu +
        torch.lgamma(shape_p) -
        shape_p * log_rate_p
    ).sum(dim=-1)
    
    return kl


def compute_kl_gaussian_beta(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    log_alpha_p: torch.Tensor,
    log_beta_p: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between Gaussian q and Beta p.
    
    KL(q || p) where:
    - q ~ N(mu_q, exp(logvar_q)) is Gaussian variational posterior
    - p ~ Beta(exp(log_alpha_p), exp(log_beta_p)) is Beta prior
    
    Note: Beta is defined on [0,1], so this uses a sigmoid transformation:
    η = sigmoid(z) where z ~ q.
    
    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of Gaussian q, shape (batch, dim)
    logvar_q : torch.Tensor
        Log-variance of Gaussian q, shape (batch, dim)
    log_alpha_p : torch.Tensor
        Log-concentration parameter α of Beta p, shape (batch, dim)
    log_beta_p : torch.Tensor
        Log-concentration parameter β of Beta p, shape (batch, dim)
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence, shape (batch,)
    """
    var_q = torch.exp(logvar_q)
    alpha_p = torch.exp(log_alpha_p)
    beta_p = torch.exp(log_beta_p)
    
    # Use sigmoid transformation: η = sigmoid(z) where z ~ N(mu_q, var_q)
    sigmoid_mu = torch.sigmoid(mu_q)
    
    # KL(q || p) = E_q[log q(z)] - E_q[log p(sigmoid(z))]
    # Beta log-pdf: (α-1)*log(η) + (β-1)*log(1-η) - B(α,β)
    kl = (
        0.5 * (logvar_q + np.log(2 * np.pi) + 1) +  # E_q[log q(z)]
        0.5 * (mu_q ** 2 + var_q) / var_q -  # Normalization
        (alpha_p - 1) * torch.log(sigmoid_mu + 1e-8) -  # E_q[log p(sigmoid(z))]
        (beta_p - 1) * torch.log(1 - sigmoid_mu + 1e-8) +
        torch.lgamma(alpha_p) + torch.lgamma(beta_p) -
        torch.lgamma(alpha_p + beta_p)
    ).sum(dim=-1)
    
    return kl


def compute_kl_gaussian_exponential(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    log_rate_p: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between Gaussian q and Exponential p.
    
    KL(q || p) where:
    - q ~ N(mu_q, exp(logvar_q)) is Gaussian variational posterior
    - p ~ Exponential(exp(log_rate_p)) is Exponential prior
    
    Note: Exponential is defined on [0,∞), so this uses a softplus transformation:
    η = softplus(z) where z ~ q.
    
    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of Gaussian q, shape (batch, dim)
    logvar_q : torch.Tensor
        Log-variance of Gaussian q, shape (batch, dim)
    log_rate_p : torch.Tensor
        Log-rate parameter of Exponential p, shape (batch, dim)
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence, shape (batch,)
    """
    var_q = torch.exp(logvar_q)
    rate_p = torch.exp(log_rate_p)
    
    # Use softplus transformation: η = softplus(z) where z ~ N(mu_q, var_q)
    softplus_mu = torch.nn.functional.softplus(mu_q)
    
    # KL(q || p) = E_q[log q(z)] - E_q[log p(softplus(z))]
    # Exponential log-pdf: log(rate) - rate*η
    kl = (
        0.5 * (logvar_q + np.log(2 * np.pi) + 1) +  # E_q[log q(z)]
        0.5 * (mu_q ** 2 + var_q) / var_q -  # Normalization
        log_rate_p +  # E_q[log p(softplus(z))]
        rate_p * softplus_mu
    ).sum(dim=-1)
    
    return kl
