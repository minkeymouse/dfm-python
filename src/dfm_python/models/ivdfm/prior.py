"""Prior network p(η_t | u_t) for iVDFM, and the KL divergences KL(q || p) of the ELBO."""

from typing import Dict, Union, List, Optional
import torch
import torch.nn as nn
import numpy as np

from ...layer.mlp import MLP
from ...utils.errors import ConfigurationError

#: Output width of the prior network per factor, by distribution.
_PARAMS_PER_FACTOR = {
    'laplace': 2,      # location, log-scale
    'gaussian': 2,     # mean, log-variance
    'student_t': 3,    # location, log-scale, log-df
    'gamma': 2,        # shape, log-rate
    'beta': 2,         # log-alpha, log-beta
    'exponential': 1,  # log-rate
}


def _clamp(x: torch.Tensor) -> torch.Tensor:
    """Clamp a log-parameter to [-10, 10] so exp() downstream cannot overflow or underflow."""
    return torch.clamp(x, min=-10.0, max=10.0)


class iVDFMPriorNetwork(nn.Module):
    """Prior network p(η_t | u_t): an MLP from u_t to the natural parameters of the innovation prior.

    Parameters
    ----------
    aux_dim : int
        Dimension of u_t.
    latent_dim : int
        Dimension r of the innovations.
    hidden_dim : int or list of int, default 100
    n_hidden_layers : int, default 1
    activation : str, default 'lrelu'
        One of 'lrelu', 'relu', 'tanh', 'sigmoid'.
    slope : float, default 0.1
        Leaky ReLU slope.
    innovation_distribution : str, default 'laplace'
        One of 'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'.
    device : str or torch.device, optional
    seed : int, optional
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
        super().__init__()

        self.aux_dim = aux_dim
        self.latent_dim = latent_dim
        self.innovation_distribution = innovation_distribution

        if innovation_distribution not in _PARAMS_PER_FACTOR:
            raise ConfigurationError(
                f"Unsupported innovation_distribution: {innovation_distribution}. "
                f"Options: 'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'"
            )
        self.prior_network = MLP(
            input_dim=aux_dim,
            output_dim=latent_dim * _PARAMS_PER_FACTOR[innovation_distribution],
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            slope=slope,
            device=device,
            seed=seed,
        )

        if device is not None:
            self.to(device)

    def forward(self, u_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prior parameters for ``u_t`` of shape (aux_dim,), (batch, aux_dim) or (batch, T, aux_dim).

        Each value has the batch (and time) shape of ``u_t`` with a trailing r.
        The keys depend on the distribution: laplace ``location, log_scale``;
        gaussian ``mu, logvar``; student_t ``location, log_scale, log_df``;
        gamma ``shape, log_rate``; beta ``log_alpha, log_beta``; exponential
        ``log_rate``. Log-parameters are clamped to [-10, 10] (``log_df`` to
        [-2, 6]) and the gamma shape to >= 1e-6.
        """
        squeeze_output = u_t.dim() == 1
        needs_reshape = u_t.dim() == 3
        if squeeze_output:
            u_t = u_t.unsqueeze(0)
        elif needs_reshape:
            batch_size, T, aux_dim = u_t.shape
            u_t = u_t.reshape(-1, aux_dim)

        params = self.prior_network(u_t)
        r = self.latent_dim
        dist = self.innovation_distribution
        if dist == 'laplace':
            out = {'location': params[:, :r], 'log_scale': _clamp(params[:, r:])}
        elif dist == 'gaussian':
            out = {'mu': params[:, :r], 'logvar': _clamp(params[:, r:])}
        elif dist == 'student_t':
            out = {
                'location': params[:, :r],
                'log_scale': _clamp(params[:, r:2 * r]),
                'log_df': torch.clamp(params[:, 2 * r:], min=-2.0, max=6.0),
            }
        elif dist == 'gamma':
            out = {'shape': torch.clamp(params[:, :r], min=1e-6), 'log_rate': _clamp(params[:, r:])}
        elif dist == 'beta':
            out = {'log_alpha': _clamp(params[:, :r]), 'log_beta': _clamp(params[:, r:])}
        else:  # exponential
            out = {'log_rate': _clamp(params[:, :r])}

        if needs_reshape:
            out = {k: v.reshape(batch_size, T, r) for k, v in out.items()}
        elif squeeze_output:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out

    def get_prior_mean(self, u_t: torch.Tensor) -> torch.Tensor:
        """Mean (location) of the prior, shape (batch, r) or (r,)."""
        params = self.forward(u_t)
        dist = self.innovation_distribution
        if dist in ('laplace', 'student_t'):
            return params['location']
        if dist == 'gaussian':
            return params['mu']
        if dist == 'gamma':
            return params['shape'] / torch.exp(params['log_rate'])
        if dist == 'beta':
            alpha = torch.exp(params['log_alpha'])
            beta = torch.exp(params['log_beta'])
            return alpha / (alpha + beta + 1e-8)
        return 1.0 / (torch.exp(params['log_rate']) + 1e-8)  # exponential

    def sample(self, u_t: torch.Tensor) -> torch.Tensor:
        """Sample innovations from p(η_t | u_t); same batch/time shape as ``forward`` with a trailing r."""
        params = self.forward(u_t)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        dist = self.innovation_distribution

        if dist == 'laplace':
            location = params['location']
            scale = torch.exp(params['log_scale'])
            u_uniform = torch.rand_like(location, device=device, dtype=dtype) - 0.5
            return location + scale * torch.sign(u_uniform) * torch.log(
                1 - 2 * torch.abs(u_uniform) + 1e-8
            )
        if dist == 'gaussian':
            mu = params['mu']
            scale = torch.exp(0.5 * params['logvar'])
            return mu + scale * torch.randn_like(mu, device=device, dtype=dtype)
        if dist == 'student_t':
            location = params['location']
            scale = torch.exp(params['log_scale'])
            df = torch.exp(params['log_df'])
            z = torch.randn_like(location, device=device, dtype=dtype)
            chi2 = torch.distributions.Gamma(df / 2, 0.5).sample().to(device=device, dtype=dtype)  # shape of df
            return location + scale * z * torch.sqrt(df / (chi2 + 1e-8))
        if dist == 'gamma':
            rate = torch.exp(params['log_rate'])
            return torch.distributions.Gamma(params['shape'], rate).sample().to(
                device=device, dtype=dtype
            )
        if dist == 'beta':
            alpha = torch.exp(params['log_alpha'])
            beta = torch.exp(params['log_beta'])
            return torch.distributions.Beta(alpha, beta).sample().to(
                device=device, dtype=dtype
            )
        rate = torch.exp(params['log_rate'])  # exponential
        u = torch.rand_like(rate, device=device, dtype=dtype)
        return -torch.log(u + 1e-8) / rate


# KL divergences KL(q || p) for q ~ N(mu_q, exp(logvar_q)); every input is (batch, dim)
# and every result (batch,), summed over dim.

def compute_kl_gaussian_laplace(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    location_p: torch.Tensor,
    log_scale_p: torch.Tensor
) -> torch.Tensor:
    """KL(q || p) for p ~ Laplace(location_p, exp(log_scale_p)), clamped at 0.

    KL = E_q[log q] - E_q[log p]
       = -0.5*log(2π*var_q) - 0.5 + log(2*scale_p) + E_q[|x - location_p|]/scale_p
    with, for x ~ N(μ, σ²) and δ = μ - location_p,
    E[|x - location_p|] = σ*sqrt(2/π)*exp(-δ²/(2σ²)) + δ*erf(δ/(σ*sqrt(2))).
    """
    var_q = torch.exp(logvar_q)
    std_q = torch.sqrt(var_q + 1e-8)
    scale_p = torch.exp(log_scale_p)

    delta = mu_q - location_p
    delta_normalized = delta / (std_q + 1e-8)

    # 2*Φ(z) - 1 = erf(z / sqrt(2))
    sqrt_2 = np.sqrt(2.0)
    erf_term = torch.erf(delta_normalized / sqrt_2)
    exp_term = torch.exp(-0.5 * delta_normalized ** 2)
    first_term = std_q * np.sqrt(2 / np.pi) * exp_term
    second_term = delta * erf_term
    expected_abs = first_term + second_term

    kl = (
        -0.5 * (logvar_q + np.log(2 * np.pi)) - 0.5  # E_q[log q]
        + log_scale_p + np.log(2)  # +log(2*scale_p)
        + expected_abs / scale_p  # +E_q[|x - location_p|]/scale_p
    ).sum(dim=-1)

    return torch.clamp(kl, min=0.0)


def compute_kl_gaussian_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor
) -> torch.Tensor:
    """KL(q || p) for p ~ N(mu_p, exp(logvar_p)), in closed form."""
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
    """KL(q || p) for p ~ Gamma(shape_p, exp(log_rate_p)) on η = softplus(z), z ~ q.

    E_q[log p(softplus(z))] is approximated at softplus(mu_q).
    """
    var_q = torch.exp(logvar_q)
    rate_p = torch.exp(log_rate_p)
    softplus_mu = torch.nn.functional.softplus(mu_q)

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
    """KL(q || p) for p ~ Beta(exp(log_alpha_p), exp(log_beta_p)) on η = sigmoid(z), z ~ q.

    E_q[log p(sigmoid(z))] is approximated at sigmoid(mu_q).
    """
    var_q = torch.exp(logvar_q)
    alpha_p = torch.exp(log_alpha_p)
    beta_p = torch.exp(log_beta_p)
    sigmoid_mu = torch.sigmoid(mu_q)

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
    """KL(q || p) for p ~ Exponential(exp(log_rate_p)) on η = softplus(z), z ~ q.

    E_q[log p(softplus(z))] is approximated at softplus(mu_q).
    """
    var_q = torch.exp(logvar_q)
    rate_p = torch.exp(log_rate_p)
    softplus_mu = torch.nn.functional.softplus(mu_q)

    # Exponential log-pdf: log(rate) - rate*η
    kl = (
        0.5 * (logvar_q + np.log(2 * np.pi) + 1) +  # E_q[log q(z)]
        0.5 * (mu_q ** 2 + var_q) / var_q -  # Normalization
        log_rate_p +  # E_q[log p(softplus(z))]
        rate_p * softplus_mu
    ).sum(dim=-1)

    return kl
