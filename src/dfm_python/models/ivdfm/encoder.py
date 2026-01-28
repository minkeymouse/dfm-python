"""Innovation encoder for iVDFM.

This module provides the innovation encoder that encodes innovations η_t
from current observation y_t and auxiliary variable u_t, matching the paper:
q(η_t | y_t, u_t).
"""

from typing import Tuple, Union, List, Optional
import torch
import torch.nn as nn

from ..base import BaseEncoder
from ...layer.mlp import MLP
from ...utils.errors import ConfigurationError
from ...logger import get_logger

_logger = get_logger(__name__)


class iVDFMInnovationEncoder(BaseEncoder):
    """Innovation encoder for iVDFM: q(η_t | y_t, u_t).
    
    Encodes innovations from current observation and auxiliary variable.
    Matches the paper's specification where the variational posterior
    conditions only on the current time step, not the full sequence.
    Uses two separate MLP networks to output mean and log-variance of the
    variational posterior over innovations.
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        aux_dim: int,
        hidden_dim: Union[int, List[int]] = 200,
        n_layers: int = 3,
        activation: str = 'lrelu',
        slope: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize innovation encoder.
        
        Parameters
        ----------
        data_dim : int
            Dimension of observed data (N)
        latent_dim : int
            Dimension of latent factors/innovations (r)
        aux_dim : int
            Dimension of auxiliary variable u_t
        hidden_dim : Union[int, List[int]]
            Hidden layer dimension(s) for MLP networks
        n_layers : int
            Number of layers in MLP networks
        activation : str
            Activation function ('lrelu', 'relu', 'tanh', 'sigmoid')
        slope : float
            Slope for leaky ReLU
        device : Optional[Union[str, torch.device]]
            Device to move model to
        seed : Optional[int]
            Random seed for weight initialization
        """
        super().__init__()
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        
        # Input dimension: current observation + auxiliary variable
        encoder_input_dim = data_dim + aux_dim
        
        # Mean network: outputs mean of innovation posterior
        self.mu_network = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            slope=slope,
            device=device,
            seed=seed,
        )
        
        # Log-variance network: outputs log-variance of innovation posterior
        # Use different seed to ensure different initialization
        logvar_seed = seed + 1 if seed is not None else None
        self.logvar_network = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            slope=slope,
            device=device,
            seed=logvar_seed,
        )
        
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)
    
    def forward(
        self,
        y_t: torch.Tensor,
        u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through innovation encoder.
        
        Parameters
        ----------
        y_t : torch.Tensor
            Current observation, shape (batch, N) or (batch, T, N)
            If (batch, T, N), processes all time steps in parallel.
        u_t : torch.Tensor
            Auxiliary variable, shape (batch, aux_dim) or (batch, T, aux_dim)
            Must match y_t's time dimension.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of innovation posterior:
            - mu: shape (batch, r) or (batch, T, r)
            - logvar: shape (batch, r) or (batch, T, r)
        """
        # Handle input shapes
        if y_t.dim() == 3:
            # (batch, T, N) -> process all time steps
            batch_size, T, N = y_t.shape
            y_flat = y_t.reshape(-1, N)  # (batch*T, N)
            u_flat = u_t.reshape(-1, self.aux_dim)  # (batch*T, aux_dim)
            needs_reshape = True
        elif y_t.dim() == 2:
            # (batch, N) -> single time step
            y_flat = y_t
            u_flat = u_t
            needs_reshape = False
        else:
            raise ValueError(f"y_t must be 2D (batch, N) or 3D (batch, T, N), got {y_t.dim()}D")
        
        # Concatenate: (batch*T, N + aux_dim) or (batch, N + aux_dim)
        xu = torch.cat([y_flat, u_flat], dim=1)
        
        # Get variational parameters
        mu = self.mu_network(xu)
        logvar = self.logvar_network(xu)
        
        # Clamp log-variance for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # Reshape back if needed
        if needs_reshape:
            mu = mu.reshape(batch_size, T, -1)
            logvar = logvar.reshape(batch_size, T, -1)
        
        return mu, logvar
    
    def sample(
        self,
        y_t: torch.Tensor,
        u_t: torch.Tensor,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample innovations using reparameterization trick.
        
        Parameters
        ----------
        y_t : torch.Tensor
            Current observation, shape (batch, N) or (batch, T, N)
        u_t : torch.Tensor
            Auxiliary variable, shape (batch, aux_dim) or (batch, T, aux_dim)
        return_params : bool
            Whether to return encoder parameters (mu, logvar) along with sample
        
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            If return_params=False: sampled innovations, shape (batch, r) or (batch, T, r)
            If return_params=True: (eta, mu, logvar) tuple
        """
        mu, logvar = self.forward(y_t, u_t)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        eta = mu + eps * std
        
        if return_params:
            return eta, mu, logvar
        else:
            return eta
