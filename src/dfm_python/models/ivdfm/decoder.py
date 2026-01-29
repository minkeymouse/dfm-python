"""Decoder for iVDFM.

This module provides the decoder that maps latent factors f_t to
observations y_t.
"""

from typing import Union, List, Optional
import torch
import torch.nn as nn

from ..base import BaseDecoder
from ...layer.mlp import MLP
from ...utils.errors import ConfigurationError
from ...logger import get_logger

_logger = get_logger(__name__)


class iVDFMDecoder(BaseDecoder):
    """Decoder for iVDFM: g(f_t) → y_t.
    
    Maps latent factors to observations using an MLP network.
    """
    
    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        hidden_dim: Union[int, List[int]] = 200,
        n_hidden_layers: int = 2,
        activation: str = 'lrelu',
        slope: float = 0.1,
        decoder_var: float = 0.01,
        use_layer_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize decoder.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of latent factors (r)
        data_dim : int
            Dimension of observed data (N)
        hidden_dim : Union[int, List[int]]
            Hidden layer dimension(s) for MLP network
        n_hidden_layers : int
            Number of hidden layers in MLP network
        activation : str
            Activation function ('lrelu', 'relu', 'tanh', 'sigmoid')
        slope : float
            Slope for leaky ReLU
        decoder_var : float
            Decoder variance (observation noise). Can be learnable or fixed.
        use_layer_norm : bool
            Whether to use layer normalization in MLP network
        device : Optional[Union[str, torch.device]]
            Device to move model to
        seed : Optional[int]
            Random seed for weight initialization
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.decoder_var = decoder_var
        
        # Decoder network: maps factors to observations
        self.decoder_network = MLP(
            input_dim=latent_dim,
            output_dim=data_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            slope=slope,
            use_layer_norm=use_layer_norm,
            device=device,
            seed=seed,
        )
        
        # Decoder variance (can be learnable parameter or fixed)
        if isinstance(decoder_var, float):
            # Fixed variance
            self.register_buffer('decoder_variance', torch.tensor(decoder_var))
        else:
            # Learnable variance (if passed as Parameter)
            self.decoder_variance = decoder_var
        
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Parameters
        ----------
        f : torch.Tensor
            Latent factors, shape (batch, r) or (batch, T, r) or (T, r) or (r,)
        
        Returns
        -------
        torch.Tensor
            Reconstructed observations, shape matches input except last dimension
            is data_dim instead of latent_dim
        """
        return self.decoder_network(f)
    
    def forward_with_variance(
        self,
        f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both mean and variance.
        
        Parameters
        ----------
        f : torch.Tensor
            Latent factors, shape (batch, r) or (batch, T, r) or (T, r) or (r,)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (mean, variance) where:
            - mean: reconstructed observations, shape matches input except last dim
            - variance: decoder variance, broadcastable to mean shape
        """
        mean = self.forward(f)
        
        # Get variance (handle both buffer and parameter cases)
        if isinstance(self.decoder_variance, torch.Tensor):
            var = self.decoder_variance
        else:
            # Cache tensor creation - avoid repeated allocations
            if not hasattr(self, '_decoder_var_tensor') or self._decoder_var_tensor.device != mean.device:
                self._decoder_var_tensor = torch.tensor(
                    self.decoder_var, 
                    device=mean.device, 
                    dtype=mean.dtype
                )
            var = self._decoder_var_tensor
        
        # Broadcast variance to match mean shape (optimized: use expand instead of unsqueeze loop)
        if var.dim() < mean.dim():
            # Add leading dimensions: (1, 1, ...) to match mean.ndim
            # Use view + expand for efficient broadcasting without copying
            shape = [1] * (mean.ndim - var.ndim) + list(var.shape)
            var = var.view(*shape).expand_as(mean)
        
        return mean, var
