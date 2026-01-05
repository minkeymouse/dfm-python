"""Base decoder class for DDFM.

This module provides the base decoder interface and factory function for creating
decoders based on architecture specifications.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..config.constants import DEFAULT_ZERO_VALUE


class BaseDecoder(nn.Module, ABC):
    """Base class for all DDFM decoders.
    
    All decoders must implement the forward method to map from latent factors
    to reconstructed observations.
    """
    
    @abstractmethod
    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        """Map latent factors to reconstructed observations.
        
        Parameters
        ----------
        factors : torch.Tensor
            Latent factors (batch_size x latent_dim)
            
        Returns
        -------
        torch.Tensor
            Reconstructed observations (batch_size x output_dim)
        """
        pass


def build_decoder(
    latent_dim: int,
    output_dim: int,
    decoder_size: Optional[Tuple[int, ...]] = None,
    activation: str = 'relu',
    use_bias: bool = True,
    seed: Optional[int] = None
) -> nn.Module:
    """Build decoder network based on architecture specification.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent factors (input to decoder)
    output_dim : int
        Dimension of output (number of target series)
    decoder_size : tuple, optional
        Hidden layer sizes. If None, creates linear decoder (single layer).
        If provided, creates MLP decoder with specified hidden layers.
    activation : str, default 'relu'
        Activation function ('relu' or 'tanh')
    use_bias : bool, default True
        Whether to use bias in the final output layer
    seed : int, optional
        Random seed for initialization
        
    Returns
    -------
    nn.Module
        Decoder network (nn.Sequential)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    decoder_layers = []
    
    if decoder_size:
        # MLP decoder with hidden layers
        prev_dim = latent_dim
        for hidden_dim in decoder_size:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            prev_dim = hidden_dim
        
        # Output layer - input dim is the last hidden layer size
        decoder_layers.append(nn.Linear(prev_dim, output_dim, bias=use_bias))
    else:
        # Linear decoder (single layer)
        decoder_layers.append(nn.Linear(latent_dim, output_dim, bias=use_bias))
    
    # Initialize weights with Xavier (GlorotNormal) matching original
    for layer in decoder_layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
    
    return nn.Sequential(*decoder_layers)

