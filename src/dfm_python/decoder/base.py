"""Base decoder class for DDFM.

This module defines the abstract base class for decoder implementations.
All decoders must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class Decoder(nn.Module, ABC):
    """Abstract base class for decoder networks in DDFM.
    
    All decoder implementations must inherit from this class and implement
    the abstract methods for extracting parameters and debugging information.
    """
    
    @abstractmethod
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Parameters
        ----------
        f : torch.Tensor
            Latent factors (T, num_factors)
            
        Returns
        -------
        torch.Tensor
            Reconstructed observations (T, output_dim)
        """
        pass
    
    @abstractmethod
    def extract_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract observation matrix C and bias from decoder.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (weight, bias) where weight is (output_dim, input_dim) and bias is (output_dim,)
        """
        pass
    
    @abstractmethod
    def get_last_linear_layer(self) -> nn.Linear:
        """Get decoder's last Linear layer (output layer).
        
        Used for extracting observation matrix H for state-space model construction.
        
        Returns
        -------
        nn.Linear
            Decoder's last Linear layer (output layer)
            
        Raises
        ------
        ConfigurationError
            If no Linear layer found in decoder
        """
        pass
    
    def get_intermediate(self) -> Optional[nn.Module]:
        """Get decoder intermediate layers (all except last layer).
        
        Used for last_neurons extraction (second-to-last layer output).
        For single-layer decoders, returns None.
        
        Returns
        -------
        Optional[nn.Module]
            Decoder intermediate layers (all except last), or None if decoder has only one layer
        """
        return None

