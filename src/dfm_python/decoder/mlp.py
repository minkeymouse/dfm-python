"""MLP (Multi-Layer Perceptron) decoder network for DDFM."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .base import Decoder
from ..utils.errors import ConfigurationError
from ..config.constants import DEFAULT_XAVIER_GAIN, DEFAULT_OUTPUT_LAYER_GAIN, DEFAULT_ZERO_VALUE, DEFAULT_CLEAN_NAN
from ..logger import get_logger

_logger = get_logger(__name__)


class MLPDecoder(Decoder):
    """MLP decoder network for DDFM."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        use_bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [output_dim]
        
        self.layers = nn.ModuleList()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ConfigurationError(f"Unknown activation: {activation}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim, bias=use_bias)
            if activation == 'relu':
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=DEFAULT_OUTPUT_LAYER_GAIN)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = f
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
    
    def extract_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract observation matrix C and bias from decoder output layer.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (weight, bias) where weight is (output_dim, input_dim) and bias is (output_dim,)
        """
        weight = self.output_layer.weight.data.cpu().numpy()
        bias = self.output_layer.bias.data.cpu().numpy() if self.output_layer.bias is not None else np.zeros(weight.shape[0])
        
        if np.any(np.isnan(weight)):
            _logger.warning("MLPDecoder: C matrix contains NaN values. Replacing with zeros.")
            weight = np.nan_to_num(weight, nan=DEFAULT_CLEAN_NAN)
        
        return weight, bias
    
    def get_last_linear_layer(self) -> nn.Linear:
        """Get decoder's last Linear layer (output layer).
        
        For MLPDecoder, this is the output_layer.
        
        Returns
        -------
        nn.Linear
            Decoder's output Linear layer
        """
        return self.output_layer
    
    def get_intermediate(self) -> nn.Sequential:
        """Get decoder intermediate layers (all except last layer).
        
        Used for last_neurons extraction (second-to-last layer output).
        Returns a Sequential module containing all hidden layers with activations.
        
        Returns
        -------
        nn.Sequential
            Decoder intermediate layers (all hidden layers with activations)
            
        Raises
        ------
        ConfigurationError
            If decoder has no intermediate layers (only output layer)
        """
        if len(self.layers) == 0:
            raise ConfigurationError(
                "MLPDecoder has no intermediate layers for last_neurons extraction. "
                "Decoder must have at least one hidden layer."
            )
        
        # Build Sequential with layers and activations
        intermediate_modules = []
        for layer in self.layers:
            intermediate_modules.append(layer)
            intermediate_modules.append(self.activation)
        
        return nn.Sequential(*intermediate_modules)

