"""MLP (Multi-Layer Perceptron) decoder network for DDFM."""

import torch
import torch.nn as nn
from typing import List, Optional

from ..utils.errors import ConfigurationError
from ..config.constants import DEFAULT_XAVIER_GAIN, DEFAULT_OUTPUT_LAYER_GAIN, DEFAULT_ZERO_VALUE


class MLPDecoder(nn.Module):
    """MLP decoder network for DDFM."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        use_bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [output_dim]
        
        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
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
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=DEFAULT_OUTPUT_LAYER_GAIN)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = f
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
        return self.output_layer(x)

