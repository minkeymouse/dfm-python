"""Multi-Layer Perceptron (MLP) for common usage across models.

This module provides a flexible MLP implementation that can be used
for encoders, decoders, and other neural network components.
"""

from numbers import Number
from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.constants import (
    DEFAULT_XAVIER_GAIN,
    DEFAULT_ZERO_VALUE,
)
from ..logger import get_logger
from ..utils.errors import ConfigurationError

_logger = get_logger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture.
    
    Supports various activation functions, custom hidden dimensions,
    and proper weight initialization.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, List[int]] = 200,
        n_layers: int = 3,
        activation: Union[str, List[str]] = 'relu',
        slope: float = 0.1,
        use_bias: bool = True,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize MLP.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        hidden_dim : Union[int, List[int]]
            Hidden layer dimension(s). If int, all hidden layers use this dimension.
            If list, specifies dimension for each hidden layer.
        n_layers : int
            Total number of layers (including input and output)
        activation : Union[str, List[str]]
            Activation function(s). Options: 'relu', 'lrelu', 'tanh', 'sigmoid', 'none'.
            If str, all layers use this activation. If list, specifies activation per layer.
        slope : float
            Slope for leaky ReLU (negative_slope parameter)
        use_bias : bool
            Whether to use bias in linear layers
        dropout : float
            Dropout probability (0.0 = no dropout)
        use_batch_norm : bool
            Whether to use batch normalization after each hidden layer
        device : Optional[Union[str, torch.device]]
            Device to move model to (None = no move)
        seed : Optional[int]
            Random seed for weight initialization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Validate and set hidden dimensions
        if isinstance(hidden_dim, Number):
            if n_layers < 2:
                raise ValueError(f"n_layers must be >= 2 when hidden_dim is int, got {n_layers}")
            self.hidden_dim = [int(hidden_dim)] * (n_layers - 1)
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != n_layers - 1:
                raise ValueError(
                    f"hidden_dim list length ({len(hidden_dim)}) must equal n_layers - 1 ({n_layers - 1})"
                )
            self.hidden_dim = [int(d) for d in hidden_dim]
        else:
            raise ValueError(f'Wrong argument type for hidden_dim: {type(hidden_dim)}')
        
        # Validate and set activations
        if isinstance(activation, str):
            self.activation = [activation] * (n_layers - 1)
        elif isinstance(activation, list):
            if len(activation) != n_layers - 1:
                raise ValueError(
                    f"activation list length ({len(activation)}) must equal n_layers - 1 ({n_layers - 1})"
                )
            self.activation = activation
        else:
            raise ValueError(f'Wrong argument type for activation: {type(activation)}')
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Build activation functions
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x, s=slope: F.leaky_relu(x, negative_slope=s))
            elif act == 'relu':
                self._act_f.append(F.relu)
            elif act == 'tanh':
                self._act_f.append(torch.tanh)
            elif act == 'sigmoid':
                self._act_f.append(torch.sigmoid)
            elif act == 'none' or act is None:
                self._act_f.append(lambda x: x)
            else:
                raise ValueError(f'Incorrect activation: {act}. Options: relu, lrelu, tanh, sigmoid, none')
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        if n_layers == 1:
            # Single layer: input -> output
            layer = nn.Linear(input_dim, output_dim, bias=use_bias)
            self._init_linear(layer)
            self.layers.append(layer)
        else:
            # First hidden layer: input -> hidden[0]
            layer = nn.Linear(input_dim, self.hidden_dim[0], bias=use_bias)
            self._init_linear(layer)
            self.layers.append(layer)
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))
            
            # Hidden layers: hidden[i-1] -> hidden[i]
            for i in range(1, len(self.hidden_dim)):
                layer = nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i], bias=use_bias)
                self._init_linear(layer)
                self.layers.append(layer)
                
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i]))
            
            # Output layer: hidden[-1] -> output
            layer = nn.Linear(self.hidden_dim[-1], output_dim, bias=use_bias)
            self._init_linear(layer, is_output=True)
            self.layers.append(layer)
        
        # Dropout layer
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Move to device if specified
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)
    
    @staticmethod
    def _init_linear(layer: nn.Linear, is_output: bool = False) -> None:
        """Initialize linear layer weights and bias.
        
        Parameters
        ----------
        layer : nn.Linear
            Linear layer to initialize
        is_output : bool
            Whether this is the output layer (uses different initialization)
        """
        # Weight initialization
        if is_output:
            # Output layer: Xavier normal with default gain
            nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
        else:
            # Hidden layers: Xavier normal
            nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
        
        # Bias initialization
        if layer.bias is not None:
            nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim) or (batch, *, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, output_dim) or (batch, *, output_dim)
        """
        # Handle multi-dimensional input (e.g., sequences)
        original_shape = x.shape
        if x.dim() > 2:
            # Flatten all dimensions except last
            x = x.view(-1, x.shape[-1])
            needs_reshape = True
        else:
            needs_reshape = False
        
        # Forward through layers
        h = x
        n_hidden = len(self.layers) - 1  # Number of hidden layers
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Apply activation, batch norm, and dropout (except on output layer)
            if i < n_hidden:
                # Activation
                h = self._act_f[i](h)
                
                # Batch normalization
                if self.use_batch_norm and self.batch_norms is not None:
                    h = self.batch_norms[i](h)
                
                # Dropout
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)
        
        # Reshape if needed
        if needs_reshape:
            output_shape = list(original_shape[:-1]) + [self.output_dim]
            h = h.view(*output_shape)
        
        return h
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.
        
        Returns
        -------
        int
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> dict:
        """Get information about MLP architecture.
        
        Returns
        -------
        dict
            Dictionary with architecture information
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dim,
            'n_layers': self.n_layers,
            'activations': self.activation,
            'use_bias': self.use_bias,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'num_parameters': self.get_num_parameters(),
        }
