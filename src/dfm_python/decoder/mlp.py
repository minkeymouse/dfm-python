"""MLP (Multi-Layer Perceptron) decoder network for DDFM.

Maps latent factors f_t back to observed variables X_t using a nonlinear MLP.
This provides more expressive power than linear decoder but loses interpretability.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = None
    nn = None

try:
    import torch
    import torch.nn as nn
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None


if _has_torch:
    class MLPDecoder(nn.Module):
        """MLP (Multi-Layer Perceptron) decoder network for DDFM.
        
        Maps latent factors f_t back to observed variables X_t using a nonlinear MLP.
        This provides more expressive power than linear decoder but loses interpretability.
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Optional[List[int]] = None,
            activation: str = 'relu',
            use_batch_norm: bool = False,
            use_bias: bool = True,
        ):
            """Initialize MLP decoder.
            
            Parameters
            ----------
            input_dim : int
                Number of factors (input dimension)
            output_dim : int
                Number of series (output dimension)
            hidden_dims : List[int], optional
                Hidden layer dimensions. Default: [output_dim] (single hidden layer)
            activation : str, default 'relu'
                Activation function ('relu', 'tanh', 'sigmoid')
            use_batch_norm : bool, default False
                Whether to use batch normalization
            use_bias : bool, default True
                Whether to use bias term
            """
            super().__init__()
            
            if hidden_dims is None:
                hidden_dims = [output_dim]  # Default: single hidden layer with same size as output
            
            self.layers = nn.ModuleList()
            self.use_batch_norm = use_batch_norm
            self.batch_norms = nn.ModuleList() if use_batch_norm else None
            
            # Activation function
            if activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Build hidden layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layer = nn.Linear(prev_dim, hidden_dim, bias=use_bias)
                # Initialize weights
                if activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
                self.layers.append(layer)
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer (linear, no activation)
            self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)  # Smaller gain for output
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0.0)
        
        def forward(self, f: "torch.Tensor") -> "torch.Tensor":
            """Forward pass through MLP decoder.
            
            Parameters
            ----------
            f : torch.Tensor
                Factors (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Reconstructed observations (batch_size x output_dim)
            """
            x = f
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
            
            # Output layer (linear, no activation)
            x = self.output_layer(x)
            return x
else:
    # Placeholder class when PyTorch is not available
    class MLPDecoder:
        """Placeholder MLPDecoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM. Install with: pip install dfm-python[deep]")

