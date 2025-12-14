"""Variational Autoencoder (VAE) layers and utilities for DDFM.

This module contains PyTorch-based encoder and decoder networks used in the
Deep Dynamic Factor Model (DDFM), along with training and conversion utilities.
"""

import numpy as np
from typing import Optional, Tuple, List, Any, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = None
    nn = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None
    optim = None

from .base import BaseEncoder
from ..logger import get_logger

_logger = get_logger(__name__)


if _has_torch:
    class Encoder(nn.Module):
        """Nonlinear encoder network for DDFM.
        
        Maps observed variables X_t to latent factors f_t using a multi-layer perceptron.
        This is the PyTorch module implementation.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            activation: str = 'tanh',
            use_batch_norm: bool = True,
        ):
            """Initialize encoder network.
            
            Parameters
            ----------
            input_dim : int
                Number of input features (number of series)
            hidden_dims : List[int]
                List of hidden layer dimensions
            output_dim : int
                Number of factors (output dimension)
            activation : str
                Activation function ('tanh', 'relu', 'sigmoid')
            use_batch_norm : bool
                Whether to use batch normalization
            """
            super().__init__()
            
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
            
            # Build layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layer = nn.Linear(prev_dim, hidden_dim)
                # Initialize weights using Xavier/Kaiming initialization for better training stability
                # Use Kaiming for ReLU, Xavier for tanh/sigmoid
                if activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
                # Initialize bias to small values
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
                self.layers.append(layer)
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer (linear, no activation)
            # Use smaller initialization for output layer to prevent large initial factors
            self.output_layer = nn.Linear(prev_dim, output_dim)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)  # Smaller gain for output
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0.0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through encoder.
            
            Parameters
            ----------
            x : torch.Tensor
                Input data (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Encoded factors (batch_size x output_dim)
            """
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
            
            # Output layer (linear, no activation)
            x = self.output_layer(x)
            return x
    
    
    class VAEEncoder(BaseEncoder):
        """Variational Autoencoder encoder wrapper for factor extraction.
        
        This class wraps the PyTorch Encoder module to provide the BaseEncoder interface.
        It handles training and encoding for DDFM.
        
        Parameters
        ----------
        n_components : int
            Number of factors to extract
        input_dim : int
            Number of input features (number of series)
        hidden_dims : List[int]
            List of hidden layer dimensions
        activation : str, default 'tanh'
            Activation function ('tanh', 'relu', 'sigmoid')
        use_batch_norm : bool, default True
            Whether to use batch normalization
        """
        
        def __init__(
            self,
            n_components: int,
            input_dim: int,
            hidden_dims: List[int],
            activation: str = 'tanh',
            use_batch_norm: bool = True,
        ):
            super().__init__(n_components)
            
            if not _has_torch:
                raise ImportError("PyTorch is required for VAEEncoder")
            
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.activation = activation
            self.use_batch_norm = use_batch_norm
            
            # Create the PyTorch encoder module
            self.encoder_module = Encoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=n_components,
                activation=activation,
                use_batch_norm=use_batch_norm,
            )
            
            # Training state
            self._is_fitted = False
        
        def fit(
            self,
            X: Union[np.ndarray, "torch.Tensor"],
            **kwargs
        ) -> "VAEEncoder":
            """Fit VAE encoder (no-op, training is done separately).
            
            Note: VAE encoders are typically trained via autoencoder training
            (encoder + decoder) before being used for factor extraction.
            This method satisfies the BaseEncoder interface but does nothing.
            
            Parameters
            ----------
            X : np.ndarray or torch.Tensor
                Training data (T x N). Not used, training is done separately.
            **kwargs
                Additional parameters (ignored)
                
            Returns
            -------
            self : VAEEncoder
                Returns self for method chaining
            """
            # VAE training is done separately via autoencoder training
            # This is just for interface compatibility
            self._is_fitted = True
            return self
        
        def encode(
            self,
            X: Union[np.ndarray, "torch.Tensor"],
            **kwargs
        ) -> "torch.Tensor":
            """Extract factors using trained VAE encoder.
            
            Parameters
            ----------
            X : np.ndarray or torch.Tensor
                Observed data (T x N) or (batch_size x T x N)
            **kwargs
                Additional parameters (ignored)
                
            Returns
            -------
            factors : torch.Tensor
                Extracted factors (T x n_components) or (batch_size x T x n_components)
            """
            if not _has_torch:
                raise ImportError("PyTorch is required for VAEEncoder")
            
            # Convert to tensor if needed
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            
            # Handle different input shapes
            original_shape = X.shape
            if X.ndim == 3:
                # (batch_size, T, N) -> (batch_size * T, N)
                batch_size, T, N = X.shape
                X = X.view(batch_size * T, N)
                factors = self.encoder_module(X)
                # Reshape back: (batch_size * T, n_components) -> (batch_size, T, n_components)
                factors = factors.view(batch_size, T, self.n_components)
            elif X.ndim == 2:
                # (T, N) -> (T, n_components)
                factors = self.encoder_module(X)
            else:
                raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
            
            return factors
        
        @property
        def encoder(self) -> Encoder:
            """Get the underlying PyTorch encoder module."""
            return self.encoder_module
    
    
    class Decoder(nn.Module):
        """Linear decoder network for DDFM.
        
        Maps latent factors f_t back to observed variables X_t using a linear transformation.
        This preserves interpretability and allows Kalman filtering.
        """
        
        def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
            """Initialize linear decoder.
            
            Parameters
            ----------
            input_dim : int
                Number of factors (input dimension)
            output_dim : int
                Number of series (output dimension)
            use_bias : bool
                Whether to use bias term
            """
            super().__init__()
            self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        def forward(self, f: torch.Tensor) -> torch.Tensor:
            """Forward pass through decoder.
            
            Parameters
            ----------
            f : torch.Tensor
                Factors (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Reconstructed observations (batch_size x output_dim)
            """
            return self.decoder(f)
    
    
else:
    # Placeholder classes when PyTorch is not available
    class Encoder:
        """Placeholder Encoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM. Install with: pip install dfm-python[deep]")
    
    class VAEEncoder(BaseEncoder):
        """Placeholder VAEEncoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for VAEEncoder. Install with: pip install dfm-python[deep]")
        
        def encode(self, X, **kwargs):
            raise ImportError("PyTorch is required for VAEEncoder")
    
    class Decoder:
        """Placeholder Decoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM. Install with: pip install dfm-python[deep]")


def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation matrix C and bias from trained decoder.
    
    Parameters
    ----------
    decoder
        Trained PyTorch decoder module
        
    Returns
    -------
    C : np.ndarray
        Loading matrix (N x m) from decoder weights
    bias : np.ndarray
        Bias terms (N,)
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for DDFM")
    
    decoder_layer = decoder.decoder
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = decoder_layer.weight.data.cpu().numpy()
    
    # Extract bias if present
    if decoder_layer.bias is not None:
        bias = decoder_layer.bias.data.cpu().numpy()
    else:
        bias = np.zeros(weight.shape[0])
    
    # C should be (N x m) for consistency with DFMResult
    # Decoder weight is already (N x m), so no transpose needed
    C = weight
    
    # Check for NaN in extracted C matrix (indicates numerical instability during training)
    if np.any(np.isnan(C)):
        nan_count = np.sum(np.isnan(C))
        nan_ratio = nan_count / C.size
        from ..logger import get_logger
        _logger = get_logger(__name__)
        _logger.warning(
            f"extract_decoder_params: C matrix contains {nan_count}/{C.size} NaN values ({nan_ratio:.1%}). "
            f"This indicates the decoder weights contain NaN, likely due to numerical instability during training. "
            f"Possible causes: (1) learning rate too high, (2) gradient explosion, (3) unstable training."
        )
        # Replace NaN with zeros to prevent further issues
        C = np.nan_to_num(C, nan=0.0)
        _logger.warning("Replaced NaN values in C matrix with zeros.")
    
    return C, bias


def convert_decoder_to_numpy(
    decoder: Any,  # nn.Module when torch is available
    has_bias: bool = True,
    factor_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PyTorch decoder to NumPy arrays for state-space model.
    
    Extracts weights and biases from a PyTorch decoder (typically nn.Linear)
    and constructs the observation matrix (emission matrix) for the state-space
    representation. Supports VAR(1) and VAR(2) factor dynamics.
    
    Parameters
    ----------
    decoder : nn.Module
        PyTorch decoder model (typically a single Linear layer or a model with
        a final Linear layer accessible via `.decoder` attribute)
    has_bias : bool
        Whether the decoder has a bias term
    factor_order : int
        Lag order for common factors. Only VAR(1) and VAR(2) are supported.
        Higher orders will raise NotImplementedError.
        
    Returns
    -------
    bias : np.ndarray
        Bias terms (N,) where N is the number of series
    emission : np.ndarray
        Emission matrix (N x state_dim) for state-space model.
        For VAR(1): [C, I] where C is loading matrix and I is identity for idio
        For VAR(2): [C, zeros, I] where zeros are for lagged factors
        
    Notes
    -----
    The emission matrix structure depends on the state vector:
    - VAR(1): x_t = [f_t, eps_t], emission = [C, I]
    - VAR(2): x_t = [f_t, f_{t-1}, eps_t], emission = [C, zeros, I]
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for decoder conversion")
    
    # Extract the actual Linear layer
    if hasattr(decoder, 'decoder'):
        # If decoder is wrapped in a class (e.g., Decoder class)
        linear_layer = decoder.decoder
    elif isinstance(decoder, nn.Linear):
        # If decoder is directly a Linear layer
        linear_layer = decoder
    else:
        # Try to find the last Linear layer
        linear_layers = [m for m in decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("No Linear layer found in decoder")
        linear_layer = linear_layers[-1]
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = linear_layer.weight.data.cpu().numpy()  # N x m
    
    # Extract bias if present
    if has_bias and linear_layer.bias is not None:
        bias = linear_layer.bias.data.cpu().numpy()  # N,
    else:
        bias = np.zeros(weight.shape[0])  # N,
    
    # Construct emission matrix for state-space model
    N, m = weight.shape
    
    if factor_order == 2:
        # VAR(2): x_t = [f_t, f_{t-1}, eps_t]
        # emission = [C, zeros, I]
        # where C is the loading matrix (N x m)
        C = weight.T  # m x N, but we need N x m for emission
        # Actually, emission should be N x (m + m + N) = N x (2m + N)
        emission = np.hstack([
            weight,  # N x m (current factors)
            np.zeros((N, m)),  # N x m (lagged factors, zero contribution)
            np.eye(N)  # N x N (idiosyncratic components)
        ])
    elif factor_order == 1:
        # VAR(1): x_t = [f_t, eps_t]
        # emission = [C, I]
        emission = np.hstack([
            weight,  # N x m (factors)
            np.eye(N)  # N x N (idiosyncratic components)
        ])
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported. "
            f"Got factor_order={factor_order}"
        )
    
    return bias, emission

