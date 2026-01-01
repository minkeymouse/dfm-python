"""Autoencoder layers and utilities for DDFM.

This module contains PyTorch-based encoder networks used in the
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
from ..utils.errors import ConfigurationError, DataValidationError
from ..utils.common import ensure_numpy, sanitize_array
from ..numeric.stability import create_scaled_identity
from ..config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_ZERO_VALUE

_logger = get_logger(__name__)


def _get_decoder_layer(decoder: Any) -> "nn.Linear":
    """Extract the Linear layer from a decoder module.
    
    Handles multiple decoder architectures:
    - Linear decoder: decoder.decoder is the Linear layer
    - MLP decoder: decoder.output_layer is the final Linear layer
    - Direct Linear: decoder is directly a Linear layer
    
    Parameters
    ----------
    decoder : Any
        Decoder module (nn.Module, nn.Linear, or object with decoder/output_layer attributes)
        
    Returns
    -------
    nn.Linear
        The Linear layer from the decoder
        
    Raises
    ------
    DataValidationError
        If decoder does not have expected attributes or is not a Linear layer
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for decoder layer extraction")
    
    if hasattr(decoder, 'decoder'):
        # Linear decoder: decoder.decoder is the Linear layer
        return decoder.decoder
    elif hasattr(decoder, 'output_layer'):
        # MLP decoder: decoder.output_layer is the final Linear layer
        return decoder.output_layer
    elif isinstance(decoder, nn.Linear):
        # If decoder is directly a Linear layer
        return decoder
    else:
        raise DataValidationError(
            f"decoder must have 'decoder' (linear), 'output_layer' (MLP), or be a Linear layer. "
            f"Got decoder type: {type(decoder)}",
            details="Decoder must be a linear layer or MLP with 'decoder' or 'output_layer' attribute, or a Linear layer directly"
        )


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
                raise ConfigurationError(
                    f"Unknown activation: {activation}",
                    details="Activation must be one of: 'relu', 'sigmoid'"
                )
            
            # Import constants once at the start of initialization
            from ..config.constants import DEFAULT_XAVIER_GAIN, DEFAULT_OUTPUT_LAYER_GAIN, DEFAULT_ZERO_VALUE
            
            # Build layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layer = nn.Linear(prev_dim, hidden_dim)
                # Initialize weights using Xavier/Kaiming initialization for better training stability
                # Use Kaiming for ReLU, Xavier for tanh/sigmoid
                if activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
                # Initialize bias to small values
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
                self.layers.append(layer)
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer (linear, no activation)
            # Use smaller initialization for output layer to prevent large initial factors
            self.output_layer = nn.Linear(prev_dim, output_dim)
            nn.init.xavier_normal_(self.output_layer.weight, gain=DEFAULT_OUTPUT_LAYER_GAIN)  # Smaller gain for output
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, DEFAULT_ZERO_VALUE)
        
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
    
    
    class AutoencoderEncoder(BaseEncoder):
        """Autoencoder encoder wrapper for factor extraction.
        
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
                raise ImportError("PyTorch is required for AutoencoderEncoder")
            
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
        ) -> "AutoencoderEncoder":
            """Fit autoencoder encoder (no-op, training is done separately).
            
            Note: Autoencoder encoders are typically trained via autoencoder training
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
            self : AutoencoderEncoder
                Returns self for method chaining
            """
            # Autoencoder training is done separately via autoencoder training
            # This is just for interface compatibility
            self._is_fitted = True
            return self
        
        def encode(
            self,
            X: Union[np.ndarray, "torch.Tensor"],
            **kwargs
        ) -> "torch.Tensor":
            """Extract factors using trained autoencoder encoder.
            
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
                raise ImportError("PyTorch is required for AutoencoderEncoder")
            
            # Convert to tensor if needed
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=DEFAULT_TORCH_DTYPE)
            
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
                raise DataValidationError(
                    f"Expected 2D or 3D input, got {X.ndim}D",
                    details="Input must be 2D (T, K) or 3D (batch, T, K) for encoder forward pass"
                )
            
            return factors
        
        @property
        def encoder(self) -> Encoder:
            """Get the underlying PyTorch encoder module."""
            return self.encoder_module
    
    
else:
    # Placeholder classes when PyTorch is not available
    class Encoder:
        """Placeholder Encoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM. Install with: pip install dfm-python[deep]")
    
    class AutoencoderEncoder(BaseEncoder):
        """Placeholder AutoencoderEncoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AutoencoderEncoder. Install with: pip install dfm-python[deep]")
        
        def encode(self, X, **kwargs):
            raise ImportError("PyTorch is required for AutoencoderEncoder")


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
    
    # Handle both linear decoder and MLP decoder
    decoder_layer = _get_decoder_layer(decoder)
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = ensure_numpy(decoder_layer.weight.data)
    
    # Extract bias if present
    if decoder_layer.bias is not None:
        bias = ensure_numpy(decoder_layer.bias.data)
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
        C = sanitize_array(C)
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
    try:
        linear_layer = _get_decoder_layer(decoder)
    except DataValidationError:
        # Fallback: Try to find the last Linear layer in the module
        linear_layers = [m for m in decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise DataValidationError(
                "No Linear layer found in decoder",
                details="Decoder must contain at least one nn.Linear layer for parameter extraction"
            )
        linear_layer = linear_layers[-1]
    
    # Extract weight matrix: (output_dim x input_dim) = (N x m)
    weight = ensure_numpy(linear_layer.weight.data)  # N x m
    
    # Extract bias if present
    if has_bias and linear_layer.bias is not None:
        bias = ensure_numpy(linear_layer.bias.data)  # N,
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
        from ..config.constants import DEFAULT_IDENTITY_SCALE
        emission = np.hstack([
            weight,  # N x m (current factors)
            np.zeros((N, m)),  # N x m (lagged factors, zero contribution)
            create_scaled_identity(N, DEFAULT_IDENTITY_SCALE)  # N x N (idiosyncratic components)
        ])
    elif factor_order == 1:
        # VAR(1): x_t = [f_t, eps_t]
        # emission = [C, I]
        from ..config.constants import DEFAULT_IDENTITY_SCALE
        emission = np.hstack([
            weight,  # N x m (factors)
            create_scaled_identity(N, DEFAULT_IDENTITY_SCALE)  # N x N (idiosyncratic components)
        ])
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported. "
            f"Got factor_order={factor_order}"
        )
    
    return bias, emission

