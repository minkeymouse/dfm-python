"""Encoder and decoder utilities for DDFM.

This module contains DDFM-specific encoder networks and decoder parameter extraction utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Any, Optional

from ..logger import get_logger
from ..utils.errors import ConfigurationError, DataValidationError
from ..utils.common import ensure_numpy, sanitize_array
from ..numeric.stability import create_scaled_identity
from ..config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_ZERO_VALUE, DEFAULT_FACTOR_ORDER

_logger = get_logger(__name__)


def _get_decoder_layer(decoder: Any) -> nn.Linear:
    """Extract the Linear layer from a decoder module."""
    if hasattr(decoder, 'decoder'):
        return decoder.decoder
    elif hasattr(decoder, 'output_layer'):
        return decoder.output_layer
    elif isinstance(decoder, nn.Linear):
        return decoder
    else:
        raise DataValidationError(
            f"decoder must have 'decoder', 'output_layer', or be a Linear layer. Got: {type(decoder)}"
        )


class Encoder(nn.Module):
    """Nonlinear encoder network for DDFM."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'tanh',
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
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
        
        from ..config.constants import DEFAULT_XAVIER_GAIN, DEFAULT_OUTPUT_LAYER_GAIN, DEFAULT_ZERO_VALUE
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            # Use Xavier (GlorotNormal) for all layers, matching original TensorFlow DDFM
            nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
            self.layers.append(layer)
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight, gain=DEFAULT_OUTPUT_LAYER_GAIN)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
        return self.output_layer(x)


class Autoencoder(nn.Module):
    """Autoencoder combining encoder and decoder with noise injection for DDFM.
    
    Implements Algorithm 1 step 3: ε_t^(mc) ~ N(0, Σ_ε)
    Following original DDFM pattern: x_sim_den = x_sim_den - eps_draws
    """
    
    def __init__(self, encoder: Encoder, decoder: Any, num_series: Optional[int] = None, Sigma_eps: Optional[torch.Tensor] = None, seed: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Noise injection for DDFM denoising training
        self._noise_samples: Optional[torch.Tensor] = None
        self._num_series = num_series
        if num_series is not None:
            from ..config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_EPSILON
            if Sigma_eps is None:
                Sigma_eps = torch.ones(num_series, dtype=DEFAULT_TORCH_DTYPE) * DEFAULT_EPSILON
            elif Sigma_eps.ndim == 0:
                Sigma_eps = torch.ones(num_series, dtype=DEFAULT_TORCH_DTYPE) * Sigma_eps.item()
            elif Sigma_eps.shape[0] != num_series:
                raise ValueError(f"Sigma_eps must have shape ({num_series},) or be scalar, got {Sigma_eps.shape}")
            self.register_buffer('Sigma_eps', Sigma_eps)
            
            if seed is not None:
                self._generator = torch.Generator()
                self._generator.manual_seed(seed)
            else:
                self._generator = None
        else:
            self.register_buffer('Sigma_eps', None)
            self._generator = None
    
    def generate_noise_samples(self, n_mc_samples: int, T: int, device: Optional[torch.device] = None) -> None:
        """Pre-generate noise samples: ε_t^(mc) ~ N(0, Σ_ε) for all MC samples."""
        if self._num_series is None:
            return
        device = device or self.Sigma_eps.device
        Sigma_eps = self.Sigma_eps.to(device)
        from ..config.constants import DEFAULT_TORCH_DTYPE
        # Move generator to device if it exists and is on a different device
        generator = self._generator
        if generator is not None and generator.device != device:
            generator = torch.Generator(device=device)
            if hasattr(self._generator, 'initial_seed'):
                generator.manual_seed(self._generator.initial_seed())
        noise = torch.randn(
            n_mc_samples, T, self._num_series,
            device=device,
            dtype=DEFAULT_TORCH_DTYPE,
            generator=generator
        ) * Sigma_eps[None, None, :]
        self._noise_samples = noise
    
    def inject_noise(self, x: torch.Tensor, sample_idx: Optional[int] = None, start_idx: Optional[int] = None, end_idx: Optional[int] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject noise by subtracting epsilon: y_t^(mc) = ỹ_t - ε_t^(mc).
        
        Following original DDFM pattern: x_sim_den = x_sim_den - eps_draws
        """
        if not training or self._noise_samples is None:
            return x, torch.ones_like(x, dtype=torch.bool)
        
        if sample_idx is None:
            raise ValueError("sample_idx is required when using pre-sampled noise")
        
        noise_samples = self._noise_samples.to(x.device)
        noise = noise_samples[sample_idx]
        if start_idx is not None and end_idx is not None:
            noise = noise[start_idx:end_idx]
        
        if noise.shape[0] < x.shape[0]:
            pad = torch.zeros(x.shape[0] - noise.shape[0], noise.shape[1], device=x.device, dtype=noise.dtype)
            noise = torch.cat([noise, pad], dim=0)
        noise = noise[:x.shape[0]]
        
        return x - noise, torch.ones_like(x, dtype=torch.bool)
    
    def update_Sigma_eps(self, Sigma_eps: torch.Tensor) -> None:
        """Update Sigma_eps (Σ_ε) buffer."""
        if self._num_series is None:
            return
        from ..config.constants import DEFAULT_TORCH_DTYPE
        if Sigma_eps.ndim == 0:
            Sigma_eps = torch.ones(self._num_series, dtype=DEFAULT_TORCH_DTYPE, device=Sigma_eps.device) * Sigma_eps.item()
        elif Sigma_eps.shape[0] != self._num_series:
            raise ValueError(f"Sigma_eps must have shape ({self._num_series},) or be scalar, got {Sigma_eps.shape}")
        self.Sigma_eps = Sigma_eps.to(self.Sigma_eps.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, factors: torch.Tensor) -> torch.Tensor:
        return self.decoder(factors)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict (inference mode) - matches original TensorFlow autoencoder.predict().
        
        Parameters
        ----------
        x : torch.Tensor
            Input data (T x N) or (batch_size, T, N)
            
        Returns
        -------
        torch.Tensor
            Reconstructed output (same shape as input)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1,
        batch_size: int = 100,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
        verbose: int = 0
    ) -> None:
        """Fit autoencoder on data - matches original TensorFlow autoencoder.fit().
        
        This method trains the autoencoder for specified number of epochs, matching
        the original TensorFlow DDFM pattern where autoencoder.fit() is called
        separately for each MC sample with epochs=1. Uses DataLoader for efficient batching.
        
        Original TensorFlow usage (DDFM/models/ddfm.py line 246):
            self.autoencoder.fit(x_sim_den[i, :, :], self.z_actual, epochs=1, batch_size=self.batch_size, verbose=0)
        
        Parameters
        ----------
        x : torch.Tensor
            Input data (T x N) - corrupted/noisy input
        y : torch.Tensor
            Target data (T x N) - clean target for reconstruction
        epochs : int, default 1
            Number of epochs to train (original DDFM uses epochs=1 per MC sample)
        batch_size : int, default 100
            Batch size for training (number of time steps per batch).
            Keras splits data into batches and processes each batch separately.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. If None, creates Adam optimizer with default lr=0.001
        loss_fn : torch.nn.Module, optional
            Loss function. If None, uses MSE loss
        mask : torch.Tensor, optional
            Missing data mask (T x N), True where data is missing
        verbose : int, default 0
            Verbosity level (0 = silent, 1 = print loss)
            
        Notes
        -----
        - This method sets model to training mode and trains for specified epochs
        - After training, model remains in training mode (unlike predict() which uses eval mode)
        - Matches original TensorFlow behavior where fit() is called in a loop for each MC sample
        - Uses DataLoader for efficient batching (sequential processing maintained with num_workers=0)
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()
        
        self.train()
        
        # Create dataset and dataloader for efficient batching
        from ..dataset.ddfm_dataset import AutoencoderDataset
        from ..dataset.dataloader import create_autoencoder_dataloader
        
        # Prepare mask: if None, create all-True mask
        if mask is None:
            T, N = y.shape
            mask = torch.ones(T, N, dtype=torch.bool, device=y.device)
        
        dataset = AutoencoderDataset(x, y, mask)
        dataloader = create_autoencoder_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # Sequential processing required
            pin_memory=torch.cuda.is_available()
        )
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_y, batch_mask in dataloader:
                # Forward pass
                reconstructed = self.forward(batch_x)
                
                # Compute loss - masked_loss_fn always requires 3 args
                if batch_mask.shape == reconstructed.shape:
                    loss = loss_fn(reconstructed, batch_y, batch_mask)
                else:
                    # Create all-True mask if shape mismatch
                    all_true_mask = torch.ones_like(reconstructed, dtype=torch.bool)
                    loss = loss_fn(reconstructed, batch_y, all_true_mask)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if verbose > 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                _logger.info(f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.6f}")


def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation matrix C and bias from trained decoder."""
    decoder_layer = _get_decoder_layer(decoder)
    weight = ensure_numpy(decoder_layer.weight.data)
    bias = ensure_numpy(decoder_layer.bias.data) if decoder_layer.bias is not None else np.zeros(weight.shape[0])
    
    C = weight
    
    if np.any(np.isnan(C)):
        nan_count = np.sum(np.isnan(C))
        nan_ratio = nan_count / C.size
        _logger.warning(
            f"extract_decoder_params: C matrix contains {nan_count}/{C.size} NaN values ({nan_ratio:.1%}). "
            f"Replacing with zeros."
        )
        C = sanitize_array(C)
    
    return C, bias


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for DDFM (matching original TensorFlow structure).
    
    This is a minimal autoencoder that combines encoder and decoder with
    simple training interface matching the original TensorFlow DDFM.
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Initialize simple autoencoder.
        
        Parameters
        ----------
        encoder : nn.Module
            Encoder network
        decoder : nn.Module
            Decoder network
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        return self.decoder(self.encoder(x))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict (inference mode) - matches original TensorFlow autoencoder.predict()."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def fit(
        self,
        x: Any,
        y: Any,
        epochs: int = 1,
        batch_size: int = 100,
        learning_rate: float = 0.005,
        optimizer_type: str = 'Adam',
        decay_learning_rate: bool = True,
        verbose: int = 0
    ) -> None:
        """Fit autoencoder on data - matches original TensorFlow autoencoder.fit().
        
        Parameters
        ----------
        x : array-like
            Input data (T x N) - corrupted/noisy input
        y : array-like
            Target data (T x N) - clean target for reconstruction
        epochs : int, default 1
            Number of epochs to train
        batch_size : int, default 100
            Batch size for training
        learning_rate : float, default 0.005
            Learning rate for optimizer
        optimizer_type : str, default 'Adam'
            Optimizer type ('Adam' or 'SGD')
        decay_learning_rate : bool, default True
            Whether to use learning rate decay
        verbose : int, default 0
            Verbosity level (0 = silent)
        """
        from ..utils.common import ensure_tensor
        from ..config.constants import DEFAULT_TORCH_DTYPE
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) if optimizer_type == 'Adam' else torch.optim.SGD(self.parameters(), lr=learning_rate)
        if decay_learning_rate:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        
        self.train()
        x_tensor = ensure_tensor(x, dtype=DEFAULT_TORCH_DTYPE)
        y_tensor = ensure_tensor(y, dtype=DEFAULT_TORCH_DTYPE)
        
        for epoch in range(epochs):
            for i in range(0, len(x_tensor), batch_size):
                batch_x = x_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = self.forward(batch_x)
                # Masked loss for missing data
                mask = ~torch.isnan(batch_y)
                if mask.any():
                    loss = nn.functional.mse_loss(pred[mask], batch_y[mask])
                else:
                    loss = nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            if decay_learning_rate:
                scheduler.step()


def convert_decoder_to_numpy(
    decoder: Any,
    has_bias: bool = True,
    factor_order: int = DEFAULT_FACTOR_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PyTorch decoder to NumPy arrays for state-space model."""
    try:
        linear_layer = _get_decoder_layer(decoder)
    except DataValidationError:
        linear_layers = [m for m in decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise DataValidationError("No Linear layer found in decoder")
        linear_layer = linear_layers[-1]
    
    weight = ensure_numpy(linear_layer.weight.data)
    bias = ensure_numpy(linear_layer.bias.data) if (has_bias and linear_layer.bias is not None) else np.zeros(weight.shape[0])
    
    N, m = weight.shape
    
    if factor_order == 1:
        from ..config.constants import DEFAULT_IDENTITY_SCALE
        emission = np.hstack([
            weight,
            create_scaled_identity(N, DEFAULT_IDENTITY_SCALE)
        ])
    else:
        raise NotImplementedError(f"Only VAR(1) (factor_order=1) is supported. Got: {factor_order}")
    
    return bias, emission

