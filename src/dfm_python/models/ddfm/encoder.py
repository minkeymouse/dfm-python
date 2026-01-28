"""Encoder and autoencoder implementation for DDFM.

This module provides the nonlinear encoder network and autoencoder
specifically for the Deep Dynamic Factor Model (DDFM).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...dataset.ddfm_dataset import AutoencoderDataset

from ..base import BaseEncoder, BaseDecoder
from .decoder import DDFMLinearDecoder, DDFMMLPDecoder
from ...utils.errors import ConfigurationError, DataValidationError
from ...config.constants import (
    DEFAULT_XAVIER_GAIN,
    DEFAULT_ZERO_VALUE,
    DEFAULT_BATCH_NORM_MOMENTUM,
    DEFAULT_BATCH_NORM_EPS,
    DEFAULT_AUTOENCODER_FIT_EPOCHS,
    DEFAULT_DDFM_BATCH_SIZE,
    DEFAULT_DDFM_LEARNING_RATE,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_CLEAN_NAN,
)
from ...logger import get_logger

_logger = get_logger(__name__)


def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation matrix C and bias from decoder.
    
    Parameters
    ----------
    decoder : BaseDecoder or nn.Linear
        Decoder instance or Linear layer
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (weight, bias) where weight is (output_dim, input_dim) and bias is (output_dim,)
        
    Raises
    ------
    DataValidationError
        If decoder doesn't have extract_params method and is not a Linear layer
    """
    if hasattr(decoder, 'extract_params'):
        return decoder.extract_params()
    
    if isinstance(decoder, nn.Linear):
        weight = decoder.weight.data.cpu().numpy()
        bias = decoder.bias.data.cpu().numpy() if decoder.bias is not None else np.zeros(weight.shape[0])
        if np.any(np.isnan(weight)):
            _logger.warning("extract_decoder_params: C matrix contains NaN values. Replacing with zeros.")
            weight = np.nan_to_num(weight, nan=DEFAULT_CLEAN_NAN)
        return weight, bias
    
    raise DataValidationError(
        f"decoder must have 'extract_params' method or be a Linear layer. Got: {type(decoder)}"
    )


class DDFMEncoder(BaseEncoder):
    """Nonlinear encoder network for DDFM.
    
    This is the encoder component used in DDFM's SimpleAutoencoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        activation: str = 'relu',
    ):
        """Initialize DDFM encoder.
        
        Parameters
        ----------
        input_dim : int
            Input dimension (observation dimension)
        encoder_dims : List[int]
            Hidden layer dimensions. Must have at least one element.
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()
        
        if len(encoder_dims) == 0:
            raise ValueError("encoder_dims must have at least one element")
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Activation function
        activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if activation not in activations:
            raise ConfigurationError(f"Unknown activation: {activation}")
        self.activation = activations[activation]
        
        # Build layers
        prev_dim = input_dim
        first_layer = nn.Linear(prev_dim, encoder_dims[0])
        self._init_linear(first_layer)
        self.layers.append(first_layer)
        prev_dim = encoder_dims[0]
        
        for dim in encoder_dims[1:]:
            self.batch_norms.append(
                nn.BatchNorm1d(prev_dim, momentum=DEFAULT_BATCH_NORM_MOMENTUM, eps=DEFAULT_BATCH_NORM_EPS)
            )
            layer = nn.Linear(prev_dim, dim)
            self._init_linear(layer)
            self.layers.append(layer)
            prev_dim = dim
    
    @staticmethod
    def _init_linear(layer: nn.Linear) -> None:
        """Initialize linear layer weights and bias.
        
        Parameters
        ----------
        layer : nn.Linear
            Linear layer to initialize
        """
        nn.init.xavier_normal_(layer.weight, gain=DEFAULT_XAVIER_GAIN)
        nn.init.constant_(layer.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data (T, input_dim) or (batch, T, input_dim)
            
        Returns
        -------
        torch.Tensor
            Encoded factors (T, encoder_dims[-1]) or (batch, T, encoder_dims[-1])
        """
        x = self.activation(self.layers[0](x))
        for i in range(1, len(self.layers)):
            x = self.batch_norms[i - 1](x)
            x = self.activation(self.layers[i](x))
        
        return x


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for DDFM.
    
    This is the DDFM autoencoder, combining DDFMEncoder and DDFM decoders
    (DDFMLinearDecoder or DDFMMLPDecoder). The encoder component is equivalent
    to DDFMEncoder, and together with a decoder forms the complete autoencoder
    used in DDFM training.
    """
    
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        """Initialize autoencoder.
        
        Parameters
        ----------
        encoder : BaseEncoder
            Encoder network
        decoder : BaseDecoder
            Decoder network
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    @classmethod
    def build(
        cls,
        input_dim: int,
        encoder_size: Tuple[int, ...],
        decoder_size: Optional[Tuple[int, ...]] = None,
        decoder_type: str = "linear",
        output_dim: Optional[int] = None,
        activation: str = 'relu',
        seed: Optional[int] = None
    ) -> "SimpleAutoencoder":
        """Build autoencoder with encoder and decoder.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        encoder_size : Tuple[int, ...]
            Encoder hidden layer dimensions
        decoder_size : Optional[Tuple[int, ...]]
            Decoder hidden layer dimensions (required if decoder_type='mlp')
        decoder_type : str
            Decoder type ('linear' or 'mlp')
        output_dim : Optional[int]
            Output dimension (defaults to input_dim)
        activation : str
            Activation function
        seed : Optional[int]
            Random seed
            
        Returns
        -------
        SimpleAutoencoder
            Constructed autoencoder
        """
        if output_dim is None:
            output_dim = input_dim
        
        if seed is not None:
            torch.manual_seed(seed)
        
        if len(encoder_size) == 0:
            raise ValueError("encoder_size must have at least one element")
        
        # Build encoder
        encoder = DDFMEncoder(input_dim, list(encoder_size), activation)
        
        # Build decoder
        latent_dim = encoder_size[-1]
        if decoder_type == "mlp":
            if decoder_size is None or len(decoder_size) == 0:
                raise ValueError("decoder_size must be provided when decoder_type='mlp'")
            decoder = DDFMMLPDecoder(latent_dim, output_dim, list(decoder_size), activation, seed=seed)
        else:
            decoder = DDFMLinearDecoder(latent_dim, output_dim, seed=seed)
        
        return cls(encoder=encoder, decoder=decoder)
    
    @classmethod
    def from_dataset(
        cls,
        dataset: 'AutoencoderDataset',
        encoder_size: Tuple[int, ...],
        decoder_size: Optional[Tuple[int, ...]] = None,
        decoder_type: str = "linear",
        activation: str = 'relu',
        seed: Optional[int] = None
    ) -> "SimpleAutoencoder":
        """Build autoencoder from dataset.
        
        Parameters
        ----------
        dataset : AutoencoderDataset
            Dataset to build from
        encoder_size : Tuple[int, ...]
            Encoder hidden layer dimensions
        decoder_size : Optional[Tuple[int, ...]]
            Decoder hidden layer dimensions
        decoder_type : str
            Decoder type ('linear' or 'mlp')
        activation : str
            Activation function
        seed : Optional[int]
            Random seed
            
        Returns
        -------
        SimpleAutoencoder
            Constructed autoencoder
        """
        return cls.build(
            input_dim=dataset.full_input.shape[1],
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            decoder_type=decoder_type,
            output_dim=dataset.y_clean.shape[1],
            activation=activation,
            seed=seed
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Reconstructed output
        """
        return self.decoder(self.encoder(x))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using inference mode (matches TensorFlow's autoencoder.predict() behavior).
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
            
        Returns
        -------
        torch.Tensor
            Predicted output
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def pretrain(
        self,
        full_input: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        use_mse_loss: bool = True
    ) -> List[float]:
        """Pre-train autoencoder on clean data.
        
        Pre-training uses clean data (no corruption) to initialize the autoencoder
        before MCMC training. This is separate from the MCMC fit() method which
        uses corrupted inputs.
        
        Parameters
        ----------
        full_input : torch.Tensor
            Full input data (X + y) for pre-training (T, input_dim)
        y : torch.Tensor
            Target data for pre-training (T, output_dim)
        epochs : int
            Number of pre-training epochs
        batch_size : int
            Batch size for pre-training
        optimizer : torch.optim.Optimizer
            Optimizer instance for pre-training
        use_mse_loss : bool, default True
            Whether to use standard MSE loss (True) or masked MSE loss (False)
            
        Returns
        -------
        List[float]
            Final epoch losses for each epoch
        """
        self.train()
        T = len(full_input)
        final_epoch_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            for i in range(0, T, batch_size):
                batch_full_input = full_input[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                y_pred = self.forward(batch_full_input)
                
                if use_mse_loss:
                    loss = torch.nn.functional.mse_loss(y_pred, batch_y, reduction='mean')
                else:
                    mask = ~torch.isnan(batch_y)
                    y_actual = torch.where(torch.isnan(batch_y), torch.zeros_like(batch_y), batch_y)
                    y_pred_masked = y_pred * mask.float()
                    loss = torch.nn.functional.mse_loss(y_pred_masked, y_actual, reduction='mean')
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            if epoch_losses:
                final_epoch_losses.append(np.mean(epoch_losses))
        
        return final_epoch_losses
    
    def fit(
        self,
        dataset: 'AutoencoderDataset',
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        optimizer_type: str = 'Adam',
        decay_learning_rate: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        target_indices: Optional[torch.Tensor] = None
    ) -> None:
        """Fit autoencoder on dataset.
        
        Parameters
        ----------
        dataset : AutoencoderDataset
            Dataset to train on
        epochs : Optional[int]
            Number of training epochs
        batch_size : Optional[int]
            Batch size
        learning_rate : Optional[float]
            Learning rate
        optimizer_type : str
            Optimizer type ('Adam' or 'SGD')
        decay_learning_rate : bool
            Whether to decay learning rate
        optimizer : Optional[torch.optim.Optimizer]
            Pre-configured optimizer (if None, creates new one)
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Pre-configured scheduler (if None and decay_learning_rate=True, creates new one)
        target_indices : Optional[torch.Tensor]
            Target indices (not used, kept for compatibility)
        """
        if epochs is None:
            epochs = DEFAULT_AUTOENCODER_FIT_EPOCHS
        if batch_size is None:
            batch_size = DEFAULT_DDFM_BATCH_SIZE
        if learning_rate is None:
            learning_rate = DEFAULT_DDFM_LEARNING_RATE
        
        if optimizer is None:
            optimizers = {
                'Adam': lambda: torch.optim.Adam(self.parameters(), lr=learning_rate),
                'SGD': lambda: torch.optim.SGD(self.parameters(), lr=learning_rate)
            }
            optimizer = optimizers.get(optimizer_type, optimizers['Adam'])()
        
        if scheduler is None and decay_learning_rate:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=DEFAULT_N_MC_SAMPLES, gamma=DEFAULT_LR_DECAY_RATE
            )
        
        self.train()
        full_input = dataset.full_input
        y_clean = dataset.y_clean
        T = len(dataset)
        
        for epoch in range(epochs):
            for i in range(0, T, batch_size):
                batch_input = full_input[i:i+batch_size]
                batch_target = y_clean[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = self.forward(batch_input)
                # Note: target_indices is not used here because the autoencoder
                # output_dim is already set to num_target_series, so pred already
                # contains only target series predictions
                
                mask = ~torch.isnan(batch_target)
                y_actual_ = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                y_predicted_ = pred * mask.float()
                loss = torch.nn.functional.mse_loss(y_predicted_, y_actual_, reduction='mean')
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
