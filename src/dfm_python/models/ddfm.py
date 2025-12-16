"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.

DDFM is a PyTorch Lightning module that inherits from BaseFactorModel.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from ..config import (
    ConfigSource,
    DFMConfig,
    MergedConfigSource,
    make_config_source,
)
from ..config.results import DDFMResult
from ..config.utils import get_periods_per_year
from ..encoder.autoencoder import Encoder, extract_decoder_params
from ..decoder.linear import Decoder
from ..decoder.mlp import MLPDecoder
from ..logger import get_logger
from ..utils.data import rem_nans_spline
from ..utils.helpers import (
    get_clock_frequency,
)
from ..utils.statespace import (
    estimate_var1,
    estimate_var2,
)
from ..utils.time import (
    TimeIndex,
)
from .base import BaseFactorModel
from .utils import (
    estimate_var_ddfm,
    validate_factors_ddfm,
    validate_training_data_ddfm,
    forecast_var_factors,
)

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class DDFMTrainingState:
    """State tracking for DDFM training."""
    factors: np.ndarray
    prediction: np.ndarray
    converged: bool
    num_iter: int
    training_loss: Optional[float] = None

# ============================================================================
# High-level API Classes
# ============================================================================

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

class DDFM(BaseFactorModel):
    """High-level API for Deep Dynamic Factor Model (PyTorch Lightning module).
    
    This class is a PyTorch Lightning module that can be used with standard
    Lightning training patterns. It inherits from BaseFactorModel and implements
    DDFM training using autoencoder and MCMC procedure.
    
    Note: Maximum supported VAR order for factor dynamics is VAR(2) (set via factor_order parameter).
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import DDFM, DDFMDataModule, DDFMTrainer
        >>> import pandas as pd
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pd.read_csv('data/finance.csv')
        >>> df_processed = df[[col for col in df.columns if col != 'date']]
        >>> 
        >>> # Step 2: Create DataModule (use DDFMDataModule for DDFM)
        >>> dm = DDFMDataModule(config_path='config/ddfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config/ddfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = DDFMTrainer(max_epochs=100)
        >>> trainer.fit(model, dm)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    
    Note on GPU Memory Usage:
        DDFM typically uses less GPU memory than DFM because:
        1. DDFM uses batch training (batch_size=100, matching original DDFM), processing data in small chunks
        2. DFM uses EM algorithm with Kalman filtering, which stores large covariance
           matrices on GPU: V (m x m x T+1), R (N x N), Q (m x m) for all time steps
        3. DDFM's neural network (encoder/decoder) is relatively small compared to
           the large covariance matrices in DFM's Kalman smoother
        4. DDFM processes data incrementally, while DFM processes the full dataset
           simultaneously during Kalman smoothing
        
        For example, with T=8000, N=22, m=2:
        - DFM: V matrix alone is (2 x 2 x 8001) = ~128KB, plus R (22 x 22) = ~4KB,
          plus all intermediate matrices during Kalman smoothing
        - DDFM: Processes batches of 32 samples at a time, so only (32 x 22) = ~3KB
          per batch on GPU, plus small encoder/decoder weights
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        learning_rate: float = 0.005,
        epochs: int = 100,
        batch_size: int = 100,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        seed: Optional[int] = None,
        decay_learning_rate: bool = True,
        min_obs_pretrain: int = 50,
        mult_epoch_pretrain: int = 1,
        loss_function: str = 'mse',
        huber_delta: float = 1.0,
        weight_decay: float = 0.0,
        grad_clip_val: float = 1.0,
        decoder: str = "linear",
        decoder_layers: Optional[List[int]] = None,
        **kwargs
    ):
        """Initialize DDFM instance.
        
        Parameters
        ----------
        config : DFMConfig, optional
            DFM configuration. Can be loaded later via load_config().
        encoder_layers : List[int], optional
            Hidden layer dimensions for encoder. Default: [64, 32]
        num_factors : int, optional
            Number of factors. If None, inferred from config.
        activation : str, default 'relu'
            Activation function ('tanh', 'relu', 'sigmoid'). Default: 'relu' (matches original DDFM)
        use_batch_norm : bool, default True
            Whether to use batch normalization in encoder
        learning_rate : float, default 0.005
            Learning rate for Adam optimizer (matches original DDFM default)
        epochs : int, default 100
            Number of epochs per MCMC iteration
        batch_size : int, default 100
            Batch size for training (matches original DDFM)
        factor_order : int, default 1
            VAR lag order for factor dynamics. Must be 1 or 2 (maximum supported order is VAR(2))
        use_idiosyncratic : bool, default True
            Whether to model idiosyncratic components
        min_obs_idio : int, default 5
            Minimum observations for idio AR(1) estimation
        max_iter : int, default 200
            Maximum number of MCMC iterations
        tolerance : float, default 0.0005
            Convergence tolerance
        disp : int, default 10
            Display progress every 'disp' iterations
        decay_learning_rate : bool, default True
            Whether to use exponential decay learning rate scheduler (matches original DDFM)
        min_obs_pretrain : int, default 50
            Minimum number of observations for pre-training without interpolation
        mult_epoch_pretrain : int, default 1
            Multiplier for number of epochs during pre-training
            Display progress every 'disp' iterations
        loss_function : str, default 'mse'
            Loss function for training ('mse', 'huber'). 
            'mse': Mean squared error (default, matches original DDFM)
            'huber': Huber loss (more robust to outliers)
        huber_delta : float, default 1.0
            Delta parameter for Huber loss (only used if loss_function='huber').
            Controls the transition point between quadratic and linear regions.
        weight_decay : float, default 0.0
            Weight decay (L2 regularization) for optimizer. Helps prevent overfitting to linear features.
            Recommended: 1e-5 to 1e-3 for deeper encoders or when encoder collapses to linear behavior.
        grad_clip_val : float, default 1.0
            Maximum gradient norm for gradient clipping. Prevents training instability.
            Set to 0.0 to disable gradient clipping.
        decoder : str, default "linear"
            Decoder type: "linear" (linear decoder) or "mlp" (nonlinear MLP decoder).
            Linear decoder preserves interpretability and allows Kalman filtering.
            MLP decoder provides more expressive power but loses interpretability.
        decoder_layers : List[int], optional
            Hidden layer dimensions for MLP decoder. Only used if decoder="mlp".
            Default: [output_dim] (single hidden layer with same size as output).
        seed : int, optional
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        super().__init__(**kwargs)
        
        # Initialize config using consolidated helper method
        # DDFM does not use block structure
        config = self._initialize_config(config)
        
        # Validate factor_order
        if factor_order not in [1, 2]:
            raise ValueError(
                f"DDFM initialization failed: factor_order must be 1 or 2, got {factor_order}. "
                f"Maximum supported VAR order is VAR(2). Please provide factor_order=1 (VAR(1)) or factor_order=2 (VAR(2))"
            )
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.decoder_type = decoder
        self.decoder_layers = decoder_layers
        self.learning_rate = learning_rate
        self.epochs_per_iter = epochs
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        self.decay_learning_rate = decay_learning_rate
        self.min_obs_pretrain = min_obs_pretrain
        self.mult_epoch_pretrain = mult_epoch_pretrain
        self.loss_function = loss_function.lower()
        self.huber_delta = huber_delta
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        
        # Validate loss function
        if self.loss_function not in ['mse', 'huber']:
            raise ValueError(
                f"DDFM initialization failed: loss_function must be 'mse' or 'huber', got '{loss_function}'"
            )
        
        # Validate gradient clipping value
        if self.grad_clip_val < 0.0:
            raise ValueError(
                f"DDFM initialization failed: grad_clip_val must be >= 0.0, got {grad_clip_val}"
            )
        
        # Determine number of factors
        # DDFM does not use block structure - num_factors is specified directly
        if num_factors is None:
            # Try to get from config num_factors (DDFM-specific parameter)
            if hasattr(config, 'num_factors') and config.num_factors is not None:
                self.num_factors = config.num_factors
            else:
                # Default to 1 if not specified
                self.num_factors = 1
            # Track that num_factors was computed from config, not explicitly set
            self._num_factors_explicit = False
        else:
            self.num_factors = num_factors
            # Track that num_factors was explicitly set
            self._num_factors_explicit = True
        
        # Initialize encoder and decoder
        # input_dim and output_dim will be set in setup() when we know data dimensions
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        
        # Training state
        self.data_processed: Optional[torch.Tensor] = None
        self.target_scaler: Optional[Any] = None
        
        # MCMC state
        self.mcmc_iteration: int = 0
        
        # Random number generator for MC sampling
        self.rng = np.random.RandomState(seed if seed is not None else 3)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize encoder and decoder when data dimensions are known.
        
        This is called by Lightning before configure_optimizers(), so we need
        to initialize encoder/decoder here if datamodule is available.
        If not available here, will be initialized in on_train_start().
        """
        # Access datamodule if available (trainer should be attached by now)
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                self._data_module = self.trainer.datamodule
                try:
                    # Get data to determine input dimension
                    # datamodule.setup() should have been called by Lightning already
                    X_torch = self._data_module.get_processed_data()
                    input_dim = X_torch.shape[1]
                    
                    # Initialize networks if not already initialized
                    if self.encoder is None or self.decoder is None:
                        self.initialize_networks(input_dim)
                        # Move to same device as data
                        device = X_torch.device
                        self.encoder = self.encoder.to(device)
                        self.decoder = self.decoder.to(device)
                        _logger.debug(f"Initialized encoder/decoder in setup() with input_dim={input_dim}")
                except (RuntimeError, AttributeError) as e:
                    # If datamodule not ready yet, will initialize in on_train_start()
                    _logger.debug(f"Could not initialize networks in setup(): {e}. Will initialize in on_train_start()")
                    pass
    
    def initialize_networks(self, input_dim: int) -> None:
        """Initialize encoder and decoder networks with error handling.
        
        Parameters
        ----------
        input_dim : int
            Number of input features (number of series)
            
        Raises
        ------
        RuntimeError
            If encoder or decoder initialization fails with clear error message
        """
        try:
            self.encoder = Encoder(
                input_dim=input_dim,
                hidden_dims=self.encoder_layers,
                output_dim=self.num_factors,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            raise RuntimeError(
                f"DDFM encoder initialization failed: failed to initialize encoder: {type(e).__name__}: {str(e)}. "
                f"Check encoder_layers={self.encoder_layers}, num_factors={self.num_factors}, "
                f"input_dim={input_dim}. "
                f"Suggestions: (1) Ensure input_dim > 0, (2) Reduce encoder_layers size if too large, "
                f"(3) Ensure num_factors > 0 and num_factors <= input_dim, "
                f"(4) Check that encoder_layers values are positive integers"
            ) from e
        
        try:
            # Create decoder based on decoder_type
            if self.decoder_type == "linear":
                self.decoder = Decoder(
                    input_dim=self.num_factors,
                    output_dim=input_dim,
                    use_bias=True,
                )
                # Validate decoder weights are not all zeros (initialization check)
                decoder_weight = self.decoder.decoder.weight.data.cpu().numpy()
            elif self.decoder_type == "mlp":
                self.decoder = MLPDecoder(
                    input_dim=self.num_factors,
                    output_dim=input_dim,
                    hidden_dims=self.decoder_layers,
                    activation=self.activation,
                    use_batch_norm=False,  # Usually not needed for decoder
                    use_bias=True,
                )
                # Validate decoder weights are not all zeros (initialization check)
                # For MLP, check the first layer
                decoder_weight = self.decoder.layers[0].weight.data.cpu().numpy()
            else:
                raise ValueError(
                    f"DDFM decoder initialization failed: decoder must be 'linear' or 'mlp', got '{self.decoder_type}'"
                )
            
            # Validate decoder weights are not all zeros (initialization check)
            if np.allclose(decoder_weight, 0.0, atol=1e-8):
                raise RuntimeError(
                    f"DDFM decoder initialization failed: decoder weights are all zeros after initialization. "
                    f"This indicates a problem with decoder initialization. "
                    f"Check: (1) Decoder class implementation, (2) Weight initialization method, "
                    f"(3) PyTorch version compatibility. "
                    f"Decoder weight shape: {decoder_weight.shape}, "
                    f"weight mean: {np.mean(decoder_weight):.6f}, "
                    f"weight std: {np.std(decoder_weight):.6f}"
                )
            
            # Log decoder initialization statistics
            decoder_weight_mean = np.mean(decoder_weight)
            decoder_weight_std = np.std(decoder_weight)
            decoder_weight_nonzero = np.count_nonzero(decoder_weight)
            _logger.debug(
                f"DDFM decoder initialized: weight shape={decoder_weight.shape}, "
                f"mean={decoder_weight_mean:.6f}, std={decoder_weight_std:.6f}, "
                f"nonzero={decoder_weight_nonzero}/{decoder_weight.size}"
            )
        except (ValueError, RuntimeError, TypeError) as e:
            raise RuntimeError(
                f"DDFM decoder initialization failed: failed to initialize decoder: {type(e).__name__}: {str(e)}. "
                f"Check num_factors={self.num_factors}, input_dim={input_dim}. "
                f"Suggestions: (1) Ensure num_factors > 0, (2) Ensure input_dim > 0, "
                f"(3) Check that num_factors <= input_dim"
            ) from e
    
    def _check_networks_initialized(self):
        """Check if encoder and decoder are initialized."""
        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: encoder and decoder must be initialized. "
                f"Ensure setup() or on_train_start() has been called."
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data (batch_size x T x N) or (T x N)
            
        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed data
        """
        self._check_networks_initialized()
        
        # Handle different input shapes
        if x.ndim == 3:
            batch_size, T, N = x.shape
            x_flat = x.view(batch_size * T, N)
            factors = self.encoder(x_flat)
            reconstructed = self.decoder(factors)
            return reconstructed.view(batch_size, T, N)
        else:
            factors = self.encoder(x)
            reconstructed = self.decoder(factors)
            return reconstructed
    
    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Training step for autoencoder.
        
        This is used for standard autoencoder training and also called
        during MCMC procedure for each MC sample.
        
        Missing data (NaN values) are handled by masking them in the loss function,
        similar to the original DDFM implementation (mse_missing).
        
        Parameters
        ----------
        batch : torch.Tensor or tuple
            Data tensor or (data, target) tuple where both are the same for reconstruction.
            Data may contain NaN values which are masked in the loss.
        batch_idx : int
            Batch index
            
        Returns
        -------
        loss : torch.Tensor
            Reconstruction loss (MSE with missing data masking)
        """
        # Handle both tuple and single tensor batches
        # DataLoader may return tuple, list, or single tensor
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
            data = batch[0]
            target = data  # For autoencoder, target is same as input
        else:
            data = batch
            target = data  # For autoencoder, target is same as input
        
        # Ensure data is on the same device as the model
        device = next(self.parameters()).device
        data = data.to(device)
        target = target.to(device)
        
        # Clip input data to prevent extreme values that cause NaN
        # Clip to reasonable range: -10 to 10 standard deviations
        # For deeper networks, use slightly tighter clipping to improve stability
        clip_range = 8.0 if len(self.encoder_layers) > 2 else 10.0
        data_clipped = torch.clamp(data, min=-clip_range, max=clip_range)
        
        # Forward pass
        reconstructed = self.forward(data_clipped)
        
        # Check for NaN/Inf in forward pass output
        if not torch.all(torch.isfinite(reconstructed)):
            nan_count = torch.sum(torch.isnan(reconstructed)).item()
            inf_count = torch.sum(torch.isinf(reconstructed)).item()
            _logger.error(f"DDFM training_step: Forward pass produced {nan_count} NaN and {inf_count} Inf values")
            loss = torch.tensor(1e6, device=data.device, dtype=data.dtype, requires_grad=True)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        # Compute loss with missing data masking
        mask = torch.isfinite(target)
        target_clean = torch.where(mask, target, torch.zeros_like(target))
        
        # Use specified loss function
        if self.loss_function == 'huber':
            # Huber loss: more robust to outliers
            # L_delta(a) = 0.5 * a^2 if |a| <= delta, else delta * (|a| - 0.5 * delta)
            diff = target_clean - reconstructed
            abs_diff = torch.abs(diff)
            huber_loss = torch.where(
                abs_diff <= self.huber_delta,
                0.5 * diff ** 2,
                self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
            )
            loss = torch.sum(huber_loss * mask) / (torch.sum(mask) + 1e-8)
        else:
            # MSE loss (default)
            squared_diff = (target_clean - reconstructed) ** 2
            loss = torch.sum(squared_diff * mask) / (torch.sum(mask) + 1e-8)
        
        # Handle NaN/Inf in loss
        if not torch.isfinite(loss):
            _logger.error(f"DDFM training_step: Loss is NaN/Inf")
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Note: Gradient clipping is handled automatically by Lightning trainer if gradient_clip_val is set
        # The grad_clip_val parameter is used in pre_train() and MCMC training for manual training loops
        
        return loss
    
    
    def _validate_factors(self, factors: np.ndarray, operation: str = "operation") -> np.ndarray:
        """Validate and normalize factors shape and content quality.
        
        This method delegates to validate_factors_ddfm() utility function.
        See utils.py for detailed documentation.
        """
        return validate_factors_ddfm(
            factors=factors,
            num_factors=self.num_factors,
            operation=operation,
        )
    
    def _validate_training_data(
        self,
        X_torch: torch.Tensor,
        operation: str = "training setup"
    ) -> None:
        """Validate data dimensions and model configuration before training starts.
        
        This method delegates to validate_training_data_ddfm() utility function.
        See utils.py for detailed documentation.
        """
        validate_training_data_ddfm(
            X_torch=X_torch,
            num_factors=self.num_factors,
            factor_order=self.factor_order,
            encoder_layers=self.encoder_layers,
            encoder=self.encoder,
            operation=operation,
        )
    
    
    def _estimate_var(
        self, 
        factors: np.ndarray, 
        factor_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate VAR dynamics with comprehensive error handling and fallback.
        
        This method delegates to estimate_var_ddfm() utility function.
        See utils.py for detailed documentation.
        """
        return estimate_var_ddfm(
            factors=factors,
            factor_order=factor_order,
            num_factors=self.num_factors,
        )
    
    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], Dict[str, Any]]:
        """Configure optimizer and learning rate scheduler for autoencoder training.
        
        Matches original DDFM implementation with exponential decay scheduler.
        
        Returns
        -------
        List[torch.optim.Optimizer] or Dict
            If decay_learning_rate=False: List containing the optimizer
            If decay_learning_rate=True: Dict with optimizer and scheduler config
        """
        if self.encoder is None or self.decoder is None:
            _logger.warning("Encoder/decoder not initialized, creating placeholder optimizer")
            dummy_param = nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.Adam([dummy_param], lr=self.learning_rate)
            if self.decay_learning_rate:
                # Create scheduler matching original DDFM: decay_rate=0.96, decay_steps=epochs
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.96
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                }
            return [optimizer]
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.decay_learning_rate:
            # Exponential decay scheduler matching original DDFM implementation
            # Original: decay_rate=0.96, decay_steps=epochs, staircase=True
            # PyTorch: gamma=0.96, step every epoch (interval='epoch', frequency=1)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.96
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        
        return [optimizer]
    
    def _create_optimizer(self, step: int = 0) -> torch.optim.Optimizer:
        """Create optimizer for autoencoder training.
        
        Helper method for internal use (e.g., in fit_mcmc()).
        For Lightning trainer setup, use configure_optimizers() instead.
        
        Parameters
        ----------
        step : int, default 0
            Current step/iteration for learning rate decay calculation
            
        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer for encoder and decoder parameters
        """
        self._check_networks_initialized()
        
        # Calculate learning rate with exponential decay if enabled
        # Original DDFM: decay_rate=0.96, decay_steps=epochs, staircase=True
        # lr = initial_lr * (decay_rate ^ floor(step / decay_steps))
        if self.decay_learning_rate:
            # For MCMC, we decay per MCMC iteration (not per epoch)
            # Each MCMC iteration uses epochs_per_iter epochs
            decay_steps = self.epochs_per_iter
            decay_rate = 0.96
            lr = self.learning_rate * (decay_rate ** (step // decay_steps))
        else:
            lr = self.learning_rate
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
    
    def pre_train(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        """Pre-train autoencoder on data without missing values.
        
        This method matches the original DDFM implementation's pre-training step.
        It trains the autoencoder on observations without missing values to provide
        a stable initialization before MCMC training.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data with missing values, shape (T x N)
        x_clean : torch.Tensor
            Clean data (interpolated), shape (T x N)
        missing_mask : np.ndarray
            Missing data mask, shape (T x N), boolean array where True indicates missing
        device : torch.device, optional
            Device to use for training. If None, uses self.device
            
        Notes
        -----
        Original DDFM pre-training procedure:
        1. Build inputs without interpolation (if enough observations)
        2. If not enough observations, use interpolated data
        3. Train autoencoder on non-missing data for epochs * mult_epoch_pretrain
        4. Uses MSE loss (not mse_missing) if enough non-missing observations
        """
        if device is None:
            device = self.device
        
        # Convert to numpy for easier missing data handling
        x_clean_np = x_clean.cpu().numpy() if isinstance(x_clean, torch.Tensor) else x_clean
        missing_mask_np = missing_mask if isinstance(missing_mask, np.ndarray) else missing_mask.cpu().numpy()
        
        # Check number of non-missing observations
        bool_no_miss = ~missing_mask_np
        n_non_missing = np.sum(bool_no_miss)
        
        # Determine if we have enough observations for pre-training without interpolation
        use_interpolated = n_non_missing < self.min_obs_pretrain
        
        if use_interpolated:
            # Use interpolated data (x_clean) for pre-training
            _logger.info(
                f"DDFM pre_train: Only {n_non_missing} non-missing observations (< {self.min_obs_pretrain}), "
                f"using interpolated data for pre-training"
            )
            inpt_pre_train = x_clean_np
            # Use mse_missing loss to handle any remaining missing values
            use_mse_missing = True
        else:
            # Use only non-missing observations (original DDFM behavior)
            _logger.info(
                f"DDFM pre_train: {n_non_missing} non-missing observations (>= {self.min_obs_pretrain}), "
                f"using non-missing data only for pre-training"
            )
            # Extract non-missing rows
            non_missing_rows = np.all(bool_no_miss, axis=1)
            inpt_pre_train = x_clean_np[non_missing_rows, :]
            # Use standard MSE loss (no missing values)
            use_mse_missing = False
        
        # Output is same as input for autoencoder (reconstruction task)
        oupt_pre_train = inpt_pre_train.copy()
        
        # Convert to torch tensors and ensure they're on the correct device
        inpt_tensor = torch.tensor(inpt_pre_train, device=device, dtype=torch.float32)
        oupt_tensor = torch.tensor(oupt_pre_train, device=device, dtype=torch.float32)
        
        # Ensure encoder and decoder are on the same device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(inpt_tensor, oupt_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Create optimizer for pre-training
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Pre-train for epochs * mult_epoch_pretrain
        num_epochs = self.epochs_per_iter * self.mult_epoch_pretrain
        _logger.info(f"DDFM pre_train: Starting pre-training for {num_epochs} epochs")
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_data, batch_target in dataloader:
                # Ensure batch data is on the correct device (should already be, but double-check)
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.forward(batch_data)
                
                # Compute loss
                if use_mse_missing:
                    # Handle missing values (though there shouldn't be any if use_interpolated=False)
                    mask = torch.where(
                        torch.isnan(batch_target),
                        torch.zeros_like(batch_target),
                        torch.ones_like(batch_target)
                    )
                    target_clean = torch.where(
                        torch.isnan(batch_target),
                        torch.zeros_like(batch_target),
                        batch_target
                    )
                    reconstructed_masked = reconstructed * mask
                    squared_diff = (target_clean - reconstructed_masked) ** 2
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
                else:
                    # Standard MSE (no missing values)
                    loss = nn.functional.mse_loss(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if self.grad_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        max_norm=self.grad_clip_val
                    )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                _logger.info(f"DDFM pre_train: Epoch {epoch + 1}/{num_epochs}, loss={avg_loss:.6f}")
        
        _logger.info(f"DDFM pre_train: Pre-training completed")
    
    def fit_mcmc(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        target_scaler: Optional[Any] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DDFMTrainingState:
        """Run MCMC iterative training procedure for DDFM.
        
        This method delegates to DDFMDenoisingTrainer for the actual denoising training procedure.
        See mcmc.py for detailed documentation of the MCMC algorithm.
        
        Parameters
        ----------
        X : torch.Tensor
            Preprocessed data with missing values, shape (T x N), where T is number
            of time periods and N is number of series. Data should already be preprocessed
            (imputation, scaling, etc.). Missing values should be NaN or handled via missing_mask.
        x_clean : torch.Tensor
            Clean data (interpolated), shape (T x N), used for initial autoencoder
            training. Should have same shape as X.
        missing_mask : np.ndarray
            Missing data mask, shape (T x N), boolean array where True indicates
            missing data. Must match shape of X.
        target_scaler : Any, optional
            Scaler instance for target series inverse transformation. Used to convert
            standardized predictions back to original scale for target series only.
            If None, no inverse transformation is applied (predictions remain standardized).
        max_iter : int, optional
            Maximum number of MCMC iterations. If None, uses self.max_iter (default: 200).
        tolerance : float, optional
            Convergence tolerance for MSE change between iterations. If None, uses
            self.tolerance (default: 0.0005). Training stops when |MSE_new - MSE_old| < tolerance.
        disp : int, optional
            Display progress every 'disp' iterations. If None, uses self.disp (default: 10).
            Set to 0 to disable progress output.
        seed : int, optional
            Random seed for reproducibility. If None, uses current random state.
            Sets both NumPy and PyTorch random seeds.
            
        Returns
        -------
        DDFMTrainingState
            Final training state containing:
            - factors: np.ndarray, shape (T x num_factors) - extracted factors
            - prediction: np.ndarray, shape (T x N) - final predictions
            - converged: bool - whether convergence was achieved
            - num_iter: int - number of iterations completed
            - training_loss: float - final training loss (MSE)
            
        Examples
        --------
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config.yaml')
        >>> # X, x_clean, missing_mask prepared from data module
        >>> state = model.fit_mcmc(X, x_clean, missing_mask, max_iter=50, tolerance=1e-4)
        >>> factors = state.factors  # (T x 2) factor estimates
        >>> print(f"Converged: {state.converged}, Iterations: {state.num_iter}")
        """
        from ..trainer.denoising import DDFMDenoisingTrainer
        
        trainer = DDFMDenoisingTrainer(self)
        return trainer.fit(
            X=X,
            x_clean=x_clean,
            missing_mask=missing_mask,
            target_scaler=target_scaler,
            max_iter=max_iter,
            tolerance=tolerance,
            disp=disp,
            seed=seed,
        )
    
    def get_result(self) -> DDFMResult:
        """Extract DDFMResult from trained model.
        
        Returns
        -------
        DDFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        if self.training_state is None:
            raise RuntimeError(
                f"{self.__class__.__name__} get_result failed: model has not been fitted yet. "
                f"Call fit_mcmc() first."
            )
        
        self._check_networks_initialized()
        
        # Extract decoder parameters (C, bias)
        C, bias = extract_decoder_params(self.decoder)
        
        # Log decoder weight statistics for monitoring and debugging
        C_mean = np.mean(C)
        C_std = np.std(C)
        C_min = np.min(C)
        C_max = np.max(C)
        C_nonzero = np.count_nonzero(C)
        C_zero_ratio = 1.0 - (C_nonzero / C.size)
        _logger.info(
            f"DDFM get_result: C matrix statistics - mean={C_mean:.6f}, std={C_std:.6f}, "
            f"min={C_min:.6f}, max={C_max:.6f}, nonzero={C_nonzero}/{C.size} ({1.0-C_zero_ratio:.1%}), "
            f"zero_ratio={C_zero_ratio:.1%}"
        )
        
        # Validate C matrix for NaN (extract_decoder_params should handle this, but double-check)
        if np.any(np.isnan(C)):
            nan_count = np.sum(np.isnan(C))
            nan_ratio = nan_count / C.size
            _logger.error(
                f"DDFM get_result: C matrix contains {nan_count}/{C.size} NaN values ({nan_ratio:.1%}) "
                f"after extraction. This indicates severe numerical instability. "
                f"The model cannot be used for prediction. Consider: (1) reducing learning rate, "
                f"(2) adding gradient clipping, (3) checking data quality, (4) reducing model complexity."
            )
            # Replace NaN with zeros as last resort (model will not work correctly)
            C = np.nan_to_num(C, nan=0.0)
            _logger.warning("Replaced NaN values in C matrix with zeros (model may not work correctly).")
        
        # Get factors and prediction
        factors = self.training_state.factors  # T x num_factors
        prediction_iter = self.training_state.prediction  # T x N
        
        # Validate and normalize factors shape
        factors = self._validate_factors(factors, operation="get_result")
        
        # Convert to numpy
        C = C.cpu().numpy() if isinstance(C, torch.Tensor) else C
        bias = bias.cpu().numpy() if isinstance(bias, torch.Tensor) else bias
        
        # Compute residuals and estimate idiosyncratic dynamics
        if self.data_processed is not None:
            x_standardized = self.data_processed.cpu().numpy()
            # Ensure shapes match
            if x_standardized.shape != prediction_iter.shape:
                _logger.warning(
                    f"{self.__class__.__name__} get_result: shape mismatch: data_processed {x_standardized.shape} vs prediction {prediction_iter.shape}. "
                    f"Using prediction shape for residuals"
                )
                residuals = np.zeros_like(prediction_iter)
            else:
                residuals = x_standardized - prediction_iter
        else:
            residuals = np.zeros_like(prediction_iter)
        
        # Estimate factor dynamics (VAR) with error handling
        A_f, Q_f = self._estimate_var(factors, self.factor_order)
        
        # For DDFM, we use simplified state-space (factor-only)
        A = A_f
        Q = Q_f
        Z_0 = factors[0, :]
        V_0 = np.cov(factors.T)
        # Ensure V_0 is always 2D (np.cov returns scalar when m=1)
        if V_0.ndim == 0:
            V_0 = np.atleast_2d(V_0)
        elif V_0.ndim == 1:
            # If 1D, reshape to (m x m)
            V_0 = np.atleast_2d(V_0).T if V_0.shape[0] == 1 else np.atleast_2d(V_0)
        
        # Estimate R from residuals
        R_diag = np.var(residuals, axis=0)
        R = np.diag(np.maximum(R_diag, 1e-8))
        
        # Compute smoothed data
        x_sm = prediction_iter  # T x N (standardized, already preprocessed)
        
        # Unstandardize: Data is already preprocessed, so for most series Mx=0, Wx=1
        # Only target series (if target_scaler provided) need inverse transformation
        n_series = C.shape[0]
        
        # Initialize with defaults (already standardized: Mx=0, Wx=1)
        Mx_clean = np.zeros(n_series)
        Wx_clean = np.ones(n_series)
        
        # If target_scaler is available, extract Mx, Wx for target series only
        if self.target_scaler is not None:
            try:
                # Try to get target indices from DataModule
                data_module = self._get_datamodule()
                target_indices = getattr(data_module, 'get_target_indices', lambda: [])()

                if len(target_indices) > 0:
                    # Get target Mx, Wx from DataModule (computed from target_scaler)
                    target_Mx, target_Wx = data_module.get_std_params()
                    
                    if target_Mx is not None and target_Wx is not None and len(target_Mx) > 0:
                        # Apply target Mx, Wx only to target indices
                        for i, tgt_idx in enumerate(target_indices):
                            if tgt_idx < n_series:
                                if i < len(target_Mx):
                                    Mx_clean[tgt_idx] = target_Mx[i]
                                if i < len(target_Wx):
                                    Wx_clean[tgt_idx] = target_Wx[i]
            except (ValueError, AttributeError):
                # DataModule not available or no target indices - use defaults
                pass
        
        X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized)
        
        # Create result object
        r = np.array([self.num_factors])
        
        result = DDFMResult(
            x_sm=x_sm,
            X_sm=X_sm,
            Z=factors,  # T x m
            C=C,
            R=R,
            A=A,
            Q=Q,
            Mx=Mx_clean,
            Wx=Wx_clean,
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=self.factor_order,
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=None,  # DDFM doesn't compute loglik in same way
            series_ids=self.config.get_series_ids() if hasattr(self.config, 'get_series_ids') else None,
            block_names=None,  # DDFM does not use block structure (DFM-specific)
            training_loss=self.training_state.training_loss,
            encoder_layers=self.encoder_layers,
            use_idiosyncratic=self.use_idiosyncratic,
        )
        
        return result
    
    def on_train_end(self) -> None:
        """Called when training ends. Automatically computes result from training state."""
        # Automatically compute result after training completes
        if self.training_state is not None:
            try:
                if not hasattr(self, '_result') or self._result is None:
                    self._result = self.get_result()
            except Exception as e:
                # Log warning but don't fail - result can be computed later if needed
                _logger.warning(
                    f"Could not automatically compute result after training: {e}. "
                    f"Result will be computed on first access to result property or predict()."
                )
    
    def on_train_start(self) -> None:
        """Called when training starts. Run MCMC training."""
        # Get processed data and target scaler from DataModule
        data_module = self._get_datamodule()
        X_torch = data_module.get_processed_data()
        target_scaler = getattr(data_module, 'target_scaler', None)
        
        # Early validation: Check data dimensions and model configuration before training
        # This catches configuration issues early with clear error messages
        self._validate_training_data(X_torch, operation="training setup")
        
        # Initialize encoder/decoder if not already done in setup()
        if self.encoder is None or self.decoder is None:
            input_dim = X_torch.shape[1]
            self.initialize_networks(input_dim)
            # Move to same device as data
            device = X_torch.device
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            _logger.debug(f"Initialized encoder/decoder in on_train_start() with input_dim={input_dim}")
        
        # Always run fit_mcmc() if training_state is None (first training run)
        if self.training_state is None:
            # Handle missing data - use imputation for MCMC training
            nan_method = getattr(self.config, 'nan_method', 1)
            nan_k = getattr(self.config, 'nan_k', 3)
            
            # Check if data has NaN values
            if isinstance(X_torch, torch.Tensor):
                has_nan = torch.any(torch.isnan(X_torch)).item()
                X_np = X_torch.cpu().numpy()
            else:
                has_nan = np.any(np.isnan(X_torch))
                X_np = X_torch
            
            if has_nan:
                _logger.info(
                    f"DDFM on_train_start: NaN values detected in training data. "
                    f"Using imputation (method={nan_method}) for MCMC initialization. "
                    f"DDFM will handle remaining missing data through state-space model."
                )
                x_clean_np, missing_mask = rem_nans_spline(
                    X_np,
                    method=nan_method,
                    k=nan_k
                )
                x_clean_torch = torch.tensor(x_clean_np, dtype=torch.float32, device=X_torch.device)
            else:
                # No NaN values - use data as-is
                x_clean_torch = X_torch if isinstance(X_torch, torch.Tensor) else torch.tensor(X_torch, dtype=torch.float32, device=X_torch.device)
                missing_mask = np.zeros(X_np.shape, dtype=bool)  # No missing data
            
            # Replace any remaining NaN/Inf with zeros (defensive check)
            device = x_clean_torch.device
            dtype = x_clean_torch.dtype
            x_clean_torch = torch.where(
                torch.isfinite(x_clean_torch),
                x_clean_torch,
                torch.tensor(0.0, device=device, dtype=dtype)
            )
            
            # Adjust missing_mask shape to match x_clean_torch
            if missing_mask.shape != x_clean_torch.shape:
                _logger.warning(f"DDFM on_train_start: missing_mask shape {missing_mask.shape} != x_clean_torch shape {x_clean_torch.shape}, adjusting")
                target_shape = x_clean_torch.shape
                # Truncate or pad rows
                if missing_mask.shape[0] > target_shape[0]:
                    missing_mask = missing_mask[:target_shape[0], :]
                elif missing_mask.shape[0] < target_shape[0]:
                    pad_rows = target_shape[0] - missing_mask.shape[0]
                    missing_mask = np.vstack([missing_mask, np.zeros((pad_rows, missing_mask.shape[1]), dtype=bool)])
                # Truncate or pad columns
                if missing_mask.shape[1] > target_shape[1]:
                    missing_mask = missing_mask[:, :target_shape[1]]
                elif missing_mask.shape[1] < target_shape[1]:
                    pad_cols = target_shape[1] - missing_mask.shape[1]
                    missing_mask = np.hstack([missing_mask, np.zeros((missing_mask.shape[0], pad_cols), dtype=bool)])
            
            # Pre-train autoencoder on non-missing data (matching original DDFM)
            # This provides stable initialization before MCMC training
            try:
                self.pre_train(
                    X=x_clean_torch,
                    x_clean=x_clean_torch,
                    missing_mask=missing_mask,
                    device=x_clean_torch.device,
                )
            except Exception as e:
                _logger.warning(
                    f"DDFM pre_train failed: {e}. Continuing with MCMC training without pre-training. "
                    f"Continuing without pre-training."
                )
            
            # Run MCMC training
            # Pass x_clean_torch as X to ensure all data arrays have consistent shape
            self.fit_mcmc(
                X=x_clean_torch,
                x_clean=x_clean_torch,
                missing_mask=missing_mask,
                target_scaler=target_scaler,
            )
        
        super().on_train_start()
    
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> 'DDFM':
        """Load configuration from various sources."""
        # Preserve explicitly set num_factors if it was set during initialization
        preserved_num_factors = None
        if hasattr(self, '_num_factors_explicit') and self._num_factors_explicit:
            preserved_num_factors = self.num_factors
        
        # Use common config loading logic
        self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        # Restore preserved num_factors if it was explicitly set
        if preserved_num_factors is not None:
            self.num_factors = preserved_num_factors
            # Keep the flag set since it's still explicitly set
            self._num_factors_explicit = True
        
        # DDFM-specific initialization is handled in __init__ or on_train_start
        # No additional setup needed here
        
        return self
    
    @staticmethod
    def _extract_state_dict_and_hparams(checkpoint: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract state_dict and hyperparameters from checkpoint."""
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict'], checkpoint.get('hyper_parameters', {})
        return checkpoint, {}
    
    @staticmethod
    def _infer_input_dim_from_data(
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        date_id_col: str = "date_id"
    ) -> int:
        """Infer input_dim from data."""
        if isinstance(data, pd.DataFrame):
            exclude_cols = [date_id_col, 'market_forward_excess_returns', 
                          'forward_returns', 'risk_free_rate']
            feature_cols = [c for c in data.columns if c not in exclude_cols]
            return len(feature_cols)
        elif isinstance(data, np.ndarray):
            return data.shape[1] if len(data.shape) > 1 else 1
        elif isinstance(data, torch.Tensor):
            return data.shape[1] if len(data.shape) > 1 else 1
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def _infer_input_dim_from_checkpoint(state_dict: Dict[str, Any]) -> Optional[int]:
        """Infer input_dim from checkpoint state_dict."""
        if not isinstance(state_dict, dict):
            return None
        
        # Check encoder.layers.0.weight shape: (hidden_dim, input_dim)
        first_layer_keys = [k for k in state_dict.keys() if 'encoder.layers.0.weight' in k]
        if first_layer_keys:
            weight = state_dict[first_layer_keys[0]]
            if isinstance(weight, torch.Tensor):
                return weight.shape[1]
        
        # Check decoder.decoder.weight shape: (output_dim, num_factors)
        decoder_keys = [k for k in state_dict.keys() 
                      if 'decoder.decoder.weight' in k or ('decoder' in k and 'layers.0.weight' in k)]
        for key in decoder_keys:
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor):
                return weight.shape[0]  # output_dim is input_dim
        
        return None
    
    @staticmethod
    def _infer_model_params_from_state_dict(
        state_dict: Dict[str, Any],
        hparams: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Infer model parameters from state_dict."""
        if not isinstance(state_dict, dict):
            return {
                'encoder_layers': hparams.get('encoder_layers') or kwargs.get('encoder_layers', [64, 32]),
                'num_factors': hparams.get('num_factors') or kwargs.get('num_factors', 3),
                'activation': hparams.get('activation') or kwargs.get('activation', 'relu'),
                'use_batch_norm': hparams.get('use_batch_norm', kwargs.get('use_batch_norm', True)),
                'decoder': hparams.get('decoder') or kwargs.get('decoder', 'linear'),
                'decoder_layers': hparams.get('decoder_layers') or kwargs.get('decoder_layers', None),
            }
        
        # Infer encoder_layers
        encoder_layer_keys = [k for k in sorted(state_dict.keys()) 
                             if 'encoder.layers' in k and 'weight' in k and 'output' not in k]
        inferred_encoder_layers = None
        if encoder_layer_keys:
            inferred_encoder_layers = []
            for key in encoder_layer_keys:
                weight = state_dict[key]
                if isinstance(weight, torch.Tensor):
                    inferred_encoder_layers.append(weight.shape[0])
        
        # Infer num_factors
        inferred_num_factors = None
        output_layer_keys = [k for k in state_dict.keys() if 'encoder.output_layer.weight' in k]
        if output_layer_keys:
            weight = state_dict[output_layer_keys[0]]
            if isinstance(weight, torch.Tensor):
                inferred_num_factors = weight.shape[0]
        
        if inferred_num_factors is None:
            decoder_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
            for key in decoder_keys:
                if 'decoder.weight' in key or ('layers' in key and '0.weight' in key):
                    weight = state_dict[key]
                    if isinstance(weight, torch.Tensor):
                        inferred_num_factors = weight.shape[1]
                        break
        
        return {
            'encoder_layers': inferred_encoder_layers or hparams.get('encoder_layers') or kwargs.get('encoder_layers', [64, 32]),
            'num_factors': inferred_num_factors or hparams.get('num_factors') or kwargs.get('num_factors', 3),
            'activation': hparams.get('activation') or kwargs.get('activation', 'relu'),
            'use_batch_norm': hparams.get('use_batch_norm', kwargs.get('use_batch_norm', True)),
            'decoder': hparams.get('decoder') or kwargs.get('decoder', 'linear'),
            'decoder_layers': hparams.get('decoder_layers') or kwargs.get('decoder_layers', None),
        }
    
    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        input_dim: Optional[int] = None,
        date_id_col: str = "date_id",
        device: str = "cpu",
        map_location: Optional[str] = None,
        **kwargs
    ) -> 'DDFM':
        """Load DDFM model from checkpoint with automatic encoder/decoder initialization.
        
        This method loads a DDFM model from checkpoint and automatically initializes
        encoder/decoder if they are not already initialized. This is useful when loading
        state_dict checkpoints that don't include the full model state.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file (.ckpt)
        data : pd.DataFrame, np.ndarray, or torch.Tensor, optional
            Data to determine input_dim. If provided, input_dim will be inferred from data.
            If None, input_dim must be provided explicitly.
        input_dim : int, optional
            Number of input features. If None and data is provided, will be inferred from data.
            If both are None, will try to infer from checkpoint metadata (if available).
        date_id_col : str, default "date_id"
            Column name for date ID (only used if data is pd.DataFrame)
        device : str, default "cpu"
            Device to load model on
        map_location : str, optional
            Map location for torch.load (overrides device if provided)
        **kwargs
            Additional arguments passed to DDFM.__init__ if creating new model instance
            
        Returns
        -------
        DDFM
            Loaded DDFM model with encoder/decoder initialized
            
        Examples
        --------
        >>> # Load from Lightning checkpoint
        >>> model = DDFM.load("checkpoint.ckpt", data=df)
        >>> 
        >>> # Load from state_dict with explicit input_dim
        >>> model = DDFM.load("checkpoint.ckpt", input_dim=250)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        map_location = map_location or device
        
        # Try to load as Lightning checkpoint first
        try:
            model = cls.load_from_checkpoint(str(checkpoint_path), map_location=map_location, **kwargs)
            # Check if encoder is initialized
            if model.encoder is not None and model.decoder is not None:
                return model
            # If encoder not initialized, fall through to manual initialization
        except Exception as e:
            # Not a Lightning checkpoint, will load as state_dict
            pass
        
        # Load checkpoint and extract state_dict/hparams
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        state_dict, hparams = cls._extract_state_dict_and_hparams(checkpoint)
        
        # Infer input_dim: prioritize checkpoint, then data, then explicit parameter
        checkpoint_input_dim = cls._infer_input_dim_from_checkpoint(state_dict)
        
        if input_dim is None:
            if checkpoint_input_dim is not None:
                input_dim = checkpoint_input_dim
                # Warn if data dimension doesn't match
                if data is not None:
                    data_input_dim = cls._infer_input_dim_from_data(data, date_id_col)
                    if data_input_dim != input_dim:
                        _logger.warning(
                            f"DDFM.load: Data input_dim ({data_input_dim}) doesn't match checkpoint input_dim ({input_dim}). "
                            f"Using checkpoint input_dim. Model may not work correctly with current data."
                        )
            elif data is not None:
                input_dim = cls._infer_input_dim_from_data(data, date_id_col)
            else:
                raise ValueError(
                    "Cannot determine input_dim. Please provide either 'data' or 'input_dim' parameter. "
                    "input_dim is required to initialize encoder/decoder."
                )
        
        # Infer model parameters from state_dict
        model_params = cls._infer_model_params_from_state_dict(state_dict, hparams, kwargs)
        
        # Create model with inferred/provided parameters
        excluded_keys = ['encoder_layers', 'num_factors', 'activation', 'use_batch_norm', 'decoder', 'decoder_layers']
        model = cls(
            **model_params,
            **{k: v for k, v in kwargs.items() if k not in excluded_keys}
        )
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Initialize encoder/decoder if not already initialized
        if model.encoder is None or model.decoder is None:
            if input_dim is None:
                raise ValueError(
                    "Cannot initialize encoder/decoder: input_dim is required. "
                    "Please provide either 'data' or 'input_dim' parameter."
                )
            model.initialize_networks(input_dim)
            # Reload state dict to get encoder/decoder weights
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        history: Optional[int] = None,
        return_series: bool = True,
        return_factors: bool = True,
        target: Optional[List[str]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the Lightning module to generate forecasts.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        history : int, optional
            Number of historical periods to use for Kalman filter update before prediction.
            If None, uses full history (default). If specified (e.g., 60), uses only the most
            recent N periods for efficiency. Initial state (Z_0, V_0) is always estimated from
            full history (including any new data beyond training period).
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
        target : List[str], optional
            List of target series IDs to return. If None, uses target_series from DataModule.
            If DataModule has no target_series, raises ValueError.
            If specified, only returns predictions for the specified target series.
            Only target series are returned (features are excluded).
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x N) if target is None, else (horizon x len(target))
            If only return_factors is True:
                Z_forecast (horizon x m)
            
        Notes
        -----
        When history is specified, the method uses only the most recent N periods for
        Kalman filter update, improving computational efficiency. The initial state
        (Z_0, V_0) is always estimated from full history (including any new data beyond
        training period), ensuring accuracy while maintaining efficiency.
        
        When target is specified, only the target series are unstandardized and returned.
        Feature series are not unstandardized (no inverse transform needed).
        """
        if self.training_state is None:
            raise ValueError(
                f"DDFM prediction failed: model has not been trained yet. "
                f"Please call trainer.fit(model, data_module) first"
            )
        
        # Convert training state to result format for prediction
        if not hasattr(self, '_result') or self._result is None:
            self._result = self.get_result()
        
        if self._result is None:
            raise ValueError(
                f"DDFM prediction failed: model has not been fitted yet. "
                f"Please call trainer.fit(model, data_module) first"
            )
        
        # Compute default horizon
        if horizon is None:
            if self._config is not None:
                from ..config.utils import get_periods_per_year
                from ..utils.helpers import get_clock_frequency
                clock = get_clock_frequency(self._config, 'm')
                horizon = get_periods_per_year(clock)
            else:
                horizon = 12  # Default to 12 periods if no config
        
        # Validate horizon
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        
        # Extract parameters
        A = self._result.A  # Factor dynamics (m x m) for VAR(1) or (m x 2m) for VAR(2)
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        p = self._result.p  # VAR order
        
        # Update factor state with history if specified
        if history is not None and history > 0:
            Z_last_updated = self._update_factor_state_with_history(
                history=history,
                result=self._result,
                kalman_filter=None  # Will be created in _update_factor_state_ddfm if needed
            )
            if Z_last_updated is not None:
                Z_last = Z_last_updated
            else:
                # Fallback to training state if update failed
                Z_last = self._result.Z[-1, :]
        else:
            # Use training state (default behavior)
            Z_last = self._result.Z[-1, :]
        
        # Determine target indices first (for optimization: only transform target series)
        # If target is None, use target_series from DataModule (if available)
        if target is None:
            # Try to get target_series from DataModule
            try:
                data_module = self._get_datamodule()
                target_series = getattr(data_module, 'target_series', None)
                if target_series is not None and len(target_series) > 0:
                    target = target_series
            except (ValueError, AttributeError):
                # DataModule not available or no target_series - raise error
                target = None
        
        # Find target indices if target is specified
        target_indices = None
        if target is not None and len(target) > 0:
            # Get series IDs from config
            if self._config is not None:
                series_ids = self._config.get_series_ids()
            else:
                # Fallback: try to get from result
                series_ids = getattr(self._result, 'series_ids', None)
                if series_ids is None:
                    raise ValueError(
                        "DDFM prediction failed: target specified but cannot determine series IDs. "
                        "Please ensure config is loaded or result contains series_ids."
                    )
            
            # Find target indices
            target_indices = []
            for tgt_id in target:
                if tgt_id in series_ids:
                    target_indices.append(series_ids.index(tgt_id))
                else:
                    _logger.warning(
                        f"DDFM prediction: target series '{tgt_id}' not found in series_ids. "
                        f"Available series: {series_ids}"
                    )
            
            if len(target_indices) == 0:
                raise ValueError(
                    f"DDFM prediction failed: none of the specified target series found. "
                    f"Target: {target}, Available: {series_ids}"
                )
        elif target is None:
            raise ValueError(
                "DDFM prediction failed: target is None but no target_series found in DataModule. "
                "Please specify target=['series_id'] or ensure DataModule has target_series set."
            )
        
        # Forecast factors using VAR dynamics (common helper)
        Z_prev = self._result.Z[-2, :] if self._result.Z.shape[0] >= 2 and p == 2 else None
        Z_forecast = self._forecast_var_factors(
            Z_last=Z_last,
            A=A,
            p=p,
            horizon=horizon,
            Z_prev=Z_prev
        )
        
        # Optimized: Transform only target series (not all series)
        # Use only target indices for C, Mx, Wx
        C_target = C[target_indices, :]  # (len(target) x m)
        Mx_target = Mx[target_indices] if len(Mx) > max(target_indices) else Mx
        Wx_target = Wx[target_indices] if len(Wx) > max(target_indices) else Wx
        
        # Transform factors to target observations only
        X_forecast_std = Z_forecast @ C_target.T  # (horizon x len(target))
        X_forecast = X_forecast_std * Wx_target + Mx_target  # (horizon x len(target))
        
        # Convert to numpy (handles torch inputs)
        X_forecast = np.asarray(
            X_forecast.detach().cpu().numpy() if hasattr(X_forecast, "detach") else X_forecast
        )
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    def update(
        self,
        X_std: np.ndarray,
        *,
        history: Optional[int] = None,
        kalman_filter: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> 'DDFM':
        """Update factor state with standardized data.
        
        This method permanently updates the last factor state (result.Z[-1, :])
        using the provided standardized data. Users should handle all preprocessing
        (masking, imputation, standardization) before calling this method.
        
        Parameters
        ----------
        X_std : np.ndarray
            Standardized data array (T x N), where T is number of time periods
            and N is number of series. Data should already be standardized using
            result.Mx and result.Wx.
        history : int, optional
            Number of recent periods to use for factor state update. If None, uses
            all provided data (default). If specified (e.g., 60), uses only the most
            recent N periods. Initial state (Z_0, V_0) is always estimated from
            full training data, but the update uses only recent history for efficiency.
        kalman_filter : Any, optional
            Kalman filter instance. If None, uses default or model's kalman filter.
            
        Returns
        -------
        DDFM
            Self for method chaining
            
        Examples
        --------
        >>> # Update state with new data, then predict
        >>> model.update(X_std).predict(horizon=1)
        >>> # Or update with only recent 12 periods
        >>> model.update(X_std, history=12)
        >>> forecast = model.predict(horizon=6)
        """
        self._check_trained()
        
        # Optionally replace scaler (e.g., if refit on new regime)
        if scaler is not None:
            self.scaler = scaler
        
        result = self.result  # Use property which ensures non-None after _check_trained()
        
        # Validate input shape
        if not isinstance(X_std, np.ndarray):
            X_std = np.asarray(X_std)
        if X_std.ndim != 2:
            raise ValueError(
                f"DDFM update(): X_std must be 2D array (T x N), "
                f"got shape {X_std.shape}"
            )
        
        # Handle NaN/Inf values
        X_std = np.where(np.isfinite(X_std), X_std, np.nan)
        
        # Filter to recent history if specified
        # Note: Initial state (Z_0, V_0) from result is estimated from full training data,
        # but we use only recent history for the update step
        if history is not None and history > 0:
            if X_std.shape[0] > history:
                X_recent = X_std[-history:, :]
                _logger.debug(
                    f"DDFM update(): Using {history} most recent periods out of {X_std.shape[0]} total periods"
                )
            else:
                X_recent = X_std
                _logger.debug(
                    f"DDFM update(): history={history} specified but data has only {X_std.shape[0]} periods, using all data"
                )
        else:
            X_recent = X_std
        
        # Update factor state using encoder to extract factors, then Kalman filter
        Z_last_updated = self._update_factor_state_ddfm(
            X_recent, result, kalman_filter
        )
        
        # Update result.Z[-1, :] permanently
        if Z_last_updated is not None:
            result.Z[-1, :] = Z_last_updated
        else:
            _logger.warning(
                f"DDFM update(): Failed to update factor state, "
                f"keeping current state"
            )
            
        return self
    
    @property
    def result(self) -> DDFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        # Check if trained and extract result from training state if needed
        self._check_trained()
        return self._result
    
    
    
    def reset(self) -> 'DDFM':
        """Reset model state."""
        super().reset()
        return self

