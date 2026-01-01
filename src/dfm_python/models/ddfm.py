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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Sequence

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from ..config import (
    ConfigSource,
    DFMConfig,
    make_config_source,
)
from ..config import DDFMResult
from ..config import get_periods_per_year
from ..encoder.simple_encoder import Encoder, extract_decoder_params
from ..decoder.linear import Decoder
from ..decoder.mlp import MLPDecoder
from ..logger import get_logger
from ..numeric.stability import rem_nans_spline
from ..utils.misc import (
    get_clock_frequency,
    TimeIndex,
)
from ..numeric.estimator import (
    estimate_var,
)
from .base import BaseFactorModel
from ..utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    ConfigurationError,
    DataError,
    DataValidationError,
    PredictionError,
    NumericalError
)
from ..utils.validation import check_condition, has_shape_with_min_dims
from ..utils.common import ensure_numpy, sanitize_array, ensure_tensor
from ..config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_ZERO_VALUE,
    MAX_WARNING_ITEMS,
    MIN_EIGENVALUE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    MAX_EIGENVALUE,
    DEFAULT_MIN_OBS_VAR,
    VAR_STABILITY_THRESHOLD,
    DEFAULT_AR_COEF,
    PERFECT_CORR_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_IDENTITY_SCALE,
    HUBER_QUADRATIC_COEFF,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_EPSILON,
    DEFAULT_DTYPE,
    DEFAULT_ENCODER_LAYERS,
    MIN_STD,
    DEFAULT_LR_DECAY_RATE,
    MIN_DIAGONAL_VARIANCE,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
    MIN_VARIABLES,
    MIN_DDFM_TIME_STEPS,
)

if TYPE_CHECKING:
    from ..datamodule import DFMDataModule

import pytorch_lightning as pl

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


class DDFM(BaseFactorModel, pl.LightningModule):
    """High-level API for Deep Dynamic Factor Model (PyTorch Lightning module).
    
    This class is a PyTorch Lightning module that can be used with standard
    Lightning training patterns. It inherits from both BaseFactorModel and
    pl.LightningModule, and implements DDFM training using autoencoder and MCMC procedure.
    
    Note: Factors use AR(1) dynamics (simplified).
    
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
        >>> trainer = DDFMTrainer(max_epochs=100)  # DEFAULT_MAX_EPOCHS
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
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_idiosyncratic: bool = True,
        min_obs_idio: Optional[int] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: int = 10,
        seed: Optional[int] = None,
        decay_learning_rate: bool = True,
        min_obs_pretrain: int = 50,
        mult_epoch_pretrain: int = 1,
        loss_function: str = 'mse',
        huber_delta: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_val: Optional[float] = None,
        decoder: str = "linear",
        decoder_layers: Optional[List[int]] = None,
        **kwargs: Any
    ) -> None:
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
        learning_rate : float, default DEFAULT_DDFM_LEARNING_RATE (0.005)
            Learning rate for Adam optimizer (matches original DDFM default)
        epochs : int, optional
            Number of epochs per MCMC iteration (default: DEFAULT_MAX_EPOCHS = 100)
        batch_size : int, default 100
            Batch size for training (matches original DDFM)
        use_idiosyncratic : bool, default True
            Whether to model idiosyncratic components
        min_obs_idio : int, default 5
            Minimum observations for idio AR(1) estimation
        max_iter : int, default 200
            Maximum number of MCMC iterations
        tolerance : float, default DEFAULT_TOLERANCE (0.0005)
            Convergence tolerance
        disp : int, default 10
            Display progress every 'disp' iterations
        decay_learning_rate : bool, default True
            Whether to use exponential decay learning rate scheduler (matches original DDFM)
        min_obs_pretrain : int, default 50
            Minimum number of observations for pre-training without interpolation
        mult_epoch_pretrain : int, default 1
            Multiplier for number of epochs during pre-training
        loss_function : str, default 'mse'
            Loss function for training ('mse', 'huber'). 
            'mse': Mean squared error (default, matches original DDFM)
            'huber': Huber loss (more robust to outliers)
        huber_delta : float, default DEFAULT_HUBER_DELTA (1.0)
            Delta parameter for Huber loss (only used if loss_function='huber').
            Controls the transition point between quadratic and linear regions.
        weight_decay : float, default DEFAULT_WEIGHT_DECAY (0.0)
            Weight decay (L2 regularization) for optimizer. Helps prevent overfitting to linear features.
            Recommended: 1e-5 to 1e-3 for deeper encoders or when encoder collapses to linear behavior.
        grad_clip_val : float, default DEFAULT_GRAD_CLIP_VAL (1.0)
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
        **kwargs : Any
            Additional arguments passed to BaseFactorModel (for API consistency with KDFM/DFM)
            
        Returns
        -------
        None
            Initializes DDFM instance in-place.
            
        Raises
        ------
        ConfigurationError
            If config validation fails or required parameters are missing.
        ValueError
            If invalid activation/decoder is specified.
        """
        BaseFactorModel.__init__(self)
        pl.LightningModule.__init__(self)
        
        # Initialize config using consolidated helper method
        # DDFM does not use block structure
        config = self._initialize_config(config)
        
        self.encoder_layers = encoder_layers or DEFAULT_ENCODER_LAYERS
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.decoder_type = decoder
        self.decoder_layers = decoder_layers
        # Import constants for defaults
        from ..config.constants import (
            DEFAULT_DDFM_LEARNING_RATE, DEFAULT_DDFM_BATCH_SIZE,
            DEFAULT_MAX_EPOCHS, DEFAULT_MAX_MCMC_ITER, DEFAULT_TOLERANCE, DEFAULT_MIN_OBS_IDIO,
            DEFAULT_HUBER_DELTA, DEFAULT_WEIGHT_DECAY, DEFAULT_GRAD_CLIP_VAL
        )
        
        # Resolve parameters using consolidated helper
        from ..utils.misc import resolve_param
        self.learning_rate = resolve_param(learning_rate, default=DEFAULT_DDFM_LEARNING_RATE)
        self.epochs_per_iter = resolve_param(epochs, default=DEFAULT_MAX_EPOCHS)
        self.batch_size = resolve_param(batch_size, default=DEFAULT_DDFM_BATCH_SIZE)
        self.factor_order = 1  # Factors always use AR(1) dynamics (simplified)
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = resolve_param(min_obs_idio, default=DEFAULT_MIN_OBS_IDIO)
        self.max_iter = resolve_param(max_iter, default=DEFAULT_MAX_MCMC_ITER)
        self.tolerance = resolve_param(tolerance, default=DEFAULT_TOLERANCE)
        self.disp = disp
        self.decay_learning_rate = decay_learning_rate
        self.min_obs_pretrain = min_obs_pretrain
        self.mult_epoch_pretrain = mult_epoch_pretrain
        self.loss_function = loss_function.lower()
        self.huber_delta = resolve_param(huber_delta, default=DEFAULT_HUBER_DELTA)
        self.weight_decay = resolve_param(weight_decay, default=DEFAULT_WEIGHT_DECAY)
        self.grad_clip_val = resolve_param(grad_clip_val, default=DEFAULT_GRAD_CLIP_VAL)
        
        # Validate loss function
        check_condition(
            self.loss_function in ['mse', 'huber'],
            ConfigurationError,
            f"DDFM initialization failed: loss_function must be 'mse' or 'huber', got '{loss_function}'",
            details="Valid loss functions are 'mse' (mean squared error) or 'huber' (Huber loss)"
        )
        
        # Validate gradient clipping value
        check_condition(
            self.grad_clip_val >= DEFAULT_ZERO_VALUE,
            ConfigurationError,
            f"DDFM initialization failed: grad_clip_val must be >= {DEFAULT_ZERO_VALUE}, got {grad_clip_val}",
            details="Gradient clipping value must be non-negative (0.0 disables clipping)"
        )
        
        # Determine number of factors
        # DDFM does not use block structure - num_factors is specified directly
        if num_factors is None:
            # Try to get from config num_factors (DDFM-specific parameter)
            from ..utils.helper import get_config_attr
            num_factors_from_config = get_config_attr(config, 'num_factors', None)
            if num_factors_from_config is not None:
                self.num_factors = num_factors_from_config
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
        # Default seed for reproducibility (when not specified)
        self.rng = np.random.RandomState(resolve_param(seed, default=DEFAULT_SEED))
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize encoder and decoder when data dimensions are known.
        
        This is called by Lightning before configure_optimizers(), so we need
        to initialize encoder/decoder here if datamodule is available.
        If not available here, will be initialized in on_train_start().
        """
        # Access datamodule if available (trainer should be attached by now)
        trainer = getattr(self, 'trainer', None)
        if trainer is not None:
            datamodule = getattr(trainer, 'datamodule', None)
            if datamodule is not None:
                self._data_module = datamodule
                try:
                    # Get data to determine input dimension
                    # datamodule.setup() should have been called by Lightning already
                    X_torch = self._data_module.get_processed_data()
                    input_dim = X_torch.shape[1]
                    
                    # Initialize networks if not already initialized
                    if not self._are_networks_initialized():
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
        # Helper method for consistent exception handling during initialization
        def _handle_initialization_error(
            component_name: str,
            error: Exception,
            component_specific_details: str
        ) -> None:
            """Handle initialization errors consistently.
            
            Args:
                component_name: Name of component being initialized (e.g., "encoder", "decoder")
                error: The exception that occurred
                component_specific_details: Component-specific error details and suggestions
            """
            raise ModelNotInitializedError(
                f"DDFM {component_name} initialization failed: failed to initialize {component_name}: {type(error).__name__}: {str(error)}",
                details=component_specific_details
            ) from error
        
        try:
            self.encoder = Encoder(
                input_dim=input_dim,
                hidden_dims=self.encoder_layers,
                output_dim=self.num_factors,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            _handle_initialization_error(
                "encoder",
                e,
                f"Check encoder_layers={self.encoder_layers}, num_factors={self.num_factors}, input_dim={input_dim}. Suggestions: (1) Ensure input_dim > 0, (2) Reduce encoder_layers size if too large, (3) Ensure num_factors > 0 and num_factors <= input_dim, (4) Check that encoder_layers values are positive integers"
            )
        
        try:
            # Create decoder based on decoder_type
            if self.decoder_type == "linear":
                self.decoder = Decoder(
                    input_dim=self.num_factors,
                    output_dim=input_dim,
                    use_bias=True,
                )
                # Validate decoder weights are not all zeros (initialization check)
                decoder_weight = ensure_numpy(self.decoder.decoder.weight.data)
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
                decoder_weight = ensure_numpy(self.decoder.layers[0].weight.data)
            else:
                check_condition(
                    False,  # Always fails - decoder_type is invalid
                    ConfigurationError,
                    f"DDFM decoder initialization failed: decoder must be 'linear' or 'mlp', got '{self.decoder_type}'",
                    details="Valid decoder types are 'linear' (LinearDecoder) or 'mlp' (MLPDecoder)"
                )
            
            # Validate decoder weights are not all zeros (initialization check)
            check_condition(
                not np.allclose(decoder_weight, DEFAULT_ZERO_VALUE, atol=MIN_EIGENVALUE),
                ModelNotInitializedError,
                f"DDFM decoder initialization failed: decoder weights are all zeros after initialization",
                details=f"This indicates a problem with decoder initialization. Check: (1) Decoder class implementation, (2) Weight initialization method, (3) PyTorch version compatibility. Decoder weight shape: {decoder_weight.shape}, weight mean: {np.mean(decoder_weight):.6f}, weight std: {np.std(decoder_weight):.6f}"
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
            _handle_initialization_error(
                "decoder",
                e,
                f"Check num_factors={self.num_factors}, input_dim={input_dim}. Suggestions: (1) Ensure num_factors > 0, (2) Ensure input_dim > 0, (3) Check that num_factors <= input_dim"
            )
    
    def _check_networks_initialized(self) -> None:
        """Check if encoder and decoder are initialized."""
        check_condition(
            self.encoder is not None and self.decoder is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__}: encoder and decoder must be initialized",
            details="Please call _initialize_encoder_decoder() before using the model. Ensure setup() or on_train_start() has been called."
        )
    
    def _are_networks_initialized(self) -> bool:
        """Check if encoder and decoder are initialized (returns bool instead of raising)."""
        return self.encoder is not None and self.decoder is not None
    
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
        if isinstance(batch, Sequence) and not isinstance(batch, str) and len(batch) == 2:
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
        from ..config.constants import DEFAULT_DDFM_CLIP_RANGE_DEEP, DEFAULT_DDFM_CLIP_RANGE_SHALLOW
        clip_range = DEFAULT_DDFM_CLIP_RANGE_DEEP if len(self.encoder_layers) > 2 else DEFAULT_DDFM_CLIP_RANGE_SHALLOW
        data_clipped = torch.clamp(data, min=-clip_range, max=clip_range)
        
        # Forward pass
        reconstructed = self.forward(data_clipped)
        
        # Check for NaN/Inf in forward pass output
        if not torch.all(torch.isfinite(reconstructed)):
            nan_count = torch.sum(torch.isnan(reconstructed)).item()
            inf_count = torch.sum(torch.isinf(reconstructed)).item()
            error_msg = (
                f"DDFM training_step failed: Forward pass produced {nan_count} NaN and {inf_count} Inf values. "
                f"This indicates numerical instability. Possible causes: (1) Learning rate too high, "
                f"(2) Gradient explosion, (3) Invalid input data, (4) Model architecture mismatch."
            )
            _logger.error(error_msg)
            from ..utils.errors import NumericalError
            raise NumericalError(
                error_msg,
                details=f"NaN count: {nan_count}, Inf count: {inf_count}. "
                       f"Consider: (1) Reducing learning rate, (2) Adding gradient clipping, "
                       f"(3) Checking input data for NaN/Inf, (4) Verifying encoder/decoder dimensions."
            )
        
        # Compute loss with missing data masking
        mask = torch.isfinite(target)
        target_clean = torch.where(mask, target, torch.zeros_like(target))
        
        # Use specified loss function
        if self.loss_function == 'huber':
            # Huber loss: more robust to outliers
            # L_delta(a) = HUBER_QUADRATIC_COEFF * a^2 if |a| <= delta, else delta * (|a| - HUBER_QUADRATIC_COEFF * delta)
            diff = target_clean - reconstructed
            abs_diff = torch.abs(diff)
            huber_loss = torch.where(
                abs_diff <= self.huber_delta,
                HUBER_QUADRATIC_COEFF * diff ** 2,
                self.huber_delta * (abs_diff - HUBER_QUADRATIC_COEFF * self.huber_delta)
            )
            loss = torch.sum(huber_loss * mask) / (torch.sum(mask) + DEFAULT_EPSILON)
        else:
            # MSE loss (default)
            squared_diff = (target_clean - reconstructed) ** 2
            loss = torch.sum(squared_diff * mask) / (torch.sum(mask) + DEFAULT_EPSILON)
        
        # Handle NaN/Inf in loss
        if not torch.isfinite(loss):
            error_msg = (
                f"DDFM training_step failed: Loss is NaN/Inf. "
                f"This indicates numerical instability in the loss computation. "
                f"Possible causes: (1) Division by zero in loss calculation, "
                f"(2) Invalid target values, (3) Model output contains extreme values."
            )
            _logger.error(error_msg)
            from ..utils.errors import NumericalError
            raise NumericalError(
                error_msg,
                details="Loss computation produced NaN/Inf. Check: (1) Target data for NaN/Inf, "
                       f"(2) Model output values, (3) Loss function implementation. "
                       f"Current loss value: {loss.item() if hasattr(loss, 'item') else loss}."
            )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Note: Gradient clipping is handled automatically by Lightning trainer if gradient_clip_val is set
        # The grad_clip_val parameter is used in pre_train() and MCMC training for manual training loops
        
        return loss
    
    
    def _validate_factors(self, factors: np.ndarray, operation: str = "operation") -> np.ndarray:
        """Validate and normalize factors shape and content quality.
        
        Inlined from validate_factors_ddfm() utility function.
        """
        factors = ensure_numpy(factors)
        if factors.ndim == 0 or factors.size == 0:
            raise DataError(
                f"DDFM {operation} failed: factors is empty or invalid (shape: {factors.shape})",
                details="This indicates training did not complete properly"
            )
        if factors.ndim == 1:
            factors = factors.reshape(-1, self.num_factors) if factors.size > 0 else factors.reshape(0, self.num_factors)
        if factors.ndim != 2:
            raise DataError(
                f"DDFM {operation} failed: factors must be 2D array (T x m), got shape {factors.shape}",
                details="Factors should be a 2D array with shape (T, m) where T is time steps and m is number of factors"
            )
        
        T, m = factors.shape
        
        # Validate factors are finite
        from ..numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(factors, name=f"factors ({operation})")
        
        return factors
    
    def _validate_training_data(
        self,
        X_torch: torch.Tensor,
        operation: str = "training setup"
    ) -> None:
        """Validate data dimensions and model configuration before training starts.
        
        Inlined from validate_training_data_ddfm() utility function.
        """
        check_condition(
            X_torch is not None,
            DataError,
            f"DDFM {operation} failed: X_torch is None",
            details="Please provide training data"
        )
        
        check_condition(
            isinstance(X_torch, torch.Tensor),
            DataError,
            f"DDFM {operation} failed: X_torch must be torch.Tensor, got {type(X_torch)}",
            details="Training data must be a torch.Tensor. Convert numpy arrays using torch.from_numpy()"
        )
        
        check_condition(
            X_torch.ndim == 2,
            DataError,
            f"DDFM {operation} failed: X_torch must be 2D (T x N), got shape {X_torch.shape}",
            details="Training data must be 2D with shape (T, N) where T is time steps and N is number of series"
        )
        
        T, N = X_torch.shape
        
        check_condition(
            T >= MIN_DDFM_TIME_STEPS,
            DataError,
            f"DDFM {operation} failed: Need at least {MIN_DDFM_TIME_STEPS} time periods, got T={T}",
            details="DDFM requires at least 2 time periods for training"
        )
        
        check_condition(
            N >= MIN_VARIABLES,
            DataError,
            f"DDFM {operation} failed: Need at least {MIN_VARIABLES} series, got N={N}",
            details="DDFM requires at least 1 series (variable) in the data"
        )
        
        check_condition(
            self.num_factors is not None and self.num_factors >= 1,
            ConfigurationError,
            f"DDFM {operation} failed: num_factors must be >= 1, got {self.num_factors}",
            details="Number of factors must be a positive integer"
        )
        
        check_condition(
            self.num_factors <= N,
            ConfigurationError,
            f"DDFM {operation} failed: num_factors ({self.num_factors}) cannot exceed number of series (N={N})",
            details="Number of factors cannot exceed the number of input series"
        )
        
        if self.encoder_layers is not None and len(self.encoder_layers) > 0:
            if self.encoder_layers[0] != N:
                _logger.warning(
                    f"DDFM {operation}: encoder_layers[0] ({self.encoder_layers[0]}) does not match input dimension (N={N}). "
                    "Encoder will be reinitialized with correct input dimension."
                )
        
        if self.encoder is not None:
            if hasattr(self.encoder, 'input_dim') and self.encoder.input_dim != N:
                raise ConfigurationError(
                    f"DDFM {operation} failed: encoder.input_dim ({self.encoder.input_dim}) must match input dimension (N={N})",
                    details="Encoder input dimension must match the number of series in the data"
                )
    
    
    def _estimate_var(
        self, 
        factors: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate VAR dynamics with comprehensive error handling and fallback.
        
        Estimates VAR(p) coefficients from factor time series using OLS regression.
        Includes fallback to identity matrix if estimation fails.
        
        Parameters
        ----------
        factors : np.ndarray
            Factor time series of shape (T, m) where T is time steps and m is number of factors
            
        Returns
        -------
        A_f : np.ndarray
            VAR transition matrix of shape (m, m) for AR(1)
        Q_f : np.ndarray
            Innovation covariance matrix of shape (m, m), positive definite
            
        Raises
        ------
        ConfigurationError
        NumericalError
            If estimation produces invalid results (handled internally with fallback)
            
        Notes
        -----
        - Uses OLS regression to estimate VAR coefficients
        - Ensures Q_f is positive definite with minimum eigenvalue >= MIN_EIGENVALUE
        - Falls back to identity matrix with small regularization if estimation fails
        - Inlined from estimate_var_ddfm() utility function for better integration
        """
        factors = ensure_numpy(factors)
        if factors.ndim == 0 or factors.size == 0 or factors.ndim < 2 or factors.shape[0] < 2 or factors.shape[1] == 0:
            m = self.num_factors if self.num_factors else 1
            from ..numeric.stability import create_scaled_identity
            return create_scaled_identity(m, DEFAULT_IDENTITY_SCALE), create_scaled_identity(m, DEFAULT_REGULARIZATION)
        
        T, m = factors.shape
        
        min_obs_required = 1 + DEFAULT_MIN_OBS_VAR - 2  # AR(1) requires at least 2 observations
        if T < min_obs_required:
            from ..numeric.stability import create_scaled_identity
            factor_var = np.var(factors, axis=0)
            factor_var = np.maximum(factor_var, MIN_STD ** 2)
            # AR(1) dynamics (simplified)
            if True:
                A_f = create_scaled_identity(m, DEFAULT_AR_COEF)
            else:
                A_f = np.hstack([create_scaled_identity(m, DEFAULT_AR_COEF), np.zeros((m, m), dtype=DEFAULT_DTYPE)])
            Q_f = np.diag(factor_var)
            return A_f, Q_f
        
        if not np.all(np.isfinite(factors)):
            factors = sanitize_array(factors)
        
        # VAR estimation error handling: raises ConfigurationError for parameter validation failures
        # This is intentionally different from _handle_initialization_error() which raises ModelNotInitializedError
        # VAR estimation failures are configuration/parameter issues, not initialization failures
        try:
            A_f, Q_f = estimate_var(factors, order=1)  # Always use AR(1)
        except (ValueError, RuntimeError) as e:
            raise ConfigurationError(
                "DDFM VAR estimation failed: factors always use AR(1) dynamics"
            ) from e
        
        if Q_f.ndim == 0:
            factor_var = np.var(factors, axis=0)
            factor_var = np.maximum(factor_var, MIN_EIGENVALUE)
            Q_f = np.diag(factor_var)
        elif Q_f.ndim != 2:
            if Q_f.size == m ** 2:
                Q_f = Q_f.reshape(m, m)
            else:
                factor_var = np.var(factors, axis=0)
                factor_var = np.maximum(factor_var, MIN_EIGENVALUE)
                Q_f = np.diag(factor_var)
        
        Q_sym = (Q_f + Q_f.T) / 2
        eigenvals_Q = np.linalg.eigvalsh(Q_sym)
        min_eigenval_Q = np.min(eigenvals_Q)
        if min_eigenval_Q < MIN_EIGENVALUE:
            from ..numeric.stability import create_scaled_identity
            Q_f = Q_sym + create_scaled_identity(m, MIN_EIGENVALUE - min_eigenval_Q)
        else:
            Q_f = Q_sym
        
        return A_f, Q_f
    
    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], Dict[str, Any]]:
        """Configure optimizer and learning rate scheduler for autoencoder training.
        
        Matches original DDFM implementation with exponential decay scheduler.
        
        Returns
        -------
        List[torch.optim.Optimizer] or Dict
            If decay_learning_rate=False: List containing the optimizer
            If decay_learning_rate=True: Dict with optimizer and scheduler config
        """
        if not self._are_networks_initialized():
            _logger.warning("Encoder/decoder not initialized, creating placeholder optimizer")
            optimizer = self._create_dummy_optimizer(self.learning_rate)
            if self.decay_learning_rate:
                return self._create_lr_scheduler(optimizer)
            return [optimizer]
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.decay_learning_rate:
            return self._create_lr_scheduler(optimizer)
        
        return [optimizer]
    
    def _create_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Create learning rate scheduler configuration for Lightning.
        
        Helper method to consolidate scheduler creation logic.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to attach scheduler to
            
        Returns
        -------
        Dict[str, Any]
            Lightning scheduler configuration dict
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=DEFAULT_LR_DECAY_RATE
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
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
            decay_rate = DEFAULT_LR_DECAY_RATE
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
        x_clean_np = ensure_numpy(x_clean)
        missing_mask_np = ensure_numpy(missing_mask)
        
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
        inpt_tensor = ensure_tensor(inpt_pre_train, device=device, dtype=DEFAULT_TORCH_DTYPE)
        oupt_tensor = ensure_tensor(oupt_pre_train, device=device, dtype=DEFAULT_TORCH_DTYPE)
        
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
            epoch_loss = DEFAULT_ZERO_VALUE
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
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + DEFAULT_EPSILON)
                else:
                    # Standard MSE (no missing values)
                    loss = nn.functional.mse_loss(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if self.grad_clip_val > DEFAULT_ZERO_VALUE:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        max_norm=self.grad_clip_val
                    )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % max(1, num_epochs // DEFAULT_LOG_INTERVAL) == 0 or epoch == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else DEFAULT_ZERO_VALUE
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
            self.tolerance (default: DEFAULT_TOLERANCE = 0.0005). Training stops when |MSE_new - MSE_old| < tolerance.
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
        >>> from dfm_python.config.constants import DEFAULT_TOLERANCE
        >>> state = model.fit_mcmc(X, x_clean, missing_mask, max_iter=50, tolerance=DEFAULT_TOLERANCE)
        >>> factors = state.factors  # (T x 2) factor estimates
        >>> print(f"Converged: {state.converged}, Iterations: {state.num_iter}")
        """
        from ..trainer.ddfm import DDFMDenoisingTrainer
        
        # Store processed data for shape validation in update()
        self.data_processed = ensure_numpy(X, dtype=DEFAULT_DTYPE)
        
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
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"{self.__class__.__name__} get_result failed: model has not been fitted yet",
            details="Please call fit_mcmc() first"
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
        C_zero_ratio = DEFAULT_IDENTITY_SCALE - (C_nonzero / C.size)
        _logger.info(
            f"DDFM get_result: C matrix statistics - mean={C_mean:.6f}, std={C_std:.6f}, "
            f"min={C_min:.6f}, max={C_max:.6f}, nonzero={C_nonzero}/{C.size} ({DEFAULT_IDENTITY_SCALE-C_zero_ratio:.1%}), "
            f"zero_ratio={C_zero_ratio:.1%}"
        )
        
        # Validate C matrix for NaN/Inf (extract_decoder_params should handle this, but double-check)
        from ..numeric.validator import validate_no_nan_inf
        try:
            validate_no_nan_inf(C, name="C matrix")
        except NumericalError as e:
            # Provide more detailed error message for C matrix
            nan_count = np.sum(np.isnan(C)) if isinstance(C, np.ndarray) else 0
            nan_ratio = nan_count / C.size if hasattr(C, 'size') and C.size > 0 else 0.0
            raise NumericalError(
                f"DDFM get_result failed: C matrix contains NaN/Inf values after extraction. "
                f"This indicates severe numerical instability. The model cannot be used for prediction.",
                details=(
                    f"NaN count: {nan_count}, NaN ratio: {nan_ratio:.1%}. "
                    f"Consider: (1) reducing learning rate, (2) adding gradient clipping, "
                    f"(3) checking data quality, (4) reducing model complexity, (5) checking encoder/decoder initialization."
                )
            ) from e
        
        # Get factors and prediction
        factors = self.training_state.factors  # T x num_factors
        prediction_iter = self.training_state.prediction  # T x N
        
        # Validate and normalize factors shape
        factors = self._validate_factors(factors, operation="get_result")
        
        # Convert to numpy
        C = ensure_numpy(C)
        bias = ensure_numpy(bias)
        
        # Compute residuals and estimate idiosyncratic dynamics
        if self.data_processed is not None:
            x_standardized = ensure_numpy(self.data_processed)
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
        A_f, Q_f = self._estimate_var(factors, 1)  # Always use AR(1)
        
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
        R = np.diag(np.maximum(R_diag, MIN_DIAGONAL_VARIANCE))
        
        # Compute smoothed data
        x_sm = prediction_iter  # T x N (standardized, already preprocessed)
        
        # Unstandardize: Data is already preprocessed, so for most series Mx=0, Wx=1
        # Only target series (if target_scaler provided) need inverse transformation
        n_series = C.shape[0]
        
        # Data is already standardized (mean=0, std=1) - no transformation needed
        # This is just for internal consistency in result structure
        
        # Target scaler is stored in self.target_scaler (from base class)
        # Use scaler.inverse_transform() for unstandardization, not Mx/Wx arrays
        
        # Create result object
        r = np.array([self.num_factors])
        
        # Get target scaler from datamodule if available
        target_scaler = getattr(self, 'target_scaler', None)
        
        result = DDFMResult(
            x_sm=x_sm,
            Z=factors,  # T x m
            C=C,
            R=R,
            A=A,
            Q=Q,
            target_scaler=target_scaler,
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=1,  # Always use AR(1)
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=None,  # DDFM doesn't compute loglik in same way
            training_loss=self.training_state.training_loss,
            encoder_layers=self.encoder_layers,
            use_idiosyncratic=self.use_idiosyncratic,
        )
        
        return result
    
    def on_train_end(self) -> None:
        """Called when training ends. Automatically computes result from training state."""
        # Automatically compute result after training completes
        # Error handling: This is a fallback pattern (graceful degradation) - logs warning instead of raising exception
        # Intentionally different from _handle_initialization_error() which raises ModelNotInitializedError
        # Result computation can fail gracefully and be retried later (on first access to result property or predict())
        if self.training_state is not None:
            try:
                if self._result is None:
                    self._result = self.get_result()
            except (ValueError, RuntimeError, AttributeError) as e:
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
        if not self._are_networks_initialized():
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
            from ..utils.helper import get_config_attr
            nan_method = get_config_attr(self.config, 'nan_method', DEFAULT_NAN_METHOD)
            nan_k = get_config_attr(self.config, 'nan_k', DEFAULT_NAN_K)
            
            # Check if data has NaN values
            X_np = ensure_numpy(X_torch)
            has_nan = np.any(np.isnan(X_np))
            
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
                x_clean_torch = ensure_tensor(x_clean_np, dtype=DEFAULT_TORCH_DTYPE, device=X_torch.device)
            else:
                # No NaN values - use data as-is
                device = X_torch.device if isinstance(X_torch, torch.Tensor) else None
                x_clean_torch = ensure_tensor(X_torch, dtype=DEFAULT_TORCH_DTYPE, device=device)
                missing_mask = np.zeros(X_np.shape, dtype=bool)  # No missing data
            
            # Replace any remaining NaN/Inf with zeros (defensive check)
            device = x_clean_torch.device
            dtype = x_clean_torch.dtype
            x_clean_torch = torch.where(
                torch.isfinite(x_clean_torch),
                x_clean_torch,
                torch.tensor(DEFAULT_ZERO_VALUE, device=device, dtype=dtype)
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
            # Error handling: This is a fallback pattern (graceful degradation) - logs warning instead of raising exception
            # Intentionally different from _handle_initialization_error() which raises ModelNotInitializedError
            # Pre-training can fail gracefully and MCMC training continues without it
            try:
                self.pre_train(
                    X=x_clean_torch,
                    x_clean=x_clean_torch,
                    missing_mask=missing_mask,
                    device=x_clean_torch.device,
                )
            except (RuntimeError, ValueError, AttributeError, OSError) as e:
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
        
        pl.LightningModule.on_train_start(self)
    
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> 'DDFM':
        """Load configuration from various sources."""
        # Preserve explicitly set num_factors if it was set during initialization
        preserved_num_factors = None
        num_factors_explicit = getattr(self, '_num_factors_explicit', None)
        if num_factors_explicit:
            preserved_num_factors = self.num_factors
        
        self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
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
    def _parse_checkpoint(checkpoint: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            return data.shape[1] if has_shape_with_min_dims(data, min_dims=2) else 1
        elif isinstance(data, torch.Tensor):
            return data.shape[1] if has_shape_with_min_dims(data, min_dims=2) else 1
        else:
            raise DataError(
                f"Unsupported data type: {type(data)}",
                details="Data must be pandas.DataFrame, numpy.ndarray, or torch.Tensor"
            )
    
    @staticmethod
    def _infer_input_dim(state_dict: Dict[str, Any]) -> Optional[int]:
        """Infer input_dim from checkpoint state_dict."""
        if not isinstance(state_dict, dict):
            raise DataValidationError(
                "Cannot infer input_dim: state_dict must be a dictionary",
                details=f"Received type: {type(state_dict).__name__}"
            )
        
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
        
        raise DataValidationError(
            "Cannot infer input_dim: no matching decoder weight keys found in state_dict",
            details=f"Checked keys: {list(state_dict.keys())[:MAX_WARNING_ITEMS]}{'...' if len(state_dict.keys()) > MAX_WARNING_ITEMS else ''}"
        )
    
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
        except (AttributeError, KeyError, RuntimeError, OSError) as e:
            # Not a Lightning checkpoint, will load as state_dict
            pass
        
        # Load checkpoint and extract state_dict/hparams
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        state_dict, hparams = cls._parse_checkpoint(checkpoint)
        
        # Infer input_dim: prioritize checkpoint, then data, then explicit parameter
        checkpoint_input_dim = cls._infer_input_dim(state_dict)
        
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
                raise ConfigurationError(
                    "Cannot determine input_dim. Please provide either 'data' or 'input_dim' parameter",
                    details="input_dim is required to initialize encoder/decoder"
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
        if not model._are_networks_initialized():
            if input_dim is None:
                raise ConfigurationError(
                    "Cannot initialize encoder/decoder: input_dim is required",
                    details="Please provide either 'data' or 'input_dim' parameter"
                )
            model.initialize_networks(input_dim)
            # Reload state dict to get encoder/decoder weights
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the Lightning module to generate forecasts.
        
        Target series are determined from the DataModule's target_series attribute,
        which should be set during DataModule initialization.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x len(target_series))
            If only return_factors is True:
                Z_forecast (horizon x m)
            
        Raises
        ------
        ValueError
            If DataModule has no target_series set
        """
        # Validate model is trained
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"DDFM prediction failed: model has not been trained yet",
            details="Please call trainer.fit(model, data_module) first"
        )
        
        # Get result (compute if needed)
        result = self._ensure_result()
        
        # Compute and validate horizon
        if horizon is None:
            horizon = self._compute_default_horizon()
        from ..numeric.validator import validate_horizon
        horizon = validate_horizon(horizon)
        
        # Extract parameters
        A = result.A  # Factor dynamics (m x m) for AR(1)
        C = result.C
        target_scaler = result.target_scaler  # Use scaler object instead of Mx/Wx
        p = result.p  # VAR order
        
        # Use training state for initial factor state
        Z_last = result.Z[-1, :]
        
        # Resolve target series from DataModule (target_series should be set at initialization)
        # _resolve_target_series() already validates and raises DataError if none found
        series_ids = self._config.get_series_ids() if self._config is not None else getattr(result, 'series_ids', None)
        target_series, target_indices = self._resolve_target_series(series_ids, result)
        
        # Additional validation: ensure target_series was set in DataModule
        if target_series is None or len(target_series) == 0:
            raise PredictionError(
                "DDFM prediction failed: no target_series found in DataModule",
                details="Please set target_series when creating the DataModule (e.g., DDFMDataModule(..., target_series=['series_id']))."
            )
        
        # Forecast factors using VAR dynamics (common helper)
        Z_prev = result.Z[-2, :] if result.Z.shape[0] >= 2 and p == 2 else None
        Z_forecast = self._forecast_var_factors(
            Z_last=Z_last,
            A=A,
            p=p,
            horizon=horizon,
            Z_prev=Z_prev
        )
        
        # Optimized: Transform only target series (not all series)
        # Use only target indices for C
        C_target = C[target_indices, :]  # (len(target) x m)
        
        # Transform factors to target observations (in standardized scale)
        X_forecast_std = Z_forecast @ C_target.T  # (horizon x len(target))
        
        # Unstandardize using scaler if available, otherwise return as-is
        target_scaler = getattr(self, 'target_scaler', None)
        if target_scaler is not None and hasattr(target_scaler, 'inverse_transform'):
            # Reshape for scaler: scaler expects (n_samples, n_features)
            X_forecast = target_scaler.inverse_transform(X_forecast_std)
        else:
            # No scaler - assume already in original scale
            X_forecast = X_forecast_std
        
        # Ensure X_forecast is numpy array and validate it's finite
        X_forecast = ensure_numpy(X_forecast)
        from ..numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(X_forecast, name="forecast X_forecast")
        
        # Validate factor forecast if returning factors
        if return_factors:
            Z_forecast = ensure_numpy(Z_forecast)
            validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    @property
    def result(self) -> DDFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        """
        result = self._ensure_result()
        # Type assertion: get_result() always returns DDFMResult for DDFM model
        assert isinstance(result, DDFMResult), f"Expected DDFMResult but got {type(result)}"
        return result
    
    
    
    def update(self, data: Union[np.ndarray, Any]) -> None:
        """Update model state with new observations via neural network forward pass.
        
        This method runs the DDFM encoder-decoder forward pass on new data to update
        the latent factors, but keeps model parameters fixed.
        
        After calling update(), the model's internal state (result.Z and data_processed)
        is extended with the new observations. Subsequent calls to predict() will use
        the updated state.
        
        **Data Shape**: The input data must be 2D with shape (T_new x N) where:
        - T_new: Number of new time steps (can be any positive integer)
        - N: Number of series (must match training data)
        
        **Supported Types**:
        - numpy.ndarray: (T_new x N) array
        - pandas.DataFrame: DataFrame with N columns, T_new rows
        - polars.DataFrame: DataFrame with N columns, T_new rows
        
        **Important**: Data must be preprocessed by the user (same preprocessing as training).
        Only target scaler is handled internally if needed.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations with shape (T_new x N) where:
            - T_new: Number of new time steps (any positive integer)
            - N: Number of series (must match training data)
            Data must be preprocessed by user (same preprocessing as training).
            
        Notes
        -----
        - This updates factors via neural network forward pass, NOT parameter retraining
        - For parameter retraining, use fit_mcmc() with concatenated data
        - After update(), predict() will use the updated factor state
        - New data must have same number of series (N) as training data
        - User must preprocess data themselves (same preprocessing as training)
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        DataValidationError
            If data shape doesn't match training data
        """
        # Validate and convert data (no preprocessing - user must preprocess)
        from ..numeric.validator import validate_and_convert_update_data
        data_new = validate_and_convert_update_data(
            data,
            self.data_processed,
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        
        # Convert to tensor
        data_tensor = ensure_tensor(data_new, dtype=DEFAULT_TORCH_DTYPE)
        device = next(self.parameters()).device
        data_tensor = data_tensor.to(device)
        
        # Run encoder to get factors
        with torch.no_grad():
            factors_new = self.encoder(data_tensor)  # (T_new x num_factors)
        
        # Convert to numpy
        factors_new_np = ensure_numpy(factors_new, dtype=DEFAULT_DTYPE)
        
        # Get current result (compute if needed)
        result = self._ensure_result()
        
        # Update model state: append new factors and data
        result.Z = np.vstack([result.Z, factors_new_np])
        self.data_processed = np.vstack([self.data_processed, data_new])
        
        # Update smoothed data (x_sm) in result
        result.x_sm = result.Z @ result.C.T
    
    
    def reset(self) -> 'DDFM':
        """Reset model state."""
        super().reset()
        return self

