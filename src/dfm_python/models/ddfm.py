"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.

DDFM is a PyTorch Lightning module that inherits from BaseFactorModel.
"""

# Standard library imports
from dataclasses import dataclass
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
    make_config_source,
)
from ..config import DDFMResult
from ..encoder.simple_encoder import Encoder, extract_decoder_params
from ..decoder.linear import Decoder
from ..decoder.mlp import MLPDecoder
from ..logger import get_logger
from ..numeric.stability import (
    create_scaled_identity,
    clean_matrix,
    check_convergence_with_tolerance,
    ensure_positive_definite,
    compute_var_safe,
    compute_cov_safe,
)
from ..numeric.estimator import (
    estimate_idio_dynamics,
    estimate_variance_unified,
    extract_idio_params_for_ddfm,
    estimate_var_with_fallback,
)
from ..numeric.validator import (
    validate_no_nan_inf,
    validate_horizon,
    validate_and_convert_update_data,
    validate_factors,
    validate_ddfm_training_data,
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
from ..utils.loss import compute_masked_loss
from ..utils.checkpoint import (
    parse_checkpoint,
    infer_input_dim_from_data,
    infer_ddfm_input_dim,
    infer_ddfm_params_from_state_dict,
)
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
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_IDENTITY_SCALE,
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
    COMPUTATION_ERROR_TYPES,
    DEFAULT_DDFM_CLIP_RANGE_DEEP,
    DEFAULT_MIN_OBS_PRETRAIN,
    DEFAULT_MULT_EPOCH_PRETRAIN,
    DEFAULT_DDFM_CLIP_RANGE_SHALLOW,
    MATRIX_TYPE_DIAGONAL,
    MATRIX_TYPE_COVARIANCE,
    DEFAULT_DISP,
    DEFAULT_IDIO_STD,
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
    
    Note: Factors use AR(1) dynamics by default (configurable via factor_order).
    
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
        n_mc_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_idiosyncratic: bool = True,
        min_obs_idio: Optional[int] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: Optional[int] = None,
        seed: Optional[int] = None,
        decay_learning_rate: bool = True,
        min_obs_pretrain: Optional[int] = None,
        mult_epoch_pretrain: Optional[int] = None,
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
        n_mc_samples : int, optional
            Number of MC samples per MCMC iteration (default: DEFAULT_N_MC_SAMPLES = 1)
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
        disp : int, optional
            Display progress every 'disp' iterations (default: DEFAULT_DISP = 10)
        decay_learning_rate : bool, default True
            Whether to use exponential decay learning rate scheduler (matches original DDFM)
        min_obs_pretrain : int, optional
            Minimum number of observations for pre-training without interpolation
            (default: 50, matches original DDFM implementation)
        mult_epoch_pretrain : int, optional
            Multiplier for number of epochs during pre-training (default: 1)
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
            DEFAULT_HUBER_DELTA, DEFAULT_WEIGHT_DECAY, DEFAULT_GRAD_CLIP_VAL,
            DEFAULT_N_MC_SAMPLES, DEFAULT_FACTOR_ORDER
        )
        
        # Resolve parameters using consolidated helper
        from ..utils.misc import resolve_param
        self.learning_rate = resolve_param(learning_rate, default=DEFAULT_DDFM_LEARNING_RATE)
        self.n_mc_samples = resolve_param(n_mc_samples, default=DEFAULT_N_MC_SAMPLES)
        self.batch_size = resolve_param(batch_size, default=DEFAULT_DDFM_BATCH_SIZE)
        self.factor_order = DEFAULT_FACTOR_ORDER  # Factors use AR(1) dynamics
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = resolve_param(min_obs_idio, default=DEFAULT_MIN_OBS_IDIO)
        self.max_iter = resolve_param(max_iter, default=DEFAULT_MAX_MCMC_ITER)
        self.tolerance = resolve_param(tolerance, default=DEFAULT_TOLERANCE)
        self.disp = resolve_param(disp, default=DEFAULT_DISP)
        self.decay_learning_rate = decay_learning_rate
        # Pre-training defaults (DDFM-specific: 50 observations, 1x multiplier)
        # These match original DDFM implementation defaults
        self.min_obs_pretrain = resolve_param(min_obs_pretrain, default=DEFAULT_MIN_OBS_PRETRAIN)
        self.mult_epoch_pretrain = resolve_param(mult_epoch_pretrain, default=DEFAULT_MULT_EPOCH_PRETRAIN)
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
            from ..utils.misc import get_config_attr
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
        
        # Enable manual optimization for alternating updates (NN params vs AR/Sigma_epsilon)
        self.automatic_optimization = False
        
        # Register buffers for state parameters (will be properly initialized in on_train_start() when data dimensions are known)
        # Phi: AR coefficients for idiosyncratic components, shape (N, N) - variable-specific AR(1)
        # Sigma_eps: Idiosyncratic covariance, shape (N,) for diagonal or (N, N) for full
        # eps_hat: Current residual estimates, shape (T, N)
        # Initialize with placeholder tensors, will be resized in on_train_start()
        self.register_buffer('Phi', torch.zeros(1, 1, dtype=DEFAULT_TORCH_DTYPE))
        self.register_buffer('Sigma_eps', torch.ones(1, dtype=DEFAULT_TORCH_DTYPE))
        self.register_buffer('eps_hat', torch.zeros(1, 1, dtype=DEFAULT_TORCH_DTYPE))
        
        # Training state for convergence checking
        self.prediction_prev_iter: Optional[torch.Tensor] = None
        self.factors_samples: Optional[torch.Tensor] = None  # Store factors from all MC samples as tensor (n_mc_samples, T, num_factors)
        
        # MC dataset (created in on_train_start)
        self._mc_dataset: Optional[Any] = None
        
        # Data structures for MC sampling (initialized in on_train_start)
        self._data_mod: Optional[torch.Tensor] = None
        self._data_mod_only_miss: Optional[torch.Tensor] = None
        self._missing_mask: Optional[np.ndarray] = None
    
    def _handle_initialization_error(
        self,
        component_name: str,
        error: Exception,
        component_specific_details: str
    ) -> None:
        """Handle initialization errors consistently.
        
        Parameters
        ----------
        component_name : str
            Name of component being initialized (e.g., "encoder", "decoder")
        error : Exception
            The exception that occurred
        component_specific_details : str
            Component-specific error details and suggestions
        """
        raise ModelNotInitializedError(
            f"DDFM {component_name} initialization failed: failed to initialize {component_name}: {type(error).__name__}: {str(error)}",
            details=component_specific_details
        ) from error
    
    def _extract_decoder_weight(self, decoder: Union[Decoder, Any]) -> np.ndarray:
        """Extract decoder weight for validation.
        
        Parameters
        ----------
        decoder : Decoder or MLPDecoder
            Decoder instance
            
        Returns
        -------
        np.ndarray
            Decoder weight matrix
        """
        if self.decoder_type == "linear":
            return ensure_numpy(decoder.decoder.weight.data)
        elif self.decoder_type == "mlp":
            return ensure_numpy(decoder.layers[0].weight.data)
        else:
            raise ConfigurationError(
                f"DDFM decoder weight extraction failed: unsupported decoder type '{self.decoder_type}'",
                details="Supported decoder types are 'linear' and 'mlp'"
            )
    
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
        except COMPUTATION_ERROR_TYPES as e:
            # Uses COMPUTATION_ERROR_TYPES for consistent error handling (ValueError, RuntimeError, TypeError are included)
            self._handle_initialization_error(
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
            elif self.decoder_type == "mlp":
                self.decoder = MLPDecoder(
                    input_dim=self.num_factors,
                    output_dim=input_dim,
                    hidden_dims=self.decoder_layers,
                    activation=self.activation,
                    use_batch_norm=False,  # Usually not needed for decoder
                    use_bias=True,
                )
            else:
                check_condition(
                    False,  # Always fails - decoder_type is invalid
                    ConfigurationError,
                    f"DDFM decoder initialization failed: decoder must be 'linear' or 'mlp', got '{self.decoder_type}'",
                    details="Valid decoder types are 'linear' (LinearDecoder) or 'mlp' (MLPDecoder)"
                )
            
            # Extract decoder weight for validation (using helper method)
            decoder_weight = self._extract_decoder_weight(self.decoder)
            
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
        except COMPUTATION_ERROR_TYPES as e:
            # Uses COMPUTATION_ERROR_TYPES for consistent error handling (ValueError, RuntimeError, TypeError are included)
            self._handle_initialization_error(
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
    
    
    def _move_networks_to_device(self, device: torch.device) -> None:
        """Move encoder and decoder to specified device.
        
        Parameters
        ----------
        device : torch.device
            Target device
        """
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def _set_networks_mode(self, train: bool = True) -> None:
        """Set encoder and decoder to train or eval mode.
        
        Parameters
        ----------
        train : bool, default True
            If True, set to train mode; otherwise set to eval mode
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
    
    def _get_buffer_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Get device and dtype from Phi buffer.
        
        Returns
        -------
        device : torch.device
            Device of Phi buffer
        dtype : torch.dtype
            Dtype of Phi buffer
        """
        return self.Phi.device, self.Phi.dtype
    
    def _initialize_buffers(self, T: int, N: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize or update Phi, Sigma_eps, and eps_hat buffers.
        
        Parameters
        ----------
        T : int
            Number of time steps
        N : int
            Number of variables
        device : torch.device
            Device for buffers
        dtype : torch.dtype
            Data type for buffers
        """
        # Initialize or update buffers based on whether they exist
        buffers = [
            ('Phi', (N, N), torch.zeros(N, N, device=device, dtype=dtype)),
            ('Sigma_eps', (N,), torch.ones(N, device=device, dtype=dtype) * DEFAULT_EPSILON),
            ('eps_hat', (T, N), torch.zeros(T, N, device=device, dtype=dtype))
        ]
        
        for name, expected_shape, init_value in buffers:
            if not hasattr(self, name):
                self.register_buffer(name, init_value)
            elif getattr(self, name).shape != expected_shape:
                # Update existing buffer if shape changed
                setattr(self, name, init_value)
    
    
    
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
        if x.ndim == 3:  # Batch format: (batch_size, T, N)
            batch_size, T, N = x.shape
            x_flat = x.view(batch_size * T, N)
            factors = self.encoder(x_flat)
            reconstructed = self.decoder(factors)
            return reconstructed.view(batch_size, T, N)
        else:
            # Assume sequence format: (T, N)
            factors = self.encoder(x)
            reconstructed = self.decoder(factors)
            return reconstructed
    
    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Training step for vectorized MC sampling with manual optimization.
        
        This method processes all MC samples in one forward pass for efficient training.
        Batch from DDFMMCDataset contains:
        - x_corrupted: (n_mc_samples, T, N) - corrupted inputs with MC noise
        - x_target: (n_mc_samples, T, N) - target observations
        - mask: (n_mc_samples, T, N) - missing data mask
        
        Parameters
        ----------
        batch : tuple
            (x_corrupted, x_target, mask) from DDFMMCDataset
        batch_idx : int
            Batch index (should be 0 for MC dataset)
            
        Returns
        -------
        loss : torch.Tensor
            Reconstruction loss (MSE with missing data masking) averaged over all MC samples
        """
        # Handle batch from MC dataset: (x_corrupted, x_target, mask)
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                f"DDFM training_step expects batch from DDFMMCDataset with 3 elements "
                f"(x_corrupted, x_target, mask), got {type(batch)} with {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'} elements"
            )
        x_corrupted, x_target, mask = batch
        
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        x_corrupted = x_corrupted.to(device)
        x_target = x_target.to(device)
        mask = mask.to(device)
        
        # DataLoader with batch_size=1 adds an extra batch dimension
        # Remove it if present: (1, n_mc_samples, T, N) -> (n_mc_samples, T, N)
        BATCH_WRAPPER_NDIM = 4  # Extra batch dimension from DataLoader
        SINGLE_BATCH = 1  # Single batch size
        BATCH_DIM = 0  # Batch dimension index
        
        if x_corrupted.ndim == BATCH_WRAPPER_NDIM and x_corrupted.shape[BATCH_DIM] == SINGLE_BATCH:
            x_corrupted = x_corrupted.squeeze(BATCH_DIM)
            x_target = x_target.squeeze(BATCH_DIM)
            mask = mask.squeeze(BATCH_DIM)
        
        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        # x_corrupted shape: (n_mc_samples, T, N)
        # Reshape for batch processing: (n_mc_samples * T, N)
        n_mc_samples, T, N = x_corrupted.shape
        x_corrupted_flat = x_corrupted.view(n_mc_samples * T, N)
        
        # Clip input data to prevent extreme values
        clip_range = DEFAULT_DDFM_CLIP_RANGE_DEEP if len(self.encoder_layers) > 2 else DEFAULT_DDFM_CLIP_RANGE_SHALLOW
        x_corrupted_flat = torch.clamp(x_corrupted_flat, min=-clip_range, max=clip_range)
        
        # Forward pass: process all MC samples in one batch
        self._set_networks_mode(train=True)
        factors_flat = self.encoder(x_corrupted_flat)  # (n_mc_samples * T, num_factors)
        reconstructed_flat = self.decoder(factors_flat)  # (n_mc_samples * T, N)
        
        # Reshape back: (n_mc_samples, T, N)
        reconstructed = reconstructed_flat.view(n_mc_samples, T, N)
        
        # Store factors as tensor for efficient averaging in on_train_epoch_end
        self.factors_samples = factors_flat.view(n_mc_samples, T, self.num_factors)  # (n_mc_samples, T, num_factors)
        
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
        loss = compute_masked_loss(
            reconstructed=reconstructed,
            target=x_target,
            mask=mask,
            loss_function=self.loss_function,
            huber_delta=self.huber_delta
        )
        
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
        
        # Manual optimization (automatic_optimization = False)
        opt = self.optimizers()
        self.manual_backward(loss)
        
        # Gradient clipping
        if self.grad_clip_val > DEFAULT_ZERO_VALUE:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=self.grad_clip_val
            )
        
        opt.step()
        opt.zero_grad()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def _update_idiosyncratic_params(
        self,
        eps: np.ndarray,
        missing_mask: np.ndarray
    ) -> None:
        """Update Phi and Sigma_eps buffers from residuals.
        
        Parameters
        ----------
        eps : np.ndarray
            Residuals (T x N)
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
        """
        if not self.use_idiosyncratic:
            # If not using idiosyncratic, set to zero/identity
            N = eps.shape[1]
            device, dtype = self._get_buffer_device_dtype()
            self.Phi = torch.zeros(N, N, device=device, dtype=dtype)
            self.Sigma_eps = torch.ones(N, device=device, dtype=dtype) * DEFAULT_EPSILON
            return
        
        try:
            A_eps, Q_eps = estimate_idio_dynamics(eps, missing_mask, self.min_obs_idio)
            
            # Extract phi and std_eps using numeric utility
            N = eps.shape[1]
            phi, std_eps = extract_idio_params_for_ddfm(A_eps, Q_eps, N)
            
            # Update buffers
            device, dtype = self._get_buffer_device_dtype()
            self.Phi = torch.tensor(phi, device=device, dtype=dtype)
            self.Sigma_eps = torch.tensor(std_eps, device=device, dtype=dtype)
            
        except Exception as e:
            _logger.warning(f"Failed to update idiosyncratic parameters: {e}. Using previous values.")
            # Keep previous values on error
    
    def _check_convergence(
        self,
        prediction_iter: np.ndarray,
        prediction_prev_iter: Optional[np.ndarray],
        data_mod_only_miss: np.ndarray
    ) -> Tuple[float, bool]:
        """Check convergence based on MSE delta using numeric utilities.
        
        Parameters
        ----------
        prediction_iter : np.ndarray
            Current prediction (T x N)
        prediction_prev_iter : np.ndarray, optional
            Previous prediction (T x N)
        data_mod_only_miss : np.ndarray
            Original data with missing values (T x N)
            
        Returns
        -------
        delta : float
            MSE delta between current and previous prediction
        converged : bool
            Whether convergence criterion is met
        """
        if prediction_prev_iter is None:
            return float('inf'), False
        
        # Use numeric utility for convergence checking with tolerance
        delta, converged = check_convergence_with_tolerance(
            y_prev=prediction_prev_iter,
            y_now=prediction_iter,
            y_actual=data_mod_only_miss,
            tolerance=self.tolerance
        )
        
        return delta, converged
    
    
    def _validate_factors(self, factors: np.ndarray, operation: str = "operation") -> np.ndarray:
        """Validate and normalize factors shape and content quality.
        
        Parameters
        ----------
        factors : np.ndarray
            Factors to validate
        operation : str, default "operation"
            Operation name for error messages
            
        Returns
        -------
        np.ndarray
            Validated and normalized factors (2D array)
        """
        return validate_factors(factors, self.num_factors, operation)
    
    def _validate_training_data(
        self,
        X_torch: torch.Tensor,
        operation: str = "training setup"
    ) -> None:
        """Validate data dimensions and model configuration before training starts."""
        validate_ddfm_training_data(
            X_torch=X_torch,
            num_factors=self.num_factors,
            encoder_layers=self.encoder_layers,
            encoder=self.encoder,
            operation=operation
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
            VAR transition matrix of shape (m, m) for AR(factor_order)
        Q_f : np.ndarray
            Innovation covariance matrix of shape (m, m), positive definite
        """
        return estimate_var_with_fallback(
            factors=factors,
            order=self.factor_order,
            num_factors=self.num_factors
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
        # Encoder/decoder should be initialized by on_train_start() before configure_optimizers is called
        # However, Lightning may call this before on_train_start() in some cases, so we handle gracefully
        if self.encoder is None or self.decoder is None:
            _logger.warning("Encoder/decoder not initialized, creating placeholder optimizer")
            # Create a dummy parameter to satisfy Lightning's optimizer requirement
            from ..config.constants import DEFAULT_ZERO_VALUE
            dummy_param = nn.Parameter(torch.tensor(DEFAULT_ZERO_VALUE))
            optimizer = torch.optim.Adam([dummy_param], lr=self.learning_rate)
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
        self._move_networks_to_device(device)
        
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
        
        # Pre-train for n_mc_samples * mult_epoch_pretrain epochs
        # Matches original DDFM: epochs * mult_epoch_pre where epochs was the number of MC samples
        num_epochs = self.n_mc_samples * self.mult_epoch_pretrain
        _logger.info(f"DDFM pre_train: Starting pre-training for {num_epochs} epochs")
        
        self._set_networks_mode(train=True)
        
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
                    ).bool()
                    # Use loss utility for consistency
                    loss = compute_masked_loss(
                        reconstructed=reconstructed,
                        target=batch_target,
                        mask=mask,
                        loss_function='mse'
                    )
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
            details="Please train the model using trainer.fit(model, datamodule) first"
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
        A_f, Q_f = self._estimate_var(factors)
        
        # For DDFM, we use simplified state-space (factor-only)
        A = A_f
        Q = Q_f
        Z_0 = factors[0, :]
        
        # Compute initial covariance using safe utility
        # factors.T converts (T, m) to (m, T) for rowvar=True
        # Handle edge case where we have insufficient data for covariance
        try:
            V_0 = compute_cov_safe(
                factors.T,
                rowvar=True,
                pairwise_complete=False,
                min_eigenval=MIN_EIGENVALUE,
                fallback_to_identity=True
            )
        except DataError as e:
            # If covariance computation fails due to insufficient data, use identity
            _logger.warning(
                f"DDFM get_result: Failed to compute initial covariance: {e}. "
                f"Using identity matrix as fallback."
            )
            m = factors.shape[1]
            V_0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE)
        
        # Ensure V_0 is always 2D (compute_cov_safe handles this, but double-check)
        if V_0.ndim == 0:
            V_0 = np.atleast_2d(V_0)
        elif V_0.ndim == 1:
            # If 1D, reshape to (m x m)
            V_0 = np.atleast_2d(V_0).T if V_0.shape[0] == 1 else np.atleast_2d(V_0)
        
        # Estimate R from residuals using unified variance estimation utility
        R = estimate_variance_unified(
            residuals=residuals,
            min_variance=MIN_DIAGONAL_VARIANCE,
            default_variance=MIN_DIAGONAL_VARIANCE,
            dtype=DEFAULT_DTYPE
        )
        
        # Compute smoothed data (already standardized)
        x_sm = prediction_iter
        
        r = np.array([self.num_factors])
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
            p=self.factor_order,
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
            except COMPUTATION_ERROR_TYPES as e:
                # Uses COMPUTATION_ERROR_TYPES for consistent error handling (ValueError, RuntimeError, AttributeError are included)
                # Log warning but don't fail - result can be computed later if needed
                _logger.warning(
                    f"Could not automatically compute result after training: {e}. "
                    f"Result will be computed on first access to result property or predict()."
                )
    
    def on_train_start(self) -> None:
        """Called when training starts. Initialize buffers and data structures for Lightning training."""
        # Get processed data and target scaler from DataModule (standard Lightning flow)
        data_module = self._get_datamodule()
        X_torch = data_module.get_processed_data()
        target_scaler = getattr(data_module, 'target_scaler', None)
        if target_scaler is not None:
            self.target_scaler = target_scaler
        
        # Early validation: Check data dimensions and model configuration before training
        # This catches configuration issues early with clear error messages
        self._validate_training_data(X_torch, operation="training setup")
        
        # Initialize encoder/decoder (data dimensions are now available)
        input_dim = X_torch.shape[1]
        self.initialize_networks(input_dim)
        self._move_networks_to_device(X_torch.device)
        _logger.debug(f"Initialized encoder/decoder with input_dim={input_dim}")
        
        # Store processed data
        self.data_processed = X_torch
        
        # Handle missing data - use DataModule's preprocessing method
        if hasattr(data_module, '_preprocess_training_data'):
            x_clean_torch, missing_mask = data_module._preprocess_training_data(X_torch)
        else:
            raise ModelNotInitializedError(
                "DDFM on_train_start failed: DataModule does not have _preprocess_training_data method. "
                "Ensure datamodule.setup() has been called before training."
            )
        
        if np.any(missing_mask):
            _logger.info(
                f"DDFM on_train_start: NaN values detected in training data. "
                f"Using imputation for initialization. "
                f"DDFM will handle remaining missing data through state-space model."
            )
        
        # Adjust missing_mask shape to match x_clean_torch
        if hasattr(data_module, '_adjust_mask_shape'):
            missing_mask = data_module._adjust_mask_shape(missing_mask, x_clean_torch.shape)
        else:
            raise ModelNotInitializedError(
                "DDFM on_train_start failed: DataModule does not have _adjust_mask_shape method. "
                "Ensure datamodule is properly initialized."
            )
        
        # Store data structures for MC sampling
        T, N = x_clean_torch.shape
        self._data_mod_only_miss = x_clean_torch.clone()
        self._data_mod = x_clean_torch.clone()
        self._missing_mask = missing_mask
        
        # Pre-train autoencoder on non-missing data (matching original DDFM)
        # This provides stable initialization before MCMC training
        try:
            self.pre_train(
                X=x_clean_torch,
                x_clean=x_clean_torch,
                missing_mask=missing_mask,
                device=x_clean_torch.device,
            )
        except (RuntimeError, ValueError, AttributeError, OSError) as e:
            _logger.warning(
                f"DDFM pre_train failed: {e}. Continuing with MCMC training without pre-training."
            )
        
        # Initialize buffers with proper dimensions
        device = x_clean_torch.device
        dtype = x_clean_torch.dtype
        self._initialize_buffers(T, N, device, dtype)
        
        # Initialize initial prediction and residuals
        self._set_networks_mode(train=False)
        with torch.no_grad():
            factors_init = self.encoder(x_clean_torch)  # (T, num_factors)
            prediction_init = self.decoder(factors_init)  # (T, N)
            
            # Update missing values with initial prediction
            bool_miss = torch.tensor(missing_mask, device=device, dtype=torch.bool)
            if bool_miss.any():
                self._data_mod_only_miss[bool_miss] = prediction_init[bool_miss]
            
            # Initialize residuals
            eps_init = self._data_mod_only_miss - prediction_init
            self.eps_hat = eps_init
        
        # Initialize prediction_prev_iter for convergence checking
        self.prediction_prev_iter = None
        
        # Initialize factors_samples tensor
        self.factors_samples = None
        
        # Initialize training state
        if self.training_state is None:
            self.training_state = DDFMTrainingState(
                factors=ensure_numpy(factors_init),
                prediction=ensure_numpy(prediction_init),
                converged=False,
                num_iter=0
            )
        
        # Initialize MCMC iteration counter
        self.mcmc_iteration = 0
        
        # Create MC dataset via DataModule (dataset/dataloader logic belongs in DataModule)
        if hasattr(data_module, 'create_mc_dataset'):
            self._mc_dataset = data_module.create_mc_dataset(
                model=self,
                data_mod=self._data_mod,
                data_mod_only_miss=self._data_mod_only_miss,
                missing_mask=self._missing_mask
            )
            # Store reference in datamodule for train_dataloader()
            data_module.mc_dataset = self._mc_dataset
        else:
            # Fallback: create directly if datamodule doesn't have create_mc_dataset method
            from ..dataset.ddfm_dataset import DDFMMCDataset
            self._mc_dataset = DDFMMCDataset.create_from_model(
                model=self,
                data_mod=self._data_mod,
                data_mod_only_miss=self._data_mod_only_miss,
                missing_mask=self._missing_mask
            )
        
        _logger.info(f"DDFM on_train_start: Initialized buffers and data structures. T={T}, N={N}, num_factors={self.num_factors}, n_mc_samples={self.n_mc_samples}")
        
        pl.LightningModule.on_train_start(self)
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch. Update AR/Sigma_epsilon and check convergence.
        
        This method:
        1. Averages factors over all MC samples
        2. Reconstructs epsilon from decoder output
        3. Updates AR/Sigma_epsilon parameters
        4. Checks convergence
        5. Updates data structures for next iteration
        """
        # Check if we have factors from MC samples
        if self.factors_samples is None:
            _logger.warning("DDFM on_train_epoch_end: No factors samples available. Skipping update.")
            return
        
        self.mcmc_iteration += 1
        
        # Average factors over all MC samples (already stored as tensor)
        factors = torch.mean(self.factors_samples, dim=0)  # (T, num_factors)
        
        # Reconstruct prediction from averaged factors
        self._set_networks_mode(train=False)
        with torch.no_grad():
            prediction_iter = self.decoder(factors)  # (T, N)
        
        # Ensure _data_mod_only_miss is on same device as prediction_iter
        device = prediction_iter.device
        if self._data_mod_only_miss.device != device:
            self._data_mod_only_miss = self._data_mod_only_miss.to(device)
        
        # Update missing values with current prediction
        bool_miss = torch.tensor(self._missing_mask, device=device, dtype=torch.bool)
        if bool_miss.any():
            self._data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Update residuals (keep as torch for efficiency, convert only when needed)
        eps_torch = self._data_mod_only_miss - prediction_iter
        self.eps_hat = eps_torch
        
        # Convert to numpy for AR/Sigma update (only when needed)
        eps_np = ensure_numpy(eps_torch)
        factors_np = ensure_numpy(factors)
        prediction_iter_np = ensure_numpy(prediction_iter)
        
        # Update AR/Sigma_epsilon parameters
        self._update_idiosyncratic_params(eps_np, self._missing_mask)
        
        # Filter data: subtract conditional AR-idio mean (idiosyncratic uses AR(1) per variable)
        MIN_TIME_STEPS_FOR_AR = 2  # Need at least 2 time steps for AR(1) filtering
        if self.use_idiosyncratic and eps_torch.shape[0] >= MIN_TIME_STEPS_FOR_AR:
            Phi = self.Phi
            phi_device = Phi.device
            # Ensure tensors are on same device as Phi (only transfer if needed)
            if eps_torch.device != phi_device:
                eps_torch = eps_torch.to(phi_device)
            if self._data_mod_only_miss.device != phi_device:
                self._data_mod_only_miss = self._data_mod_only_miss.to(phi_device)
            
            eps_prev = eps_torch[:-1, :]
            # Clone only the portion we'll modify to avoid unnecessary memory copy
            data_mod_torch = self._data_mod_only_miss.clone()
            data_mod_torch[1:] = data_mod_torch[1:] - eps_prev @ Phi
            self._data_mod = data_mod_torch
        else:
            # No filtering needed - clone to avoid shared reference issues
            self._data_mod = self._data_mod_only_miss.clone()
        
        # Update MC dataset with new filtered data
        if self._mc_dataset is not None:
            self._mc_dataset.data_mod = self._data_mod
            # Also update datamodule's reference if available
            try:
                data_module = self._get_datamodule()
                if hasattr(data_module, 'mc_dataset'):
                    data_module.mc_dataset = self._mc_dataset
            except ModelNotInitializedError:
                # Datamodule not available, skip update
                pass
        
        # Check convergence
        delta, converged = self._check_convergence(
            prediction_iter_np,
            ensure_numpy(self.prediction_prev_iter) if self.prediction_prev_iter is not None else None,
            ensure_numpy(self._data_mod_only_miss)
        )
        
        # Store prediction for next iteration (clone already detaches from computation graph)
        self.prediction_prev_iter = prediction_iter.clone()
        
        # Update training state
        self.training_state = DDFMTrainingState(
            factors=factors_np,
            prediction=prediction_iter_np,
            converged=converged,
            num_iter=self.mcmc_iteration,
            training_loss=float(delta) if np.isfinite(delta) else float('inf')
        )
        
        # Log progress
        if self.mcmc_iteration % self.disp == 0:
            _logger.info(
                f"Iteration {self.mcmc_iteration}/{self.max_iter}: loss={self.training_state.training_loss:.6f}, "
                f"delta={delta:.6f}, converged={converged}"
            )
        
        # Stop training if converged
        if converged:
            _logger.info(
                f"Convergence achieved in {self.mcmc_iteration} iterations: "
                f"loss={self.training_state.training_loss:.6f}, delta={delta:.6f} < {self.tolerance}"
            )
            # Signal to trainer to stop
            if hasattr(self.trainer, 'should_stop'):
                self.trainer.should_stop = True
        
        # Clear factors_samples for next epoch
        self.factors_samples = None
        
        pl.LightningModule.on_train_epoch_end(self)
    
    
    
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
        state_dict, hparams = parse_checkpoint(checkpoint)
        
        # Infer input_dim: prioritize checkpoint, then data, then explicit parameter
        checkpoint_input_dim = infer_ddfm_input_dim(state_dict)
        
        if input_dim is None:
            if checkpoint_input_dim is not None:
                input_dim = checkpoint_input_dim
                # Warn if data dimension doesn't match
                if data is not None:
                    data_input_dim = infer_input_dim_from_data(data, date_id_col)
                    if data_input_dim != input_dim:
                        _logger.warning(
                            f"DDFM.load: Data input_dim ({data_input_dim}) doesn't match checkpoint input_dim ({input_dim}). "
                            f"Using checkpoint input_dim. Model may not work correctly with current data."
                        )
            elif data is not None:
                input_dim = infer_input_dim_from_data(data, date_id_col)
            else:
                raise ConfigurationError(
                    "Cannot determine input_dim. Please provide either 'data' or 'input_dim' parameter",
                    details="input_dim is required to initialize encoder/decoder"
                )
        
        # Infer model parameters from state_dict
        model_params = infer_ddfm_params_from_state_dict(state_dict, hparams, kwargs)
        
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
        
        # Forecast factors using VAR dynamics
        MIN_TIME_STEPS_FOR_AR2 = 2  # Minimum time steps needed for AR(2) prediction
        AR_ORDER_2 = 2
        Z_prev = result.Z[-2, :] if result.Z.shape[0] >= MIN_TIME_STEPS_FOR_AR2 and p == AR_ORDER_2 else None
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
        - For parameter retraining, retrain the model using trainer.fit(model, datamodule) with concatenated data
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

