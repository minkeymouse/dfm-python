"""Identifiable Variational Dynamic Factor Model (iVDFM).

Implements the iVDFM framework that combines identifiable latent-variable modeling
with explicit stochastic dynamics. Identifiability is achieved by applying iVAE
conditions to the innovation process driving dynamics.
"""

import time
from pathlib import Path
from typing import Optional, Any, Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseFactorModel
from ...logger import get_logger
from ...logger.ivdfm_logger import iVDFMTrainLogger
from ...config.constants import DEFAULT_LOSS_LOG_PRECISION
from ...config.types import to_tensor, to_numpy
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_SEED,
    DEFAULT_DTYPE,
    DEFAULT_TOLERANCE,
    DEFAULT_IVDFM_LATENT_DIM,
    DEFAULT_IVDFM_AUX_DIM,
    DEFAULT_REGULARIZATION,
)
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError, DataValidationError
from ...utils.validation import check_condition
from ...utils.loss import compute_elbo_loss
from ...layer.pca import extract_pca_factors
from .encoder import iVDFMInnovationEncoder
from .decoder import iVDFMDecoder
from .prior import iVDFMPriorNetwork
from ...ssm.companion import iVDFMCompanionSSM
from ...dataset.ivdfm_dataset import iVDFMDataset
from ...config.schema.model import iVDFMConfig
from ...config.schema.params import iVDFMModelState
from ...config.schema.results import iVDFMResult
from ...numeric.builder import build_ivdfm_optimizer

_logger = get_logger(__name__)


class iVDFM(BaseFactorModel, nn.Module):
    """Identifiable Variational Dynamic Factor Model.
    
    Combines identifiable latent-variable modeling (iVAE) with explicit
    stochastic dynamics. Identifiability is achieved by applying conditional
    exponential-family priors to innovations rather than states.
    """
    
    def __init__(
        self,
        data_dim: Optional[int] = None,
        num_factors: Optional[int] = None,  # Aligned with config: num_factors (not latent_dim)
        context_dim: Optional[int] = None,
        window: Optional[int] = None,
        config: Optional[iVDFMConfig] = None,
        # Network architecture (used if config is None)
        encoder_hidden_dim: Union[int, list] = 200,
        encoder_n_hidden_layers: int = 2,
        decoder_hidden_dim: Union[int, list] = 200,
        decoder_n_hidden_layers: int = 2,
        prior_hidden_dim: Union[int, list] = 100,
        prior_n_hidden_layers: int = 1,
        activation: str = 'lrelu',
        slope: float = 0.1,
        # Dynamics parameters
        factor_order: int = 1,  # AR order for factors (p in AR(p))
        # Prior/innovation parameters
        innovation_distribution: str = 'laplace',  # 'laplace', 'student_t', etc.
        decoder_var: float = 0.01,
        # Training parameters
        # NOTE: these default to None so they don't override config values
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        # Device
        device: Optional[torch.device] = None,
        seed: int = DEFAULT_SEED,
        **kwargs  # Allow additional parameters to override config
    ):
        """Initialize iVDFM model.
        
        Parameters
        ----------
        data_dim : int
            Dimension of observed data (N in paper)
        num_factors : int
            Number of latent factors (r in paper)
        context_dim : int
            Dimension of context variable u_t
        window : int
            Length of sliding windows (T in paper)
        config : Optional[Any]
            Configuration object
        encoder_hidden_dim : Union[int, list]
            Hidden dimensions for innovation encoder
        encoder_n_hidden_layers : int
            Number of hidden layers in innovation encoder
        decoder_hidden_dim : Union[int, list]
            Hidden dimensions for decoder
        decoder_n_hidden_layers : int
            Number of hidden layers in decoder
        prior_hidden_dim : Union[int, list]
            Hidden dimensions for prior network
        prior_n_hidden_layers : int
            Number of hidden layers in prior network
        activation : str
            Activation function ('lrelu', 'relu', 'tanh')
        slope : float
            Slope for leaky ReLU
        factor_order : int
            AR order for factor dynamics (p in AR(p))
        innovation_distribution : str
            Distribution for innovations ('laplace', 'student_t', etc.)
        decoder_var : float
            Decoder variance (observation noise)
        learning_rate : float
            Learning rate for optimizer
        optimizer : str
            Optimizer type ('Adam', 'AdamW', 'SGD')
        batch_size : int
            Batch size for training
        max_epochs : int
            Maximum number of training epochs
        tolerance : float
            Convergence tolerance
        device : Optional[torch.device]
            Device for computation (None for auto-detect)
        seed : int
            Random seed
        """
        BaseFactorModel.__init__(self)
        nn.Module.__init__(self)
        
        # Build config: start with provided config or defaults, then override with parameters and kwargs
        if config is not None:
            if isinstance(config, iVDFMConfig):
                from dataclasses import asdict
                config_dict = asdict(config)
            elif isinstance(config, dict):
                config_dict = config.copy()
            else:
                raise ConfigurationError(
                    f"config must be iVDFMConfig instance or dict, got {type(config)}"
                )
        else:
            config_dict = {}
        
        # Directly update config_dict with explicit parameters (aligned with config keys)
        # Only add if parameter is not None (to allow using config defaults)
        if num_factors is not None:
            config_dict['num_factors'] = num_factors
        if encoder_hidden_dim is not None:
            config_dict['encoder_hidden_dim'] = encoder_hidden_dim
        if encoder_n_hidden_layers is not None:
            config_dict['encoder_n_hidden_layers'] = encoder_n_hidden_layers
        if decoder_hidden_dim is not None:
            config_dict['decoder_hidden_dim'] = decoder_hidden_dim
        if decoder_n_hidden_layers is not None:
            config_dict['decoder_n_hidden_layers'] = decoder_n_hidden_layers
        if prior_hidden_dim is not None:
            config_dict['prior_hidden_dim'] = prior_hidden_dim
        if prior_n_hidden_layers is not None:
            config_dict['prior_n_hidden_layers'] = prior_n_hidden_layers
        if activation is not None:
            config_dict['activation'] = activation
        if slope is not None:
            config_dict['slope'] = slope
        if factor_order is not None:
            config_dict['factor_order'] = factor_order
        if innovation_distribution is not None:
            config_dict['innovation_distribution'] = innovation_distribution
        if decoder_var is not None:
            config_dict['decoder_var'] = decoder_var
        if learning_rate is not None:
            config_dict['learning_rate'] = learning_rate
        if optimizer is not None:
            config_dict['optimizer'] = optimizer
        if batch_size is not None:
            config_dict['batch_size'] = batch_size
        if max_epochs is not None:
            config_dict['max_epochs'] = max_epochs
        if tolerance is not None:
            config_dict['tolerance'] = tolerance
        if seed is not None:
            config_dict['seed'] = seed
        
        # Special handling for dimensions (can be None)
        # Only add to config_dict if explicitly provided (not None)
        if data_dim is not None:
            config_dict['data_dim'] = data_dim
        if context_dim is not None:
            config_dict['context_dim'] = context_dim
        # Note: if context_dim is None, don't add to config_dict - will use None directly
        if window is not None:
            config_dict['window'] = window
        
        # Override with kwargs (highest precedence)
        # Backward compatibility: map latent_dim -> num_factors if provided in kwargs
        if 'latent_dim' in kwargs and 'num_factors' not in kwargs:
            kwargs['num_factors'] = kwargs.pop('latent_dim')
            _logger.warning(
                "Parameter 'latent_dim' is deprecated. Use 'num_factors' instead. "
                "This mapping will be removed in a future version."
            )
        config_dict.update(kwargs)
        
        # Preserve f0_init_method and ar_init_method even if None (they have defaults in schema)
        # Remove None values to use defaults, but keep initialization methods
        f0_init_method_val = config_dict.get('f0_init_method')
        ar_init_method_val = config_dict.get('ar_init_method')
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        # Restore initialization methods if they were explicitly provided (even if None)
        if 'f0_init_method' in kwargs:
            config_dict['f0_init_method'] = f0_init_method_val
        if 'ar_init_method' in kwargs:
            config_dict['ar_init_method'] = ar_init_method_val
        
        # Create config object
        try:
            self._config = iVDFMConfig.from_dict(config_dict) if config_dict else iVDFMConfig()
        except Exception:
            # Fallback: create with defaults and update
            self._config = iVDFMConfig()
            for key, value in config_dict.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
        
        # Extract all parameters from config (simplified, unified approach)
        self.data_dim = data_dim  # Can be None, inferred during fit
        # Keep latent_dim as internal attribute name (more intuitive), but use num_factors from config
        self.latent_dim = self._config.num_factors or DEFAULT_IVDFM_LATENT_DIM
        # context_dim: preserve None if not provided
        # Note: context_dim is NOT in iVDFMConfig schema (replaced by time_context + context columns).
        # context_dim is inferred from dataset during fit() based on time_context + custom context columns.
        # If user explicitly passes context_dim parameter, use it; otherwise None (will be inferred from dataset).
        self.context_dim = context_dim
        # time_context: dimension of time-based features (separate from custom context columns)
        self.time_context = self._get_config_attr('time_context', 1)
        # window: if None, use full T (full series as one sequence)
        # Priority: explicit parameter > config value > None (which dataset will convert to T)
        if window is not None:
            self.window = window
        elif self._config is not None and hasattr(self._config, 'window'):
            self.window = self._config.window
        else:
            self.window = None  # Will be converted to T by dataset
        # Extract config attributes (network architecture)
        self.encoder_hidden_dim = self._config.encoder_hidden_dim
        self.encoder_n_hidden_layers = self._config.encoder_n_hidden_layers
        self.decoder_hidden_dim = self._config.decoder_hidden_dim
        self.decoder_n_hidden_layers = self._config.decoder_n_hidden_layers
        self.prior_hidden_dim = self._config.prior_hidden_dim
        self.prior_n_hidden_layers = self._config.prior_n_hidden_layers
        self.activation = self._config.activation
        self.slope = self._config.slope
        
        # Extract config attributes (dynamics and distribution)
        self.factor_order = self._config.factor_order
        self.innovation_distribution = self._config.innovation_distribution
        self.decoder_var = self._config.decoder_var
        self.beta_kl = self._get_config_attr('beta_kl', 1.0)
        self.use_layer_norm = self._get_config_attr('use_layer_norm', False)
        
        # Extract config attributes (training)
        self.learning_rate = self._config.learning_rate
        self.optimizer_type = self._config.optimizer
        self.optimizer_weight_decay = self._config.optimizer_weight_decay
        self.optimizer_momentum = self._config.optimizer_momentum
        self.batch_size = self._config.batch_size
        self.max_epochs = self._config.max_epochs
        self.tolerance = self._config.tolerance
        self.patience = self._get_config_attr('patience', None)  # Early stopping patience (None = disabled)
        
        # Extract config attributes (scheduler)
        self.scheduler_type = self._config.scheduler_type
        self.scheduler_step_size = self._config.scheduler_step_size
        self.scheduler_gamma = self._config.scheduler_gamma
        self.scheduler_patience = self._config.scheduler_patience
        self.scheduler_factor = self._config.scheduler_factor
        self.scheduler_min_lr = self._config.scheduler_min_lr
        
        # Extract initialization method attributes (for use during fit)
        self.f0_init_method = self._get_config_attr('f0_init_method', None)
        self.ar_init_method = self._get_config_attr('ar_init_method', None)
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Random seed
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize components (only if dimensions are available)
        # If data_dim or context_dim is None, components will be built during fit
        if self.data_dim is not None and self.context_dim is not None:
            self._build_components()
        else:
            # Components will be built during fit when dimensions are known
            self.innovation_encoder = None
            self.prior_network = None
            self.decoder = None
            self.ssm = None
        
        # Optimizer (built during training)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Training state
        self.training_state: Optional[Dict] = None
        self.factors: Optional[np.ndarray] = None
        self.innovations: Optional[np.ndarray] = None
        
        # Move to device
        self.to(self.device)
    
    def _get_config_attr(self, attr: str, default: Any = None) -> Any:
        """Helper to safely get config attribute with default."""
        return getattr(self._config, attr, default) if self._config is not None else default
    
    
    def _set_random_f0(self) -> None:
        """Set f0 to random values (fallback for initialization failures)."""
        with torch.no_grad():
            self.ssm.f0.data = torch.randn(self.latent_dim, device=self.device) * 0.1
    
    def _normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """Normalize f0 vector: pad if needed, apply sign convention.
        
        Parameters
        ----------
        f0 : np.ndarray
            Raw f0 vector (may be shorter than latent_dim)
            
        Returns
        -------
        np.ndarray
            Normalized f0 vector (latent_dim,)
        """
        # Pad if needed
        if f0.shape[0] < self.latent_dim:
            f0 = np.pad(f0, (0, self.latent_dim - f0.shape[0]))
        
        # Deterministic sign convention for reproducibility
        if np.sum(f0) < 0:
            f0 = -f0
        
        return f0
    
    def _set_ssm_ar_params(self, ar_coeffs: np.ndarray, B_diag: np.ndarray) -> None:
        """Set SSM AR parameters from numpy arrays.
        
        Parameters
        ----------
        ar_coeffs : np.ndarray
            AR coefficients (r, p) for AR(p) or (r,) for AR(1)
        B_diag : np.ndarray
            Innovation standard deviations (r,)
        """
        with torch.no_grad():
            if self.factor_order == 1:
                # AR(1): initialize A and B
                self.ssm.A.data = torch.tensor(
                    ar_coeffs[:, 0] if ar_coeffs.ndim > 1 else ar_coeffs,
                    dtype=DEFAULT_TORCH_DTYPE,
                    device=self.device
                )
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            else:
                # AR(p): initialize ar_coeffs and B
                self.ssm.ar_coeffs.data = torch.tensor(
                    ar_coeffs, dtype=DEFAULT_TORCH_DTYPE, device=self.device
                )
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
    
    def _build_components(self):
        """Build model components: encoder, decoder, prior, SSM."""
        # Validate dimensions are available
        if self.data_dim is None or self.context_dim is None:
            raise ModelNotInitializedError(
                "Cannot build components: data_dim and context_dim must be set. "
                "Call fit() first to infer dimensions from data."
            )
        
        # Innovation encoder: q(η_t | y_t, u_t)
        self.innovation_encoder = iVDFMInnovationEncoder(
            data_dim=self.data_dim,
            latent_dim=self.latent_dim,
            aux_dim=self.context_dim,
            hidden_dim=self.encoder_hidden_dim,
            n_hidden_layers=self.encoder_n_hidden_layers,
            activation=self.activation,
            slope=self.slope,
            use_layer_norm=self.use_layer_norm,
            device=self.device,
            seed=self.seed,
        )
        
        # Prior network: p(η_t | u_t)
        self.prior_network = iVDFMPriorNetwork(
            aux_dim=self.context_dim,  # Parameter name in encoder/prior (uses aux_dim internally)
            latent_dim=self.latent_dim,
            hidden_dim=self.prior_hidden_dim,
            n_hidden_layers=self.prior_n_hidden_layers,
            activation=self.activation,
            slope=self.slope,
            innovation_distribution=self.innovation_distribution,
            device=self.device,
            seed=self.seed,
        )
        
        # Decoder: g(f_t) → y_t
        self.decoder = iVDFMDecoder(
            latent_dim=self.latent_dim,
            data_dim=self.data_dim,
            hidden_dim=self.decoder_hidden_dim,
            n_hidden_layers=self.decoder_n_hidden_layers,
            activation=self.activation,
            slope=self.slope,
            decoder_var=self.decoder_var,
            use_layer_norm=self.use_layer_norm,
            device=self.device,
            seed=self.seed,
        )
        
        # SSM: maps innovations to factors via deterministic dynamics
        self.ssm = iVDFMCompanionSSM(
            latent_dim=self.latent_dim,
            factor_order=self.factor_order,
            device=self.device,
        )
    
    def _initialize_f0_from_data(self, dataset: 'iVDFMDataset', method: str = 'single_window') -> None:
        """Initialize f_0 (initial factor state) using PCA on data.
        
        Supports three methods:
        - 'single_window': PCA on most recent window (baseline)
        - 'multi_window': Average PCA factors across multiple recent windows
        - 'rolling': Expanding window PCA (all data up to current)
        
        Parameters
        ----------
        dataset : iVDFMDataset
            Dataset containing training data
        method : str, default 'single_window'
            Initialization method: 'single_window', 'multi_window', or 'rolling'
        """
        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total
        if T_init < 2:
            self._set_random_f0()
            _logger.warning(f"Insufficient data for PCA init (T={T_init}). Using random f_0.")
            return

        max_components = min(self.data_dim, T_init, self.latent_dim)
        f0_mean = None
        
        try:
            if method == 'single_window':
                # Baseline: single window PCA
                y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
                f_init, _ = extract_pca_factors(y_win, n_components=max_components)
                f0_mean = np.mean(f_init, axis=0)  # (max_components,)
                
            elif method == 'multi_window':
                # Multi-window: average PCA factors across multiple windows
                n_windows = 3  # Use 3 recent windows
                window_size = T_init
                f0_list = []
                
                for w in range(n_windows):
                    start_idx = max(0, T_total - window_size * (w + 1))
                    end_idx = T_total - window_size * w
                    if end_idx - start_idx < 2:
                        continue
                    
                    y_win = dataset.data[start_idx:end_idx, :]
                    try:
                        f_init, _ = extract_pca_factors(y_win, n_components=max_components)
                        f0_w = np.mean(f_init, axis=0)
                        f0_list.append(f0_w)
                    except Exception:
                        continue
                
                if len(f0_list) > 0:
                    # Average across windows
                    f0_mean = np.mean(f0_list, axis=0)
                else:
                    raise ValueError("No valid windows for multi-window PCA")
                    
            elif method == 'rolling':
                # Rolling: expanding window PCA (all data up to current)
                y_win = dataset.data[:T_total, :]  # All data
                f_init_all, eigenvectors = extract_pca_factors(y_win, n_components=max_components)
                # Use most recent window's factors for f0
                f_init_recent = f_init_all[T_total - T_init:T_total, :]
                f0_mean = np.mean(f_init_recent, axis=0)
            else:
                raise ValueError(f"Unknown f0 initialization method: {method}")
                
        except Exception as e:
            self._set_random_f0()
            _logger.warning(f"PCA init ({method}) failed: {e}. Using random f_0.")
            return

        # Normalize f0 (pad, sign convention)
        if f0_mean is None:
            f0_mean = np.zeros(self.latent_dim, dtype=DEFAULT_DTYPE)
        f0_mean = self._normalize_f0(f0_mean)

        # Set f0
        with torch.no_grad():
            self.ssm.f0.data = torch.tensor(f0_mean, dtype=DEFAULT_TORCH_DTYPE, device=self.device)

        _logger.info(
            f"Initialized f_0 using {method} PCA. "
            f"f_0 range: [{f0_mean.min():.4f}, {f0_mean.max():.4f}]"
        )
    
    def _initialize_ar_coeffs_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Initialize AR coefficients using OLS estimation from PCA factors.
        
        Extracts PCA factors from data, estimates AR(p) coefficients for each factor,
        and initializes SSM parameters.
        
        Parameters
        ----------
        dataset : iVDFMDataset
            Dataset containing training data
        """
        from ...numeric.estimator import estimate_arp_ols
        
        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total
        
        if T_init < self.factor_order + 1:
            _logger.warning(f"Insufficient data for AR init (T={T_init}, order={self.factor_order}). Skipping.")
            return
        
        try:
            # Extract data window and compute PCA factors
            y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
            max_components = min(self.data_dim, T_init, self.latent_dim)
            factors, _ = extract_pca_factors(y_win, n_components=max_components)
            
            # Pad if needed
            if factors.shape[1] < self.latent_dim:
                factors = np.pad(factors, ((0, 0), (0, self.latent_dim - factors.shape[1])))
            
            # Estimate AR(p) coefficients using OLS
            ar_coeffs, B_diag = estimate_arp_ols(
                factors=factors,
                order=self.factor_order,
                regularization=DEFAULT_REGULARIZATION,
                dtype=DEFAULT_DTYPE
            )
            
            # Initialize SSM parameters
            self._set_ssm_ar_params(ar_coeffs, B_diag)
            
            _logger.info(
                f"Initialized AR({self.factor_order}) coefficients using OLS from PCA factors. "
                f"AR coeffs range: [{ar_coeffs.min():.4f}, {ar_coeffs.max():.4f}], "
                f"B range: [{B_diag.min():.4f}, {B_diag.max():.4f}]"
            )
        except Exception as e:
            _logger.warning(f"AR coefficient initialization failed: {e}. Using random initialization.")
    
    def _build_optimizer(self):
        """Build optimizer and scheduler using builder utility."""
        self.optimizer, self.scheduler = build_ivdfm_optimizer(
            model=self,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            max_epochs=self.max_epochs,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_momentum=self.optimizer_momentum,
            scheduler_type=self.scheduler_type,
            scheduler_step_size=self.scheduler_step_size,
            scheduler_gamma=self.scheduler_gamma,
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            scheduler_min_lr=self.scheduler_min_lr,
        )
    
    
    def forward(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through iVDFM.
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Context variable sequence, shape (batch, T, context_dim)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - y_pred: predicted observations, shape (batch, T, N)
            - eta: innovations, shape (batch, T, r)
            - factors: latent factors, shape (batch, T, r)
            - encoder_params: encoder parameters (list of dicts)
            - prior_params: prior parameters (list of dicts)
        """
        batch_size, T, _ = y_1T.shape
        
        # Encode innovations: q(η_t | y_t, u_t) for all time steps
        # Encoder processes (batch, T, N) and (batch, T, context_dim) directly
        mu_all, logvar_all = self.innovation_encoder.forward(y_1T, u_1T)
        # mu_all, logvar_all: (batch, T, r)
        
        # Sample innovations using reparameterization trick
        std_all = torch.exp(0.5 * logvar_all)
        eps = torch.randn_like(mu_all)
        eta_1T = mu_all + eps * std_all  # (batch, T, r)
        
        # Get prior parameters for all time steps: p(η_t | u_t)
        # Prior network processes (batch, T, context_dim) and returns (batch, T, r) per param
        prior_params_all = self.prior_network(u_1T)  # Dict with batched params (batch, T, r)
        
        # Build time-indexed parameter lists for ELBO computation
        # Optimized: pre-extract keys to avoid repeated dict iteration
        prior_keys = list(prior_params_all.keys())
        encoder_params_list = [
            {'mu': mu_all[:, t, :], 'logvar': logvar_all[:, t, :]}
            for t in range(T)
        ]
        prior_params_list = [
            {key: prior_params_all[key][:, t, :] for key in prior_keys}
            for t in range(T)
        ]
        
        # Compute factors deterministically using SSM
        factors_1T = self.ssm.forward(eta_1T)  # (batch, T, r)
        
        # Decode observations
        y_pred = self.decoder(factors_1T)  # (batch, T, N)
        
        return {
            'y_pred': y_pred,
            'eta': eta_1T,
            'factors': factors_1T,
            'encoder_params': encoder_params_list,
            'prior_params': prior_params_list,
        }
    
    def elbo(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor,
        N: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Evidence Lower Bound (ELBO).
        
        ELBO = E[log p(y_t | f_t)] - Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Context variable sequence, shape (batch, T, context_dim)
        N : int
            Total number of samples in dataset (for TC computation, currently unused)
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss (ELBO) and dictionary of component losses
        """
        # Single forward pass
        outputs = self.forward(y_1T, u_1T)
        y_pred = outputs['y_pred']
        encoder_params = outputs['encoder_params']
        prior_params = outputs['prior_params']
        
        # ELBO
        elbo, loss_dict = compute_elbo_loss(
            y_true=y_1T,
            y_pred=y_pred,
            encoder_params=encoder_params,
            prior_params=prior_params,
            innovation_distribution=self.innovation_distribution,
            decoder_variance=self.decoder_var,
            beta_kl=self.beta_kl,
            reduction='mean'
        )
        
        return elbo, loss_dict
    
    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame, iVDFMDataset],
        *args,
        **kwargs
    ) -> 'iVDFM':
        """Fit iVDFM model.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, pd.DataFrame, iVDFMDataset]
            Time series data, or a pre-built `iVDFMDataset`.
            Prefer passing `iVDFMDataset` when you need explicit control over
            `time_idx`, `variables`, `covariates`, and `context` splitting.
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        iVDFM
            Fitted model
        """
        if isinstance(data, iVDFMDataset):
            dataset = data
        else:
            # Context is handled by iVDFMDataset (via config / embedded columns).
            cfg_context = self._get_config_attr("context")
            cfg_scaler = self._get_config_attr("scaler")
            time_context = self._get_config_attr("time_context", 1)
            stride = self._get_config_attr("stride", 1)
            dataset = iVDFMDataset(
                data=data, window=self.window, stride=stride,
                context=cfg_context, time_context=time_context,
                scaler=cfg_scaler, device=self.device,
            )
        
        # Update data_dim and context_dim from dataset
        actual_data_dim = dataset.target_length
        actual_context_dim = dataset.context_length
        
        # Update time_context from config if not already set
        if not hasattr(self, 'time_context') or self.time_context is None:
            self.time_context = self._get_config_attr("time_context", 1)
        
        # Update dimensions
        need_rebuild = False
        if self.data_dim is None:
            self.data_dim = actual_data_dim
            need_rebuild = True
        elif self.data_dim != actual_data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, dataset has {actual_data_dim}",
                details="If data_dim was set in __init__, it must match the observation dimension after extracting context columns"
            )
        
        if self.context_dim is None:
            # Infer from dataset (this is the final dimension: time_context + custom context columns)
            self.context_dim = actual_context_dim
            need_rebuild = True
        elif self.context_dim != actual_context_dim:
            # If context_dim was explicitly set, it must match
            # But if dataset inferred a different dimension from context columns, use dataset's value
            _logger.warning(
                f"context_dim mismatch: model expects {self.context_dim}, dataset has {actual_context_dim}. "
                f"Using dataset's inferred dimension {actual_context_dim}."
            )
            self.context_dim = actual_context_dim
            need_rebuild = True
        
        # Rebuild components if dimensions were updated
        if need_rebuild:
            self._build_components()
        
        # Initialize f_0 (initial factor state) using PCA on initial data
        # Get initialization method from instance attribute (set during __init__)
        f0_init_method = self.f0_init_method if self.f0_init_method is not None else 'single_window'
        _logger.info(f"Using f0_init_method: {f0_init_method} (from config: {getattr(self._config, 'f0_init_method', 'NOT_SET')})")
        self._initialize_f0_from_data(dataset, method=f0_init_method)
        
        # Initialize AR coefficients from data if requested
        ar_init_method = self.ar_init_method
        if ar_init_method == 'ols':
            self._initialize_ar_coeffs_from_data(dataset)
        
        # Build optimizer
        self._build_optimizer()
        
        # Create data loader using dataset's method
        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        # Initialize training logger
        train_logger = iVDFMTrainLogger(verbose=True)
        train_logger.start(
            config={
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_type,
            },
            data_info={
                'num_sequences': len(dataset),
                'data_dim': self.data_dim,
                'context_dim': self.context_dim,
                'latent_dim': self.latent_dim,
            }
        )
        
        # Training loop
        self.train()
        N_total = len(dataset)  # Total number of sequences
        
        self._num_iter = 0
        self.loss_now = None
        self._elbo = None
        self._converged = False
        
        # Early stopping setup
        best_elbo = float('inf')  # Track best ELBO (lower is better, since it's a loss to minimize)
        epochs_without_improvement = 0
        best_model_state = None
        
        early_stop_enabled = self.patience is not None and self.patience > 0
        patience_str = f"early stopping patience={self.patience}" if early_stop_enabled else "early stopping disabled"
        _logger.info(f"Starting training: {self.max_epochs} epochs, {len(dataloader)} batches/epoch, {patience_str}")
        
        import time
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            epoch_recon_losses = []
            epoch_kl_losses = []
            epoch_elbos = []
            
            for batch_idx, (y_batch, u_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                # Forward pass
                elbo, loss_dict = self.elbo(y_batch, u_batch, N_total)
                
                # Check for NaN/Inf
                if torch.isnan(elbo) or torch.isinf(elbo):
                    _logger.error(
                        f"NaN/Inf detected in ELBO at epoch {epoch}, batch {batch_idx}. "
                        f"ELBO={elbo.item()}, Recon={loss_dict['reconstruction'].item()}, "
                        f"KL={loss_dict['kl'].item()}"
                    )
                    raise ValueError("Training failed: NaN/Inf in loss")
                
                # Backward pass
                elbo.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                
                self.optimizer.step()
                
                # Store losses (ensure KL is non-negative)
                recon_val = loss_dict['reconstruction'].item()
                kl_val = loss_dict['kl'].item()
                elbo_val = elbo.item()
                
                # KL should never be negative; fail fast if it happens.
                if kl_val < 0:
                    raise ValueError(
                        f"Training failed: negative KL at epoch {epoch}, batch {batch_idx}: "
                        f"KL={kl_val:.6f} (Recon={recon_val:.6f}, ELBO={elbo_val:.6f})"
                    )
                
                epoch_recon_losses.append(recon_val)
                epoch_kl_losses.append(kl_val)
                epoch_elbos.append(elbo_val)
            
            # Update state
            self._num_iter = epoch + 1
            self.loss_now = np.mean(epoch_recon_losses) if epoch_recon_losses else None
            kl_loss_mean = np.mean(epoch_kl_losses) if epoch_kl_losses else None
            self._elbo = np.mean(epoch_elbos) if epoch_elbos else None
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler (different schedulers need different inputs)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self._elbo if self._elbo is not None else float('inf'))
                else:
                    self.scheduler.step()
            
            # Logging with detailed information
            elapsed_time = time.time() - start_time
            # Prepare log kwargs
            log_kwargs = {
                'learning_rate': current_lr,
                'time_elapsed': f"{elapsed_time:.2f}s"
            }
            
            train_logger.log_epoch(
                epoch=epoch + 1,
                elbo=self._elbo,
                recon_loss=self.loss_now,
                kl_loss=kl_loss_mean,
                **log_kwargs
            )
            
            # Progress indicator every 10 epochs (redundant with train_logger, but provides percentage)
            # Only log if train_logger is not verbose or we want periodic progress updates
            if (epoch + 1) % 10 == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.max_epochs
                _logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"Recon: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"KL: {kl_loss_mean:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Early stopping: track best ELBO and check for improvement
            if early_stop_enabled and self._elbo is not None:
                # ELBO is a loss to minimize, so lower is better
                if self._elbo < best_elbo:
                    best_elbo = self._elbo
                    epochs_without_improvement = 0
                    # Save best model state (deep copy)
                    import copy
                    best_model_state = {
                        'model_state_dict': copy.deepcopy(self.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
                        'epoch': epoch + 1,
                        'elbo': self._elbo,
                    }
                    _logger.debug(f"New best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} at epoch {epoch + 1}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        _logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}: "
                            f"no improvement for {epochs_without_improvement} epochs "
                            f"(best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f}, "
                            f"current ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f})"
                        )
                        # Restore best model state
                        if best_model_state is not None:
                            self.load_state_dict(best_model_state['model_state_dict'])
                            if best_model_state['optimizer_state_dict'] is not None and self.optimizer is not None:
                                self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                            self._elbo = best_model_state['elbo']
                            self._num_iter = best_model_state['epoch']
                            _logger.info(f"Restored best model from epoch {best_model_state['epoch']}")
                        
                        self._converged = True
                        train_logger.log_convergence(
                            converged=True,
                            num_epochs=epoch + 1,
                            final_loss=best_elbo,
                            reason="early_stopping"
                        )
                        break
            
            # Check convergence (tolerance-based)
            if epoch > 0 and self._check_convergence(epoch_elbos):
                self._converged = True
                train_logger.log_convergence(
                    converged=True,
                    num_epochs=epoch + 1,
                    final_loss=self._elbo,
                    reason="converged"
                )
                _logger.info(f"Converged at epoch {epoch + 1}")
                break
        
        # Final logging
        if not self._converged:
            train_logger.log_convergence(
                converged=False,
                num_epochs=self.max_epochs,
                final_loss=self._elbo,
                reason="max_epochs"
            )
        
        # Extract factors and innovations
        self.eval()
        with torch.no_grad():
            # Use full dataset for final extraction
            all_factors = []
            all_innovations = []
            for y_batch, u_batch in dataloader:
                outputs = self.forward(y_batch, u_batch)
                all_factors.append(to_numpy(outputs['factors']))
                all_innovations.append(to_numpy(outputs['eta']))
            
            # Concatenate all batches
            if all_factors:
                self.factors = np.concatenate(all_factors, axis=0)
                self.innovations = np.concatenate(all_innovations, axis=0)
        
        # Store training state
        self.training_state = iVDFMModelState.from_model(self)
        
        return self
    
    def _check_convergence(self, elbo_history: list, window: int = 5) -> bool:
        """Check if training has converged.
        
        Parameters
        ----------
        elbo_history : list
            History of ELBO values
        window : int
            Window size for convergence check
        
        Returns
        -------
        bool
            True if converged
        """
        if len(elbo_history) < window + 1:
            return False
        
        recent = elbo_history[-window:]
        previous = elbo_history[-window-1:-1]
        
        # Check if improvement is below tolerance
        improvement = np.mean(recent) - np.mean(previous)
        return abs(improvement) < self.tolerance
    
    def predict(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        context_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Union[List[str], List[int]]] = None,
        horizon: int = 1,
        deterministic: bool = True,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Predict future values.
        
        Parameters
        ----------
        data : Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]]
            Historical data for prediction. If provided, will extract last factor state.
            If DataFrame, can include context variables via context.
        context_data : Optional[Union[np.ndarray, torch.Tensor]]
            Context variables for prediction horizon, shape (horizon, context_dim).
            If None, will be generated as time-based context.
        context : Optional[Union[List[str], List[int]]]
            Column names (DataFrame) or indices (array) for context variables in data.
            Used only if data is provided and is DataFrame/array with context columns.
        horizon : int
            Prediction horizon (number of steps ahead)
        deterministic : bool, default True
            If True, uses zero innovations (deterministic forecast).
            If False, samples innovations from prior network using context variables.
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        np.ndarray
            Predictions, shape (horizon, data_dim)
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before prediction")
        
        if self.factors is None:
            raise ModelNotTrainedError("Factors not available. Model must be trained with fit()")
        
        # Validate horizon
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        
        self.eval()
        with torch.no_grad():
            # Get last factor state from training
            # factors shape: (batch, T, r) or (T, r)
            if self.factors.ndim == 3:
                # Average over batches if needed
                factors_avg = np.mean(self.factors, axis=0)  # (T, r)
            else:
                factors_avg = self.factors  # (T, r)
            
            # Get last factor state
            f_last = factors_avg[-1, :]  # (r,)
            f_last_tensor = torch.from_numpy(f_last).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, r) for batch dimension
            
            # Generate context variables for forecast horizon
            if context_data is None:
                # Generate time-based context: continue from last training time step
                T_train = factors_avg.shape[0]
                time_indices = np.arange(T_train, T_train + horizon, dtype=np.float32)
                if T_train > 1:
                    # Normalize by training length
                    time_indices = time_indices / (T_train - 1)
                else:
                    time_indices = time_indices / T_train
                
                # Create time features using time_context (not full context_dim which may include custom context)
                if self.time_context == 1:
                    u_future = time_indices.reshape(-1, 1)
                else:
                    features = [time_indices.reshape(-1, 1)]
                    for i in range(1, self.time_context):
                        freq = 2 * np.pi * (i + 1) / T_train if T_train > 1 else 2 * np.pi * (i + 1)
                        periodic = np.sin(freq * np.arange(T_train, T_train + horizon, dtype=np.float32))
                        features.append(periodic.reshape(-1, 1))
                    u_future = np.hstack(features)
                
                # If model has custom context columns, user must provide context_data for prediction
                # (time features alone won't match full context_dim)
                if self.context_dim > self.time_context:
                    _logger.warning(
                        f"Model has custom context columns (context_dim={self.context_dim} > time_context={self.time_context}). "
                        f"Only time features generated. Provide context_data parameter for full context."
                    )
            else:
                # Use provided context_data
                if isinstance(context_data, torch.Tensor):
                    u_future = context_data.cpu().numpy()
                else:
                    u_future = context_data
                
                if u_future.shape[0] != horizon:
                    raise ValueError(
                        f"context_data shape[0] ({u_future.shape[0]}) must match horizon ({horizon})"
                    )
                if u_future.shape[1] != self.context_dim:
                    raise ValueError(
                        f"context_data shape[1] ({u_future.shape[1]}) must match context_dim ({self.context_dim})"
                    )
            
            u_future_tensor = torch.from_numpy(u_future).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, horizon, context_dim)
            
            # Generate innovations for forecast horizon
            if deterministic:
                # Deterministic forecast: zero innovations
                eta_future = torch.zeros(1, horizon, self.latent_dim, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            else:
                # Sample innovations from prior network
                eta_future_list = []
                for h in range(horizon):
                    u_h = u_future_tensor[:, h, :]  # (1, context_dim)
                    prior_params_h = self.prior_network(u_h)  # Dict with distribution parameters
                    
                    # Sample from prior distribution
                    if self.innovation_distribution == 'laplace':
                        location = prior_params_h['location']
                        log_scale = prior_params_h['log_scale']
                        scale = torch.exp(log_scale)
                        # Sample from Laplace
                        u_uniform = torch.rand(1, self.latent_dim, device=self.device) - 0.5
                        eta_h = location + scale * torch.sign(u_uniform) * torch.log(1 - 2 * torch.abs(u_uniform) + 1e-8)
                    elif self.innovation_distribution == 'gaussian':
                        mu = prior_params_h['mu']
                        logvar = prior_params_h['logvar']
                        scale = torch.exp(0.5 * logvar)
                        eta_h = mu + scale * torch.randn(1, self.latent_dim, device=self.device)
                    elif self.innovation_distribution == 'student_t':
                        location = prior_params_h['location']
                        log_scale = prior_params_h['log_scale']
                        log_df = prior_params_h['log_df']
                        scale = torch.exp(log_scale)
                        df = torch.exp(log_df)
                        # Sample from Student-t: location + scale * t(df)
                        # Using normal / sqrt(chi2/df) approximation
                        z = torch.randn(1, self.latent_dim, device=self.device)
                        chi2 = torch.distributions.Gamma(df/2, 0.5).sample((1, self.latent_dim)).to(self.device)
                        eta_h = location + scale * z * torch.sqrt(df / (chi2 + 1e-8))
                    elif self.innovation_distribution == 'gamma':
                        shape = prior_params_h['shape']
                        log_rate = prior_params_h['log_rate']
                        rate = torch.exp(log_rate)
                        # Sample from Gamma
                        eta_h = torch.distributions.Gamma(shape, rate).sample((1,)).to(self.device)
                    elif self.innovation_distribution == 'beta':
                        log_alpha = prior_params_h['log_alpha']
                        log_beta = prior_params_h['log_beta']
                        alpha = torch.exp(log_alpha)
                        beta = torch.exp(log_beta)
                        # Sample from Beta
                        eta_h = torch.distributions.Beta(alpha, beta).sample((1,)).to(self.device)
                    elif self.innovation_distribution == 'exponential':
                        log_rate = prior_params_h['log_rate']
                        rate = torch.exp(log_rate)
                        # Sample from Exponential
                        u = torch.rand(1, self.latent_dim, device=self.device)
                        eta_h = -torch.log(u + 1e-8) / rate
                    else:
                        # Default: Gaussian
                        mu = prior_params_h.get('mu', torch.zeros(1, self.latent_dim, device=self.device))
                        logvar = prior_params_h.get('logvar', torch.zeros(1, self.latent_dim, device=self.device))
                        scale = torch.exp(0.5 * logvar)
                        eta_h = mu + scale * torch.randn(1, self.latent_dim, device=self.device)
                    
                    eta_future_list.append(eta_h)
                
                eta_future = torch.stack(eta_future_list, dim=1)  # (1, horizon, r)
            
            # Forecast factors using SSM
            factors_future = self.ssm.forward_closed_loop(
                f_current=f_last_tensor,  # (1, r)
                eta_future=eta_future,  # (1, horizon, r)
                horizon=horizon
            )  # (1, horizon, r)
            
            # Decode factors to observations
            y_pred = self.decoder(factors_future)  # (1, horizon, data_dim)
            
            # Convert to numpy and remove batch dimension
            y_pred_np = to_numpy(y_pred.squeeze(0))  # (horizon, data_dim)
            
            return y_pred_np
    
    def update(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        *args,
        append: bool = True,
        store_full_history: bool = True,
        verbose: bool = False,
        **kwargs
    ) -> None:
        """Update model state with new observations (online learning).
        
        This method performs a forward pass on new data to update the model's
        internal state (factors and innovations). It does NOT retrain the model
        parameters - for that, call fit() again.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, pd.DataFrame]
            New observation data, shape (T_new, N) or (T_new, N_total) if context provided.
            If DataFrame, columns can include context variables.
        append : bool, default True
            If True, append newly inferred factors/innovations to existing state.
            If False, overwrite the stored state with the newly inferred state.
        store_full_history : bool, default True
            If True, store the inferred full factor/innovation trajectories for the
            provided data windows (legacy behavior). If False, store only the last
            inferred factor/innovation state (shape (1, r)), which is sufficient for
            forecasting and much faster/more memory-efficient.
        verbose : bool, default False
            If True, emit an info log after updating state.
        *args
            Additional arguments (unused).
        **kwargs
            Additional keyword arguments (unused).
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before update")

        # Context is handled by iVDFMDataset (via config / embedded columns).
        cfg_context = self._get_config_attr("context")
        cfg_scaler = self._get_config_attr("scaler")
        time_context = self._get_config_attr("time_context", 1)
        stride = self._get_config_attr("stride", 1)
        dataset = iVDFMDataset(
            data=data, window=self.window, stride=stride,
            context=cfg_context, time_context=time_context,
            scaler=cfg_scaler, device=self.device,
        )
        
        # Validate dimensions match
        if dataset.target_length != self.data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, new data has {dataset.target_length}"
            )
        if dataset.context_length != self.context_dim:
            raise DataValidationError(
                f"context_dim mismatch: model expects {self.context_dim}, new data has {dataset.context_length}"
            )
        
        # Create data loader using dataset's method (no shuffling for update)
        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Forward pass to get new factors and innovations
        self.eval()
        with torch.no_grad():
            if store_full_history:
                all_factors: list[np.ndarray] = []
                all_innovations: list[np.ndarray] = []
            else:
                last_factor_state: Optional[np.ndarray] = None  # (r,)
                last_innovation_state: Optional[np.ndarray] = None  # (r,)
            
            for y_batch, u_batch in dataloader:
                outputs = self.forward(y_batch, u_batch)
                if store_full_history:
                    all_factors.append(to_numpy(outputs['factors']))
                    all_innovations.append(to_numpy(outputs['eta']))
                else:
                    # Keep only the last state (batch, time step) for memory efficiency
                    f_btr = to_numpy(outputs["factors"])
                    e_btr = to_numpy(outputs["eta"])
                    if f_btr.size > 0:
                        last_factor_state = f_btr[-1, -1, :].copy()
                    if e_btr.size > 0:
                        last_innovation_state = e_btr[-1, -1, :].copy()
            
            if store_full_history:
                # Concatenate all batches
                if not all_factors:
                    _logger.warning("No data processed in update")
                    return

                new_factors = np.concatenate(all_factors, axis=0)  # (num_windows, T, r)
                new_innovations = np.concatenate(all_innovations, axis=0)

                # Legacy behavior averaged over windows; keep for backward compatibility.
                # NOTE: This is a heuristic aggregation and may not preserve full temporal order.
                if new_factors.ndim == 3:
                    new_factors = np.mean(new_factors, axis=0)  # (T, r)
                    new_innovations = np.mean(new_innovations, axis=0)
            else:
                if last_factor_state is None or last_innovation_state is None:
                    _logger.warning("No data processed in update")
                    return
                new_factors = last_factor_state.reshape(1, -1)
                new_innovations = last_innovation_state.reshape(1, -1)

            # Update model state: append or overwrite
            if append and self.factors is not None and self.innovations is not None:
                factors_existing = np.mean(self.factors, axis=0) if self.factors.ndim == 3 else self.factors
                innovations_existing = np.mean(self.innovations, axis=0) if self.innovations.ndim == 3 else self.innovations
                self.factors = np.concatenate([factors_existing, new_factors], axis=0)
                self.innovations = np.concatenate([innovations_existing, new_innovations], axis=0)
            else:
                self.factors = new_factors
                self.innovations = new_innovations

            # Update training state
            self.training_state = iVDFMModelState.from_model(self)

            if verbose:
                _logger.info(
                    f"Model state updated with {len(dataset)} new sequences. "
                    f"Factors shape: {self.factors.shape}, Innovations shape: {self.innovations.shape}"
                )
    
    def get_result(self) -> iVDFMResult:
        """Extract result from trained model.
        
        Returns
        -------
        iVDFMResult
            Model result object with all state-space parameters
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        
        # Extract factors and innovations
        if self.factors is None or self.innovations is None:
            raise ModelNotTrainedError("Factors and innovations not available")

        # Decode factors to reconstructions (nonlinear decoder)
        with torch.no_grad():
            z_tensor = torch.from_numpy(self.factors).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            y_hat = self.decoder(z_tensor)
            recon = to_numpy(y_hat)

        # Minimal config snapshot for reproducibility (plain python)
        cfg_snapshot = None
        try:
            from dataclasses import asdict
            cfg_snapshot = asdict(self._config) if self._config is not None else None
        except Exception:
            cfg_snapshot = None

        # Training diagnostics
        num_epochs = int(self._num_iter) if self._num_iter is not None else None
        converged = bool(self._converged)
        training_loss = self.loss_now
        training_elbo = self._elbo

        # Provide weights-only payload (recommended by PyTorch)
        weights = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        # Shared BaseResult fields
        # - use Z for factors (common convention in the codebase)
        # - use x_sm for reconstructions in model space
        z_np = self.factors
        x_sm = recon
        if isinstance(z_np, np.ndarray) and z_np.ndim == 3:
            # (batch, T, r) -> (T, r) for summary/interop
            z_np = np.mean(z_np, axis=0)
        if isinstance(x_sm, np.ndarray) and x_sm.ndim == 3:
            # (batch, T, N) -> (T, N)
            x_sm = np.mean(x_sm, axis=0)

        # Compute full_state (augmented state for companion form)
        # For p=1: full_state = factors (T, r)
        # For p>1: full_state = augmented state (T, r*p) with lags
        full_state = None
        if z_np is not None and isinstance(z_np, np.ndarray):
            T, r = z_np.shape
            if self.factor_order == 1:
                full_state = z_np  # (T, r)
            else:
                # Construct augmented state: s_t[i*p : (i+1)*p] = [f_t[i], f_{t-1}[i], ..., f_{t-p+1}[i]]
                full_state = np.zeros((T, r * self.factor_order), dtype=z_np.dtype)
                f0_np = self.ssm.f0.data.cpu().numpy() if hasattr(self.ssm.f0, 'data') else self.ssm.f0
                for t in range(T):
                    for i in range(r):
                        for lag in range(self.factor_order):
                            idx = t - lag
                            if idx >= 0:
                                full_state[t, i * self.factor_order + lag] = z_np[idx, i]
                            else:
                                # Use f0 for negative indices
                                if f0_np is not None and i < len(f0_np):
                                    full_state[t, i * self.factor_order + lag] = f0_np[i]

        return iVDFMResult(
            innovations=self.innovations,
            reconstructions=recon,
            full_state=full_state,
            x_sm=x_sm,
            Z=z_np,
            r=np.array([self.latent_dim], dtype=int),
            p=int(self.factor_order),
            target_scaler=getattr(self._config, "target_scaler", None) if self._config else None,
            num_iter=int(self._num_iter) if self._num_iter is not None else 0,
            objective=training_elbo,
            training_elbo=training_elbo,
            training_loss=training_loss,
            num_epochs=num_epochs,
            converged=converged,
            config=cfg_snapshot,
            model_state_dict=weights,
        )
    
    def save(self, path: Union[str, Path], *, weights_only: bool = False) -> None:
        """Save model to file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save model
        weights_only : bool
            If True, save *only* the model weights (state_dict) to a .pt file.
            This is the recommended, refactor-safe format.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if weights_only:
            # Pure weights-only file (PyTorch-recommended).
            torch.save(self.state_dict(), path)
            _logger.info(f"Model weights saved to {path}")
            return

        # Backward-compatible checkpoint-style save (contains non-tensors).
        # Save all parameters needed to reconstruct the model architecture
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    # Core dimensions
                    "data_dim": self.data_dim,
                    "latent_dim": self.latent_dim,
                    "context_dim": self.context_dim,
                    "time_context": self.time_context,  # Time feature dimension
                    "window": self.window,
                    # Dynamics parameters
                    "factor_order": self.factor_order,
                    "innovation_distribution": self.innovation_distribution,
                    # Architecture parameters (needed for model reconstruction)
                    "encoder_hidden_dim": self.encoder_hidden_dim,
                    "encoder_n_hidden_layers": self.encoder_n_hidden_layers,
                    "decoder_hidden_dim": self.decoder_hidden_dim,
                    "decoder_n_hidden_layers": self.decoder_n_hidden_layers,
                    "prior_hidden_dim": self.prior_hidden_dim,
                    "prior_n_hidden_layers": self.prior_n_hidden_layers,
                    "activation": self.activation,
                    "slope": self.slope,
                },
                "training_state": self.training_state,
            },
            path,
        )
        _logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], *args, **kwargs) -> 'iVDFM':
        """Load model from file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load model from
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        iVDFM
            Loaded model instance
        """
        path = Path(path)

        # Try weights-only load first (safe for untrusted sources).
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and all(hasattr(v, "dtype") for v in state_dict.values()):
                # weights-only file: caller must provide architecture args/kwargs.
                model = cls(*args, **kwargs)
                model.load_state_dict(state_dict)
                _logger.info(f"Model weights loaded from {path}")
                return model
        except Exception:
            # Fall back to legacy checkpoint below.
            pass

        # Legacy checkpoint load (contains non-tensors): requires weights_only=False.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        config = checkpoint.get("config", {})
        model = cls(**config, **kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.training_state = checkpoint.get("training_state")
        
        # Restore factors and innovations from training_state if available
        if model.training_state is not None:
            if hasattr(model.training_state, 'factors') and model.training_state.factors is not None:
                model.factors = model.training_state.factors
            if hasattr(model.training_state, 'innovations') and model.training_state.innovations is not None:
                model.innovations = model.training_state.innovations
        
        _logger.info(f"Model loaded from {path}")
        return model
