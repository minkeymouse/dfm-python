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
    DEFAULT_IVDFM_SEQUENCE_LENGTH,
    DEFAULT_IVDFM_AUX_DIM,
)
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError, DataValidationError
from ...utils.validation import check_condition
from ...utils.loss import compute_ivdfm_elbo
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
        sequence_length: Optional[int] = None,
        config: Optional[iVDFMConfig] = None,
        # Network architecture (used if config is None)
        encoder_hidden_dim: Union[int, list] = 200,
        encoder_n_layers: int = 3,
        decoder_hidden_dim: Union[int, list] = 200,
        decoder_n_layers: int = 3,
        prior_hidden_dim: Union[int, list] = 100,
        prior_n_layers: int = 2,
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
        sequence_length : int
            Length of time series sequences (T in paper)
        config : Optional[Any]
            Configuration object
        encoder_hidden_dim : Union[int, list]
            Hidden dimensions for innovation encoder
        encoder_n_layers : int
            Number of layers in innovation encoder
        decoder_hidden_dim : Union[int, list]
            Hidden dimensions for decoder
        decoder_n_layers : int
            Number of layers in decoder
        prior_hidden_dim : Union[int, list]
            Hidden dimensions for prior network
        prior_n_layers : int
            Number of layers in prior network
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
        if encoder_n_layers is not None:
            config_dict['encoder_n_layers'] = encoder_n_layers
        if decoder_hidden_dim is not None:
            config_dict['decoder_hidden_dim'] = decoder_hidden_dim
        if decoder_n_layers is not None:
            config_dict['decoder_n_layers'] = decoder_n_layers
        if prior_hidden_dim is not None:
            config_dict['prior_hidden_dim'] = prior_hidden_dim
        if prior_n_layers is not None:
            config_dict['prior_n_layers'] = prior_n_layers
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
        if sequence_length is not None:
            config_dict['sequence_length'] = sequence_length
        
        # Override with kwargs (highest precedence)
        # Backward compatibility: map latent_dim -> num_factors if provided in kwargs
        if 'latent_dim' in kwargs and 'num_factors' not in kwargs:
            kwargs['num_factors'] = kwargs.pop('latent_dim')
            _logger.warning(
                "Parameter 'latent_dim' is deprecated. Use 'num_factors' instead. "
                "This mapping will be removed in a future version."
            )
        config_dict.update(kwargs)
        
        # Remove None values to use defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
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
        # context_dim: preserve None if not provided, use config value only if explicitly set
        self.context_dim = context_dim if context_dim is not None else (self._config.context_dim if 'context_dim' in config_dict else None)
        self.sequence_length = self._config.sequence_length if sequence_length is None else sequence_length
        self.encoder_hidden_dim = self._config.encoder_hidden_dim
        self.encoder_n_layers = self._config.encoder_n_layers
        self.decoder_hidden_dim = self._config.decoder_hidden_dim
        self.decoder_n_layers = self._config.decoder_n_layers
        self.prior_hidden_dim = self._config.prior_hidden_dim
        self.prior_n_layers = self._config.prior_n_layers
        self.activation = self._config.activation
        self.slope = self._config.slope
        self.factor_order = self._config.factor_order
        self.innovation_distribution = self._config.innovation_distribution
        self.decoder_var = self._config.decoder_var
        self.learning_rate = self._config.learning_rate
        self.optimizer_type = self._config.optimizer
        self.optimizer_weight_decay = self._config.optimizer_weight_decay
        self.optimizer_momentum = self._config.optimizer_momentum
        self.batch_size = self._config.batch_size
        self.max_epochs = self._config.max_epochs
        self.tolerance = self._config.tolerance
        self.scheduler_type = self._config.scheduler_type
        self.scheduler_step_size = self._config.scheduler_step_size
        self.scheduler_gamma = self._config.scheduler_gamma
        self.scheduler_patience = self._config.scheduler_patience
        self.scheduler_factor = self._config.scheduler_factor
        self.scheduler_min_lr = self._config.scheduler_min_lr
        
        
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
            n_layers=self.encoder_n_layers,
            activation=self.activation,
            slope=self.slope,
            device=self.device,
            seed=self.seed,
        )
        
        # Prior network: p(η_t | u_t)
        self.prior_network = iVDFMPriorNetwork(
            aux_dim=self.context_dim,  # Parameter name in encoder/prior (uses aux_dim internally)
            latent_dim=self.latent_dim,
            hidden_dim=self.prior_hidden_dim,
            n_layers=self.prior_n_layers,
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
            n_layers=self.decoder_n_layers,
            activation=self.activation,
            slope=self.slope,
            decoder_var=self.decoder_var,
            device=self.device,
            seed=self.seed,
        )
        
        # SSM: maps innovations to factors via deterministic dynamics
        self.ssm = iVDFMCompanionSSM(
            latent_dim=self.latent_dim,
            factor_order=self.factor_order,
            device=self.device,
        )
    
    def _initialize_f0_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Initialize f_0 (initial factor state) using PCA on recent data.
        
        Use the most recent `sequence_length` window, extract PCA factors, and
        set f_0 to the mean factor vector from that window.
        
        Parameters
        ----------
        dataset : iVDFMDataset
            Dataset containing training data
        """
        from ...layer.pca import fit_pca
        
        T_total = len(dataset.data)
        T_init = min(self.sequence_length, T_total)
        if T_init < 2:
            with torch.no_grad():
                self.ssm.f0.data = torch.randn(self.latent_dim, device=self.device) * 0.1
            _logger.warning(f"Insufficient data for PCA init (T={T_init}). Using random f_0.")
            return

        # Extract most recent window
        y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
        
        # Center the data
        y_mean = np.mean(y_win, axis=0, keepdims=True)
        y_centered = y_win - y_mean
        
        # PCA: extract up to min(data_dim, T_init, latent_dim) components, then pad if needed
        max_components = min(self.data_dim, T_init, self.latent_dim)
        try:
            _, eigenvectors, _, _ = fit_pca(
                X=y_centered,
                n_components=max_components,
                block_idx=None
            )

            f_init = y_centered @ eigenvectors  # (T_init, max_components)
            f0_mean = np.mean(f_init, axis=0)  # (max_components,)
        except Exception as e:
            with torch.no_grad():
                self.ssm.f0.data = torch.randn(self.latent_dim, device=self.device) * 0.1
            _logger.warning(f"PCA init failed: {e}. Using random f_0.")
            return

        if f0_mean.shape[0] < self.latent_dim:
            f0_mean = np.pad(f0_mean, (0, self.latent_dim - f0_mean.shape[0]))

        # (Optional) deterministic sign convention for reproducibility
        if np.sum(f0_mean) < 0:
            f0_mean = -f0_mean

        with torch.no_grad():
            self.ssm.f0.data = torch.tensor(f0_mean, dtype=DEFAULT_TORCH_DTYPE, device=self.device)

        _logger.info(
            f"Initialized f_0 using PCA on most recent {T_init} time steps. "
            f"f_0 range: [{f0_mean.min():.4f}, {f0_mean.max():.4f}]"
        )
    
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
        encoder_params_list = []
        prior_params_list = []
        for t in range(T):
            encoder_params_list.append({
                'mu': mu_all[:, t, :],      # (batch, r)
                'logvar': logvar_all[:, t, :]  # (batch, r)
            })
            prior_params_t = {}
            for key, value in prior_params_all.items():
                # value shape: (batch, T, r) or (batch, T, ...)
                prior_params_t[key] = value[:, t, :]  # (batch, r)
            prior_params_list.append(prior_params_t)
        
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
            ELBO loss and dictionary of component losses
        """
        # Use abstracted ELBO computation from utils
        elbo, loss_dict = compute_ivdfm_elbo(
            model=self,
            y_1T=y_1T,
            u_1T=u_1T,
            innovation_distribution=self.innovation_distribution,
            decoder_variance=self.decoder_var,
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
            cfg_context = getattr(self._config, "context", None) if self._config is not None else None
            cfg_scaler = getattr(self._config, "scaler", None) if self._config is not None else None

            # Create dataset (handles DataFrame/array conversion and context extraction)
            dataset = iVDFMDataset(
                data=data,
                sequence_length=self.sequence_length,
                context=cfg_context,
                context_dim=self.context_dim if self.context_dim is not None else DEFAULT_IVDFM_AUX_DIM,
                scaler=cfg_scaler,
                device=self.device,
            )
        
        # Update data_dim and context_dim from dataset
        actual_data_dim = dataset.target_length
        actual_context_dim = dataset.context_length
        
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
            # Infer from dataset
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
        self._initialize_f0_from_data(dataset)
        
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
        
        _logger.info(f"Starting training: {self.max_epochs} epochs, {len(dataloader)} batches/epoch")
        
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
                    # ReduceLROnPlateau needs a metric (use ELBO as loss)
                    self.scheduler.step(self._elbo if self._elbo is not None else float('inf'))
                else:
                    # Other schedulers (StepLR, CosineAnnealingLR, ExponentialLR) step on epoch
                    self.scheduler.step()
            
            # Logging with detailed information
            elapsed_time = time.time() - start_time
            train_logger.log_epoch(
                epoch=epoch + 1,
                elbo=self._elbo,
                recon_loss=self.loss_now,
                kl_loss=kl_loss_mean,
                learning_rate=current_lr,
                time_elapsed=f"{elapsed_time:.2f}s"
            )
            
            # Progress indicator every epoch (if verbose) or every 10 epochs
            if (epoch + 1) % max(1, self.max_epochs // 20) == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.max_epochs
                _logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"Recon: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"KL: {kl_loss_mean:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Check convergence
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
                
                # Create time features
                if self.context_dim == 1:
                    u_future = time_indices.reshape(-1, 1)
                else:
                    features = [time_indices.reshape(-1, 1)]
                    for i in range(1, self.context_dim):
                        freq = 2 * np.pi * (i + 1) / T_train if T_train > 1 else 2 * np.pi * (i + 1)
                        periodic = np.sin(freq * np.arange(T_train, T_train + horizon, dtype=np.float32))
                        features.append(periodic.reshape(-1, 1))
                    u_future = np.hstack(features)
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
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before update")

        # Context is handled by iVDFMDataset (via config / embedded columns).
        cfg_context = getattr(self._config, "context", None) if self._config is not None else None
        cfg_scaler = getattr(self._config, "scaler", None) if self._config is not None else None

        # Create dataset for new data
        dataset = iVDFMDataset(
            data=data,
            sequence_length=self.sequence_length,
            context=cfg_context,
            context_dim=self.context_dim if self.context_dim is not None else DEFAULT_IVDFM_AUX_DIM,
            scaler=cfg_scaler,
            device=self.device,
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
            all_factors = []
            all_innovations = []
            
            for y_batch, u_batch in dataloader:
                outputs = self.forward(y_batch, u_batch)
                all_factors.append(to_numpy(outputs['factors']))
                all_innovations.append(to_numpy(outputs['eta']))
            
            # Concatenate all batches
            if all_factors:
                new_factors = np.concatenate(all_factors, axis=0)  # (batch*num_sequences, T, r) or (num_sequences, T, r)
                new_innovations = np.concatenate(all_innovations, axis=0)
                
                # Average over batches if 3D to get (T_total, r)
                if new_factors.ndim == 3:
                    # Average over batch dimension: (batch, T, r) -> (T, r)
                    new_factors = np.mean(new_factors, axis=0)
                    new_innovations = np.mean(new_innovations, axis=0)
                
                # Update model state
                # If factors already exist, append; otherwise replace
                if self.factors is not None:
                    # Normalize existing factors to 2D if needed
                    factors_existing = self.factors
                    innovations_existing = self.innovations
                    
                    if factors_existing.ndim == 3:
                        # Average over batch dimension
                        factors_existing = np.mean(factors_existing, axis=0)
                        innovations_existing = np.mean(innovations_existing, axis=0)
                    
                    # Concatenate along time axis
                    self.factors = np.concatenate([factors_existing, new_factors], axis=0)  # (T_old + T_new, r)
                    self.innovations = np.concatenate([innovations_existing, new_innovations], axis=0)
                else:
                    # No existing factors: use new data
                    self.factors = new_factors
                    self.innovations = new_innovations
                
                # Update training state
                self.training_state = iVDFMModelState.from_model(self)
                
                _logger.info(
                    f"Model state updated with {len(dataset)} new sequences. "
                    f"Factors shape: {self.factors.shape}, Innovations shape: {self.innovations.shape}"
                )
            else:
                _logger.warning("No data processed in update")
    
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
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "data_dim": self.data_dim,
                    "latent_dim": self.latent_dim,
                    "context_dim": self.context_dim,
                    "sequence_length": self.sequence_length,
                    "factor_order": self.factor_order,
                    "innovation_distribution": self.innovation_distribution,
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
        
        _logger.info(f"Model loaded from {path}")
        return model
