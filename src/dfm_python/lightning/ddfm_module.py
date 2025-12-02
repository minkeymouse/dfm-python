"""PyTorch Lightning module for Deep Dynamic Factor Model.

This module provides DDFMLightningModule which integrates autoencoder training
and MCMC procedure with PyTorch Lightning for training deep DFM models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import pytorch_lightning as pl
from dataclasses import dataclass

from ..config import DFMConfig
from ..models.results import DDFMResult
from ..encoder.vae import Encoder, Decoder, extract_decoder_params
from ..utils.statespace import estimate_idiosyncratic_dynamics
from ..logger import get_logger

_logger = get_logger(__name__)


@dataclass
class DDFMTrainingState:
    """State tracking for DDFM training."""
    factors: np.ndarray
    prediction: np.ndarray
    converged: bool
    num_iter: int
    training_loss: Optional[float] = None


class DDFMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Deep Dynamic Factor Model.
    
    This module implements DDFM training using:
    1. Standard Lightning hooks for autoencoder training
    2. Custom MCMC procedure for iterative factor extraction
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    encoder_layers : List[int], optional
        Hidden layer dimensions for encoder. Default: [64, 32]
    num_factors : int, optional
        Number of factors. If None, inferred from config.
    activation : str, default 'tanh'
        Activation function ('tanh', 'relu', 'sigmoid')
    use_batch_norm : bool, default True
        Whether to use batch normalization in encoder
    learning_rate : float, default 0.001
        Learning rate for Adam optimizer
    epochs : int, default 100
        Number of epochs per MCMC iteration
    batch_size : int, default 32
        Batch size for training
    factor_order : int, default 1
        VAR lag order for factor dynamics (1 or 2)
    use_idiosyncratic : bool, default True
        Whether to model idiosyncratic components
    min_obs_idio : int, default 5
        Minimum observations for idio AR(1) estimation
    """
    
    def __init__(
        self,
        config: DFMConfig,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        activation: str = 'tanh',
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.encoder_layers = encoder_layers or [64, 32]
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs_per_iter = epochs
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        
        # Determine number of factors
        if num_factors is None:
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                self.num_factors = int(np.sum(config.factors_per_block))
            else:
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    self.num_factors = int(np.sum(blocks[:, 0]))
                else:
                    self.num_factors = 1
        else:
            self.num_factors = num_factors
        
        # Initialize encoder and decoder
        # Note: input_dim and output_dim will be set in setup() when we know data dimensions
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        
        # Training state
        self.training_state: Optional[DDFMTrainingState] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        
        # MCMC state
        self.current_mcmc_data: Optional[torch.Tensor] = None
        self.mcmc_iteration: int = 0
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize encoder and decoder when data dimensions are known."""
        # This will be called after data is loaded
        # Encoder/decoder will be initialized in fit_mcmc() when we have data
        pass
    
    def initialize_networks(self, input_dim: int) -> None:
        """Initialize encoder and decoder networks.
        
        Parameters
        ----------
        input_dim : int
            Number of input features (number of series)
        """
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=self.encoder_layers,
            output_dim=self.num_factors,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        
        self.decoder = Decoder(
            input_dim=self.num_factors,
            output_dim=input_dim,
            use_bias=True,
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
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Encoder and decoder must be initialized before forward pass")
        
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
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for autoencoder.
        
        This is used for standard autoencoder training and also called
        during MCMC procedure for each MC sample.
        
        Parameters
        ----------
        batch : tuple
            (data, target) where both are the same for reconstruction
        batch_idx : int
            Batch index
            
        Returns
        -------
        loss : torch.Tensor
            Reconstruction loss (MSE)
        """
        data, target = batch
        
        # Forward pass
        reconstructed = self.forward(data)
        
        # Compute loss (MSE)
        loss = nn.functional.mse_loss(reconstructed, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer for autoencoder training."""
        if self.encoder is None or self.decoder is None:
            return None
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
        
        return optimizer
    
    def fit_mcmc(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        seed: Optional[int] = None,
    ) -> DDFMTrainingState:
        """Run MCMC iterative training procedure.
        
        This method implements the MCMC procedure for DDFM training.
        It alternates between estimating idiosyncratic dynamics, generating
        MC samples, training the autoencoder, and checking convergence.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data with missing values (T x N)
        x_clean : torch.Tensor
            Clean data (interpolated, T x N) used for initial training
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
        Mx : np.ndarray, optional
            Mean values for unstandardization (N,)
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization (N,)
        max_iter : int, default 200
            Maximum number of MCMC iterations
        tolerance : float, default 0.0005
            Convergence tolerance (MSE threshold)
        disp : int, default 10
            Display progress every 'disp' iterations
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        DDFMTrainingState
            Final training state with factors and convergence info
        """
        self.Mx = Mx
        self.Wx = Wx
        self.data_processed = X
        
        device = X.device
        dtype = X.dtype
        T, N = X.shape
        
        # Initialize networks if not done
        if self.encoder is None or self.decoder is None:
            self.initialize_networks(N)
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
        
        # Random number generator for MC sampling
        rng = np.random.RandomState(seed if seed is not None else 3)
        
        # Convert to numpy for MCMC procedure (some operations are easier in numpy)
        x_standardized_np = X.cpu().numpy()
        x_clean_np = x_clean.cpu().numpy()
        bool_no_miss = ~missing_mask
        
        # Initialize data structures
        data_mod_only_miss = x_standardized_np.copy()  # Original with missing values
        data_mod = x_clean_np.copy()  # Clean data (will be modified during MCMC)
        
        # Initial prediction
        x_tensor = x_clean.to(device)
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            factors_init = self.encoder(x_tensor).cpu().numpy()
            factors_tensor = torch.tensor(factors_init, device=device, dtype=dtype)
            prediction_iter = self.decoder(factors_tensor).cpu().numpy()
        
        # Initialize factors
        factors = factors_init.copy()
        
        # Update missing values with initial prediction
        bool_miss = missing_mask
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Initial residuals
        eps = data_mod_only_miss - prediction_iter
        
        # MCMC loop
        iter_count = 0
        not_converged = True
        prediction_prev_iter = None
        delta = float('inf')
        loss_now = float('inf')
        
        _logger.info(f"Starting MCMC training: max_iter={max_iter}, tolerance={tolerance}, epochs_per_iter={self.epochs_per_iter}")
        
        while not_converged and iter_count < max_iter:
            iter_count += 1
            self.mcmc_iteration = iter_count
            
            # Get idiosyncratic distribution
            if self.use_idiosyncratic:
                A_eps, Q_eps = estimate_idiosyncratic_dynamics(eps, missing_mask, self.min_obs_idio)
                # Convert to format expected by MCMC procedure
                phi = A_eps if A_eps.ndim == 2 else np.diag(A_eps) if A_eps.ndim == 1 else np.eye(N)
                mu_eps = np.zeros(N)
                if Q_eps.ndim == 2:
                    std_eps = np.sqrt(np.diag(Q_eps))
                elif Q_eps.ndim == 1:
                    std_eps = np.sqrt(Q_eps)
                else:
                    std_eps = np.ones(N) * 0.1
            else:
                phi = np.zeros((N, N))
                mu_eps = np.zeros(N)
                std_eps = np.ones(N) * 1e-8
            
            # Subtract conditional AR-idio mean from x
            if self.use_idiosyncratic and eps.shape[0] > 1:
                data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
                data_mod[:1] = data_mod_only_miss[:1]
            else:
                data_mod = data_mod_only_miss.copy()
            
            # Generate MC samples for idio (dims = epochs_per_iter x T x N)
            eps_draws = np.zeros((self.epochs_per_iter, T, N))
            for t in range(T):
                eps_draws[:, t, :] = rng.multivariate_normal(
                    mu_eps, np.diag(std_eps), size=self.epochs_per_iter
                )
            
            # Initialize noisy inputs
            x_sim_den = np.zeros((self.epochs_per_iter, T, N))
            
            # Loop over MC samples
            factors_samples = []
            for i in range(self.epochs_per_iter):
                x_sim_den[i, :, :] = data_mod.copy()
                # Corrupt input data by subtracting sampled idio innovations
                x_sim_den[i, :, :] = x_sim_den[i, :, :] - eps_draws[i, :, :]
                
                # Train autoencoder on corrupted sample (1 epoch)
                # Convert to torch and create dataset
                x_sample = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                dataset = torch.utils.data.TensorDataset(x_sample, x_sample)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )
                
                # Train for 1 epoch
                self.encoder.train()
                self.decoder.train()
                optimizer = self.configure_optimizers()
                
                for batch_data, batch_target in dataloader:
                    optimizer.zero_grad()
                    reconstructed = self.forward(batch_data)
                    loss = nn.functional.mse_loss(reconstructed, batch_target)
                    loss.backward()
                    optimizer.step()
                
                # Extract factors from this sample
                x_sample_tensor = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                self.encoder.eval()
                with torch.no_grad():
                    factors_sample = self.encoder(x_sample_tensor).cpu().numpy()
                factors_samples.append(factors_sample)
            
            # Update factors: average over all MC samples
            factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
            
            # Check convergence
            self.decoder.eval()
            with torch.no_grad():
                factors_tensor = torch.tensor(factors, device=device, dtype=dtype)
                prediction_iter = self.decoder(factors_tensor).cpu().numpy()
            
            if iter_count > 1:
                # Compute MSE on non-missing values
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    mse = np.nanmean((prediction_prev_iter[mask] - prediction_iter[mask]) ** 2)
                    delta = mse
                    loss_now = mse
                else:
                    delta = float('inf')
                    loss_now = float('inf')
                
                if iter_count % disp == 0:
                    _logger.info(
                        f"Iteration {iter_count}/{max_iter}: loss={loss_now:.6f}, delta={delta:.6f}"
                    )
                
                if delta < tolerance:
                    not_converged = False
                    _logger.info(
                        f"Convergence achieved in {iter_count} iterations: "
                        f"loss={loss_now:.6f}, delta={delta:.6f} < {tolerance}"
                    )
            else:
                # First iteration: compute initial loss
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    loss_now = np.nanmean((data_mod_only_miss[mask] - prediction_iter[mask]) ** 2)
                else:
                    loss_now = float('inf')
            
            # Store previous prediction for convergence checking
            prediction_prev_iter = prediction_iter.copy()
            
            # Update missing values with current prediction
            if bool_miss.any():
                data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
            
            # Update residuals
            eps = data_mod_only_miss - prediction_iter
        
        if not_converged:
            _logger.warning(
                f"Convergence not achieved within {max_iter} iterations. "
                f"Final delta: {delta:.6f if iter_count > 1 else 'N/A'}"
            )
        
        converged = not not_converged
        
        # Store final state
        self.training_state = DDFMTrainingState(
            factors=factors,
            prediction=prediction_iter,
            converged=converged,
            num_iter=iter_count,
            training_loss=loss_now
        )
        
        return self.training_state
    
    def get_result(self) -> DDFMResult:
        """Extract DDFMResult from trained model.
        
        Returns
        -------
        DDFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        if self.training_state is None:
            raise RuntimeError("Model must be fitted before getting results. Call fit_mcmc() first.")
        
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Encoder and decoder must be initialized")
        
        # Extract decoder parameters (C, bias)
        C, bias = extract_decoder_params(self.decoder)
        
        # Get factors and prediction
        factors = self.training_state.factors  # T x num_factors
        prediction_iter = self.training_state.prediction  # T x N
        
        # Convert to numpy
        C = C.cpu().numpy() if isinstance(C, torch.Tensor) else C
        bias = bias.cpu().numpy() if isinstance(bias, torch.Tensor) else bias
        
        # Compute residuals and estimate idiosyncratic dynamics
        if self.data_processed is not None:
            x_standardized = self.data_processed.cpu().numpy()
            residuals = x_standardized - prediction_iter
        else:
            residuals = np.zeros_like(prediction_iter)
        
        # Estimate factor dynamics (VAR)
        from ..utils.statespace import estimate_var1, estimate_var2
        
        if self.factor_order == 1:
            A_f, Q_f = estimate_var1(factors)
        elif self.factor_order == 2:
            A_f, Q_f = estimate_var2(factors)
        else:
            raise ValueError(f"factor_order must be 1 or 2, got {self.factor_order}")
        
        # For DDFM, we use simplified state-space (factor-only)
        A = A_f
        Q = Q_f
        Z_0 = factors[0, :]
        V_0 = np.cov(factors.T)
        
        # Estimate R from residuals
        R_diag = np.var(residuals, axis=0)
        R = np.diag(np.maximum(R_diag, 1e-8))
        
        # Compute smoothed data
        x_sm = prediction_iter  # T x N (standardized)
        
        # Unstandardize
        Wx_clean = np.where(np.isnan(self.Wx), 1.0, self.Wx) if self.Wx is not None else np.ones(C.shape[0])
        Mx_clean = np.where(np.isnan(self.Mx), 0.0, self.Mx) if self.Mx is not None else np.zeros(C.shape[0])
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
            Mx=self.Mx if self.Mx is not None else np.zeros(C.shape[0]),
            Wx=self.Wx if self.Wx is not None else np.ones(C.shape[0]),
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=self.factor_order,
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=None,  # DDFM doesn't compute loglik in same way
            series_ids=self.config.get_series_ids() if hasattr(self.config, 'get_series_ids') else None,
            block_names=getattr(self.config, 'block_names', None),
            training_loss=self.training_state.training_loss,
            encoder_layers=self.encoder_layers,
            use_idiosyncratic=self.use_idiosyncratic,
        )
        
        return result

