"""PyTorch Lightning Trainer and Denoising Trainer for Deep Dynamic Factor Model (DDFM).

This module provides:
- DDFMTrainer: A specialized PyTorch Lightning Trainer class for DDFM models
- DDFMDenoisingTrainer: A custom denoising (MCMC-based) training procedure for DDFM
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np
import torch

from ..logger import get_logger
from ..config import DFMConfig, DDFMConfig
from ..numeric.estimator import estimate_idio_dynamics
from . import (
    _create_base,
    _extract_train_params,
    _validate_config,
    DDFM_TRAINER_DEFAULTS
)

if TYPE_CHECKING:
    from ..models.ddfm import DDFM, DDFMTrainingState

_logger = get_logger(__name__)


class DDFMTrainer(pl.Trainer):
    """Specialized PyTorch Lightning Trainer for DDFM models.
    
    This trainer provides sensible defaults for training DDFM models using
    neural networks (autoencoders). It includes appropriate callbacks and
    logging for deep learning training.
    
    Default Values:
        - max_epochs: 100 (training epochs)
        - enable_progress_bar: True
        - enable_model_summary: True (useful for debugging DDFM architecture)
        - logger: True (uses TensorBoardLogger with CSVLogger fallback)
        - accelerator: 'auto'
        - devices: 'auto'
        - precision: 32
        - gradient_clip_val: 1.0 (default, for numerical stability)
        - accumulate_grad_batches: 1
    
    These defaults are optimized for DDFM neural network training. The trainer
    automatically sets up early stopping (patience=20), learning rate monitor,
    and model checkpoint callbacks.
    
    Parameters
    ----------
    max_epochs : int, default 100
        Maximum number of training epochs
    enable_progress_bar : bool, default True
        Whether to show progress bar during training
    enable_model_summary : bool, default True
        Whether to print model summary (useful for debugging DDFM architecture)
    logger : bool or Logger, default True
        Whether to use a logger. Can be False, True (uses TensorBoardLogger), or a Logger instance
    callbacks : List[Callback], optional
        Additional callbacks beyond defaults
    accelerator : str, default 'auto'
        Accelerator type ('cpu', 'gpu', 'auto', etc.)
    devices : int or List[int], default 'auto'
        Device configuration
    precision : str or int, default 32
        Training precision (16, 32, 'bf16', etc.)
    gradient_clip_val : float, optional, default 1.0
        Gradient clipping value for numerical stability. Default 1.0 helps prevent
        gradient explosion that can cause NaN values during training.
    accumulate_grad_batches : int, default 1
        Number of batches to accumulate gradients before optimizer step
    **kwargs
        Additional arguments passed to pl.Trainer
    
    Examples
    --------
    >>> from dfm_python.trainer import DDFMTrainer
    >>> from dfm_python import DDFM, DDFMDataModule
    >>> 
    >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> dm = DDFMDataModule(config_path='config.yaml', data=df)
    >>> trainer = DDFMTrainer(max_epochs=100, enable_progress_bar=True)
    >>> trainer.fit(model, dm)
    """
    
    def __init__(
            self,
            max_epochs: int = 100,
            enable_progress_bar: bool = True,
            enable_model_summary: bool = True,
            logger: Optional[Any] = True,
            callbacks: Optional[List[Any]] = None,
            accelerator: str = 'auto',
            devices: Any = 'auto',
            precision: Any = 32,
            gradient_clip_val: Optional[float] = 1.0,  # Default: 1.0 for numerical stability
            accumulate_grad_batches: int = 1,
            **kwargs
    ):
        # Setup DDFM-specific callbacks (learning rate monitor and checkpoint)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            filename='ddfm-{epoch:02d}-{train_loss:.4f}'
        )
        
        # Use common trainer base setup with DDFM-specific parameters
        # DDFM uses 'train_loss' metric, TensorBoard logger, patience=20, and additional callbacks
        trainer_config = _create_base(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            logger=logger,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            early_stopping_patience=20,  # More patience for neural network training
            early_stopping_min_delta=1e-6,  # Minimum change for improvement
            early_stopping_monitor='train_loss',  # DDFM uses 'train_loss' metric
            logger_type='tensorboard',  # DDFM uses TensorBoard logger
            logger_name='ddfm',
            additional_callbacks=[lr_monitor, checkpoint],  # DDFM-specific callbacks
            gradient_clip_val=gradient_clip_val,  # DDFM-specific parameter
            accumulate_grad_batches=accumulate_grad_batches,  # DDFM-specific parameter
            **kwargs
        )
        
        # Store attributes for testing/verification
        # Note: These are stored as instance attributes to allow tests to verify
        # default values. The parent Trainer class also stores these, but storing
        # them here ensures they're accessible even if parent implementation changes.
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        
        # Call parent constructor with configured parameters
        super().__init__(**trainer_config)
    
    @classmethod
    def from_config(
        cls,
        config: Union[DFMConfig, DDFMConfig],
        **kwargs
    ) -> 'DDFMTrainer':
        """Create DDFMTrainer from DDFMConfig or DFMConfig.
        
        Extracts training parameters from config and creates trainer with
        appropriate settings for neural network training. Parameters can be overridden via kwargs.
        
        Parameters
        ----------
        config : Union[DFMConfig, DDFMConfig]
            Configuration object (can be DDFMConfig or DDFMConfig with DDFM parameters)
        **kwargs
            Additional arguments to override config values.
            Supported parameters: max_epochs, enable_progress_bar, enable_model_summary, gradient_clip_val.
            For additional Trainer parameters, use __init__() directly.
            
        Returns
        -------
        DDFMTrainer
            Configured trainer instance
        """
        # Validate config before processing
        _validate_config(config, trainer_name="DDFMTrainer")
        
        # Extract training parameters from config and kwargs
        # Handle both DDFMConfig and DFMConfig with ddfm_* parameters
        # Note: Don't use max_iter for DDFM (only epochs/ddfm_epochs)
        # Use constants from trainer/__init__.py to ensure single source of truth
        # These defaults match __init__() defaults for consistency
        # Note: _extract_train_params() modifies kwargs by popping extracted keys
        # After extraction, only extracted parameters are used (kwargs are consumed)
        params = _extract_train_params(config, kwargs, DDFM_TRAINER_DEFAULTS, use_max_iter=False)
        
        # Create trainer with extracted parameters
        # All relevant parameters are extracted, so kwargs are not passed through
        # If additional Trainer parameters are needed, use __init__() directly
        return cls(**params)


# ============================================================================
# Denoising Training Procedure
# ============================================================================

class DDFMDenoisingTrainer:
    """Denoising (MCMC-based) training procedure for DDFM.
    
    This class encapsulates the iterative denoising training procedure for DDFM,
    which alternates between:
    1. Estimating idiosyncratic dynamics (AR parameters)
    2. Generating Monte Carlo samples from the state-space model
    3. Training the autoencoder (encoder/decoder) on corrupted MC samples (denoising)
    4. Extracting factors from trained encoder
    5. Checking convergence based on MSE between predictions and data
    
    The procedure continues until convergence (MSE change < tolerance) or
    maximum iterations reached.
    """
    
    def __init__(self, model: 'DDFM'):
        """Initialize denoising trainer with DDFM model instance.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance to train. The trainer will access model's
            encoder, decoder, and other attributes/methods.
        """
        self.model = model
    
    def fit(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        target_scaler: Optional[Any] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> 'DDFMTrainingState':
        """Run denoising iterative training procedure for DDFM.
        
        This method implements the denoising (MCMC-based) procedure for
        Deep Dynamic Factor Model training. It alternates between:
        1. Estimating idiosyncratic dynamics (AR parameters)
        2. Generating Monte Carlo samples from the state-space model
        3. Training the autoencoder (encoder/decoder) on corrupted MC samples (denoising)
        4. Extracting factors from trained encoder
        5. Checking convergence based on MSE between predictions and data
        
        The procedure continues until convergence (MSE change < tolerance) or
        maximum iterations reached.
        
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
            Maximum number of iterations. If None, uses model.max_iter (default: 200).
        tolerance : float, optional
            Convergence tolerance for MSE change between iterations. If None, uses
            model.tolerance (default: 0.0005). Training stops when |MSE_new - MSE_old| < tolerance.
        disp : int, optional
            Display progress every 'disp' iterations. If None, uses model.disp (default: 10).
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
            
        Raises
        ------
        ValueError
            If X and x_clean have mismatched shapes.
            If missing_mask shape doesn't match X shape.
            If numerical issues occur (NaN/Inf propagation) that cannot be handled.
        """
        # Store target scaler for inverse transformation in prediction
        if target_scaler is not None:
            self.model.target_scaler = target_scaler
        
        # Store processed data
        self.model.data_processed = X
        
        device = X.device
        dtype = X.dtype
        T, N = X.shape
        
        # Validate shape consistency between X, x_clean, and missing_mask
        if x_clean.shape != X.shape:
            raise ValueError(
                f"{self.model.__class__.__name__} fit_mcmc failed: shape mismatch between X ({X.shape}) and x_clean ({x_clean.shape}). "
                f"Both X and x_clean must have the same shape (T x N). "
                f"This indicates data preprocessing inconsistency. "
                f"Please ensure x_clean is created from the same data as X."
            )
        if missing_mask.shape != (T, N):
            raise ValueError(
                f"{self.model.__class__.__name__} fit_mcmc failed: shape mismatch between X ({X.shape}) and missing_mask ({missing_mask.shape}). "
                f"missing_mask must have shape (T x N) matching X. "
                f"This indicates missing_mask was created from data with different shape. "
                f"Please ensure missing_mask is created from the same data passed as X parameter."
            )
        
        # Use instance attributes if not provided
        max_iter = max_iter if max_iter is not None else self.model.max_iter
        tolerance = tolerance if tolerance is not None else self.model.tolerance
        disp = disp if disp is not None else self.model.disp
        
        # Initialize networks if not done
        if self.model.encoder is None or self.model.decoder is None:
            self.model.initialize_networks(N)
        
        # Ensure encoder and decoder are on the correct device
        self.model.encoder = self.model.encoder.to(device)
        self.model.decoder = self.model.decoder.to(device)
        
        # Random number generator for MC sampling
        rng = np.random.RandomState(seed if seed is not None else (self.model.rng.randint(0, 2**31) if hasattr(self.model.rng, 'randint') else 3))
        
        # Convert to numpy for denoising procedure
        x_standardized_np = X.cpu().numpy()
        x_clean_np = x_clean.cpu().numpy()
        bool_no_miss = ~missing_mask
        
        # Initialize data structures
        data_mod_only_miss = x_standardized_np.copy()  # Original with missing values
        data_mod = x_clean_np.copy()  # Clean data (will be modified during denoising)
        z_actual = x_standardized_np.copy()  # Actual observations (target for training)
        
        # Initial prediction
        from ..utils.misc import check_finite_array
        from ..config.constants import MAX_EIGENVALUE
        x_tensor = x_clean.to(device)
        self.model.encoder.eval()
        self.model.decoder.eval()
        with torch.no_grad():
            factors_init = self.model.encoder(x_tensor).cpu().numpy()
            try:
                factors_init = check_finite_array(factors_init, "initial factors", context="at iteration 0", fallback=None)
            except ValueError:
                factors_init = np.nan_to_num(factors_init, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
            
            factors_tensor = torch.tensor(factors_init, device=device, dtype=dtype)
            prediction_iter = self.model.decoder(factors_tensor).cpu().numpy()
            try:
                prediction_iter = check_finite_array(prediction_iter, "initial prediction", context="at iteration 0", fallback=None)
            except ValueError:
                prediction_iter = np.nan_to_num(prediction_iter, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
        
        # Initialize factors
        factors = factors_init.copy()
        
        # Update missing values with initial prediction
        bool_miss = missing_mask
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Initial residuals
        eps = data_mod_only_miss - prediction_iter
        try:
            eps = check_finite_array(eps, "initial residuals", context="at iteration 0", fallback=None)
        except ValueError:
            eps = np.nan_to_num(eps, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
        
        # Denoising loop
        iter_count = 0
        not_converged = True
        prediction_prev_iter = None
        delta = float('inf')
        loss_now = float('inf')
        
        # Check for very small dataset and warn about potential instability
        if T < 10:
            _logger.warning(
                f"{self.model.__class__.__name__} denoising training: very small dataset (T={T} < 10) may cause unstable sampling. "
                f"With only {T} time periods, encoder/decoder training per iteration may have high variance. "
                f"Factor extraction and VAR estimation will use fallback strategies. Results may be less reliable. "
                f"Monitor convergence carefully. Consider reducing num_factors or using smaller encoder_layers for better stability"
            )
        
        _logger.info(f"Starting denoising training: max_iter={max_iter}, tolerance={tolerance}, epochs_per_iter={self.model.epochs_per_iter}")
        
        # Create optimizer once per iteration (reused across MC samples)
        optimizer = None
        
        while not_converged and iter_count < max_iter:
            iter_count += 1
            self.model.mcmc_iteration = iter_count
            
            # Create optimizer for this iteration (reused across MC samples)
            optimizer = self.model._create_optimizer()
            
            # Get idiosyncratic distribution
            if self.model.use_idiosyncratic:
                A_eps, Q_eps = estimate_idio_dynamics(eps, missing_mask, self.model.min_obs_idio)
                try:
                    A_eps = check_finite_array(A_eps, f"idiosyncratic AR coefficients (A_eps)", context=f"at iteration {iter_count}", fallback=None)
                except ValueError:
                    A_eps = np.nan_to_num(A_eps, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
                try:
                    Q_eps = check_finite_array(Q_eps, f"idiosyncratic innovation covariance (Q_eps)", context=f"at iteration {iter_count}", fallback=None)
                except ValueError:
                    Q_eps = np.nan_to_num(Q_eps, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
                
                # Convert to format expected by denoising procedure
                phi = A_eps if A_eps.ndim == 2 else np.diag(A_eps) if A_eps.ndim == 1 else np.eye(N)
                mu_eps = np.zeros(N)
                if Q_eps.ndim == 2:
                    std_eps = np.sqrt(np.diag(Q_eps))
                elif Q_eps.ndim == 1:
                    std_eps = np.sqrt(Q_eps)
                else:
                    std_eps = np.ones(N) * 0.1
                
                # Ensure std_eps is finite and positive
                std_eps = np.maximum(std_eps, 1e-8)
                try:
                    std_eps = check_finite_array(std_eps, f"idiosyncratic std (std_eps)", context=f"at iteration {iter_count}", fallback=None)
                except ValueError:
                    std_eps = np.nan_to_num(std_eps, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
            else:
                phi = np.zeros((N, N))
                mu_eps = np.zeros(N)
                std_eps = np.ones(N) * 1e-8
            
            # Subtract conditional AR-idio mean from x
            if self.model.use_idiosyncratic and eps.shape[0] > 1:
                data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
                data_mod[:1] = data_mod_only_miss[:1]
            else:
                data_mod = data_mod_only_miss.copy()
            
            # Generate MC samples for idio (dims = epochs_per_iter x T x N)
            eps_draws = np.zeros((self.model.epochs_per_iter, T, N))
            try:
                for t in range(T):
                    eps_draws[:, t, :] = rng.multivariate_normal(
                        mu_eps, np.diag(std_eps), size=self.model.epochs_per_iter
                    )
                try:
                    eps_draws = check_finite_array(eps_draws, f"MC samples (eps_draws)", context=f"at iteration {iter_count}", fallback=None)
                except ValueError:
                    eps_draws = np.nan_to_num(eps_draws, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
            except (ValueError, np.linalg.LinAlgError) as e:
                _logger.warning(
                    f"{self.model.__class__.__name__} denoising iteration {iter_count}: failed to generate MC samples: {e}. "
                    f"Using zero samples as fallback"
                )
                eps_draws = np.zeros((self.model.epochs_per_iter, T, N))
            
            # Initialize noisy inputs
            x_sim_den = np.zeros((self.model.epochs_per_iter, T, N))
            
            # Loop over MC samples
            factors_samples = []
            for i in range(self.model.epochs_per_iter):
                x_sim_den[i, :, :] = data_mod.copy()
                # Corrupt input data by subtracting sampled idio innovations (denoising)
                x_sim_den[i, :, :] = x_sim_den[i, :, :] - eps_draws[i, :, :]
                
                # Train autoencoder on corrupted sample (1 epoch) - denoising training
                x_sample = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                z_actual_tensor = torch.tensor(z_actual, device=device, dtype=dtype)
                dataset = torch.utils.data.TensorDataset(x_sample, z_actual_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.model.batch_size, shuffle=True
                )
                
                # Train for 1 epoch
                self.model.encoder.train()
                self.model.decoder.train()
                
                for batch_data, batch_target in dataloader:
                    optimizer.zero_grad()
                    reconstructed = self.model.forward(batch_data)
                    # Use missing-aware loss
                    mask = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), torch.ones_like(batch_target))
                    target_clean = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                    reconstructed_masked = reconstructed * mask
                    squared_diff = (target_clean - reconstructed_masked) ** 2
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
                    loss.backward()
                    # Gradient clipping to prevent NaN and improve stability
                    if self.model.grad_clip_val > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                            max_norm=self.model.grad_clip_val
                        )
                    optimizer.step()
                # Extract factors from this sample
                x_sample_tensor = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                self.model.encoder.eval()
                with torch.no_grad():
                    factors_sample = self.model.encoder(x_sample_tensor).cpu().numpy()
                    try:
                        factors_sample = check_finite_array(
                            factors_sample, 
                            f"factor sample {i+1}/{self.model.epochs_per_iter}", 
                            context=f"at iteration {iter_count}",
                            fallback=None
                        )
                    except ValueError:
                        factors_sample = np.nan_to_num(factors_sample, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
                factors_samples.append(factors_sample)
            
            # Update factors: average over all MC samples
            factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
            try:
                factors = check_finite_array(factors, "averaged factors", context=f"at iteration {iter_count}", fallback=factors_init)
            except ValueError:
                if factors_init is not None:
                    factors = factors_init.copy()
                else:
                    factors = np.nan_to_num(factors, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
            
            # Clip extreme factor values to prevent numerical instability
            clip_threshold = 10.0
            factor_mean = np.mean(factors, axis=0)
            factor_std = np.std(factors, axis=0)
            factor_std = np.maximum(factor_std, 1e-8)
            
            clipped_count = 0
            for i in range(factors.shape[1]):
                lower_bound = factor_mean[i] - clip_threshold * factor_std[i]
                upper_bound = factor_mean[i] + clip_threshold * factor_std[i]
                before_clip = factors[:, i].copy()
                factors[:, i] = np.clip(factors[:, i], lower_bound, upper_bound)
                clipped_count += np.sum((before_clip != factors[:, i]))
            
            if clipped_count > 0:
                _logger.warning(
                    f"{self.model.__class__.__name__} denoising iteration {iter_count}: clipped {clipped_count} extreme factor values (>{clip_threshold} std devs). "
                    f"This prevents numerical instability in encoder/decoder forward passes. "
                    f"If clipping occurs frequently, consider: (1) Reducing learning_rate, (2) Using smaller encoder_layers, (3) Checking data scaling"
                )
            
            # Check convergence
            self.model.decoder.eval()
            with torch.no_grad():
                factors_tensor = torch.tensor(factors, device=device, dtype=dtype)
                prediction_iter = self.model.decoder(factors_tensor).cpu().numpy()
                try:
                    prediction_iter = check_finite_array(
                        prediction_iter, 
                        "prediction_iter", 
                        context=f"at iteration {iter_count}",
                        fallback=prediction_prev_iter if prediction_prev_iter is not None else prediction_iter
                    )
                except ValueError:
                    if prediction_prev_iter is not None:
                        prediction_iter = prediction_prev_iter.copy()
                    else:
                        prediction_iter = np.nan_to_num(prediction_iter, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
            
            if iter_count > 1:
                # Compute MSE on non-missing values
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    mse = np.nanmean((prediction_prev_iter[mask] - prediction_iter[mask]) ** 2)
                    if not np.isfinite(mse):
                        _logger.warning(
                            f"{self.model.__class__.__name__} denoising iteration {iter_count}: MSE is not finite ({mse}). "
                            f"Using previous delta value"
                        )
                        mse = delta if np.isfinite(delta) else tolerance * 10
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
                    if not np.isfinite(loss_now):
                        _logger.warning(
                            f"{self.model.__class__.__name__} denoising iteration {iter_count}: initial loss is not finite ({loss_now}). "
                            f"Using large default value"
                        )
                        loss_now = 1e6
                else:
                    loss_now = float('inf')
            
            # Store previous prediction for convergence checking
            prediction_prev_iter = prediction_iter.copy()
            
            # Update missing values with current prediction
            if bool_miss.any():
                data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
            
            # Update residuals
            eps = data_mod_only_miss - prediction_iter
            try:
                eps = check_finite_array(eps, "residuals (eps)", context=f"at iteration {iter_count}", fallback=None)
            except ValueError:
                eps = np.nan_to_num(eps, nan=0.0, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
        
        if not_converged:
            delta_str = f"{delta:.6f}" if iter_count > 1 else "N/A"
            _logger.warning(
                f"{self.model.__class__.__name__} denoising training: convergence not achieved within {max_iter} iterations. "
                f"Final delta: {delta_str}"
            )
        
        converged = not not_converged
        
        # Validate and normalize factors shape before storing
        factors = self.model._validate_factors(factors, operation="fit_mcmc")
        
        # Import DDFMTrainingState here to avoid circular dependency
        from ..models.ddfm import DDFMTrainingState
        
        # Store final state
        self.model.training_state = DDFMTrainingState(
            factors=factors,
            prediction=prediction_iter,
            converged=converged,
            num_iter=iter_count,
            training_loss=loss_now
        )
        
        return self.model.training_state

