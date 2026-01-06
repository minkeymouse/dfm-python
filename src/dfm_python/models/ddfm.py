"""Deep Dynamic Factor Model (DDFM) using PyTorch."""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Tuple
import pandas as pd

from .base import BaseFactorModel
from ..logger import get_logger
from ..numeric.stability import convergence_checker
from ..numeric.estimator import get_idio, get_transition_params
from ..encoder.simple_autoencoder import SimpleAutoencoder
from ..config.schema.params import DDFMFitParams, DDFMTrainingState
from ..config.schema.results import DDFMResult
from ..config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_DDFM_OBSERVATION_NOISE,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPS,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_DTYPE,
    DEFAULT_DDFM_LEARNING_RATE,
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_DDFM_WINDOW_SIZE,
    DEFAULT_MAX_MCMC_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_DISP,
    DEFAULT_SEED,
    DEFAULT_MCMC_EPOCHS,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_INF_VALUE,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_MULT_EPOCH_PRETRAIN,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_LOSS_LOG_PRECISION,
    DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
)
from ..utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError
from ..utils.validation import check_condition
from ..numeric.validator import validate_horizon, validate_no_nan_inf, validate_update_data_shape
from ..numeric.estimator import forecast_ar1_factors
from ..utils.helper import interpolate_array
from ..config.types import to_tensor, to_numpy

# Import DDFMDataset for isinstance check
from ..dataset.ddfm_dataset import DDFMDataset

_logger = get_logger(__name__)


class DDFM(BaseFactorModel, nn.Module):
    """Deep Dynamic Factor Model (DDFM) using PyTorch.
    
    Implements the original DDFM algorithm with MCMC-based denoising training
    and sequential MC sample processing. Uses plain PyTorch (nn.Module) for
    better control over training loop compared to PyTorch Lightning.
    
    The model uses an autoencoder architecture to extract factors from
    multivariate time series data, with Bayesian inference via MCMC sampling.
    """
    
    def __init__(
        self,
        dataset: DDFMDataset,
        config: Optional[Any] = None,
        encoder_size: Optional[tuple] = None,
        decoder_type: str = "linear",
        seed: int = DEFAULT_SEED,
        activation: str = 'relu',
        learning_rate: float = DEFAULT_DDFM_LEARNING_RATE,
        optimizer: str = 'Adam',
        n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
        window_size: int = DEFAULT_DDFM_WINDOW_SIZE,
        max_epoch_pre_train: int = DEFAULT_PRETRAIN_EPOCHS,
        max_iter: int = DEFAULT_MAX_MCMC_ITER,
        tolerance: float = DEFAULT_TOLERANCE,
        disp: int = DEFAULT_DISP,
        min_target_interporate_ratio: Optional[float] = DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
    ):
        """Initialize DDFM model."""
        BaseFactorModel.__init__(self)
        nn.Module.__init__(self)
        
        # Validate dataset is DDFMDataset instance
        if not isinstance(dataset, DDFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DDFMDataset, got {type(dataset).__name__}"
            )
        
        # Store config and dataset
        self._config = config
        self._dataset = dataset
        
        # Use DEFAULT_ENCODER_LAYERS if encoder_size not provided
        if encoder_size is None:
            encoder_size = tuple(DEFAULT_ENCODER_LAYERS)
        self.encoder_size = encoder_size
        self.decoder_size = None
        
        # Validate decoder_type
        if decoder_type not in ("linear", "mlp"):
            raise ConfigurationError(f"decoder_type must be 'linear' or 'mlp', got '{decoder_type}'")
        self.decoder_type = decoder_type
        if decoder_type == "mlp":
            self.decoder_size = tuple(reversed(self.encoder_size[:-1])) if len(self.encoder_size) > 1 else None
            
        self.activation = activation
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_epoch_pre_train = max_epoch_pre_train
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        
        self.target_scaler = dataset.target_scaler
        self.min_target_interporate_ratio = min_target_interporate_ratio
        
        # Calculate dimensions using dataset shape properties
        # input_dim = X features + y targets (full_input concatenates them)
        # output_dim = only y targets (decoder only reconstructs targets, not features)
        self.num_series = self._dataset.data_shape[1]  # Total number of series
        self.input_dim = self._dataset.data_shape[1]  # full_input = X + y_corrupted
        self.output_dim = self._dataset.target_shape[1]  # Decoder outputs only targets

        self.initializer_seed = seed
        
        # Optimizer setup
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None  # Built in _build_optimizer()
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None  # Built in _build_optimizer()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_state = DDFMTrainingState()
    
    def _get_averaged_factors(self) -> np.ndarray:
        """Get factors averaged across MC samples if 3D, otherwise return as-is."""
        if self.factors.ndim == 3:
            return np.mean(self.factors, axis=0)  # Average across MC samples
        return self.factors
    
    
    @property
    def _has_factors(self) -> bool:
        """Check if factors attribute exists and is not None."""
        return getattr(self, 'factors', None) is not None
    
    
    def _update_imputed_and_eps(self, prediction_iter_full: np.ndarray) -> None:
        """Update data_imputed with predictions and compute eps (idiosyncratic residuals)."""
        # Step 1: Fill missing values with predictions (matching TensorFlow: data_mod_only_miss update)
        # TensorFlow: data_mod_only_miss.values[lags_input:][bool_miss] = prediction_iter[bool_miss]
        if self.missing_mask.any():
            self.data_imputed.values[self.missing_mask] = prediction_iter_full[self.missing_mask]
        
        # Step 2: Compute eps (residuals) = data_imputed - prediction (matching TensorFlow)
        # TensorFlow: eps = data_mod_only_miss.values[lags_input:] - prediction_iter
        # For exchange rate data (no missing), data_mod_only_miss = original data
        # But TensorFlow uses data_mod_only_miss (which may have predictions for missing values)
        # We use data_imputed to match TensorFlow's data_mod_only_miss behavior
        # Note: For data with no missing values, data_imputed = original data, so this is equivalent
        # But for data with missing values, this ensures we use the imputed values (matching TensorFlow)
        eps_full = self.data_imputed.values - prediction_iter_full
        
        # Extract only target series residuals (eps is only defined for target series)
        # TensorFlow indexes with [lags_input:], but for lags_input=0 this is the same
        self.eps = eps_full[:, self.target_indices]
    
    def _update_previous_predictions(
        self, 
        prediction_iter_target: np.ndarray, 
        prediction_iter_full: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Update previous prediction state for convergence checking.
        
        Consolidates duplicate pattern of copying predictions for next iteration.
        
        Parameters
        ----------
        prediction_iter_target : np.ndarray
            Target prediction for current iteration
        prediction_iter_full : np.ndarray
            Full prediction for current iteration
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (prediction_prev_iter, prediction_prev_iter_full)
        """
        return (
            prediction_iter_target.copy(),
            prediction_iter_full.copy() if self._dataset.all_columns_are_targets else None
        )
    
    def _register_state_space_buffers(self, buffers: dict) -> None:
        """Register state-space parameter buffers.
        
        Consolidates repetitive register_buffer() calls into a single helper method.
        
        Parameters
        ----------
        buffers : dict
            Dictionary mapping buffer names to numpy arrays
        """
        for name, arr in buffers.items():
            self.register_buffer(name, to_tensor(arr, dtype=DEFAULT_TORCH_DTYPE))
    
    def _build_optimizer(self) -> None:
        """Build optimizer and scheduler for training."""
        optimizers = {
            'Adam': lambda: torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=self.learning_rate,
                betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
                eps=DEFAULT_ADAM_EPS
            ),
            'AdamW': lambda: torch.optim.AdamW(
                self.autoencoder.parameters(),
                lr=self.learning_rate,
                betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
                eps=DEFAULT_ADAM_EPS
            ),
            'SGD': lambda: torch.optim.SGD(self.autoencoder.parameters(), lr=self.learning_rate)
        }
        self.optimizer = optimizers.get(self.optimizer_type, optimizers['SGD'])()
        # CRITICAL: TensorFlow's ExponentialDecay with decay_steps=epochs (n_mc_samples=10) and staircase=True
        # decays every 10 optimizer steps (batches), not every 10 epochs
        # This is a significant difference from StepLR which decays every N epochs
        # For now, disable scheduler to test if this is causing the prediction shrinking issue
        # TODO: Implement proper per-step decay to match TensorFlow's behavior
        self.scheduler = None
    
    def _interpolate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate DataFrame values in-place and return."""
        # Copy needed: interpolation modifies values in-place, preserve original
        df_interpolated = df.copy()
        df_interpolated.values[:] = interpolate_array(df_interpolated.values)
        return df_interpolated
    
    def _select_convergence_predictions(
        self,
        prediction_prev_full: Optional[np.ndarray],
        prediction_prev_target: np.ndarray,
        prediction_iter_full: np.ndarray,
        prediction_iter_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select appropriate predictions for convergence checking.
        
        When all columns are targets, uses full predictions (all columns).
        When only some columns are targets, uses target predictions only.
        This ensures consistency between training target and convergence checking target.
        
        Parameters
        ----------
        prediction_prev_full : Optional[np.ndarray]
            Previous full prediction (all columns) or None if not all columns are targets
        prediction_prev_target : np.ndarray
            Previous target prediction (target columns only)
        prediction_iter_full : np.ndarray
            Current full prediction (all columns)
        prediction_iter_target : np.ndarray
            Current target prediction (target columns only)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (prediction_prev, prediction_iter) selected based on _all_columns_are_targets flag
        """
        if self._dataset.all_columns_are_targets:
            return prediction_prev_full, prediction_iter_full
        else:
            return prediction_prev_target, prediction_iter_target
    
    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize MCMC state: interpolate data, make initial prediction, compute initial eps."""
        self.data_denoised_interpolated = self._interpolate_dataframe(self.data_denoised)
        self.data_imputed = self.data_denoised_interpolated.copy()
        
        # Initial prediction: use ORIGINAL data (matching TensorFlow: data_tmp from data_mod)
        # TensorFlow: build_inputs() creates data_tmp from data_mod, then uses data_tmp.values
        # For lags_input=0 and no missing values, data_tmp = data_mod = original scaled data
        # We use self.data.values directly (matching TensorFlow's data_tmp.values)
        # 
        # CRITICAL: TensorFlow's autoencoder.predict() uses training mode for BatchNorm
        # We must match this behavior to get identical initial predictions
        data_original_tensor = to_tensor(self.data.values, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        self.autoencoder.train()  # Set to training mode (matching TensorFlow's predict())
        with torch.no_grad():
            prediction_iter_full_tensor = self.autoencoder.forward(data_original_tensor)
        prediction_iter_target_tensor = prediction_iter_full_tensor if self._dataset.all_columns_are_targets else prediction_iter_full_tensor[:, self._target_col_tensor]
        prediction_iter_full = to_numpy(prediction_iter_full_tensor)
        prediction_iter_target = to_numpy(prediction_iter_target_tensor)
        
        # Update imputed data/eps (eps computed as original_data - prediction, matching TensorFlow)
        self._update_imputed_and_eps(prediction_iter_full)
        
        # Initialize previous prediction for convergence checking
        prediction_prev_iter, prediction_prev_iter_full = self._update_previous_predictions(
            prediction_iter_target, prediction_iter_full
        )
        
        return prediction_prev_iter, prediction_prev_iter_full
                
    def fit(self) -> None:
        """Fit DDFM: builds model, pre-trains, and trains in one method."""
        # Track training time
        start_time = time.time()
        
        # Build autoencoder
        self.autoencoder = SimpleAutoencoder.build(
            input_dim=self.input_dim,
            encoder_size=self.encoder_size,
            decoder_size=self.decoder_size,
            decoder_type=self.decoder_type,
            output_dim=self.output_dim,
            activation=self.activation,
            seed=self.initializer_seed
        )
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.autoencoder.to(self.device)
        
        # Compile autoencoder for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.autoencoder = torch.compile(self.autoencoder, mode='reduce-overhead')
                _logger.debug("Autoencoder compiled with torch.compile for optimization")
            except Exception as e:
                _logger.warning(f"torch.compile failed, continuing without compilation: {e}")
        
        self._build_optimizer()
        
        # Pre-train: interpolate if target_nan_ratio is high, otherwise keep NaN
        if self._dataset.target_nan_ratio > self.min_target_interporate_ratio:
            data_pre_train = self._interpolate_dataframe(self._dataset.data)
        else:
            data_pre_train = self._dataset.data
        
        # Create pre-training dataset (handles X/y splitting and tensor conversion)
        pre_train_dataset = self._dataset.create_pretrain_dataset(data_pre_train, device=self.device)
        
        self.autoencoder.fit(
            dataset=pre_train_dataset,
            epochs=self.max_epoch_pre_train,
            batch_size=self.window_size,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            optimizer=self.optimizer,
            scheduler=None,
            target_indices=None
        )
        
        # CRITICAL: Do NOT rebuild optimizer after pre-training (matching TensorFlow)
        # TensorFlow re-compiles with the SAME optimizer instance, preserving state
        # (momentum, second-moment estimates, etc.)
        # Rebuilding would reset optimizer state, causing different learning dynamics
        # The optimizer state from pre-training should persist into MCMC training
        
        # Initialize data structures needed for MCMC
        # Copy needed: self.data is modified during MCMC (denoising, imputation), preserve original dataset
        self.data = self._dataset.data.copy()
        # Copy needed: self.data_denoised is modified during denoising iterations, preserve baseline
        self.data_denoised = self.data.copy()
        self.missing_mask = self.data.isna().values
        self.target_indices = self._dataset.target_indices
        if not self._dataset.all_columns_are_targets:
            self._target_col_tensor = torch.tensor(self.target_indices, device=self.device, dtype=torch.long)
        self.rng = np.random.RandomState(self.initializer_seed)
        
        # Initialize MCMC state (interpolation, initial prediction, eps computation)
        prediction_prev_iter, prediction_prev_iter_full = self._initialize_mcmc_state()
        
        # Set y_actual to ORIGINAL data (matching TensorFlow: z_actual = self.data[lags_input:].values)
        # This is the ground truth target for reconstruction, NOT interpolated data
        # TensorFlow uses original scaled data starting from lags_input
        # We use self.data (which is a copy of _dataset.data) to match TensorFlow's self.data
        # Note: self.data is set above as self._dataset.data.copy(), so it's the scaled original data
        # For DDFM, lags_input is typically 0, so this is equivalent to using all data
        lags_input = getattr(self, 'lags_input', 0)  # Default to 0 if not set (for backward compatibility)
        if self._dataset.all_columns_are_targets:
            # All columns are targets: use original data starting from lags_input (matching TensorFlow)
            self.y_actual = self.data.values[lags_input:]
        else:
            # Only some columns are targets: use target columns from original data starting from lags_input
            # For non-target case, we need to extract target columns from self.data
            self.y_actual = self.data.values[lags_input:, self.target_indices]
        
        converged = False
        self._num_iter = 0  # Store actual iteration count for get_result()
        
        # MCMC loop
        while not converged and self._num_iter < self.max_iter:
            # Get idio distr (Phi is AR(1) coefficient matrix)
            # Use dataset's observed_y directly (target columns only)
            Phi, mu_eps, std_eps = get_idio(self.eps, self._dataset.observed_y)
            
            # Apply denoising step: subtract conditional AR-idio mean from data_imputed
            # TensorFlow: data_mod[lags_input + 1:] = data_mod_only_miss[lags_input + 1:] - eps[:-1, :] @ phi
            # For lags_input=0, this becomes: data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
            # We use data_imputed (which matches data_mod_only_miss) for denoising
            eps_expanded = np.zeros((self.eps.shape[0], self.num_series))
            eps_expanded[:, self.target_indices] = self.eps
            self.data_denoised.values[1:] = self.data_imputed.values[1:] - eps_expanded[:-1, :] @ Phi
            # TensorFlow also sets: data_mod[:lags_input + 1] = data_mod_only_miss[:lags_input + 1]
            # For lags_input=0, this sets data_mod[0] = data_mod_only_miss[0]
            # We initialize data_denoised as a copy of data, so first row is already correct
            
            # Update interpolated data (needed for creating datasets)
            self.data_denoised_interpolated = self._interpolate_dataframe(self.data_denoised)
            
            # Pre-generate MC samples
            # Use denoised data (not interpolated) for corruption, matching TensorFlow's data_tmp
            # TensorFlow: data_tmp is built from data_mod (denoised) via build_inputs()
            # build_inputs() interpolates only if there are missing values
            # For exchange rate data (no missing), data_tmp = denoised data
            # So we use data_denoised (not interpolated) for corruption
            X_features_df, y_tmp = self._dataset.split_features_and_targets(self.data_denoised)
            X_features = pd.DataFrame() if X_features_df is None else X_features_df
            
            autoencoder_datasets = self._dataset.create_autoencoder_datasets_list(
                n_mc_samples=self.n_mc_samples,
                mu_eps=mu_eps,
                std_eps=std_eps,
                X=X_features,
                y_tmp=y_tmp,
                y_actual=self.y_actual,
                rng=self.rng,
                device=self.device
            )
            
            # Train on each MC sample sequentially
            # CRITICAL: TensorFlow uses ExponentialDecay with decay_steps=epochs (10) and staircase=True
            # This means learning rate decays every 10 optimizer steps (batches), not every 10 epochs
            # Our StepLR with step_size=n_mc_samples (10) decays every 10 epochs, which is different
            # However, TensorFlow's decay_steps=epochs refers to the number of MC samples (epochs parameter)
            # So we need to step the scheduler after each MC sample to match TensorFlow's behavior
            # Actually, TensorFlow's ExponentialDecay is applied per optimizer step, so it decays
            # every 10 batches across all MC samples. We need to match this behavior.
            self.autoencoder.train()
            target_indices = self._target_col_tensor if not self._dataset.all_columns_are_targets else None
            for ae_dataset in autoencoder_datasets:
                self.autoencoder.fit(
                    dataset=ae_dataset,
                    epochs=DEFAULT_MCMC_EPOCHS,
                    batch_size=self.window_size,
                    learning_rate=self.learning_rate,
                    optimizer_type=self.optimizer_type,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,  # Pass scheduler to use DDFM's learning rate decay
                    target_indices=target_indices
                )
            
            # Step scheduler after all MC samples (matching TensorFlow's decay_steps=epochs behavior)
            # TensorFlow's ExponentialDecay with decay_steps=epochs (10) means it decays every 10 optimizer steps
            # Since we have n_mc_samples (10) MC samples per iteration, we step after all MC samples
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Extract factors and compute predictions
            # CRITICAL: TensorFlow's encoder is a separate model that shares layers with autoencoder
            # When autoencoder.fit() is called, all layers (including encoder's) are in training mode
            # When encoder() is called directly, it uses the same layers, so still in training mode
            # This means BatchNorm uses batch statistics, not running statistics
            # 
            # We must keep training mode to match TensorFlow, even though it's slower
            # The slowness comes from BatchNorm computing batch mean/var on-the-fly
            # This is necessary for correctness - running stats would give different results
            # 
            # Note: We use torch.no_grad() to prevent gradient computation, but keep training mode
            # This ensures BatchNorm uses batch statistics (matching TensorFlow)
            with torch.no_grad():
                factors_list = [self.encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
                factors_tensor = torch.stack(factors_list, dim=0)
                
                predictions_full_tensor = torch.stack([self.decoder(f) for f in factors_list], dim=0)
                prediction_iter_full_tensor = predictions_full_tensor.mean(dim=0)
                prediction_iter_target_tensor = prediction_iter_full_tensor if self._dataset.all_columns_are_targets else prediction_iter_full_tensor[:, self._target_col_tensor]
                
                prediction_iter_full = to_numpy(prediction_iter_full_tensor)
                prediction_iter_target = to_numpy(prediction_iter_target_tensor)
                self.factors = to_numpy(factors_tensor)
            
            # Update imputed data/eps
            self._update_imputed_and_eps(prediction_iter_full)
            
            # Check convergence
            prediction_prev, prediction_iter = self._select_convergence_predictions(
                prediction_prev_iter_full, prediction_prev_iter,
                prediction_iter_full, prediction_iter_target
            )
            delta, self.loss_now = convergence_checker(prediction_prev, prediction_iter, self.y_actual)
            
            if self._num_iter % self.disp == 0:
                _logger.info(f'iteration: {self._num_iter} - loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f} - delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f}')
            
            if delta < self.tolerance:
                converged = True
                self._converged = True
                _logger.info(f'Convergence achieved in {self._num_iter} iterations')
            
            # Update previous prediction for next iteration
            prediction_prev_iter, prediction_prev_iter_full = self._update_previous_predictions(
                prediction_iter_target, prediction_iter_full
            )
            
            # Store last iteration's datasets for last_neurons extraction (if MLP decoder)
            if self.decoder_type == "mlp":
                self._last_iter_datasets = autoencoder_datasets
            
            self._num_iter += 1  # Increment iteration count (used for convergence check and get_result())
        
        # Extract last neurons (for MLP decoder: second-to-last layer output)
        if self.decoder_type == "linear":
            self.last_neurons = self.factors
        else:
            decoder_intermediate = nn.Sequential(*list(self.decoder.children())[:-1])
            self.autoencoder.eval()
            with torch.no_grad():
                last_neurons_list = [
                    decoder_intermediate(self.encoder(ae_dataset.full_input))
                    for ae_dataset in self._last_iter_datasets
                ]
            self.last_neurons = np.array([to_numpy(ln) for ln in last_neurons_list])
        
        # Store training time
        self._training_time = time.time() - start_time
    
    def build_state_space(self) -> None:
        """Build state-space model from trained autoencoder."""
        f_t = self._get_averaged_factors()
        eps_t = self.eps
        num_factors = f_t.shape[1]
        
        # Extract decoder's last linear layer weights
        linear_layers = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ConfigurationError("No Linear layer found in decoder")
        linear_layer = linear_layers[-1]
        
        # Get weight matrix: shape (N, m) where N=num_series, m=num_factors
        weight = to_numpy(linear_layer.weight.data)
        
        # Extract observation matrix H (first m columns of weight matrix)
        H = weight[:, :num_factors]
        
        # Get transition equation params (factor_order is fixed to 1)
        # F_full includes both factors and idiosyncratic components: shape (m + N, m + N)
        # Use dataset's observed_y directly (target columns only)
        F_full, Q_full, mu_0_full, Sigma_0_full, _ = get_transition_params(f_t, eps_t, bool_no_miss=self._dataset.observed_y)
        
        # Extract factor-only transition matrix F (top-left block of F_full)
        # F_full structure: [[A_f, 0], [0, Phi]] where A_f is (m x m) factor transition
        F = F_full[:num_factors, :num_factors]  # Factor transition matrix (m x m)
        Q = Q_full[:num_factors, :num_factors]  # Factor process noise (m x m)
        mu_0 = mu_0_full[:num_factors]  # Initial factor mean (m,)
        Sigma_0 = Sigma_0_full[:num_factors, :num_factors]  # Initial factor covariance (m x m)
        
        R = np.eye(eps_t.shape[1]) * DEFAULT_DDFM_OBSERVATION_NOISE
        
        # Register state-space parameters as buffers (for checkpoint saving/loading)
        self._register_state_space_buffers({
            '_state_space_F': F,
            '_state_space_Q': Q,
            '_state_space_mu_0': mu_0,
            '_state_space_Sigma_0': Sigma_0,
            '_state_space_H': H,
            '_state_space_R': R
        })
        
        # Store in DDFMFitParams dataclass for convenient numpy access
        self.state_space_params = DDFMFitParams(
            F=F,
            Q=Q,
            mu_0=mu_0,
            Sigma_0=Sigma_0,
            H=H,
            R=R
        )
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dictionary and restore state_space_params dataclass."""
        result = super().load_state_dict(state_dict, strict=strict)
        # Restore state_space_params from buffers if they exist
        if getattr(self, '_state_space_F', None) is not None:
            # Convert state space buffers to numpy using to_numpy utility for consistency
            self.state_space_params = DDFMFitParams(
                F=to_numpy(self._state_space_F),
                Q=to_numpy(self._state_space_Q),
                mu_0=to_numpy(self._state_space_mu_0),
                Sigma_0=to_numpy(self._state_space_Sigma_0),
                H=to_numpy(self._state_space_H),
                R=to_numpy(self._state_space_R)
            )
        else:
            self.state_space_params = None
        return result
    
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values using trained state-space model. Requires build_state_space() to be called."""
        
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        # Validate state-space model is built
        check_condition(
            self.state_space_params is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} prediction failed: state-space model has not been built",
            details="Please call build_state_space() after training to enable prediction"
        )
        
        # Validate horizon
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        horizon = validate_horizon(horizon)
        
        # Get state-space parameters
        params = self.state_space_params
        F = params.F  # Transition matrix (m x m)
        H = params.H  # Observation matrix (N x m) where N is num_target_series
        
        # Get last factor state from training (average across MC samples)
        factors_avg = self._get_averaged_factors()
        
        # Get last factor state: (num_factors,)
        Z_last = factors_avg[-1, :] if factors_avg.shape[0] > 0 else params.mu_0
        
        # Validate factor state
        validate_no_nan_inf(Z_last, name="last factor state Z_last")
        validate_no_nan_inf(F, name="transition matrix F")
        validate_no_nan_inf(H, name="observation matrix H")
        
        # Forecast factors forward using AR(1) dynamics
        Z_forecast = forecast_ar1_factors(Z_last, F, horizon, dtype=DEFAULT_DTYPE)
        
        # Transform factors to observations (target series only)
        # H shape: (num_target_series, num_factors)
        # Z_forecast shape: (horizon, num_factors)
        # y_forecast shape: (horizon, num_target_series)
        # Note: y_forecast_std is in SCALED space (same scale as training data)
        y_forecast_std = Z_forecast @ H.T
        
        # Inverse transform target series to original scale (only during forecasting)
        # Training and evaluation use scaled data; only forecasts are unscaled
        target_scaler = getattr(self._dataset, 'target_scaler', None)
        
        if target_scaler is None:
            raise ConfigurationError(
                f"{self.__class__.__name__} forecast failed: target_scaler is None",
                details="Dataset must provide target_scaler for proper forecast scaling"
            )
        
        # Scaler is already fitted in dataset.__init__, just apply inverse transform
        y_forecast = target_scaler.inverse_transform(y_forecast_std)
        
        # Ensure numpy array and validate
        y_forecast = np.asarray(y_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(y_forecast, name="forecast y_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        
        # Return based on flags
        if return_series and return_factors:
            return y_forecast, Z_forecast
        if return_series:
            return y_forecast
        return Z_forecast
    
    def get_result(self):
        """Extract DDFMResult from trained model."""
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} get_result failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        # Validate state-space model is built
        check_condition(
            self.state_space_params is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} get_result failed: state-space model has not been built",
            details="Please call build_state_space() after training"
        )
        
        # Get state-space parameters
        params = self.state_space_params
        F = params.F  # Transition matrix
        H = params.H  # Observation matrix
        Q = params.Q  # Process noise covariance
        R = params.R  # Observation noise covariance
        mu_0 = params.mu_0  # Initial state mean
        Sigma_0 = params.Sigma_0  # Initial state covariance
        
        # Get factors (average across MC samples if 3D)
        Z = self._get_averaged_factors()
        
        # Compute smoothed data: x_sm = Z @ H.T
        # Note: x_sm is in SCALED space (same scale as training data)
        # Use get_x_sm_original_scale() on result to get unscaled values if needed
        x_sm = Z @ H.T  # (T, num_target_series)
        
        # Expand to full data shape for compatibility
        x_sm_full = np.zeros((x_sm.shape[0], self.num_series))
        x_sm_full[:, self.target_indices] = x_sm
        
        # Get target scaler from dataset
        target_scaler = getattr(self._dataset, 'target_scaler', None)
        
        # Create result
        # For DDFM: A = F (transition), C = H (observation), r = [num_factors], p = 1
        f_avg = self._get_averaged_factors()
        num_factors = f_avg.shape[1]
        return DDFMResult(
            x_sm=x_sm_full,
            Z=Z,
            C=H.T,  # Transpose H to get (num_target_series x num_factors) loading matrix
            R=R,
            A=F,
            Q=Q,
            Z_0=mu_0,
            V_0=Sigma_0,
            r=np.array([num_factors]),  # Single block with num_factors
            p=DEFAULT_FACTOR_ORDER,  # AR(1) dynamics
            target_scaler=target_scaler,
            converged=getattr(self, '_converged', False),
            num_iter=getattr(self, '_num_iter', self.max_iter),  # Use actual iteration count if available
            loglik=-DEFAULT_INF_VALUE  # DDFM doesn't compute log-likelihood
        )
    
    def update(self, dataset: DDFMDataset) -> None:
        """Update model factors with new data using neural network forward pass."""
        from ..utils.errors import DataValidationError
        
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} update failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        # Validate autoencoder is built
        check_condition(
            self.autoencoder is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} update failed: model has not been built",
            details="Please call build_model() first"
        )
        
        # Validate dataset has same number of features
        # DDFMDataset has self.data (DataFrame), not data_processed attribute
        new_data = np.asarray(dataset.data.values)
        training_data = np.asarray(self.data.values)
        validate_update_data_shape(
            data=new_data,
            training_data=training_data,
            model_name=self.__class__.__name__
        )
        
        # Validate target_series match
        if dataset.target_series != self._dataset.target_series:
            raise DataValidationError(
                f"target_series mismatch: new dataset has {dataset.target_series}, "
                f"but training dataset has {self._dataset.target_series}"
            )
        
        # Convert new data to tensor and move to GPU
        new_data_tensor = to_tensor(new_data, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        # Extract factors from new data using encoder
        self.autoencoder.eval()
        with torch.no_grad():
            new_factors = self.encoder(new_data_tensor)  # (T_new, num_factors)
        
        # Convert to numpy
        new_factors_np = to_numpy(new_factors)
        
        # Update factors: append new factors to existing ones
        # self.factors shape: (n_mc_samples, T, num_factors) or (T, num_factors)
        if self.factors.ndim == 3:
            # Expand new_factors to match MC dimension: (1, T_new, num_factors)
            new_factors_expanded = np.expand_dims(new_factors_np, axis=0)
            # Concatenate: (n_mc_samples, T + T_new, num_factors)
            self.factors = np.concatenate([self.factors, new_factors_expanded], axis=1)
        else:
            # Concatenate: (T + T_new, num_factors)
            self.factors = np.vstack([self.factors, new_factors_np])