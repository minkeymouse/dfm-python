"""Deep Dynamic Factor Model (DDFM) using PyTorch.

Implements the original DDFM algorithm with MCMC-based denoising training
and sequential MC sample processing.
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Tuple
import pandas as pd

from .base import BaseFactorModel
from ..logger import get_logger
from ..numeric.stability import convergence_checker
from ..numeric.estimator import get_idio, get_transition_params
from ..numeric.statistic import (
    compute_variance_mean, compute_array_stats, compute_tensor_stats,
    average_3d_array, extract_batchnorm_statistics, diagnose_variance_collapse
)
from ..encoder.simple_autoencoder import SimpleAutoencoder
from ..config.schema.params import DDFMStateSpaceParams, DDFMTrainingState
from ..config.schema.results import DDFMResult
from sklearn.preprocessing import StandardScaler
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
    DEFAULT_VARIANCE_COLLAPSE_THRESHOLD,
    DEFAULT_FACTOR_COLLAPSE_THRESHOLD,
    DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX,
    DEFAULT_SCALE_RATIO_MIN,
    DEFAULT_STANDARDIZATION_MEAN_THRESHOLD,
    DEFAULT_STANDARDIZATION_STD_MIN,
    DEFAULT_STANDARDIZATION_STD_MAX,
    DEFAULT_STANDARDIZED_TARGET_STD,
    DEFAULT_TARGET_PREDICTION_STD,
    DEFAULT_VARIANCE_COLLAPSE_STD,
    DEFAULT_TARGET_CONVERGENCE_ITERATIONS,
    DEFAULT_TARGET_DDFM_LOSS,
    DEFAULT_DDFM_LOSS_MULTIPLIER,
)
from ..utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError
from ..utils.validation import check_condition
from ..numeric.validator import validate_horizon, validate_no_nan_inf, validate_update_data_shape
from ..numeric.estimator import forecast_ar1_factors
from ..utils.helper import interpolate_array, interpolate_dataframe
from ..config.types import to_tensor, to_numpy

from ..dataset.ddfm_dataset import DDFMDataset

_logger = get_logger(__name__)


class DDFM(BaseFactorModel, nn.Module):
    """Deep Dynamic Factor Model using PyTorch."""
    
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
        
        if not isinstance(dataset, DDFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DDFMDataset, got {type(dataset).__name__}"
            )
        
        self._config = config
        self._dataset = dataset
        
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
        self.state_space_params = None
        
        self.lags_input = 0
    
    def _get_averaged_factors(self) -> np.ndarray:
        """Get factors averaged across MC samples if 3D, otherwise return as-is."""
        return average_3d_array(self.factors, axis=0)
    
    
    
    
    
    @property
    def _has_factors(self) -> bool:
        """Check if factors attribute exists and is not None."""
        return getattr(self, 'factors', None) is not None
    
    def _update_imputed_and_eps(self, y_pred_full: np.ndarray) -> None:
        """Update data_imputed with predictions and compute eps (idiosyncratic residuals).
        
        Only target series (y) are imputed - features (X) are not imputed since they're only
        used for encoder input. Uses dataset's missing_y to identify missing target values.
        """
        # Only impute target series: use dataset's missing_y for target columns
        missing_y = self._dataset.missing_y
        if missing_y.any():
            # When all columns are targets, y_pred_full has full shape
            # When there are features, y_pred_full only has target columns
            if self._dataset.all_columns_are_targets:
                missing_mask_full = missing_y
                self.data_imputed.values[missing_mask_full] = y_pred_full[missing_mask_full]
            else:
                # Only update target columns: y_pred_full has shape (T, num_target_series)
                # missing_y has shape (T, num_target_series)
                self.data_imputed.values[:, self.target_indices][missing_y] = y_pred_full[missing_y]
        
        # Compute eps: y_actual - y_pred
        # When all columns are targets, y_pred_full has full shape
        # When there are features, y_pred_full only has target columns
        if self._dataset.all_columns_are_targets:
            eps_full = self.data_imputed.values - y_pred_full
            self.eps = eps_full[:, self.target_indices]
        else:
            # y_pred_full only has target columns, extract target columns from data_imputed
            self.eps = self.data_imputed.values[:, self.target_indices] - y_pred_full
    
    def _update_previous_predictions(
        self, 
        y_pred: np.ndarray, 
        y_pred_full: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Update previous prediction state for convergence checking.
        
        Consolidates duplicate pattern of copying predictions for next iteration.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Target prediction for current iteration
        y_pred_full : np.ndarray
            Full prediction for current iteration
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (y_pred_prev, y_pred_prev_full)
        """
        return (
            y_pred.copy(),
            y_pred_full.copy() if self._dataset.all_columns_are_targets else None
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
        """Build optimizer and scheduler for training.
        
        Creates optimizer (Adam/AdamW/SGD) and learning rate scheduler (LambdaLR).
        
        **Learning Rate Decay Implementation:**
        TensorFlow's ExponentialDecay with decay_steps=n_mc_samples (DEFAULT_N_MC_SAMPLES) and staircase=True
        decays every n_mc_samples optimizer steps (batches), not every n_mc_samples epochs.
        
        **Implementation (Fixed 2026-01-07):**
        - Scheduler steps after each batch in autoencoder.fit() (simple_autoencoder.py:269)
        - LambdaLR scheduler uses step count (number of batches) to compute decay
        - Decays every n_mc_samples scheduler steps (batches) → matches TensorFlow behavior
        - Learning rate multiplier: decay_rate ^ (step // n_mc_samples)
        - Mathematical verification: Matches TensorFlow's ExponentialDecay(decay_steps=n_mc_samples, decay_rate=0.96, staircase=True)
        """
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
        
        def lr_lambda(step: int) -> float:
            """Compute learning rate multiplier for per-batch decay (matches TensorFlow behavior).
            
            TensorFlow: ExponentialDecay(decay_steps=n_mc_samples, decay_rate=DEFAULT_LR_DECAY_RATE, staircase=True)
            - Decays every n_mc_samples optimizer steps (batches)
            
            Our implementation (fixed 2026-01-07):
            - Scheduler steps after each batch in autoencoder.fit() (simple_autoencoder.py:269)
            - step parameter is scheduler step count (number of batches completed)
            - Decays every n_mc_samples scheduler steps (batches) → matches TensorFlow behavior
            - Mathematical equivalence: DEFAULT_LR_DECAY_RATE ^ (step // n_mc_samples) matches TensorFlow's staircase=True behavior
            
            Returns:
                Learning rate multiplier: DEFAULT_LR_DECAY_RATE ^ (step // n_mc_samples)
            """
            # Decay every n_mc_samples scheduler steps (batches)
            # Scheduler steps after each batch, so step count equals batch count
            decay_steps = step // self.n_mc_samples
            return DEFAULT_LR_DECAY_RATE ** decay_steps
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
    
    def _build_inputs_for_pretrain(self, interpolate: bool = True) -> pd.DataFrame:
        """Build inputs for pre-training.
        
        For lags_input=0, returns the data (no lagged features).
        For lags_input > 0, creates lagged features.
        
        Parameters
        ----------
        interpolate : bool
            Whether to interpolate missing values using spline interpolation
            
        Returns
        -------
        pd.DataFrame
            Full input data (X + y) with lagged features (if lags_input > 0) and optionally interpolated
        """
        if self.lags_input == 0:
            full_input_data = self._dataset.data.copy()
        else:
            # Create lagged features
            new_dict = {}
            for col_name in self._dataset.data.columns:
                new_dict[col_name] = self._dataset.data[col_name]
                for lag in range(self.lags_input):
                    new_dict[f'{col_name}_lag{lag + 1}'] = self._dataset.data[col_name].shift(lag + 1)
            full_input_data = pd.DataFrame(new_dict, index=self._dataset.data.index)
            # Drop initial nans from lagging
            full_input_data = full_input_data[self.lags_input:]
        
        if interpolate and full_input_data.isna().sum().sum() > 0:
            full_input_data = full_input_data.interpolate(method='spline', limit_direction='both', order=3)
        
        return full_input_data
    
    
    def _select_convergence_predictions(
        self,
        y_pred_prev_full: Optional[np.ndarray],
        y_pred_prev: np.ndarray,
        y_pred_full: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select appropriate predictions for convergence checking."""
        if self._dataset.all_columns_are_targets:
            return y_pred_prev_full, y_pred_full
        else:
            return y_pred_prev, y_pred
    
    def _extract_target_predictions(self, y_pred_full_tensor: torch.Tensor) -> torch.Tensor:
        """Extract target predictions from full prediction tensor."""
        if self._dataset.all_columns_are_targets:
            return y_pred_full_tensor
        else:
            # When covariates are present, autoencoder output_dim is num_target_series,
            # so y_pred_full_tensor already contains only target predictions
            # Check if the prediction shape matches target shape
            if y_pred_full_tensor.shape[-1] == len(self._dataset.target_series):
                return y_pred_full_tensor
            # Otherwise, extract target columns (legacy case)
            if y_pred_full_tensor.dim() == 1:
                return y_pred_full_tensor[self._target_col_tensor]
            elif y_pred_full_tensor.dim() == 2:
                return y_pred_full_tensor.index_select(1, self._target_col_tensor)
            else:
                return y_pred_full_tensor.index_select(-1, self._target_col_tensor)
    
    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize MCMC state: interpolate data, make initial prediction, compute initial eps."""
        self.data_denoised_interpolated = interpolate_dataframe(self.data_denoised)
        self.data_imputed = self.data_denoised_interpolated.copy()
        
        # Match TensorFlow: build_inputs() then predict on full input data
        # For lags_input=0, _build_inputs_for_pretrain returns self.data (no lags)
        # Use interpolate=True to match TensorFlow's build_inputs() behavior
        full_input_data = self._build_inputs_for_pretrain(interpolate=True)
        full_input = to_numpy(full_input_data)
        full_input_tensor = to_tensor(full_input, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_pred_full_tensor = self.autoencoder.predict(full_input_tensor)
        y_pred_tensor = y_pred_full_tensor
        y_pred_full = to_numpy(y_pred_full_tensor)
        y_pred = to_numpy(y_pred_tensor)
        
        # Update imputed data with predictions (fill missing values)
        # Only target series (y) are imputed - features (X) are not imputed
        missing_y = self._dataset.missing_y
        if missing_y.any():
            # When all columns are targets, y_pred_full has full shape
            # When there are features, y_pred_full only has target columns
            if self._dataset.all_columns_are_targets:
                missing_mask_full = missing_y
                self.data_imputed.values[missing_mask_full] = y_pred_full[missing_mask_full]
            else:
                # Only update target columns: y_pred_full has shape (T, num_target_series)
                # missing_y has shape (T, num_target_series)
                self.data_imputed.values[:, self.target_indices][missing_y] = y_pred_full[missing_y]
        
        # Compute eps: Match TensorFlow's self.eps = self.data_tmp[self.data.columns].values - prediction_iter
        # For lags_input=0, full_input_data is self.data, so extract original columns directly
        # For all-targets case, y_pred_full is already full shape
        # eps = y_actual_full - y_pred_full (idiosyncratic residuals)
        if self.lags_input == 0:
            y_actual_full = to_numpy(self.data)
        else:
            # If lags_input > 0, extract only original columns (matching TensorFlow's self.data.columns)
            y_actual_full = full_input_data[self._dataset.data.columns].values
        eps_full = y_actual_full - y_pred_full
        self.eps = eps_full[:, self.target_indices]
        y_pred_prev, y_pred_prev_full = self._update_previous_predictions(
            y_pred, y_pred_full
        )
        
        return y_pred_prev, y_pred_prev_full
                
    def fit(self) -> None:
        """Fit DDFM: builds model, pre-trains, and trains in one method."""
        start_time = time.time()
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
        
        
        self._build_optimizer()
        
        # Pre-train autoencoder on clean data
        min_obs = 50
        mult_epoch_pre = 1
        pretrain_epochs = self.n_mc_samples * mult_epoch_pre
        
        data_pre_train = self._build_inputs_for_pretrain(interpolate=False)
        data_pre_train_dropped = data_pre_train.dropna()
        use_mse_loss = len(data_pre_train_dropped) >= min_obs
        
        if not use_mse_loss:
            data_pre_train = self._build_inputs_for_pretrain(interpolate=True)
            data_pre_train_dropped = data_pre_train.dropna()
        
        full_input_pre_train = data_pre_train_dropped.values
        
        if self._dataset.all_columns_are_targets:
            y_pre_train = data_pre_train_dropped.values
        else:
            y_pre_train = data_pre_train_dropped[self._dataset.target_series].values
        
        assert full_input_pre_train.shape[1] == self.input_dim, \
            f"Input dimension mismatch: {full_input_pre_train.shape[1]} != {self.input_dim}"
        assert y_pre_train.shape[1] == self.output_dim, \
            f"Output dimension mismatch: {y_pre_train.shape[1]} != {self.output_dim}"
        
        full_input_tensor = torch.from_numpy(full_input_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_tensor = torch.from_numpy(y_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        final_epoch_losses = self.autoencoder.pretrain(
            full_input=full_input_tensor,
            y=y_tensor,
            epochs=pretrain_epochs,
            batch_size=self.window_size,
            optimizer=self.optimizer,
            use_mse_loss=use_mse_loss
        )
        
        if final_epoch_losses:
            _logger.info(f'Pre-training completed: final loss={final_epoch_losses[-1]:.{DEFAULT_LOSS_LOG_PRECISION}f}')
        
        self.data = self._dataset.data.copy()
        self.data_denoised = self.data.copy()
        # Note: missing_y is already available from dataset (target series only)
        # We don't need to store missing_mask separately - use dataset.missing_y directly
        self.target_indices = self._dataset.target_indices
        if not self._dataset.all_columns_are_targets:
            self._target_col_tensor = torch.tensor(self.target_indices, device=self.device, dtype=torch.long)
        self.rng = np.random.RandomState(self.initializer_seed)
        y_pred_prev, y_pred_prev_full = self._initialize_mcmc_state()
        if self._dataset.all_columns_are_targets:
            # All columns are targets
            self.y_actual = self.data.values[self.lags_input:]
        else:
            # Only some columns are targets: use target columns from original data starting from lags_input
            # For non-target case, we need to extract target columns from self.data
            self.y_actual = self.data.values[self.lags_input:, self.target_indices]
        
        converged = False
        self._num_iter = 0
        self.prediction_std = None
        self.factor_std = None
        while not converged and self._num_iter < self.max_iter:
            Phi, mu_eps, std_eps = get_idio(self.eps, self._dataset.observed_y)
            # Denoise: subtract conditional AR-idio mean from target series only
            # Features (X) are only used for encoder input, not for denoising
            # eps @ Phi gives (T-1, num_target_series), update only target columns
            eps_denoise = self.eps[:-1, :] @ Phi  # (T-1, num_target_series)
            self.data_denoised.values[self.lags_input+1:, self.target_indices] = \
                self.data_imputed.values[self.lags_input+1:, self.target_indices] - eps_denoise
            self.data_denoised_interpolated = interpolate_dataframe(self.data_denoised)
            
            # Generate MC samples using denoised data
            X_features_df, y_tmp = self._dataset.split_features_and_targets(self.data_denoised)
            X_features = X_features_df if X_features_df is not None else pd.DataFrame()
            
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
                    scheduler=self.scheduler,
                    target_indices=target_indices
                )
            
            with torch.no_grad():
                factors_list = [self.encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
                factors_tensor = torch.stack(factors_list, dim=0)
                
                y_pred_samples_tensor = torch.stack([self.decoder(f) for f in factors_list], dim=0)
                
                # Validate MC sample dimension
                if factors_tensor.shape[0] != self.n_mc_samples:
                    raise ValueError(
                        f"MC samples dimension mismatch: factors_tensor.shape[0]={factors_tensor.shape[0]} != n_mc_samples={self.n_mc_samples}"
                    )
                if y_pred_samples_tensor.shape[0] != self.n_mc_samples:
                    raise ValueError(
                        f"MC samples dimension mismatch: y_pred_samples_tensor.shape[0]={y_pred_samples_tensor.shape[0]} != n_mc_samples={self.n_mc_samples}"
                    )
                
                y_pred_full_tensor, y_pred_std_tensor = compute_tensor_stats(y_pred_samples_tensor)
                y_pred_tensor = self._extract_target_predictions(y_pred_full_tensor)
                y_pred_full = to_numpy(y_pred_full_tensor)
                y_pred = to_numpy(y_pred_tensor)
                y_pred_std = to_numpy(y_pred_std_tensor)
                self.factors = to_numpy(factors_tensor)
                self.prediction_std = y_pred_std
                
                _, factors_std_tensor = compute_tensor_stats(factors_tensor)
                self.factor_std = to_numpy(factors_std_tensor)
                
                y_pred_std_mean_check = compute_variance_mean(y_pred_std)
                should_check_variance = (
                    (y_pred_std_mean_check is not None and y_pred_std_mean_check < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD) or
                    (self._num_iter % self.disp == 0)
                )
                if should_check_variance:
                    factors_mean_tensor, _ = compute_tensor_stats(factors_tensor)
                    factors_mean = to_numpy(factors_mean_tensor)
                    variance_diagnostics = diagnose_variance_collapse(
                        prediction_std=y_pred_std,
                        prediction_mean=y_pred_full,
                        factors_mean=factors_mean,
                        y_actual=self.y_actual,
                        target_scaler=self._get_target_scaler(),
                        encoder=self.encoder,
                        decoder=self.decoder,
                        factors_std=self.factor_std
                    )
                    if variance_diagnostics['variance_collapse_detected']:
                        _logger.warning(f"Variance collapse detected at iteration {self._num_iter}: {', '.join(variance_diagnostics['warnings'])}")
            
            self._update_imputed_and_eps(y_pred_full)
            
            if self._num_iter > 1:
                y_pred_prev, y_pred = self._select_convergence_predictions(
                    y_pred_prev_full, y_pred_prev,
                    y_pred_full, y_pred
                )
                delta, self.loss_now = convergence_checker(y_pred_prev, y_pred, self.y_actual)
                
                _logger.info(f'iteration: {self._num_iter} - delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f} - loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                
                if self._num_iter % self.disp == 0:
                    prediction_std_mean = compute_variance_mean(self.prediction_std)
                    factor_std_mean = compute_variance_mean(self.factor_std) if self.factor_std is not None else None
                    log_parts = [f'iteration: {self._num_iter}', f'loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}', f'delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f}']
                    if prediction_std_mean is not None:
                        log_parts.append(f'pred_std: {prediction_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    if factor_std_mean is not None:
                        log_parts.append(f'factor_std: {factor_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    _logger.info(' - '.join(log_parts))
                
                if delta < self.tolerance:
                    converged = True
                    self._converged = True
                    _logger.info(f'Convergence achieved in {self._num_iter + 1} iterations')
            
            y_pred_prev, y_pred_prev_full = self._update_previous_predictions(
                y_pred, y_pred_full
            )
            
            if self.decoder_type == "mlp":
                self._last_iter_datasets = autoencoder_datasets
            
            self._num_iter += 1
        
        # Extract last neurons (for MLP decoder: second-to-last layer output)
        if self.decoder_type == "linear":
            self.last_neurons = self.factors
        else:
            decoder_intermediate = self.decoder.get_intermediate()
            if decoder_intermediate is None:
                raise ConfigurationError(
                    f"Decoder {type(self.decoder).__name__} has no intermediate layers for last_neurons extraction"
                )
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
        
        linear_layer = self.decoder.get_last_linear_layer()
        weight = to_numpy(linear_layer.weight.data)
        
        H = weight[:, :num_factors]
        
        # Get transition equation params (factor_order is fixed to 1)
        # F_full includes both factors and idiosyncratic components: shape (m + N, m + N)
        # Use dataset's observed_y directly (target columns only)
        F_full, Q_full, mu_0_full, Sigma_0_full, _ = get_transition_params(f_t, eps_t, bool_no_miss=self._dataset.observed_y)
        
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
        
        self.state_space_params = DDFMStateSpaceParams(
            F=F,
            Q=Q,
            mu_0=mu_0,
            Sigma_0=Sigma_0,
            H=H,
            R=R
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save DDFM model to file.
        
        Saves the complete model state using the defined dataclasses:
        - PyTorch model state_dict (autoencoder weights and registered buffers)
        - Configuration
        - Training state (DDFMTrainingState dataclass)
               - State-space parameters (DDFMStateSpaceParams dataclass)
        - Result (DDFMResult dataclass, if model is trained)
        
        Parameters
        ----------
        path : str or Path
            Path to save the model checkpoint file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sync training_state dataclass with current model state
        self.training_state.sync_from_model(self)
        
        # Get result dataclass if model is trained
        result = None
        if self._has_factors and self.state_space_params is not None:
            try:
                result = self.get_result()
            except (ModelNotTrainedError, ModelNotInitializedError):
                pass
        
        # Collect checkpoint using dataclasses
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self._config,
            'training_state': self.training_state,
            'state_space_params': self.state_space_params,
            'result': result,
            'encoder_size': self.encoder_size,
            'decoder_type': self.decoder_type,
            'decoder_size': self.decoder_size,
        }
        
        torch.save(checkpoint, path)
        _logger.info(f"DDFM model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], dataset: DDFMDataset) -> 'DDFM':
        """Load DDFM model from checkpoint file.
        
        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file
        dataset : DDFMDataset
            Dataset instance (required for model initialization)
            
        Returns
        -------
        DDFM
            Loaded DDFM model instance
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')
        
        # Extract architecture and config
        encoder_size = checkpoint.get('encoder_size')
        decoder_type = checkpoint.get('decoder_type', 'linear')
        config = checkpoint.get('config')
        
        # Create model instance
        model = cls(
            dataset=dataset,
            config=config,
            encoder_size=encoder_size,
            decoder_type=decoder_type
        )
        
        # Load PyTorch state
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Restore dataclasses
        model.training_state = checkpoint.get('training_state', DDFMTrainingState())
        model.state_space_params = checkpoint.get('state_space_params')
        
        # Restore instance attributes from training_state if trained
        if model.training_state.factors is not None:
            model.factors = model.training_state.factors
            model.eps = model.training_state.eps
            model.last_neurons = model.training_state.last_neurons
            model._num_iter = model.training_state.num_iter
            model.loss_now = model.training_state.loss_now
            model._converged = model.training_state.converged
        
        # Restore result
        if checkpoint.get('result') is not None:
            model._result = checkpoint['result']
        
        _logger.info(f"DDFM model loaded from {path}")
        return model
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dictionary and restore state_space_params dataclass."""
        result = super().load_state_dict(state_dict, strict=strict)
        if getattr(self, '_state_space_F', None) is not None:
            self.state_space_params = DDFMStateSpaceParams(
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
        
        check_condition(
            getattr(self, 'state_space_params', None) is not None,
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
        # Use last row if factors exist, otherwise use initial state
        Z_last = factors_avg[-1, :] if len(factors_avg) > 0 else params.mu_0
        
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
        y_forecast_std = Z_forecast @ H.T
        
        # Inverse transform target series to original scale (only during forecasting)
        target_scaler = self._get_target_scaler()
        
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
        
        check_condition(
            getattr(self, 'state_space_params', None) is not None,
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
        # Use get_x_sm_original_scale() on result to get unscaled values if needed
        # x_sm contains only target series (features X are only for encoder input, not in results)
        x_sm = Z @ H.T  # (T, num_target_series)
        
        # Get target scaler from dataset
        target_scaler = self._get_target_scaler()
        
        # Create result
        # For DDFM: A = F (transition), C = H (observation), r = [num_factors], p = 1
        # Z already computed above, reuse it instead of calling _get_averaged_factors() again
        num_factors = Z.shape[1]
        return DDFMResult(
            x_sm=x_sm,
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
        
        if self.factors.ndim == 3:
            n_mc_samples = self.factors.shape[0]
            new_factors_expanded = np.expand_dims(new_factors_np, axis=0)
            new_factors_expanded = np.repeat(new_factors_expanded, n_mc_samples, axis=0)
            self.factors = np.concatenate([self.factors, new_factors_expanded], axis=1)
        else:
            self.factors = np.vstack([self.factors, new_factors_np])