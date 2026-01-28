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

from ..base import BaseFactorModel
from ...logger import get_logger
from ...numeric.stability import convergence_checker
from ...numeric.builder import build_ddfm_state_space
from ...numeric.statistic import (
    compute_variance_mean, compute_array_stats,
    average_3d_array
)
from ...numeric.builder import build_ddfm_optimizer
from .encoder import SimpleAutoencoder
from .sampling import run_mcmc_iteration
from ...config.schema.params import DDFMModelState
from ...config.schema.results import DDFMResult
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
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
    DEFAULT_FACTOR_ORDER,
    DEFAULT_INF_VALUE,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_MULT_EPOCH_PRETRAIN,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_LOSS_LOG_PRECISION,
    DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
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
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError
from ...utils.validation import check_condition
from ...numeric.validator import validate_no_nan_inf, validate_update_data_shape
from ...utils.helper import interpolate_array, interpolate_dataframe
from ...config.types import to_tensor, to_numpy

from ...dataset.ddfm_dataset import DDFMDataset

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
        interpolation_method: str = 'linear',
        interpolation_limit: Optional[int] = 10,
        interpolation_limit_direction: str = 'both',
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
        
        self.scaler = getattr(dataset, 'scaler', None)
        self.min_target_interporate_ratio = min_target_interporate_ratio
        
        # Interpolation parameters (config takes precedence over function arguments)
        self.interpolation_method, self.interpolation_limit, self.interpolation_limit_direction = \
            self._get_interpolation_params(config, interpolation_method, interpolation_limit, interpolation_limit_direction)
        
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
        self.training_state = DDFMModelState()
        
        self.lags_input = 0
    
    def _get_interpolation_params(
        self,
        config: Optional[Any],
        default_method: str,
        default_limit: Optional[int],
        default_direction: str
    ) -> Tuple[str, Optional[int], str]:
        """Get interpolation parameters from config or defaults.
        
        Parameters
        ----------
        config : Optional[Any]
            Config object (may have interpolation attributes)
        default_method : str
            Default interpolation method
        default_limit : Optional[int]
            Default interpolation limit
        default_direction : str
            Default interpolation direction
            
        Returns
        -------
        Tuple[str, Optional[int], str]
            (method, limit, direction)
        """
        if config is not None:
            return (
                getattr(config, 'interpolation_method', default_method),
                getattr(config, 'interpolation_limit', default_limit),
                getattr(config, 'interpolation_limit_direction', default_direction)
            )
        return default_method, default_limit, default_direction
    
    def _get_averaged_factors(self) -> np.ndarray:
        """Get factors averaged across MC samples if 3D, otherwise return as-is."""
        return average_3d_array(self.factors, axis=0)
    
    
    @property
    def _has_factors(self) -> bool:
        """Check if factors attribute exists and is not None."""
        return getattr(self, 'factors', None) is not None
    
    def _impute_missing_targets(self, y_pred_full: np.ndarray) -> None:
        """Impute missing target values with predictions.
        
        Parameters
        ----------
        y_pred_full : np.ndarray
            Predictions for target series (T, num_target_series) or full data (T, num_series)
        """
        missing_y = self._dataset.missing_y
        if not missing_y.any():
            return
        
        if self._dataset.all_columns_are_targets:
            # y_pred_full has full shape (T, num_series)
            self.data_imputed.values[missing_y] = y_pred_full[missing_y]
        else:
            # y_pred_full only has target columns (T, num_target_series)
            self.data_imputed.values[:, self.target_indices][missing_y] = y_pred_full[missing_y]
    
    def _update_imputed_and_eps(self, y_pred_full: np.ndarray) -> None:
        """Update data_imputed with predictions and compute eps (idiosyncratic residuals).
        
        Only target series (y) are imputed - features (X) are not imputed since they're only
        used for encoder input. Uses dataset's missing_y to identify missing target values.
        """
        # Impute missing target values
        self._impute_missing_targets(y_pred_full)
        
        # Compute eps: y_actual - y_pred
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
    
    def _build_optimizer(self) -> None:
        """Build optimizer and scheduler for training."""
        self.optimizer, self.scheduler = build_ddfm_optimizer(
            model=self.autoencoder,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            n_mc_samples=self.n_mc_samples
        )
    
    def _build_inputs_for_pretrain(self, interpolate: bool = True) -> pd.DataFrame:
        """Build inputs for pre-training.
        
        Returns the data as-is (user provides data with lagged features if needed).
        Optionally interpolates missing values.
        
        Parameters
        ----------
        interpolate : bool
            Whether to interpolate missing values using configured interpolation method
            
        Returns
        -------
        pd.DataFrame
            Input data, optionally interpolated
        """
        full_input_data = self._dataset.data.copy()
        
        if interpolate and full_input_data.isna().sum().sum() > 0:
            full_input_data = interpolate_dataframe(
                full_input_data,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        
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
        """Extract target predictions from full prediction tensor.
        
        When covariates exist, decoder output_dim is already num_target_series,
        so y_pred_full_tensor already contains only target predictions.
        """
        # Decoder output_dim is always num_target_series, so predictions are already target-only
        return y_pred_full_tensor
    
    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize MCMC state: interpolate data, make initial prediction, compute initial eps."""
        self.data_denoised_interpolated = interpolate_dataframe(
            self.data_denoised,
            method=self.interpolation_method,
            limit=self.interpolation_limit,
            limit_direction=self.interpolation_limit_direction
        )
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
        self._impute_missing_targets(y_pred_full)
        
        # Compute eps: idiosyncratic residuals on *targets only*.
        # When covariates exist, the decoder predicts targets only (output_dim = num_target_series),
        # so we must subtract predictions from the target slice, not from the full (X+y) matrix.
        if self.lags_input == 0:
            # IMPORTANT: use interpolated data for residuals so eps does not contain NaNs
            # Mixed-frequency data can have large missing blocks; eps NaNs propagate into AR-idio
            # estimation and can cause SVD failures when building the state space.
            if self._dataset.all_columns_are_targets:
                y_actual_full = to_numpy(self.data_denoised_interpolated)
            else:
                y_actual_full = to_numpy(self.data_denoised_interpolated[self._dataset.target_series])
        else:
            # If lags_input > 0, match target-only shape when covariates exist
            if self._dataset.all_columns_are_targets:
                y_actual_full = full_input_data[self._dataset.data.columns].values
            else:
                y_actual_full = full_input_data[self._dataset.target_series].values
        eps_full = y_actual_full - y_pred_full
        if self._dataset.all_columns_are_targets:
            self.eps = eps_full[:, self.target_indices]
        else:
            # eps_full is already target-only
            self.eps = eps_full
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
            # Use interpolated data to avoid NaNs in convergence and diagnostics
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:]
        else:
            # Only some columns are targets: use target columns from original data starting from lags_input
            # For non-target case, we need to extract target columns from self.data
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:, self.target_indices]
        
        converged = False
        self._num_iter = 0
        self.prediction_std = None
        self.factor_std = None
        while not converged and self._num_iter < self.max_iter:
            # Run MCMC iteration: denoise, sample, train, predict
            (
                self.factors, y_pred, y_pred_full, self.prediction_std, 
                self.factor_std, self.data_denoised_interpolated, autoencoder_datasets
            ) = run_mcmc_iteration(
                eps=self.eps,
                data_imputed=self.data_imputed,
                data_denoised=self.data_denoised,
                dataset=self._dataset,
                encoder=self.encoder,
                decoder=self.decoder,
                autoencoder=self.autoencoder,
                y_actual=self.y_actual,
                lags_input=self.lags_input,
                n_mc_samples=self.n_mc_samples,
                window_size=self.window_size,
                learning_rate=self.learning_rate,
                optimizer_type=self.optimizer_type,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extract_target_predictions=self._extract_target_predictions,
                interpolation_method=self.interpolation_method,
                interpolation_limit=self.interpolation_limit,
                interpolation_limit_direction=self.interpolation_limit_direction,
                target_scaler=self.scaler,
                num_iter=self._num_iter,
                disp=self.disp,
                device=self.device,
                rng=self.rng
            )
            
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
            
            # Store last iteration datasets for MLP decoder (needed for last_neurons extraction)
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
        
        # Get decoder weight
        linear_layer = self.decoder.get_last_linear_layer()
        decoder_weight = to_numpy(linear_layer.weight.data)
        
        # Build state-space parameters
        F, Q, mu_0, Sigma_0, H, R = build_ddfm_state_space(
            factors=f_t,
            eps=eps_t,
            decoder_weight=decoder_weight,
            observed_y=self._dataset.observed_y,
            model_name=self.__class__.__name__
        )
        
        # Store state-space parameters in training_state (single source of truth)
        self.training_state.F = F
        self.training_state.Q = Q
        self.training_state.mu_0 = mu_0
        self.training_state.Sigma_0 = Sigma_0
        self.training_state.H = H
        self.training_state.R = R
    
    def save(self, path: Union[str, Path]) -> None:
        """Save DDFM model to file.
        
        Saves the complete model state using the defined dataclasses:
        - PyTorch model state_dict (autoencoder weights)
        - Configuration
        - Training state (DDFMModelState dataclass) - single source of truth for all model state
               including training state and state-space parameters
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
        result = self._result if self._result is not None else (self.get_result() if self._has_factors and self.training_state.F is not None else None)
        
        # Extract dataset metadata if available (needed for forecasting)
        dataset_metadata = None
        if hasattr(self, '_dataset') and self._dataset is not None:
            dataset = self._dataset
            dataset_metadata = {
                'colnames': list(dataset.data.columns) if hasattr(dataset, 'data') and dataset.data is not None else None,
                'covariates': dataset.covariates if hasattr(dataset, 'covariates') else None,
                'time_idx': dataset.time_idx if hasattr(dataset, 'time_idx') else None,
            }
        
        # Add DDFM-specific dimensions to metadata
        if dataset_metadata is not None:
            dataset_metadata.update({
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'num_series': self.num_series,
                'window_size': self.window_size,  # Also save window_size for dummy data creation
            })
        
        # Collect checkpoint using dataclasses
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self._config,
            'training_state': self.training_state,
            'result': result,
            'encoder_size': self.encoder_size,
            'decoder_type': self.decoder_type,
            'decoder_size': self.decoder_size,
            'dataset_metadata': dataset_metadata,  # Save dataset metadata in model checkpoint
            'scaler': self.scaler,  # Save scaler for forecast scaling
        }
        
        torch.save(checkpoint, path)
        _logger.info(f"DDFM model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], dataset: Optional[DDFMDataset] = None) -> 'DDFM':
        """Load DDFM model from checkpoint file.
        
        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file
        dataset : DDFMDataset, optional
            Dataset instance. If None, a minimal dataset will be created from checkpoint metadata.
            
        Returns
        -------
        DDFM
            Loaded DDFM model instance
        """
        path = Path(path)
        # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which blocks
        # unpickling non-tensor objects (e.g., our config dataclass) unless allowlisted.
        # These checkpoints are produced locally by this project, so load them fully.
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions don't support weights_only
            checkpoint = torch.load(path, map_location='cpu')
        
        # Extract architecture and config
        encoder_size = checkpoint.get('encoder_size')
        decoder_type = checkpoint.get('decoder_type', 'linear')
        config = checkpoint.get('config')
        
        # Create minimal dataset from metadata if dataset not provided
        if dataset is None:
            dataset = cls._create_minimal_dataset_from_metadata(checkpoint)
        
        # Create model instance
        model = cls(
            dataset=dataset,
            config=config,
            encoder_size=encoder_size,
            decoder_type=decoder_type
        )
        
        # Build autoencoder modules before loading state_dict so attributes exist
        # (needed for update()/predict() paths that use encoder/decoder).
        model.autoencoder = SimpleAutoencoder.build(
            input_dim=model.input_dim,
            encoder_size=model.encoder_size,
            decoder_size=model.decoder_size,
            decoder_type=model.decoder_type,
            output_dim=model.output_dim,
            activation=model.activation,
            seed=model.initializer_seed
        )
        model.encoder = model.autoencoder.encoder
        model.decoder = model.autoencoder.decoder
        
        # Load PyTorch state
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Ensure modules are on the same device expected by update()/predict()
        model.autoencoder.to(model.device)
        
        # Restore dataclasses
        model.training_state = checkpoint.get('training_state', DDFMModelState())
        
        # Restore instance attributes from training_state if trained
        if model.training_state.factors is not None:
            model.factors = model.training_state.factors
            model.eps = model.training_state.eps
            model.last_neurons = model.training_state.last_neurons
            model._num_iter = model.training_state.num_iter
            model.loss_now = model.training_state.loss_now
            model._converged = model.training_state.converged
        
        # Reconstruct runtime attributes needed for update()/forecast paths
        # (these are not part of state_dict/training_state but are expected by methods).
        # For minimal datasets created from metadata, we still need these attributes
        model.data = model._dataset.data.copy()
        model.target_indices = model._dataset.target_indices
        if not model._dataset.all_columns_are_targets:
            model._target_col_tensor = torch.tensor(model.target_indices, device=model.device, dtype=torch.long)
        
        # Restore checkpoint metadata (uses base class helper)
        model._restore_checkpoint_metadata(checkpoint)
        
        # Restore result
        if checkpoint.get('result') is not None:
            model._result = checkpoint['result']
        
        # Ensure scaler is set (from checkpoint or dataset)
        if model.scaler is None:
            # Try to get from checkpoint first
            if checkpoint.get('scaler') is not None:
                model.scaler = checkpoint['scaler']
            # Fallback to dataset scaler
            elif dataset is not None and hasattr(dataset, 'scaler') and dataset.scaler is not None:
                model.scaler = dataset.scaler
            else:
                _logger.warning("DDFM model loaded without scaler - forecasts may not be properly scaled")
        
        _logger.info(f"DDFM model loaded from {path}")
        return model
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dictionary."""
        return super().load_state_dict(state_dict, strict=strict)
    
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        data: Optional[Union[np.ndarray, Any]] = None,
        update: bool = True,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values using trained state-space model. Requires build_state_space() to be called.
        
        **New Data Initialization**: If `data` is provided:
        - If `update=True` (default): Updates the model's internal state with new data
          via neural network forward pass, then forecasts from the updated state.
        - If `update=False`: Uses data only for initializing factor state without modifying model state.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast
        data : DDFMDataset, optional
            New preprocessed dataset to use for updating model state. If provided:
            - With `update=True` (default): Updates model state, then forecasts from updated state.
            - With `update=False`: Uses data only for initializing factor state without modifying model state.
        update : bool, default True
            If True and `data` is provided, updates the model's internal state with new data
            before forecasting. If False, uses data only for initializing factor state without
            modifying model state.
        """
        
        # Validate model is trained
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        
        check_condition(
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} prediction failed: state-space model has not been built",
            details="Please call build_state_space() after training to enable prediction"
        )
        
        # Handle data update if provided
        if data is not None and update:
            # Update model state with new data, then use updated state for forecasting
            if not isinstance(data, DDFMDataset):
                raise TypeError(
                    f"DDFM.predict() requires data to be a DDFMDataset when update=True. "
                    f"Got {type(data)} instead."
                )
            self._update(data)
        
        # Validate and resolve horizon
        from ...numeric.validator import validate_horizon
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        horizon = validate_horizon(horizon)
        
        # Get state-space parameters from training_state (single source of truth)
        F = self.training_state.F  # Transition matrix (m x m)
        H = self.training_state.H  # Observation matrix (N x m) where N is num_target_series
        # Defensive: training instability can introduce NaNs/Infs in state-space params
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get last factor state from training (average across MC samples)
        factors_avg = self._get_averaged_factors()
        
        # Get last factor state: (num_factors,)
        # Use last row if factors exist, otherwise use initial state
        Z_last = factors_avg[-1, :] if len(factors_avg) > 0 else self.training_state.mu_0
        # Defensive: factors can contain NaNs if upstream training/update was numerically unstable.
        # Replace with 0s so forecasting can still proceed (reporting/evaluation use-case).
        Z_last = np.nan_to_num(Z_last, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate factor state
        validate_no_nan_inf(Z_last, name="last factor state Z_last")
        validate_no_nan_inf(F, name="transition matrix F")
        validate_no_nan_inf(H, name="observation matrix H")
        
        # Forecast factors forward using AR(1) dynamics
        from ...numeric.estimator import forecast_ar1_factors
        Z_forecast = forecast_ar1_factors(Z_last, F, horizon, dtype=DEFAULT_DTYPE)
        
        # Transform factors to observations (target series only)
        # H shape: (num_target_series, num_factors)
        # Z_forecast shape: (horizon, num_factors)
        # y_forecast shape: (horizon, num_target_series)
        y_forecast_std = Z_forecast @ H.T
        
        # Inverse transform target series to original scale
        if self.scaler is None:
            raise ConfigurationError(
                f"{self.__class__.__name__} forecast failed: scaler is None",
                details="Dataset must provide scaler for proper forecast scaling"
            )
        
        # Scaler was fitted on target series only, so inverse_transform directly
        y_forecast = self.scaler.inverse_transform(y_forecast_std)
        y_forecast = np.nan_to_num(y_forecast, nan=0.0, posinf=0.0, neginf=0.0)
        
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
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} get_result failed: state-space model has not been built",
            details="Please call build_state_space() after training"
        )
        
        # Get state-space parameters from training_state (single source of truth)
        F = self.training_state.F  # Transition matrix
        H = self.training_state.H  # Observation matrix
        Q = self.training_state.Q  # Process noise covariance
        R = self.training_state.R  # Observation noise covariance
        mu_0 = self.training_state.mu_0  # Initial state mean
        Sigma_0 = self.training_state.Sigma_0  # Initial state covariance
        
        # Get factors (average across MC samples if 3D)
        Z = self._get_averaged_factors()
        
        # Compute smoothed data: x_sm = Z @ H.T
        # Use get_x_sm_original_scale() on result to get unscaled values if needed
        # x_sm contains only target series (features X are only for encoder input, not in results)
        x_sm = Z @ H.T  # (T, num_target_series)
        
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
            target_scaler=self.scaler,  # Store scaler in target_scaler field (DFMResult API)
            converged=getattr(self, '_converged', False),
            num_iter=getattr(self, '_num_iter', self.max_iter),  # Use actual iteration count if available
            loglik=-DEFAULT_INF_VALUE  # DDFM doesn't compute log-likelihood
        )

    def update(self, data: Union[np.ndarray, Any], *args, **kwargs) -> None:
        """Update model state with new observations.

        For DDFM we only support updating with a `DDFMDataset`, which guarantees
        consistent preprocessing and column order.
        """
        if not isinstance(data, DDFMDataset):
            raise TypeError(f"DDFM.update() expects a DDFMDataset, got {type(data)}")
        self._update(data)
    
    @classmethod
    def _create_minimal_dataset_from_metadata(cls, checkpoint: dict) -> DDFMDataset:
        """Create minimal DDFMDataset from checkpoint metadata.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary containing dataset_metadata and scaler
            
        Returns
        -------
        DDFMDataset
            Minimal dataset with correct structure for model loading
        """
        dataset_metadata = checkpoint.get('dataset_metadata')
        if dataset_metadata is None:
            raise ValueError(
                "Cannot load DDFM model: dataset_metadata not found in checkpoint. "
                "Either provide a dataset parameter or ensure the checkpoint was saved with dataset_metadata."
            )
        
        # Extract metadata
        colnames = dataset_metadata.get('colnames')
        covariates = dataset_metadata.get('covariates', [])
        time_idx = dataset_metadata.get('time_idx')
        
        if colnames is None:
            raise ValueError("Cannot create dataset: colnames not found in checkpoint metadata")
        
        # Create minimal dummy data with correct column structure
        # We only need the shape properties, not actual data
        window_size = dataset_metadata.get('window_size', checkpoint.get('window_size', 10))
        num_rows = max(10, window_size)
        dummy_data = pd.DataFrame(
            np.zeros((num_rows, len(colnames))),
            columns=colnames,
            index=pd.date_range('2000-01-01', periods=num_rows, freq='W')
        )
        
        # If time_idx is a column name, ensure it exists in dummy_data
        # Otherwise, use None (DDFMDataset will use index)
        if time_idx and time_idx not in colnames:
            time_idx = None
        
        # Get scaler from checkpoint if available
        scaler = checkpoint.get('scaler')
        
        # Create minimal dataset with metadata
        return DDFMDataset(
            data=dummy_data,
            time_idx=time_idx if time_idx else 'date',  # Use 'date' as default if not specified
            covariates=covariates if covariates else None,
            scaler=scaler  # Restore scaler from checkpoint
        )
    
    def _refit_scalers_with_new_data(
        self,
        dataset: DDFMDataset,
        new_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Re-fit target scaler with combined training and new data.
        
        Parameters
        ----------
        dataset : DDFMDataset
            New dataset with data to update
        new_data_df : pd.DataFrame
            New data DataFrame (may be interpolated)
            
        Returns
        -------
        pd.DataFrame
            Updated new_data_df with re-scaled target values
        """
        target_cols = list(self._dataset.target_series)
        
        # Get unstandardized new data from dataset for scaler re-fitting
        # Dataset stores original (unstandardized) data in data_original before scaling
        if hasattr(dataset, 'data_original') and dataset.data_original is not None:
            new_data_unstd_df = dataset.data_original.copy()
            new_targets_unstd = new_data_unstd_df[target_cols].values if target_cols else np.array([]).reshape(new_data_unstd_df.shape[0], 0)
            _logger.debug(f"DDFM._refit_scalers_with_new_data(): Using data_original for scaler re-fitting ({new_data_unstd_df.shape[0]} observations)")
        else:
            # Fallback: inverse-standardize the already-scaled data
            _logger.warning("DDFM._refit_scalers_with_new_data(): dataset.data_original not available, inverse-standardizing scaled data (may have precision loss)")
            new_targets_unstd = new_data_df[target_cols].values if target_cols else np.array([]).reshape(new_data_df.shape[0], 0)
            # Inverse-standardize targets
            if self.scaler is not None and target_cols and hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                new_targets_unstd = new_targets_unstd * self.scaler.scale_[np.newaxis, :] + self.scaler.mean_[np.newaxis, :]
        
        # Get training data (already scaled, need to inverse-standardize for re-fitting)
        training_data_df = self.data.copy()
        
        # Re-fit target scaler on combined data (training + new)
        if self.scaler is not None and target_cols and len(new_targets_unstd) > 0:
            # Inverse-standardize training target data
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                training_targets_scaled = training_data_df[target_cols].values
                training_targets_unstd = training_targets_scaled * self.scaler.scale_[np.newaxis, :] + self.scaler.mean_[np.newaxis, :]
            else:
                training_targets_unstd = training_data_df[target_cols].values
            
            # Combine training + new target data for re-fitting
            combined_targets = np.vstack([training_targets_unstd, new_targets_unstd])
            
            # Re-fit scaler on combined data
            self.scaler.fit(combined_targets)
            _logger.debug(f"DDFM._refit_scalers_with_new_data(): Re-fitted target scaler on {combined_targets.shape[0]} observations (training + new data)")
            
            # Transform new data with updated scaler
            target_vals = self.scaler.transform(new_targets_unstd)
            new_data_df[target_cols] = target_vals
            # Update dataset's data and y with newly transformed values (keep in sync)
            dataset.data[target_cols] = target_vals
            if hasattr(dataset, 'y') and dataset.y is not None:
                dataset.y = new_data_df[target_cols].values
                _logger.debug(f"DDFM._refit_scalers_with_new_data(): Updated dataset.y with re-scaled target values")
        
        return new_data_df
    
    def _update(self, dataset: DDFMDataset) -> None:
        """Update model factors with new data using neural network forward pass."""
        from ...utils.errors import DataValidationError
        
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
        
        # Validate dataset has same number of features and column order
        # DDFMDataset has self.data (DataFrame), not data_processed attribute
        # CRITICAL: Column order must match training data exactly for correct scaling
        new_data_df = dataset.data.copy()
        training_data_df = self.data.copy()
        
        # Validate column count matches
        if len(new_data_df.columns) != len(training_data_df.columns):
            raise DataValidationError(
                f"DDFM._update(): Column count mismatch!\n"
                f"Training data has {len(training_data_df.columns)} columns, "
                f"but update data has {len(new_data_df.columns)} columns.\n"
                f"Update data must have exactly the same columns as training data.\n"
                f"Training columns (first 5): {list(training_data_df.columns[:5])}\n"
                f"Update columns (first 5): {list(new_data_df.columns[:5])}"
            )
        
        # Validate column order matches exactly
        if list(new_data_df.columns) != list(training_data_df.columns):
            # Find first mismatch
            first_mismatch = next((i for i, (expected, actual) in enumerate(zip(training_data_df.columns, new_data_df.columns)) if expected != actual), None)
            if first_mismatch is not None:
                raise DataValidationError(
                    f"DDFM._update(): Column order mismatch at index {first_mismatch}!\n"
                    f"Expected '{training_data_df.columns[first_mismatch]}', got '{new_data_df.columns[first_mismatch]}'.\n"
                    f"Update data columns must be in the EXACT same order as training data.\n"
                    f"Expected order (first 5): {list(training_data_df.columns[:5])}\n"
                    f"Got order (first 5): {list(new_data_df.columns[:5])}\n"
                    f"Reorder columns to match training order before creating DDFMDataset."
                )
        
        # Validate all columns are numeric (same format as training)
        non_numeric_cols = [col for col in new_data_df.columns if not pd.api.types.is_numeric_dtype(new_data_df[col])]
        if non_numeric_cols:
            raise DataValidationError(
                f"DDFM._update(): Non-numeric columns found: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}.\n"
                f"All columns must be numeric to match training data format."
            )
        
        # Ensure no NaNs/Infs are fed into the encoder during update.
        # Mixed-frequency data updates often include missing values; without interpolation,
        # encoder outputs NaNs and forecasting fails.
        if new_data_df.isna().sum().sum() > 0:
            new_data_df = interpolate_dataframe(
                new_data_df,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        
        # Re-fit scalers with combined data (training + new) to update statistics
        new_data_df = self._refit_scalers_with_new_data(dataset, new_data_df)
        new_data = np.asarray(new_data_df.values)
        training_data = np.asarray(self.data.values)
        
        # Validate data shape matches training format
        validate_update_data_shape(
            data=new_data,
            training_data=training_data,
            model_name=self.__class__.__name__
        )
        
        # Additional validation: ensure 2D array format
        if new_data.ndim != 2:
            raise DataValidationError(
                f"DDFM._update(): Data must be 2D array (time_steps × features), got shape {new_data.shape}.\n"
                f"Expected format: (T, {training_data.shape[1]}) where T is number of time steps."
            )
        
        # Validate target_series match (or covariates match, which determines target_series)
        # Check if target_series match, or if covariates match (which would result in same target_series)
        new_targets = getattr(dataset, 'target_series', None)
        new_covariates = getattr(dataset, 'covariates', None) or []
        train_targets = getattr(self._dataset, 'target_series', None)
        train_covariates = getattr(self._dataset, 'covariates', None) or []
        
        # If target_series don't match, check if it's due to covariates difference
        if new_targets != train_targets:
            # If covariates match, target_series should also match (unless data columns changed)
            if new_covariates != train_covariates:
                raise DataValidationError(
                    f"target_series mismatch: new dataset has {new_targets}, "
                    f"but training dataset has {train_targets}. "
                    f"This may be due to covariates difference: new={new_covariates}, train={train_covariates}"
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