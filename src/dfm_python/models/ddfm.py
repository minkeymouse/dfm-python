"""Deep Dynamic Factor Model (DDFM) using PyTorch.

Implements the original DDFM algorithm with MCMC-based denoising training
and sequential MC sample processing.
"""

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
from ..utils.helper import interpolate_array
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
        if self.factors.ndim == 3:
            return np.mean(self.factors, axis=0)  # Average across MC samples
        return self.factors
    
    def _compute_variance_mean(self, variance_array: Optional[np.ndarray]) -> Optional[float]:
        """Compute mean of variance array for logging.
        
        Consolidates duplicate pattern: float(np.mean(variance_array)) used for
        prediction_std and factor_std logging.
        
        Parameters
        ----------
        variance_array : np.ndarray, optional
            Variance array (prediction_std or factor_std)
            
        Returns
        -------
        float, optional
            Mean of variance array, or None if array is None
        """
        if variance_array is None:
            return None
        return float(np.mean(variance_array))
    
    def _compute_array_stats(self, array: np.ndarray, use_nan: bool = False) -> Tuple[float, float, float, float]:
        """Compute statistics (mean, std, min, max) for numpy array."""
        if use_nan:
            return (
                float(np.nanmean(array)),
                float(np.nanstd(array)),
                float(np.nanmin(array)),
                float(np.nanmax(array))
            )
        else:
            return (
                float(np.mean(array)),
                float(np.std(array)),
                float(np.min(array)),
                float(np.max(array))
            )
    
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of tensor along dimension 0 (MC samples dimension).
        
        Consolidates duplicate pattern: tensor.mean(dim=0) and tensor.std(dim=0) used
        for predictions_full_tensor and factors_tensor statistics.
        
        **CRITICAL**: Uses `unbiased=False` to match TensorFlow's `tf.reduce_std` behavior.
        TensorFlow's `tf.reduce_std` uses population std (unbiased=False) by default, while
        PyTorch's `tensor.std()` defaults to sample std (unbiased=True, Bessel's correction).
        This mismatch can cause systematic differences in prediction_std computation, especially
        with small MC sample counts (n_mc_samples=10), potentially explaining variance collapse.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor with shape (n_mc_samples, ...) where first dimension is MC samples
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (mean, std) computed along dimension 0, where std uses population std (unbiased=False)
            to match TensorFlow's tf.reduce_std behavior
        """
        return tensor.mean(dim=0), tensor.std(dim=0, unbiased=False)
    
    def _extract_batchnorm_statistics(self) -> list:
        """Extract BatchNorm statistics (running_mean, running_var) from encoder/decoder.
        
        Consolidates BatchNorm statistics inspection pattern used in variance collapse diagnostics.
        
        Returns
        -------
        list[dict]
            List of BatchNorm statistics dictionaries: {
                'module': str ('encoder' or 'decoder'),
                'layer': str (layer name),
                'running_mean_abs': float,
                'running_var_mean': float
            }
        """
        batchnorm_stats = []
        for module_name, module in [('encoder', self.encoder), ('decoder', self.decoder)]:
            for name, submodule in module.named_modules():
                if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    running_mean = to_numpy(submodule.running_mean) if submodule.running_mean.numel() > 0 else None
                    running_var = to_numpy(submodule.running_var) if submodule.running_var.numel() > 0 else None
                    if running_mean is not None and running_var is not None:
                        mean_abs = float(np.mean(np.abs(running_mean)))
                        var_mean = self._compute_variance_mean(running_var)
                        batchnorm_stats.append({
                            'module': module_name,
                            'layer': name,
                            'running_mean_abs': mean_abs,
                            'running_var_mean': var_mean
                        })
        return batchnorm_stats
    
    def _get_decoder_intermediate(self) -> nn.Sequential:
        """Get decoder intermediate layers (all except last layer).
        
        Extracts decoder structure assuming decoder is nn.Sequential.
        Used for last_neurons extraction (second-to-last layer output).
        
        Returns
        -------
        nn.Sequential
            Decoder intermediate layers (all children except last)
            
        Raises
        ------
        ConfigurationError
            If decoder is not Sequential or has no children
        """
        if not isinstance(self.decoder, nn.Sequential):
            raise ConfigurationError(
                f"Decoder must be nn.Sequential for last_neurons extraction, "
                f"got {type(self.decoder).__name__}"
            )
        decoder_children = list(self.decoder.children())
        if len(decoder_children) < 2:
            raise ConfigurationError(
                f"Decoder must have at least 2 layers for last_neurons extraction, "
                f"got {len(decoder_children)} layers"
            )
        return nn.Sequential(*decoder_children[:-1])
    
    def _get_decoder_last_linear_layer(self) -> nn.Linear:
        """Get decoder's last Linear layer.
        
        Searches decoder modules recursively to find Linear layers.
        Returns the last Linear layer (assumed to be the output layer).
        
        Returns
        -------
        nn.Linear
            Decoder's last Linear layer
            
        Raises
        ------
        ConfigurationError
            If no Linear layer found in decoder
        """
        linear_layers = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ConfigurationError("No Linear layer found in decoder")
        # Return last Linear layer (assumed to be output layer)
        return linear_layers[-1]
    
    def _diagnose_variance_collapse(
        self,
        prediction_std: np.ndarray,
        prediction_mean: np.ndarray,
        factors_mean: np.ndarray,
        factors_std: Optional[np.ndarray] = None
    ) -> dict:
        """Diagnose root cause of variance collapse.
        
        Detects when prediction std is too low (std ~DEFAULT_VARIANCE_COLLAPSE_STD vs target ~DEFAULT_TARGET_PREDICTION_STD).
        Uses constants DEFAULT_VARIANCE_COLLAPSE_STD and DEFAULT_TARGET_PREDICTION_STD for target values in diagnostics.
        
        Provides actionable diagnostics to identify why prediction variance is too low.
        Checks: (1) decoder output scale vs data scale, (2) BatchNorm statistics,
        (3) factor magnitudes, (4) per-time-step variance patterns.
        
        Parameters
        ----------
        prediction_std : np.ndarray
            Prediction std across MC samples (shape: (T, N))
        prediction_mean : np.ndarray
            Prediction mean across MC samples (shape: (T, N))
        factors_mean : np.ndarray
            Factor mean across MC samples (shape: (T, m))
        factors_std : np.ndarray, optional
            Factor std across MC samples (shape: (T, m))
            
        Returns
        -------
        dict
            Diagnostic information: {
                'prediction_std_mean': float,
                'data_std_mean': float,
                'scale_ratio': float,
                'factors_mean_abs': float,
                'factors_std_mean': float (if factors_std provided),
                'variance_collapse_detected': bool,
                'warnings': list[str]
            }
        """
        diagnostics = {
            'prediction_std_mean': None,
            'variance_collapse_detected': False,
            'warnings': []
        }
        
        # Validate prediction_std before computing mean to avoid numpy warnings
        if not isinstance(prediction_std, np.ndarray):
            diagnostics['warnings'].append(f"Invalid prediction_std type: {type(prediction_std).__name__}, expected np.ndarray")
            return diagnostics
        
        if prediction_std.ndim == 0 or prediction_std.size == 0:
            diagnostics['warnings'].append(f"Invalid prediction_std shape: {prediction_std.shape}, expected 2D array (T, N)")
            return diagnostics
        
        # Handle 1D arrays (single time step or single series)
        if prediction_std.ndim == 1:
            diagnostics['warnings'].append(f"1D prediction_std array (shape: {prediction_std.shape}), per-time-step analysis skipped")
            return diagnostics
        
        # Validate 2D array shape
        if prediction_std.ndim != 2:
            diagnostics['warnings'].append(f"Invalid prediction_std dimensions: {prediction_std.ndim}, expected 2D array (T, N)")
            return diagnostics
        
        diagnostics['prediction_std_mean'] = self._compute_variance_mean(prediction_std)
        
        # Check 1: Decoder output scale vs data scale
        data_mean, data_std, _, _ = self._compute_array_stats(self.y_actual)
        diagnostics['data_std_mean'] = data_std
        diagnostics['data_mean_abs'] = abs(data_mean)
        
        # Check if data is standardized (mean ≈ 0, std ≈ DEFAULT_TARGET_PREDICTION_STD)
        target_scaler = self._get_target_scaler()
        is_standardized = (
            isinstance(target_scaler, StandardScaler) and
            abs(data_mean) < DEFAULT_STANDARDIZATION_MEAN_THRESHOLD and
            DEFAULT_STANDARDIZATION_STD_MIN <= data_std <= DEFAULT_STANDARDIZATION_STD_MAX
        )
        diagnostics['is_standardized'] = is_standardized
        
        if not is_standardized:
            diagnostics['warnings'].append(
                f"Data standardization assumption not verified: data_mean={data_mean:.6f}, data_std={data_std:.6f} "
                f"(expected: mean≈0, std≈{DEFAULT_TARGET_PREDICTION_STD} for StandardScaler). Diagnostics may be inaccurate."
            )
            target_std = data_std
        else:
            target_std = DEFAULT_TARGET_PREDICTION_STD
        
        diagnostics['scale_ratio'] = diagnostics['prediction_std_mean'] / data_std if data_std > 0 else float('inf')
        
        if diagnostics['prediction_std_mean'] < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD:
            diagnostics['variance_collapse_detected'] = True
            diagnostics['warnings'].append(
                f"Variance collapse detected: prediction_std={diagnostics['prediction_std_mean']:.6f} << target ~{target_std:.6f}"
            )
        
        if diagnostics['scale_ratio'] < DEFAULT_SCALE_RATIO_MIN:
            diagnostics['warnings'].append(f"Scale mismatch: prediction_std/data_std={diagnostics['scale_ratio']:.6f} << target ~{target_std:.6f}")
        
        factors_mean_abs = float(np.mean(np.abs(factors_mean)))
        diagnostics['factors_mean_abs'] = factors_mean_abs
        if factors_mean_abs < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
            diagnostics['warnings'].append(f"Factor collapse: |factors_mean|={factors_mean_abs:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
        
        if factors_std is not None:
            factors_std_mean = self._compute_variance_mean(factors_std)
            diagnostics['factors_std_mean'] = factors_std_mean
            if factors_std_mean < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
                diagnostics['warnings'].append(f"Factor variance collapse: factors_std={factors_std_mean:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
        
        batchnorm_stats = self._extract_batchnorm_statistics()
        for stat in batchnorm_stats:
            if stat['running_var_mean'] < DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD:
                diagnostics['warnings'].append(
                    f"BatchNorm signal suppression in {stat['module']}.{stat['layer']}: running_var={stat['running_var_mean']:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}"
                )
        diagnostics['batchnorm_stats'] = batchnorm_stats
        
        if len(prediction_std) > 1:
            per_timestep_std = np.mean(prediction_std, axis=1)
            timestep_collapse_count = int(np.sum(per_timestep_std < DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD))
            timestep_collapse_ratio = timestep_collapse_count / len(per_timestep_std)
            diagnostics['timestep_collapse_count'] = timestep_collapse_count
            diagnostics['timestep_collapse_ratio'] = timestep_collapse_ratio
            if timestep_collapse_ratio > DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD:
                diagnostics['warnings'].append(
                    f"Localized variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD} "
                    f"(ratio={timestep_collapse_ratio:.2%})"
                )
            elif timestep_collapse_count > 0:
                diagnostics['warnings'].append(
                    f"Partial variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD}"
                )
        
        return diagnostics
    
    
    @property
    def _has_factors(self) -> bool:
        """Check if factors attribute exists and is not None."""
        return getattr(self, 'factors', None) is not None
    
    def _get_target_scaler(self):
        """Get target scaler from dataset.
        
        Consolidates duplicate pattern of extracting target_scaler from dataset.
        
        Returns
        -------
        target_scaler or None
            Target scaler from dataset, or None if not available
        """
        return getattr(self._dataset, 'target_scaler', None)
    
    def _expand_to_full_series_shape(self, target_array: np.ndarray) -> np.ndarray:
        """Expand target array to full series shape by padding with zeros.
        
        Consolidates duplicate pattern of creating zero array and assigning to target_indices.
        If all columns are targets, returns the input array as-is (no expansion needed).
        
        Parameters
        ----------
        target_array : np.ndarray
            Array with shape (T, num_target_series)
            
        Returns
        -------
        np.ndarray
            Array with shape (T, num_series) with target_array values in target_indices columns.
            If all columns are targets, returns target_array unchanged.
        """
        if self._dataset.all_columns_are_targets:
            # All columns are targets: no expansion needed, return as-is
            return target_array
        full_array = np.zeros((target_array.shape[0], self.num_series))
        full_array[:, self.target_indices] = target_array
        return full_array
    
    def _update_imputed_and_eps(self, prediction_iter_full: np.ndarray) -> None:
        """Update data_imputed with predictions and compute eps (idiosyncratic residuals)."""
        if self.missing_mask.any():
            self.data_imputed.values[self.missing_mask] = prediction_iter_full[self.missing_mask]
        eps_full = self.data_imputed.values - prediction_iter_full
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
            Input data with lagged features (if lags_input > 0) and optionally interpolated
        """
        if self.lags_input == 0:
            data_tmp = self._dataset.data.copy()
        else:
            # Create lagged features
            new_dict = {}
            for col_name in self._dataset.data.columns:
                new_dict[col_name] = self._dataset.data[col_name]
                for lag in range(self.lags_input):
                    new_dict[f'{col_name}_lag{lag + 1}'] = self._dataset.data[col_name].shift(lag + 1)
            data_tmp = pd.DataFrame(new_dict, index=self._dataset.data.index)
            # Drop initial nans from lagging
            data_tmp = data_tmp[self.lags_input:]
        
        if interpolate and data_tmp.isna().sum().sum() > 0:
            data_tmp = data_tmp.interpolate(method='spline', limit_direction='both', order=3)
        
        return data_tmp
    
    def _interpolate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate DataFrame values in-place and return."""
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
        """Select appropriate predictions for convergence checking."""
        if self._dataset.all_columns_are_targets:
            return prediction_prev_full, prediction_iter_full
        else:
            return prediction_prev_target, prediction_iter_target
    
    def _extract_target_predictions(self, prediction_full_tensor: torch.Tensor) -> torch.Tensor:
        """Extract target predictions from full prediction tensor."""
        if self._dataset.all_columns_are_targets:
            return prediction_full_tensor
        else:
            # When covariates are present, autoencoder output_dim is num_target_series,
            # so prediction_full_tensor already contains only target predictions
            # Check if the prediction shape matches target shape
            if prediction_full_tensor.shape[-1] == len(self._dataset.target_series):
                return prediction_full_tensor
            # Otherwise, extract target columns (legacy case)
            if prediction_full_tensor.dim() == 1:
                return prediction_full_tensor[self._target_col_tensor]
            elif prediction_full_tensor.dim() == 2:
                return prediction_full_tensor.index_select(1, self._target_col_tensor)
            else:
                return prediction_full_tensor.index_select(-1, self._target_col_tensor)
    
    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize MCMC state: interpolate data, make initial prediction, compute initial eps."""
        self.data_denoised_interpolated = self._interpolate_dataframe(self.data_denoised)
        self.data_imputed = self.data_denoised_interpolated.copy()
        
        # Match TensorFlow: build_inputs() then predict on data_tmp.values
        # For lags_input=0, _build_inputs_for_pretrain returns self.data (no lags)
        # Use interpolate=True to match TensorFlow's build_inputs() behavior
        data_tmp = self._build_inputs_for_pretrain(interpolate=True)
        data_original = to_numpy(data_tmp)
        data_original_tensor = to_tensor(data_original, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        prediction_iter_full_tensor = self.autoencoder.predict(data_original_tensor)
        prediction_iter_target_tensor = prediction_iter_full_tensor
        prediction_iter_full = to_numpy(prediction_iter_full_tensor)
        prediction_iter_target = to_numpy(prediction_iter_target_tensor)
        
        # Update imputed data with predictions (fill missing values)
        if self.missing_mask.any():
            self.data_imputed.values[self.missing_mask] = prediction_iter_full[self.missing_mask]
        
        # Compute eps: Match TensorFlow's self.eps = self.data_tmp[self.data.columns].values - prediction_iter
        # For lags_input=0, data_tmp is self.data, so data_tmp[self.data.columns] is just self.data
        # For all-targets case, prediction_iter_full is already full shape
        if self.lags_input == 0:
            data_for_eps = to_numpy(self.data)
        else:
            # If lags_input > 0, extract only original columns (matching TensorFlow's self.data.columns)
            data_for_eps = data_tmp[self._dataset.data.columns].values
        eps_full = data_for_eps - prediction_iter_full
        self.eps = eps_full[:, self.target_indices]
        prediction_prev_iter, prediction_prev_iter_full = self._update_previous_predictions(
            prediction_iter_target, prediction_iter_full
        )
        
        return prediction_prev_iter, prediction_prev_iter_full
                
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
        
        min_obs = 50
        mult_epoch_pre = 1
        pretrain_epochs = self.n_mc_samples * mult_epoch_pre
        
        data_pre_train = self._build_inputs_for_pretrain(interpolate=False)
        data_pre_train_dropped = data_pre_train.dropna()
        use_mse_loss = len(data_pre_train_dropped) >= min_obs
        
        if not use_mse_loss:
            data_pre_train = self._build_inputs_for_pretrain(interpolate=True)
            data_pre_train_dropped = data_pre_train.dropna()
        
        inpt_pre_train = data_pre_train_dropped.values
        
        if self._dataset.all_columns_are_targets:
            oupt_pre_train = data_pre_train_dropped.values
        else:
            oupt_pre_train = data_pre_train_dropped[self._dataset.target_series].values
        
        assert inpt_pre_train.shape[1] == self.input_dim, \
            f"Input dimension mismatch: {inpt_pre_train.shape[1]} != {self.input_dim}"
        assert oupt_pre_train.shape[1] == self.output_dim, \
            f"Output dimension mismatch: {oupt_pre_train.shape[1]} != {self.output_dim}"
        
        inpt_tensor = torch.from_numpy(inpt_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        oupt_tensor = torch.from_numpy(oupt_pre_train).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        self.autoencoder.train()
        final_epoch_losses = []
        for epoch in range(pretrain_epochs):
            epoch_losses = []
            for i in range(0, len(inpt_tensor), self.window_size):
                batch_input = inpt_tensor[i:i+self.window_size]
                batch_target = oupt_tensor[i:i+self.window_size]
                
                self.optimizer.zero_grad()
                pred = self.autoencoder(batch_input)
                
                if use_mse_loss:
                    loss = torch.nn.functional.mse_loss(pred, batch_target, reduction='mean')
                else:
                    mask = ~torch.isnan(batch_target)
                    y_actual_ = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                    y_predicted_ = pred * mask.float()
                    loss = torch.nn.functional.mse_loss(y_predicted_, y_actual_, reduction='mean')
                
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            
            if epoch_losses:
                final_epoch_losses.append(np.mean(epoch_losses))
        
        if final_epoch_losses:
            _logger.info(f'Pre-training completed: final loss={final_epoch_losses[-1]:.{DEFAULT_LOSS_LOG_PRECISION}f}')
        
        self.data = self._dataset.data.copy()
        self.data_denoised = self.data.copy()
        self.missing_mask = self.data.isna().values
        self.target_indices = self._dataset.target_indices
        if not self._dataset.all_columns_are_targets:
            self._target_col_tensor = torch.tensor(self.target_indices, device=self.device, dtype=torch.long)
        self.rng = np.random.RandomState(self.initializer_seed)
        prediction_prev_iter, prediction_prev_iter_full = self._initialize_mcmc_state()
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
            # For all-targets case (exchange rate), use eps directly like TensorFlow
            # For covariates case, expand eps to full series shape
            if self._dataset.all_columns_are_targets:
                self.data_denoised.values[self.lags_input+1:] = self.data_imputed.values[self.lags_input+1:] - self.eps[:-1, :] @ Phi
            else:
                eps_expanded = self._expand_to_full_series_shape(self.eps)
                self.data_denoised.values[self.lags_input+1:] = self.data_imputed.values[self.lags_input+1:] - eps_expanded[:-1, :] @ Phi
            self.data_denoised_interpolated = self._interpolate_dataframe(self.data_denoised)
            
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
                
                predictions_full_tensor = torch.stack([self.decoder(f) for f in factors_list], dim=0)
                
                # Validate MC sample dimension
                if factors_tensor.shape[0] != self.n_mc_samples:
                    raise ValueError(
                        f"MC samples dimension mismatch: factors_tensor.shape[0]={factors_tensor.shape[0]} != n_mc_samples={self.n_mc_samples}"
                    )
                if predictions_full_tensor.shape[0] != self.n_mc_samples:
                    raise ValueError(
                        f"MC samples dimension mismatch: predictions_full_tensor.shape[0]={predictions_full_tensor.shape[0]} != n_mc_samples={self.n_mc_samples}"
                    )
                
                prediction_iter_full_tensor, prediction_iter_std_tensor = self._compute_tensor_stats(predictions_full_tensor)
                prediction_iter_target_tensor = self._extract_target_predictions(prediction_iter_full_tensor)
                prediction_iter_full = to_numpy(prediction_iter_full_tensor)
                prediction_iter_target = to_numpy(prediction_iter_target_tensor)
                prediction_iter_std = to_numpy(prediction_iter_std_tensor)
                self.factors = to_numpy(factors_tensor)
                self.prediction_std = prediction_iter_std
                
                _, factors_std_tensor = self._compute_tensor_stats(factors_tensor)
                self.factor_std = to_numpy(factors_std_tensor)
                
                prediction_std_mean_check = self._compute_variance_mean(prediction_iter_std)
                should_check_variance = (
                    (prediction_std_mean_check is not None and prediction_std_mean_check < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD) or
                    (self._num_iter % self.disp == 0)
                )
                if should_check_variance:
                    factors_mean_tensor, _ = self._compute_tensor_stats(factors_tensor)
                    factors_mean = to_numpy(factors_mean_tensor)
                    variance_diagnostics = self._diagnose_variance_collapse(
                        prediction_std=prediction_iter_std,
                        prediction_mean=prediction_iter_full,
                        factors_mean=factors_mean,
                        factors_std=self.factor_std
                    )
                    if variance_diagnostics['variance_collapse_detected']:
                        _logger.warning(f"Variance collapse detected at iteration {self._num_iter}: {', '.join(variance_diagnostics['warnings'])}")
            
            self._update_imputed_and_eps(prediction_iter_full)
            
            if self._num_iter > 1:
                prediction_prev, prediction_iter = self._select_convergence_predictions(
                    prediction_prev_iter_full, prediction_prev_iter,
                    prediction_iter_full, prediction_iter_target
                )
                delta, self.loss_now = convergence_checker(prediction_prev, prediction_iter, self.y_actual)
                
                _logger.info(f'iteration: {self._num_iter} - delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f} - loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                
                if self._num_iter % self.disp == 0:
                    prediction_std_mean = self._compute_variance_mean(self.prediction_std)
                    factor_std_mean = self._compute_variance_mean(self.factor_std) if self.factor_std is not None else None
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
            
            prediction_prev_iter, prediction_prev_iter_full = self._update_previous_predictions(
                prediction_iter_target, prediction_iter_full
            )
            
            if self.decoder_type == "mlp":
                self._last_iter_datasets = autoencoder_datasets
            
            self._num_iter += 1
        
        # Extract last neurons (for MLP decoder: second-to-last layer output)
        if self.decoder_type == "linear":
            self.last_neurons = self.factors
        else:
            decoder_intermediate = self._get_decoder_intermediate()
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
        
        linear_layer = self._get_decoder_last_linear_layer()
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
        if getattr(self, '_state_space_F', None) is not None:
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
        x_sm = Z @ H.T  # (T, num_target_series)
        
        # Expand to full data shape for compatibility
        x_sm_full = self._expand_to_full_series_shape(x_sm)
        
        # Get target scaler from dataset
        target_scaler = self._get_target_scaler()
        
        # Create result
        # For DDFM: A = F (transition), C = H (observation), r = [num_factors], p = 1
        # Z already computed above, reuse it instead of calling _get_averaged_factors() again
        num_factors = Z.shape[1]
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
        
        if self.factors.ndim == 3:
            n_mc_samples = self.factors.shape[0]
            new_factors_expanded = np.expand_dims(new_factors_np, axis=0)
            new_factors_expanded = np.repeat(new_factors_expanded, n_mc_samples, axis=0)
            self.factors = np.concatenate([self.factors, new_factors_expanded], axis=1)
        else:
            self.factors = np.vstack([self.factors, new_factors_np])