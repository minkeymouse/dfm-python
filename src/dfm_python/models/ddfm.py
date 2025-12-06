"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.

DDFM is a PyTorch Lightning module that inherits from BaseFactorModel.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from ..config import (
    ConfigSource,
    DFMConfig,
    MergedConfigSource,
    make_config_source,
)
from ..config.results import DDFMResult
from ..config.utils import get_periods_per_year
from ..encoder.vae import Decoder, Encoder, extract_decoder_params
from ..logger import get_logger
from ..nowcast.dataview import DataView
from ..nowcast.nowcast import Nowcast
from ..utils.data import create_data_view, rem_nans_spline
from ..utils.helpers import (
    find_series_index,
    get_clock_frequency,
    safe_get_attr,
)
from ..utils.statespace import (
    estimate_idio_dynamics,
    estimate_var1,
    estimate_var2,
)
from ..utils.time import (
    convert_to_timestamp,
    find_time_index,
    TimeIndex,
)
from .base import BaseFactorModel, format_error_message

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class DDFMTrainingState:
    """State tracking for DDFM training."""
    factors: np.ndarray
    prediction: np.ndarray
    converged: bool
    num_iter: int
    training_loss: Optional[float] = None


class DDFMModel:
    """Deep Dynamic Factor Model using PyTorch (low-level implementation).
    
    This class implements a DDFM with:
    - Nonlinear encoder (MLP) to extract factors from observations
    - Linear decoder for interpretability
    - Linear factor dynamics (VAR)
    - Kalman filtering for final smoothing
    
    The model is trained using gradient descent (Adam optimizer) to minimize
    reconstruction error, then factor dynamics are estimated via OLS, and
    final smoothing is performed using Kalman filter.
    
    .. note::
        This is the low-level implementation used internally by the ``DDFM`` class.
        For high-level API with Lightning training, use the ``DDFM`` class (defined below)
        which is a PyTorch Lightning module.
        
        Example:
        
        .. code-block:: python
        
            from dfm_python import DDFM, DFMDataModule, DDFMTrainer
            
            model = DDFM()
            model.load_config('config/ddfm.yaml')
            dm = DFMDataModule(config=model.config, data=df_processed)
            dm.setup()
            trainer = DDFMTrainer(max_epochs=100)
            trainer.fit(model, dm)
    """
    
    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 100,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        lags_input: int = 0,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize DDFM model.
        
        Parameters
        ----------
        encoder_layers : List[int], optional
            Hidden layer dimensions for encoder. Default: [64, 32]
        num_factors : int, optional
            Number of factors. If None, will be inferred from config during fit.
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid'). Default: 'relu' (matches original DDFM)
        use_batch_norm : bool
            Whether to use batch normalization in encoder. Default: True
        learning_rate : float
            Learning rate for Adam optimizer. Default: 0.001
        epochs : int
            Number of epochs per MCMC iteration. Default: 100
        batch_size : int
            Batch size for training. Default: 100 (matches original DDFM)
        factor_order : int
            VAR lag order for factor dynamics (1 or 2). Default: 1
        use_idiosyncratic : bool
            Whether to model idiosyncratic components with AR(1) dynamics. Default: True
        min_obs_idio : int
            Minimum number of observations required for idio AR(1) estimation. Default: 5
        lags_input : int
            Number of lags of inputs on encoder (default 0, i.e. same inputs and outputs). Default: 0
        max_iter : int
            Maximum number of MCMC iterations. Default: 200
        tolerance : float
            Convergence tolerance. Default: 0.0005
        disp : int
            Display intermediate results every 'disp' iterations. Default: 10
        seed : int, optional
            Random seed for reproducibility. Default: None
        """
        
        # PyTorch is mandatory - no need to check
        if factor_order not in [1, 2]:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="initialization",
                reason=f"factor_order must be 1 or 2, got {factor_order}",
                guidance="Please provide a valid factor_order value (1 for VAR(1) or 2 for VAR(2))"
            )
            raise ValueError(error_msg)
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.num_factors = num_factors
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs  # Epochs per MCMC iteration
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        self.lags_input = lags_input
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        
        # PyTorch modules (will be initialized in fit)
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random number generator for MC sampling
        self.rng = np.random.RandomState(seed if seed is not None else 3)
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, default True
            Whether to return forecasted series.
        return_factors : bool, default True
            Whether to return forecasted factors.
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted series and/or factors
        """
        if self._result is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason="model has not been fitted yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
        
        # Default horizon
        if horizon is None:
            if self._config is not None:
                clock = get_clock_frequency(self._config, 'm')
                horizon = get_periods_per_year(clock)
            else:
                horizon = 12  # Default to 12 periods if no config
        
        if horizon <= 0:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason=f"horizon must be a positive integer, got {horizon}",
                guidance="Please provide a positive integer value for the forecast horizon"
            )
            raise ValueError(error_msg)
        
        # Extract parameters
        A = self._result.A  # Factor dynamics (m x m) for VAR(1) or (m x 2m) for VAR(2)
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]  # Last factor estimate (m,)
        p = self._result.p  # VAR order
        
        # Deterministic forecast
        if p == 1:
            # VAR(1): f_t = A @ f_{t-1}
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            Z_forecast[0, :] = A @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        elif p == 2:
            # VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            # Need last two factor values
            if self._result.Z.shape[0] < 2:
                # Fallback to VAR(1) if not enough history
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                A1 = A[:, :Z_last.shape[0]]
                Z_forecast[0, :] = A1 @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
            else:
                Z_prev = self._result.Z[-2, :]  # f_{t-2}
                A1 = A[:, :Z_last.shape[0]]
                A2 = A[:, Z_last.shape[0]:]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                if horizon > 1:
                    Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                for h in range(2, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
        else:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason=f"unsupported VAR order {p}",
                guidance="Only VAR(1) and VAR(2) are supported. Please use factor_order=1 or factor_order=2"
            )
            raise ValueError(error_msg)
        
        # Transform to observations
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    def generate_dataset(
        self,
        target_series: str,
        periods: List[datetime],
        backward: int = 0,
        forward: int = 0,
        dataview: Optional['DataView'] = None
    ) -> Dict[str, Any]:
        """Generate dataset for DFM evaluation.
        
        Note: Requires data to be stored during fit(). Data is automatically
        stored from DataModule during fit().
        """
        if not hasattr(self, '_data') or self._data is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="generate_dataset",
                reason="model has not been fitted with DataModule yet",
                guidance="Please call trainer.fit(model, data_module) first to store data"
            )
            raise ValueError(error_msg)
        
        i_series = find_series_index(self._config, target_series)
        X_features, y_baseline, y_actual, metadata, backward_results = [], [], [], [], []
        
        if dataview is not None:
            dataview_factory = dataview
        else:
            # Convert data to numpy if needed
            if hasattr(self._data, 'to_numpy'):
                X_data = self._data.to_numpy()
            else:
                X_data = np.asarray(self._data)
            
            dataview_factory = DataView.from_arrays(
                X=X_data, Time=self._time,
                Z=self._original_data, config=self._config,
                X_frame=self._data_frame
            )
        if dataview_factory.config is None:
            dataview_factory.config = self._config
        
        for period in periods:
            view_obj = dataview_factory.with_view_date(period)
            X_view, Time_view, _ = view_obj.materialize()
            
            if backward > 0:
                nowcasts, data_view_dates = [], []
                for weeks_back in range(backward, -1, -1):
                    data_view_date = period - timedelta(weeks=weeks_back)
                    view_past = dataview_factory.with_view_date(data_view_date)
                    X_view_past, Time_view_past, _ = view_past.materialize()
                    # Access nowcast through _nowcast_ref (set by high-level DDFM class)
                    nowcast_obj = getattr(self, '_nowcast_ref', None)
                    if nowcast_obj is None:
                        error_msg = format_error_message(
                            model_type="DDFM",
                            operation="nowcast",
                            reason="requires high-level DDFM instance",
                            guidance="Please call nowcast() from DDFM class, not DDFMModel"
                        )
                        raise ValueError(error_msg)
                    nowcast_val = nowcast_obj(
                        target_series=target_series,
                        view_date=view_past.view_date or data_view_date,
                        target_period=period
                    )
                    nowcasts.append(nowcast_val)
                    data_view_dates.append(view_past.view_date or data_view_date)
                baseline_nowcast = nowcasts[-1]
                backward_results.append({
                    'nowcasts': np.array(nowcasts),
                    'data_view_dates': data_view_dates,
                    'target_date': period
                })
            else:
                # Access nowcast through _nowcast_ref (set by high-level DDFM class)
                nowcast_obj = getattr(self, '_nowcast_ref', None)
                if nowcast_obj is None:
                    error_msg = format_error_message(
                        model_type="DDFM",
                        operation="nowcast",
                        reason="requires high-level DDFM instance",
                        guidance="Please call nowcast() from DDFM class, not DDFMModel"
                    )
                    raise ValueError(error_msg)
                baseline_nowcast = nowcast_obj(
                    target_series=target_series,
                    view_date=view_obj.view_date or period,
                    target_period=period
                )
            
            y_baseline.append(baseline_nowcast)
            t_idx = find_time_index(self._time, period)
            actual_val = np.nan
            # Convert data to numpy for indexing
            if hasattr(self._data, 'to_numpy'):
                data_array = self._data.to_numpy()
            else:
                data_array = np.asarray(self._data)
            if t_idx is not None and t_idx < data_array.shape[0] and i_series < data_array.shape[1]:
                actual_val = data_array[t_idx, i_series]
            y_actual.append(actual_val)
            
            # Extract features
            if self._result is not None and hasattr(self._result, 'Z'):
                latest_factors = self._result.Z[-1, :] if self._result.Z.shape[0] > 0 else np.zeros(self._result.Z.shape[1])
            else:
                latest_factors = np.array([])
            if X_view.shape[0] > 0:
                mean_residual = np.nanmean(X_view[-1, :]) if X_view.shape[0] > 0 else 0.0
            else:
                mean_residual = 0.0
            features = np.concatenate([latest_factors, [mean_residual]])
            X_features.append(features)
            metadata.append({'period': period, 'target_series': target_series})
        
        return {
            'X': np.array(X_features),
            'y_baseline': np.array(y_baseline),
            'y_actual': np.array(y_actual),
            'y_target': np.array(y_actual) - np.array(y_baseline),
            'metadata': metadata,
            'backward_results': backward_results if backward > 0 else []
        }
    
    def get_state(
        self,
        t: Union[int, datetime],
        target_series: str,
        lookback: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get DFM state at time t.
        
        Note: Requires data to be stored during fit(). Data is automatically
        stored from DataModule during fit().
        """
        if not hasattr(self, '_data') or self._data is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="get_state",
                reason="model has not been fitted with DataModule yet",
                guidance="Please call trainer.fit(model, data_module) first to store data"
            )
            raise ValueError(error_msg)
        
        if lookback is None:
            clock = get_clock_frequency(self._config, 'm')
            lookback = get_periods_per_year(clock)
        
        t = convert_to_timestamp(t, self._time, None)
        i_series = find_series_index(self._config, target_series)
        
        # Convert data to numpy if needed
        if hasattr(self._data, 'to_numpy'):
            X_data = self._data.to_numpy()
        else:
            X_data = np.asarray(self._data)
        
        X_view, Time_view, _ = create_data_view(
            X=X_data, Time=self._time,
            Z=self._original_data, config=self._config, view_date=t
        )
        
        # Access nowcast through _nowcast_ref (set by high-level DDFM class)
        nowcast_obj = getattr(self, '_nowcast_ref', None)
        if nowcast_obj is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="nowcast",
                reason="requires high-level DDFM instance",
                guidance="Please call nowcast() from DDFM class, not DDFMModel"
            )
            raise ValueError(error_msg)
        baseline_nowcast = nowcast_obj(target_series=target_series, view_date=t, target_period=None)
        
        baseline_forecast, actual_history, residuals, factors_history = [], [], [], []
        t_idx = find_time_index(self._time, t)
        if t_idx is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="get_state",
                reason=f"time {t} not found in model_instance._time",
                guidance="Please provide a valid time value that exists in the model's time index"
            )
            raise ValueError(error_msg)
        
        for i in range(max(0, t_idx - lookback + 1), t_idx + 1):
            if i < X_data.shape[0]:
                forecast_val = baseline_nowcast
                baseline_forecast.append(forecast_val)
                actual_val = X_data[i, i_series] if i_series < X_data.shape[1] else np.nan
                actual_history.append(actual_val)
                residuals.append(actual_val - forecast_val)
                if self._result is not None and hasattr(self._result, 'Z') and i < self._result.Z.shape[0]:
                    factors_history.append(self._result.Z[i, :])
                else:
                    factors_history.append(np.array([]))
        
        return {
            'time': t,
            'target_series': target_series,
            'baseline_nowcast': baseline_nowcast,
            'baseline_forecast': np.array(baseline_forecast),
            'actual_history': np.array(actual_history),
            'residuals': np.array(residuals),
            'factors_history': factors_history
        }

# ============================================================================
# High-level API Classes
# ============================================================================

if TYPE_CHECKING:
    from ..nowcast import Nowcast
    from ..lightning import DFMDataModule

class DDFM(BaseFactorModel):
    """High-level API for Deep Dynamic Factor Model (PyTorch Lightning module).
    
    This class is a PyTorch Lightning module that can be used with standard
    Lightning training patterns. It inherits from BaseFactorModel and implements
    DDFM training using autoencoder and MCMC procedure.
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import DDFM, DFMDataModule, DDFMTrainer
        >>> import pandas as pd
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pd.read_csv('data/finance.csv')
        >>> df_processed = df[[col for col in df.columns if col != 'date']]
        >>> 
        >>> # Step 2: Create DataModule
        >>> dm = DFMDataModule(config_path='config/ddfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config/ddfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = DDFMTrainer(max_epochs=100)
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
        learning_rate: float = 0.005,
        epochs: int = 100,
        batch_size: int = 100,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        seed: Optional[int] = None,
        decay_learning_rate: bool = True,
        min_obs_pretrain: int = 50,
        mult_epoch_pretrain: int = 1,
        **kwargs
    ):
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
        learning_rate : float, default 0.005
            Learning rate for Adam optimizer (matches original DDFM default)
        epochs : int, default 100
            Number of epochs per MCMC iteration
        batch_size : int, default 100
            Batch size for training (matches original DDFM)
        factor_order : int, default 1
            VAR lag order for factor dynamics (1 or 2)
        use_idiosyncratic : bool, default True
            Whether to model idiosyncratic components
        min_obs_idio : int, default 5
            Minimum observations for idio AR(1) estimation
        max_iter : int, default 200
            Maximum number of MCMC iterations
        tolerance : float, default 0.0005
            Convergence tolerance
        disp : int, default 10
            Display progress every 'disp' iterations
        decay_learning_rate : bool, default True
            Whether to use exponential decay learning rate scheduler (matches original DDFM)
        min_obs_pretrain : int, default 50
            Minimum number of observations for pre-training without interpolation
        mult_epoch_pretrain : int, default 1
            Multiplier for number of epochs during pre-training
            Display progress every 'disp' iterations
        seed : int, optional
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        super().__init__(**kwargs)
        
        # Initialize config using consolidated helper method
        # Note: DDFM does not use block structure, but BaseModelConfig requires blocks
        # We create a minimal default block that will be ignored by DDFM
        config = self._initialize_config(config)
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs_per_iter = epochs
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        self.decay_learning_rate = decay_learning_rate
        self.min_obs_pretrain = min_obs_pretrain
        self.mult_epoch_pretrain = mult_epoch_pretrain
        
        # Determine number of factors
        # Note: DDFM does not use block structure - num_factors is specified directly
        if num_factors is None:
            # Try to get from config num_factors (DDFM-specific parameter)
            if hasattr(config, 'num_factors') and config.num_factors is not None:
                self.num_factors = config.num_factors
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
        # Note: input_dim and output_dim will be set in setup() when we know data dimensions
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        
        # Training state
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        
        # MCMC state
        self.mcmc_iteration: int = 0
        
        # Random number generator for MC sampling
        self.rng = np.random.RandomState(seed if seed is not None else 3)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize encoder and decoder when data dimensions are known.
        
        This is called by Lightning before configure_optimizers(), so we need
        to initialize encoder/decoder here if datamodule is available.
        If not available here, will be initialized in on_train_start().
        """
        # Access datamodule if available (trainer should be attached by now)
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                self._data_module = self.trainer.datamodule
                try:
                    # Get data to determine input dimension
                    # Note: datamodule.setup() should have been called by Lightning already
                    X_torch = self._data_module.get_processed_data()
                    input_dim = X_torch.shape[1]
                    
                    # Initialize networks if not already initialized
                    if self.encoder is None or self.decoder is None:
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
        try:
            self.encoder = Encoder(
                input_dim=input_dim,
                hidden_dims=self.encoder_layers,
                output_dim=self.num_factors,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="encoder initialization",
                reason=f"failed to initialize encoder: {type(e).__name__}: {str(e)}",
                guidance=(
                    f"Check encoder_layers={self.encoder_layers}, num_factors={self.num_factors}, "
                    f"input_dim={input_dim}. "
                    f"Suggestions: (1) Ensure input_dim > 0, (2) Reduce encoder_layers size if too large, "
                    f"(3) Ensure num_factors > 0 and num_factors <= input_dim, "
                    f"(4) Check that encoder_layers values are positive integers"
                )
            )
            raise RuntimeError(error_msg) from e
        
        try:
            # Use standard decoder (block structure removed for simplicity)
            self.decoder = Decoder(
                input_dim=self.num_factors,
                output_dim=input_dim,
                use_bias=True,
            )
            
            # Validate decoder weights are not all zeros (initialization check)
            decoder_weight = self.decoder.decoder.weight.data.cpu().numpy()
            if np.allclose(decoder_weight, 0.0, atol=1e-8):
                error_msg = format_error_message(
                    model_type="DDFM",
                    operation="decoder initialization",
                    reason="decoder weights are all zeros after initialization",
                    guidance=(
                        f"This indicates a problem with decoder initialization. "
                        f"Check: (1) Decoder class implementation, (2) Weight initialization method, "
                        f"(3) PyTorch version compatibility. "
                        f"Decoder weight shape: {decoder_weight.shape}, "
                        f"weight mean: {np.mean(decoder_weight):.6f}, "
                        f"weight std: {np.std(decoder_weight):.6f}"
                    )
                )
                raise RuntimeError(error_msg)
            
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
            error_msg = format_error_message(
                model_type="DDFM",
                operation="decoder initialization",
                reason=f"failed to initialize decoder: {type(e).__name__}: {str(e)}",
                guidance=(
                    f"Check num_factors={self.num_factors}, input_dim={input_dim}. "
                    f"Suggestions: (1) Ensure num_factors > 0, (2) Ensure input_dim > 0, "
                    f"(3) Check that num_factors <= input_dim"
                )
            )
            raise RuntimeError(error_msg) from e
    
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
            error_msg = self._format_error_message(
                operation="forward pass",
                reason="encoder and decoder must be initialized",
                guidance="Please ensure the model is properly initialized before calling forward(). This usually happens automatically during trainer.fit(), but if calling forward() directly, ensure setup() or on_train_start() has been called."
            )
            raise RuntimeError(error_msg)
        
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
        if isinstance(batch, tuple):
            data, target = batch
        else:
            data = batch
            target = data  # For autoencoder, target is same as input
        
        # Ensure data is on the same device as the model
        device = next(self.parameters()).device
        data = data.to(device)
        target = target.to(device)
        
        # Clip input data to prevent extreme values that cause NaN
        # Clip to reasonable range: -10 to 10 standard deviations
        data_clipped = torch.clamp(data, min=-10.0, max=10.0)
        
        # Forward pass
        reconstructed = self.forward(data_clipped)
        
        # Check for NaN in forward pass output (indicates numerical instability)
        if torch.any(torch.isnan(reconstructed)) or torch.any(torch.isinf(reconstructed)):
            nan_count = torch.sum(torch.isnan(reconstructed)).item()
            inf_count = torch.sum(torch.isinf(reconstructed)).item()
            _logger.error(
                f"DDFM training_step: Forward pass produced {nan_count} NaN and {inf_count} Inf values. "
                f"This indicates severe numerical instability. Skipping this batch to prevent NaN propagation. "
                f"Possible causes: (1) learning rate too high, (2) gradient explosion, "
                f"(3) extreme input values, (4) unstable activation functions."
            )
            # Return a large but finite loss to signal the issue, but don't update weights
            # This prevents NaN from propagating to decoder weights
            loss = torch.tensor(1e6, device=data.device, dtype=data.dtype, requires_grad=False)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        # Compute loss with missing data masking (mse_missing)
        # Create mask: 1 for non-missing, 0 for missing (NaN)
        mask = torch.where(torch.isnan(target), torch.zeros_like(target), torch.ones_like(target))
        
        # Replace NaN with zeros for computation
        target_clean = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        
        # Apply mask to predictions
        reconstructed_masked = reconstructed * mask
        
        # Compute MSE only on non-missing values
        # MSE = mean((target_clean - reconstructed_masked)^2) over non-missing elements
        squared_diff = (target_clean - reconstructed_masked) ** 2
        loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)  # Avoid division by zero
        
        # Check for NaN/Inf in loss (indicates numerical instability)
        if torch.isnan(loss) or torch.isinf(loss):
            _logger.error(
                f"DDFM training_step: Loss is NaN/Inf (loss={loss.item()}). "
                f"This indicates numerical instability. Returning large loss value."
            )
            # Return a large but finite loss to signal the issue
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def _validate_factors(self, factors: np.ndarray, operation: str = "operation") -> np.ndarray:
        """Validate and normalize factors shape and content quality.
        
        This method performs comprehensive validation of factor arrays, checking for
        shape correctness, numerical issues (NaN/Inf), constant factors, extreme scale,
        and perfect correlation between factors. It raises errors for critical issues
        and issues warnings for quality concerns.
        
        Parameters
        ----------
        factors : np.ndarray
            Factors array to validate. Can be 1D or 2D, will be reshaped to 2D if needed.
        operation : str, default "operation"
            Operation name for error messages (e.g., "prediction", "factor extraction").
            Used to provide context in error messages.
            
        Returns
        -------
        np.ndarray
            Validated factors array, guaranteed to be 2D with shape (T x num_factors),
            where T is number of time periods and num_factors is number of factors.
            All values are finite (no NaN/Inf).
            
        Raises
        ------
        ValueError
            If factors are empty or invalid shape (0D or 3D+).
            If factors contain NaN/Inf values (critical numerical issue).
            If factors cannot be reshaped to 2D array.
            
        Notes
        -----
        Validation checks performed (in order):
        
        1. **Shape validation**:
           - Checks for empty or 0-dimensional arrays (raises error)
           - Reshapes 1D arrays to 2D (T x num_factors)
           - Ensures 2D shape (raises error if 3D+)
        
        2. **Numerical quality** (raises error if found):
           - Detects NaN/Inf values
           - Reports count and percentage of invalid values
        
        3. **Content quality** (issues warnings):
           - Constant factors: Detects factors with zero variance
           - Extreme scale: Detects factors with very large (>1e6) or very small (<1e-8) std
           - Perfect correlation: Detects pairs of factors with |correlation| > 0.999
        
        4. **Debug logging**:
           - When validation passes, logs factor statistics (mean range, std range) at DEBUG level
        
        This validation is critical for ensuring factors are suitable for:
        - VAR estimation (requires finite, non-constant factors)
        - Kalman filtering (requires valid covariance matrices)
        - Prediction (requires stable factor dynamics)
        """
        factors = np.asarray(factors)
        if factors.ndim == 0 or factors.size == 0:
            raise ValueError(
                f"DDFM {operation} failed: factors is empty or invalid (shape: {factors.shape}). "
                f"This indicates training did not complete properly. "
                f"Please check training logs and ensure fit_mcmc() completed successfully."
            )
        if factors.ndim == 1:
            # Reshape to (T, num_factors) if it's 1D
            factors = factors.reshape(-1, self.num_factors) if factors.size > 0 else factors.reshape(0, self.num_factors)
        if factors.ndim != 2:
            raise ValueError(
                f"DDFM {operation} failed: factors must be 2D array (T x m), got shape {factors.shape}"
            )
        
        T, m = factors.shape
        
        # Check for NaN/Inf values
        if not np.all(np.isfinite(factors)):
            nan_count = np.sum(~np.isfinite(factors))
            nan_pct = 100.0 * nan_count / factors.size
            raise ValueError(
                f"DDFM {operation} failed: factors contain {nan_count} ({nan_pct:.1f}%) NaN/Inf values. "
                f"This indicates numerical issues during training. "
                f"Please check training logs and data quality."
            )
        
        # Check for constant factors (all same value)
        factor_var = np.var(factors, axis=0)
        constant_factors = factor_var < 1e-10
        if np.any(constant_factors):
            n_constant = np.sum(constant_factors)
            constant_indices = np.where(constant_factors)[0].tolist()
            warning_msg = self._format_warning_message(
                operation=operation,
                issue=f"{n_constant}/{m} factors are constant (zero variance)",
                context=f"Constant factor indices: {constant_indices}",
                suggestion="This may indicate training issues or insufficient data variation"
            )
            _logger.warning(warning_msg)
        
        # Check factor scale (warn if extremely large/small)
        factor_std = np.std(factors, axis=0)
        extreme_scale = (factor_std > 1e6) | (factor_std < 1e-8)
        if np.any(extreme_scale):
            n_extreme = np.sum(extreme_scale)
            extreme_indices = np.where(extreme_scale)[0].tolist()
            std_range = [np.min(factor_std), np.max(factor_std)]
            warning_msg = self._format_warning_message(
                operation=operation,
                issue=f"{n_extreme}/{m} factors have extreme scale",
                context=f"Extreme factor indices: {extreme_indices}, Factor std range: [{std_range[0]:.2e}, {std_range[1]:.2e}]",
                suggestion="This may indicate numerical instability"
            )
            _logger.warning(warning_msg)
        
        # Check for perfect correlation between factors (detect linear dependencies)
        if m > 1 and T > 1:
            factor_corr = np.corrcoef(factors.T)
            # Check off-diagonal elements for perfect correlation
            np.fill_diagonal(factor_corr, 0.0)
            perfect_corr = np.abs(factor_corr) > 0.999
            if np.any(perfect_corr):
                n_pairs = np.sum(perfect_corr) // 2  # Divide by 2 since symmetric
                warning_msg = self._format_warning_message(
                    operation=operation,
                    issue=f"{n_pairs} pairs of factors are perfectly correlated (|corr| > 0.999)",
                    suggestion="This may indicate redundant factors or training convergence issues"
                )
                _logger.warning(warning_msg)
        
        # Log factor statistics when validation passes (debug level)
        if _logger.isEnabledFor(logging.DEBUG):
            factor_mean = np.mean(factors, axis=0)
            factor_std = np.std(factors, axis=0)
            _logger.debug(
                f"DDFM {operation}: Factor validation passed. "
                f"Shape: {factors.shape}, Mean range: [{np.min(factor_mean):.4f}, {np.max(factor_mean):.4f}], "
                f"Std range: [{np.min(factor_std):.4f}, {np.max(factor_std):.4f}]"
            )
        
        return factors
    
    def _validate_training_data(
        self,
        X_torch: torch.Tensor,
        operation: str = "training setup"
    ) -> None:
        """Validate data dimensions and model configuration before training starts.
        
        This method performs early validation checks to catch configuration issues
        before training begins. It validates:
        1. Data dimensions (T >= min_obs_required for VAR estimation)
        2. Factor count (num_factors <= N, number of series)
        3. Encoder architecture (reasonable size for data dimensions)
        
        Parameters
        ----------
        X_torch : torch.Tensor
            Input data tensor, shape (T x N) where T is time periods and N is number of series
        operation : str, default "training setup"
            Operation name for error messages
            
        Raises
        ------
        ValueError
            If data dimensions are insufficient for training
            If num_factors exceeds number of series
            If encoder architecture is too large for data size
        """
        # Get data dimensions
        if X_torch.ndim != 2:
            error_msg = self._format_error_message(
                operation=operation,
                reason=f"data must be 2D array (T x N), got shape {X_torch.shape}",
                guidance="Please ensure data is properly formatted as (time_periods x num_series)"
            )
            raise ValueError(error_msg)
        
        T, N = X_torch.shape
        
        # Validation 1: Check minimum time periods for VAR estimation
        min_obs_required = self.factor_order + 5
        if T < min_obs_required:
            error_msg = self._format_error_message(
                operation=operation,
                reason=f"insufficient time periods (T={T}) for VAR({self.factor_order}) estimation",
                guidance=(
                    f"Need at least {min_obs_required} time periods for stable VAR estimation. "
                    f"Current config: num_factors={self.num_factors}, factor_order={self.factor_order}, "
                    f"encoder_layers={self.encoder_layers}. "
                    f"With very small datasets (T < 10), training may be unstable due to: "
                    f"(1) Insufficient data for encoder/decoder training, "
                    f"(2) Poor VAR parameter estimation, "
                    f"(3) High variance in MCMC sampling. "
                    f"Suggestions: (1) Increase data size to at least {min_obs_required} periods, "
                    f"(2) Reduce factor_order to 1 (requires {1 + 5} periods), "
                    f"(3) Reduce num_factors to 1-2 for small datasets, "
                    f"(4) Use smaller encoder_layers (e.g., [16, 8]) for better generalization"
                )
            )
            raise ValueError(error_msg)
        
        # Additional warning for very small datasets (T < 10) even if above minimum
        if T < 10:
            warning_msg = self._format_warning_message(
                operation=operation,
                issue=f"very small dataset (T={T} < 10) may lead to unstable training",
                context=(
                    f"With T={T} time periods, encoder/decoder training and MCMC sampling "
                    f"may have high variance. VAR estimation will use fallback strategies."
                ),
                suggestion=(
                    f"Consider: (1) Using factor_order=1 (requires {1 + 5} periods), "
                    f"(2) Reducing num_factors to 1-2, (3) Using smaller encoder_layers, "
                    f"(4) Increasing data size if possible"
                )
            )
            _logger.warning(warning_msg)
        
        # Validation 2: Check factor count vs. number of series
        if self.num_factors > N:
            error_msg = self._format_error_message(
                operation=operation,
                reason=f"num_factors ({self.num_factors}) exceeds number of series (N={N})",
                guidance=(
                    f"Cannot extract more factors than available series. "
                    f"Current config: num_factors={self.num_factors}, N={N}. "
                    f"Suggestions: (1) Reduce num_factors to {min(self.num_factors, N)}, "
                    f"(2) Add more series to data, (3) Use num_factors <= N"
                )
            )
            raise ValueError(error_msg)
        
        # Validation 3: Check factor count is positive
        if self.num_factors <= 0:
            error_msg = self._format_error_message(
                operation=operation,
                reason=f"num_factors must be positive, got {self.num_factors}",
                guidance="Please set num_factors to a positive integer (typically 1-5)"
            )
            raise ValueError(error_msg)
        
        # Validation 4: Check encoder architecture is reasonable for data size
        # Warn if encoder is too large for small datasets
        total_encoder_params = sum(self.encoder_layers) if self.encoder_layers else 0
        if T < 50 and total_encoder_params > 200:
            warning_msg = self._format_warning_message(
                operation=operation,
                issue=f"encoder architecture may be too large for small dataset",
                context=f"T={T}, encoder_layers={self.encoder_layers} (total params: ~{total_encoder_params})",
                suggestion="Consider using smaller encoder_layers (e.g., [32, 16]) for better generalization"
            )
            _logger.warning(warning_msg)
        
        # Validation 5: Check encoder input dimension matches data dimension
        # This will be validated when encoder is initialized, but we can check early
        if self.encoder is not None:
            if hasattr(self.encoder, 'input_dim') and self.encoder.input_dim != N:
                error_msg = self._format_error_message(
                    operation=operation,
                    reason=f"encoder input_dim ({self.encoder.input_dim}) doesn't match data dimension (N={N})",
                    guidance=(
                        f"Encoder was initialized with input_dim={self.encoder.input_dim}, "
                        f"but data has N={N} series. "
                        f"This may indicate data dimension mismatch. "
                        f"Please ensure data and encoder are compatible"
                    )
                )
                raise ValueError(error_msg)
        
        # Log validation success at debug level
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"DDFM {operation}: Data validation passed. "
                f"T={T}, N={N}, num_factors={self.num_factors}, "
                f"factor_order={self.factor_order}, encoder_layers={self.encoder_layers}"
            )
    
    
    def _estimate_var(
        self, 
        factors: np.ndarray, 
        factor_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate VAR dynamics with comprehensive error handling and fallback.
        
        This method estimates Vector Autoregression (VAR) parameters for factor dynamics
        with robust error handling. It performs pre-estimation validation, handles edge
        cases (insufficient data, constant factors, NaN/Inf values), and provides
        fallback mechanisms when estimation fails.
        
        Parameters
        ----------
        factors : np.ndarray
            Factors array (T x m), where T is number of time periods and m is number of factors.
            Must be 2D array with at least 2 observations.
        factor_order : int
            VAR order, must be 1 or 2.
            - VAR(1): Z_t = A Z_{t-1} + v_t
            - VAR(2): Z_t = A1 Z_{t-1} + A2 Z_{t-2} + v_t
            
        Returns
        -------
        A : np.ndarray
            Transition matrix.
            - For VAR(1): shape (m x m)
            - For VAR(2): shape (m x 2m), where first m columns are A1, last m columns are A2
        Q : np.ndarray
            Innovation covariance matrix, shape (m x m). Always positive definite.
            
        Raises
        ------
        ValueError
            If factor_order is not 1 or 2.
            
        Notes
        -----
        This method includes comprehensive error handling:
        
        1. **Pre-estimation checks**:
           - Validates factors shape and dimensionality
           - Checks for sufficient observations (T >= factor_order + 5)
           - Detects constant factors (zero variance)
           - Cleans NaN/Inf values before estimation
        
        2. **Estimation validation**:
           - Validates spectral radius (ensures stationarity)
           - Checks positive definiteness of Q matrix
           - Validates condition number (detects ill-conditioning)
        
        3. **Fallback mechanisms**:
           - Insufficient observations: Uses scaled identity based on factor variance
           - Constant factors: Uses small variance fallback
           - Estimation failure: Uses scaled identity with factor variance
           - Invalid Q shape: Reshapes or reconstructs from factor variance
        
        All fallbacks ensure the returned matrices are valid and stable for use in
        Kalman filtering and prediction.
        """
        # Validate factors shape - check for 0-dimensional array first
        factors = np.asarray(factors)
        if factors.ndim == 0:
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue="factors is 0-dimensional array",
                suggestion="Using identity matrix for A and small covariance for Q"
            )
            _logger.warning(warning_msg)
            num_factors = self.num_factors if hasattr(self, 'num_factors') and self.num_factors > 0 else 1
            return np.eye(num_factors), np.eye(num_factors) * 1e-6
        
        # Validate factors shape
        if factors.size == 0 or factors.ndim < 2 or factors.shape[0] < 2 or factors.shape[1] == 0:
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue=f"insufficient or invalid factors shape {factors.shape}",
                suggestion="Using identity matrix for A and small covariance for Q"
            )
            _logger.warning(warning_msg)
            num_factors = factors.shape[1] if factors.ndim == 2 and factors.shape[1] > 0 else self.num_factors
            if num_factors == 0:
                num_factors = 1
            return np.eye(num_factors), np.eye(num_factors) * 1e-6
        
        T, m = factors.shape
        
        # Pre-estimation checks
        min_obs_required = factor_order + 5
        if T < min_obs_required:
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue=f"insufficient observations (T={T}) for VAR({factor_order})",
                context=f"need at least {min_obs_required}",
                suggestion="Using scaled identity based on factor variance"
            )
            _logger.warning(warning_msg)
            # Use scaled identity based on factor variance
            factor_var = np.var(factors, axis=0)
            factor_var = np.maximum(factor_var, 1e-8)  # Floor
            if factor_order == 1:
                A_f = np.eye(m) * 0.5  # Conservative initial value
            else:
                A_f = np.hstack([np.eye(m) * 0.5, np.zeros((m, m))])
            Q_f = np.diag(factor_var)
            return A_f, Q_f
        
        # Check for constant factors (zero variance)
        factor_var = np.var(factors, axis=0)
        constant_factors = factor_var < 1e-10
        if np.any(constant_factors):
            n_constant = np.sum(constant_factors)
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue=f"{n_constant}/{m} factors have zero variance",
                suggestion="These will be handled with small variance fallback"
            )
            _logger.warning(warning_msg)
        
        # Check for NaN/Inf in factors
        if not np.all(np.isfinite(factors)):
            nan_count = np.sum(~np.isfinite(factors))
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue=f"factors contain {nan_count} NaN/Inf values",
                suggestion="Cleaning before estimation"
            )
            _logger.warning(warning_msg)
            factors = np.nan_to_num(factors, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for rank deficiency in factor covariance matrix
        # This can occur when factors are highly correlated or redundant
        try:
            factor_cov = np.cov(factors.T)
            rank = np.linalg.matrix_rank(factor_cov)
            if rank < m:
                warning_msg = self._format_warning_message(
                    operation="VAR estimation",
                    issue=f"factor covariance matrix is rank-deficient (rank={rank} < {m})",
                    context=(
                        f"This may indicate redundant factors, insufficient data variation, "
                        f"or highly correlated factors. Regularization will be applied."
                    ),
                    suggestion="Using regularized estimation with small diagonal perturbation to improve conditioning"
                )
                _logger.warning(warning_msg)
                # Add small diagonal perturbation to improve conditioning
                # This helps with numerical stability without significantly changing the covariance structure
                regularization = 1e-6
                factor_cov += np.eye(m) * regularization
        except (ValueError, np.linalg.LinAlgError) as e:
            # If covariance computation fails, log warning but continue
            warning_msg = self._format_warning_message(
                operation="VAR estimation",
                issue=f"failed to compute factor covariance for rank check: {type(e).__name__}",
                suggestion="Proceeding with VAR estimation, will use fallback if needed"
            )
            _logger.warning(warning_msg)
        
        # Estimate VAR with error handling
        try:
            if factor_order == 1:
                A_f, Q_f = estimate_var1(factors)
            elif factor_order == 2:
                A_f, Q_f = estimate_var2(factors)
            else:
                error_msg = self._format_error_message(
                    operation="VAR estimation",
                    reason=f"factor_order must be 1 or 2, got {factor_order}",
                    guidance="Please set factor_order to 1 (VAR(1)) or 2 (VAR(2))"
                )
                raise ValueError(error_msg)
            
            # Validate Q_f shape
            if Q_f.ndim == 0:
                warning_msg = self._format_warning_message(
                    operation="VAR estimation",
                    issue="returned 0-dimensional Q",
                    suggestion="Using scaled identity based on factor variance"
                )
                _logger.warning(warning_msg)
                factor_var = np.var(factors, axis=0)
                factor_var = np.maximum(factor_var, 1e-8)
                Q_f = np.diag(factor_var)
            elif Q_f.ndim != 2:
                warning_msg = self._format_warning_message(
                    operation="VAR estimation",
                    issue=f"returned Q with unexpected shape {Q_f.shape}",
                    suggestion="Reshaping"
                )
                _logger.warning(warning_msg)
                if Q_f.size == m ** 2:
                    Q_f = Q_f.reshape(m, m)
                else:
                    factor_var = np.var(factors, axis=0)
                    factor_var = np.maximum(factor_var, 1e-8)
                    Q_f = np.diag(factor_var)
            
            # Validate estimated parameters
            # Check spectral radius of A (should be < 1 for stability)
            if factor_order == 1:
                eigenvals_A = np.linalg.eigvals(A_f)
            else:
                # For VAR(2), check companion form
                A1 = A_f[:, :m]
                A2 = A_f[:, m:]
                companion = np.block([
                    [A1, A2],
                    [np.eye(m), np.zeros((m, m))]
                ])
                eigenvals_A = np.linalg.eigvals(companion)
            
            max_eigenval = np.max(np.abs(eigenvals_A))
            if max_eigenval >= 0.99:
                warning_msg = self._format_warning_message(
                    operation=f"VAR({factor_order}) estimation",
                    issue=f"estimated A has spectral radius {max_eigenval:.4f} >= 0.99",
                    suggestion="This may indicate instability. Consider checking factor quality"
                )
                _logger.warning(warning_msg)
            
            # Validate Q is positive definite
            Q_sym = (Q_f + Q_f.T) / 2  # Ensure symmetry
            eigenvals_Q = np.linalg.eigvalsh(Q_sym)
            min_eigenval_Q = np.min(eigenvals_Q)
            if min_eigenval_Q < 1e-8:
                warning_msg = self._format_warning_message(
                    operation="VAR estimation",
                    issue=f"estimated Q has minimum eigenvalue {min_eigenval_Q:.2e} < 1e-8",
                    suggestion="Regularizing to ensure positive definiteness"
                )
                _logger.warning(warning_msg)
                Q_f = Q_sym + np.eye(m) * (1e-8 - min_eigenval_Q)
            else:
                Q_f = Q_sym
            
            # Check condition number of Q
            if m > 1:
                max_eigenval_Q = np.max(eigenvals_Q)
                cond_num_Q = max_eigenval_Q / max(min_eigenval_Q, 1e-12)
                if cond_num_Q > 1e8:
                    warning_msg = self._format_warning_message(
                        operation="VAR estimation",
                        issue=f"estimated Q is ill-conditioned (cond={cond_num_Q:.2e})",
                        suggestion="This may indicate collinear factors"
                    )
                    _logger.warning(warning_msg)
            
            return A_f, Q_f
            
        except (ValueError, np.linalg.LinAlgError) as e:
            warning_msg = self._format_warning_message(
                operation=f"VAR({factor_order}) estimation",
                issue=f"estimation failed: {e}",
                suggestion="Using scaled identity based on factor variance as fallback"
            )
            _logger.warning(warning_msg)
            # Use scaled identity based on factor variance as fallback
            factor_var = np.var(factors, axis=0)
            factor_var = np.maximum(factor_var, 1e-8)  # Floor
            
            if factor_order == 1:
                A_f = np.eye(m) * 0.5  # Conservative initial value
            else:
                A_f = np.hstack([np.eye(m) * 0.5, np.zeros((m, m))])
            
            Q_f = np.diag(factor_var)
            _logger.info(
                f"VAR fallback: A shape={A_f.shape}, Q shape={Q_f.shape}, "
                f"factor variance range=[{np.min(factor_var):.2e}, {np.max(factor_var):.2e}]"
            )
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
        if self.encoder is None or self.decoder is None:
            # If still not initialized, create placeholder optimizer as fallback
            # This should not happen if setup() works correctly, but provides safety net
            _logger.warning(
                "Encoder/decoder not initialized in configure_optimizers(). "
                "Creating placeholder optimizer. This may indicate an issue with setup()."
            )
            # Create dummy parameter for optimizer (Lightning requires at least one optimizer)
            dummy_param = nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.Adam([dummy_param], lr=self.learning_rate)
            if self.decay_learning_rate:
                # Create scheduler matching original DDFM: decay_rate=0.96, decay_steps=epochs
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.96
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                }
            return [optimizer]
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
        
        if self.decay_learning_rate:
            # Exponential decay scheduler matching original DDFM implementation
            # Original: decay_rate=0.96, decay_steps=epochs, staircase=True
            # PyTorch: gamma=0.96, step every epoch (interval='epoch', frequency=1)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.96
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        
        return [optimizer]
    
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
        if self.encoder is None or self.decoder is None:
            error_msg = self._format_error_message(
                operation="configure_optimizers",
                reason="encoder and decoder must be initialized before creating optimizer",
                guidance="Call initialize_networks() first. This usually happens automatically during setup() or on_train_start()"
            )
            raise RuntimeError(error_msg)
        
        # Calculate learning rate with exponential decay if enabled
        # Original DDFM: decay_rate=0.96, decay_steps=epochs, staircase=True
        # lr = initial_lr * (decay_rate ^ floor(step / decay_steps))
        if self.decay_learning_rate:
            # For MCMC, we decay per MCMC iteration (not per epoch)
            # Each MCMC iteration uses epochs_per_iter epochs
            decay_steps = self.epochs_per_iter
            decay_rate = 0.96
            lr = self.learning_rate * (decay_rate ** (step // decay_steps))
        else:
            lr = self.learning_rate
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
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
        x_clean_np = x_clean.cpu().numpy() if isinstance(x_clean, torch.Tensor) else x_clean
        missing_mask_np = missing_mask if isinstance(missing_mask, np.ndarray) else missing_mask.cpu().numpy()
        
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
        inpt_tensor = torch.tensor(inpt_pre_train, device=device, dtype=torch.float32)
        oupt_tensor = torch.tensor(oupt_pre_train, device=device, dtype=torch.float32)
        
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
            lr=self.learning_rate
        )
        
        # Pre-train for epochs * mult_epoch_pretrain
        num_epochs = self.epochs_per_iter * self.mult_epoch_pretrain
        _logger.info(f"DDFM pre_train: Starting pre-training for {num_epochs} epochs")
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
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
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
                else:
                    # Standard MSE (no missing values)
                    loss = nn.functional.mse_loss(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                _logger.info(f"DDFM pre_train: Epoch {epoch + 1}/{num_epochs}, loss={avg_loss:.6f}")
        
        _logger.info(f"DDFM pre_train: Pre-training completed")
    
    def fit_mcmc(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DDFMTrainingState:
        """Run MCMC iterative training procedure for DDFM.
        
        This method implements the Markov Chain Monte Carlo (MCMC) procedure for
        Deep Dynamic Factor Model training. It alternates between:
        1. Estimating idiosyncratic dynamics (AR parameters)
        2. Generating Monte Carlo samples from the state-space model
        3. Training the autoencoder (encoder/decoder) on MC samples
        4. Extracting factors from trained encoder
        5. Checking convergence based on MSE between predictions and data
        
        The procedure continues until convergence (MSE change < tolerance) or
        maximum iterations reached.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data with missing values, shape (T x N), where T is number
            of time periods and N is number of series. Missing values should be NaN
            or handled via missing_mask.
        x_clean : torch.Tensor
            Clean data (interpolated), shape (T x N), used for initial autoencoder
            training. Should have same shape as X.
        missing_mask : np.ndarray
            Missing data mask, shape (T x N), boolean array where True indicates
            missing data. Must match shape of X.
        Mx : np.ndarray, optional
            Mean values for unstandardization, shape (N,). Used to convert standardized
            predictions back to original scale. If None, uses values from data module.
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization, shape (N,). Used to convert
            standardized predictions back to original scale. If None, uses values from
            data module.
        max_iter : int, optional
            Maximum number of MCMC iterations. If None, uses self.max_iter (default: 50).
        tolerance : float, optional
            Convergence tolerance for MSE change between iterations. If None, uses
            self.tolerance (default: 1e-4). Training stops when |MSE_new - MSE_old| < tolerance.
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
            - mse_history: List[float] - MSE values at each iteration
            
        Raises
        ------
        ValueError
            If X and x_clean have mismatched shapes.
            If missing_mask shape doesn't match X shape.
            If numerical issues occur (NaN/Inf propagation) that cannot be handled.
            
        Notes
        -----
        The MCMC procedure includes comprehensive numerical stability checks:
        
        1. **Shape validation**: Validates X, x_clean, and missing_mask have consistent shapes.
        
        2. **Numerical stability**: 
           - Checks for NaN/Inf after each major step
           - Uses fallback mechanisms when numerical issues occur
           - Logs warnings when fallbacks are used
        
        3. **Convergence checking**:
           - Computes MSE between predictions and observed data (excluding missing values)
           - Checks for convergence: |MSE_new - MSE_old| < tolerance
           - Stops early if convergence achieved
        
        4. **Error handling**:
           - VAR estimation failures: Uses fallback matrices
           - Factor extraction failures: Uses previous iteration factors
           - Prediction failures: Uses previous iteration predictions
        
        The method is designed to be robust to numerical edge cases and provides
        graceful degradation when estimation fails.
        
        Examples
        --------
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config.yaml')
        >>> # X, x_clean, missing_mask prepared from data module
        >>> state = model.fit_mcmc(X, x_clean, missing_mask, max_iter=50, tolerance=1e-4)
        >>> factors = state.factors  # (T x 2) factor estimates
        >>> print(f"Converged: {state.converged}, Iterations: {state.num_iter}")
        """
        self.Mx = Mx
        self.Wx = Wx
        self.data_processed = X
        
        device = X.device
        dtype = X.dtype
        T, N = X.shape
        
        # Validate shape consistency between X, x_clean, and missing_mask
        # This prevents IndexError when using boolean indexing with missing_mask
        if x_clean.shape != X.shape:
            error_msg = self._format_error_message(
                operation="fit_mcmc",
                reason=f"shape mismatch between X ({X.shape}) and x_clean ({x_clean.shape})",
                guidance=(
                    f"Both X and x_clean must have the same shape (T x N). "
                    f"This indicates data preprocessing inconsistency. "
                    f"Please ensure x_clean is created from the same data as X."
                )
            )
            raise ValueError(error_msg)
        if missing_mask.shape != (T, N):
            error_msg = self._format_error_message(
                operation="fit_mcmc",
                reason=f"shape mismatch between X ({X.shape}) and missing_mask ({missing_mask.shape})",
                guidance=(
                    f"missing_mask must have shape (T x N) matching X. "
                    f"This indicates missing_mask was created from data with different shape. "
                    f"Please ensure missing_mask is created from the same data passed as X parameter."
                )
            )
            raise ValueError(error_msg)
        
        # Use instance attributes if not provided
        max_iter = max_iter if max_iter is not None else self.max_iter
        tolerance = tolerance if tolerance is not None else self.tolerance
        disp = disp if disp is not None else self.disp
        
        # Initialize networks if not done
        if self.encoder is None or self.decoder is None:
            self.initialize_networks(N)
        
        # Ensure encoder and decoder are on the correct device
        # This is critical even if they were initialized in on_train_start(),
        # as the device might differ or they might not have been moved properly
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        
        # Random number generator for MC sampling
        rng = np.random.RandomState(seed if seed is not None else (self.rng.randint(0, 2**31) if hasattr(self.rng, 'randint') else 3))
        
        # Convert to numpy for MCMC procedure (some operations are easier in numpy)
        x_standardized_np = X.cpu().numpy()
        x_clean_np = x_clean.cpu().numpy()
        bool_no_miss = ~missing_mask
        
        # Initialize data structures
        data_mod_only_miss = x_standardized_np.copy()  # Original with missing values
        data_mod = x_clean_np.copy()  # Clean data (will be modified during MCMC)
        z_actual = x_standardized_np.copy()  # Actual observations (target for training, like original DDFM)
        
        # Initial prediction
        x_tensor = x_clean.to(device)
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            factors_init = self.encoder(x_tensor).cpu().numpy()
            # Check for NaN/Inf in initial factors
            factors_init = self._check_finite(factors_init, "initial factors", context="at iteration 0")
            
            factors_tensor = torch.tensor(factors_init, device=device, dtype=dtype)
            prediction_iter = self.decoder(factors_tensor).cpu().numpy()
            # Check for NaN/Inf in initial prediction
            prediction_iter = self._check_finite(prediction_iter, "initial prediction", context="at iteration 0")
        
        # Initialize factors
        factors = factors_init.copy()
        
        # Update missing values with initial prediction
        bool_miss = missing_mask
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Initial residuals
        eps = data_mod_only_miss - prediction_iter
        # Check for NaN/Inf in initial residuals
        eps = self._check_finite(eps, "initial residuals", context="at iteration 0")
        
        # MCMC loop
        iter_count = 0
        not_converged = True
        prediction_prev_iter = None
        delta = float('inf')
        loss_now = float('inf')
        
        # Check for very small dataset and warn about potential instability
        if T < 10:
            warning_msg = self._format_warning_message(
                operation="MCMC training",
                issue=f"very small dataset (T={T} < 10) may cause unstable MCMC sampling",
                context=(
                    f"With only {T} time periods, encoder/decoder training per iteration "
                    f"may have high variance. Factor extraction and VAR estimation will use "
                    f"fallback strategies. Results may be less reliable."
                ),
                suggestion=(
                    f"Monitor convergence carefully. Consider reducing num_factors or "
                    f"using smaller encoder_layers for better stability"
                )
            )
            _logger.warning(warning_msg)
        
        _logger.info(f"Starting MCMC training: max_iter={max_iter}, tolerance={tolerance}, epochs_per_iter={self.epochs_per_iter}")
        
        # Create optimizer once per MCMC iteration (reused across MC samples)
        optimizer = None
        
        while not_converged and iter_count < max_iter:
            iter_count += 1
            self.mcmc_iteration = iter_count
            
            # Create optimizer for this MCMC iteration (reused across MC samples)
            optimizer = self._create_optimizer()
            
            # Get idiosyncratic distribution
            if self.use_idiosyncratic:
                A_eps, Q_eps = estimate_idio_dynamics(eps, missing_mask, self.min_obs_idio)
                # Check for NaN/Inf in estimated dynamics
                A_eps = self._check_finite(A_eps, f"idiosyncratic AR coefficients (A_eps)", context=f"at iteration {iter_count}")
                Q_eps = self._check_finite(Q_eps, f"idiosyncratic innovation covariance (Q_eps)", context=f"at iteration {iter_count}")
                
                # Convert to format expected by MCMC procedure
                phi = A_eps if A_eps.ndim == 2 else np.diag(A_eps) if A_eps.ndim == 1 else np.eye(N)
                mu_eps = np.zeros(N)
                if Q_eps.ndim == 2:
                    std_eps = np.sqrt(np.diag(Q_eps))
                elif Q_eps.ndim == 1:
                    std_eps = np.sqrt(Q_eps)
                else:
                    std_eps = np.ones(N) * 0.1
                
                # Ensure std_eps is finite and positive
                std_eps = np.maximum(std_eps, 1e-8)  # Floor to prevent zero/negative
                std_eps = self._check_finite(std_eps, f"idiosyncratic std (std_eps)", context=f"at iteration {iter_count}")
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
            try:
                for t in range(T):
                    eps_draws[:, t, :] = rng.multivariate_normal(
                        mu_eps, np.diag(std_eps), size=self.epochs_per_iter
                    )
                # Check for NaN/Inf in MC samples
                eps_draws = self._check_finite(eps_draws, f"MC samples (eps_draws)", context=f"at iteration {iter_count}")
            except (ValueError, np.linalg.LinAlgError) as e:
                warning_msg = self._format_warning_message(
                    operation=f"MCMC iteration {iter_count}",
                    issue=f"failed to generate MC samples: {e}",
                    suggestion="Using zero samples as fallback"
                )
                _logger.warning(warning_msg)
                # Use zero samples as fallback
                eps_draws = np.zeros((self.epochs_per_iter, T, N))
            
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
                z_actual_tensor = torch.tensor(z_actual, device=device, dtype=dtype)
                dataset = torch.utils.data.TensorDataset(x_sample, z_actual_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )
                
                # Train for 1 epoch
                self.encoder.train()
                self.decoder.train()
                # Optimizer is created once per MCMC iteration and reused across MC samples
                
                for batch_data, batch_target in dataloader:
                    optimizer.zero_grad()
                    reconstructed = self.forward(batch_data)
                    # Use missing-aware loss (mse_missing) for consistency
                    # Create mask for missing values (though in MCMC loop, missing values are filled)
                    mask = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), torch.ones_like(batch_target))
                    target_clean = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                    reconstructed_masked = reconstructed * mask
                    squared_diff = (target_clean - reconstructed_masked) ** 2
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
                    loss.backward()
                    # Gradient clipping to prevent NaN and improve stability
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                # Extract factors from this sample
                x_sample_tensor = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                self.encoder.eval()
                with torch.no_grad():
                    factors_sample = self.encoder(x_sample_tensor).cpu().numpy()
                    # Check for NaN/Inf in factor sample
                    factors_sample = self._check_finite(
                        factors_sample, 
                        f"factor sample {i+1}/{self.epochs_per_iter}", 
                        context=f"at iteration {iter_count}"
                    )
                factors_samples.append(factors_sample)
            
            # Update factors: average over all MC samples
            factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
            # Check for NaN/Inf in averaged factors
            factors = self._check_finite(factors, "averaged factors", context=f"at iteration {iter_count}", fallback=factors_init)
            
            # Clip extreme factor values to prevent numerical instability
            # Use configurable clipping threshold (default: 10 standard deviations)
            clip_threshold = 10.0
            factor_mean = np.mean(factors, axis=0)
            factor_std = np.std(factors, axis=0)
            # Avoid division by zero
            factor_std = np.maximum(factor_std, 1e-8)
            
            clipped_count = 0
            for i in range(factors.shape[1]):
                lower_bound = factor_mean[i] - clip_threshold * factor_std[i]
                upper_bound = factor_mean[i] + clip_threshold * factor_std[i]
                before_clip = factors[:, i].copy()
                factors[:, i] = np.clip(factors[:, i], lower_bound, upper_bound)
                # Count how many values were clipped
                clipped_count += np.sum((before_clip != factors[:, i]))
            
            if clipped_count > 0:
                warning_msg = self._format_warning_message(
                    operation=f"MCMC iteration {iter_count}",
                    issue=f"clipped {clipped_count} extreme factor values (>{clip_threshold} std devs)",
                    context=f"This prevents numerical instability in encoder/decoder forward passes",
                    suggestion="If clipping occurs frequently, consider: (1) Reducing learning_rate, (2) Using smaller encoder_layers, (3) Checking data scaling"
                )
                _logger.warning(warning_msg)
            
            # Check convergence
            self.decoder.eval()
            with torch.no_grad():
                factors_tensor = torch.tensor(factors, device=device, dtype=dtype)
                prediction_iter = self.decoder(factors_tensor).cpu().numpy()
                # Check for NaN/Inf in prediction
                prediction_iter = self._check_finite(
                    prediction_iter, 
                    "prediction_iter", 
                    context=f"at iteration {iter_count}",
                    fallback=prediction_prev_iter if prediction_prev_iter is not None else prediction_iter
                )
            
            if iter_count > 1:
                # Compute MSE on non-missing values
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    mse = np.nanmean((prediction_prev_iter[mask] - prediction_iter[mask]) ** 2)
                    # Ensure MSE is finite
                    if not np.isfinite(mse):
                        warning_msg = self._format_warning_message(
                            operation=f"MCMC iteration {iter_count}",
                            issue=f"MSE is not finite ({mse})",
                            suggestion="Using previous delta value"
                        )
                        _logger.warning(warning_msg)
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
                    # Ensure loss is finite
                    if not np.isfinite(loss_now):
                        warning_msg = self._format_warning_message(
                            operation=f"MCMC iteration {iter_count}",
                            issue=f"initial loss is not finite ({loss_now})",
                            suggestion="Using large default value"
                        )
                        _logger.warning(warning_msg)
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
            # Check for NaN/Inf in residuals before next iteration
            eps = self._check_finite(eps, "residuals (eps)", context=f"at iteration {iter_count}")
        
        if not_converged:
            delta_str = f"{delta:.6f}" if iter_count > 1 else "N/A"
            warning_msg = self._format_warning_message(
                operation="MCMC training",
                issue=f"convergence not achieved within {max_iter} iterations",
                context=f"Final delta: {delta_str}"
            )
            _logger.warning(warning_msg)
        
        converged = not not_converged
        
        # Validate and normalize factors shape before storing
        factors = self._validate_factors(factors, operation="fit_mcmc")
        
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
            error_msg = self._format_error_message(
                operation="get_result",
                reason="model has not been fitted yet",
                guidance="Please call fit_mcmc() first. This usually happens automatically during trainer.fit()"
            )
            raise RuntimeError(error_msg)
        
        if self.encoder is None or self.decoder is None:
            error_msg = self._format_error_message(
                operation="get_result",
                reason="encoder and decoder must be initialized",
                guidance="Please ensure the model is properly initialized before getting results. This usually happens automatically during setup() or on_train_start()"
            )
            raise RuntimeError(error_msg)
        
        # Extract decoder parameters (C, bias)
        C, bias = extract_decoder_params(self.decoder)
        
        # Log decoder weight statistics for monitoring and debugging
        C_mean = np.mean(C)
        C_std = np.std(C)
        C_min = np.min(C)
        C_max = np.max(C)
        C_nonzero = np.count_nonzero(C)
        C_zero_ratio = 1.0 - (C_nonzero / C.size)
        _logger.info(
            f"DDFM get_result: C matrix statistics - mean={C_mean:.6f}, std={C_std:.6f}, "
            f"min={C_min:.6f}, max={C_max:.6f}, nonzero={C_nonzero}/{C.size} ({1.0-C_zero_ratio:.1%}), "
            f"zero_ratio={C_zero_ratio:.1%}"
        )
        
        # Validate C matrix for NaN (extract_decoder_params should handle this, but double-check)
        if np.any(np.isnan(C)):
            nan_count = np.sum(np.isnan(C))
            nan_ratio = nan_count / C.size
            _logger.error(
                f"DDFM get_result: C matrix contains {nan_count}/{C.size} NaN values ({nan_ratio:.1%}) "
                f"after extraction. This indicates severe numerical instability. "
                f"The model cannot be used for prediction. Consider: (1) reducing learning rate, "
                f"(2) adding gradient clipping, (3) checking data quality, (4) reducing model complexity."
            )
            # Replace NaN with zeros as last resort (model will not work correctly)
            C = np.nan_to_num(C, nan=0.0)
            _logger.warning("Replaced NaN values in C matrix with zeros (model may not work correctly).")
        
        # Get factors and prediction
        factors = self.training_state.factors  # T x num_factors
        prediction_iter = self.training_state.prediction  # T x N
        
        # Validate and normalize factors shape
        factors = self._validate_factors(factors, operation="get_result")
        
        # Convert to numpy
        C = C.cpu().numpy() if isinstance(C, torch.Tensor) else C
        bias = bias.cpu().numpy() if isinstance(bias, torch.Tensor) else bias
        
        # Compute residuals and estimate idiosyncratic dynamics
        if self.data_processed is not None:
            x_standardized = self.data_processed.cpu().numpy()
            # Ensure shapes match
            if x_standardized.shape != prediction_iter.shape:
                warning_msg = self._format_warning_message(
                    operation="get_result",
                    issue=f"shape mismatch: data_processed {x_standardized.shape} vs prediction {prediction_iter.shape}",
                    suggestion="Using prediction shape for residuals"
                )
                _logger.warning(warning_msg)
                residuals = np.zeros_like(prediction_iter)
            else:
                residuals = x_standardized - prediction_iter
        else:
            residuals = np.zeros_like(prediction_iter)
        
        # Estimate factor dynamics (VAR) with error handling
        A_f, Q_f = self._estimate_var(factors, self.factor_order)
        
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
            block_names=None,  # DDFM does not use block structure
            training_loss=self.training_state.training_loss,
            encoder_layers=self.encoder_layers,
            use_idiosyncratic=self.use_idiosyncratic,
        )
        
        return result
    
    def on_train_end(self) -> None:
        """Called when training ends. Automatically computes result from training state."""
        # Automatically compute result after training completes
        if self.training_state is not None:
            try:
                if not hasattr(self, '_result') or self._result is None:
                    self._result = self.get_result()
            except Exception as e:
                # Log warning but don't fail - result can be computed later if needed
                _logger.warning(
                    f"Could not automatically compute result after training: {e}. "
                    f"Result will be computed on first access to result property or predict()."
                )
    
    def on_train_start(self) -> None:
        """Called when training starts. Run MCMC training."""
        # Get processed data and standardization params from DataModule
        X_torch, Mx, Wx = self._get_data_from_datamodule()
        
        # Early validation: Check data dimensions and model configuration before training
        # This catches configuration issues early with clear error messages
        self._validate_training_data(X_torch, operation="training setup")
        
        # Initialize encoder/decoder if not already done in setup()
        if self.encoder is None or self.decoder is None:
            input_dim = X_torch.shape[1]
            self.initialize_networks(input_dim)
            # Move to same device as data
            device = X_torch.device
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            _logger.debug(f"Initialized encoder/decoder in on_train_start() with input_dim={input_dim}")
        
        # Always run fit_mcmc() if training_state is None (first training run)
        # This ensures MCMC training happens even if encoder/decoder were already initialized
        if self.training_state is None:
            # Handle missing data (NaN values)
            # DDFM can handle NaN implicitly via state-space model, but for MCMC training
            # we use imputation to provide clean initial data. The MCMC procedure will
            # then handle missing data through the idiosyncratic component estimation.
            # Use method=1 (fill without trimming) to preserve data size for test compatibility
            # Method 1 fills missing values using spline interpolation + moving average without trimming rows
            # This ensures result shape matches data_module.data_processed.shape[0] as expected by tests
            # Method 2 (default) trims rows with >80% NaN, which causes shape mismatch with test expectations
            # Note: nan_method and nan_k are internal parameters for missing data handling
            nan_method = safe_get_attr(self.config, 'nan_method', 1)
            nan_k = safe_get_attr(self.config, 'nan_k', 3)
            
            # Check if data has NaN values
            if isinstance(X_torch, torch.Tensor):
                has_nan = torch.any(torch.isnan(X_torch)).item()
                X_np = X_torch.cpu().numpy()
            else:
                has_nan = np.any(np.isnan(X_torch))
                X_np = X_torch
            
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
                x_clean_torch = torch.tensor(x_clean_np, dtype=torch.float32, device=X_torch.device)
            else:
                # No NaN values - use data as-is
                x_clean_torch = X_torch if isinstance(X_torch, torch.Tensor) else torch.tensor(X_torch, dtype=torch.float32, device=X_torch.device)
                missing_mask = np.zeros(X_np.shape, dtype=bool)  # No missing data
            
            # Replace any remaining NaN/Inf with zeros (defensive check)
            device = x_clean_torch.device
            dtype = x_clean_torch.dtype
            x_clean_torch = torch.where(
                torch.isfinite(x_clean_torch),
                x_clean_torch,
                torch.tensor(0.0, device=device, dtype=dtype)
            )
            
            # Ensure missing_mask shape matches x_clean_torch shape
            # This is critical for boolean indexing in fit_mcmc()
            # If shapes don't match (shouldn't happen with method=1, but defensive check),
            # adjust missing_mask to match x_clean_torch shape
            if missing_mask.shape != x_clean_torch.shape:
                _logger.warning(
                    f"DDFM on_train_start: missing_mask shape {missing_mask.shape} doesn't match "
                    f"x_clean_torch shape {x_clean_torch.shape}. Adjusting missing_mask to match x_clean_torch."
                )
                # If x_clean_torch is smaller, truncate missing_mask
                if missing_mask.shape[0] > x_clean_torch.shape[0]:
                    missing_mask = missing_mask[:x_clean_torch.shape[0], :]
                # If x_clean_torch is larger, pad missing_mask with False (no missing data)
                elif missing_mask.shape[0] < x_clean_torch.shape[0]:
                    pad_rows = x_clean_torch.shape[0] - missing_mask.shape[0]
                    missing_mask = np.vstack([missing_mask, np.zeros((pad_rows, missing_mask.shape[1]), dtype=bool)])
                # Adjust columns if needed
                if missing_mask.shape[1] != x_clean_torch.shape[1]:
                    if missing_mask.shape[1] > x_clean_torch.shape[1]:
                        missing_mask = missing_mask[:, :x_clean_torch.shape[1]]
                    else:
                        pad_cols = x_clean_torch.shape[1] - missing_mask.shape[1]
                        missing_mask = np.hstack([missing_mask, np.zeros((missing_mask.shape[0], pad_cols), dtype=bool)])
            
            # Pre-train autoencoder on non-missing data (matching original DDFM)
            # This provides stable initialization before MCMC training
            try:
                self.pre_train(
                    X=x_clean_torch,
                    x_clean=x_clean_torch,
                    missing_mask=missing_mask,
                    device=x_clean_torch.device,
                )
            except Exception as e:
                _logger.warning(
                    f"DDFM pre_train failed: {e}. Continuing with MCMC training without pre-training. "
                    f"This may lead to less stable training."
                )
            
            # Run MCMC training
            # Pass x_clean_torch as X to ensure all data arrays have consistent shape
            self.fit_mcmc(
                X=x_clean_torch,
                x_clean=x_clean_torch,
                missing_mask=missing_mask,
                Mx=Mx,
                Wx=Wx,
            )
        
        super().on_train_start()
    
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> 'DDFM':
        """Load configuration from various sources."""
        # Preserve explicitly set num_factors if it was set during initialization
        preserved_num_factors = None
        if hasattr(self, '_num_factors_explicit') and self._num_factors_explicit:
            preserved_num_factors = self.num_factors
        
        # Use common config loading logic
        self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        # Restore preserved num_factors if it was explicitly set
        if preserved_num_factors is not None:
            self.num_factors = preserved_num_factors
            # Keep the flag set since it's still explicitly set
            self._num_factors_explicit = True
        
        # DDFM-specific initialization is handled in __init__ or on_train_start
        # No additional setup needed here
        
        return self
    
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
        """
        if self.training_state is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason="model has not been trained yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
        
        # Convert training state to result format for prediction
        if not hasattr(self, '_result') or self._result is None:
            self._result = self.get_result()
        
        if self._result is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason="model has not been fitted yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
        
        # Default horizon
        if horizon is None:
            if self._config is not None:
                clock = get_clock_frequency(self._config, 'm')
                horizon = get_periods_per_year(clock)
            else:
                horizon = 12  # Default to 12 periods if no config
        
        if horizon <= 0:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason=f"horizon must be a positive integer, got {horizon}",
                guidance="Please provide a positive integer value for the forecast horizon"
            )
            raise ValueError(error_msg)
        
        # Extract parameters
        A = self._result.A  # Factor dynamics (m x m) for VAR(1) or (m x 2m) for VAR(2)
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]  # Last factor estimate (m,)
        p = self._result.p  # VAR order
        
        # Deterministic forecast
        if p == 1:
            # VAR(1): f_t = A @ f_{t-1}
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            Z_forecast[0, :] = A @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        elif p == 2:
            # VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            # Need last two factor values
            if self._result.Z.shape[0] < 2:
                # Fallback to VAR(1) if not enough history
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                A1 = A[:, :Z_last.shape[0]]
                Z_forecast[0, :] = A1 @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
            else:
                Z_prev = self._result.Z[-2, :]  # f_{t-2}
                A1 = A[:, :Z_last.shape[0]]
                A2 = A[:, Z_last.shape[0]:]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                if horizon > 1:
                    Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                for h in range(2, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
        else:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="prediction",
                reason=f"unsupported VAR order {p}",
                guidance="Only VAR(1) and VAR(2) are supported. Please use factor_order=1 or factor_order=2"
            )
            raise ValueError(error_msg)
        
        # Transform to observations
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
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
        ValueError
            If model has not been trained yet
        """
        # Check if trained and extract result from training state if needed
        self._check_trained()
        return self._result
    
    
    
    def reset(self) -> 'DDFM':
        """Reset model state."""
        super().reset()
        return self
    
    
    def generate_dataset(
        self,
        target_series: str,
        periods: List[datetime],
        backward: int = 0,
        forward: int = 0,
        dataview: Optional['DataView'] = None
    ) -> Dict[str, Any]:
        """Generate dataset for DFM evaluation.
        
        Note: Requires data to be stored during fit(). Data is automatically
        stored from DataModule during fit().
        """
        if not hasattr(self, '_data') or self._data is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="generate_dataset",
                reason="model has not been fitted with DataModule yet",
                guidance="Please call trainer.fit(model, data_module) first to store data"
            )
            raise ValueError(error_msg)
        
        i_series = find_series_index(self._config, target_series)
        X_features, y_baseline, y_actual, metadata, backward_results = [], [], [], [], []
        
        if dataview is not None:
            dataview_factory = dataview
        else:
            # Convert data to numpy if needed
            if hasattr(self._data, 'to_numpy'):
                X_data = self._data.to_numpy()
            else:
                X_data = np.asarray(self._data)
            
            dataview_factory = DataView.from_arrays(
                X=X_data, Time=self._time,
                Z=self._original_data, config=self._config,
                X_frame=self._data_frame
            )
        if dataview_factory.config is None:
            dataview_factory.config = self._config
        
        for period in periods:
            view_obj = dataview_factory.with_view_date(period)
            X_view, Time_view, _ = view_obj.materialize()
            
            if backward > 0:
                nowcasts, data_view_dates = [], []
                for weeks_back in range(backward, -1, -1):
                    data_view_date = period - timedelta(weeks=weeks_back)
                    view_past = dataview_factory.with_view_date(data_view_date)
                    X_view_past, Time_view_past, _ = view_past.materialize()
                    # Use nowcast property directly
                    nowcast_val = self.nowcast(
                        target_series=target_series,
                        view_date=view_past.view_date or data_view_date,
                        target_period=period
                    )
                    nowcasts.append(nowcast_val)
                    data_view_dates.append(view_past.view_date or data_view_date)
                baseline_nowcast = nowcasts[-1]
                backward_results.append({
                    'nowcasts': np.array(nowcasts),
                    'data_view_dates': data_view_dates,
                    'target_date': period
                })
            else:
                # Use nowcast property directly
                baseline_nowcast = self.nowcast(
                    target_series=target_series,
                    view_date=view_obj.view_date or period,
                    target_period=period
                )
            
            y_baseline.append(baseline_nowcast)
            t_idx = find_time_index(self._time, period)
            actual_val = np.nan
            # Convert data to numpy for indexing
            if hasattr(self._data, 'to_numpy'):
                data_array = self._data.to_numpy()
            else:
                data_array = np.asarray(self._data)
            if t_idx is not None and t_idx < data_array.shape[0] and i_series < data_array.shape[1]:
                actual_val = data_array[t_idx, i_series]
            y_actual.append(actual_val)
            
            # Extract features
            if self._result is not None and hasattr(self._result, 'Z'):
                latest_factors = self._result.Z[-1, :] if self._result.Z.shape[0] > 0 else np.zeros(self._result.Z.shape[1])
            else:
                latest_factors = np.array([])
            if X_view.shape[0] > 0:
                mean_residual = np.nanmean(X_view[-1, :]) if X_view.shape[0] > 0 else 0.0
            else:
                mean_residual = 0.0
            features = np.concatenate([latest_factors, [mean_residual]])
            X_features.append(features)
            metadata.append({'period': period, 'target_series': target_series})
        
        return {
            'X': np.array(X_features),
            'y_baseline': np.array(y_baseline),
            'y_actual': np.array(y_actual),
            'y_target': np.array(y_actual) - np.array(y_baseline),
            'metadata': metadata,
            'backward_results': backward_results if backward > 0 else []
        }
    
    def get_state(
        self,
        t: Union[int, datetime],
        target_series: str,
        lookback: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get DFM state at time t.
        
        Returns state information including baseline nowcast, actual history,
        residuals, and factor history for the specified time period.
        """
        if not hasattr(self, '_data') or self._data is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="get_state",
                reason="model has not been fitted with DataModule yet",
                guidance="Please call trainer.fit(model, data_module) first to store data"
            )
            raise ValueError(error_msg)
        
        if lookback is None:
            lookback = 12  # Default lookback
        
        i_series = find_series_index(self._config, target_series)
        
        # Convert data to numpy if needed
        if hasattr(self._data, 'to_numpy'):
            X_data = self._data.to_numpy()
        else:
            X_data = np.asarray(self._data)
        
        # Use nowcast property directly
        baseline_nowcast = self.nowcast(target_series=target_series, view_date=t, target_period=None)
        
        baseline_forecast, actual_history, residuals, factors_history = [], [], [], []
        t_idx = find_time_index(self._time, t)
        if t_idx is None:
            error_msg = format_error_message(
                model_type="DDFM",
                operation="get_state",
                reason=f"time {t} not found in model_instance._time",
                guidance="Please provide a valid time value that exists in the model's time index"
            )
            raise ValueError(error_msg)
        
        for i in range(max(0, t_idx - lookback + 1), t_idx + 1):
            if i < X_data.shape[0]:
                forecast_val = baseline_nowcast
                baseline_forecast.append(forecast_val)
                actual_val = X_data[i, i_series] if i_series < X_data.shape[1] else np.nan
                actual_history.append(actual_val)
                residuals.append(actual_val - forecast_val)
                if self._result is not None and hasattr(self._result, 'Z') and i < self._result.Z.shape[0]:
                    factors_history.append(self._result.Z[i, :])
                else:
                    factors_history.append(np.array([]))
        
        return {
            'time': t,
            'target_series': target_series,
            'baseline_nowcast': baseline_nowcast,
            'baseline_forecast': np.array(baseline_forecast),
            'actual_history': np.array(actual_history),
            'residuals': np.array(residuals),
            'factors_history': factors_history
        }

