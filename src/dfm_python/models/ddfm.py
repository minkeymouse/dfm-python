"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
from ..logger import get_logger

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# PyTorch is mandatory - no optional import needed

from .base import BaseFactorModel
from ..config import (
    DFMConfig, DEFAULT_BLOCK_NAME,
    make_config_source, ConfigSource, MergedConfigSource,
)
from ..config.results import DDFMResult
from ..config.utils import get_periods_per_year
from ..utils.data import rem_nans_spline
from ..utils.helpers import (
    safe_get_attr,
    get_clock_frequency,
    resolve_param,
)
from ..encoder.vae import Encoder, Decoder, extract_decoder_params
from ..nowcast.dataview import DataView

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)




class DDFMModel(BaseFactorModel):
    """Deep Dynamic Factor Model using PyTorch (low-level implementation).
    
    This class implements a DDFM with:
    - Nonlinear encoder (MLP) to extract factors from observations
    - Linear decoder for interpretability
    - Linear factor dynamics (VAR)
    - Kalman filtering for final smoothing
    
    The model is trained using gradient descent (Adam optimizer) to minimize
    reconstruction error, then factor dynamics are estimated via OLS, and
    final smoothing is performed using Kalman filter.
    
    Note: This is the low-level implementation. For high-level API with
    load_config, load_data, train methods, use the DDFM class (defined below).
    """
    
    def __init__(
        self,
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
            Activation function ('tanh', 'relu', 'sigmoid'). Default: 'tanh'
        use_batch_norm : bool
            Whether to use batch normalization in encoder. Default: True
        learning_rate : float
            Learning rate for Adam optimizer. Default: 0.001
        epochs : int
            Number of epochs per MCMC iteration. Default: 100
        batch_size : int
            Batch size for training. Default: 32
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
        super().__init__()
        
        # PyTorch is mandatory - no need to check
        if factor_order not in [1, 2]:
            raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
        
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
    
    def fit(self, data_module: 'DFMDataModule', config: DFMConfig, **kwargs) -> DDFMResult:
        """Fit the DDFM model.
        
        Training process:
        1. Get preprocessed data from DataModule
        2. Train autoencoder (encoder + decoder) to minimize reconstruction error
        3. Extract factors using trained encoder
        4. Extract decoder parameters (C, bias) directly from trained decoder
        5. Compute residuals and estimate idiosyncratic AR(1) dynamics
        6. Estimate factor dynamics (VAR(1) or VAR(2)) via OLS
        7. Build complete state-space model (factor + idio)
        8. Apply Kalman smoothing for final state estimates
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        config : DFMConfig
            Configuration object. Used to determine number of factors if not specified.
        **kwargs
            Additional parameters:
            - epochs: Override default epochs
            - batch_size: Override default batch size
            - learning_rate: Override default learning rate
            
        Returns
        -------
        DDFMResult
            Estimation results with DDFM-specific fields (training_loss, encoder_layers, etc.).
        """
        # PyTorch is mandatory - no need to check
        from ..lightning import DFMDataModule
        
        if not isinstance(data_module, DFMDataModule):
            raise TypeError(f"data_module must be DFMDataModule, got {type(data_module)}")
        
        # Ensure DataModule is set up
        if data_module.data_processed is None:
            data_module.setup()
        
        # Store config and DataModule for later use
        self._config = config
        self._data_module = data_module
        
        # Get processed data from DataModule
        X_torch = data_module.get_processed_data()
        X = X_torch.numpy() if isinstance(X_torch, torch.Tensor) else X_torch
        
        # Store raw data and time index from DataModule for utility methods (generate_dataset, get_state)
        # Temporary storage for utility methods
        self._data = data_module.data
        self._time = data_module.time_index
        # For Z (original data), use the same as data (raw, before transformation)
        if hasattr(self._data, 'to_numpy'):
            self._original_data = self._data.to_numpy()
        else:
            self._original_data = np.asarray(self._data)
        self._data_frame = self._data if isinstance(self._data, pl.DataFrame) else None
        
        # Override hyperparameters from kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        
        # Determine number of factors
        if self.num_factors is None:
            # Infer from config (sum of factors per block)
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                num_factors = int(np.sum(config.factors_per_block))
            else:
                # Default: use first block's factors or 1
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    num_factors = int(np.sum(blocks[:, 0]))  # First block
                else:
                    num_factors = 1
        else:
            num_factors = self.num_factors
        
        # Get standardization parameters from DataModule (may be None)
        Mx, Wx = data_module.get_standardization_params()
        
        # Handle case where standardization params might be None
        # (if transformer doesn't include StandardScaler)
        if Mx is None or Wx is None:
            # Use zeros/ones as defaults (no standardization)
            N = X_torch.shape[1] if isinstance(X_torch, torch.Tensor) else X_torch.shape[1]
            Mx = np.zeros(N, dtype=np.float32)
            Wx = np.ones(N, dtype=np.float32)
        
        # Data is already transformed and standardized in DataModule
        # Convert torch tensor to numpy for processing
        if isinstance(X_torch, torch.Tensor):
            x_standardized = X_torch.cpu().numpy()
        else:
            x_standardized = X
        
        T, N = x_standardized.shape
        
        # Step 1: Handle missing data (simple interpolation for now)
        nan_method = kwargs.get('nan_method', safe_get_attr(config, 'nan_method', 2))
        nan_k = kwargs.get('nan_k', safe_get_attr(config, 'nan_k', 3))
        x_clean, _ = rem_nans_spline(x_standardized, method=nan_method, k=nan_k)
        
        # Convert to torch tensors
        x_standardized_torch = torch.tensor(x_standardized, dtype=torch.float32, device=self.device)
        x_clean_torch = torch.tensor(x_clean, dtype=torch.float32, device=self.device)
        missing_mask = np.isnan(x_standardized)
        
        # Step 3: Create Lightning module
        from ..lightning.ddfm_module import DDFMLightningModule
        
        lightning_module = DDFMLightningModule(
            config=config,
            encoder_layers=self.encoder_layers,
            num_factors=num_factors,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            factor_order=self.factor_order,
            use_idiosyncratic=self.use_idiosyncratic,
            min_obs_idio=self.min_obs_idio,
        )
        
        # Move to device
        lightning_module = lightning_module.to(self.device)
        
        # Step 4: Optional pre-training
        pre_train_epochs = kwargs.get('pre_train_epochs', None)
        if pre_train_epochs is not None and pre_train_epochs > 0:
            _logger.info(f"Pre-training DDFM autoencoder: {pre_train_epochs} epochs")
            # Initialize networks first
            lightning_module.initialize_networks(N)
            # Create dataset and dataloader for pre-training
            dataset = torch.utils.data.TensorDataset(x_clean_torch, x_clean_torch)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            # Train using DDFM trainer
            from ..trainer import DDFMTrainer
            trainer = DDFMTrainer(
                max_epochs=pre_train_epochs,
                enable_progress_bar=True,
                logger=False,
                enable_model_summary=False,
            )
            trainer.fit(lightning_module, dataloader)
        
        # Step 5: MCMC iterative training procedure
        _logger.info(f"Starting MCMC iterative training: epochs_per_iter={epochs}, max_iter={self.max_iter}")
        
        # Run MCMC training
        lightning_module.fit_mcmc(
            X=x_standardized_torch,
            x_clean=x_clean_torch,
            missing_mask=missing_mask,
            Mx=Mx,
            Wx=Wx,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            disp=self.disp,
            seed=self.rng.randint(0, 2**31) if hasattr(self.rng, 'randint') else None,
        )
        
        # Step 6: Extract results
        result = lightning_module.get_result()
        
        # Store encoder/decoder
        self.encoder = lightning_module.encoder
        self.decoder = lightning_module.decoder
        
        # Store result
        self._result = result
        return result
    
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
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Default horizon
        if horizon is None:
            if self._config is not None:
                clock = get_clock_frequency(self._config, 'm')
                horizon = get_periods_per_year(clock)
            else:
                horizon = 12  # Default to 12 periods if no config
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
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
            raise ValueError(f"Unsupported VAR order: {p}")
        
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
            raise ValueError("Model must be fitted with DataModule before calling generate_dataset()")
        
        from ..utils.helpers import find_series_index
        from ..utils.time import find_time_index
        from ..nowcast.dataview import DataView
        
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
                        raise ValueError("nowcast() requires high-level DDFM instance. Call from DDFM class, not DDFMModel.")
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
                    raise ValueError("nowcast() requires high-level DDFM instance. Call from DDFM class, not DDFMModel.")
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
            raise ValueError("Model must be fitted with DataModule before calling get_state()")
        
        from ..config.utils import get_periods_per_year
        from ..utils.helpers import find_series_index
        from ..utils.time import find_time_index, convert_to_timestamp
        # create_data_view is already imported at module level
        
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
            raise ValueError("nowcast() requires high-level DDFM instance. Call from DDFM class, not DDFMModel.")
        baseline_nowcast = nowcast_obj(target_series=target_series, view_date=t, target_period=None)
        
        baseline_forecast, actual_history, residuals, factors_history = [], [], [], []
        t_idx = find_time_index(self._time, t)
        if t_idx is None:
            raise ValueError(f"Time {t} not found in model_instance._time")
        
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

# Note: PyTorch is now mandatory for DDFM - placeholder removed


# ============================================================================
# High-level API Classes
# ============================================================================
# Note: transformations.utils removed - use DFMDataModule for data loading
from ..config.results import DFMResult
from ..utils.helpers import (
    safe_get_method,
    safe_get_attr,
    get_clock_frequency,
    _validate_config_loaded,
    _validate_result_loaded,
)
from ..utils.time import TimeIndex

if TYPE_CHECKING:
    from ..nowcast import Nowcast
    from ..lightning import DFMDataModule

class DDFM(BaseFactorModel):
    """High-level API for Deep Dynamic Factor Model.
    
    This class provides a unified interface for loading configuration, data,
    training, and prediction. It uses DDFMModel internally for the actual
    model implementation.
    
    Example:
        >>> from dfm_python.lightning import DFMDataModule
        >>> from sktime.transformations.compose import ColumnTransformer
        >>> 
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config.yaml')
        >>> 
        >>> # Create transformer (user must provide)
        >>> transformer = ColumnTransformer([...])  # User-defined
        >>> 
        >>> # Create DataModule
        >>> data_module = DFMDataModule(config=model.config, transformer=transformer, data_path='data.csv')
        >>> data_module.setup()
        >>> 
        >>> # Train
        >>> model.train(data_module, epochs=100)
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
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
        """Initialize DDFM instance."""
        super().__init__()
        self._model_impl = DDFMModel(
            encoder_layers=encoder_layers,
            num_factors=num_factors,
            activation=activation,
            use_batch_norm=use_batch_norm,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            factor_order=factor_order,
            use_idiosyncratic=use_idiosyncratic,
            min_obs_idio=min_obs_idio,
            **kwargs
        )
        self._data_module: Optional['DFMDataModule'] = None
        self._nowcast: Optional['Nowcast'] = None
    
    @property
    def nowcast(self) -> 'Nowcast':
        """Get nowcasting manager instance."""
        if self._nowcast is None:
            _validate_config_loaded(self._config)
            if self._data_module is None:
                raise ValueError("DataModule must be provided via train() before accessing nowcast")
            _validate_result_loaded(self._result)
            from ..nowcast.nowcast import Nowcast
            self._nowcast = Nowcast(model=self, data_module=self._data_module)
        return self._nowcast
    
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
        from ..config import make_config_source, MergedConfigSource
        # Handle base and override merging
        if base is not None or override is not None:
            if base is None:
                raise ValueError("base must be provided when override is specified")
            base_source = make_config_source(source=base)
            override_source = make_config_source(source=override) if override is not None else None
            if override_source is not None:
                config_source = MergedConfigSource(base_source, override_source)
            else:
                config_source = base_source
        else:
            config_source = make_config_source(
                source=source,
                yaml=yaml,
                mapping=mapping,
                hydra=hydra,
            )
        self._config = config_source.load()
        return self
    
    
    def fit(self, data_module: 'DFMDataModule', config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the DDFM model (implements abstract method from BaseFactorModel)."""
        self._config = config
        self._data_module = data_module
        self._result = self._model_impl.fit(data_module, config, **kwargs)
        return self._result
    
    def train(
        self,
        data_module: 'DFMDataModule',
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> 'DDFM':
        """Train the DDFM model.
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        learning_rate : float, optional
            Learning rate for optimizer
        **kwargs
            Additional parameters
        """
        from ..lightning import DFMDataModule
        _validate_config_loaded(self._config)
        
        if not isinstance(data_module, DFMDataModule):
            raise TypeError(f"data_module must be DFMDataModule, got {type(data_module)}")
        
        # Ensure DataModule is set up
        if data_module.data_processed is None:
            data_module.setup()
        
        self._data_module = data_module
        
        train_kwargs = {}
        if epochs is not None:
            train_kwargs['epochs'] = epochs
        if batch_size is not None:
            train_kwargs['batch_size'] = batch_size
        if learning_rate is not None:
            train_kwargs['learning_rate'] = learning_rate
        train_kwargs.update(kwargs)
        
        self._result = self._model_impl.fit(
            data_module,
            self._config,
            **train_kwargs
        )
        return self
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values (implements abstract method from BaseFactorModel)."""
        from ..utils.helpers import _validate_result_loaded
        _validate_result_loaded(self._result)
        return self._model_impl.predict(
            horizon=horizon,
            return_series=return_series,
            return_factors=return_factors
        )
    
    def plot(self, **kwargs) -> 'DDFM':
        """Plot common visualizations.
        
        .. note::
            Plot functionality is not yet implemented. This method is a placeholder
            for future visualization features. Use external plotting libraries
            (matplotlib, plotly) with model results for visualization.
        """
        from ..utils.helpers import _validate_result_loaded
        _validate_result_loaded(self._result)
        _logger.warning("Plot functionality not yet implemented. Use external plotting libraries with model results.")
        return self
    
    def reset(self) -> 'DDFM':
        """Reset model state."""
        self._config = None
        self._data_module = None
        self._result = None
        self._nowcast = None
        return self
    
    def load_pickle(self, path: Union[str, Path], **kwargs) -> 'DDFM':
        """Load a saved model from pickle file.
        
        Note: DataModule is not saved in pickle. Users must create a new DataModule
        and call train() with it after loading the model.
        """
        import pickle  # Import locally to avoid unnecessary dependency
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self._config = payload.get('config')
        self._result = payload.get('result')
        # Note: data_module is not loaded - users must provide it via train()
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
        
        Delegates to model_impl.generate_dataset() with access to high-level nowcast.
        """
        # Store nowcast reference in model_impl for the method call
        # This allows generate_dataset to access nowcast property
        setattr(self._model_impl, '_nowcast_ref', self.nowcast)
        try:
            result = self._model_impl.generate_dataset(
                target_series=target_series,
                periods=periods,
                backward=backward,
                forward=forward,
                dataview=dataview
            )
        finally:
            # Clean up
            if hasattr(self._model_impl, '_nowcast_ref'):
                delattr(self._model_impl, '_nowcast_ref')
        return result
    
    def get_state(
        self,
        t: Union[int, datetime],
        target_series: str,
        lookback: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get DFM state at time t.
        
        Delegates to model_impl.get_state() with access to high-level nowcast.
        """
        # Store nowcast reference in model_impl for the method call
        setattr(self._model_impl, '_nowcast_ref', self.nowcast)
        try:
            result = self._model_impl.get_state(
                t=t,
                target_series=target_series,
                lookback=lookback
            )
        finally:
            # Clean up
            if hasattr(self._model_impl, '_nowcast_ref'):
                delattr(self._model_impl, '_nowcast_ref')
        return result

