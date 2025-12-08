"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.

All factor models are PyTorch Lightning modules, enabling standard Lightning
training patterns: trainer.fit(model, datamodule).
"""

from abc import abstractmethod
from typing import Optional, Union, Tuple, Any, Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd

if TYPE_CHECKING:
    from ..config.results import NowcastResult

from ..config import DFMConfig, make_config_source, ConfigSource, MergedConfigSource
from ..config.results import BaseResult
from ..config.schema import SeriesConfig, DEFAULT_BLOCK_NAME
from ..logger import get_logger
from .utils import create_nowcast_result
from ..utils.time import parse_timestamp, get_latest_time
from ..utils.data import create_data_view

_logger = get_logger(__name__)


class BaseFactorModel(pl.LightningModule):
    """Base class for all factor models (PyTorch Lightning module).
    
    This base class provides the common interface that all factor models
    (DFM, DDFM, etc.) must implement. It inherits from pl.LightningModule,
    ensuring all models can be used with standard Lightning training patterns:
    trainer.fit(model, datamodule).
    
    Attributes
    ----------
    _config : Optional[DFMConfig]
        Current configuration object
    _result : Optional[BaseResult]
        Last fit result
    training_state : Optional[Any]
        Training state (model-specific, e.g., DFMTrainingState or DDFMTrainingState)
    """
    
    def __init__(self, **kwargs):
        """Initialize factor model instance."""
        super().__init__(**kwargs)
        self._config: Optional[DFMConfig] = None
        self._result: Optional[BaseResult] = None
        self.training_state: Optional[Any] = None
        self._data_module: Optional[Any] = None
    
    @property
    def config(self) -> DFMConfig:
        """Get model configuration.
        
        Raises
        ------
        ValueError
            If model configuration has not been set
        """
        if not hasattr(self, '_config') or self._config is None:
            model_type = self.__class__.__name__
            raise ValueError(
                f"{model_type} config access failed: model configuration has not been set. "
                "Please call load_config() or pass config to __init__() first."
            )
        return self._config
    
    
    def _check_trained(self) -> None:
        """Check if model is trained, raise error if not.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        if self._result is None:
            # Try to extract result from training state if available
            if hasattr(self, 'training_state') and self.training_state is not None:
                try:
                    self._result = self.get_result()
                    return
                except (NotImplementedError, AttributeError):
                    # get_result() not implemented or failed, model not fully trained
                    pass
            
            raise ValueError(
                f"{self.__class__.__name__} operation failed: model has not been trained yet. "
                f"Please call trainer.fit(model, data_module) first"
            )
    
    def _check_finite(
        self, 
        arr: np.ndarray, 
        name: str, 
        context: Optional[str] = None,
        fallback: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Check array for NaN/Inf values and apply fallback if needed.
        
        This is a shared utility method for numerical stability checks across all models.
        
        Parameters
        ----------
        arr : np.ndarray
            Array to check
        name : str
            Name of array for error messages
        context : str, optional
            Additional context for error messages (e.g., "at iteration 5", "during MCMC")
        fallback : np.ndarray, optional
            Fallback array to use if NaN/Inf detected. If None, replaces NaN/Inf with finite values.
            
        Returns
        -------
        np.ndarray
            Cleaned array (or fallback if provided)
        """
        if not np.all(np.isfinite(arr)):
            nan_count = np.sum(~np.isfinite(arr))
            context_str = f" {context}" if context else ""
            _logger.warning(
                f"{self.__class__.__name__} numerical stability check: {name} contains {nan_count} NaN/Inf values{context_str}. "
                f"Shape: {arr.shape}"
            )
            if fallback is not None:
                _logger.info(f"{self.__class__.__name__}: Using fallback for {name}")
                return fallback
            else:
                # Replace NaN/Inf with finite values as last resort
                arr_clean = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                _logger.warning(
                    f"{self.__class__.__name__} numerical stability check: replaced NaN/Inf in {name} with finite values"
                )
                return arr_clean
        return arr
    
    def _create_temp_config(self, block_name: Optional[str] = None) -> DFMConfig:
        """Create a temporary configuration for model initialization.
        
        This helper method creates a minimal default configuration when no config
        is provided during model initialization. The temporary config will typically
        be replaced later via load_config().
        
        Parameters
        ----------
        block_name : str, optional
            Name for the default block. If None, uses DEFAULT_BLOCK_NAME.
            
        Returns
        -------
        DFMConfig
            Minimal default configuration with a single temporary series and block
        """
        if block_name is None:
            block_name = DEFAULT_BLOCK_NAME
        
        return DFMConfig(
            series=[SeriesConfig(series_id='temp', frequency='m', transformation='lin', blocks=[1])],
            blocks={block_name: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
        )
    
    def _initialize_config(self, config: Optional[DFMConfig] = None) -> DFMConfig:
        """Initialize configuration with common pattern.
        
        This helper method consolidates the common pattern of creating a temporary
        config if none is provided and setting the internal config. Subclasses
        should call this method in their __init__() before model-specific initialization.
        
        Parameters
        ----------
        config : DFMConfig, optional
            Configuration object. If None, creates a temporary config.
            
        Returns
        -------
        DFMConfig
            Configuration object (either provided or created temporary config)
        """
        # If config not provided, create a temporary config that will be replaced via load_config
        if config is None:
            config = self._create_temp_config()
        
        # Set internal config (config property is read-only, accessed via property getter)
        self._config = config
        
        return config
    
    def _get_data_from_datamodule(self) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray]]:
        """Get processed data and standardization parameters from DataModule.
        
        Returns
        -------
        X_torch : torch.Tensor
            Processed data tensor (T x N)
        Mx : np.ndarray or None
            Mean values for unstandardization (N,)
        Wx : np.ndarray or None
            Standard deviation values for unstandardization (N,)
            
        Raises
        ------
        ValueError
            If DataModule is not available
        """
        data_module = self._get_datamodule()
        X_torch = data_module.get_processed_data()
        Mx, Wx = data_module.get_std_params()
        return X_torch, Mx, Wx
    
    def _get_datamodule(self):
        """Get DataModule from model or trainer.
        
        Returns
        -------
        DataModule
            DataModule instance
            
        Raises
        ------
        ValueError
            If DataModule is not available
        """
        data_module = self._data_module
        if data_module is None and hasattr(self, 'trainer') and self.trainer is not None:
            data_module = getattr(self.trainer, 'datamodule', None)
        
        if data_module is None:
            raise ValueError(
                f"{self.__class__.__name__}: DataModule not available. "
                f"Please ensure DataModule is attached to model or trainer"
            )
        return data_module
    
    def _forecast_var_factors(
        self,
        Z_last: np.ndarray,
        A: np.ndarray,
        p: int,
        horizon: int,
        Z_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecast factors using VAR dynamics.
        
        Supports VAR(1) and VAR(2) factor dynamics.
        
        Parameters
        ----------
        Z_last : np.ndarray
            Last factor state (m,)
        A : np.ndarray
            Transition matrix. For VAR(1): (m x m), for VAR(2): (m x 2m)
        p : int
            VAR order (1 or 2)
        horizon : int
            Number of periods to forecast
        Z_prev : np.ndarray, optional
            Previous factor state for VAR(2) (m,). Required if p == 2.
            
        Returns
        -------
        np.ndarray
            Forecasted factors (horizon x m)
        """
        if p == 1:
            # VAR(1): f_t = A @ f_{t-1}
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            Z_forecast[0, :] = A @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        elif p == 2:
            # VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            if Z_prev is None:
                # Fallback to VAR(1) if not enough history
                A1 = A[:, :Z_last.shape[0]]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
            else:
                A1 = A[:, :Z_last.shape[0]]
                A2 = A[:, Z_last.shape[0]:]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                if horizon > 1:
                    Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                for h in range(2, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
        else:
            raise ValueError(
                f"{self.__class__.__name__} prediction failed: unsupported VAR order {p}. "
                f"Only VAR(1) and VAR(2) are supported. Please use factor_order=1 or factor_order=2"
            )
        return Z_forecast
    
    def _transform_factors_to_observations(
        self,
        Z_forecast: np.ndarray,
        C: np.ndarray,
        Wx: np.ndarray,
        Mx: np.ndarray
    ) -> np.ndarray:
        """Transform forecasted factors to observed series.
        
        Parameters
        ----------
        Z_forecast : np.ndarray
            Forecasted factors (horizon x m)
        C : np.ndarray
            Loading matrix (N x m)
        Wx : np.ndarray
            Standard deviation values for unstandardization (N,)
        Mx : np.ndarray
            Mean values for unstandardization (N,)
            
        Returns
        -------
        np.ndarray
            Forecasted observations (horizon x N)
        """
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        return X_forecast
    
    def _standardize_data(self, X: np.ndarray, Mx: Optional[np.ndarray], Wx: Optional[np.ndarray]) -> np.ndarray:
        """Standardize data using Mx and Wx.
        
        Parameters
        ----------
        X : np.ndarray
            Data to standardize (T x N)
        Mx : np.ndarray or None
            Mean values (N,)
        Wx : np.ndarray or None
            Standard deviation values (N,)
            
        Returns
        -------
        np.ndarray
            Standardized data (T x N)
        """
        if Mx is None or Wx is None:
            return X
        Wx_safe = np.where(Wx == 0, 1.0, Wx)
        return (X - Mx) / Wx_safe
    
    def _update_factor_state_with_history(
        self,
        history: int,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Update factor state using recent N periods of data.
        
        Parameters
        ----------
        history : int
            Number of recent periods to use for state update
        result : BaseResult
            Model result containing parameters
        kalman_filter : Any, optional
            Kalman filter instance
            
        Returns
        -------
        np.ndarray or None
            Updated last factor state (m,), or None if update failed
        """
        try:
            data_module = self._get_datamodule()
            X_torch = data_module.get_processed_data()
            X_data = X_torch.cpu().numpy() if isinstance(X_torch, torch.Tensor) else np.asarray(X_torch)
            
            # Slice to recent N periods
            X_recent = X_data[-history:, :] if X_data.shape[0] > history else X_data
            
            # Standardize and handle NaN
            X_recent_std = self._standardize_data(X_recent, result.Mx, result.Wx)
            X_recent_std = np.where(np.isfinite(X_recent_std), X_recent_std, np.nan)
            
            # Model-specific state update
            if hasattr(self, 'encoder') and self.encoder is not None:
                return self._update_factor_state_ddfm(X_recent_std, result, kalman_filter)
            else:
                return self._update_factor_state_dfm(X_recent_std, result, kalman_filter)
        except Exception as e:
            _logger.warning(
                f"{self.__class__.__name__} predict(): Failed to update factor state with history={history}, "
                f"using training state instead. Error: {type(e).__name__}: {str(e)}"
            )
            return None
    
    def _get_kalman_filter(self, kalman_filter: Optional[Any] = None):
        """Get or create Kalman filter instance."""
        if kalman_filter is not None:
            return kalman_filter
        if hasattr(self, 'kalman') and self.kalman is not None:
            return self.kalman
        from ..ssm.kalman import KalmanFilter
        return KalmanFilter(
            min_eigenval=1e-8,
            inv_regularization=1e-6,
            cholesky_regularization=1e-8
        )
    
    def _result_to_torch_params(self, result: 'BaseResult', C: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """Convert result parameters to torch tensors.
        
        Parameters
        ----------
        result : BaseResult
            Model result containing parameters
        C : np.ndarray, optional
            Override C matrix (for DDFM where C=I)
        R : np.ndarray, optional
            Override R matrix (for DDFM where R=small noise)
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of torch tensors: A, C, Q, R, Z_0, V_0
        """
        return {
            'A': torch.tensor(result.A, dtype=torch.float32),
            'C': torch.tensor(C if C is not None else result.C, dtype=torch.float32),
            'Q': torch.tensor(result.Q, dtype=torch.float32),
            'R': torch.tensor(R if R is not None else result.R, dtype=torch.float32),
            'Z_0': torch.tensor(result.Z_0, dtype=torch.float32),
            'V_0': torch.tensor(result.V_0, dtype=torch.float32)
        }
    
    def _update_factor_state_dfm(
        self,
        X_recent_std: np.ndarray,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> np.ndarray:
        """Update factor state for DFM using Kalman filter.
        
        Parameters
        ----------
        X_recent_std : np.ndarray
            Standardized recent data (T x N)
        result : BaseResult
            Model result containing parameters
        kalman_filter : Any, optional
            Kalman filter instance
            
        Returns
        -------
        np.ndarray
            Updated last factor state (m,)
        """
        # Convert to torch: (N x T) format for Kalman filter
        Y = torch.tensor(X_recent_std.T, dtype=torch.float32)
        params = self._result_to_torch_params(result)
        
        # Run Kalman smoother
        kf = self._get_kalman_filter(kalman_filter)
        zsmooth, _, _, _ = kf(Y, params['A'], params['C'], params['Q'],
                              params['R'], params['Z_0'], params['V_0'])
        
        return zsmooth.T[-1, :].cpu().numpy()
    
    def _update_factor_state_ddfm(
        self,
        X_recent_std: np.ndarray,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Update factor state for DDFM using encoder and Kalman filter.
        
        Parameters
        ----------
        X_recent_std : np.ndarray
            Standardized recent data (T x N)
        result : BaseResult
            Model result containing parameters
        kalman_filter : Any, optional
            Kalman filter instance
            
        Returns
        -------
        np.ndarray or None
            Updated last factor state (m,), or None if encoder unavailable
        """
        if not hasattr(self, 'encoder') or self.encoder is None:
            _logger.warning(
                f"{self.__class__.__name__} predict(): Encoder not available, "
                "using training state instead"
            )
            return None
        
        # Extract factors using encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.eval()
        X_tensor = torch.tensor(X_recent_std, device=device, dtype=torch.float32)
        with torch.no_grad():
            factors_raw = self.encoder(X_tensor).cpu().numpy()  # (T x m)
        
        # For DDFM: C = I (identity), R = small noise
        m = factors_raw.shape[1]
        C = np.eye(m)
        R = np.eye(m) * 1e-8
        
        # Convert to torch: (m x T) format for Kalman filter
        Y = torch.tensor(factors_raw.T, dtype=torch.float32)
        params = self._result_to_torch_params(result, C=C, R=R)
        
        # Run Kalman smoother
        kf = self._get_kalman_filter(kalman_filter)
        zsmooth, _, _, _ = kf(Y, params['A'], params['C'], params['Q'],
                              params['R'], params['Z_0'], params['V_0'])
        
        return zsmooth.T[-1, :].cpu().numpy()
    
    @property
    def result(self) -> BaseResult:
        """Get last fit result.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        self._check_trained()
        return self._result
    
    @abstractmethod
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        history: Optional[int] = None,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, uses default based on clock frequency.
        history : int, optional
            Number of historical periods to use for Kalman filter update before prediction.
            If None, uses full history (default). If specified (e.g., 60), uses only the most
            recent N periods for efficiency. Initial state (Z_0, V_0) is always estimated from
            full history (including any new data beyond training period).
        return_series : bool, default True
            Whether to return forecasted series.
        return_factors : bool, default True
            Whether to return forecasted factors.
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted series (and optionally factors). Shape depends on model.
            - If both return_series and return_factors are True: (X_forecast, Z_forecast)
            - If only return_series is True: X_forecast
            - If only return_factors is True: Z_forecast
            
        Notes
        -----
        When history is specified, the method uses only the most recent N periods for
        Kalman filter update, improving computational efficiency. The initial state
        (Z_0, V_0) is always estimated from full history (including any new data beyond
        training period), ensuring accuracy while maintaining efficiency.
        """
        pass
    
    def nowcast(
        self,
        target_series: str,
        view_date: Optional[Union[datetime, str]] = None,
        target_period: Optional[Union[datetime, str]] = None,
        *,
        return_result: bool = False,
        horizon: int = 1
    ) -> Union[float, 'NowcastResult']:
        """Calculate nowcast for target series.
        
        This method performs nowcasting by:
        1. Creating a data view with masking based on release dates
        2. Re-running Kalman filter with masked data to update factor state
        3. Calling predict() with the updated state
        
        This is equivalent to: masking → Kalman filter update → predict()
        
        Parameters
        ----------
        target_series : str
            Target series ID to nowcast
        view_date : datetime or str, optional
            Data view date (when data is available). If None, uses latest available.
        target_period : datetime or str, optional
            Target period for nowcast. If None, uses latest available period.
        return_result : bool, default False
            If True, returns NowcastResult with additional information.
            If False, returns only the nowcast value (float).
        horizon : int, default 1
            Forecast horizon. For nowcasting, typically 1 (current period).
            
        Returns
        -------
        float or NowcastResult
            Nowcast value if return_result=False, or NowcastResult if return_result=True
            
        Examples
        --------
        >>> model = DFM()
        >>> trainer.fit(model, data_module)
        >>> # Simple nowcast
        >>> value = model.nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
        >>> # With full result
        >>> result = model.nowcast('gdp', view_date='2024-01-15', return_result=True)
        """
        self._check_trained()
        
        # Get DataModule
        data_module = self._get_datamodule()
        raw_data = data_module.data
        time_index = data_module.time_index
        
        # Convert to numpy
        X = raw_data.to_numpy() if hasattr(raw_data, 'to_numpy') else np.asarray(raw_data)
        
        # Parse view_date
        if view_date is None:
            view_date_dt = get_latest_time(time_index)
        elif isinstance(view_date, str):
            view_date_dt = parse_timestamp(view_date)
        else:
            view_date_dt = view_date
        
        # Create data view with masking
        X_view, Time_view, _ = create_data_view(
            X=X,
            Time=time_index,
            Z=None,
            config=self.config,
            view_date=view_date_dt,
            X_frame=raw_data if isinstance(raw_data, pd.DataFrame) else None
        )
        
        # Standardize masked data
        result = self._result
        X_view_std = self._standardize_data(X_view, result.Mx, result.Wx)
        X_view_std = np.where(np.isfinite(X_view_std), X_view_std, np.nan)
        
        # Update factor state using masked data (model-specific)
        if hasattr(self, 'encoder') and self.encoder is not None:
            # DDFM: Extract factors via encoder, then use Kalman filter
            Z_last_updated = self._update_factor_state_ddfm(X_view_std, result, None)
        else:
            # DFM: Use Kalman filter directly on standardized data
            Z_last_updated = self._update_factor_state_dfm(
                X_view_std, result, getattr(self, 'kalman', None)
            )
        
        if Z_last_updated is None:
            Z_last_updated = result.Z[-1, :]
            _logger.warning(
                f"{self.__class__.__name__} nowcast(): Failed to update factor state, "
                f"using training state instead"
            )
        
        # Temporarily update result.Z[-1, :] with updated state
        original_Z_last = result.Z[-1, :].copy()
        result.Z[-1, :] = Z_last_updated
        
        try:
            # Call predict() with updated state
            X_forecast = self.predict(horizon=horizon, return_series=True, return_factors=False)
            
            # Extract nowcast value for target series
            series_ids = [s.series_id for s in self.config.series if s.series_id]
            try:
                target_idx = series_ids.index(target_series)
            except ValueError:
                available = ', '.join(str(sid) for sid in series_ids[:5])
                raise ValueError(
                    f"{self.__class__.__name__} nowcast failed: target_series '{target_series}' not found. "
                    f"Available: {available}..."
                )
            
            # Get nowcast value (first period of forecast)
            if isinstance(X_forecast, np.ndarray):
                nowcast_value = float(X_forecast[0, target_idx])
            else:
                nowcast_value = float(X_forecast[0][target_idx])
            
            if return_result:
                # Calculate data availability
                data_availability = {
                    'n_available': int(np.sum(np.isfinite(X_view))),
                    'n_missing': int(np.sum(np.isnan(X_view)))
                }
                
                # Create NowcastResult using utility function
                return create_nowcast_result(
                    target_series=target_series,
                    target_period=target_period,
                    view_date=view_date_dt,
                    nowcast_value=nowcast_value,
                    factors_at_view=Z_last_updated,
                    data_availability=data_availability,
                    dfm_result=result
                )
            else:
                return nowcast_value
        finally:
            # Restore original state
            result.Z[-1, :] = original_Z_last
    
    def get_result(self) -> BaseResult:
        """Extract result from trained model.
        
        This method should be implemented by subclasses to extract model-specific
        results (DFMResult, DDFMResult, etc.) from the training state.
        
        Returns
        -------
        BaseResult
            Model-specific result object
        """
        raise NotImplementedError("Subclasses must implement get_result()")
    
    
    def _load_config_common(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> DFMConfig:
        """Common logic for loading configuration from various sources.
        
        This method handles the common pattern of creating a config source,
        loading the configuration, updating the internal config, and computing
        the number of factors. Subclasses should call this method and then
        perform any model-specific initialization.
        
        Parameters
        ----------
        source : str, Path, Dict, DFMConfig, or ConfigSource, optional
            Configuration source (YAML path, dict, config object, etc.)
        yaml : str or Path, optional
            YAML file path (alternative to source)
        mapping : Dict, optional
            Dictionary configuration (alternative to source)
        hydra : Dict or DictConfig, optional
            Hydra configuration (alternative to source)
        base : str, Path, Dict, or ConfigSource, optional
            Base configuration for merging
        override : str, Path, Dict, or ConfigSource, optional
            Override configuration for merging with base
        
        Returns
        -------
        DFMConfig
            Loaded configuration object
        
        Raises
        ------
        ValueError
            If base is None when override is specified
        """
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
        new_config = config_source.load()
        
        # Update internal config
        self._config = new_config
        
        # Recompute number of factors from new config
        if hasattr(new_config, 'factors_per_block') and new_config.factors_per_block:
            self.num_factors = int(np.sum(new_config.factors_per_block))
        else:
            blocks = new_config.get_blocks_array()
            if blocks.shape[1] > 0:
                self.num_factors = int(np.sum(blocks[:, 0]))
            else:
                self.num_factors = 1
        
        return new_config
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state.
        
        Clears configuration, data module, result, nowcast, and training state.
        Returns self for method chaining.
        
        Returns
        -------
        BaseFactorModel
            Self for method chaining
        """
        self._config = None
        self._data_module = None
        self._result = None
        if hasattr(self, 'training_state'):
            self.training_state = None
        return self
    
    def load_pickle(self, path: Union[str, Path], **kwargs) -> 'BaseFactorModel':
        """Load a saved model from pickle file.
        
        Note: DataModule is not saved in pickle. Users must create a new DataModule
        and call trainer.fit() with it after loading the model.
        
        Parameters
        ----------
        path : str or Path
            Path to the pickle file to load
        **kwargs
            Additional keyword arguments (reserved for future use)
            
        Returns
        -------
        BaseFactorModel
            Self for method chaining
        """
        import pickle  # Import locally to avoid unnecessary dependency
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self._config = payload.get('config')
        self._result = payload.get('result')
        # Note: data_module is not loaded - users must provide it via trainer.fit()
        return self

