"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any, Dict, List
from pathlib import Path
import numpy as np

from ..config import DFMConfig, make_config_source, ConfigSource, MergedConfigSource
from ..config.results import BaseResult
from ..config.schema import SeriesConfig, DEFAULT_BLOCK_NAME
from ..logger import get_logger

_logger = get_logger(__name__)


class BaseFactorModel(ABC):
    """Abstract base class for all factor models.
    
    This base class provides the common interface that all factor models
    (DFM, DDFM, etc.) must implement. It is a pure abstract class without
    any framework dependencies.
    
    Attributes
    ----------
    _config : Optional[DFMConfig]
        Current configuration object
    _result : Optional[BaseResult]
        Last fit result
    training_state : Optional[Any]
        Training state (model-specific, e.g., DFMTrainingState or DDFMTrainingState)
    """
    
    def __init__(self):
        """Initialize factor model instance."""
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
                f"{self.__class__.__name__} operation failed: model has not been trained yet."
            )
    
    def _create_temp_config(self, block_name: Optional[str] = None) -> DFMConfig:
        """Create a temporary configuration for model initialization.
        
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
            series=[SeriesConfig(series_id='temp', frequency='m')],
            blocks={block_name: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
        )
    
    def _initialize_config(self, config: Optional[DFMConfig] = None) -> DFMConfig:
        """Initialize configuration with common pattern.
        
        Parameters
        ----------
        config : DFMConfig, optional
            Configuration object. If None, creates a temporary config.
            
        Returns
        -------
        DFMConfig
            Configuration object (either provided or created temporary config)
        """
        if config is None:
            config = self._create_temp_config()
        
        self._config = config
        return config
    
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
        
        Supports VAR(1) and VAR(2) factor dynamics (maximum supported order is VAR(2)).
        
        Parameters
        ----------
        Z_last : np.ndarray
            Last factor state (m,)
        A : np.ndarray
            Transition matrix. For VAR(1): (m x m), for VAR(2): (m x 2m)
        p : int
            VAR order. Must be 1 or 2 (maximum supported order is VAR(2))
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
                f"Maximum supported VAR order is VAR(2). Please use factor_order=1 (VAR(1)) or factor_order=2 (VAR(2))"
            )
        return Z_forecast
    
    def _transform_factors_to_observations(
        self,
        Z_forecast: np.ndarray,
        C: np.ndarray,
        Wx: Optional[np.ndarray],
        Mx: Optional[np.ndarray]
    ) -> np.ndarray:
        """Transform forecasted factors to observed series.
        
        Parameters
        ----------
        Z_forecast : np.ndarray
            Forecasted factors (horizon x m)
        C : np.ndarray
            Loading matrix (N x m)
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization (N,)
        Mx : np.ndarray, optional
            Mean values for unstandardization (N,)
            
        Returns
        -------
        np.ndarray
            Forecasted observations (horizon x N)
        """
        X_forecast_std = Z_forecast @ C.T  # (horizon x N)
        
        n_series = C.shape[0]
        
        # Handle Wx with shape validation
        if Wx is not None and len(Wx) != n_series:
            if len(Wx) > n_series:
                Wx_clean = Wx[:n_series]
            else:
                Wx_clean = np.ones(n_series)
                Wx_clean[:len(Wx)] = Wx
        else:
            Wx_clean = Wx if Wx is not None else np.ones(n_series)
        
        # Handle Mx with shape validation
        if Mx is not None and len(Mx) != n_series:
            if len(Mx) > n_series:
                Mx_clean = Mx[:n_series]
            else:
                Mx_clean = np.zeros(n_series)
                Mx_clean[:len(Mx)] = Mx
        else:
            Mx_clean = Mx if Mx is not None else np.zeros(n_series)
        
        X_forecast = X_forecast_std * Wx_clean + Mx_clean
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
        return_factors: bool = True,
        target: Optional[List[str]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, uses default based on clock frequency.
        target : List[str], optional
            List of target series IDs to return. If None, uses target_series from DataModule.
            If DataModule has no target_series, raises ValueError.
            If specified, only returns predictions for the specified target series.
            Only target series are returned (features are excluded).
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
        """
        pass
    
    @abstractmethod
    def update(
        self,
        X_std: np.ndarray,
        *,
        history: Optional[int] = None,
        kalman_filter: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> 'BaseFactorModel':
        """Update factor state with standardized data.
        
        This method permanently updates the last factor state (result.Z[-1, :])
        using the provided standardized data. Users should handle all preprocessing
        (masking, imputation, standardization) before calling this method.
        
        Parameters
        ----------
        X_std : np.ndarray
            Standardized data array (T x N), where T is number of time periods
            and N is number of series. Data should already be standardized using
            result.Mx and result.Wx.
        history : int, optional
            Number of recent periods to use for factor state update. If None, uses
            all provided data (default). If specified (e.g., 60), uses only the most
            recent N periods. Initial state (Z_0, V_0) is always estimated from
            full training data, but the update uses only recent history for efficiency.
        kalman_filter : Any, optional
            Kalman filter instance. If None, uses default or model's kalman filter.
        scaler : Any, optional
            If provided, attach/replace the model's scaler (e.g., refit on new regime).
            When None, keep the existing scaler from training.
            
        Returns
        -------
        BaseFactorModel
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def get_result(self) -> BaseResult:
        """Extract result from trained model.
        
        Returns
        -------
        BaseResult
            Model-specific result object
        """
        pass
    
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
        elif hasattr(new_config, 'num_factors') and new_config.num_factors is not None:
            self.num_factors = new_config.num_factors
        elif hasattr(new_config, 'get_blocks_array'):
            blocks = new_config.get_blocks_array()
            if blocks.shape[1] > 0:
                self.num_factors = int(np.sum(blocks[:, 0]))
            else:
                self.num_factors = 1
        else:
            self.num_factors = 1
        
        return new_config
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state.
        
        Clears configuration, data module, result, and training state.
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
