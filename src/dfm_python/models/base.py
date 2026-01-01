"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.

API Differences Between Models
------------------------------
The models in this package (KDFM, DFM, DDFM) have intentionally different APIs
to reflect their different architectures and use cases. They are NOT polymorphic
and cannot be used interchangeably.

**KDFM (Kernelized Dynamic Factor Model)**:
- Training: Uses `training_step()` method (PyTorch Lightning training loop)
- Prediction: `predict(horizon, last_observation)` - REQUIRES `last_observation` parameter
- Result extraction: `get_result()` returns KDFMResult with IRF data
- Architecture: Companion matrix parameterization, structural identification layer
- Use case: Direct IRF estimation, explicit structural shock analysis

**DFM (Dynamic Factor Model)**:
- Training: Uses `fit(data)` method (statsmodels-style)
- Prediction: `predict(horizon)` - NO `last_observation` parameter
- Result extraction: `get_result()` returns DFMResult with factor loadings
- Architecture: Traditional factor model with EM algorithm
- Use case: Dimensionality reduction, factor extraction

**DDFM (Deep Dynamic Factor Model)**:
- Training: Uses `fit_mcmc(data)` method (MCMC sampling)
- Prediction: `predict(horizon)` - NO `last_observation` parameter
- Result extraction: `get_result()` returns DDFMResult with uncertainty quantification
- Architecture: Deep learning + Bayesian inference
- Use case: Probabilistic forecasting with uncertainty quantification

**Why Different APIs?**
These models serve different purposes and have fundamentally different architectures.
KDFM's requirement for `last_observation` reflects its need to compute initial factor
state from the last observed data point, while DFM/DDFM use their internal state.
This design choice enables KDFM's direct IRF estimation capability but requires
explicit state management.

**Usage Examples**:
    # KDFM
    model = KDFM(config=config)
    model.initialize_from_data(data)
    # Training via PyTorch Lightning Trainer
    trainer.fit(model, datamodule)
    forecasts = model.predict(horizon=8, last_observation=last_obs)
    
    # DFM
    model = DFM(config=config)
    model.fit(data)
    forecasts = model.predict(horizon=8)
    
    # DDFM
    model = DDFM(config=config)
    model.fit_mcmc(data)
    forecasts = model.predict(horizon=8)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any, Dict, List, TYPE_CHECKING
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
else:
    try:
        from torch import Tensor
    except ImportError:
        Tensor = Any

if TYPE_CHECKING:
    from ..datamodule import KDFMDataModule, DFMDataModule, DDFMDataModule

from ..config import (
    DFMConfig, make_config_source, ConfigSource,
    BaseResult
)
from ..config.constants import DEFAULT_BLOCK_NAME, DEFAULT_DTYPE, DEFAULT_FORECAST_HORIZON, MAX_WARNING_ITEMS, MAX_ERROR_ITEMS, DEFAULT_CLOCK_FREQUENCY
from ..logger import get_logger
from ..utils.errors import ConfigurationError, ModelNotTrainedError, ModelNotInitializedError
from ..utils.validation import check_has_attr, check_condition

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
        self.data_processed: Optional[np.ndarray] = None  # Store processed training data for shape validation
    
    @property
    def config(self) -> DFMConfig:
        """Get model configuration.
        
        Returns
        -------
        DFMConfig
            Current model configuration object
            
        Raises
        ------
        ConfigurationError
            If model configuration has not been set
        """
        model_type = self.__class__.__name__
        check_has_attr(self, '_config', model_type, error_class=ConfigurationError)
        if self._config is None:
            raise ConfigurationError(
                f"{model_type} config access failed: model configuration has not been set",
                details="Please call load_config() or pass config to __init__() first"
            )
        return self._config
    
    def _check_trained(self) -> None:
        """Check if model is trained, raise error if not.
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        """
        if self._result is None:
            # Try to extract result from training state if available
            training_state = getattr(self, 'training_state', None)
            if training_state is not None:
                try:
                    self._result = self.get_result()
                    return
                except (NotImplementedError, AttributeError):
                    # get_result() not implemented or failed, model not fully trained
                    pass
            
            raise ModelNotTrainedError(
                f"{self.__class__.__name__} operation failed: model has not been trained yet",
                details="Please call fit() or train the model before accessing results"
            )
    
    def _ensure_result(self) -> BaseResult:
        """Ensure result exists, computing it if necessary.
        
        This helper method ensures that self._result is available. If it's None,
        it attempts to compute it from training_state. This is a common pattern
        used in update(), predict(), and result property.
        
        Returns
        -------
        BaseResult
            The model result (computed if necessary)
            
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        """
        if self._result is None:
            self._check_trained()  # This will try to compute result if training_state exists
            if self._result is None:
                # If still None after _check_trained, explicitly compute
                self._result = self.get_result()
        return self._result
    
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
            frequency={'temp': DEFAULT_CLOCK_FREQUENCY},
            blocks={block_name: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
        )
    
    def _initialize_config(self, config: Optional[DFMConfig] = None) -> DFMConfig:
        """Initialize configuration with common pattern.
        
        Parameters
        ----------
        config : DFMConfig, optional
            Configuration to use. If None, creates temporary config.
            
        Returns
        -------
        DFMConfig
            Initialized configuration
        """
        if config is None:
            config = self._create_temp_config()
        
        self._config = config
        return config
    
    def _load_config_common(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> DFMConfig:
        """Common config loading logic shared by all models."""
        config_source = make_config_source(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
        )
        
        new_config = config_source.load()
        self._config = new_config
        return new_config
    
    def _get_datamodule(self) -> Union['KDFMDataModule', 'DFMDataModule', 'DDFMDataModule']:
        """Get DataModule from model or trainer.
        
        This method attempts to retrieve the DataModule from:
        1. Model's _data_module attribute (if set directly)
        2. Trainer's datamodule attribute (if model has trainer and trainer.fit() was called)
        
        This is a common helper used by predict() methods to access data preprocessing
        parameters (Mx, Wx) and target series configuration.
        
        **Type Note**: Uses TYPE_CHECKING to avoid circular imports. The return type
        is `Union[KDFMDataModule, DFMDataModule, DDFMDataModule]` depending on model type.
        TYPE_CHECKING allows proper type hints without runtime circular dependencies.
        
        Returns
        -------
        Any
            DataModule instance (KDFMDataModule, DFMDataModule, or DDFMDataModule)
            - KDFMDataModule: For KDFM models
            - DFMDataModule: For DFM models
            - DDFMDataModule: For DDFM models
            
        Raises
        ------
        ModelNotInitializedError
            If DataModule is not available from any source. This typically means:
            - Trainer.fit() has not been called yet
            - DataModule was not attached to trainer
            - DataModule was not set directly on model
            - Model was not properly initialized before use
            
        Examples
        --------
        >>> # After trainer.fit(), DataModule is available from trainer
        >>> data_module = model._get_datamodule()
        >>> target_series = data_module.target_series
        >>> target_scaler = data_module.target_scaler
        """
        data_module = getattr(self, '_data_module', None)
        
        if data_module is None:
            trainer = getattr(self, 'trainer', None)
            if trainer is not None:
                data_module = getattr(trainer, 'datamodule', None)
        
        if data_module is None:
            raise ModelNotInitializedError(
                f"{self.__class__.__name__}: DataModule not available",
                details=(
                    "DataModule is required for data access and preprocessing. "
                    "Please ensure: (1) DataModule is attached to trainer, "
                    "(2) Trainer.fit() has been called, or (3) DataModule is set directly on model."
                )
            )
        return data_module
    
    def _forecast_var_factors(
        self,
        Z_last: np.ndarray,
        A: np.ndarray,
        p: int = 1,
        horizon: int = 1,
        Z_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecast factors using AR(1) dynamics.
        
        This is a convenience wrapper around forecast_ar1_factors() from numeric.estimator.
        The `p` and `Z_prev` parameters are kept for backward compatibility but are ignored
        (AR(1) dynamics only).
        
        Parameters
        ----------
        Z_last : np.ndarray
            Last factor state of shape (m,)
        A : np.ndarray
            Transition matrix of shape (m, m)
        p : int, default 1
            AR order (ignored, always uses AR(1))
        horizon : int, default 1
            Number of periods to forecast
        Z_prev : np.ndarray, optional
            Ignored (unused parameter)
            
        Returns
        -------
        np.ndarray
            Forecasted factors of shape (horizon, m)
        """
        from ..numeric.estimator import forecast_ar1_factors
        return forecast_ar1_factors(Z_last, A, horizon, dtype=DEFAULT_DTYPE)
    
    # Legacy method removed: _transform_factors_to_observations()
    # This method used Mx/Wx arrays which are no longer used.
    # Models now use target_scaler.inverse_transform() directly in their predict() methods.
    
    def _resolve_target_series(
        self,
        series_ids: Optional[List[str]] = None,
        result: Optional[BaseResult] = None
    ) -> Tuple[Optional[List[str]], Optional[List[int]]]:
        """Resolve target series from DataModule.
        
        This is a convenience wrapper around resolve_target_series() from utils.misc.
        It gets the DataModule and series_ids, then calls the utility function.
        
        Parameters
        ----------
        series_ids : List[str], optional
            Available series IDs from config or result. Used for validation.
        result : BaseResult, optional
            Result object that may contain series_ids. Used as fallback.
            
        Returns
        -------
        Tuple[Optional[List[str]], Optional[List[int]]]
            Tuple of (target_series_ids, target_indices) where:
            - target_series_ids: List of target series IDs (None if not resolved)
            - target_indices: List of indices into series_ids (None if not resolved)
            
        Raises
        ------
        DataError
            If target series are not found in available series
        """
        from ..utils.misc import resolve_target_series
        
        # Get DataModule (may raise ModelNotInitializedError)
        data_module = None
        try:
            data_module = self._get_datamodule()
        except (ModelNotInitializedError, AttributeError):
            pass
        
        # Get series_ids for validation
        if series_ids is None:
            if result is not None:
                series_ids = getattr(result, 'series_ids', None)
            if series_ids is None and self._config is not None:
                series_ids = self._config.get_series_ids()
        
        # Call utility function
        return resolve_target_series(
            datamodule=data_module,
            series_ids=series_ids,
            result=result,
            model_name=self.__class__.__name__
        )
    
    def _compute_default_horizon(self, default: Optional[int] = None) -> int:
        """Compute default forecast horizon from clock frequency.
        
        This is a convenience wrapper around compute_default_horizon() from utils.misc.
        
        Parameters
        ----------
        default : int, optional
            Default value to use if clock frequency cannot be determined.
            If None, uses DEFAULT_FORECAST_HORIZON constant.
            
        Returns
        -------
        int
            Default horizon in periods (typically 1 year worth of periods)
        """
        from ..utils.misc import compute_default_horizon
        return compute_default_horizon(self._config, default=default)
    
    # Legacy method removed: _standardize_data()
    # This method used Mx/Wx arrays which are no longer used.
    # Data standardization should be done using sklearn scalers before passing to models.
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state."""
        self._config = None
        self._result = None
        self.training_state = None
        self._data_module = None
        return self
    
    
    @abstractmethod
    def update(self, data: Union[np.ndarray, Any]) -> None:
        """Update model state with new observations.
        
        This method updates the model's internal state (factors) with new observations,
        but keeps model parameters fixed. The implementation differs by model type:
        - DFM: Uses Kalman filtering/smoothing
        - DDFM: Uses neural network forward pass
        - KDFM: Uses companion matrix forward pass
        
        **Data Shape**: The input data must be 2D with shape (T_new x N) where:
        - T_new: Number of new time steps (can be any positive integer)
        - N: Number of series (must match training data)
        
        **Supported Types**:
        - numpy.ndarray: (T_new x N) array
        - pandas.DataFrame: DataFrame with N columns, T_new rows
        - polars.DataFrame: DataFrame with N columns, T_new rows
        
        **Important**: Data must be preprocessed by the user (same preprocessing as training).
        Only target scaler is handled internally if needed.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations with shape (T_new x N) where:
            - T_new: Number of new time steps (any positive integer)
            - N: Number of series (must match training data)
            Data must be preprocessed by user (same preprocessing as training).
            
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        DataValidationError
            If data shape doesn't match training data (N must match)
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    @abstractmethod
    def get_result(self) -> BaseResult:
        """Extract result from trained model.
        
        Returns
        -------
        BaseResult
            Model-specific result object
        """
        raise NotImplementedError("Subclasses must implement get_result()")
