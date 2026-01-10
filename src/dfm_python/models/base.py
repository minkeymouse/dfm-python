"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.

API Differences Between Models
------------------------------
The models in this package (DFM, DDFM) have intentionally different APIs
to reflect their different architectures and use cases. They are NOT polymorphic
and cannot be used interchangeably.

**DFM (Dynamic Factor Model)**:
- Training: Uses `fit(data)` method (statsmodels-style)
- Prediction: `predict(horizon)` - NO `last_observation` parameter
- Result extraction: `get_result()` returns DFMResult with factor loadings
- Architecture: Traditional factor model with EM algorithm
- Use case: Dimensionality reduction, factor extraction

**DDFM (Deep Dynamic Factor Model)**:
- Training: Uses `fit()` method (consolidates model building, pre-training, and training)
- Prediction: `predict(horizon)` - NO `last_observation` parameter
- Result extraction: `get_result()` returns DDFMResult with uncertainty quantification
- Architecture: Deep learning + Bayesian inference
- Use case: Probabilistic forecasting with uncertainty quantification

**Usage Examples**:
    # DFM
    model = DFM(config=config)
    dataset = DFMDataset(config=config, data=data)
    model.fit(X=dataset.get_processed_data(), dataset=dataset)
    forecasts = model.predict(horizon=8)
    
    # DDFM
    model = DDFM(dataset=dataset, config=config)
    model.fit()  # Builds model, pre-trains, and trains
    model.build_state_space()  # Required for prediction
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
    from ..dataset.dfm_dataset import DFMDataset
    from ..dataset.ddfm_dataset import DDFMDataset

from ..config import (
    DFMConfig, make_config_source, ConfigSource,
    BaseResult
)
from ..config.constants import DEFAULT_DTYPE, DEFAULT_FORECAST_HORIZON, MAX_WARNING_ITEMS, MAX_ERROR_ITEMS
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
        Training state (model-specific):
        - DFM: DFMStateSpaceParams (state-space parameters A, C, Q, R, Z_0, V_0)
        - DDFM: DDFMTrainingState (training intermediate state with factors, eps, etc.)
    """
    
    def __init__(self):
        """Initialize factor model instance."""
        self._config: Optional[DFMConfig] = None
        self._result: Optional[BaseResult] = None
        self.training_state: Optional[Any] = None
        self._dataset: Optional[Any] = None
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
        """Ensure result exists, computing it if necessary."""
        if self._result is None:
            self._check_trained()
            if self._result is None:
                self._result = self.get_result()
        return self._result
    
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
    
    def _get_dataset(self) -> Union['DFMDataset', 'DDFMDataset']:
        """Get Dataset from model.
        
        This method retrieves the Dataset from the model's _dataset attribute.
        The Dataset should be set during training (model.train() or model.fit()).
        
        This is a common helper used by predict() methods to access data preprocessing
        parameters and target series configuration.
        
        **Type Note**: Uses TYPE_CHECKING to avoid circular imports. The return type
        is `Union[DFMDataset, DDFMDataset]` depending on model type.
        TYPE_CHECKING allows proper type hints without runtime circular dependencies.
        
        Returns
        -------
        Any
            Dataset instance (DFMDataset or DDFMDataset)
            - DFMDataset: For DFM models
            - DDFMDataset: For DDFM models
            
        Raises
        ------
        ModelNotInitializedError
            If Dataset is not available. This typically means:
            - Model.train() or model.fit() has not been called yet
            - Dataset was not set during training
            - Model was not properly initialized before use
            
        Examples
        --------
        >>> # After model.train(), Dataset is available
        >>> dataset = model._get_dataset()
        >>> target_series = dataset.target_series
        >>> target_scaler = dataset.target_scaler
        """
        dataset = getattr(self, '_dataset', None)
        
        if dataset is None:
            raise ModelNotInitializedError(
                f"{self.__class__.__name__}: Dataset not available",
                details=(
                    "Dataset is required for data access and preprocessing. "
                    "Please ensure: (1) Dataset is passed to train()/fit(), "
                    "(2) Model has been trained, or (3) Dataset is set directly on model."
                )
            )
        return dataset
    
    def _get_target_scaler(self):
        """Get target scaler from dataset.
        
        Consolidates duplicate pattern of extracting target_scaler from dataset.
        This is a common helper used across all models (DFM, DDFM) for
        accessing the target scaler for inverse transformation during prediction.
        
        Returns
        -------
        target_scaler or None
            Target scaler from dataset, or None if not available
        """
        dataset = getattr(self, '_dataset', None)
        if dataset is None:
            return None
        return getattr(dataset, 'target_scaler', None)
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state."""
        self._config = None
        self._result = None
        self.training_state = None
        self._dataset = None
        return self
    
    
    @abstractmethod
    def update(self, data: Union[np.ndarray, Any]) -> None:
        """Update model state with new observations.
        
        This method updates the model's internal state (factors) with new observations,
        but keeps model parameters fixed. The implementation differs by model type:
        - DFM: Uses Kalman filtering/smoothing
        - DDFM: Uses neural network forward pass
        
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
