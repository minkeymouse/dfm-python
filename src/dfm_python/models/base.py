"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.

All factor models are PyTorch Lightning modules, enabling standard Lightning
training patterns: trainer.fit(model, datamodule).
"""

from abc import abstractmethod
from typing import Optional, Union, Tuple, Any, Dict, TYPE_CHECKING
from pathlib import Path
import numpy as np
import pytorch_lightning as pl

if TYPE_CHECKING:
    from ..nowcast import Nowcast

from ..config import DFMConfig, make_config_source, ConfigSource, MergedConfigSource
from ..config.results import BaseResult
from ..config.schema import SeriesConfig, DEFAULT_BLOCK_NAME
from ..logger import get_logger

_logger = get_logger(__name__)


def format_error_message(
    model_type: str,
    operation: str,
    reason: str,
    guidance: Optional[str] = None
) -> str:
    """Format standardized error message (standalone utility function).
    
    This is a standalone utility function that can be used by any class,
    including low-level classes that don't inherit from BaseFactorModel.
    It provides a consistent error message format across all models.
    
    Format: "{ModelType} {operation} failed: {reason}. {guidance}"
    
    Parameters
    ----------
    model_type : str
        Name of the model class (e.g., "DDFM", "DDFMModel", "DFM")
    operation : str
        Name of the operation that failed (e.g., "prediction", "config loading")
    reason : str
        Reason for the failure (e.g., "model has not been trained yet")
    guidance : str, optional
        Actionable guidance for the user (e.g., "Please call trainer.fit(model, data_module) first")
        
    Returns
    -------
    str
        Formatted error message
        
    Examples
    --------
    >>> format_error_message("DDFM", "prediction", "model has not been trained yet",
    ...                      "Please call trainer.fit(model, data_module) first")
    'DDFM prediction failed: model has not been trained yet. Please call trainer.fit(model, data_module) first'
    """
    message = f"{model_type} {operation} failed: {reason}."
    if guidance:
        message += f" {guidance}"
    return message


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
    
    def _format_error_message(
        self,
        operation: str,
        reason: str,
        guidance: Optional[str] = None
    ) -> str:
        """Format standardized error message.
        
        This helper method provides a consistent error message format across all models.
        It uses the standalone format_error_message() utility function for consistency.
        Format: "{ModelType} {operation} failed: {reason}. {guidance}"
        
        Parameters
        ----------
        operation : str
            Name of the operation that failed (e.g., "prediction", "config loading")
        reason : str
            Reason for the failure (e.g., "model has not been trained yet")
        guidance : str, optional
            Actionable guidance for the user (e.g., "Please call trainer.fit(model, data_module) first")
            
        Returns
        -------
        str
            Formatted error message
        """
        model_type = self.__class__.__name__
        return format_error_message(model_type, operation, reason, guidance)
    
    def _format_warning_message(
        self,
        operation: str,
        issue: str,
        context: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> str:
        """Format standardized warning message.
        
        This helper method provides a consistent warning message format across all models.
        Format: "{ModelType} {operation}: {issue}. {context} {suggestion}"
        
        Parameters
        ----------
        operation : str
            Name of the operation (e.g., "factor validation", "training", "prediction")
        issue : str
            Description of the issue (e.g., "3/5 factors are constant")
        context : str, optional
            Additional context (e.g., "Constant factor indices: [0, 2]")
        suggestion : str, optional
            Actionable suggestion for the user (e.g., "This may indicate training issues")
            
        Returns
        -------
        str
            Formatted warning message
        """
        model_type = self.__class__.__name__
        message = f"{model_type} {operation}: {issue}"
        if context:
            message += f". {context}"
        if suggestion:
            message += f". {suggestion}"
        return message
    
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
            
            error_msg = self._format_error_message(
                operation="operation",
                reason="model has not been trained yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
    
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
            warning_msg = self._format_warning_message(
                operation="numerical stability check",
                issue=f"{name} contains {nan_count} NaN/Inf values{context_str}",
                context=f"Shape: {arr.shape}"
            )
            _logger.warning(warning_msg)
            if fallback is not None:
                _logger.info(f"{self.__class__.__name__}: Using fallback for {name}")
                return fallback
            else:
                # Replace NaN/Inf with finite values as last resort
                arr_clean = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                warning_msg = self._format_warning_message(
                    operation="numerical stability check",
                    issue=f"replaced NaN/Inf in {name} with finite values"
                )
                _logger.warning(warning_msg)
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
        
        This helper method consolidates the common pattern of retrieving processed data
        and standardization parameters (Mx, Wx) from the DataModule. It's used by both
        DFM and DDFM in their on_train_start() methods.
        
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
        RuntimeError
            If DataModule is not available or setup() has not been called
        """
        if self._data_module is None:
            # Try to get from trainer if available
            if hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                    self._data_module = self.trainer.datamodule
                else:
                    error_msg = self._format_error_message(
                        operation="data retrieval",
                        reason="DataModule is not available",
                        guidance="Please ensure trainer.fit(model, data_module) is called with a DataModule"
                    )
                    raise RuntimeError(error_msg)
            else:
                error_msg = self._format_error_message(
                    operation="data retrieval",
                    reason="DataModule is not available and trainer is not attached",
                    guidance="Please ensure trainer.fit(model, data_module) is called with a DataModule"
                )
                raise RuntimeError(error_msg)
        
        # Get processed data
        X_torch = self._data_module.get_processed_data()
        
        # Get standardization parameters
        Mx, Wx = self._data_module.get_std_params()
        
        return X_torch, Mx, Wx
    
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
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, uses default based on clock frequency.
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

