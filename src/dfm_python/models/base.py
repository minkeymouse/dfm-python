"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.

All factor models are PyTorch Lightning modules, enabling standard Lightning
training patterns: trainer.fit(model, datamodule).
"""

from abc import abstractmethod
from typing import Optional, Union, Tuple, Any, Dict
from pathlib import Path
import numpy as np
import pytorch_lightning as pl

from ..config import DFMConfig, make_config_source, ConfigSource, MergedConfigSource
from ..config.results import BaseResult


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
    
    @property
    def config(self) -> Optional[DFMConfig]:
        """Get current configuration."""
        return self._config
    
    def _format_error_message(
        self,
        operation: str,
        reason: str,
        guidance: Optional[str] = None
    ) -> str:
        """Format standardized error message.
        
        This helper method provides a consistent error message format across all models.
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
        message = f"{model_type} {operation} failed: {reason}."
        if guidance:
            message += f" {guidance}"
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

