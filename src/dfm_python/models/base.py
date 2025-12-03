"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any
import numpy as np

from ..config import DFMConfig
from ..config.results import BaseResult


class BaseFactorModel(ABC):
    """Base class for all factor models.
    
    This abstract base class defines the common interface that all factor models
    must implement. It ensures that DFM, DDFM, and future model types can be
    used interchangeably through the same API.
    
    Attributes
    ----------
    _config : Optional[DFMConfig]
        Current configuration object
    _result : Optional[BaseResult]
        Last fit result
    """
    
    def __init__(self):
        """Initialize factor model instance."""
        self._config: Optional[DFMConfig] = None
        self._result: Optional[BaseResult] = None
    
    @property
    def config(self) -> Optional[DFMConfig]:
        """Get current configuration."""
        return self._config
    
    @property
    def result(self) -> Optional[BaseResult]:
        """Get last fit result."""
        return self._result
    
    @abstractmethod
    def fit(self, data_module: Any, config: DFMConfig, **kwargs) -> BaseResult:
        """Fit the factor model.
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        config : DFMConfig
            Configuration object specifying model structure and parameters.
        **kwargs
            Additional model-specific parameters that override config values.
            
        Returns
        -------
        BaseResult
            Estimation results including parameters, factors, and diagnostics.
            Returns DFMResult for linear DFM, DDFMResult for DDFM.
        """
        pass
    
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

