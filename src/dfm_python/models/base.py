"""Base interface for factor models.

This module defines the common interface that all factor models (DFM, DDFM, etc.)
must implement, ensuring consistent API across different model types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import numpy as np

from ..config import DFMConfig
from ..dfm import DFMResult


class BaseFactorModel(ABC):
    """Base class for all factor models.
    
    This abstract base class defines the common interface that all factor models
    must implement. It ensures that DFM, DDFM, and future model types can be
    used interchangeably through the same API.
    
    Attributes
    ----------
    _config : Optional[DFMConfig]
        Current configuration object
    _data : Optional[np.ndarray]
        Current data matrix (T x N)
    _result : Optional[DFMResult]
        Last fit result
    _time : Optional[TimeIndex]
        Time index for data (optional, for models that support it)
    """
    
    def __init__(self):
        """Initialize factor model instance."""
        self._config: Optional[DFMConfig] = None
        self._data: Optional[np.ndarray] = None
        self._result: Optional[DFMResult] = None
        self._time: Optional[Any] = None  # TimeIndex type, avoiding circular import
    
    @property
    def config(self) -> Optional[DFMConfig]:
        """Get current configuration."""
        return self._config
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get current data matrix (T x N)."""
        return self._data
    
    @property
    def result(self) -> Optional[DFMResult]:
        """Get last fit result."""
        return self._result
    
    @property
    def time(self) -> Optional[Any]:
        """Get time index (if available)."""
        return self._time
    
    @abstractmethod
    def fit(self, X: np.ndarray, config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the factor model.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N), where T is time periods and N is number of series.
        config : DFMConfig
            Configuration object specifying model structure and parameters.
        **kwargs
            Additional model-specific parameters that override config values.
            
        Returns
        -------
        DFMResult
            Estimation results including parameters, factors, and diagnostics.
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: Optional[int] = None, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, uses default based on clock frequency.
        **kwargs
            Additional forecast parameters.
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted series (and optionally factors). Shape depends on model.
        """
        pass

