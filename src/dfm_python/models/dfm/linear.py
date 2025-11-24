"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
It inherits from BaseFactorModel to provide a consistent interface.
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

from ..base import BaseFactorModel
from ...config import DFMConfig
from ...dfm import DFMResult, DFMParams, _dfm_core
from ...engine.kalman import run_kf

_logger = logging.getLogger(__name__)


class DFMLinear(BaseFactorModel):
    """Linear Dynamic Factor Model using EM algorithm.
    
    This class implements the standard linear DFM with EM estimation.
    It inherits from BaseFactorModel to provide a consistent interface
    with other factor models (e.g., DDFM).
    
    The model assumes:
    - Linear observation equation: y_t = C Z_t + e_t
    - Linear factor dynamics: Z_t = A Z_{t-1} + v_t
    - Gaussian innovations
    
    Parameters are estimated via Expectation-Maximization (EM) algorithm.
    """
    
    def fit(self, X: np.ndarray, config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the linear DFM model using EM algorithm.
        
        This method performs the complete EM workflow:
        1. Initialization via PCA and OLS
        2. EM iterations until convergence
        3. Final Kalman smoothing
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N), where T is time periods and N is number of series.
        config : DFMConfig
            Unified DFM configuration object.
        **kwargs
            Additional parameters that override config values:
            - threshold: EM convergence threshold
            - max_iter: Maximum EM iterations
            - ar_lag: AR lag for factors
            - nan_method: Missing data handling method
            - And other DFM-specific parameters
            
        Returns
        -------
        DFMResult
            Estimation results including parameters, factors, and diagnostics.
        """
        # Store config and data
        self._config = config
        self._data = X
        
        # Create DFMParams from kwargs
        params = DFMParams.from_kwargs(**kwargs)
        
        # Call the core estimation function
        result = _dfm_core(X, config, params=params)
        
        self._result = result
        return result
    
    def predict(self, horizon: Optional[int] = None, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values using the fitted model.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x N)
            If only return_factors is True:
                Z_forecast (horizon x m)
        """
        if self._result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return_series = kwargs.get('return_series', True)
        return_factors = kwargs.get('return_factors', True)
        
        # Default horizon: 1 year of periods based on clock frequency
        if horizon is None:
            from ...engine.utils import get_periods_per_year
            from ...engine.helpers import get_clock_frequency
            clock = get_clock_frequency(self._config, 'm')
            horizon = get_periods_per_year(clock)
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        # Extract model parameters
        A = self._result.A
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]
        
        # Deterministic forecast: iteratively apply transition matrix A
        Z_forecast = np.zeros((horizon, Z_last.shape[0]))
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Transform factors to observed series: X = Z @ C^T, then denormalize
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast

