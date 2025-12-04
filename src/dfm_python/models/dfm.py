"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
DFM is a PyTorch Lightning module that inherits from BaseFactorModel.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any, List, Dict, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass
from ..logger import get_logger

from .base import BaseFactorModel
from ..config import DFMConfig, SeriesConfig, validate_frequency
from ..config.results import DFMResult, FitParams
from ..ssm.kalman import KalmanFilter
from ..ssm.em import EMAlgorithm, EMStepParams

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class DFMTrainingState:
    """State tracking for DFM training."""
    A: torch.Tensor
    C: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    Z_0: torch.Tensor
    V_0: torch.Tensor
    loglik: float
    num_iter: int
    converged: bool


class DFMLinear:
    """Linear Dynamic Factor Model using EM algorithm (low-level implementation).
    
    This class implements the standard linear DFM with EM estimation.
    It provides low-level functionality for DFM estimation.
    
    The model assumes:
    - Linear observation equation: y_t = C Z_t + e_t
    - Linear factor dynamics: Z_t = A Z_{t-1} + v_t
    - Gaussian innovations
    
    Parameters are estimated via Expectation-Maximization (EM) algorithm.
    
    Note: This class provides the low-level implementation of the linear DFM.
    The high-level DFM class (PyTorch Lightning module) should be used for training.
    
    Note: DFMLinear.fit() is deprecated. Use DFM class with trainer.fit(model, dm) instead.
    """
    
    def fit(
        self,
        data_module: 'DFMDataModule',
        config: Optional[DFMConfig] = None,
        *,
        fit_params: Optional[FitParams] = None,
        **kwargs
    ) -> DFMResult:
        """Fit the linear DFM model using EM algorithm.
        
        This method performs the complete EM workflow:
        1. Initialization via PCA and OLS
        2. EM iterations until convergence
        3. Final Kalman smoothing
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        config : DFMConfig, optional
            Unified DFM configuration object. If None, uses config from data_module.
        fit_params : FitParams, optional
            Fit parameters object. If None, parameters are extracted from kwargs or config.
        **kwargs
            Additional parameters that override config values. Merged into fit_params if provided.
            
        Returns
        -------
        DFMResult
            Estimation results including parameters, factors, and diagnostics.
        """
        from ..lightning import DFMDataModule
        
        if not isinstance(data_module, DFMDataModule):
            raise TypeError(
                f"DFM fit failed: data_module must be DFMDataModule, got {type(data_module).__name__}. "
                f"Please provide a valid DFMDataModule instance."
            )
        
        # Ensure DataModule is set up
        if data_module.data_processed is None:
            data_module.setup()
        
        # Use config from data_module if not provided
        if config is None:
            config = data_module.config
        
        # Extract fit parameters from kwargs or use provided FitParams
        fit_params = self._prepare_fit_params(fit_params, **kwargs)
        
        # Get processed data and standardization params from DataModule
        X_torch = data_module.get_processed_data()
        Mx, Wx = data_module.get_standardization_params()
        
        # Handle case where standardization params might be None
        # (if transformer doesn't include StandardScaler)
        if Mx is None or Wx is None:
            # Use zeros/ones as defaults (no standardization)
            N = X_torch.shape[1]
            Mx = np.zeros(N, dtype=np.float32)
            Wx = np.ones(N, dtype=np.float32)
        
        # Get training parameters
        num_factors, training_params = self._extract_training_params(config, fit_params)
        
        # Train model using Lightning
        result = self._train_with_lightning(X_torch, config, num_factors, training_params, Mx, Wx)
        
        # Store results
        self._result = result
        return result
    
    def _prepare_fit_params(self, fit_params: Optional[FitParams], **kwargs) -> FitParams:
        """Prepare fit parameters from kwargs or provided FitParams."""
        if fit_params is None:
            return FitParams.from_kwargs(**kwargs)
        # Merge kwargs into fit_params
        fit_dict = fit_params.to_dict()
        fit_dict.update({k: v for k, v in kwargs.items() if v is not None})
        return FitParams.from_kwargs(**fit_dict)
    
    def _extract_training_params(
        self, config: DFMConfig, fit_params: FitParams
    ) -> Tuple[int, Dict[str, Any]]:
        """Extract training parameters from config and fit_params."""
        # Get parameters for Lightning module
        threshold_val = fit_params.threshold if fit_params.threshold is not None else getattr(config, 'threshold', 1e-4)
        max_iter_val = fit_params.max_iter if fit_params.max_iter is not None else getattr(config, 'max_iter', 100)
        nan_method_val = fit_params.nan_method if fit_params.nan_method is not None else getattr(config, 'nan_method', 2)
        nan_k_val = fit_params.nan_k if fit_params.nan_k is not None else getattr(config, 'nan_k', 3)
        
        # Determine number of factors
        if hasattr(config, 'factors_per_block') and config.factors_per_block:
            num_factors = int(np.sum(config.factors_per_block))
        else:
            blocks = config.get_blocks_array()
            if blocks.shape[1] > 0:
                num_factors = int(np.sum(blocks[:, 0]))
            else:
                num_factors = 1
        
        training_params = {
            'threshold': threshold_val,
            'max_iter': max_iter_val,
            'nan_method': nan_method_val,
            'nan_k': nan_k_val
        }
        
        return num_factors, training_params
    
    def _train_with_lightning(
        self,
        X_torch: torch.Tensor,
        config: DFMConfig,
        num_factors: int,
        training_params: Dict[str, Any],
        Mx: np.ndarray,
        Wx: np.ndarray
    ) -> DFMResult:
        """Train model using PyTorch Lightning."""
        from ..lightning.dfm_module import DFMLightningModule
        
        lightning_module = DFMLightningModule(
            config=config,
            num_factors=num_factors,
            **training_params
        )
        
        # Run EM algorithm
        lightning_module.fit_em(X_torch, Mx=Mx, Wx=Wx)
        
        # Extract results
        return lightning_module.get_result()
    
    def _create_default_config(
        self,
        X: np.ndarray,
        time_index: Optional[Union[List[datetime], np.ndarray, pl.Series]] = None,
        series_ids: Optional[List[str]] = None,
        num_factors: int = 1,
        clock: Optional[str] = None,
        transformation: str = 'lin'
    ) -> DFMConfig:
        """Create default configuration with smart defaults.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N)
        time_index : list of datetime, np.ndarray, or pl.Series, optional
            Time index for frequency inference
        series_ids : list of str, optional
            Series identifiers (auto-generated if None)
        num_factors : int, default 1
            Number of factors in global block
        clock : str, optional
            Clock frequency (inferred from time_index if None)
        transformation : str, default 'lin'
            Transformation code for all series
            
        Returns
        -------
        DFMConfig
            Configuration with smart defaults
        """
        T, N = X.shape
        
        # Auto-generate series_ids if not provided
        if series_ids is None:
            series_ids = [f"series_{i}" for i in range(N)]
        
        # Infer clock frequency if not provided
        if clock is None:
            clock = self._infer_frequency(time_index)
        
        # Validate clock
        clock = validate_frequency(clock)
        
        # Create series configs (all series in global block)
        series_configs = []
        for i, series_id in enumerate(series_ids):
            series_configs.append(
                SeriesConfig(
                    frequency=clock,  # All series use clock frequency
                    transformation=transformation,
                    blocks=['Block_Global'],  # All series load on global block
                    series_id=series_id,
                    series_name=series_id
                )
            )
        
        # Create block config (single global block)
        block_configs = {
            'Block_Global': {
                'factors': num_factors,
                'ar_lag': 1,
                'clock': clock
            }
        }
        
        # Create DFMConfig with defaults
        config = DFMConfig(
            series=series_configs,
            blocks=block_configs,
            clock=clock,
            ar_lag=1,
            threshold=1e-5,
            max_iter=5000,
            nan_method=2,
            nan_k=3
        )
        
        return config
    
    def _infer_frequency(
        self,
        time_index: Optional[Union[List[datetime], np.ndarray, pl.Series]] = None
    ) -> str:
        """Infer frequency from time index.
        
        Parameters
        ----------
        time_index : list of datetime, np.ndarray, or pl.Series, optional
            Time index to analyze
            
        Returns
        -------
        str
            Inferred frequency code ('d', 'w', 'm', 'q', 'sa', 'a')
            Defaults to 'm' if inference fails or time_index is None.
        """
        if time_index is None or len(time_index) < 2:
            _logger.warning("Cannot infer frequency: time_index is None or too short. Defaulting to 'm' (monthly).")
            return 'm'
        
        # Convert to list of datetime if needed
        if isinstance(time_index, pl.Series):
            time_list = time_index.to_list()
        elif isinstance(time_index, np.ndarray):
            time_list = time_index.tolist()
        else:
            time_list = list(time_index)
        
        # Ensure all elements are datetime
        from ..utils.time import parse_timestamp
        try:
            time_list = [parse_timestamp(t) for t in time_list]
        except (ValueError, TypeError):
            _logger.warning("Cannot parse time_index. Defaulting to 'm' (monthly).")
            return 'm'
        
        if len(time_list) < 2:
            return 'm'
        
        # Calculate median time difference
        diffs = []
        for i in range(1, len(time_list)):
            diff = (time_list[i] - time_list[i-1]).total_seconds()
            diffs.append(diff)
        
        if not diffs:
            return 'm'
        
        median_diff_seconds = np.median(diffs)
        median_diff_days = median_diff_seconds / (24 * 3600)
        
        # Infer frequency based on median difference
        if median_diff_days <= 1.5:
            return 'd'  # Daily
        elif median_diff_days <= 5:
            return 'w'  # Weekly
        elif median_diff_days <= 35:
            return 'm'  # Monthly
        elif median_diff_days <= 100:
            return 'q'  # Quarterly
        elif median_diff_days <= 200:
            return 'sa'  # Semi-annual
        else:
            return 'a'  # Annual
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True,
        result: Optional[DFMResult] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        result : DFMResult, optional
            Model result to use for prediction. If None, uses self._result.
            This allows the high-level DFM class to pass its result.
            
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
        # Use provided result or fall back to self._result
        if result is None:
            if self._result is None:
                raise ValueError(
                    "DFM prediction failed: model has not been fitted yet. "
                    "Please call fit() first or provide a result object."
                )
            result = self._result
        else:
            # Use provided result
            pass
        
        # Default horizon: 1 year of periods based on clock frequency
        if horizon is None:
            from ..config.utils import get_periods_per_year
            from ..utils.helpers import get_clock_frequency
            if self._config is None:
                clock = 'm'  # Default to monthly
            else:
                clock = get_clock_frequency(self._config, 'm')
            horizon = get_periods_per_year(clock)
        
        if horizon <= 0:
            raise ValueError(
                f"DFM prediction failed: horizon must be a positive integer, got {horizon}. "
                f"Please provide a positive integer value for the forecast horizon."
            )
        
        # Extract model parameters
        A = result.A
        C = result.C
        Wx = result.Wx
        Mx = result.Mx
        Z_last = result.Z[-1, :]
        
        # Validate that model is properly trained (Z_last should not contain NaN)
        if np.any(np.isnan(Z_last)):
            nan_count = np.sum(np.isnan(Z_last))
            nan_ratio = nan_count / len(Z_last)
            raise ValueError(
                f"DFM prediction failed: {nan_count}/{len(Z_last)} factors contain NaN values ({nan_ratio:.1%}). "
                f"This usually indicates the model did not converge during training. "
                f"Try increasing max_iter, checking data quality, or removing series with high missing data."
            )
        
        # Validate parameters are finite
        if np.any(~np.isfinite(A)) or np.any(~np.isfinite(C)):
            raise ValueError(
                "DFM prediction failed: model parameters (A or C) contain NaN or Inf values. "
                "This indicates the model did not train successfully. "
                "Please check training convergence and data quality."
            )
        
        # Deterministic forecast: iteratively apply transition matrix A
        Z_forecast = np.zeros((horizon, Z_last.shape[0]))
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Transform factors to observed series: X = Z @ C^T, then denormalize
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        # Validate forecast results are finite
        if np.any(~np.isfinite(X_forecast)):
            nan_count = np.sum(~np.isfinite(X_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in forecast. "
                f"This may indicate numerical instability. "
                f"Please check model parameters and data quality."
            )
        if return_factors and np.any(~np.isfinite(Z_forecast)):
            nan_count = np.sum(~np.isfinite(Z_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in factor forecast. "
                f"This may indicate numerical instability in factor dynamics. "
                f"Please check model parameters and training convergence."
            )
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast




# ============================================================================
# High-level API Classes
# ============================================================================

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, TYPE_CHECKING

from ..config import (
    DFMConfig,
    make_config_source,
    ConfigSource,
    MergedConfigSource,
)
from ..nowcast.dataview import DataView
from ..utils.helpers import (
    safe_get_method,
    safe_get_attr,
    get_clock_frequency,
)
from ..utils.time import TimeIndex

if TYPE_CHECKING:
    from omegaconf import DictConfig


class DFM(BaseFactorModel):
    """High-level API for Linear Dynamic Factor Model (PyTorch Lightning module).
    
    This class is a PyTorch Lightning module that can be used with standard
    Lightning training patterns. It inherits from BaseFactorModel and implements
    the EM algorithm for DFM estimation.
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import DFM, DFMDataModule, DFMTrainer
        >>> import polars as pl
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pl.read_csv('data/finance.csv')
        >>> df_processed = df.select([col for col in df.columns if col != 'date'])
        >>> 
        >>> # Step 2: Create DataModule
        >>> dm = DFMDataModule(config_path='config/dfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = DFM()
        >>> model.load_config('config/dfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = DFMTrainer(max_epochs=100)
        >>> trainer.fit(model, dm)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        num_factors: Optional[int] = None,
        threshold: float = 1e-4,
        max_iter: int = 100,
        nan_method: int = 2,
        nan_k: int = 3,
        **kwargs
    ):
        """Initialize DFM instance.
        
        Parameters
        ----------
        config : DFMConfig, optional
            DFM configuration. Can be loaded later via load_config().
        num_factors : int, optional
            Number of factors. If None, inferred from config.
        threshold : float, default 1e-4
            EM convergence threshold
        max_iter : int, default 100
            Maximum EM iterations
        nan_method : int, default 2
            Missing data handling method
        nan_k : int, default 3
            Spline interpolation order
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        super().__init__(**kwargs)
        
        # If config not provided, create a placeholder that will be set via load_config
        if config is None:
            from ..config.schema import DFMConfig, SeriesConfig
            config = DFMConfig(
                series=[SeriesConfig(series_id='placeholder', frequency='m', transformation='lin', blocks=[1])],
                blocks={'Block_0': {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
            )
        
        # Set internal config (config property is read-only, accessed via property getter)
        self._config = config
        self.threshold = threshold
        self.max_iter = max_iter
        self.nan_method = nan_method
        self.nan_k = nan_k
        
        # Determine number of factors
        if num_factors is None:
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                self.num_factors = int(np.sum(config.factors_per_block))
            else:
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    self.num_factors = int(np.sum(blocks[:, 0]))
                else:
                    self.num_factors = 1
        else:
            self.num_factors = num_factors
        
        # Get model structure
        self.r = torch.tensor(
            config.factors_per_block if config.factors_per_block is not None
            else np.ones(config.get_blocks_array().shape[1]),
            dtype=torch.float32
        )
        self.p = getattr(config, 'ar_lag', 1)
        self.blocks = torch.tensor(config.get_blocks_array(), dtype=torch.float32)
        
        # Compose modules as components
        self.kalman = KalmanFilter(
            min_eigenval=1e-8,
            inv_regularization=1e-6,
            cholesky_regularization=1e-8
        )
        self.em = EMAlgorithm(
            kalman=self.kalman,  # Share same KalmanFilter instance
            regularization_scale=1e-6
        )
        
        # Parameters will be initialized in setup() or fit_em()
        self.A: Optional[torch.nn.Parameter] = None
        self.C: Optional[torch.nn.Parameter] = None
        self.Q: Optional[torch.nn.Parameter] = None
        self.R: Optional[torch.nn.Parameter] = None
        self.Z_0: Optional[torch.nn.Parameter] = None
        self.V_0: Optional[torch.nn.Parameter] = None
        
        # Training state
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        
        # Use manual optimization for EM algorithm
        self.automatic_optimization = False
        
        # Low-level implementation for utility methods
        self._model_impl = DFMLinear()
        self._data_module: Optional['DFMDataModule'] = None
        self._nowcast: Optional['Nowcast'] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model parameters.
        
        This is called by Lightning before training starts.
        Parameters are initialized from data if available.
        """
        # Parameters will be initialized during fit_em() or first training step
        pass
    
    def initialize_from_data(self, X: torch.Tensor) -> None:
        """Initialize parameters from data using PCA and OLS.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        """
        opt_nan = {'method': self.nan_method, 'k': self.nan_k}
        
        # Use self.em.initialize_parameters() with direct tensor operations (no CPU transfers)
        A, C, Q, R, Z_0, V_0 = self.em.initialize_parameters(
            X,
            r=self.r.to(X.device),
            p=self.p,
            blocks=self.blocks.to(X.device),
            opt_nan=opt_nan,
            R_mat=None,
            q=None,
            nQ=0,
            i_idio=None,
            clock=getattr(self.config, 'clock', 'm'),
            tent_weights_dict=None,
            frequencies=None,
            idio_chain_lengths=None,
            config=self.config
        )
        
        # Convert to Parameters
        self.A = nn.Parameter(A)
        self.C = nn.Parameter(C)
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)
        self.Z_0 = nn.Parameter(Z_0)
        self.V_0 = nn.Parameter(V_0)
    
    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Perform one EM iteration.
        
        For DFM, each "step" is actually one EM iteration. The batch contains
        the full time series data.
        
        Parameters
        ----------
        batch : torch.Tensor or tuple
            Data tensor (T x N) or (data, target) tuple where data is (T x N) time series
        batch_idx : int
            Batch index (should be 0 for full sequence)
            
        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood (to minimize)
        """
        # Handle both tuple and single tensor batches
        if isinstance(batch, tuple):
            data, _ = batch
        else:
            data = batch
        # data is (batch_size, T, N) or (T, N) depending on DataLoader
        if data.ndim == 3:
            # Take first batch (should only be one for time series)
            data = data[0]
        
        # Initialize parameters if not done yet
        if self.A is None:
            self.initialize_from_data(data)
        
        # Prepare data for EM step
        # EM expects y as (N x T), but data is (T x N)
        y = data.T  # (N x T)
        
        # Create EM step parameters
        em_params = EMStepParams(
            y=y,
            A=self.A,
            C=self.C,
            Q=self.Q,
            R=self.R,
            Z_0=self.Z_0,
            V_0=self.V_0,
            r=self.r.to(y.device),
            p=self.p,
            R_mat=None,
            q=None,
            nQ=0,
            i_idio=torch.ones(y.shape[0], device=y.device, dtype=y.dtype),
            blocks=self.blocks.to(y.device),
            tent_weights_dict={},
            clock=getattr(self.config, 'clock', 'm'),
            frequencies=None,
            idio_chain_lengths=torch.zeros(y.shape[0], device=y.device, dtype=y.dtype),
            config=self.config
        )
        
        # Perform EM step - use self.em(...) instead of em_step(...)
        C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = self.em(em_params)
        
        # Update parameters (EM doesn't use gradients, so we update directly)
        with torch.no_grad():
            self.A.data = A_new
            self.C.data = C_new
            self.Q.data = Q_new
            self.R.data = R_new
            self.Z_0.data = Z_0_new
            self.V_0.data = V_0_new
        
        # Log metrics
        self.log('loglik', loglik, on_step=True, on_epoch=True, prog_bar=True)
        self.log('em_iteration', float(self.current_epoch), on_step=True, on_epoch=True)
        
        # Return negative log-likelihood as loss (to minimize)
        return -torch.tensor(loglik, device=data.device, dtype=data.dtype)
    
    def on_train_epoch_end(self) -> None:
        """Check convergence after each epoch (EM iteration)."""
        if self.training_state is None:
            return
        
        # Check convergence - use self.em.check_convergence() instead of em_converged()
        converged, change = self.em.check_convergence(
            self.training_state.loglik,
            self.training_state.loglik,  # Previous loglik (would need to track)
            self.threshold,
            verbose=False
        )
        
        if converged:
            self.training_state.converged = True
            _logger.info(f"EM algorithm converged at iteration {self.current_epoch}")
    
    def fit_em(
        self,
        X: torch.Tensor,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None
    ) -> DFMTrainingState:
        """Run full EM algorithm until convergence.
        
        This method runs the complete EM algorithm outside of Lightning's
        training loop, which is more natural for EM. Called by trainer during fit().
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        Mx : np.ndarray, optional
            Mean values for unstandardization (N,)
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization (N,)
            
        Returns
        -------
        DFMTrainingState
            Final training state with parameters and convergence info
        """
        self.Mx = Mx
        self.Wx = Wx
        
        # Ensure data is on same device as model (Lightning handles this automatically)
        X = X.to(self.device)
        self.data_processed = X
        
        device = X.device
        dtype = X.dtype
        
        # Initialize parameters
        self.initialize_from_data(X)
        
        # Prepare data for EM
        y = X.T  # (N x T)
        
        # Initialize state
        previous_loglik = float('-inf')
        num_iter = 0
        converged = False
        
        # EM loop
        while num_iter < self.max_iter and not converged:
            # Create EM step parameters
            em_params = EMStepParams(
                y=y,
                A=self.A,
                C=self.C,
                Q=self.Q,
                R=self.R,
                Z_0=self.Z_0,
                V_0=self.V_0,
                r=self.r.to(device),
                p=self.p,
                R_mat=None,
                q=None,
                nQ=0,
                i_idio=torch.ones(y.shape[0], device=device, dtype=dtype),
                blocks=self.blocks.to(device),
                tent_weights_dict={},
                clock=getattr(self.config, 'clock', 'm'),
                frequencies=None,
                idio_chain_lengths=torch.zeros(y.shape[0], device=device, dtype=dtype),
                config=self.config
            )
            
            # Perform EM step - use self.em(...) instead of em_step(...)
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = self.em(em_params)
            
            # Update parameters
            with torch.no_grad():
                self.A.data = A_new
                self.C.data = C_new
                self.Q.data = Q_new
                self.R.data = R_new
                self.Z_0.data = Z_0_new
                self.V_0.data = V_0_new
            
            # Check convergence - use self.em.check_convergence() instead of em_converged()
            if num_iter > 2:
                converged, change = self.em.check_convergence(
                    loglik,
                    previous_loglik,
                    self.threshold,
                    verbose=(num_iter % 10 == 0)
                )
            else:
                change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else 0.0
            
            previous_loglik = loglik
            num_iter += 1
            
            # Log metrics using Lightning (enables TensorBoard, WandB, etc.)
            # Note: on_step=False because fit_em may be called from on_train_start
            self.log('train/loglik', loglik, on_step=False, on_epoch=True)
            self.log('train/em_iteration', float(num_iter), on_step=False, on_epoch=True)
            self.log('train/loglik_change', change, on_step=False, on_epoch=True)
            
            if num_iter % 10 == 0:
                _logger.info(
                    f"EM iteration {num_iter}/{self.max_iter}: "
                    f"loglik={loglik:.4f}, change={change:.2e}"
                )
        
        # Store final state
        self.training_state = DFMTrainingState(
            A=self.A.data.clone(),
            C=self.C.data.clone(),
            Q=self.Q.data.clone(),
            R=self.R.data.clone(),
            Z_0=self.Z_0.data.clone(),
            V_0=self.V_0.data.clone(),
            loglik=loglik,
            num_iter=num_iter,
            converged=converged
        )
        
        return self.training_state
    
    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        if self.training_state is None:
            raise RuntimeError(
                "DFM get_result failed: model has not been fitted yet. "
                "Please call fit_em() first."
            )
        
        if self.data_processed is None:
            raise RuntimeError(
                "DFM get_result failed: data not available. "
                "Please ensure fit_em() was called with data."
            )
        
        # Get final smoothed factors using Kalman filter
        y = self.data_processed.T  # (N x T)
        
        # Run final Kalman smoothing with converged parameters - use self.kalman(...) instead of kalman_filter_smooth(...)
        zsmooth, Vsmooth, _, _ = self.kalman(
            y,
            self.training_state.A,
            self.training_state.C,
            self.training_state.Q,
            self.training_state.R,
            self.training_state.Z_0,
            self.training_state.V_0
        )
        
        # zsmooth is (m x (T+1)), transpose to ((T+1) x m)
        Zsmooth = zsmooth.T
        Z = Zsmooth[1:, :].cpu().numpy()  # T x m (skip initial state)
        
        # Convert parameters to numpy
        A = self.training_state.A.cpu().numpy()
        C = self.training_state.C.cpu().numpy()
        Q = self.training_state.Q.cpu().numpy()
        R = self.training_state.R.cpu().numpy()
        Z_0 = self.training_state.Z_0.cpu().numpy()
        V_0 = self.training_state.V_0.cpu().numpy()
        r = self.r.cpu().numpy()
        
        # Compute smoothed data
        x_sm = Z @ C.T  # T x N (standardized smoothed data)
        
        # Unstandardize
        Wx_clean = np.where(np.isnan(self.Wx), 1.0, self.Wx) if self.Wx is not None else np.ones(C.shape[0])
        Mx_clean = np.where(np.isnan(self.Mx), 0.0, self.Mx) if self.Mx is not None else np.zeros(C.shape[0])
        X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized smoothed data)
        
        # Create result object
        result = DFMResult(
            x_sm=x_sm,
            X_sm=X_sm,
            Z=Z,
            C=C,
            R=R,
            A=A,
            Q=Q,
            Mx=self.Mx if self.Mx is not None else np.zeros(C.shape[0]),
            Wx=self.Wx if self.Wx is not None else np.ones(C.shape[0]),
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=self.p,
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=self.training_state.loglik,
            series_ids=self.config.get_series_ids() if hasattr(self.config, 'get_series_ids') else None,
            block_names=getattr(self.config, 'block_names', None)
        )
        
        return result
    
    def configure_optimizers(self):
        """Configure optimizers.
        
        EM algorithm doesn't use standard optimizers, but Lightning requires
        this method. Return empty list.
        """
        return []
    
    @property
    def nowcast(self) -> 'Nowcast':
        """Get nowcasting manager instance."""
        if self._nowcast is None:
            if self._config is None:
                raise ValueError(
                    "DFM nowcast access failed: configuration has not been loaded yet. "
                    "Please call load_config() first."
                )
            if self._data_module is None:
                raise ValueError(
                    "DFM nowcast access failed: DataModule has not been provided yet. "
                    "Please provide DataModule via trainer.fit() before accessing nowcast."
                )
            if self.training_state is None:
                raise ValueError(
                    "DFM nowcast access failed: model has not been trained yet. "
                    "Please call trainer.fit() first."
                )
            from ..nowcast import Nowcast
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
    ) -> 'DFM':
        """Load configuration from various sources.
        
        After loading config, the model needs to be re-initialized with the new config.
        For standard Lightning pattern, pass config directly to __init__.
        """
        # Use common config loading logic
        new_config = self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        # DFM-specific: Initialize r and blocks tensors
        self.r = torch.tensor(
            new_config.factors_per_block if new_config.factors_per_block is not None
            else np.ones(new_config.get_blocks_array().shape[1]),
            dtype=torch.float32
        )
        self.blocks = torch.tensor(new_config.get_blocks_array(), dtype=torch.float32)
        
        return self
    
    
    def on_train_start(self) -> None:
        """Called when training starts. Run EM algorithm."""
        # Store data_module reference for later use (nowcast, predict, etc.)
        if hasattr(self.trainer, 'datamodule'):
            self._data_module = self.trainer.datamodule
            
            # Get processed data and standardization params from DataModule
            X_torch = self._data_module.get_processed_data()
            Mx, Wx = self._data_module.get_standardization_params()
            
            # Handle case where standardization params might be None
            if Mx is None or Wx is None:
                N = X_torch.shape[1]
                Mx = np.zeros(N, dtype=np.float32)
                Wx = np.ones(N, dtype=np.float32)
            
            # Run EM algorithm
            self.fit_em(X_torch, Mx=Mx, Wx=Wx)
        
        super().on_train_start()
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the Lightning module to generate forecasts.
        """
        if self.training_state is None:
            error_msg = self._format_error_message(
                operation="prediction",
                reason="model has not been trained yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
        
        # Convert training state to result format for prediction
        if not hasattr(self, '_result') or self._result is None:
            self._result = self.get_result()
        
        return self._model_impl.predict(
            horizon=horizon,
            return_series=return_series,
            return_factors=return_factors,
            result=self._result
        )
    
    @property
    def result(self) -> DFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        # Check if trained and extract result from training state if needed
        self._check_trained()
        return self._result
    
    @property
    def config(self) -> DFMConfig:
        """Get model configuration."""
        if not hasattr(self, '_config') or self._config is None:
            raise ValueError(
                "DFM config access failed: model configuration has not been set. "
                "Please call load_config() or pass config to __init__() first."
            )
        return self._config
    
    def plot(self, **kwargs) -> 'DFM':
        """Plot common visualizations."""
        if self.training_state is None:
            error_msg = self._format_error_message(
                operation="plotting",
                reason="model has not been trained yet",
                guidance="Please call trainer.fit(model, data_module) first"
            )
            raise ValueError(error_msg)
        _logger.info("Plot functionality not yet implemented")
        return self
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        self._config = None
        self._data_module = None
        self._result = None
        self._nowcast = None
        if hasattr(self, 'training_state'):
            self.training_state = None
        return self
    
    def load_pickle(self, path: Union[str, Path], **kwargs) -> 'DFM':
        """Load a saved model from pickle file.
        
        Note: DataModule is not saved in pickle. Users must create a new DataModule
        and use trainer.fit() with it after loading the model.
        """
        import pickle
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self._config = payload.get('config')
        self._result = payload.get('result')
        # Note: data_module is not loaded - users must provide it via trainer.fit()
        return self




def _dump_yaml_to_file(path: Path, payload: Dict[str, Any]) -> None:
    """Helper function to dump YAML file."""
    try:
        import yaml  # type: ignore
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except ImportError:
        try:
            from omegaconf import OmegaConf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Either PyYAML or omegaconf is required for YAML generation. "
                "Install with: pip install pyyaml or pip install omegaconf"
            ) from exc
        cfg = OmegaConf.create(payload)
        OmegaConf.save(cfg, path)


def from_spec(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    series_filename: Optional[str] = None,
    blocks_filename: Optional[str] = None
) -> Tuple[Path, Path]:
    """Convert spec CSV file to YAML configuration files."""
    from ..config.adapter import from_spec as _from_spec
    return _from_spec(csv_path, output_dir, series_filename, blocks_filename)

