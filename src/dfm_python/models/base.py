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
from typing import Optional, Union, Tuple, Any, Dict, List
from pathlib import Path
import numpy as np

from ..config import (
    DFMConfig, make_config_source, ConfigSource, MergedConfigSource,
    BaseResult, SeriesConfig
)
from ..functional.dfm_block import DEFAULT_BLOCK_NAME
from ..logger import get_logger
from ..utils.errors import ConfigurationError, ModelNotTrainedError, ModelNotInitializedError

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
        
        Returns
        -------
        DFMConfig
            Current model configuration object
            
        Raises
        ------
        ConfigurationError
            If model configuration has not been set
        """
        if not hasattr(self, '_config') or self._config is None:
            model_type = self.__class__.__name__
            raise ConfigurationError(
                f"{model_type} config access failed: model configuration has not been set",
                details="Please call load_config() or pass config to __init__() first"
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
            
            raise ModelNotTrainedError(
                f"{self.__class__.__name__} operation failed: model has not been trained yet",
                details="Please call fit() or train the model before accessing results"
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
        
        from ..config.constants import DEFAULT_CLOCK_FREQUENCY
        return DFMConfig(
            series=[SeriesConfig(series_id='temp', frequency=DEFAULT_CLOCK_FREQUENCY)],
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
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> DFMConfig:
        """Common config loading logic shared by all models."""
        config_source = make_config_source(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        if isinstance(config_source, MergedConfigSource):
            new_config = config_source.get_config()
        else:
            new_config = config_source.get_config()
        
        self._config = new_config
        return new_config
    
    def _get_datamodule(self) -> Any:  # type: ignore[return]  # Returns Union[KDFMDataModule, DFMDataModule, DDFMDataModule] but Any avoids circular imports
        """Get DataModule from model or trainer.
        
        This method attempts to retrieve the DataModule from:
        1. Model's _data_module attribute (if set directly)
        2. Trainer's datamodule attribute (if model has trainer and trainer.fit() was called)
        
        This is a common helper used by predict() methods to access data preprocessing
        parameters (Mx, Wx) and target series configuration.
        
        **Type Note**: Returns `Any` to avoid circular imports. Actual return type is
        `Union[KDFMDataModule, DFMDataModule, DDFMDataModule]` depending on model type.
        The `# type: ignore[return]` comment suppresses the type checker warning.
        This is a known limitation due to circular import constraints - the return type
        is type-safe at runtime but cannot be expressed in the type system without
        circular dependencies.
        
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
        >>> Mx, Wx = data_module.get_std_params()
        >>> target_series = data_module.get_target_series()
        """
        data_module = getattr(self, '_data_module', None)
        
        if data_module is None and hasattr(self, 'trainer') and self.trainer is not None:
            data_module = getattr(self.trainer, 'datamodule', None)
        
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
        p: int,
        horizon: int,
        Z_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecast factors using VAR dynamics.
        
        Supports VAR(1) and VAR(2) factor dynamics (maximum supported order is VAR(2)).
        Uses iterative matrix multiplication for efficient computation.
        
        **VAR(1) Dynamics**: f_t = A @ f_{t-1}
        **VAR(2) Dynamics**: f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
        
        This method is used by both KDFM and DDFM for factor forecasting, ensuring
        consistent VAR dynamics computation across models.
        
        Parameters
        ----------
        Z_last : np.ndarray
            Last factor state of shape (m,) where m is number of factors.
            Must be 1D array with m >= 1.
        A : np.ndarray
            Transition matrix:
            - For VAR(1): shape (m, m)
            - For VAR(2): shape (m, 2m) where A[:, :m] = A1, A[:, m:] = A2
            Must be 2D array with compatible dimensions.
        p : int
            VAR order. Must be 1 or 2 (maximum supported order is VAR(2)).
            Values outside this range will raise ConfigurationError.
        horizon : int
            Number of periods to forecast. Must be >= 1.
        Z_prev : np.ndarray, optional
            Previous factor state for VAR(2) of shape (m,). Required if p == 2.
            If None and p == 2, falls back to VAR(1) dynamics.
            
        Returns
        -------
        np.ndarray
            Forecasted factors of shape (horizon, m) where:
            - horizon: Number of forecast periods
            - m: Number of factors (matches Z_last.shape[0])
            
        Raises
        ------
        ConfigurationError
            If VAR order p is not 1 or 2 (unsupported order)
        DataValidationError
            If input shapes are incompatible or invalid
        NumericalError
            If matrix operations produce NaN/Inf values
            If forecast computation produces NaN/Inf values
            
        Examples
        --------
        >>> # VAR(1) forecast
        >>> Z_last = np.array([1.0, 2.0, 3.0])  # 3 factors
        >>> A = np.eye(3) * 0.9  # Stable transition matrix
        >>> Z_forecast = model._forecast_var_factors(Z_last, A, p=1, horizon=5)
        >>> assert Z_forecast.shape == (5, 3)
        """
        from ..utils.errors import DataValidationError, NumericalError
        from ..numeric.validator import validate_no_nan_inf
        
        # Validate inputs
        if not isinstance(Z_last, np.ndarray) or Z_last.ndim != 1:
            raise DataValidationError(
                f"Z_last must be 1D numpy array, got shape {Z_last.shape if isinstance(Z_last, np.ndarray) else 'not array'}"
            )
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise DataValidationError(
                f"A must be 2D numpy array, got shape {A.shape if isinstance(A, np.ndarray) else 'not array'}"
            )
        
        m = Z_last.shape[0]
        if m < 1:
            raise DataValidationError(f"Z_last must have at least 1 factor, got m={m}")
        
        if horizon < 1:
            raise DataValidationError(f"horizon must be >= 1, got {horizon}")
        
        validate_no_nan_inf(Z_last, name="Z_last")
        validate_no_nan_inf(A, name="transition matrix A")
        
        if Z_prev is not None:
            validate_no_nan_inf(Z_prev, name="Z_prev")
            if Z_prev.shape != (m,):
                raise DataValidationError(
                    f"Z_prev must have shape ({m},), got {Z_prev.shape}"
                )
        try:
            if p == 1:
                # Validate VAR(1) shape
                if A.shape != (m, m):
                    raise DataValidationError(
                        f"For VAR(1), A must have shape ({m}, {m}), got {A.shape}"
                    )
                
                # VAR(1): f_t = A @ f_{t-1}
                Z_forecast = np.zeros((horizon, m), dtype=np.float64)
                Z_forecast[0, :] = A @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
                    
            elif p == 2:
                # Validate VAR(2) shape
                if A.shape != (m, 2 * m):
                    raise DataValidationError(
                        f"For VAR(2), A must have shape ({m}, {2*m}), got {A.shape}"
                    )
                
                # VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
                A1 = A[:, :m]
                A2 = A[:, m:]
                
                if Z_prev is None:
                    # Fallback to VAR(1) if not enough history
                    Z_forecast = np.zeros((horizon, m), dtype=np.float64)
                    Z_forecast[0, :] = A1 @ Z_last
                    for h in range(1, horizon):
                        Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
                else:
                    Z_forecast = np.zeros((horizon, m), dtype=np.float64)
                    Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                    if horizon > 1:
                        Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                    for h in range(2, horizon):
                        Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
            else:
                raise ConfigurationError(
                    f"{self.__class__.__name__} prediction failed: unsupported VAR order {p}",
                    details="Maximum supported VAR order is VAR(2). Please use factor_order=1 (VAR(1)) or factor_order=2 (VAR(2))"
                )
            
            # Validate output
            validate_no_nan_inf(Z_forecast, name="forecasted factors")
            
            if Z_forecast.shape != (horizon, m):
                raise DataValidationError(
                    f"Forecast output has unexpected shape {Z_forecast.shape}, expected ({horizon}, {m})"
                )
            
            return Z_forecast
            
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            # Specific exceptions for forecast computation failures
            if isinstance(e, (ConfigurationError, DataValidationError)):
                raise
            raise NumericalError(
                f"Forecast computation failed: {e}",
                details=(
                    f"Z_last shape: {Z_last.shape}, A shape: {A.shape}, "
                    f"p={p}, horizon={horizon}. Check: (1) Matrix dimensions, "
                    f"(2) Numerical stability, (3) Input validity."
                )
            ) from e
    
    def _transform_factors_to_observations(
        self,
        Z_forecast: np.ndarray,
        C: np.ndarray,
        Wx: Optional[np.ndarray] = None,
        Mx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Transform forecasted factors to observed series.
        
        This method applies the loading matrix to transform factors to observations,
        then optionally unstandardizes the results using mean (Mx) and standard
        deviation (Wx) parameters. This is a common operation used by KDFM, DFM, and DDFM
        for converting factor forecasts to observation forecasts.
        
        **Transformation**: X = Z @ C^T
        **Unstandardization**: X_original = X_standardized * Wx + Mx
        
        **Note**: If Wx is provided but Mx is None, Mx defaults to zeros (assumes
        data was standardized with zero mean).
        
        Parameters
        ----------
        Z_forecast : np.ndarray
            Forecasted factors of shape (horizon, m) where:
            - horizon: Number of forecast periods
            - m: Number of factors
        C : np.ndarray
            Loading matrix of shape (N, m) where:
            - N: Number of observed variables
            - m: Number of factors (must match Z_forecast.shape[1])
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization of shape (N,).
            If None, no unstandardization is applied (assumes data is already in original scale).
        Mx : np.ndarray, optional
            Mean values for unstandardization of shape (N,).
            If None, no unstandardization is applied (assumes data is already in original scale).
            Note: If Wx is provided but Mx is None, Mx defaults to zeros.
            
        Returns
        -------
        np.ndarray
            Forecasted observations of shape (horizon, N) where:
            - horizon: Number of forecast periods
            - N: Number of observed variables (matches C.shape[0])
            
        Raises
        ------
        DataValidationError
            If input shapes are incompatible or invalid
        NumericalError
            If transformation produces NaN/Inf values
            
        Examples
        --------
        >>> # Transform factors to observations
        >>> Z_forecast = np.random.randn(5, 3)  # 5 periods, 3 factors
        >>> C = np.random.randn(7, 3)  # 7 variables, 3 factors
        >>> Wx = np.ones(7) * 10.0  # Standard deviations
        >>> Mx = np.zeros(7)  # Means
        >>> X_forecast = model._transform_factors_to_observations(Z_forecast, C, Wx, Mx)
        >>> assert X_forecast.shape == (5, 7)
        """
        from ..utils.errors import DataValidationError, NumericalError
        from ..numeric.validator import validate_no_nan_inf
        
        # Validate inputs
        if not isinstance(Z_forecast, np.ndarray) or Z_forecast.ndim != 2:
            raise DataValidationError(
                f"Z_forecast must be 2D numpy array, got shape {Z_forecast.shape if isinstance(Z_forecast, np.ndarray) else 'not array'}"
            )
        if not isinstance(C, np.ndarray) or C.ndim != 2:
            raise DataValidationError(
                f"C must be 2D numpy array, got shape {C.shape if isinstance(C, np.ndarray) else 'not array'}"
            )
        
        horizon, m = Z_forecast.shape
        N, m_C = C.shape
        
        if m != m_C:
            raise DataValidationError(
                f"Shape mismatch: Z_forecast has {m} factors, but C has {m_C} factors"
            )
        
        validate_no_nan_inf(Z_forecast, name="Z_forecast")
        validate_no_nan_inf(C, name="loading matrix C")
        
        if Wx is not None:
            if not isinstance(Wx, np.ndarray) or Wx.ndim != 1:
                raise DataValidationError(
                    f"Wx must be 1D numpy array, got shape {Wx.shape if isinstance(Wx, np.ndarray) else 'not array'}"
                )
            if Wx.shape[0] != N:
                raise DataValidationError(
                    f"Wx must have {N} elements (matches C.shape[0]), got {Wx.shape[0]}"
                )
            validate_no_nan_inf(Wx, name="Wx")
        
        if Mx is not None:
            if not isinstance(Mx, np.ndarray) or Mx.ndim != 1:
                raise DataValidationError(
                    f"Mx must be 1D numpy array, got shape {Mx.shape if isinstance(Mx, np.ndarray) else 'not array'}"
                )
            if Mx.shape[0] != N:
                raise DataValidationError(
                    f"Mx must have {N} elements (matches C.shape[0]), got {Mx.shape[0]}"
                )
            validate_no_nan_inf(Mx, name="Mx")
        try:
            # Transform factors to observations: X = Z @ C^T
            X_forecast_std = Z_forecast @ C.T  # (horizon x N)
            
            # Validate transformation output
            validate_no_nan_inf(X_forecast_std, name="X_forecast_std")
            
            if X_forecast_std.shape != (horizon, N):
                raise DataValidationError(
                    f"Transformation output has unexpected shape {X_forecast_std.shape}, expected ({horizon}, {N})"
                )
            
            # Unstandardize if parameters provided
            if Wx is not None or Mx is not None:
                # Handle Wx with shape validation
                if Wx is not None and len(Wx) != N:
                    if len(Wx) > N:
                        Wx_clean = Wx[:N]
                    else:
                        Wx_clean = np.ones(N, dtype=np.float64)
                        Wx_clean[:len(Wx)] = Wx
                else:
                    Wx_clean = Wx if Wx is not None else np.ones(N, dtype=np.float64)
                
                # Handle Mx with shape validation
                if Mx is not None and len(Mx) != N:
                    if len(Mx) > N:
                        Mx_clean = Mx[:N]
                    else:
                        Mx_clean = np.zeros(N, dtype=np.float64)
                        Mx_clean[:len(Mx)] = Mx
                else:
                    Mx_clean = Mx if Mx is not None else np.zeros(N, dtype=np.float64)
                
                # Unstandardize: X_original = X_standardized * Wx + Mx
                X_forecast = X_forecast_std * Wx_clean + Mx_clean
            else:
                # No unstandardization
                X_forecast = X_forecast_std
            
            # Validate final output
            validate_no_nan_inf(X_forecast, name="X_forecast")
            
            if X_forecast.shape != (horizon, N):
                raise DataValidationError(
                    f"Final forecast output has unexpected shape {X_forecast.shape}, expected ({horizon}, {N})"
                )
            
            return X_forecast
            
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            # Specific exceptions for forecast computation failures
            if isinstance(e, (DataValidationError, NumericalError)):
                raise
            raise NumericalError(
                f"Factor-to-observation transformation failed: {e}",
                details=(
                    f"Z_forecast shape: {Z_forecast.shape}, C shape: {C.shape}, "
                    f"Wx shape: {Wx.shape if Wx is not None else None}, "
                    f"Mx shape: {Mx.shape if Mx is not None else None}. "
                    f"Check: (1) Matrix dimensions, (2) Numerical stability, (3) Input validity."
                )
            ) from e
    
    def _resolve_target_series(
        self,
        target: Optional[List[str]],
        series_ids: Optional[List[str]] = None,
        result: Optional[BaseResult] = None
    ) -> Tuple[Optional[List[str]], Optional[List[int]]]:
        """Resolve target series from various sources.
        
        This helper method consolidates the logic for resolving target series
        across all models. It attempts to get target series from:
        1. Explicit target parameter
        2. DataModule's target_series attribute
        3. Result's series_ids (if available)
        
        Parameters
        ----------
        target : List[str], optional
            Explicit target series IDs. If provided, this takes precedence.
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
        ValueError
            If target cannot be resolved and is required
        DataError
            If target series are not found in available series
        """
        from ..utils.errors import DataError
        
        # Use explicit target if provided
        if target is not None and len(target) > 0:
            target_series = target
        else:
            # Try to get from DataModule
            try:
                data_module = self._get_datamodule()
                target_series = getattr(data_module, 'target_series', None)
                if target_series is not None and len(target_series) > 0:
                    target_series = target_series if isinstance(target_series, list) else [target_series]
            except (ModelNotInitializedError, AttributeError):
                target_series = None
        
        # If still None, try result
        if target_series is None and result is not None:
            if hasattr(result, 'series_ids') and result.series_ids is not None:
                # Use all series as fallback (not ideal, but better than error)
                target_series = result.series_ids
        
        # Get series_ids for validation
        if series_ids is None:
            if result is not None and hasattr(result, 'series_ids'):
                series_ids = result.series_ids
            elif self._config is not None:
                series_ids = self._config.get_series_ids()
        
        # Resolve indices if we have both target_series and series_ids
        target_indices = None
        if target_series is not None and series_ids is not None:
            target_indices = []
            for tgt_id in target_series:
                if tgt_id in series_ids:
                    target_indices.append(series_ids.index(tgt_id))
                else:
                    # Log warning but continue - user may want partial results
                    _logger.warning(
                        f"{self.__class__.__name__} prediction: target series '{tgt_id}' not found in series_ids. "
                        f"Available: {series_ids[:10]}{'...' if len(series_ids) > 10 else ''}. "
                        f"Skipping this target series."
                    )
            
            if len(target_indices) == 0:
                raise DataError(
                    f"{self.__class__.__name__} prediction failed: none of the specified target series found",
                    details=f"Target: {target_series}, Available: {series_ids[:20]}{'...' if len(series_ids) > 20 else ''}"
                )
        
        return target_series, target_indices
    
    def _compute_default_horizon(self, default: Optional[int] = None) -> int:
        """Compute default forecast horizon from clock frequency.
        
        Parameters
        ----------
        default : int, optional
            Default value to use if clock frequency cannot be determined.
            If None, uses 12 periods.
            
        Returns
        -------
        int
            Default horizon in periods (typically 1 year worth of periods)
        """
        if default is None:
            default = 12
        
        try:
            from ..config import get_periods_per_year
            from ..utils.misc import get_clock_frequency
            
            if self._config is not None:
                clock = get_clock_frequency(self._config)
                return get_periods_per_year(clock)
        except (AttributeError, ImportError, ValueError):
            _logger.debug(f"Could not determine horizon from clock frequency, using default={default}")
        
        return default
    
    def _compute_default_horizon(self, default: Optional[int] = None) -> int:
        """Compute default forecast horizon from clock frequency.
        
        Parameters
        ----------
        default : int, optional
            Default value to use if clock frequency cannot be determined.
            If None, uses 12 periods.
            
        Returns
        -------
        int
            Default horizon in periods (typically 1 year worth of periods)
        """
        if default is None:
            default = 12
        
        try:
            from ..config import get_periods_per_year
            from ..utils.misc import get_clock_frequency
            
            if self._config is not None:
                clock = get_clock_frequency(self._config)
                return get_periods_per_year(clock)
        except (AttributeError, ImportError, ValueError):
            _logger.debug(f"Could not determine horizon from clock frequency, using default={default}")
        
        return default
    
    @staticmethod
    def _resolve_param(override: Optional[Any], config_value: Optional[Any] = None, default: Any = None) -> Any:
        """Resolve parameter value with priority: override > config > default.
        
        This helper consolidates the common pattern of resolving parameters
        from multiple sources (constructor override, config, default constant).
        
        Note: This is a wrapper around the standalone resolve_param function
        from utils.misc for consistency within model classes.
        
        Parameters
        ----------
        override : Any, optional
            Override value (highest priority). If not None, this value is returned.
        config_value : Any, optional
            Config value (medium priority). Used if override is None.
        default : Any, optional
            Default value (lowest priority). Used if both override and config_value are None.
            
        Returns
        -------
        Any
            Resolved parameter value
            
        Examples
        --------
        >>> # Resolve with override
        >>> value = BaseFactorModel._resolve_param(override=0.01, default=0.001)
        0.01
        
        >>> # Resolve with default
        >>> value = BaseFactorModel._resolve_param(override=None, default=0.001)
        0.001
        
        >>> # Resolve with config
        >>> value = BaseFactorModel._resolve_param(override=None, config_value=0.005, default=0.001)
        0.005
        """
        from ..utils.misc import resolve_param
        return resolve_param(override, config_value, default)
    
    @staticmethod
    @staticmethod
    def _clean_mx_wx(
        Mx: Optional[np.ndarray],
        Wx: Optional[np.ndarray],
        n_series: int,
        dtype: Optional[type] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clean Mx/Wx arrays by replacing NaN with defaults.
        
        This helper consolidates the common pattern of cleaning Mx/Wx arrays
        that may contain NaN values or be None. Used consistently across DFM,
        DDFM, and KDFM models for standardizing data preprocessing.
        
        Parameters
        ----------
        Mx : np.ndarray, optional
            Mean values array (may contain NaN or be None)
        Wx : np.ndarray, optional
            Standard deviation values array (may contain NaN or be None)
        n_series : int
            Number of series (for creating default arrays)
        dtype : type, optional
            Data type for arrays. If None, uses DEFAULT_DTYPE.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cleaned (Mx_clean, Wx_clean) arrays with NaN replaced by defaults.
            Mx_clean: mean values (default 0.0), Wx_clean: std values (default 1.0)
            
        Examples
        --------
        >>> Mx = np.array([1.0, np.nan, 2.0])
        >>> Wx = np.array([0.5, np.nan, 1.0])
        >>> Mx_clean, Wx_clean = BaseFactorModel._clean_mx_wx(Mx, Wx, n_series=3)
        >>> assert np.allclose(Mx_clean, [1.0, 0.0, 2.0])
        >>> assert np.allclose(Wx_clean, [0.5, 1.0, 1.0])
        """
        from ..config.constants import DEFAULT_DTYPE
        if dtype is None:
            dtype = DEFAULT_DTYPE
        
        # Constants for default values
        DEFAULT_WX_VALUE = 1.0  # Default standard deviation when Wx is missing
        DEFAULT_MX_VALUE = 0.0  # Default mean when Mx is missing
        
        # Clean Wx: replace NaN with default, or create default array if None
        if Wx is not None:
            Wx_clean = np.where(np.isnan(Wx), DEFAULT_WX_VALUE, Wx).astype(dtype)
        else:
            Wx_clean = np.ones(n_series, dtype=dtype) * DEFAULT_WX_VALUE
        
        # Clean Mx: replace NaN with default, or create default array if None
        if Mx is not None:
            Mx_clean = np.where(np.isnan(Mx), DEFAULT_MX_VALUE, Mx).astype(dtype)
        else:
            Mx_clean = np.zeros(n_series, dtype=dtype)
        
        return Mx_clean, Wx_clean
    
    def _standardize_data(self, X: np.ndarray, Mx: Optional[np.ndarray], Wx: Optional[np.ndarray]) -> np.ndarray:
        """Standardize data using mean and standard deviation.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N)
        Mx : np.ndarray, optional
            Mean values (N,)
        Wx : np.ndarray, optional
            Standard deviation values (N,)
            
        Returns
        -------
        np.ndarray
            Standardized data (T x N)
        """
        if Mx is not None and Wx is not None:
            # Standardize: X_std = (X - Mx) / Wx
            # Use 1.0 as default when Wx is zero (avoid division by zero)
            DEFAULT_WX_FALLBACK = 1.0
            X_std = (X - Mx) / np.where(Wx != 0, Wx, DEFAULT_WX_FALLBACK)
            return X_std
        return X
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state."""
        self._config = None
        self._result = None
        self.training_state = None
        self._data_module = None
        return self
    
    @abstractmethod
    def get_result(self) -> BaseResult:
        """Extract result from trained model.
        
        Returns
        -------
        BaseResult
            Model-specific result object
        """
        raise NotImplementedError("Subclasses must implement get_result()")
