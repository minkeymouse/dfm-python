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
import torch
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
    
    def _compute_default_horizon(self) -> int:
        """Compute default horizon based on clock frequency.
        
        Returns
        -------
        int
            Default horizon (1 year of periods based on clock frequency)
        """
        from ..config.utils import get_periods_per_year
        from ..utils.helpers import get_clock_frequency
        clock = get_clock_frequency(self.config, 'm')
        return get_periods_per_year(clock)
    
    def _validate_horizon(self, horizon: int) -> None:
        """Validate horizon parameter.
        
        Parameters
        ----------
        horizon : int
            Horizon value to validate
            
        Raises
        ------
        ValueError
            If horizon is not a positive integer
        """
        if horizon <= 0:
            raise ValueError(
                self._format_error_message(
                    operation="prediction",
                    reason=f"horizon must be a positive integer, got {horizon}",
                    guidance="Please provide a positive integer value for the forecast horizon"
                )
            )
    
    def _forecast_var_factors(
        self,
        Z_last: np.ndarray,
        A: np.ndarray,
        p: int,
        horizon: int,
        Z_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecast factors using VAR dynamics.
        
        Supports VAR(1) and VAR(2) factor dynamics.
        
        Parameters
        ----------
        Z_last : np.ndarray
            Last factor state (m,)
        A : np.ndarray
            Transition matrix. For VAR(1): (m x m), for VAR(2): (m x 2m)
        p : int
            VAR order (1 or 2)
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
                self._format_error_message(
                    operation="prediction",
                    reason=f"unsupported VAR order {p}",
                    guidance="Only VAR(1) and VAR(2) are supported. Please use factor_order=1 or factor_order=2"
                )
            )
        return Z_forecast
    
    def _transform_factors_to_observations(
        self,
        Z_forecast: np.ndarray,
        C: np.ndarray,
        Wx: np.ndarray,
        Mx: np.ndarray
    ) -> np.ndarray:
        """Transform forecasted factors to observed series.
        
        Parameters
        ----------
        Z_forecast : np.ndarray
            Forecasted factors (horizon x m)
        C : np.ndarray
            Loading matrix (N x m)
        Wx : np.ndarray
            Standard deviation values for unstandardization (N,)
        Mx : np.ndarray
            Mean values for unstandardization (N,)
            
        Returns
        -------
        np.ndarray
            Forecasted observations (horizon x N)
        """
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        return X_forecast
    
    def _update_factor_state_with_history(
        self,
        history: int,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Update factor state using recent N periods of data.
        
        This method re-runs the Kalman filter (for DFM) or extracts factors
        via encoder and runs Kalman filter (for DDFM) using only the most
        recent N periods of data for efficiency.
        
        Parameters
        ----------
        history : int
            Number of recent periods to use for state update
        result : BaseResult
            Model result containing parameters (A, C, Q, R, Z_0, V_0)
        kalman_filter : Any, optional
            Kalman filter instance. If None, creates a new one.
            
        Returns
        -------
        np.ndarray or None
            Updated last factor state (m,), or None if update failed
        """
        # Get data from DataModule
        if not hasattr(self, '_data_module') or self._data_module is None:
            # Try to get from trainer
            if hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                    self._data_module = self.trainer.datamodule
        
        if not hasattr(self, '_data_module') or self._data_module is None:
            _logger.warning(
                f"{self.__class__.__name__} predict(): DataModule not available for history={history}, "
                f"using training state instead"
            )
            return None
        
        try:
            # Get processed data (T x N)
            X_torch = self._data_module.get_processed_data()
            
            # Convert to numpy if needed
            if isinstance(X_torch, torch.Tensor):
                X_data = X_torch.cpu().numpy()
            else:
                X_data = np.asarray(X_torch)
            
            # Slice to recent N periods
            if X_data.shape[0] > history:
                X_recent = X_data[-history:, :]
                _logger.debug(
                    f"{self.__class__.__name__} predict(): Using recent {history} periods "
                    f"(out of {X_data.shape[0]}) for state update"
                )
            else:
                X_recent = X_data
                _logger.debug(
                    f"{self.__class__.__name__} predict(): Using all {X_data.shape[0]} periods "
                    f"(less than history={history}) for state update"
                )
            
            # Standardize data using result's Mx and Wx
            Mx = result.Mx
            Wx = result.Wx
            if Mx is not None and Wx is not None:
                # Check if data is already standardized (mean ~0, std ~1)
                if np.abs(np.nanmean(X_recent)) > 0.1 or np.nanstd(X_recent) > 2.0:
                    # Data not standardized, standardize it
                    X_recent_std = (X_recent - Mx) / np.where(Wx != 0, Wx, 1.0)
                else:
                    # Data already standardized
                    X_recent_std = X_recent
            else:
                X_recent_std = X_recent
            
            # Handle NaN values (missing data)
            X_recent_std = np.where(np.isfinite(X_recent_std), X_recent_std, np.nan)
            
            # Model-specific state update
            # Use __class__.__name__ to avoid circular imports
            # Check if model has encoder attribute (DDFM-specific)
            if hasattr(self, 'encoder') and self.encoder is not None:
                # DDFM: Extract factors via encoder, then use Kalman filter
                return self._update_factor_state_ddfm(
                    X_recent_std, result, kalman_filter
                )
            else:
                # DFM: Use Kalman filter directly on standardized data
                return self._update_factor_state_dfm(
                    X_recent_std, result, kalman_filter
                )
        except Exception as e:
            _logger.warning(
                f"{self.__class__.__name__} predict(): Failed to update factor state with history={history}, "
                f"using training state instead. Error: {type(e).__name__}: {str(e)}"
            )
            return None
    
    def _update_factor_state_dfm(
        self,
        X_recent_std: np.ndarray,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> np.ndarray:
        """Update factor state for DFM using Kalman filter.
        
        Parameters
        ----------
        X_recent_std : np.ndarray
            Standardized recent data (T x N)
        result : BaseResult
            Model result containing parameters
        kalman_filter : Any, optional
            Kalman filter instance
            
        Returns
        -------
        np.ndarray
            Updated last factor state (m,)
        """
        import torch
        
        # Convert to torch tensor: (N x T) format for Kalman filter
        Y = torch.tensor(X_recent_std.T, dtype=torch.float32)  # (N x T)
        
        # Extract parameters
        A = result.A
        C = result.C
        Q = result.Q
        R = result.R
        Z_0 = result.Z_0
        V_0 = result.V_0
        
        # Convert parameters to torch
        A_torch = torch.tensor(A, dtype=torch.float32)
        C_torch = torch.tensor(C, dtype=torch.float32)
        Q_torch = torch.tensor(Q, dtype=torch.float32)
        R_torch = torch.tensor(R, dtype=torch.float32)
        Z_0_torch = torch.tensor(Z_0, dtype=torch.float32)
        V_0_torch = torch.tensor(V_0, dtype=torch.float32)
        
        # Re-run Kalman filter with recent data
        if kalman_filter is None:
            if hasattr(self, 'kalman') and self.kalman is not None:
                kalman_filter = self.kalman
            else:
                from ..ssm.kalman import KalmanFilter
                kalman_filter = KalmanFilter(
                    min_eigenval=1e-8,
                    inv_regularization=1e-6,
                    cholesky_regularization=1e-8
                )
        
        # Run Kalman smoother
        zsmooth, Vsmooth, _, _ = kalman_filter(
            Y, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch
        )
        
        # zsmooth is (m x (T+1)), transpose to ((T+1) x m)
        Zsmooth = zsmooth.T  # ((T+1) x m)
        
        # Get last factor state (skip initial state at index 0)
        Z_last = Zsmooth[-1, :].cpu().numpy()  # (m,)
        
        return Z_last
    
    def _update_factor_state_ddfm(
        self,
        X_recent_std: np.ndarray,
        result: 'BaseResult',
        kalman_filter: Optional[Any] = None
    ) -> np.ndarray:
        """Update factor state for DDFM using encoder and Kalman filter.
        
        For DDFM, we first extract factors using the encoder, then run
        Kalman filter on those factors to get smoothed state.
        
        Parameters
        ----------
        X_recent_std : np.ndarray
            Standardized recent data (T x N)
        result : BaseResult
            Model result containing parameters
        kalman_filter : Any, optional
            Kalman filter instance
            
        Returns
        -------
        np.ndarray
            Updated last factor state (m,)
        """
        import torch
        
        # Extract factors using encoder
        # Note: encoder is a DDFM-specific attribute, accessed via model instance
        if not hasattr(self, 'encoder') or self.encoder is None:
            _logger.warning(
                f"{self.__class__.__name__} predict(): Encoder not available for history update, "
                "using training state instead"
            )
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.eval()
        
        # Convert to torch and extract factors
        X_tensor = torch.tensor(X_recent_std, device=device, dtype=torch.float32)
        with torch.no_grad():
            factors_raw = self.encoder(X_tensor).cpu().numpy()  # (T x m)
        
        # Extract parameters for Kalman filter
        A = result.A
        Q = result.Q
        Z_0 = result.Z_0
        V_0 = result.V_0
        p = result.p  # VAR order
        
        # For DDFM, the measurement equation is identity (factors are directly observed)
        # So C = I (identity matrix), R = small noise
        m = factors_raw.shape[1]
        C = np.eye(m)  # Identity matrix (factors are directly observed)
        R = np.eye(m) * 1e-8  # Small noise for numerical stability
        
        # Convert to torch tensor: (m x T) format for Kalman filter
        # Note: For DDFM, we filter factors directly, so Y = factors.T
        Y = torch.tensor(factors_raw.T, dtype=torch.float32)  # (m x T)
        
        # Convert parameters to torch
        A_torch = torch.tensor(A, dtype=torch.float32)
        C_torch = torch.tensor(C, dtype=torch.float32)
        Q_torch = torch.tensor(Q, dtype=torch.float32)
        R_torch = torch.tensor(R, dtype=torch.float32)
        Z_0_torch = torch.tensor(Z_0, dtype=torch.float32)
        V_0_torch = torch.tensor(V_0, dtype=torch.float32)
        
        # Create or use Kalman filter
        if kalman_filter is None:
            from ..ssm.kalman import KalmanFilter
            kalman_filter = KalmanFilter(
                min_eigenval=1e-8,
                inv_regularization=1e-6,
                cholesky_regularization=1e-8
            )
        
        # Run Kalman smoother on factors
        zsmooth, Vsmooth, _, _ = kalman_filter(
            Y, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch
        )
        
        # zsmooth is (m x (T+1)), transpose to ((T+1) x m)
        Zsmooth = zsmooth.T  # ((T+1) x m)
        
        # Get last factor state (skip initial state at index 0)
        Z_last = Zsmooth[-1, :].cpu().numpy()  # (m,)
        
        return Z_last
    
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
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, uses default based on clock frequency.
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
            
        Notes
        -----
        When history is specified, the method uses only the most recent N periods for
        Kalman filter update, improving computational efficiency. The initial state
        (Z_0, V_0) is always estimated from full history (including any new data beyond
        training period), ensuring accuracy while maintaining efficiency.
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
    
    @property
    def nowcast(self) -> 'Nowcast':
        """Get nowcast manager for generating nowcasts.
        
        Returns
        -------
        Nowcast
            Nowcast manager object that provides methods for:
            - Generating nowcasts (forecasts for current period)
            - Computing revision impacts
            - Analyzing data releases
            - News decomposition
            
        Raises
        ------
        ValueError
            If model has not been trained yet. Call trainer.fit(model, data_module) first.
            
        Examples
        --------
        >>> model = DFM()
        >>> model.load_config('config.yaml')
        >>> trainer.fit(model, data_module)
        >>> nowcast = model.nowcast
        >>> value = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
        """
        self._check_trained()
        
        # Import here to avoid circular dependency
        try:
            from ..nowcast import Nowcast
        except ImportError:
            # Fallback: try importing from src.nowcasting if dfm_python.nowcast doesn't exist
            try:
                import sys
                from pathlib import Path
                # Add src to path if not already there
                src_path = Path(__file__).parent.parent.parent.parent / 'src'
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                from nowcasting import Nowcast
            except ImportError:
                error_msg = self._format_error_message(
                    operation="nowcast access",
                    reason="Nowcast class not available",
                    guidance="Please ensure nowcasting module is available"
                )
                raise ValueError(error_msg)
        
        # Get DataModule from model
        data_module = getattr(self, '_data_module', None)
        if data_module is None:
            # Try to get from trainer if available
            if hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                    data_module = self.trainer.datamodule
        
        return Nowcast(self, data_module=data_module)
    
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

