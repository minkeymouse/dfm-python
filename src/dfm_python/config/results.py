"""Result structures for Dynamic Factor Model estimation.

This module contains dataclasses for storing DFM estimation results and parameters.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC
from datetime import datetime

from .schema import DFMConfig


@dataclass
class BaseResult(ABC):
    """Base class for all factor model result structures.
    
    This abstract base class defines the common interface and fields
    shared by all factor model results (DFM, DDFM, etc.).
    
    Attributes
    ----------
    x_sm : np.ndarray
        Standardized smoothed data matrix (T x N), where T is time periods
        and N is number of series. Data is standardized (zero mean, unit variance).
    X_sm : np.ndarray
        Unstandardized smoothed data matrix (T x N). This is the original-scale
        version of x_sm, computed as X_sm = x_sm * Wx + Mx.
    Z : np.ndarray
        Smoothed factor estimates (T x m), where m is the state dimension.
        Columns represent different factors (common factors and idiosyncratic components).
    C : np.ndarray
        Observation/loading matrix (N x m). Each row corresponds to a series,
        each column to a factor. C[i, j] gives the loading of series i on factor j.
    R : np.ndarray
        Covariance matrix for observation equation residuals (N x N).
        Typically diagonal, representing idiosyncratic variances.
    A : np.ndarray
        Transition matrix (m x m) for the state equation. Describes how factors
        evolve over time: Z_t = A @ Z_{t-1} + error.
    Q : np.ndarray
        Covariance matrix for transition equation residuals (m x m).
        Describes the covariance of factor innovations.
    Mx : np.ndarray
        Series means (N,). Used for standardization: x = (X - Mx) / Wx.
    Wx : np.ndarray
        Series standard deviations (N,). Used for standardization.
    Z_0 : np.ndarray
        Initial state vector (m,). Starting values for factors at t=0.
    V_0 : np.ndarray
        Initial covariance matrix (m x m) for factors. Uncertainty about Z_0.
    r : np.ndarray
        Number of factors per block (n_blocks,). Each element specifies
        how many factors are in each block structure.
    p : int
        Number of lags in the autoregressive structure of factors. Typically p=1.
    converged : bool, optional
        Whether estimation algorithm converged.
    num_iter : int, optional
        Number of iterations performed.
    loglik : float, optional
        Final log-likelihood value.
    rmse : float, optional
        Overall RMSE on original scale (averaged across all series).
    rmse_per_series : np.ndarray, optional
        RMSE per series on original scale (N,).
    rmse_std : float, optional
        Overall RMSE on standardized scale (averaged across all series).
    rmse_std_per_series : np.ndarray, optional
        RMSE per series on standardized scale (N,).
    series_ids : List[str], optional
        Series identifiers for metadata.
    block_names : List[str], optional
        Block names for metadata.
    time_index : object, optional
        Time index for data (typically a TimeIndex).
    """
    x_sm: np.ndarray      # Standardized smoothed data (T x N)
    X_sm: np.ndarray      # Unstandardized smoothed data (T x N)
    Z: np.ndarray         # Smoothed factors (T x m)
    C: np.ndarray         # Observation matrix (N x m)
    R: np.ndarray         # Covariance for observation residuals (N x N)
    A: np.ndarray         # Transition matrix (m x m)
    Q: np.ndarray         # Covariance for transition residuals (m x m)
    Mx: np.ndarray        # Series means (N,)
    Wx: np.ndarray        # Series standard deviations (N,)
    Z_0: np.ndarray       # Initial state (m,)
    V_0: np.ndarray       # Initial covariance (m x m)
    r: np.ndarray         # Number of factors per block
    p: int                # Number of lags
    converged: bool = False  # Whether algorithm converged
    num_iter: int = 0     # Number of iterations completed
    loglik: float = -np.inf  # Final log-likelihood
    rmse: Optional[float] = None  # Overall RMSE (original scale)
    rmse_per_series: Optional[np.ndarray] = None  # RMSE per series (original scale)
    rmse_std: Optional[float] = None  # Overall RMSE (standardized scale)
    rmse_std_per_series: Optional[np.ndarray] = None  # RMSE per series (standardized scale)
    # Optional metadata for object-oriented access
    series_ids: Optional[List[str]] = None
    block_names: Optional[List[str]] = None
    time_index: Optional[object] = None  # Typically a TimeIndex

    # ----------------------------
    # Convenience methods (OOP)
    # ----------------------------
    def num_series(self) -> int:
        """Return number of series (rows in C)."""
        return int(self.C.shape[0])

    def num_state(self) -> int:
        """Return state dimension (columns in Z/C)."""
        return int(self.Z.shape[1])

    def num_factors(self) -> int:
        """Return number of primary factors (sum of r)."""
        try:
            return int(np.sum(self.r))
        except Exception:
            return self.num_state()

    def to_pandas_factors(self, time_index: Optional[object] = None, factor_names: Optional[List[str]] = None):
        """Return factors as pandas DataFrame.
        
        Parameters
        ----------
        time_index : TimeIndex, list, or compatible, optional
            Time index to use for rows. If None, uses stored time_index if available.
        factor_names : List[str], optional
            Column names. Defaults to F1..Fm.
        """
        try:
            import pandas as pd
            from ..utils.time import TimeIndex
            
            cols = factor_names if factor_names is not None else [f"F{i+1}" for i in range(self.num_state())]
            
            # Create DataFrame with factors as columns
            df_dict = {col: self.Z[:, i] for i, col in enumerate(cols)}
            
            # Add time column if time_index provided
            time_to_use = time_index if time_index is not None else self.time_index
            if time_to_use is not None:
                if isinstance(time_to_use, TimeIndex):
                    time_list = time_to_use.to_list()
                elif hasattr(time_to_use, '__iter__') and not isinstance(time_to_use, (str, bytes)):
                    time_list = list(time_to_use)
                else:
                    try:
                        time_list = [time_to_use[i] for i in range(len(time_to_use))]
                    except (TypeError, AttributeError):
                        time_list = []
                if time_list:
                    df_dict['time'] = time_list
            
            return pd.DataFrame(df_dict)
        except (ImportError, ValueError, TypeError):
            return self.Z


    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None):
        """Return smoothed data (original scale) as pandas DataFrame."""
        try:
            import pandas as pd
            from ..utils.time import TimeIndex
            
            # Get column names: use provided series_ids, fallback to stored IDs, or generate defaults
            if series_ids is not None:
                cols = series_ids
            elif self.series_ids is not None:
                cols = self.series_ids
            else:
                cols = [f"S{i+1}" for i in range(self.num_series())]
            
            # Create DataFrame with series as columns
            df_dict = {col: self.X_sm[:, i] for i, col in enumerate(cols)}
            
            # Add time column if time_index provided
            time_to_use = time_index if time_index is not None else self.time_index
            if time_to_use is not None:
                if isinstance(time_to_use, TimeIndex):
                    time_list = time_to_use.to_list()
                elif hasattr(time_to_use, '__iter__') and not isinstance(time_to_use, (str, bytes)):
                    time_list = list(time_to_use)
                else:
                    try:
                        time_list = [time_to_use[i] for i in range(len(time_to_use))]
                    except (TypeError, AttributeError):
                        time_list = []
                if time_list:
                    df_dict['time'] = time_list
            
            return pd.DataFrame(df_dict)
        except (ImportError, ValueError, TypeError):
            return self.X_sm


    def save(self, path: str) -> None:
        """Save result to a pickle file."""
        import pickle
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except (IOError, OSError, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to save result to {path}: {e}")


@dataclass
class DFMResult(BaseResult):
    """DFM estimation results structure.
    
    This dataclass contains all outputs from the DFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Inherits all fields and methods from BaseResult. This class is specifically
    for linear DFM results estimated using the EM algorithm.
    
    Attributes
    ----------
    converged : bool
        Whether EM algorithm converged.
    num_iter : int
        Number of EM iterations performed.
    
    Examples
    --------
    >>> from dfm_python import DFM
    >>> model = DFM()
    >>> Res = model.fit(X, config, threshold=1e-4)
    >>> # Access smoothed factors
    >>> common_factor = Res.Z[:, 0]
    >>> # Access factor loadings for first series
    >>> loadings = Res.C[0, :]
    >>> # Reconstruct smoothed series from factors
    >>> reconstructed = Res.Z @ Res.C.T
    """
    # All fields inherited from BaseResult
    # converged and num_iter have specific meaning for EM algorithm


@dataclass
class DDFMResult(BaseResult):
    """DDFM estimation results structure.
    
    This dataclass contains all outputs from the DDFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Inherits all fields and methods from BaseResult. This class is specifically
    for Deep Dynamic Factor Model results estimated using gradient descent.
    
    Attributes
    ----------
    converged : bool
        Whether MCMC/gradient descent algorithm converged.
    num_iter : int
        Number of MCMC iterations or epochs performed.
    training_loss : float, optional
        Final training loss from neural network training.
    encoder_layers : List[int], optional
        Architecture of the encoder network used.
    use_idiosyncratic : bool, optional
        Whether idiosyncratic components were modeled.
    
    Examples
    --------
    >>> from dfm_python import DDFM
    >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> Res = model.fit(X, config, epochs=100)
    >>> # Access smoothed factors
    >>> common_factor = Res.Z[:, 0]
    >>> # Access factor loadings
    >>> loadings = Res.C[0, :]
    """
    # All fields inherited from BaseResult
    # Additional DDFM-specific fields
    training_loss: Optional[float] = None  # Final training loss
    encoder_layers: Optional[List[int]] = None  # Encoder architecture
    use_idiosyncratic: Optional[bool] = None  # Whether idio components were used


@dataclass
class NowcastResult:
    """Result from a single nowcast calculation.
    
    This dataclass contains all information from a nowcast operation,
    including the nowcast value, metadata about the data view, and
    optional diagnostic information.
    
    Attributes
    ----------
    target_series : str
        Target series ID that was nowcasted.
    target_period : datetime
        Target period for the nowcast (the period being estimated).
    view_date : datetime
        View date (when data is available). This determines which
        data points are masked/unmasked in the nowcast calculation.
    nowcast_value : float
        The calculated nowcast value for the target series.
    confidence_interval : Tuple[float, float], optional
        Confidence interval (lower, upper) for the nowcast, if available.
    factors_at_view : np.ndarray, optional
        Factor values at the view_date (m,). These are the updated
        factor states after applying the data view masking.
    dfm_result : BaseResult, optional
        Full DFM/DDFM result for this view. Can be used for further
        analysis or diagnostics.
    data_availability : Dict[str, int], optional
        Dictionary with keys 'n_available' and 'n_missing' indicating
        how many data points were available vs missing in the data view.
    
    Examples
    --------
    >>> from dfm_python import DFM
    >>> import numpy as np
    >>> model = DFM()
    >>> trainer.fit(model, data_module)
    >>> # Update state with new data, then predict
    >>> X_std = np.random.randn(10, 5)  # Standardized data
    >>> model.update(X_std)
    >>> forecast = model.predict(horizon=1)
    >>> print(f"Forecast: {forecast[0, 0]}")
    """
    target_series: str
    target_period: datetime
    view_date: datetime
    nowcast_value: float
    confidence_interval: Optional[Tuple[float, float]] = None  # (lower, upper)
    factors_at_view: Optional[np.ndarray] = None  # Factor values at view_date
    dfm_result: Optional[BaseResult] = None  # Full DFM/DDFM result for this view
    data_availability: Optional[Dict[str, int]] = None  # n_available, n_missing


@dataclass
class FitParams:
    """Parameter overrides for DFM model fitting.
    
    This dataclass groups all optional parameters that can override
    DFMConfig values during model fitting. This reduces method signature
    complexity and improves code readability.
    
    All parameters are optional. If None, the corresponding value
    from DFMConfig will be used during parameter resolution.
    
    This class provides parameter overrides for DFM estimation
    for consistency across the codebase.
    """
    # Convergence parameters
    threshold: Optional[float] = None
    max_iter: Optional[int] = None
    
    # Model structure
    ar_lag: Optional[int] = None
    num_factors: Optional[int] = None
    
    # Missing data handling
    nan_method: Optional[int] = None
    nan_k: Optional[int] = None
    clock: Optional[str] = None
    
    # AR coefficient clipping
    clip_ar_coefficients: Optional[bool] = None
    ar_clip_min: Optional[float] = None
    ar_clip_max: Optional[float] = None
    
    # Data clipping
    clip_data_values: Optional[bool] = None
    data_clip_threshold: Optional[float] = None
    
    # Regularization
    use_regularization: Optional[bool] = None
    regularization_scale: Optional[float] = None
    min_eigenvalue: Optional[float] = None
    max_eigenvalue: Optional[float] = None
    
    # Damping
    use_damped_updates: Optional[bool] = None
    damping_factor: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> 'FitParams':
        """Create FitParams from keyword arguments.
        
        Filters kwargs to only include valid parameter names,
        ignoring any extra arguments.
        """
        valid_params = {
            'threshold', 'max_iter', 'ar_lag', 'num_factors',
            'nan_method', 'nan_k', 'clock',
            'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
            'clip_data_values', 'data_clip_threshold',
            'use_regularization', 'regularization_scale',
            'min_eigenvalue', 'max_eigenvalue',
            'use_damped_updates', 'damping_factor'
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        return cls(**filtered)



