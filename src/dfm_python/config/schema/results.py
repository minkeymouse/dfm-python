"""Result structures for Dynamic Factor Model estimation.

This module contains model-specific result dataclasses:
- DFMResult(BaseResult): Results for linear DFM
- DDFMResult(BaseResult): Results for Deep DFM
- KDFMResult(BaseResult): Results for Kernelized DFM
"""

import numpy as np
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .model import DFMConfig



# ============================================================================
# Base Result Structure
# ============================================================================

@dataclass
class BaseResult(ABC):
    """Base class for all factor model result structures.
    
    This abstract base class defines the common interface and fields
    shared by all factor model results (DFM, DDFM, KDFM, etc.).
    
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

    def num_periods(self) -> int:
        """Return number of time periods (rows in Z/x_sm)."""
        return int(self.Z.shape[0])
    
    def num_factors(self) -> int:
        """Return number of primary factors (sum of r)."""
        try:
            return int(np.sum(self.r))
        except Exception:
            return self.num_state()
    
    def to_pandas_factors(self, time_index: Optional[object] = None, factor_names: Optional[List[str]] = None):
        """Return factors as pandas DataFrame."""
        try:
            import pandas as pd
            cols = factor_names or [f"F{i+1}" for i in range(self.num_state())]
            df_dict = {col: self.Z[:, i] for i, col in enumerate(cols)}
            if time_index is not None or self.time_index is not None:
                time_to_use = time_index if time_index is not None else self.time_index
                if hasattr(time_to_use, '__iter__') and not isinstance(time_to_use, (str, bytes)):
                    df_dict['time'] = list(time_to_use)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.Z
    
    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None):
        """Return smoothed data as pandas DataFrame."""
        try:
            import pandas as pd
            cols = series_ids or self.series_ids or [f"S{i+1}" for i in range(self.num_series())]
            df_dict = {col: self.X_sm[:, i] for i, col in enumerate(cols)}
            if time_index is not None or self.time_index is not None:
                time_to_use = time_index if time_index is not None else self.time_index
                if hasattr(time_to_use, '__iter__') and not isinstance(time_to_use, (str, bytes)):
                    df_dict['time'] = list(time_to_use)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.X_sm



# ============================================================================
# Model-Specific Result Classes
# ============================================================================
# BaseResult is imported from base.py - no duplicate definition needed

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
class KDFMResult(BaseResult):
    """KDFM estimation results structure.
    
    This dataclass contains all outputs from the KDFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Inherits all fields and methods from BaseResult. This class is specifically
    for KDFM results estimated using gradient descent.
    
    Attributes
    ----------
    S : np.ndarray, optional
        Structural identification matrix (K x K)
    structural_shocks : np.ndarray, optional
        Structural shocks ε_t (T x K)
    irf_reduced : np.ndarray, optional
        Reduced-form IRFs (horizon x K x K)
    irf_structural : np.ndarray, optional
        Structural IRFs (horizon x K x K)
    ar_coeffs : np.ndarray, optional
        Extracted VAR coefficients (p x K x K)
    ma_coeffs : np.ndarray, optional
        Extracted MA coefficients (q x K x K), only if q > 0
    """
    # KDFM-specific fields
    S: Optional[np.ndarray] = None  # Structural identification matrix
    structural_shocks: Optional[np.ndarray] = None  # ε_t (T x K)
    irf_reduced: Optional[np.ndarray] = None  # Reduced-form IRFs
    irf_structural: Optional[np.ndarray] = None  # Structural IRFs
    ar_coeffs: Optional[np.ndarray] = None  # Extracted VAR coefficients
    ma_coeffs: Optional[np.ndarray] = None  # Extracted MA coefficients (if q > 0)


# ============================================================================
# Parameter Override Structure
# ============================================================================

@dataclass
class FitParams:
    """Parameter overrides for DFM estimation.
    
    This dataclass allows overriding configuration parameters at fit time
    without modifying the configuration object itself.
    
    Attributes
    ----------
    max_iter : int, optional
        Maximum EM iterations (overrides config.max_iter)
    threshold : float, optional
        Convergence threshold (overrides config.threshold)
    regularization_scale : float, optional
        Regularization scale (overrides config.regularization_scale)
    """
    max_iter: Optional[int] = None
    threshold: Optional[float] = None
    regularization_scale: Optional[float] = None

