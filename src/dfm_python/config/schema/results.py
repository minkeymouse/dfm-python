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
    
    This abstract base class defines the core model outputs shared by all
    factor model results (DFM, DDFM, KDFM, etc.). Only essential model parameters
    and outputs are included - no user-specific metadata.
    
    Attributes
    ----------
    x_sm : np.ndarray
        Standardized smoothed data matrix (T x N), where T is time periods
        and N is number of series. Data is standardized (zero mean, unit variance).
        This is the internal representation used by the model.
        To get unstandardized data: use get_x_sm_original_scale(target_scaler) method.
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
    target_scaler : Any, optional
        Sklearn scaler instance (StandardScaler, RobustScaler, etc.) for target series only.
        Used for unstandardization: X = scaler.inverse_transform(x).
        If None, assumes data is already in original scale.
    Z_0 : np.ndarray
        Initial state vector (m,). Starting values for factors at t=0.
    V_0 : np.ndarray
        Initial covariance matrix (m x m) for factors. Uncertainty about Z_0.
    r : np.ndarray
        Number of factors per block (n_blocks,). Each element specifies
        how many factors are in each block structure.
    p : int
        Number of lags in the autoregressive structure of factors. Typically p=1.
    converged : bool
        Whether estimation algorithm converged.
    num_iter : int
        Number of iterations performed.
    loglik : float
        Final log-likelihood value.
    """
    # Core state-space model parameters (required fields)
    x_sm: np.ndarray      # Standardized smoothed data (T x N)
    Z: np.ndarray         # Smoothed factors (T x m)
    C: np.ndarray         # Observation matrix (N x m)
    R: np.ndarray         # Covariance for observation residuals (N x N)
    A: np.ndarray         # Transition matrix (m x m)
    Q: np.ndarray         # Covariance for transition residuals (m x m)
    Z_0: np.ndarray       # Initial state (m,)
    V_0: np.ndarray       # Initial covariance (m x m)
    r: np.ndarray         # Number of factors per block
    p: int                # Number of lags
    # Optional fields (must come after required fields)
    target_scaler: Optional[Any] = None  # Sklearn scaler for target series unstandardization
    # Training diagnostics
    converged: bool = False  # Whether algorithm converged
    num_iter: int = 0     # Number of iterations completed
    loglik: float = -np.inf  # Final log-likelihood

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
        except (ValueError, AttributeError, TypeError):
            return self.num_state()
    
    def get_x_sm_original_scale(self, target_scaler: Optional[Any] = None) -> np.ndarray:
        """Get unstandardized smoothed data using target scaler.
        
        Parameters
        ----------
        target_scaler : Any, optional
            Sklearn scaler instance. If None, uses self.target_scaler.
            If both are None, returns x_sm as-is (assumes already in original scale).
        
        Returns
        -------
        np.ndarray
            Unstandardized smoothed data (T x N)
        """
        scaler = target_scaler if target_scaler is not None else self.target_scaler
        if scaler is not None and hasattr(scaler, 'inverse_transform'):
            return scaler.inverse_transform(self.x_sm)
        # No scaler - assume already in original scale
        return self.x_sm
    
    def to_pandas_factors(self, time_index: Optional[object] = None, factor_names: Optional[List[str]] = None):
        """Return factors as pandas DataFrame."""
        try:
            import pandas as pd
            cols = factor_names or [f"F{i+1}" for i in range(self.num_state())]
            df_dict = {col: self.Z[:, i] for i, col in enumerate(cols)}
            if time_index is not None:
                if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                    df_dict['time'] = list(time_index)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.Z
    
    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None, target_scaler: Optional[Any] = None):
        """Return smoothed data as pandas DataFrame."""
        try:
            import pandas as pd
            x_sm_original = self.get_x_sm_original_scale(target_scaler)
            cols = series_ids or [f"S{i+1}" for i in range(self.num_series())]
            df_dict = {col: x_sm_original[:, i] for i, col in enumerate(cols)}
            if time_index is not None:
                if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                    df_dict['time'] = list(time_index)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.get_x_sm_original_scale(target_scaler)
    
    def summary(self) -> str:
        """Return a formatted summary of the model results.
        
        Returns
        -------
        str
            Formatted string containing model summary including:
            - Model type and structure
            - Data dimensions (series, factors, periods)
            - Training diagnostics (convergence, iterations, log-likelihood)
            - Factor structure (AR order, factors per block)
        """
        # Determine model type from class name
        model_type = self.__class__.__name__.replace('Result', '')
        
        # Build summary lines
        lines = []
        lines.append("=" * 80)
        lines.append(f"{model_type} Model Summary")
        lines.append("=" * 80)
        lines.append("")
        
        # Data dimensions
        lines.append("Data Dimensions:")
        lines.append(f"  Series: {self.num_series()}")
        lines.append(f"  Factors: {self.num_factors()} (total state dimension: {self.num_state()})")
        lines.append(f"  Time periods: {self.num_periods()}")
        lines.append("")
        
        # Factor structure
        lines.append("Factor Structure:")
        if hasattr(self.r, '__len__') and len(self.r) > 0:
            if len(self.r) == 1:
                lines.append(f"  Factors per block: {self.r[0]}")
            else:
                lines.append(f"  Factors per block: {self.r}")
        else:
            lines.append(f"  Total factors: {self.num_factors()}")
        lines.append(f"  AR order: {self.p}")
        lines.append("")
        
        # Training diagnostics
        lines.append("Training Diagnostics:")
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Iterations: {self.num_iter}")
        lines.append(f"  Log-likelihood: {self.loglik:.4f}")
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)



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
    
    def summary(self) -> str:
        """Return a formatted summary of the DFM results."""
        return super().summary()


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
    
    def summary(self) -> str:
        """Return a formatted summary of the DDFM results."""
        summary_text = super().summary()
        lines = summary_text.split("\n")
        
        # Insert DDFM-specific information before the final separator
        insert_idx = len(lines) - 1
        if self.training_loss is not None:
            lines.insert(insert_idx, "")
            lines.insert(insert_idx, "Neural Network Training:")
            lines.insert(insert_idx + 1, f"  Final training loss: {self.training_loss:.4f}")
            if self.encoder_layers is not None:
                lines.insert(insert_idx + 2, f"  Encoder architecture: {self.encoder_layers}")
        
        return "\n".join(lines)

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
    
    def summary(self) -> str:
        """Return a formatted summary of the KDFM results."""
        summary_text = super().summary()
        lines = summary_text.split("\n")
        
        # Insert KDFM-specific information before the final separator
        insert_idx = len(lines) - 1
        kdfm_info = []
        
        if self.ar_coeffs is not None:
            ar_order = self.ar_coeffs.shape[0] if self.ar_coeffs.ndim > 0 else 0
            kdfm_info.append(f"  VAR order: {ar_order}")
        if self.ma_coeffs is not None:
            ma_order = self.ma_coeffs.shape[0] if self.ma_coeffs.ndim > 0 else 0
            kdfm_info.append(f"  MA order: {ma_order}")
        if self.irf_reduced is not None:
            kdfm_info.append("  IRFs computed: Reduced-form")
        if self.irf_structural is not None:
            kdfm_info.append("  IRFs computed: Structural")
        
        if kdfm_info:
            lines.insert(insert_idx, "")
            lines.insert(insert_idx, "KDFM-Specific:")
            for info in kdfm_info:
                lines.insert(insert_idx + 1, info)
        
        return "\n".join(lines)


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

