"""Result structures for Dynamic Factor Model estimation.

This module contains model-specific result dataclasses:
- DFMResult(BaseResult): Results for linear DFM
- DDFMResult(BaseResult): Results for Deep DFM
- KDFMResult(BaseResult): Results for Kernelized DFM

Base class (BaseResult) is in config/base.py
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .base import BaseResult
from .schema import DFMConfig


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



