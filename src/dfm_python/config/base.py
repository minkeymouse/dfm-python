"""Base configuration and result classes.

This module contains:
- Base classes: BaseModelConfig, BaseResult, SeriesConfig
- Common constants: DEFAULT_BLOCK_NAME
- Shared structures used across all model types
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC

from .utils import validate_frequency


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BLOCK_NAME = 'Block_0'


# ============================================================================
# Series Configuration
# ============================================================================

@dataclass
class SeriesConfig:
    """Configuration for a single time series.
    
    This is a generic DFM configuration - no API or database-specific fields.
    For API/database integration, implement adapters in your application layer.
    
    Note: Transformation is handled by preprocessing pipeline, not in SeriesConfig.
    Note: Blocks are defined in DFMConfig, not in SeriesConfig.
    
    Attributes
    ----------
    frequency : str
        Series frequency: 'm' (monthly), 'q' (quarterly), 'sa' (semi-annual), 'a' (annual)
    series_id : str, optional
        Unique identifier (auto-generated if None)
    series_name : str, optional
        Human-readable name (defaults to series_id if None)
    units : str, optional
        Units of measurement (optional metadata for display purposes only).
        Used in news decomposition output for readability. Not used in model estimation.
    release_date : int, optional
        Release date information for pseudo real-time nowcasting.
        - Positive value (1-31): Day of month when data is released
        - Negative value: Days before end of previous month when data is released
        Example: 25 = released on 25th of each month, -5 = released 5 days before end of previous month
    """
    # Required fields (no defaults)
    frequency: str
    # Optional fields (with defaults - must come after required fields)
    series_id: Optional[str] = None  # Auto-generated if None: "series_0", "series_1", etc.
    series_name: Optional[str] = None  # Optional metadata for display
    units: Optional[str] = None  # Optional metadata for display only (used in news.py output)
    release_date: Optional[int] = None  # Release date for pseudo real-time nowcasting
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self.frequency = validate_frequency(self.frequency)
        # Auto-generate series_name if not provided
        if self.series_name is None and self.series_id:
            self.series_name = self.series_id


# ============================================================================
# Base Model Configuration
# ============================================================================

@dataclass
class BaseModelConfig:
    """Base configuration class with shared model structure.
    
    This base class contains the model structure that is common to all
    factor models (DFM, DDFM, KDFM):
    - Series definitions
    - Clock frequency
    - Data preprocessing (missing data handling)
    
    Note: Blocks are DFM-specific and are NOT included in BaseModelConfig.
    DFMConfig adds block structure, while DDFMConfig and KDFMConfig do not use blocks.
    
    Subclasses (DFMConfig, DDFMConfig, KDFMConfig) add model-specific training parameters.
    """
    # ========================================================================
    # Model Structure (WHAT - defines the model)
    # ========================================================================
    series: List[SeriesConfig]  # Series specifications
    
    # ========================================================================
    # Shared Data Handling Parameters
    # ========================================================================
    nan_method: int = 2  # Missing data handling method (1-5). Preprocessing step before Kalman Filter-based handling
    nan_k: int = 3  # Spline parameter for NaN interpolation (cubic spline)
    clock: str = 'm'  # Base frequency for nowcasting (global clock): 'd', 'w', 'm', 'q', 'sa', 'a' (defaults to 'm' for monthly)
    scaler: Optional[str] = 'standard'  # Unified scaler type for all series: 'standard', 'robust', 'minmax', 'maxabs', 'quantile', or None (no scaling). Default: 'standard' for unified scaling.
    
    def __post_init__(self):
        """Validate basic model structure.
        
        This method performs basic validation of the model configuration:
        - Ensures at least one series is specified
        - Validates clock frequency
        - Auto-generates series_id if not provided
        
        Raises
        ------
        ValueError
            If any validation check fails, with a descriptive error message
            indicating what needs to be fixed.
        """
        # Import frequency hierarchy for validation
        from .utils import FREQUENCY_HIERARCHY
        
        if not self.series:
            raise ValueError(
                "Model configuration must contain at least one series. "
                "Please add series definitions to your configuration."
            )
        
        # Validate global clock
        self.clock = validate_frequency(self.clock)
        
        # Auto-generate series_id if not provided
        for i, s in enumerate(self.series):
            if s.series_id is None:
                s.series_id = f"series_{i}"
            if s.series_name is None:
                s.series_name = s.series_id
    
    # ========================================================================
    # Helper Methods (snake_case - recommended)
    # ========================================================================
    
    def get_series_ids(self) -> List[str]:
        """Get list of series IDs (snake_case - recommended)."""
        return [s.series_id if s.series_id is not None else f"series_{i}" 
                for i, s in enumerate(self.series)]
    
    def get_series_names(self) -> List[str]:
        """Get list of series names (snake_case - recommended)."""
        return [s.series_name if s.series_name is not None else (s.series_id or f"series_{i}")
                for i, s in enumerate(self.series)]
    
    def get_frequencies(self) -> List[str]:
        """Get list of frequencies (snake_case - recommended)."""
        return [s.frequency for s in self.series]
    
    def validate_and_report(self) -> Dict[str, Any]:
        """Validate configuration and return structured report with issues and suggestions.
        
        This method performs validation checks without raising exceptions, returning
        a structured report that can be used for debugging and user guidance.
        
        Returns
        -------
        Dict[str, Any]
            Report dictionary with keys:
            - 'valid': bool - Whether configuration is valid
            - 'errors': List[str] - List of error messages
            - 'warnings': List[str] - List of warning messages
            - 'suggestions': List[str] - List of actionable suggestions
        """
        from .utils import FREQUENCY_HIERARCHY
        
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for empty series
        if not self.series:
            report['valid'] = False
            report['errors'].append("Model configuration must contain at least one series.")
            report['suggestions'].append("Add series definitions to your configuration.")
            return report
        
        return report


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
            raise IOError(f"Failed to save result to {path}: {e}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseResult':
        """Load result from a pickle file."""
        import pickle
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (IOError, OSError, pickle.PickleError) as e:
            raise IOError(f"Failed to load result from {path}: {e}")

