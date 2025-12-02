"""Parameter dataclasses for DFM models.

This module contains parameter configuration classes:
- Params: Global estimation parameters with defaults
- FitParams: Optional parameter overrides for model fitting
"""

from dataclasses import dataclass
from typing import Optional
from .utils import validate_frequency


@dataclass
class Params:
    """Global estimation parameters for DFM (main settings only).
    
    This dataclass contains only the global knobs that control the EM algorithm
    and numerical stability. It does NOT include series or block definitions.
    Use this with spec CSV files to provide main settings while the spec CSV
    defines all series and their block memberships.
    
    All parameters have defaults matching the standard YAML configuration.
    
    Attributes
    ----------
    ar_lag : int
        AR lag for factor transition equation (typically 1)
    threshold : float
        EM convergence threshold (default: 1e-5)
    max_iter : int
        Maximum EM iterations (default: 5000)
    nan_method : int
        Missing data handling method (1-5, default: 2 = spline interpolation).
        This preprocessing step is followed by Kalman Filter-based missing data
        handling during DFM estimation, following standard practice in state-space
        models (Mariano & Murasawa 2003, Harvey 1989).
    nan_k : int
        Spline parameter for NaN interpolation (default: 3 for cubic spline)
    clock : str
        Base frequency for all latent factors (default: 'm' for monthly)
    clip_ar_coefficients : bool
        Enable AR coefficient clipping for stationarity (default: True)
    ar_clip_min : float
        Minimum AR coefficient (default: -0.99)
    ar_clip_max : float
        Maximum AR coefficient (default: 0.99)
    warn_on_ar_clip : bool
        Warn when AR coefficients are clipped (default: True)
    clip_data_values : bool
        Enable clipping of extreme data values (default: True)
    data_clip_threshold : float
        Clip values beyond this many standard deviations (default: 100.0)
    warn_on_data_clip : bool
        Warn when data values are clipped (default: True)
    use_regularization : bool
        Enable regularization for numerical stability (default: True)
    regularization_scale : float
        Scale factor for ridge regularization (default: 1e-5)
    min_eigenvalue : float
        Minimum eigenvalue for positive definite matrices (default: 1e-8)
    max_eigenvalue : float
        Maximum eigenvalue cap (default: 1e6)
    warn_on_regularization : bool
        Warn when regularization is applied (default: True)
    use_damped_updates : bool
        Enable damped updates when likelihood decreases (default: True)
    damping_factor : float
        Damping factor: 0.8 = 80% new, 20% old (default: 0.8)
    warn_on_damped_update : bool
        Warn when damped updates are used (default: True)
    augment_idio : bool
        Enable state augmentation with idiosyncratic components (default: True)
    augment_idio_slow : bool
        Enable tent-length chains for slower-frequency series (default: True)
    idio_rho0 : float
        Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var : float
        Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
    
    Examples
    --------
    >>> from dfm_python.config import Params
    >>> params = Params(max_iter=100, threshold=1e-4)
    >>> dfm.from_spec('data/spec.csv', params=params)
    """
    # Estimation parameters
    ar_lag: int = 1
    threshold: float = 1e-5
    max_iter: int = 5000
    nan_method: int = 2
    nan_k: int = 3
    clock: str = 'm'
    
    # Numerical stability - AR clipping
    clip_ar_coefficients: bool = True
    ar_clip_min: float = -0.99
    ar_clip_max: float = 0.99
    warn_on_ar_clip: bool = True
    
    # Numerical stability - Data clipping
    clip_data_values: bool = True
    data_clip_threshold: float = 100.0
    warn_on_data_clip: bool = True
    
    # Numerical stability - Regularization
    use_regularization: bool = True
    regularization_scale: float = 1e-5
    min_eigenvalue: float = 1e-8
    max_eigenvalue: float = 1e6
    warn_on_regularization: bool = True
    
    # Numerical stability - Damped updates
    use_damped_updates: bool = True
    damping_factor: float = 0.8
    warn_on_damped_update: bool = True
    
    # Idiosyncratic component augmentation
    augment_idio: bool = True  # Enable state augmentation with idiosyncratic components (default: True)
    augment_idio_slow: bool = True  # Enable tent-length chains for slower-frequency series (default: True)
    idio_rho0: float = 0.1  # Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var: float = 1e-8  # Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
    
    def __post_init__(self):
        """Validate parameters."""
        self.clock = validate_frequency(self.clock)
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")
        if self.ar_clip_min >= self.ar_clip_max:
            raise ValueError(f"ar_clip_min ({self.ar_clip_min}) must be < ar_clip_max ({self.ar_clip_max})")
        if not (-1 < self.ar_clip_min < 1):
            raise ValueError(f"ar_clip_min must be in (-1, 1), got {self.ar_clip_min}")
        if not (-1 < self.ar_clip_max < 1):
            raise ValueError(f"ar_clip_max must be in (-1, 1), got {self.ar_clip_max}")


@dataclass
class FitParams:
    """Parameters for fitting DFM models.
    
    This dataclass groups all optional parameters that can override
    config values during model fitting. This reduces method signature
    complexity and improves code readability.
    
    All parameters are optional. If None, the corresponding value
    from DFMConfig will be used during parameter resolution.
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
            'nan_method', 'nan_k',
            'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
            'clip_data_values', 'data_clip_threshold',
            'use_regularization', 'regularization_scale',
            'min_eigenvalue', 'max_eigenvalue',
            'use_damped_updates', 'damping_factor'
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        return cls(**filtered)

