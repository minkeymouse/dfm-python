"""Dynamic Factor Model (DFM) estimation using Expectation-Maximization algorithm.

This module implements the core DFM estimation framework, including:
- Initial parameter estimation via PCA and OLS
- EM algorithm for iterative parameter refinement
- Kalman filtering and smoothing for factor extraction
- Clock-based mixed-frequency handling with tent kernels
- Robust numerical stability and error handling

The implementation uses a clock-based approach, where all latent factors
evolve at a common clock frequency, with lower-frequency observations
mapped to higher-frequency latent states via deterministic tent kernels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict, Union, List
import warnings
import logging
import pandas as pd

from .kalman import run_kf
from .config import DFMConfig
from .core import calculate_rmse
from .utils import group_series_by_frequency
from .core.numeric import (
    _ensure_symmetric,
    _compute_principal_components,
    _clean_matrix,
    _ensure_positive_definite,
    _compute_regularization_param,
    _apply_ar_clipping,
    _cap_max_eigenvalue,
    _estimate_ar_coefficient,
    _safe_divide,
    _check_finite,
)
from .core import (
    _display_dfm_tables,
    diagnose_series,
    print_series_diagnosis,
)
from .core.em import (
    init_conditions,
    em_step,
    em_converged,
    NaNHandlingOptions,
)
from .core.helpers import safe_get_method, safe_get_attr, resolve_param, safe_mean_std, standardize_data

from .data import rem_nans_spline
from .utils import (
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
)

_logger = logging.getLogger(__name__)


@dataclass
class DFMResult:
    """DFM estimation results structure.
    
    This dataclass contains all outputs from the DFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
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
    rmse : float, optional
        Overall RMSE on original scale (averaged across all series).
    rmse_per_series : np.ndarray, optional
        RMSE per series on original scale (N,).
    rmse_std : float, optional
        Overall RMSE on standardized scale (averaged across all series).
    rmse_std_per_series : np.ndarray, optional
        RMSE per series on standardized scale (N,).
    converged : bool, optional
        Whether EM algorithm converged.
    num_iter : int, optional
        Number of EM iterations performed.
    loglik : float, optional
        Final log-likelihood value.
    
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
    converged: bool = False  # Whether EM algorithm converged
    num_iter: int = 0     # Number of iterations completed
    loglik: float = -np.inf  # Final log-likelihood
    rmse: Optional[float] = None  # Overall RMSE (original scale)
    rmse_per_series: Optional[np.ndarray] = None  # RMSE per series (original scale)
    rmse_std: Optional[float] = None  # Overall RMSE (standardized scale)
    rmse_std_per_series: Optional[np.ndarray] = None  # RMSE per series (standardized scale)
    # Optional metadata for object-oriented access
    series_ids: Optional[List[str]] = None
    block_names: Optional[List[str]] = None
    time_index: Optional[object] = None  # Typically a pandas.DatetimeIndex

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
        time_index : pandas.DatetimeIndex or compatible, optional
            Index to use for rows. If None, uses stored time_index if available.
        factor_names : List[str], optional
            Column names. Defaults to F1..Fm.
        """
        try:
            import pandas as pd
            idx = time_index if time_index is not None else self.time_index
            cols = factor_names if factor_names is not None else [f"F{i+1}" for i in range(self.num_state())]
            return pd.DataFrame(self.Z, index=idx, columns=cols)
        except (ImportError, ValueError, TypeError):
            return self.Z

    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None):
        """Return smoothed data (original scale) as pandas DataFrame."""
        try:
            import pandas as pd
            idx = time_index if time_index is not None else self.time_index
            cols = series_ids if series_ids is not None else (self.series_ids if self.series_ids is not None else [f"S{i+1}" for i in range(self.num_series())])
            return pd.DataFrame(self.X_sm, index=idx, columns=cols)
        except (ImportError, ValueError, TypeError):
            return self.X_sm

    def save(self, path: str) -> None:
        """Save result to a pickle file."""
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except (IOError, OSError, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to save DFMResult to {path}: {e}")


@dataclass
class DFMParams:
    """DFM estimation parameter overrides.
    
    All parameters are optional. If None, the corresponding value
    from DFMConfig will be used during parameter resolution.
    
    This dataclass groups all parameter overrides that can be passed
    to `_dfm_core()` and `_prepare_data_and_params()` to reduce
    function parameter count and improve readability.
    """
    threshold: Optional[float] = None
    max_iter: Optional[int] = None
    ar_lag: Optional[int] = None
    nan_method: Optional[int] = None
    nan_k: Optional[int] = None
    clock: Optional[str] = None
    clip_ar_coefficients: Optional[bool] = None
    ar_clip_min: Optional[float] = None
    ar_clip_max: Optional[float] = None
    clip_data_values: Optional[bool] = None
    data_clip_threshold: Optional[float] = None
    use_regularization: Optional[bool] = None
    regularization_scale: Optional[float] = None
    min_eigenvalue: Optional[float] = None
    max_eigenvalue: Optional[float] = None
    use_damped_updates: Optional[bool] = None
    damping_factor: Optional[float] = None
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> 'DFMParams':
        """Create DFMParams from keyword arguments.
        
        Filters kwargs to only include valid parameter names,
        ignoring any extra arguments.
        """
        valid_params = {
            'threshold', 'max_iter', 'ar_lag', 'nan_method', 'nan_k',
            'clock', 'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
            'clip_data_values', 'data_clip_threshold', 'use_regularization',
            'regularization_scale', 'min_eigenvalue', 'max_eigenvalue',
            'use_damped_updates', 'damping_factor'
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        return cls(**filtered)


@dataclass
class EMAlgorithmParams:
    """Parameters for EM algorithm execution.
    
    This dataclass groups all parameters required for running the EM algorithm,
    reducing function parameter count and improving readability.
    
    All parameters are required (no optional fields) since the EM algorithm
    needs all of them to execute.
    """
    # Data
    y: np.ndarray
    y_est: np.ndarray
    
    # Model parameters
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    r: np.ndarray
    p: int
    
    # Structure parameters
    R_mat: Optional[np.ndarray]
    q: Optional[np.ndarray]
    nQ: int
    i_idio: np.ndarray
    blocks: np.ndarray
    tent_weights_dict: Dict[str, np.ndarray]
    clock: str
    frequencies: Optional[np.ndarray]
    idio_chain_lengths: np.ndarray
    
    # Config and algorithm parameters
    config: DFMConfig
    threshold: float
    max_iter: int
    use_damped_updates: bool
    damping_factor: float


# Core functions are imported directly from core modules - no proxy functions needed


class DFM:
    """Core Dynamic Factor Model class.
    
    This class provides the main DFM estimation functionality. The core algorithm
    is implemented in the `fit()` method, which performs EM estimation.
    """
    
    def __init__(self):
        """Initialize DFM instance."""
        self._config: Optional[DFMConfig] = None
        self._data: Optional[np.ndarray] = None
        self._result: Optional[DFMResult] = None
    
    def fit(self,
            X: np.ndarray,
            config: DFMConfig,
            threshold: Optional[float] = None,
            max_iter: Optional[int] = None,
            ar_lag: Optional[int] = None,
            nan_method: Optional[int] = None,
            nan_k: Optional[int] = None,
            clock: Optional[str] = None,
            clip_ar_coefficients: Optional[bool] = None,
            ar_clip_min: Optional[float] = None,
            ar_clip_max: Optional[float] = None,
            clip_data_values: Optional[bool] = None,
            data_clip_threshold: Optional[float] = None,
            use_regularization: Optional[bool] = None,
            regularization_scale: Optional[float] = None,
            min_eigenvalue: Optional[float] = None,
            max_eigenvalue: Optional[float] = None,
            use_damped_updates: Optional[bool] = None,
            damping_factor: Optional[float] = None,
            **kwargs) -> DFMResult:
        """Fit the DFM model using EM algorithm.
        
        This is the core estimation method. It performs the complete EM workflow:
        1. Initialization via PCA and OLS
        2. EM iterations until convergence
        3. Final Kalman smoothing
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N), where T is time periods and N is number of series.
        config : DFMConfig
            Unified DFM configuration object.
        threshold : float, optional
            EM convergence threshold. If None, uses config.threshold.
        max_iter : int, optional
            Maximum EM iterations. If None, uses config.max_iter.
        **kwargs
            Additional parameters (ar_lag, nan_method, etc.) that override config values.
            
        Returns
        -------
        DFMResult
            Estimation results including parameters, factors, and diagnostics.
        """
        # Store config and data for later use
        self._config = config
        self._data = X
        
        # Create DFMParams from individual parameters
        params = DFMParams(
            threshold=threshold,
            max_iter=max_iter,
            ar_lag=ar_lag,
            nan_method=nan_method,
            nan_k=nan_k,
            clock=clock,
            clip_ar_coefficients=clip_ar_coefficients,
            ar_clip_min=ar_clip_min,
            ar_clip_max=ar_clip_max,
            clip_data_values=clip_data_values,
            data_clip_threshold=data_clip_threshold,
            use_regularization=use_regularization,
            regularization_scale=regularization_scale,
            min_eigenvalue=min_eigenvalue,
            max_eigenvalue=max_eigenvalue,
            use_damped_updates=use_damped_updates,
            damping_factor=damping_factor,
        )
        
        # Merge kwargs into params if provided
        if kwargs:
            # Update params with kwargs (only valid parameter names)
            valid_params = {
                'threshold', 'max_iter', 'ar_lag', 'nan_method', 'nan_k',
                'clock', 'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
                'clip_data_values', 'data_clip_threshold', 'use_regularization',
                'regularization_scale', 'min_eigenvalue', 'max_eigenvalue',
                'use_damped_updates', 'damping_factor'
            }
            for k, v in kwargs.items():
                if k in valid_params and hasattr(params, k):
                    setattr(params, k, v)
        
        # Call the core _dfm_core() function logic
        result = _dfm_core(X, config, params=params)
        
        self._result = result
        return result
    
    @property
    def result(self) -> Optional[DFMResult]:
        """Get the last fit result."""
        return self._result
    
    @property
    def config(self) -> Optional[DFMConfig]:
        """Get the current configuration."""
        return self._config


def _prepare_data_and_params(
    X: np.ndarray,
    config: DFMConfig,
    params: Optional[DFMParams] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare data and resolve all parameters from config and overrides.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (T x N)
    config : DFMConfig
        Configuration object
    params : DFMParams, optional
        Parameter overrides. If None, all values from config are used.
    
    Returns
    -------
    X_clean : np.ndarray
        Cleaned input data (Inf replaced with NaN)
    blocks : np.ndarray
        Block structure array (N x n_blocks)
    params_dict : dict
        Dictionary of resolved parameters
    """
    # Clean input data
    inf_mask = np.isinf(X)
    if np.any(inf_mask):
        X = np.where(inf_mask, np.nan, X)
        warnings.warn("Data contains Inf values, replaced with NaN", UserWarning)
    
    blocks = config.get_blocks_array()
    T, N = X.shape
    
    # Initialize params if not provided
    if params is None:
        params = DFMParams()
    
    # Resolve all parameters
    params_dict = {
        'p': resolve_param(params.ar_lag, config.ar_lag),
        'r': (np.array(config.factors_per_block) 
              if config.factors_per_block is not None 
              else np.ones(blocks.shape[1])),
        'nan_method': resolve_param(params.nan_method, config.nan_method),
        'nan_k': resolve_param(params.nan_k, config.nan_k),
        'threshold': resolve_param(params.threshold, config.threshold),
        'max_iter': resolve_param(params.max_iter, config.max_iter),
        'clock': resolve_param(params.clock, config.clock),
        'clip_ar_coefficients': resolve_param(params.clip_ar_coefficients, config.clip_ar_coefficients),
        'ar_clip_min': resolve_param(params.ar_clip_min, config.ar_clip_min),
        'ar_clip_max': resolve_param(params.ar_clip_max, config.ar_clip_max),
        'clip_data_values': resolve_param(params.clip_data_values, config.clip_data_values),
        'data_clip_threshold': resolve_param(params.data_clip_threshold, config.data_clip_threshold),
        'use_regularization': resolve_param(params.use_regularization, config.use_regularization),
        'regularization_scale': resolve_param(params.regularization_scale, config.regularization_scale),
        'min_eigenvalue': resolve_param(params.min_eigenvalue, config.min_eigenvalue),
        'max_eigenvalue': resolve_param(params.max_eigenvalue, config.max_eigenvalue),
        'use_damped_updates': resolve_param(params.use_damped_updates, config.use_damped_updates),
        'damping_factor': resolve_param(params.damping_factor, config.damping_factor),
        'T': T,
        'N': N,
    }
    
    # Display blocks structure if debug logging enabled
    if _logger.isEnabledFor(logging.DEBUG):
        try:
            series_names = config.get_series_names()
            block_names = (config.block_names if len(config.block_names) == blocks.shape[1] 
                          else [f'Block_{i+1}' for i in range(blocks.shape[1])])
            df = pd.DataFrame(blocks,
                             index=[name.replace(' ', '_') for name in series_names],
                             columns=block_names)
            _logger.debug('Block Loading Structure')
            _logger.debug(f'\n{df.to_string()}')
            _logger.debug(f'Blocks shape: {blocks.shape}')
        except Exception as e:
            _logger.debug(f'Error displaying block structure: {e}')
    
    return X, blocks, params_dict


def _prepare_aggregation_structure(
    config: DFMConfig,
    clock: str
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int, np.ndarray]:
    """Prepare aggregation structure for mixed-frequency handling.
    
    Returns
    -------
    tent_weights_dict : dict
        Dictionary mapping frequency pairs to tent weights
    R_mat : np.ndarray or None
        Constraint matrix for tent kernel aggregation
    q : np.ndarray or None
        Constraint vector for tent kernel aggregation
    frequencies : np.ndarray or None
        Array of frequencies for each series
    i_idio : np.ndarray
        Indicator array (1 for clock frequency, 0 for slower frequencies)
    nQ : int
        Number of slower-frequency series
    idio_chain_lengths : np.ndarray
        Array of idiosyncratic chain lengths per series (0, 1, or tent length)
    """
    from .utils import compute_idio_chain_lengths
    
    agg_info = get_aggregation_structure(config, clock=clock)
    tent_weights_dict = agg_info.get('tent_weights', {})
    frequencies = np.array(config.get_frequencies()) if config.series else None
    
    # Find R_mat and q for tent kernel constraints
    R_mat = None
    q = None
    if agg_info['structures']:
        max_periods = 0
        for (slower_freq, clock_freq), (R, q_vec) in agg_info['structures'].items():
            if R is not None:
                n_periods = R.shape[1]
                if n_periods > max_periods:
                    max_periods = n_periods
                    R_mat = R
                    q = q_vec
    
    # Compute i_idio and nQ
    if frequencies is not None:
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        N = len(frequencies)
        i_idio = np.array([
            1 if j >= len(frequencies) or FREQUENCY_HIERARCHY.get(frequencies[j], 3) <= clock_hierarchy
            else 0
            for j in range(N)
        ])
        nQ = N - np.sum(i_idio)
    else:
        i_idio = np.ones(config.get_blocks_array().shape[0])
        nQ = 0
    
    # Compute idio chain lengths
    idio_chain_lengths = compute_idio_chain_lengths(config, clock, tent_weights_dict)
    
    return tent_weights_dict, R_mat, q, frequencies, i_idio, nQ, idio_chain_lengths


def _run_em_algorithm(
    params: EMAlgorithmParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Run EM algorithm until convergence.
    
    Parameters
    ----------
    params : EMAlgorithmParams
        All parameters required for EM algorithm execution
    
    Returns
    -------
    A, C, Q, R, Z_0, V_0 : np.ndarray
        Final parameter estimates
    loglik : float
        Final log-likelihood
    num_iter : int
        Number of iterations completed
    converged : bool
        Whether convergence was achieved
    """
    # Extract parameters from dataclass
    y = params.y
    y_est = params.y_est
    A = params.A
    C = params.C
    Q = params.Q
    R = params.R
    Z_0 = params.Z_0
    V_0 = params.V_0
    r = params.r
    p = params.p
    R_mat = params.R_mat
    q = params.q
    nQ = params.nQ
    i_idio = params.i_idio
    blocks = params.blocks
    tent_weights_dict = params.tent_weights_dict
    clock = params.clock
    frequencies = params.frequencies
    idio_chain_lengths = params.idio_chain_lengths
    config = params.config
    threshold = params.threshold
    max_iter = params.max_iter
    use_damped_updates = params.use_damped_updates
    damping_factor = params.damping_factor
    
    previous_loglik = -np.inf
    num_iter = 0
    converged = False
    
    while num_iter < max_iter and not converged:
        # Create EMStepParams dataclass for em_step()
        from .core.em import EMStepParams
        em_step_params = EMStepParams(
            y=y_est,
            A=A,
            C=C,
            Q=Q,
            R=R,
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=p,
            R_mat=R_mat,
            q=q,
            nQ=nQ,
            i_idio=i_idio,
            blocks=blocks,
            tent_weights_dict=tent_weights_dict,
            clock=clock,
            frequencies=frequencies,
            idio_chain_lengths=idio_chain_lengths,
            config=config
        )
        C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_step_params)
        # Note: em_step returns (C, R, A, Q, Z_0, V_0, loglik) in this order
        
        # Handle likelihood decreases with damped updates
        if num_iter > 0 and loglik < previous_loglik - 1e-3:
            if use_damped_updates:
                damping = damping_factor
                C = damping * C_new + (1 - damping) * C
                R = damping * R_new + (1 - damping) * R
                A = damping * A_new + (1 - damping) * A
                Q = damping * Q_new + (1 - damping) * Q
                Z_0 = damping * Z_0_new + (1 - damping) * Z_0
                V_0 = damping * V_0_new + (1 - damping) * V_0
                
                if loglik < previous_loglik - 0.1:
                    try:
                        _, _, _, loglik_damped = run_kf(y_est, A, C, Q, R, Z_0, V_0)
                        if loglik_damped > previous_loglik:
                            loglik = loglik_damped
                        else:
                            loglik = previous_loglik
                    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                        _logger.debug(f"Likelihood recomputation failed: {type(e).__name__}, using damped update")
            else:
                loglik = previous_loglik
        else:
            C, R, A, Q = C_new, R_new, A_new, Q_new
            Z_0, V_0 = Z_0_new, V_0_new
        
        if num_iter > 2:
            converged, _ = em_converged(loglik, previous_loglik, threshold, True)
        
        if (num_iter % 10 == 0) and (num_iter > 0):
            pct_change = 100 * ((loglik - previous_loglik) / abs(previous_loglik)) if previous_loglik != 0 else 0
            _logger.info(f'Iteration {num_iter}/{max_iter}: loglik={loglik:.6f} ({pct_change:6.2f}% change)')
        
        previous_loglik = loglik
        num_iter += 1
    
    if num_iter < max_iter:
        _logger.info(f'Convergence achieved at iteration {num_iter}')
    else:
        _logger.warning(f'Stopped at maximum iterations ({max_iter}) without convergence')
    
    return A, C, Q, R, Z_0, V_0, loglik, num_iter, converged


def _dfm_core(
    X: np.ndarray,
    config: DFMConfig,
    params: Optional[DFMParams] = None,
    **kwargs
) -> DFMResult:
    """Estimate dynamic factor model using EM algorithm.
    
    This is the main function for estimating a Dynamic Factor Model (DFM). It implements
    the complete EM algorithm workflow:
    
    1. **Initialization**: Compute initial parameter estimates via PCA and OLS
    2. **EM Iterations**: Iteratively update parameters until convergence
    3. **Final Smoothing**: Run Kalman smoother with final parameters to obtain
       smoothed factors and data
    
    The DFM models observed time series as:
    
    .. math::
        y_t = C Z_t + e_t,   e_t \\sim N(0, R)
        Z_t = A Z_{t-1} + v_t,   v_t \\sim N(0, Q)
    
    where:
    - y_t is the n x 1 vector of observed series at time t
    - Z_t is the m x 1 vector of unobserved factors
    - C is the n x m loading matrix
    - A is the m x m transition matrix
    - R and Q are covariance matrices
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N), where T is time periods and N is number of series.
        Data can contain missing values (NaN), which are handled via spline interpolation.
        Missing values are allowed but excessive missing data (>50%) will trigger warnings.
    config : DFMConfig
        Unified DFM configuration object containing:
        - Model structure: Blocks (N x n_blocks), Frequency (per series), 
          Transformation (per series), factors_per_block
        - Estimation parameters: ar_lag, threshold, max_iter, nan_method, nan_k
        Typically obtained from `load_config()`.
    params : DFMParams, optional
        Parameter overrides. If None, all values from config are used.
        All parameters in DFMParams are optional and override corresponding config values.
    **kwargs
        Additional parameter overrides (merged into params if provided).
        Valid parameter names: threshold, max_iter, ar_lag, nan_method, nan_k,
        clock, clip_ar_coefficients, ar_clip_min, ar_clip_max, clip_data_values,
        data_clip_threshold, use_regularization, regularization_scale,
        min_eigenvalue, max_eigenvalue, use_damped_updates, damping_factor.
    
    Returns
    -------
    DFMResult
        Dataclass containing all estimation results:
        - x_sm, X_sm: Smoothed data (standardized and unstandardized)
        - Z: Smoothed factor estimates
        - C, A, Q, R: Estimated parameters
        - Mx, Wx: Standardization parameters
        - Z_0, V_0: Initial state and covariance
        - r, p: Model structure parameters
        
    Raises
    ------
    ValueError
        If inputs are invalid (dimensions, data quality, parameters).
        Also raised during EM iterations if numerical issues occur (e.g., NaN/Inf).
    TypeError
        If input types are incorrect (e.g., X is not numpy array).
        
    Notes
    -----
    - The function automatically standardizes data: x = (X - mean) / std
    - Initial conditions are computed via `init_conditions()`
    - EM iterations continue until convergence or max_iter=5000
    - Missing data is handled by the Kalman filter during estimation
    - Convergence messages and progress are printed during execution
    
    Examples
    --------
    >>> from dfm_python import DFM
    >>> from dfm_python.data import load_config, load_data  # Preferred import
    >>> import pandas as pd
    >>> # Load configuration from YAML or create DFMConfig directly
    >>> config = load_config('config.yaml')
    >>> # Load data from file
    >>> X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))
    >>> # Estimate DFM
    >>> model = DFM()
    >>> Res = model.fit(X, config, threshold=1e-4)
    >>> # Access results
    >>> factors = Res.Z  # (T x m) factor estimates
    >>> loadings = Res.C  # (N x m) factor loadings
    >>> smoothed_data = Res.X_sm  # (T x N) smoothed data
    >>> # Compute common factor (first factor)
    >>> common_factor = Res.Z[:, 0]
    >>> # Project factor onto a series
    >>> series_idx = 0
    >>> series_factor = Res.Z @ Res.C[series_idx, :].T
    """
    _logger.info('Estimating the dynamic factor model (DFM)')
    
    # Merge kwargs into params if provided
    if kwargs:
        if params is None:
            params = DFMParams.from_kwargs(**kwargs)
        else:
            # Update params with kwargs (only valid parameter names)
            valid_params = {
                'threshold', 'max_iter', 'ar_lag', 'nan_method', 'nan_k',
                'clock', 'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
                'clip_data_values', 'data_clip_threshold', 'use_regularization',
                'regularization_scale', 'min_eigenvalue', 'max_eigenvalue',
                'use_damped_updates', 'damping_factor'
            }
            for k, v in kwargs.items():
                if k in valid_params and hasattr(params, k):
                    setattr(params, k, v)
    
    # Step 1: Prepare data and resolve parameters
    X, blocks, params_dict = _prepare_data_and_params(X, config, params)
    
    # Extract parameters from dict for clarity
    p = params_dict['p']
    r = params_dict['r']
    nan_method = params_dict['nan_method']
    nan_k = params_dict['nan_k']
    threshold = params_dict['threshold']
    max_iter = params_dict['max_iter']
    clock = params_dict['clock']
    clip_data_values = params_dict['clip_data_values']
    data_clip_threshold = params_dict['data_clip_threshold']
    use_damped_updates = params_dict['use_damped_updates']
    damping_factor = params_dict['damping_factor']
    T, N = params_dict['T'], params_dict['N']
    
    # Step 2: Prepare aggregation structure
    tent_weights_dict, R_mat, q, frequencies, i_idio, nQ, idio_chain_lengths = _prepare_aggregation_structure(
        config, clock
    )
    
    # Step 3: Standardize data
    x_standardized, Mx, Wx = standardize_data(X, clip_data_values, data_clip_threshold)
    
    # Step 4: Initial conditions
    opt_nan = {'method': nan_method, 'k': nan_k}
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x_standardized, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
        clock=clock, tent_weights_dict=tent_weights_dict, frequencies=frequencies,
        idio_chain_lengths=idio_chain_lengths, config=config
    )
    
    # Verify initial conditions
    if not _check_finite(A, "A") or not _check_finite(C, "C") or not _check_finite(Q, "Q") or not _check_finite(R, "R"):
        _logger.warning("Initial conditions contain NaN/Inf - this should not happen after init_conditions()")
    
    # Step 5: Prepare data for EM (with and without missing values)
    # y contains missing data (NaNs) - handled by Kalman Filter during estimation
    # y_est is used for initial conditions only (missing data removed)
    y = x_standardized.T  # n x T (with missing data - standard DFM approach)
    opt_nan_est = {'method': 3, 'k': nan_k}  # Remove all-NaN rows only for initial conditions
    x_est, _ = rem_nans_spline(x_standardized, method=opt_nan_est['method'], k=opt_nan_est['k'])
    y_est = x_est.T  # n x T (missing data removed)
    
    # Step 6: Run EM algorithm
    em_params = EMAlgorithmParams(
        y=y,
        y_est=y_est,
        A=A,
        C=C,
        Q=Q,
        R=R,
        Z_0=Z_0,
        V_0=V_0,
        r=r,
        p=p,
        R_mat=R_mat,
        q=q,
        nQ=nQ,
        i_idio=i_idio,
        blocks=blocks,
        tent_weights_dict=tent_weights_dict,
        clock=clock,
        frequencies=frequencies,
        idio_chain_lengths=idio_chain_lengths,
        config=config,
        threshold=threshold,
        max_iter=max_iter,
        use_damped_updates=use_damped_updates,
        damping_factor=damping_factor,
    )
    A, C, Q, R, Z_0, V_0, loglik, num_iter, converged = _run_em_algorithm(em_params)
    
    # Step 7: Final Kalman smoothing
    zsmooth, _, _, _ = run_kf(y, A, C, Q, R, Z_0, V_0)
    Zsmooth = zsmooth.T  # m x (T+1) -> (T+1) x m
    
    # Step 8: Compute smoothed data
    x_sm = Zsmooth[1:, :] @ C.T  # T x N (standardized smoothed data)
    Wx_clean = np.where(np.isnan(Wx), 1.0, Wx)
    Mx_clean = np.where(np.isnan(Mx), 0.0, Mx)
    X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized smoothed data)
    
    # Create DFMResult object
    Res = DFMResult(
        x_sm=x_sm,
        X_sm=X_sm,
        Z=Zsmooth[1:, :],  # T x m (skip initial state)
        C=C,
        R=R,
        A=A,
        Q=Q,
        Mx=Mx,
        Wx=Wx,
        Z_0=Z_0,
        V_0=V_0,
        r=r,
        p=p,
        converged=converged,
        num_iter=num_iter,
        loglik=loglik,
        series_ids=safe_get_method(config, 'get_series_ids', []),
        block_names=safe_get_attr(config, 'block_names', None)
    )
    
    # Display diagnostic tables if debug logging is enabled
    if _logger.isEnabledFor(logging.DEBUG):
        _display_dfm_tables(Res, config, nQ)
    
    return Res


# Diagnostic functions are imported directly from core - no proxy functions needed

