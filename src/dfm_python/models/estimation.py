"""Dynamic Factor Model (DFM) estimation using PyTorch Lightning.

This module provides the core DFM estimation function (_dfm_core) which now
uses PyTorch Lightning modules for training. The legacy NumPy-based implementations
have been replaced with PyTorch versions for better performance and GPU support.

The implementation uses a clock-based approach, where all latent factors
evolve at a common clock frequency, with lower-frequency observations
mapped to higher-frequency latent states via deterministic tent kernels.

Note: This module now acts as a wrapper around Lightning modules. The actual
estimation logic is in dfm_python.lightning modules.
"""

import numpy as np
from typing import Tuple, Optional, Any, Dict, Union
import warnings
import logging
import polars as pl
from ..logger import get_logger

from ..config import DFMConfig
from ..utils.time import calculate_rmse
from ..utils.diagnostics import (
    _display_dfm_tables,
)
# Use Lightning modules instead of legacy NumPy implementations
from ..lightning import DFMLightningModule
from ..utils.helpers import (
    safe_get_method, safe_get_attr, resolve_param, safe_mean_std,
    get_series_names, get_frequencies_from_config, ParameterResolver
)

from ..config.structure import (
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
)

from ..config.results import DFMResult, DFMParams

_logger = get_logger(__name__)


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
    
    # Use ParameterResolver for consistent parameter resolution
    resolver = ParameterResolver(config, params)
    params_dict = resolver.resolve_estimation_params()
    
    # Add model structure parameters
    params_dict['r'] = (np.array(config.factors_per_block) 
                        if config.factors_per_block is not None 
                        else np.ones(blocks.shape[1]))
    params_dict['T'] = T
    params_dict['N'] = N
    
    # Display blocks structure if debug logging enabled
    if _logger.isEnabledFor(logging.DEBUG):
        try:
            series_names = get_series_names(config)
            block_names = (config.block_names if len(config.block_names) == blocks.shape[1] 
                          else [f'Block_{i+1}' for i in range(blocks.shape[1])])
            # Create polars DataFrame (no index concept, use row names as column)
            df_dict = {block_names[i]: blocks[:, i].tolist() for i in range(blocks.shape[1])}
            df_dict['series'] = [name.replace(' ', '_') for name in series_names]
            df = pl.DataFrame(df_dict)
            _logger.debug('Block Loading Structure')
            _logger.debug(f'\n{df}')
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
    from .structure import compute_idio_chain_lengths
    
    agg_info = get_aggregation_structure(config, clock=clock)
    tent_weights_dict = agg_info.get('tent_weights', {})
    frequencies = np.array(get_frequencies_from_config(config)) if config.series else None
    
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




def _dfm_core(
    X: np.ndarray,
    config: DFMConfig,
    params: Optional[DFMParams] = None,
    Mx: Optional[np.ndarray] = None,
    Wx: Optional[np.ndarray] = None,
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
        Data should already be transformed and standardized (use DFMDataModule with custom transformer).
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
    Mx : np.ndarray, optional
        Mean values used for standardization (N,). Required if data is already standardized.
    Wx : np.ndarray, optional
        Standard deviation values used for standardization (N,). Required if data is already standardized.
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
    - The function expects data to already be transformed and standardized.
      Use DFMDataModule with custom sktime transformer to apply transformations and standardization.
    - Initial conditions are computed via `init_conditions()`
    - EM iterations continue until convergence or max_iter=5000
    - Missing data is handled by the Kalman filter during estimation
    - Convergence messages and progress are printed during execution
    
    Examples
    --------
    >>> from dfm_python import DFM
    >>> from dfm_python.dataloader import load_data  # Preferred import
    >>> from dfm_python.config import load_config  # Preferred import
    >>> from datetime import datetime
    >>> # Load configuration from YAML or create DFMConfig directly
    >>> config = load_config('config.yaml')
    >>> # Load data from file
    >>> X, Time, Z = load_data('data.csv', config, sample_start=datetime(2000, 1, 1))
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
    
    # Extract key parameters
    r = params_dict['r']
    nan_method = params_dict['nan_method']
    nan_k = params_dict['nan_k']
    threshold = params_dict['threshold']
    max_iter = params_dict['max_iter']
    clock = params_dict['clock']
    
    # Step 2: Prepare aggregation structure (for diagnostics only)
    _, _, _, _, _, nQ, _ = _prepare_aggregation_structure(config, clock)
    
    # Step 3: Validate preprocessing
    if Mx is None or Wx is None:
        raise ValueError(
            "Mx and Wx must be provided. Data should be preprocessed using DFMDataModule with custom transformer. "
            "before calling _dfm_core()."
        )
    
    # Step 4-8: Use Lightning module for estimation
    import torch
    
    # Convert to torch tensor
    X_torch = torch.tensor(X, dtype=torch.float32)
    
    # Create Lightning module
    lightning_module = DFMLightningModule(
        config=config,
        num_factors=int(np.sum(r)),
        threshold=threshold,
        max_iter=max_iter,
        nan_method=nan_method,
        nan_k=nan_k
    )
    
    # Run EM algorithm
    lightning_module.fit_em(X_torch, Mx=Mx, Wx=Wx)
    
    # Extract results
    Res = lightning_module.get_result()
    
    # Display diagnostic tables if debug logging is enabled
    if _logger.isEnabledFor(logging.DEBUG):
        _display_dfm_tables(Res, config, nQ)
    
    return Res

