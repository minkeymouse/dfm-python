"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
DFM inherits from BaseFactorModel since all calculations are performed in NumPy using pykalman.
"""

# Standard library imports
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from torch import Tensor
else:
    try:
        from torch import Tensor
    except ImportError:
        Tensor = Any

# NumPy-based Kalman filter (pykalman) - now a required dependency
from ..ssm.kalman import DFMKalmanFilter

# Local imports
from ..config import (
    DFMConfig,
    ConfigSource,
    DFMResult,
)
from ..config.schema.params import DFMStateSpaceParams, DFMModelState
from ..numeric.tent import get_agg_structure, get_tent_weights, get_slower_freq_tent_weights
from ..config.constants import (
    FREQUENCY_HIERARCHY,
    TENT_WEIGHTS_LOOKUP,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_MAX_ITER,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
    DEFAULT_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_BLOCK_NAME,
)
from ..logger import get_logger
from .base import BaseFactorModel
from ..utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    ConfigurationError,
    DataError,
    PredictionError,
    NumericalError
)
from ..utils.validation import check_condition, has_shape_with_min_dims
from ..utils.helper import handle_linear_algebra_error

# Import EM config from functional module
from ..functional.em import _DEFAULT_EM_CONFIG as _EM_CONFIG
from ..functional.dfm_block import (
    build_lag_matrix,
    initialize_block_loadings,
    initialize_block_transition,
    build_slower_freq_observation_matrix,
    build_slower_freq_idiosyncratic_chain
)
from ..numeric.stability import ensure_covariance_stable, ensure_process_noise_stable
from ..numeric.stability import create_scaled_identity
from ..numeric.estimator import (
    estimate_ar1_unified,
    estimate_variance_unified,
)

if TYPE_CHECKING:
    from ..dataset.dfm_dataset import DFMDataset

_logger = get_logger(__name__)


class DFM(BaseFactorModel):
    """Linear Dynamic Factor Model using EM algorithm with NumPy and pykalman."""
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        num_factors: Optional[int] = None,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        nan_method: Optional[int] = None,
        nan_k: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize DFM instance.
        
        Parameters
        ----------
        config : DFMConfig, optional
            DFM configuration. Can be loaded later via load_config().
        num_factors : int, optional
            Number of factors. If None, inferred from config.
        threshold : float, optional
            EM convergence threshold. Defaults to DEFAULT_CONVERGENCE_THRESHOLD.
        max_iter : int, optional
            Maximum EM iterations. Defaults to DEFAULT_MAX_ITER.
        nan_method : int, optional
            Missing data handling method (internal, defaults to DEFAULT_NAN_METHOD).
        nan_k : int, optional
            Spline interpolation order (internal, defaults to DEFAULT_NAN_K).
        **kwargs : Any
            Additional arguments passed to BaseFactorModel (for API consistency with DDFM).
            
        Returns
        -------
        None
            Initializes DFM instance in-place.
            
        Raises
        ------
        ConfigurationError
            If config validation fails or required parameters are missing.
        ValueError
            If mixed_freq=True and frequency pairs are not in TENT_WEIGHTS_LOOKUP.
        """
        super().__init__()
        
        # Initialize config - create temp config if None
        if config is None:
            config = self._create_temp_config()
        self._config = config
        
        # Resolve parameters using consolidated helper
        # If parameters not explicitly passed, use config values if available, otherwise use defaults
        from ..utils.misc import resolve_param
        # Check config for max_iter and threshold if not explicitly passed
        config_max_iter = getattr(config, 'max_iter', None) if config is not None else None
        config_threshold = getattr(config, 'threshold', None) if config is not None else None
        self.threshold = resolve_param(threshold, default=resolve_param(config_threshold, default=DEFAULT_CONVERGENCE_THRESHOLD))
        self.max_iter = resolve_param(max_iter, default=resolve_param(config_max_iter, default=DEFAULT_MAX_ITER))
        self.nan_method = resolve_param(nan_method, default=DEFAULT_NAN_METHOD)
        self.nan_k = resolve_param(nan_k, default=DEFAULT_NAN_K)
        # Mixed frequency: auto-detected from Dataset or config during fit()
        self._mixed_freq: Optional[bool] = None  # Internal property, auto-detected
        
        # Mixed frequency parameters (set during fit)
        self._constraint_matrix = None  # R_mat: constraint matrix for tent kernel aggregation
        self._constraint_vector = None  # q: constraint vector for tent kernel aggregation
        self._n_slower_freq = 0  # Number of slower-frequency series
        self._tent_weights_dict = None
        self._frequencies = None
        self._idio_indicator = None  # i_idio: indicator for idiosyncratic components
        
        # Determine number of factors
        # Conditional logic: Initialize num_factors from config if not provided (not validation)
        if num_factors is None:
            from ..utils.misc import get_config_attr
            factors_per_block = get_config_attr(config, 'factors_per_block', None)
            if factors_per_block is not None:
                self.num_factors = int(np.sum(factors_per_block))
            else:
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    self.num_factors = int(np.sum(blocks[:, 0]))
                else:
                    self.num_factors = 1
        else:
            self.num_factors = num_factors
        
        # Get model structure (stored as NumPy arrays)
        self.r = np.array(
            config.factors_per_block if config.factors_per_block is not None
            else np.ones(config.get_blocks_array().shape[1]),
            dtype=DEFAULT_DTYPE
        )
        self.p = DEFAULT_FACTOR_ORDER  # Factors always use AR(1) dynamics (simplified)
        self.blocks = np.array(config.get_blocks_array(), dtype=DEFAULT_DTYPE)
        
        # Parameters stored as NumPy arrays (no PyTorch dependencies)
        # Set during fit() and required for prediction
        self.A: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.Z_0: Optional[np.ndarray] = None
        self.V_0: Optional[np.ndarray] = None
        
        
        # Training state
        self.data_processed: Optional[np.ndarray] = None
        self.target_scaler: Optional[Any] = None  # Fitted sklearn scaler for target series inverse transformation
    
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
        
        return DFMConfig(
            frequency={'temp': DEFAULT_CLOCK_FREQUENCY},
            blocks={block_name: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
        )
    
    def _check_parameters_initialized(self) -> None:
        """Check if model parameters are initialized (required for prediction).
        
        Raises
        ------
        ModelNotInitializedError
            If parameters are not initialized
        """
        from ..numeric.validator import validate_parameters_initialized
        validate_parameters_initialized(
            {
                'A': self.A, 'C': self.C, 'Q': self.Q,
                'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
            },
            model_name=self.__class__.__name__
        )
    
    def _rebuild_blocks_array(self, columns: Optional[List[str]], N_actual: int) -> None:
        """Rebuild blocks array to match data dimensions.
        
        Parameters
        ----------
        columns : Optional[List[str]]
            Column names if available
        N_actual : int
            Expected number of series
        """
        if columns is not None:
            # Clear cache and rebuild from config
            if hasattr(self._config, '_cached_blocks'):
                self._config._cached_blocks = None
            blocks_array = self._config.get_blocks_array(columns=columns)
            self.blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)
            _logger.info(f"Rebuilt blocks array: shape={self.blocks.shape}")
            self._log_blocks_diagnostics(columns, N_actual)
        else:
            # Fallback: pad or truncate to match dimensions
            n_blocks = self.blocks.shape[1]
            if self.blocks.shape[0] < N_actual:
                padding = np.zeros((N_actual - self.blocks.shape[0], n_blocks), dtype=DEFAULT_DTYPE)
                self.blocks = np.vstack([self.blocks, padding])
                _logger.warning(f"Padded blocks array with zeros: {N_actual - self.blocks.shape[0]} rows")
            elif self.blocks.shape[0] > N_actual:
                self.blocks = self.blocks[:N_actual, :]
                _logger.warning(f"Truncated blocks array: {self.blocks.shape[0]} -> {N_actual} rows")
    
    def _log_blocks_diagnostics(self, columns: Optional[List[str]], N_actual: int) -> None:
        """Log diagnostics about blocks array.
        
        Parameters
        ----------
        columns : Optional[List[str]]
            Column names if available
        N_actual : int
            Number of series
        """
        n_in_block = np.sum(self.blocks[:, 0] > 0) if self.blocks.shape[1] > 0 else 0
        _logger.info(f"Blocks array: shape={self.blocks.shape}, series in Block_Global: {n_in_block}/{N_actual}")
        
        if n_in_block < N_actual:
            missing_indices = np.where(self.blocks[:, 0] == 0)[0]
            if columns and len(columns) > 0:
                missing_series = [columns[i] for i in missing_indices[:10]]
                _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} ({missing_series[:5]}...)")
            else:
                _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} (indices: {missing_indices[:10].tolist()}...)")
    
    
    def _validate_initialization_numerics(
        self,
        A: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Z_0: np.ndarray,
        V_0: np.ndarray
    ) -> None:
        """Validate numerical stability of initialized parameters."""
        from ..numeric.validator import validate_no_nan_inf
        
        # Check for NaN/Inf using consolidated validation function
        for name, param in [('A', A), ('C', C), ('Q', Q), ('R', R), ('Z_0', Z_0), ('V_0', V_0)]:
            try:
                validate_no_nan_inf(param, name=name)
            except DataValidationError as e:
                # Re-raise as NumericalError for consistency
                raise NumericalError(
                    f"Initialization contains non-finite values in {name}",
                    details=str(e)
                ) from e
        
        # Check for extreme values that could cause numerical issues
        max_abs_values = {
            'A': np.abs(A).max(),
            'C': np.abs(C).max(),
            'Q': np.abs(Q).max(),
            'R': np.abs(R).max(),
            'V_0': np.abs(V_0).max()
        }
        
        from ..config.constants import MAX_EIGENVALUE
        extreme_threshold = MAX_EIGENVALUE * 1e3  # Allow values up to 1e6 for validation warnings
        for name, max_val in max_abs_values.items():
            if max_val > extreme_threshold:
                _logger.warning(
                    f"Large values detected in {name}: max(abs)={max_val:.2e}. "
                    f"This may indicate scaling issues or numerical instability."
                )
        
        # Check covariance matrices are positive definite
        for name, cov in [('Q', Q), ('R', R), ('V_0', V_0)]:
            try:
                eigenvals = np.linalg.eigvals(cov)
                min_eigenval = np.min(eigenvals)
                if min_eigenval < 0:
                    _logger.warning(
                        f"{name} has negative eigenvalues (min={min_eigenval:.2e}). "
                        f"Matrix may not be positive definite."
                    )
                if min_eigenval < 1e-8:
                    _logger.warning(
                        f"{name} has very small eigenvalues (min={min_eigenval:.2e}). "
                        f"This may cause numerical instability."
                    )
            except Exception as e:
                _logger.warning(f"Could not check eigenvalues for {name}: {e}")
        
        # Check data scaling (via observation matrix)
        C_scale = np.abs(C).mean()
        if C_scale > 100 or C_scale < 0.01:
            _logger.warning(
                f"Observation matrix C has unusual scale (mean(abs)={C_scale:.2e}). "
                f"This may indicate data scaling issues."
            )
    
    def _update_parameters(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
                          R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> None:
        """Update model parameters from NumPy arrays.
        
        Parameters
        ----------
        A, C, Q, R, Z_0, V_0 : np.ndarray
            Parameter arrays
        """
        self.A = np.asarray(A, dtype=DEFAULT_DTYPE) if A is not None else None
        self.C = np.asarray(C, dtype=DEFAULT_DTYPE) if C is not None else None
        self.Q = np.asarray(Q, dtype=DEFAULT_DTYPE) if Q is not None else None
        self.R = np.asarray(R, dtype=DEFAULT_DTYPE) if R is not None else None
        self.Z_0 = np.asarray(Z_0, dtype=DEFAULT_DTYPE) if Z_0 is not None else None
        self.V_0 = np.asarray(V_0, dtype=DEFAULT_DTYPE) if V_0 is not None else None
    
    def _initialize_clock_freq_idio(
        self,
        res: np.ndarray,
        data_with_nans: np.ndarray,
        n_clock_freq: int,
        idio_indicator: Optional[np.ndarray],
        T: int,
        dtype: type = DEFAULT_DTYPE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize clock frequency idiosyncratic components (AR(1) for each series).
        
        Returns
        -------
        BM, SM, initViM
        """
        n_idio_clock = n_clock_freq if idio_indicator is None else int(np.sum(idio_indicator))
        BM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        SM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        
        idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
        default_ar_coef = _EM_CONFIG.slower_freq_ar_coef
        default_noise = _EM_CONFIG.default_process_noise
        
        for i, idx in enumerate(idio_indices):
            res_i = data_with_nans[:, idx]
            non_nan_mask = ~np.isnan(res_i)
            if np.sum(non_nan_mask) > 1:
                first_non_nan = np.where(non_nan_mask)[0][0]
                last_non_nan = np.where(non_nan_mask)[0][-1]
                res_i_clean = res[first_non_nan:last_non_nan + 1, idx]
                
                if len(res_i_clean) > 1:
                    def _estimate_ar1_for_idio() -> np.ndarray:
                        # Use unified AR(1) estimation with raw data
                        y_ar = res_i_clean[1:]
                        x_ar = res_i_clean[:-1].reshape(-1, 1)
                        A_diag, Q_diag = estimate_ar1_unified(
                            y=y_ar.reshape(-1, 1),  # (T-1 x 1)
                            x=x_ar,  # (T-1 x 1)
                            V_smooth=None,  # Raw data mode
                            regularization=_EM_CONFIG.matrix_regularization,
                            min_variance=default_noise,
                            default_ar_coef=default_ar_coef,
                            default_noise=default_noise,
                            dtype=dtype
                        )
                        return (A_diag[0] if len(A_diag) > 0 else default_ar_coef,
                                Q_diag[0] if len(Q_diag) > 0 else default_noise)
                    
                    BM[i, i], SM[i, i] = handle_linear_algebra_error(
                        _estimate_ar1_for_idio, "AR(1) estimation for idiosyncratic component",
                        fallback_func=lambda: (default_ar_coef, default_noise)
                    )
                else:
                    BM[i, i] = default_ar_coef
                    SM[i, i] = default_noise
            else:
                BM[i, i] = default_ar_coef
                SM[i, i] = default_noise
        
        # Initial covariance for clock frequency idio
        def _compute_initViM() -> np.ndarray:
            eye_BM = create_scaled_identity(n_idio_clock, DEFAULT_IDENTITY_SCALE, dtype=dtype)
            BM_sq = BM ** 2
            diag_inv = DEFAULT_IDENTITY_SCALE / np.diag(eye_BM - BM_sq)
            diag_inv = np.where(np.isfinite(diag_inv), diag_inv, np.full_like(diag_inv, DEFAULT_IDENTITY_SCALE))
            return np.diag(diag_inv) @ SM
        
        initViM = handle_linear_algebra_error(
            _compute_initViM, "initial covariance computation",
            fallback_func=lambda: SM.copy()
        )
        
        return BM, SM, initViM
    
    def _initialize_block_factors(
        self,
        data_for_extraction: np.ndarray,
        data_with_nans: np.ndarray,
        blocks: np.ndarray,
        r: np.ndarray,
        n_blocks: int,
        n_clock_freq: int,
        tent_kernel_size: int,
        p: int,
        R_mat: Optional[np.ndarray],
        q: Optional[np.ndarray],
        N: int,
        T: int,
        indNaN: np.ndarray,
        max_lag_size: int,
        dtype: type = DEFAULT_DTYPE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize factors and transition matrices block-by-block using sequential PCA.
        
        **Block-by-block extraction process:**
        - Block 1: Extracts factors from original data (data_for_extraction starts as original data)
        - Block 2+: Extracts factors from residuals (data_for_extraction becomes residuals after each block)
        
        This ensures each block captures different variance components, with factors orthogonal across blocks.
        
        Parameters
        ----------
        data_for_extraction : np.ndarray
            Data matrix (T x N). For Block 1, this is the original data (after cleaning).
            For subsequent blocks, this becomes residuals after removing previous blocks' contributions.
        data_with_nans : np.ndarray
            Data matrix with NaNs preserved (T x N)
        blocks : np.ndarray
            Block structure array (N x n_blocks)
        r : np.ndarray
            Number of factors per block (n_blocks,)
        n_blocks : int
            Number of blocks
        n_clock_freq : int
            Number of clock frequency series
        tent_kernel_size : int
            Tent kernel size for mixed-frequency aggregation
        p : int
            VAR lag order
        R_mat : np.ndarray, optional
            Constraint matrix for tent kernel aggregation
        q : np.ndarray, optional
            Constraint vector for tent kernel aggregation
        N : int
            Total number of series
        T : int
            Number of time steps
        indNaN : np.ndarray
            Boolean array indicating missing values
        max_lag_size : int
            Maximum lag size for loading matrix
        dtype : type
            Data type
            
        Returns
        -------
        A_factors : np.ndarray
            Block-diagonal transition matrix for factors
        Q_factors : np.ndarray
            Block-diagonal process noise covariance for factors
        V_0_factors : np.ndarray
            Block-diagonal initial state covariance for factors
        C : np.ndarray
            Observation/loading matrix (N x total_factor_dim)
        """
        C_list = []
        A_list = []
        Q_list = []
        V_0_list = []
        
        # Process each block sequentially
        # Block 1: data_for_extraction = original data
        # Block 2+: data_for_extraction = residuals after previous blocks
        for block_idx in range(n_blocks):
            num_factors_block = int(r[block_idx])
            block_series_indices = np.where(blocks[:, block_idx] > 0)[0]
            clock_freq_indices = block_series_indices[block_series_indices < n_clock_freq]
            slower_freq_indices = block_series_indices[block_series_indices >= n_clock_freq]
            
            _logger.info(f"  Initializing block {block_idx + 1}/{n_blocks}: "
                        f"{num_factors_block} factors, {len(block_series_indices)} series "
                        f"({len(clock_freq_indices)} clock, {len(slower_freq_indices)} slower)")
            
            # Extract factors and loadings for this block
            # Block 1: Uses original data (data_for_extraction = original data)
            # Block 2+: Uses residuals (data_for_extraction = residuals after previous blocks)
            C_i, factors = initialize_block_loadings(
                data_for_extraction, data_with_nans, clock_freq_indices, slower_freq_indices,
                num_factors_block, tent_kernel_size, R_mat, q,
                N, max_lag_size, _EM_CONFIG.matrix_regularization, dtype
            )
            
            # Build lag matrix for transition equation
            lag_matrix = build_lag_matrix(factors, T, num_factors_block, tent_kernel_size, p, dtype)
            slower_freq_factors = lag_matrix[:, :num_factors_block * tent_kernel_size]
            
            # Pad and align factors
            if tent_kernel_size > 1 and slower_freq_factors.shape[0] < T:
                padding = np.zeros((tent_kernel_size - 1, slower_freq_factors.shape[1]), dtype=dtype)
                slower_freq_factors = np.vstack([padding, slower_freq_factors])
                if slower_freq_factors.shape[0] < T:
                    additional_padding = np.zeros((T - slower_freq_factors.shape[0], slower_freq_factors.shape[1]), dtype=dtype)
                    slower_freq_factors = np.vstack([slower_freq_factors, additional_padding])
                slower_freq_factors = slower_freq_factors[:T, :]
            
            # Update data_for_extraction: remove this block's contribution to get residuals for next block
            # After Block 1: data_for_extraction becomes residuals (original_data - Block1_contribution)
            # After Block 2: data_for_extraction becomes residuals (original_data - Block1 - Block2)
            if data_for_extraction.shape[0] != slower_freq_factors.shape[0]:
                slower_freq_factors = slower_freq_factors[:data_for_extraction.shape[0], :]
            data_for_extraction = data_for_extraction - slower_freq_factors @ C_i[:, :num_factors_block * tent_kernel_size].T
            data_with_nans = data_for_extraction.copy()
            data_with_nans[indNaN] = np.nan
            
            C_list.append(C_i)
            
            # Initialize transition matrices
            A_i, Q_i, V_0_i = initialize_block_transition(
                lag_matrix, factors, num_factors_block, max_lag_size, p, T,
                _EM_CONFIG.regularization, _EM_CONFIG.default_transition_coef,
                _EM_CONFIG.default_process_noise, _EM_CONFIG.matrix_regularization,
                _EM_CONFIG.eigenval_floor, dtype
            )
            
            A_list.append(A_i)
            Q_list.append(Q_i)
            V_0_list.append(V_0_i)
        
        # Concatenate loadings
        C = np.hstack(C_list) if C_list else np.zeros((N, 0), dtype=dtype)
        
        # Build block-diagonal matrices
        if A_list:
            A_factors = block_diag(*A_list)
            Q_factors = block_diag(*Q_list)
            V_0_factors = block_diag(*V_0_list)
        else:
            empty_matrix = np.zeros((0, 0), dtype=dtype)
            A_factors = Q_factors = V_0_factors = empty_matrix
        
        return A_factors, Q_factors, V_0_factors, C
    
    def _initialize_slower_freq_idio(
        self,
        R: np.ndarray,
        n_clock_freq: int,
        n_slower_freq: int,
        tent_kernel_size: int,
        dtype: type = DEFAULT_DTYPE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize slower frequency idiosyncratic components (tent kernel chain).
        
        Returns
        -------
        BQ, SQ, initViQ
        """
        if n_slower_freq == 0:
            empty_matrix = np.zeros((0, 0), dtype=dtype)
            return empty_matrix, empty_matrix, empty_matrix
        
        rho0 = _EM_CONFIG.slower_freq_ar_coef
        sig_e = np.diag(R[n_clock_freq:, n_clock_freq:]) / _EM_CONFIG.slower_freq_variance_denominator
        sig_e = np.where(np.isfinite(sig_e), sig_e, _EM_CONFIG.default_observation_noise)
        
        return build_slower_freq_idiosyncratic_chain(n_slower_freq, tent_kernel_size, rho0, sig_e, dtype)
    
    def _find_slower_frequency(
        self,
        clock: str,
        tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> Optional[str]:
        """Find slower frequency from tent_weights_dict or hierarchy.
        
        Parameters
        ----------
        clock : str
            Clock frequency
        tent_weights_dict : Optional[Dict[str, np.ndarray]]
            Dictionary of tent weights by frequency
            
        Returns
        -------
        Optional[str]
            Slower frequency if found, None otherwise
        """
        # Try tent_weights_dict first
        if tent_weights_dict:
            slower_freq = next((freq for freq in tent_weights_dict.keys() if freq != clock), None)
            if slower_freq is not None:
                return slower_freq
        
        # Try slower frequencies from hierarchy (sorted by hierarchy, ascending)
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        slower_freqs = sorted(
            [freq for freq in FREQUENCY_HIERARCHY if FREQUENCY_HIERARCHY[freq] > clock_hierarchy],
            key=lambda f: FREQUENCY_HIERARCHY[f]
        )
        for freq in slower_freqs:
            if get_tent_weights(freq, clock) is not None:
                return freq
        
        # Fallback: first available slower frequency
        for freq in FREQUENCY_HIERARCHY:
            if FREQUENCY_HIERARCHY[freq] > clock_hierarchy and get_tent_weights(freq, clock) is not None:
                return freq
        
        return None
    
    def _add_idiosyncratic_observation_matrix(
        self,
        C: np.ndarray,
        N: int,
        n_clock_freq: int,
        n_slower_freq: int,
        idio_indicator: Optional[np.ndarray],
        clock: str,
        tent_kernel_size: int,
        tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
        dtype: type = DEFAULT_DTYPE
    ) -> np.ndarray:
        """Add idiosyncratic components to observation matrix C.
        
        Returns
        -------
        C : np.ndarray
            Updated observation matrix with idiosyncratic components
        """
        # Clock frequency: identity matrix for each series
        if idio_indicator is not None:
            eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
            idio_indicator_bool = idio_indicator.astype(bool)
            C = np.hstack([C, eyeN[:, idio_indicator_bool]])
        else:
            # Default: all clock frequency series have idiosyncratic components
            if n_clock_freq > 0:
                eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
                C = np.hstack([C, eyeN[:, :n_clock_freq]])
        
        # Slower frequency: tent kernel chain observation matrix
        if n_slower_freq > 0:
            # Determine slower frequency using helper method
            slower_freq = self._find_slower_frequency(clock, tent_weights_dict)
            
            # Get tent weights
            if tent_weights_dict and slower_freq in tent_weights_dict:
                tent_weights = tent_weights_dict[slower_freq].astype(dtype)
            else:
                tent_weights = get_slower_freq_tent_weights(slower_freq or 'q', clock, tent_kernel_size, dtype)
            
            C_slower_freq = build_slower_freq_observation_matrix(N, n_clock_freq, n_slower_freq, tent_weights, dtype)
            C = np.hstack([C, C_slower_freq])
        
        return C
    
    def _initialize_observation_noise(
        self,
        data_with_nans: np.ndarray,
        N: int,
        idio_indicator: Optional[np.ndarray],
        n_clock_freq: int,
        dtype: type = DEFAULT_DTYPE
    ) -> np.ndarray:
        """Initialize observation noise covariance R from residuals.
        
        Missing values (NaN) are handled via nan-aware statistics - only valid observations
        are used for variance estimation. NaN will be handled by Kalman filter during EM.
        
        Returns
        -------
        R : np.ndarray
            Observation noise covariance (N x N, diagonal)
        """
        # Ensure 2D
        if data_with_nans.ndim != 2:
            data_with_nans = data_with_nans.reshape(-1, N) if data_with_nans.size > 0 else np.zeros((1, N), dtype=dtype)
        
        T_res, N_res = data_with_nans.shape
        default_obs_noise = _EM_CONFIG.default_observation_noise
        
        # Use unified variance estimation with raw residuals (handles NaN via nan-aware stats)
        if T_res <= 1:
            from ..numeric.stability import create_scaled_identity
            R = create_scaled_identity(N_res, default_obs_noise, dtype)
        else:
            # Compute residuals (data itself, since we're initializing from raw data)
            # estimate_variance_unified uses nan-aware variance if residuals contain NaN
            R = estimate_variance_unified(
                residuals=data_with_nans,  # Raw data as "residuals" for initialization (may contain NaN)
                X=None,  # Not using smoothed expectations mode
                EZ=None,
                C=None,
                V_smooth=None,
                min_variance=default_obs_noise,
                default_variance=default_obs_noise,
                dtype=dtype
            )
        
        # Set variances for idiosyncratic series to default
        idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
        all_indices = np.unique(np.concatenate([idio_indices, np.arange(n_clock_freq, N, dtype=np.int32)]))
        R[np.ix_(all_indices, all_indices)] = np.diag(np.full(len(all_indices), default_obs_noise, dtype=dtype))
        
        return R
    
    def _initialize_parameters(
        self,
        x: np.ndarray,
        r: np.ndarray,
        p: int,
        blocks: np.ndarray,
        opt_nan: Dict[str, Any],
        R_mat: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        n_slower_freq: int = 0,
        idio_indicator: Optional[np.ndarray] = None,
        clock: str = DEFAULT_CLOCK_FREQUENCY,
        tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize DFM state-space parameters using sequential residual-based PCA."""
        T, N = x.shape
        dtype = DEFAULT_DTYPE
        
        n_blocks = blocks.shape[1]
        n_clock_freq = N - n_slower_freq  # Number of clock frequency series
        
        # Kalman filter handles missing values natively via masked arrays - preserve NaN
        x_clean = np.where(np.isinf(x), np.nan, x)  # Replace Inf with NaN, keep existing NaN
        
        # Check data scale for numerical stability
        # Detect potential RobustScaler issues (IQR≈0) vs StandardScaler (std≈1, mean≈0)
        valid_mask = np.isfinite(x_clean)
        if valid_mask.any():
            data_std = np.nanstd(x_clean)
            data_mean = np.nanmean(x_clean)
            data_median = np.nanmedian(x_clean)
            data_iqr = np.nanpercentile(x_clean, 75) - np.nanpercentile(x_clean, 25)
            
            # Check for RobustScaler issues: IQR≈0 indicates potential scaling problems
            # StandardScaler: mean≈0, std≈1, IQR≈1.35 (for normal distribution)
            # RobustScaler with IQR≈0: can produce extreme values
            has_zero_iqr = data_iqr < 1e-6
            
            # Check for scale mismatch
            has_scale_mismatch = (
                data_std > 10 or abs(data_mean) > 3 or abs(data_median) > 3 or
                (data_std < 0.01 and not has_zero_iqr)  # Very small std (might indicate no scaling)
            )
            
            if has_scale_mismatch or has_zero_iqr:
                _logger.warning(f"  Data scale check: mean={data_mean:.2f}, median={data_median:.2f}, "
                              f"std={data_std:.2f}, IQR={data_iqr:.2e}")
                if has_zero_iqr:
                    _logger.warning(f"  ⚠ CRITICAL: IQR≈0 detected (IQR={data_iqr:.2e}). "
                                  f"This may indicate RobustScaler issues with near-constant series.")
                    _logger.warning(f"  RobustScaler with IQR≈0 can produce extreme scaled values → large covariances → numerical instability.")
                    _logger.warning(f"  Recommendation: Use StandardScaler or check for constant/zero-variance series before scaling.")
                else:
                    _logger.warning(f"  Data may not be properly standardized (expected: mean≈0, std≈1 for StandardScaler). "
                                  f"This may cause numerical instability.")
                    _logger.warning(f"  Check: Is data being scaled multiple times or not at all?")
        
        # Initialize data for factor extraction
        # Block 1: original data. Subsequent blocks: residuals. NaN preserved for Kalman filter
        data_for_extraction = x_clean.copy()
        data_with_nans = x_clean.copy()
        indNaN = np.isnan(x_clean)  # Track NaN positions for initialization (only)
        
        # Determine tent kernel size
        if R_mat is not None:
            tent_kernel_size = R_mat.shape[1]
        elif tent_weights_dict:
            # Use first available tent weights
            first_weights = next(iter(tent_weights_dict.values()))
            tent_kernel_size = len(first_weights)
        else:
            tent_kernel_size = _EM_CONFIG.tent_kernel_size
        # State dimension per factor = max(p + 1, tent_kernel_size)
        max_lag_size = max(p + 1, tent_kernel_size)
        
        # Set initial observations as NaN for slower-frequency aggregation
        if tent_kernel_size > 1:
            data_with_nans[:tent_kernel_size-1, :] = np.nan
        
        # Initialize factors and loadings block-by-block
        # Block 1 uses original data, subsequent blocks use residuals
        A_factors, Q_factors, V_0_factors, C = self._initialize_block_factors(
            data_for_extraction, data_with_nans, blocks, r, n_blocks, n_clock_freq, tent_kernel_size,
            p, R_mat, q, N, T, indNaN, max_lag_size, dtype
        )
        
        # === IDIOSYNCRATIC COMPONENTS ===
        C = self._add_idiosyncratic_observation_matrix(
            C, N, n_clock_freq, n_slower_freq, idio_indicator, clock, tent_kernel_size, tent_weights_dict, dtype
        )
        
        # Initialize R (observation noise covariance) from final residuals
        R = self._initialize_observation_noise(data_with_nans, N, idio_indicator, n_clock_freq, dtype)
        
        # === IDIOSYNCRATIC TRANSITION MATRICES ===
        # Clock frequency: AR(1) for each series
        # Use final residuals (after all blocks) for idiosyncratic component initialization
        BM, SM, initViM = self._initialize_clock_freq_idio(
            data_for_extraction, data_with_nans, n_clock_freq, idio_indicator, T, dtype=dtype
        )
        
        # Slower frequency: tent kernel chain
        BQ, SQ, initViQ = self._initialize_slower_freq_idio(
            R, n_clock_freq, n_slower_freq, tent_kernel_size, dtype=dtype
        )
        
        # Combine all transition matrices
        def _construct_block_diagonal() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A = block_diag(A_factors, BM, BQ)
            Q = block_diag(Q_factors, SM, SQ)
            V_0 = block_diag(V_0_factors, initViM, initViQ)
            return A, Q, V_0
        
        def _raise_block_diagonal_error(*args, **kwargs) -> None:
            error_msg = "Failed to construct block-diagonal matrices. Check matrix dimensions and ensure all blocks are valid."
            _logger.error(error_msg)
            raise NumericalError(error_msg)
        
        A, Q, V_0 = handle_linear_algebra_error(
            _construct_block_diagonal, "block-diagonal matrix construction",
            fallback_func=_raise_block_diagonal_error
        )
        
        # Initial state: Z_0 = zeros
        m = int(A.shape[0]) if A.size > 0 and has_shape_with_min_dims(A, min_dims=1) else 0
        Z_0 = np.zeros(m, dtype=DEFAULT_DTYPE)
        
        # Ensure Q is positive definite and bounded (SQ has sparse structure which can cause zero eigenvalues)
        # This is critical for Kalman filter stability - prevents both singularity and explosion
        Q = ensure_process_noise_stable(Q, min_eigenval=_EM_CONFIG.eigenval_floor, warn=True, dtype=DEFAULT_DTYPE)
        
        # Ensure V_0 is positive definite
        V_0 = ensure_covariance_stable(V_0, min_eigenval=_EM_CONFIG.eigenval_floor)
        
        return A, C, Q, R, Z_0, V_0
    
    def fit(
        self,
        X: Union[np.ndarray, Any],
        dataset: Optional[Any] = None,
        checkpoint_callback: Optional[Any] = None
    ) -> DFMStateSpaceParams:
        """Fit model using EM algorithm (wrapper around pykalman).
        
        Uses pykalman for E-step (Kalman filter/smoother) and custom M-step
        that preserves block structure and mixed-frequency constraints.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor, optional
            Standardized data (T x N). If dataset is provided, X can be None.
        dataset : DFMDataset, optional
            Custom DFMDataset instance. If provided, initialization parameters will be
            extracted from the dataset instead of computing them directly.
            Target scaler is obtained from dataset.target_scaler for inverse transformation.
            
        Returns
        -------
        DFMStateSpaceParams
            Fitted state-space parameters (A, C, Q, R, Z_0, V_0)
        """
        # Clear all caches for fresh training run (ensures no stale data from previous runs)
        if self._config is not None and hasattr(self._config, '_cached_blocks'):
            self._config._cached_blocks = None
        
        # Extract initialization parameters (from dataset or X)
        if dataset is not None:
            # Use dataset's built-in method - it already handles all parameter extraction
            self._dataset = dataset
            from ..utils.misc import get_target_scaler
            self.target_scaler = get_target_scaler(dataset=dataset)
            
            init_params = dataset.get_initialization_params()
            if self._mixed_freq is None:
                self._mixed_freq = init_params.get('is_mixed_freq', False)
            
            X_np = init_params['X']
            R_mat = init_params['R_mat']
            q = init_params['q']
            n_slower_freq = init_params['n_slower_freq']
            tent_weights_dict = init_params['tent_weights_dict']
            frequencies_np = init_params['frequencies']
            idio_indicator = init_params['idio_indicator']
            opt_nan = init_params['opt_nan']
            clock = init_params['clock']
            n_clock_freq = X_np.shape[1] - n_slower_freq
        else:
            # Create a temporary dataset to reuse its parameter extraction logic
            # This avoids duplicating mixed-frequency setup code
            from ..dataset.dfm_dataset import DFMDataset
            temp_dataset = DFMDataset(config=self._config, data=X)
            init_params = temp_dataset.get_initialization_params()
            
            X_np = init_params['X']
            R_mat = init_params['R_mat']
            q = init_params['q']
            n_slower_freq = init_params['n_slower_freq']
            tent_weights_dict = init_params['tent_weights_dict']
            frequencies_np = init_params['frequencies']
            idio_indicator = init_params['idio_indicator']
            opt_nan = init_params['opt_nan']
            clock = init_params['clock']
            n_clock_freq = X_np.shape[1] - n_slower_freq
            
            if self._mixed_freq is None:
                self._mixed_freq = init_params.get('is_mixed_freq', False)
        
        self.data_processed = X_np
        
        # Rebuild blocks array to match actual data dimensions
        N_actual = X_np.shape[1]
        # Get column names from dataset if available (it already stores _processed_columns)
        columns = None
        if dataset is not None and hasattr(dataset, '_processed_columns'):
            columns = list(dataset._processed_columns) if len(dataset._processed_columns) == N_actual else None
        
        if self.blocks.shape[0] != N_actual:
            self._rebuild_blocks_array(columns, N_actual)
        else:
            self._log_blocks_diagnostics(columns, N_actual)
        
        # Store for reuse in EM steps
        self._constraint_matrix = R_mat
        self._constraint_vector = q
        self._n_slower_freq = n_slower_freq
        self._n_clock_freq = n_clock_freq if 'n_clock_freq' in locals() else (X_np.shape[1] - n_slower_freq)
        self._tent_weights_dict = tent_weights_dict
        self._frequencies = frequencies_np
        self._idio_indicator = idio_indicator
        
        # Compute max_lag_size for state dimension (used in both initialization and EM)
        # For mixed-frequency data, use tent_kernel_size; otherwise use p+1
        # State dimension per factor = max(p + 1, tent_kernel_size)
        if R_mat is not None:
            tent_kernel_size = R_mat.shape[1]
        elif tent_weights_dict:
            first_weights = next(iter(tent_weights_dict.values()))
            tent_kernel_size = len(first_weights)
        else:
            tent_kernel_size = 1  # No tent kernel for single-frequency data
        self._max_lag_size = max(self.p + 1, tent_kernel_size)
        
        # Initialize parameters (required for EM algorithm)
        _logger.info("Initializing DFM parameters...")
        _logger.info(f"  Data: {X_np.shape[0]} time steps × {X_np.shape[1]} series")
        _logger.info(f"  Blocks: {self.blocks.shape[1]}, Factors: {self.num_factors}, "
                    f"Mixed freq: {self._mixed_freq}")
        A_np, C_np, Q_np, R_np, Z_0_np, V_0_np = self._initialize_parameters(
            X_np, self.r, self.p, self.blocks, opt_nan, R_mat, q, n_slower_freq, idio_indicator,
            clock, tent_weights_dict
        )
        
        # Validate numerical stability before proceeding
        self._validate_initialization_numerics(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        self._update_parameters(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        self._check_parameters_initialized()
        
        _logger.info(f"Initialization complete: state_dim={self.A.shape[0]}, obs_dim={self.C.shape[0]}, "
                    f"factors={self.num_factors}, max_lag={self._max_lag_size}, mixed_freq={self._mixed_freq}")
        
        # Run EM algorithm using DFMKalmanFilter.em() method
        # Create Kalman filter and run EM algorithm
        try:
            kalman_filter = DFMKalmanFilter(
                transition_matrices=self.A,
                observation_matrices=self.C,
                transition_covariance=self.Q,
                observation_covariance=self.R,
                initial_state_mean=self.Z_0,
                initial_state_covariance=self.V_0
            )
            _logger.debug("Kalman filter created successfully")
        except Exception as e:
            _logger.error(f"Failed to create Kalman filter: {e}", exc_info=True)
            _logger.error(f"  Parameter shapes - A: {self.A.shape}, C: {self.C.shape}, Q: {self.Q.shape}, R: {self.R.shape}")
            _logger.error(f"  Z_0: {self.Z_0.shape}, V_0: {self.V_0.shape}")
            raise
        
        # Create EMConfig from DFMConfig (uses consolidated parameters)
        em_config = self._config.to_em_config() if self._config is not None else _EM_CONFIG
        
        try:
            final_state = kalman_filter.em(
                X=X_np,
                initial_params={
                    'A': self.A, 'C': self.C, 'Q': self.Q,
                    'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
                },
                max_iter=self.max_iter,
                threshold=self.threshold,
                config=em_config,
                blocks=self.blocks,
                r=self.r,
                p=self.p,
                p_plus_one=self._max_lag_size,  # Use max_lag_size for state dimension (accounts for tent_kernel_size)
                R_mat=self._constraint_matrix,
                q=self._constraint_vector,
                n_clock_freq=self._n_clock_freq,
                n_slower_freq=n_slower_freq,
                idio_indicator=self._idio_indicator,
                tent_weights_dict=self._tent_weights_dict,
                checkpoint_callback=checkpoint_callback
            )
        except Exception as e:
            _logger.error(f"EM algorithm failed: {e}", exc_info=True)
            _logger.error(f"  Initialization parameters:")
            _logger.error(f"    A shape: {self.A.shape}, C shape: {self.C.shape}")
            _logger.error(f"    Q shape: {self.Q.shape}, R shape: {self.R.shape}")
            _logger.error(f"    Z_0 shape: {self.Z_0.shape}, V_0 shape: {self.V_0.shape}")
            _logger.error(f"    Blocks shape: {self.blocks.shape}, r: {self.r}")
            _logger.error(f"    p: {self.p}, max_lag_size: {self._max_lag_size}")
            raise
        
        # Update model parameters from final state
        self._update_parameters(
            final_state['A'], final_state['C'], final_state['Q'],
            final_state['R'], final_state['Z_0'], final_state['V_0']
        )
        
        # Store state-space parameters
        self.training_state = DFMStateSpaceParams(
            A=final_state['A'], C=final_state['C'], Q=final_state['Q'],
            R=final_state['R'], Z_0=final_state['Z_0'], V_0=final_state['V_0']
        )
        
        # Store training metadata separately (not in state-space params)
        self._training_loglik = final_state['loglik']
        self._training_num_iter = final_state['num_iter']
        self._training_converged = final_state['converged']
        
        return self.training_state
    
    def _create_kalman_filter(
        self,
        initial_state_mean: Optional[np.ndarray] = None,
        initial_state_covariance: Optional[np.ndarray] = None
    ) -> DFMKalmanFilter:
        """Create Kalman filter with current training state parameters.
        
        Parameters
        ----------
        initial_state_mean : np.ndarray, optional
            Initial state mean. If None, uses training_state.Z_0
        initial_state_covariance : np.ndarray, optional
            Initial state covariance. If None, uses training_state.V_0
            
        Returns
        -------
        DFMKalmanFilter
            Configured Kalman filter instance
        """
        if initial_state_mean is None:
            initial_state_mean = self.training_state.Z_0
        if initial_state_covariance is None:
            initial_state_covariance = self.training_state.V_0
        
        return DFMKalmanFilter(
            transition_matrices=self.training_state.A,
            observation_matrices=self.training_state.C,
            transition_covariance=self.training_state.Q,
            observation_covariance=self.training_state.R,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )
    
    def _compute_smoothed_factors(self) -> np.ndarray:
        """Compute smoothed factors using Kalman filter.
        
        Returns
        -------
        np.ndarray
            Smoothed factors (T x m)
        """
        check_condition(
            self.training_state is not None and self.data_processed is not None,
            ModelNotTrainedError,
            "Model not fitted or data not available",
            details="Please call fit() method before computing smoothed factors"
        )
        
        kalman_final = self._create_kalman_filter()
        y_masked = np.ma.masked_invalid(self.data_processed)
        smoothed_state_means, _ = kalman_final.smooth(y_masked)
        return smoothed_state_means
    
    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        # Compute smoothed factors (validates training_state and data_processed internally)
        Z = self._compute_smoothed_factors()
        
        # Get parameters
        A = self.training_state.A
        C = self.training_state.C
        Q = self.training_state.Q
        R = self.training_state.R
        Z_0 = self.training_state.Z_0
        V_0 = self.training_state.V_0
        
        # Compute smoothed data
        x_sm = Z @ C.T
        
        # Get target scaler from dataset if available
        from ..utils.misc import get_target_scaler
        target_scaler = get_target_scaler(model=self)
        
        # Get training metadata from instance attributes
        converged = getattr(self, '_training_converged', False)
        num_iter = getattr(self, '_training_num_iter', 0)
        loglik = getattr(self, '_training_loglik', 0.0)
        
        return DFMResult(
            x_sm=x_sm, Z=Z, C=C, R=R, A=A, Q=Q,
            target_scaler=target_scaler,
            Z_0=Z_0, V_0=V_0, r=self.r, p=self.p,
            converged=converged,
            num_iter=num_iter,
            loglik=loglik
        )
    
    
    def update(self, data: Union[np.ndarray, Any]) -> None:
        """Update model state with new observations via Kalman filtering/smoothing.
        
        This method runs Kalman filtering/smoothing on new data to update the
        latent factors, but keeps model parameters (A, C, Q, R) fixed.
        
        After calling update(), the model's internal state (result.Z and data_processed)
        is extended with the new observations. Subsequent calls to predict() will use
        the updated state.
        
        **Data Shape**: The input data must be 2D with shape (T_new x N) where:
        - T_new: Number of new time steps (can be any positive integer)
        - N: Number of series (must match training data)
        
        **Supported Types**:
        - numpy.ndarray: (T_new x N) array
        - pandas.DataFrame: DataFrame with N columns, T_new rows
        - polars.DataFrame: DataFrame with N columns, T_new rows
        
        **Important**: Data must be preprocessed by the user (same preprocessing as training).
        Only target scaler is handled internally if needed.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations with shape (T_new x N) where:
            - T_new: Number of new time steps (any positive integer)
            - N: Number of series (must match training data)
            Data must be preprocessed by user (same preprocessing as training).
            
        Notes
        -----
        - This updates factors via filtering/smoothing, NOT parameter retraining
        - For parameter retraining, use fit() with concatenated data
        - After update(), predict() will use the updated factor state
        - New data must have same number of series (N) as training data
        - User must preprocess data themselves (same preprocessing as training)
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        DataValidationError
            If data shape doesn't match training data
        """
        # Validate and convert data (no preprocessing - user must preprocess)
        from ..numeric.validator import validate_and_convert_update_data
        data_new = validate_and_convert_update_data(
            data, 
            self.data_processed, 
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        
        # Get current result (compute if needed)
        result = self._ensure_result()
        
        # Get last smoothed state from training as initial state for new data
        # Check both shape dimensions and length to safely access last row
        Z_last = result.Z[-1, :] if (has_shape_with_min_dims(result.Z, min_dims=1) and result.Z.shape[0] > 0) else result.Z_0
        V_last = result.V_0  # Use original V_0 or could compute from last state covariance
        
        # Create Kalman filter with current parameters and new initial state
        kalman_new = self._create_kalman_filter(
            initial_state_mean=Z_last,
            initial_state_covariance=V_last
        )
        
        # Run filter and smooth on new data
        y_masked = np.ma.masked_invalid(data_new)
        Z_new, V_smooth_new, _, _ = kalman_new.filter_and_smooth(y_masked)
        
        # Update model state: append new factors and data
        # Concatenate new factors to existing result.Z
        result.Z = np.vstack([result.Z, Z_new])
        
        # Append new data to data_processed
        self.data_processed = np.vstack([self.data_processed, data_new])
        
        # Update smoothed data (x_sm) in result
        result.x_sm = result.Z @ result.C.T
        
        # Invalidate cached result (or it's already updated above)
        # The result object is updated in place, so _result is still valid
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> 'DFM':
        """Load configuration from various sources.
        
        After loading config, the model needs to be re-initialized with the new config.
        For standard pattern, pass config directly to __init__.
        """
        new_config = self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
        )
        
        # DFM-specific: Initialize r and blocks arrays
        self.r = np.array(
            new_config.factors_per_block if new_config.factors_per_block is not None
            else np.ones(new_config.get_blocks_array().shape[1]),
            dtype=DEFAULT_DTYPE
        )
        self.blocks = np.array(new_config.get_blocks_array(), dtype=DEFAULT_DTYPE)
        
        return self
    
    
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        data: Optional[Union[np.ndarray, Tensor, Any]] = None,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the model to generate forecasts.
        
        Target series are determined from the Dataset's target_series attribute,
        which should be set during Dataset initialization.
        
        **New Data Initialization**: If `data` is provided, the method runs a
        Kalman filter forward pass on the new data to compute the initial factor
        state for forecasting. This does NOT modify the model's internal state.
        Use `update()` method if you want to update the model state with new data.
        
        **Important**: If `data` is provided, it must be preprocessed by the user
        (same preprocessing as training). Only target scaler is handled internally.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        data : np.ndarray, torch.Tensor, pandas.DataFrame, or polars.DataFrame, optional
            New preprocessed observations to use for initializing forecast. If provided,
            runs Kalman filter forward pass to compute Z_last from the new data.
            Does NOT modify model state (use update() for that).
            Must have shape (T_new x N) where N matches training data.
            Data must be preprocessed by user (same preprocessing as training).
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x len(target_series))
            If only return_factors is True:
                Z_forecast (horizon x m)
            
        Raises
        ------
        ValueError
            If Dataset has no target_series set
        ModelNotTrainedError
            If model has not been trained yet
        DataValidationError
            If data shape doesn't match training data
        """
        # Validate model is trained
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() first"
        )
        
        # Validate parameters are initialized
        self._check_parameters_initialized()
        
        # Get result (compute if needed)
        result = self._ensure_result()
        
        check_condition(
            result.Z is not None,
            ModelNotTrainedError,
            "DFM prediction failed: result.Z is not available",
            details="This may indicate the model was not properly trained or result object is corrupted"
        )
        
        if horizon is None:
            from ..utils.misc import compute_default_horizon
            horizon = compute_default_horizon(self._config)
        from ..numeric.validator import validate_horizon
        horizon = validate_horizon(horizon)
        
        # Extract model parameters
        A = result.A
        C = result.C
        target_scaler = result.target_scaler  # Use scaler object instead of Mx/Wx
        p = result.p  # VAR order (always available after training)
        
        # Determine initial factor state
        if data is not None:
            # Use new data to compute initial factor state via Kalman filter forward pass
            # This does NOT modify model state
            # User must preprocess data themselves (same preprocessing as training)
            from ..numeric.validator import validate_and_convert_update_data
            data_new = validate_and_convert_update_data(
                data,
                self.data_processed,
                dtype=DEFAULT_DTYPE,
                model_name=self.__class__.__name__
            )
            
            # Get last smoothed state from training as initial state for filtering
            # Check both shape dimensions and length to safely access last row
            Z_initial = result.Z[-1, :] if (has_shape_with_min_dims(result.Z, min_dims=1) and result.Z.shape[0] > 0) else result.Z_0
            V_initial = result.V_0
            
            # Create Kalman filter with current parameters
            kalman_filter = self._create_kalman_filter(
                initial_state_mean=Z_initial,
                initial_state_covariance=V_initial
            )
            
            # Run filter (forward pass only, not smooth) on new data
            y_masked = np.ma.masked_invalid(data_new)
            filtered_states, _ = kalman_filter.filter(y_masked)
            
            # Extract last filtered state as initial state for forecasting
            # Check both shape dimensions and length to safely access last row (consistent with other shape checks)
            Z_last = filtered_states[-1, :] if (has_shape_with_min_dims(filtered_states, min_dims=1) and filtered_states.shape[0] > 0) else Z_initial
        else:
            # Use training state for initial factor state
            # For DFM, we use the last smoothed state from training
            # Check both shape dimensions and length to safely access last row
            Z_last = result.Z[-1, :] if (has_shape_with_min_dims(result.Z, min_dims=1) and result.Z.shape[0] > 0) else np.zeros(result.A.shape[0], dtype=DEFAULT_DTYPE)
        
        # Validate factor state and parameters are finite
        from ..numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(Z_last, name="factor state Z_last")
        validate_no_nan_inf(A, name="transition matrix A")
        validate_no_nan_inf(C, name="observation matrix C")
        
        from ..utils.misc import resolve_target_series
        series_ids = self._config.get_series_ids() if self._config is not None else result.series_ids
        dataset = None
        try:
            dataset = self._get_dataset()
        except (ModelNotInitializedError, AttributeError):
            pass
        target_series, target_indices = resolve_target_series(dataset, series_ids, result, self.__class__.__name__)
        
        # Additional validation: ensure target_series was set in Dataset
        if target_series is None or len(target_series) == 0:
            raise ValueError(
                "DFM prediction failed: no target_series found in Dataset. "
                "Please set target_series when creating the Dataset (e.g., DFMDataset(..., target_series=['series_id']))."
            )
        
        from ..numeric.estimator import forecast_ar1_factors
        Z_forecast = forecast_ar1_factors(Z_last, A, horizon, dtype=DEFAULT_DTYPE)
        
        # Optimized: Transform only target series (not all series)
        # Use only target indices for C
        C_target = C[target_indices, :]  # (len(target) x m)
        
        # Transform factors to target observations (in standardized scale)
        X_forecast_std = Z_forecast @ C_target.T  # (horizon x len(target))
        
        # Unscale target series using fitted scaler if available
        if target_scaler is not None and hasattr(target_scaler, 'inverse_transform'):
            X_forecast = target_scaler.inverse_transform(X_forecast_std)
        else:
            X_forecast = X_forecast_std
        
        # Ensure X_forecast is numpy array and validate it's finite
        from ..config.types import to_numpy
        X_forecast = to_numpy(X_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(X_forecast, name="forecast X_forecast")
        
        # Validate forecast values are within reasonable bounds (optional, if scaler available)
        if target_scaler is not None and hasattr(target_scaler, 'scale_'):
            try:
                scale_vals = target_scaler.scale_
                if scale_vals is not None and len(scale_vals) > 0:
                    extreme_threshold_std = _EM_CONFIG.extreme_forecast_threshold
                    n_series = X_forecast.shape[1] if X_forecast.ndim > 1 else 1
                    for i in range(min(n_series, len(scale_vals))):
                        if scale_vals[i] > 0:
                            series_forecast = X_forecast[:, i] if X_forecast.ndim > 1 else X_forecast
                            abs_deviations = np.abs(series_forecast) / scale_vals[i]
                            max_deviation = np.max(abs_deviations) if len(abs_deviations) > 0 else DEFAULT_ZERO_VALUE
                            if max_deviation > extreme_threshold_std:
                                _logger.warning(
                                    f"DFM prediction: Extreme forecast for target series {i} "
                                    f"(max deviation: {max_deviation:.1f} std devs). "
                                    f"Possible numerical instability."
                                )
            except (AttributeError, TypeError, ValueError):
                pass  # Skip validation if scaler attributes unavailable
        
        if return_factors:
            validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    @property
    def result(self) -> DFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        """
        result = self._ensure_result()
        # Type assertion: get_result() always returns DFMResult for DFM model
        assert isinstance(result, DFMResult), f"Expected DFMResult but got {type(result)}"
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save DFM model to file.
        
        Saves the complete model state using the defined dataclasses:
        - State-space parameters (DFMStateSpaceParams dataclass)
        - Model state (DFMModelState dataclass)
        - Training metadata (loglik, num_iter, converged as simple fields)
        - Result (DFMResult dataclass, if model is trained)
        - Configuration
        
        Parameters
        ----------
        path : str or Path
            Path to save the model checkpoint file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get result dataclass if model is trained
        result = None
        if self.training_state is not None:
            try:
                result = self.get_result()
            except (ModelNotTrainedError, ModelNotInitializedError):
                pass
        
        # Collect checkpoint using dataclasses
        checkpoint = {
            'state_space_params': self.training_state,  # DFMStateSpaceParams
            'model_state': DFMModelState.from_model(self),
            'result': result,
            'config': self._config,
            'threshold': self.threshold,
            'max_iter': self.max_iter,
            'nan_method': self.nan_method,
            'nan_k': self.nan_k,
            'data_processed': self.data_processed,
            'target_scaler': self.target_scaler,
            # Training metadata stored as simple fields
            'training_loglik': getattr(self, '_training_loglik', None),
            'training_num_iter': getattr(self, '_training_num_iter', None),
            'training_converged': getattr(self, '_training_converged', None),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        _logger.info(f"DFM model saved to {path}")
    
    @staticmethod
    def _extract_state_space_from_old_format(old_training_state: Any) -> DFMStateSpaceParams:
        """Extract state-space parameters from old checkpoint format.
        
        Parameters
        ----------
        old_training_state : Any
            Old training state (object with A,C,Q,R,Z_0,V_0 or dict)
            
        Returns
        -------
        DFMStateSpaceParams
            State-space parameters
        """
        if hasattr(old_training_state, 'A'):
            return DFMStateSpaceParams(
                A=old_training_state.A,
                C=old_training_state.C,
                Q=old_training_state.Q,
                R=old_training_state.R,
                Z_0=old_training_state.Z_0,
                V_0=old_training_state.V_0
            )
        elif isinstance(old_training_state, dict):
            return DFMStateSpaceParams(
                A=old_training_state.get('A'),
                C=old_training_state.get('C'),
                Q=old_training_state.get('Q'),
                R=old_training_state.get('R'),
                Z_0=old_training_state.get('Z_0'),
                V_0=old_training_state.get('V_0')
            )
        else:
            # Assume it's already DFMStateSpaceParams
            return old_training_state
    
    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[DFMConfig] = None) -> 'DFM':
        """Load DFM model from checkpoint file.
        
        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file
        config : DFMConfig, optional
            Configuration (if None, loaded from checkpoint)
            
        Returns
        -------
        DFM
            Loaded DFM model instance
        """
        path = Path(path)
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Use checkpoint config if not provided
        if config is None:
            config = checkpoint.get('config')
        
        # Get model state (new format preferred, fallback to old format)
        model_state = checkpoint.get('model_state')
        if model_state is None:
            # Old format: reconstruct from individual fields
            model_state = DFMModelState(
                num_factors=checkpoint.get('num_factors'),
                r=checkpoint.get('r'),
                p=checkpoint.get('p'),
                blocks=checkpoint.get('blocks'),
                mixed_freq=checkpoint.get('_mixed_freq'),
                constraint_matrix=checkpoint.get('_constraint_matrix'),
                constraint_vector=checkpoint.get('_constraint_vector'),
                n_slower_freq=checkpoint.get('_n_slower_freq', 0),
                n_clock_freq=checkpoint.get('_n_clock_freq'),
                tent_weights_dict=checkpoint.get('_tent_weights_dict'),
                frequencies=checkpoint.get('_frequencies'),
                idio_indicator=checkpoint.get('_idio_indicator'),
                max_lag_size=checkpoint.get('_max_lag_size'),
            )
        elif isinstance(model_state, dict):
            model_state = DFMModelState(**{k: v for k, v in model_state.items() if v is not None})
        
        # Create model instance
        model = cls(
            config=config,
            num_factors=model_state.num_factors,
            threshold=checkpoint.get('threshold'),
            max_iter=checkpoint.get('max_iter'),
            nan_method=checkpoint.get('nan_method'),
            nan_k=checkpoint.get('nan_k')
        )
        
        # Apply model state (structure and mixed-frequency params)
        model_state.apply_to_model(model)
        
        # Restore state-space parameters (new format preferred)
        state_space_params = checkpoint.get('state_space_params')
        old_training_state = checkpoint.get('training_state')
        
        if state_space_params is not None:
            if isinstance(state_space_params, dict):
                state_space_params = DFMStateSpaceParams(**{k: v for k, v in state_space_params.items() if v is not None})
            model.training_state = state_space_params
            state_space_params.apply_to_model(model)
        elif old_training_state is not None:
            # Backward compatibility: extract from old format
            model.training_state = cls._extract_state_space_from_old_format(old_training_state)
            model.training_state.apply_to_model(model)
        
        # Restore training metadata
        model._training_loglik = checkpoint.get('training_loglik') or (
            getattr(old_training_state, 'loglik', None) if old_training_state is not None else None
        )
        model._training_num_iter = checkpoint.get('training_num_iter') or (
            getattr(old_training_state, 'num_iter', None) if old_training_state is not None else None
        )
        model._training_converged = checkpoint.get('training_converged') or (
            getattr(old_training_state, 'converged', None) if old_training_state is not None else None
        )
        
        # Restore data and preprocessing
        model.data_processed = checkpoint.get('data_processed')
        model.target_scaler = checkpoint.get('target_scaler')
        
        # Restore result
        if checkpoint.get('result') is not None:
            model._result = checkpoint['result']
        
        _logger.info(f"DFM model loaded from {path}")
        return model
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        super().reset()
        return self

