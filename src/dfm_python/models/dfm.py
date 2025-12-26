"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
DFM inherits from BaseFactorModel (not PyTorch Lightning) since all
calculations are performed in NumPy using pykalman.
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
from scipy.linalg import block_diag

# NumPy-based Kalman filter (pykalman) - now a required dependency
from ..ssm.kalman import DFMKalmanFilter

# Local imports
from ..config import (
    DFMConfig,
    make_config_source,
    ConfigSource,
)
from ..config.results import DFMResult
from ..config.utils import get_agg_structure, get_tent_weights, FREQUENCY_HIERARCHY, TENT_WEIGHTS_LOOKUP
from ..logger import get_logger
from .base import BaseFactorModel

# Constants
_FREQ_TO_INT = {'d': 1, 'w': 2, 'm': 3, 'q': 4, 'sa': 5, 'a': 6}
_DEFAULT_DTYPE = np.float32

# Import EM algorithm from functional module
from .functional.em import EMConfig, em_step, run_em_algorithm, _DEFAULT_EM_CONFIG as _EM_CONFIG
from .functional.block import (
    build_lag_matrix,
    initialize_block_loadings,
    initialize_block_transition
)

if TYPE_CHECKING:
    from ..datamodule import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class DFMTrainingState:
    """State tracking for DFM training."""
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    loglik: float
    num_iter: int
    converged: bool


class DFM(BaseFactorModel):
    """High-level API for Linear Dynamic Factor Model.
    
    This class implements the EM algorithm for DFM estimation using NumPy and pykalman.
    It inherits from BaseFactorModel (not PyTorch Lightning) since all calculations
    are performed in NumPy.
    
    **Note**: All calculations are performed in NumPy (using pykalman) for better
    numerical stability. Parameters are stored as NumPy arrays (no PyTorch dependencies).
    
    **Block Structure**: The model supports block-structured factors (factors organized
    in blocks). Block structure is established during initialization and is preserved
    during EM updates. pykalman handles the E-step (Kalman filter/smoother), while
    the M-step uses custom code that maintains block structure, mixed-frequency handling,
    and idiosyncratic components.
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import DFM, DFMDataModule, DFMTrainer
        >>> import pandas as pd
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pd.read_csv('data/finance.csv')
        >>> df_processed = df[[col for col in df.columns if col != 'date']]
        >>> 
        >>> # Step 2: Create DataModule
        >>> dm = DFMDataModule(config_path='config/dfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = DFM()
        >>> model.load_config('config/dfm_config.yaml')
        >>> 
        >>> # Step 4: Fit model
        >>> model.fit(X_torch, Mx=Mx, Wx=Wx)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        num_factors: Optional[int] = None,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        nan_method: int = 2,
        nan_k: int = 3,
        mixed_freq: bool = False
    ):
        """Initialize DFM instance.
        
        Parameters
        ----------
        config : DFMConfig, optional
            DFM configuration. Can be loaded later via load_config().
        num_factors : int, optional
            Number of factors. If None, inferred from config.
        threshold : float, default 1e-4
            EM convergence threshold
        max_iter : int, default 100
            Maximum EM iterations
        nan_method : int, default 2
            Missing data handling method
        nan_k : int, default 3
            Spline interpolation order
        mixed_freq : bool, default False
            If True, use tent kernels for mixed-frequency data. If False, treat all series as clock frequency.
            When True, raises ValueError if any frequency pair is not in TENT_WEIGHTS_LOOKUP.
        """
        super().__init__()
        
        # Initialize config using consolidated helper method
        config = self._initialize_config(config)
        
        self.threshold = threshold
        self.max_iter = max_iter
        self.nan_method = nan_method
        self.nan_k = nan_k
        self.mixed_freq = mixed_freq
        
        # Mixed frequency parameters (set during fit)
        self._constraint_matrix = None  # R_mat: constraint matrix for tent kernel aggregation
        self._constraint_vector = None  # q: constraint vector for tent kernel aggregation
        self._n_slower_freq = 0  # nQ: number of slower-frequency series
        self._tent_weights_dict = None
        self._frequencies = None
        self._idio_indicator = None  # i_idio: indicator for idiosyncratic components
        self._idio_chain_lengths = None
        
        # Determine number of factors
        if num_factors is None:
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                self.num_factors = int(np.sum(config.factors_per_block))
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
            dtype=np.float32
        )
        self.p = getattr(config, 'ar_lag', 1)
        self.blocks = np.array(config.get_blocks_array(), dtype=np.float32)
        
        # Use NumPy for all calculations (pykalman is now a required dependency)
        # PyKalman instance will be created when needed
        
        # Parameters stored as NumPy arrays (no PyTorch dependencies)
        self.A: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.Z_0: Optional[np.ndarray] = None
        self.V_0: Optional[np.ndarray] = None
        
        # DFM Kalman filter instance (created when needed)
        self._kalman_filter: Optional[DFMKalmanFilter] = None
        
        # Training state
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[np.ndarray] = None
    
    def _update_parameters(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
                          R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> None:
        """Update model parameters from NumPy arrays.
        
        Parameters
        ----------
        A, C, Q, R, Z_0, V_0 : np.ndarray
            Parameter arrays
        """
        def ensure_dtype(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return arr.astype(_DEFAULT_DTYPE) if arr is not None else None
        
        self.A = ensure_dtype(A)
        self.C = ensure_dtype(C)
        self.Q = ensure_dtype(Q)
        self.R = ensure_dtype(R)
        self.Z_0 = ensure_dtype(Z_0)
        self.V_0 = ensure_dtype(V_0)
    
    
    
    
    def _initialize_idiosyncratic_components(
        self,
        res: np.ndarray,
        data_with_nans: np.ndarray,
        R: np.ndarray,
        n_clock_freq: int,
        n_slower_freq: int,
        i_idio: Optional[np.ndarray],
        T: int,
        clock: str = 'm',
        dtype: type = np.float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize idiosyncratic transition matrices (clock frequency AR(1) and slower frequency tent kernel chain).
        
        Returns
        -------
        BM, SM, BQ, SQ, initViM, initViQ
        """
        # Clock frequency: AR(1) for each series
        n_idio_clock = n_clock_freq if i_idio is None else int(np.sum(i_idio))
        BM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        SM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        
        ii_idio = np.where(i_idio > 0)[0] if i_idio is not None else np.arange(n_clock_freq, dtype=np.int32)
        
        for i, idx in enumerate(ii_idio):
            res_i = data_with_nans[:, idx]
            non_nan_mask = ~np.isnan(res_i)
            if np.sum(non_nan_mask) > 1:
                first_non_nan = np.where(non_nan_mask)[0][0] if np.any(non_nan_mask) else 0
                last_non_nan = np.where(non_nan_mask)[0][-1] if np.any(non_nan_mask) else T - 1
                res_i_clean = res[first_non_nan:last_non_nan + 1, idx]
                
                # Default values (used as fallback)
                default_ar_coef = _EM_CONFIG.quarterly_ar_coef
                default_noise = _EM_CONFIG.default_process_noise
                
                if len(res_i_clean) > 1:
                    try:
                        y_ar = res_i_clean[1:]
                        x_ar = res_i_clean[:-1].reshape(-1, 1)
                        BM[i, i] = np.linalg.solve(
                            x_ar.T @ x_ar + np.eye(1, dtype=dtype) * _EM_CONFIG.matrix_regularization,
                            x_ar.T @ y_ar
                        ).item()
                        residuals_ar = y_ar - x_ar.squeeze() * BM[i, i]
                        SM[i, i] = np.var(residuals_ar, ddof=0) if len(residuals_ar) > 1 else default_noise
                    except (np.linalg.LinAlgError, ValueError):
                        BM[i, i] = default_ar_coef
                        SM[i, i] = default_noise
                else:
                    BM[i, i] = default_ar_coef
                    SM[i, i] = default_noise
            else:
                BM[i, i] = _EM_CONFIG.quarterly_ar_coef
                SM[i, i] = _EM_CONFIG.default_process_noise
        
        # Slower frequency: tent kernel size-state chain
        from .functional.block import (
            get_tent_kernel_size,
            build_quarterly_idiosyncratic_chain
        )
        rho0 = _EM_CONFIG.quarterly_ar_coef
        chain_size = get_tent_kernel_size(
            R_mat=None,  # Not available in this context
            tent_weights_dict=None,  # Not available in this context
            default_size=_EM_CONFIG.tent_kernel_size
        )
        if n_slower_freq > 0:
            sig_e = np.diag(R[n_clock_freq:, n_clock_freq:]) / _EM_CONFIG.quarterly_variance_denominator
            sig_e = np.where(np.isfinite(sig_e), sig_e, _EM_CONFIG.default_observation_noise)
            BQ, SQ, initViQ = build_quarterly_idiosyncratic_chain(
                n_slower_freq, chain_size, rho0, sig_e, dtype
            )
        else:
            empty_matrix = np.zeros((0, 0), dtype=dtype)
            BQ = SQ = initViQ = empty_matrix
        
        # Clock frequency initial covariance
        try:
            eye_BM = np.eye(n_idio_clock, dtype=dtype)
            BM_sq = BM ** 2
            diag_inv = 1.0 / np.diag(eye_BM - BM_sq)
            diag_inv = np.where(np.isfinite(diag_inv), diag_inv, np.ones_like(diag_inv))
            initViM = np.diag(diag_inv) @ SM
        except (np.linalg.LinAlgError, ValueError):
            initViM = SM.copy()
        
        return BM, SM, BQ, SQ, initViM, initViQ
    
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
        i_idio: Optional[np.ndarray] = None,
        clock: str = 'm',
        tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
        frequencies: Optional[np.ndarray] = None,
        idio_chain_lengths: Optional[np.ndarray] = None,
        config: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize DFM parameters using pure NumPy (residual-based PCA approach).
        
        Implements MATLAB InitCond() approach:
        1. Start with residuals = spline-interpolated data
        2. For each block: compute PCA on residuals, extract factors, update residuals
        3. Build transition matrices block-by-block
        4. Handle idiosyncratic components (monthly AR(1) and quarterly 5-state chains)
        
        Parameters
        ----------
        x : np.ndarray
            Standardized data matrix (T x N)
        r : np.ndarray
            Number of factors per block (n_blocks,)
        p : int
            AR lag order (typically 1)
        blocks : np.ndarray
            Block structure array (N x n_blocks)
        opt_nan : dict
            Missing data handling options {'method': int, 'k': int}
        R_mat : np.ndarray, optional
            Constraint matrix for tent kernel aggregation
        q : np.ndarray, optional
            Constraint vector for tent kernel aggregation
        n_slower_freq : int
            Number of slower-frequency series
        i_idio : np.ndarray, optional
            Indicator array (1 for clock frequency, 0 for slower frequencies)
        clock : str
            Clock frequency ('d', 'w', 'm', 'q', 'sa', 'a')
        tent_weights_dict : dict, optional
            Dictionary mapping frequency pairs to tent weights
        frequencies : np.ndarray, optional
            Array of frequencies for each series
        idio_chain_lengths : np.ndarray, optional
            Array of idiosyncratic chain lengths per series
        config : Any, optional
            Configuration object
            
        Returns
        -------
        A : np.ndarray
            Initial transition matrix (m x m)
        C : np.ndarray
            Initial observation/loading matrix (N x m)
        Q : np.ndarray
            Initial process noise covariance (m x m)
        R : np.ndarray
            Initial observation noise covariance (N x N)
        Z_0 : np.ndarray
            Initial state vector (m,)
        V_0 : np.ndarray
            Initial state covariance (m x m)
        """
        from ..utils.data import rem_nans_spline
        from ..encoder.pca import compute_principal_components
        
        T, N = x.shape
        dtype = np.float32
        
        n_blocks = blocks.shape[1]
        n_clock_freq = N - n_slower_freq  # Number of clock frequency series
        
        # Handle missing data for initialization
        x_clean, indNaN = rem_nans_spline(x, method=opt_nan.get('method', 2), k=opt_nan.get('k', 3))
        
        # Remove any remaining NaN/inf
        x_clean = np.where(np.isfinite(x_clean), x_clean, 0.0)
        
        # Initialize residuals: res = x_clean (spline-interpolated data)
        res = x_clean.copy()  # T x N
        data_with_nans = x_clean.copy()
        data_with_nans[indNaN] = np.nan
        
        # Determine tent kernel size for slower-frequency aggregation
        tent_kernel_size = _EM_CONFIG.tent_kernel_size
        if R_mat is not None:
            tent_kernel_size = R_mat.shape[1]
        elif tent_weights_dict is not None:
            # Get first slower frequency key from tent_weights_dict
            slower_freq_keys = [k for k in tent_weights_dict.keys() if k != clock]
            if slower_freq_keys:
                tent_kernel_size = len(tent_weights_dict[slower_freq_keys[0]])
        max_lag_size = max(p, tent_kernel_size)
        
        # Set first tent_kernel_size-1 observations as NaN for slower-to-clock frequency aggregation
        if tent_kernel_size > 1:
            data_with_nans[:tent_kernel_size-1, :] = np.nan
        
        # Initialize output matrices
        C_list = []  # Will concatenate block loadings
        A_list = []  # Will build block-diagonal transition matrix
        Q_list = []  # Will build block-diagonal process noise
        V_0_list = []  # Will build block-diagonal initial covariance
        
        # Process each block sequentially (residual-based approach)
        for block_idx in range(n_blocks):
            num_factors_block = int(r[block_idx])
            
            # Find series indices loading on this block
            block_series_indices = np.where(blocks[:, block_idx] > 0)[0]
            clock_freq_indices = block_series_indices[block_series_indices < n_clock_freq]
            slower_freq_indices = block_series_indices[block_series_indices >= n_clock_freq]
            
            # Initialize loadings and extract factors
            C_i, factors = initialize_block_loadings(
                res, data_with_nans, clock_freq_indices, slower_freq_indices,
                num_factors_block, tent_kernel_size, R_mat, q,
                N, max_lag_size, _EM_CONFIG.matrix_regularization, dtype
            )
            
            # Build lag matrix and update residuals
            lag_matrix = build_lag_matrix(factors, T, num_factors_block, tent_kernel_size, p, dtype)
            slower_freq_factors = lag_matrix[:, :num_factors_block * tent_kernel_size] if lag_matrix.shape[1] >= num_factors_block * tent_kernel_size else lag_matrix
            
            # Pad slower frequency factors
            if tent_kernel_size > 1:
                slower_freq_factors_padded = np.vstack([
                    np.zeros((tent_kernel_size - 1, num_factors_block * tent_kernel_size), dtype=dtype),
                    slower_freq_factors[:T - (tent_kernel_size - 1), :num_factors_block * tent_kernel_size] if T > (tent_kernel_size - 1) else slower_freq_factors[:, :num_factors_block * tent_kernel_size]
                ])
                if len(slower_freq_factors_padded) < T:
                    slower_freq_factors_padded = np.vstack([
                        slower_freq_factors_padded,
                        np.zeros((T - len(slower_freq_factors_padded), num_factors_block * tent_kernel_size), dtype=dtype)
                    ])
                slower_freq_factors = slower_freq_factors_padded[:T, :]
            
            # Update residuals
            if res.shape[0] != slower_freq_factors.shape[0]:
                slower_freq_factors = slower_freq_factors[:res.shape[0], :] if res.shape[0] < slower_freq_factors.shape[0] else np.vstack([
                    slower_freq_factors, np.zeros((res.shape[0] - slower_freq_factors.shape[0], slower_freq_factors.shape[1]), dtype=dtype)
                ])
            res = res - slower_freq_factors @ C_i[:, :num_factors_block * tent_kernel_size].T
            data_with_nans = res.copy()
            data_with_nans[indNaN] = np.nan
            
            C_list.append(C_i)
            
            # Initialize transition matrices for this block
            A_i, Q_i, V_0_i = initialize_block_transition(
                lag_matrix, factors, num_factors_block, max_lag_size, p, T,
                _EM_CONFIG.regularization, _EM_CONFIG.default_transition_coef,
                _EM_CONFIG.default_process_noise, _EM_CONFIG.matrix_regularization,
                _EM_CONFIG.eigenval_floor, dtype
            )
            
            A_list.append(A_i)
            Q_list.append(Q_i)
            V_0_list.append(V_0_i)
        
        # Concatenate C matrices
        C = np.hstack(C_list) if C_list else np.zeros((N, 0), dtype=dtype)
        
        # Build block-diagonal A, Q, V_0
        if A_list:
            try:
                A_factors = block_diag(*A_list)
                Q_factors = block_diag(*Q_list)
                V_0_factors = block_diag(*V_0_list)
            except (ValueError, np.linalg.LinAlgError) as e:
                _logger.error(f"block_diag failed: {e}")
                raise
        else:
            empty_matrix = np.zeros((0, 0), dtype=dtype)
            A_factors = Q_factors = V_0_factors = empty_matrix
        
        # === IDIOSYNCRATIC COMPONENTS ===
        # Add identity matrix for clock frequency idiosyncratic series
        if i_idio is not None:
            eyeN = np.eye(N, dtype=dtype)
            i_idio_bool = i_idio.astype(bool)
            eyeN_idio = eyeN[:, i_idio_bool]  # N x n_idio
            C = np.hstack([C, eyeN_idio])
        else:
            # Default: all clock frequency series have idiosyncratic components
            eyeN_clock_freq = np.eye(N, dtype=dtype)[:, :n_clock_freq] if n_clock_freq > 0 else np.zeros((N, 0), dtype=dtype)
            C = np.hstack([C, eyeN_clock_freq])
        
        # Add slower frequency idiosyncratic chains (tent_kernel_size-state chain)
        if n_slower_freq > 0:
            from .functional.block import (
                get_tent_kernel_size,
                get_quarterly_tent_weights,
                build_quarterly_observation_matrix
            )
            tent_kernel_size = get_tent_kernel_size(
                R_mat=R_mat,
                tent_weights_dict=tent_weights_dict,
                default_size=_EM_CONFIG.tent_kernel_size
            )
            tent_weights = get_quarterly_tent_weights(clock, tent_kernel_size, dtype)
            C_slower_freq = build_quarterly_observation_matrix(N, n_clock_freq, n_slower_freq, tent_weights, dtype)
            C = np.hstack([C, C_slower_freq])
        
        # Initialize R (observation noise covariance) from final residuals
        if data_with_nans.ndim != 2:
            data_with_nans = data_with_nans.reshape(-1, data_with_nans.shape[-1]) if data_with_nans.ndim > 2 else data_with_nans.reshape(1, -1)
        
        T_res, N_res = data_with_nans.shape
        default_obs_noise = _EM_CONFIG.default_observation_noise
        
        if T_res <= 1:
            var_res = np.full(N_res, default_obs_noise, dtype=dtype)
        else:
            var_res = np.array([
                np.var(data_with_nans[:, i][np.isfinite(data_with_nans[:, i])], ddof=0) 
                if np.sum(np.isfinite(data_with_nans[:, i])) > 1 else default_obs_noise
                for i in range(N_res)
            ], dtype=dtype)
            var_res = np.where(np.isfinite(var_res), var_res, default_obs_noise)
        
        R = np.diag(var_res)
        R = np.where(np.isfinite(R), R, default_obs_noise)
        
        # Set all variances to default observation noise
        default_obs_noise = _EM_CONFIG.default_observation_noise
        i_idio_indices = np.where(i_idio > 0)[0] if i_idio is not None else np.arange(n_clock_freq, dtype=np.int32)
        all_indices = np.unique(np.concatenate([i_idio_indices, np.arange(n_clock_freq, N, dtype=np.int32)]))
        for idx in all_indices:
            R[idx, idx] = default_obs_noise
        
        # === IDIOSYNCRATIC TRANSITION MATRICES ===
        BM, SM, BQ, SQ, initViM, initViQ = self._initialize_idiosyncratic_components(
            res, data_with_nans, R, n_clock_freq, n_slower_freq, i_idio, T, clock=clock, dtype=dtype
        )
        
        # Combine all transition matrices
        try:
            A = block_diag(A_factors, BM, BQ)
            Q = block_diag(Q_factors, SM, SQ)
            V_0 = block_diag(V_0_factors, initViM, initViQ)
        except (ValueError, np.linalg.LinAlgError) as e:
            _logger.error(f"block_diag failed: {e}")
            raise
        
        # Initial state: Z_0 = zeros
        m = int(A.shape[0]) if A.size > 0 and len(A.shape) > 0 else 0
        Z_0 = np.zeros(m, dtype=dtype)
        
        # Ensure V_0 is positive definite
        from .functional.em import ensure_covariance_stable
        V_0 = ensure_covariance_stable(V_0, min_eigenval=_EM_CONFIG.eigenval_floor)
        
        return A.astype(np.float32), C.astype(np.float32), Q.astype(np.float32), R.astype(np.float32), Z_0.astype(np.float32), V_0.astype(np.float32)
    
    def fit(
        self,
        X: Union[np.ndarray, Any],
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None,
        datamodule: Optional[Any] = None
    ) -> DFMTrainingState:
        """Fit model using EM algorithm (wrapper around pykalman).
        
        Uses pykalman for E-step (Kalman filter/smoother) and custom M-step
        that preserves block structure and mixed-frequency constraints.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor, optional
            Standardized data (T x N). If datamodule is provided, X can be None.
        Mx : np.ndarray, optional
            Mean values for unstandardization (N,). If datamodule is provided, Mx can be None.
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization (N,). If datamodule is provided, Wx can be None.
        datamodule : DFMDataModule, optional
            Custom DFMDataModule instance. If provided, initialization parameters will be
            extracted from the datamodule instead of computing them directly.
            
        Returns
        -------
        DFMTrainingState
            Final training state with parameters and convergence info
        """
        # Use datamodule if provided
        if datamodule is not None:
            init_params = datamodule.get_initialization_params()
            X_np = init_params['X']
            Mx = init_params['Mx'] if Mx is None else Mx
            Wx = init_params['Wx'] if Wx is None else Wx
            R_mat = init_params['R_mat']
            q = init_params['q']
            n_slower_freq = init_params['n_slower_freq']
            tent_weights_dict = init_params['tent_weights_dict']
            frequencies_np = init_params['frequencies']
            i_idio = init_params['i_idio']
            idio_chain_lengths = init_params['idio_chain_lengths']
            opt_nan = init_params['opt_nan']
            clock = init_params['clock']
        else:
            # Convert to NumPy
            if hasattr(X, 'cpu') and hasattr(X, 'numpy') and not isinstance(X, np.ndarray):
                X_np = X.cpu().numpy()
            else:
                X_np = np.asarray(X, dtype=np.float32)
            
            # Setup mixed-frequency parameters (fallback if no datamodule)
            from ..config.utils import get_agg_structure, get_tent_weights, FREQUENCY_HIERARCHY, TENT_WEIGHTS_LOOKUP
            clock = getattr(self.config, 'clock', 'm')
            
            if not self.mixed_freq:
                R_mat = None
                q = None
                n_slower_freq = 0
                tent_weights_dict = None
                frequencies_np = None
                i_idio = np.ones(X_np.shape[1], dtype=np.float32)
                idio_chain_lengths = np.zeros(X_np.shape[1], dtype=np.int32)
            else:
                agg_structure = get_agg_structure(self.config, clock=clock)
                frequencies_list = [s.frequency for s in self.config.series]
                frequencies_set = set(frequencies_list)
                clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                
                missing_pairs = [
                    (freq, clock) for freq in frequencies_set
                    if FREQUENCY_HIERARCHY.get(freq, 3) > clock_hierarchy and get_tent_weights(freq, clock) is None
                ]
                if missing_pairs:
                    raise ValueError(
                        f"mixed_freq=True but the following frequency pairs are not in TENT_WEIGHTS_LOOKUP: {missing_pairs}. "
                        f"Available pairs: {list(TENT_WEIGHTS_LOOKUP.keys())}. "
                        f"Either add the missing pairs to TENT_WEIGHTS_LOOKUP or set mixed_freq=False."
                    )
                
                tent_weights_dict = {k: np.array(v, dtype=np.float32) for k, v in agg_structure['tent_weights'].items()}
                
                R_mat = None
                q = None
                if agg_structure['structures']:
                    first_structure = list(agg_structure['structures'].values())[0]
                    R_mat = np.array(first_structure[0], dtype=np.float32)
                    q = np.array(first_structure[1], dtype=np.float32)
                
                n_slower_freq = sum(1 for freq in frequencies_list if FREQUENCY_HIERARCHY.get(freq, 3) > clock_hierarchy)
                i_idio = np.array([1 if freq == clock else 0 for freq in frequencies_list], dtype=np.float32)
                frequencies_np = np.array([_FREQ_TO_INT.get(f, 3) for f in frequencies_list], dtype=np.int32)
                idio_chain_lengths = np.zeros(X_np.shape[1], dtype=np.int32)
            
            opt_nan = {'method': self.nan_method, 'k': self.nan_k}
        
        self.Mx = Mx
        self.Wx = Wx
        self.data_processed = X_np
        
        # Store for reuse in EM steps
        self._constraint_matrix = R_mat
        self._constraint_vector = q
        self._n_slower_freq = n_slower_freq
        self._tent_weights_dict = tent_weights_dict
        self._frequencies = frequencies_np
        self._i_idio = i_idio
        self._idio_chain_lengths = idio_chain_lengths
        
        # Initialize parameters
        A_np, C_np, Q_np, R_np, Z_0_np, V_0_np = self._initialize_parameters(
            X_np, self.r, self.p, self.blocks, opt_nan, R_mat, q, n_slower_freq, i_idio,
            clock, tent_weights_dict, frequencies_np, idio_chain_lengths, self.config
        )
        self._update_parameters(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        # Run EM algorithm (pykalman E-step + custom M-step)
        max_iter_val = self.max_iter if self.max_iter is not None else 100
        threshold_val = self.threshold if self.threshold is not None else 1e-4
        
        if self.A is None or self.C is None or self.Q is None or self.R is None or self.Z_0 is None or self.V_0 is None:
            raise RuntimeError("DFM fit failed: parameters not initialized.")
        
        initial_params: Dict[str, np.ndarray] = {
            'A': self.A, 'C': self.C, 'Q': self.Q,
            'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
        }
        
        def get_params_fn() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            from typing import cast
            return cast(Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       (self.A, self.C, self.Q, self.R, self.Z_0, self.V_0))
        
        final_state = run_em_algorithm(
            X=X_np,
            initial_params=initial_params,
            update_params_fn=self._update_parameters,
            get_params_fn=get_params_fn,
            max_iter=max_iter_val,
            threshold=threshold_val,
            config=_EM_CONFIG
        )
        
        self.training_state = DFMTrainingState(
            A=final_state['A'], C=final_state['C'], Q=final_state['Q'],
            R=final_state['R'], Z_0=final_state['Z_0'], V_0=final_state['V_0'],
            loglik=final_state['loglik'], num_iter=final_state['num_iter'],
            converged=final_state['converged']
        )
        
        return self.training_state
    
    def _compute_smoothed_factors(self) -> np.ndarray:
        """Compute smoothed factors using Kalman filter.
        
        Returns
        -------
        np.ndarray
            Smoothed factors (T x m)
        """
        if self.training_state is None or self.data_processed is None:
            raise RuntimeError("Model not fitted or data not available")
        
        kalman_final = DFMKalmanFilter(
            transition_matrices=self.training_state.A,
            observation_matrices=self.training_state.C,
            transition_covariance=self.training_state.Q,
            observation_covariance=self.training_state.R,
            initial_state_mean=self.training_state.Z_0,
            initial_state_covariance=self.training_state.V_0
        )
        
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
        if self.training_state is None:
            raise RuntimeError("DFM get_result failed: model has not been fitted yet. Please call fit() first.")
        if self.data_processed is None:
            raise RuntimeError("DFM get_result failed: data not available. Please ensure fit() was called with data.")
        
        # Compute smoothed factors
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
        Wx_clean = np.where(np.isnan(self.Wx), 1.0, self.Wx) if self.Wx is not None else np.ones(C.shape[0])
        Mx_clean = np.where(np.isnan(self.Mx), 0.0, self.Mx) if self.Mx is not None else np.zeros(C.shape[0])
        X_sm = x_sm * Wx_clean + Mx_clean
        
        return DFMResult(
            x_sm=x_sm, X_sm=X_sm, Z=Z, C=C, R=R, A=A, Q=Q,
            Mx=self.Mx if self.Mx is not None else np.zeros(C.shape[0]),
            Wx=self.Wx if self.Wx is not None else np.ones(C.shape[0]),
            Z_0=Z_0, V_0=V_0, r=self.r, p=self.p,
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=self.training_state.loglik,
            series_ids=self.config.get_series_ids() if hasattr(self.config, 'get_series_ids') else None,
            block_names=getattr(self.config, 'block_names', None)
        )
    
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> 'DFM':
        """Load configuration from various sources.
        
        After loading config, the model needs to be re-initialized with the new config.
        For standard Lightning pattern, pass config directly to __init__.
        """
        # Use common config loading logic
        new_config = self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        # DFM-specific: Initialize r and blocks arrays
        self.r = np.array(
            new_config.factors_per_block if new_config.factors_per_block is not None
            else np.ones(new_config.get_blocks_array().shape[1]),
            dtype=np.float32
        )
        self.blocks = np.array(new_config.get_blocks_array(), dtype=np.float32)
        
        return self
    
    
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        history: Optional[int] = None,
        return_series: bool = True,
        return_factors: bool = True,
        target: Optional[List[str]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the Lightning module to generate forecasts.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        history : int, optional
            Number of historical periods to use for Kalman filter update before prediction.
            If None, uses full history (default). If specified (e.g., 60), uses only the most
            recent N periods for efficiency. Initial state (Z_0, V_0) is always estimated from
            full history (including any new data beyond training period).
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
        target : List[str], optional
            List of target series IDs to return. If None, uses target_series from DataModule.
            If DataModule has no target_series, raises ValueError.
            If specified, only returns predictions for the specified target series.
            Only target series are returned (features are excluded).
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x N)
            If only return_factors is True:
                Z_forecast (horizon x m)
            
        Notes
        -----
        When history is specified, the method uses only the most recent N periods for
        Kalman filter update, improving computational efficiency. The initial state
        (Z_0, V_0) is always estimated from full history (including any new data beyond
        training period), ensuring accuracy while maintaining efficiency.
        """
        if self.training_state is None:
            raise ValueError(
                f"{self.__class__.__name__} prediction failed: model has not been trained yet. "
                f"Please call trainer.fit(model, data_module) first"
            )
        
        # Get result (only call get_result() if _result is None)
        if not hasattr(self, '_result') or self._result is None:
            self._result = self.get_result()
        
        result = self._result
        
        if not hasattr(result, 'Z') or result.Z is None:
            raise ValueError(
                "DFM prediction failed: result.Z is not available. "
                "This may indicate the model was not properly trained or result object is corrupted."
            )
        
        # Compute default horizon
        if horizon is None:
            from ..config.utils import get_periods_per_year
            from ..utils.helpers import get_clock_frequency
            clock = get_clock_frequency(self.config, 'm')
            horizon = get_periods_per_year(clock)
        
        # Validate horizon
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        
        # Extract model parameters
        A = result.A
        C = result.C
        Wx = result.Wx
        Mx = result.Mx
        p = getattr(result, 'p', 1)  # VAR order, default to 1 for DFM
        
        # Use training state for initial factor state
        # For DFM, we use the last smoothed state from training
        # History-based updates can be added later if needed
        Z_last = result.Z[-1, :] if result.Z.shape[0] > 0 else np.zeros(result.A.shape[0], dtype=np.float32)
        
        # Validate factor state
        if np.any(np.isnan(Z_last)):
            nan_count = np.sum(np.isnan(Z_last))
            nan_ratio = nan_count / len(Z_last)
            raise ValueError(
                f"DFM prediction failed: {nan_count}/{len(Z_last)} factors contain NaN ({nan_ratio:.1%}). "
                f"Model may not have converged. Try increasing max_iter or checking data quality."
            )
        
        # Validate parameters are finite
        if np.any(~np.isfinite(A)) or np.any(~np.isfinite(C)):
            raise ValueError(
                "DFM prediction failed: model parameters (A or C) contain NaN/Inf. "
                "Check training convergence and data quality."
            )
        
        # Resolve target indices
        target_indices = self._resolve_target_indices(target, result)  # type: ignore[arg-type]
        
        # Forecast factors using VAR dynamics (common helper)
        Z_prev = result.Z[-2, :] if result.Z.shape[0] >= 2 and p == 2 else None
        Z_forecast = self._forecast_var_factors(
            Z_last=Z_last,
            A=A,
            p=p,
            horizon=horizon,
            Z_prev=Z_prev
        )
        
        # Optimized: Transform only target series (not all series)
        # Use only target indices for C, Mx, Wx
        C_target = C[target_indices, :]  # (len(target) x m)
        max_target_idx = max(target_indices) if target_indices and len(target_indices) > 0 else 0
        Mx_target = Mx[target_indices] if Mx is not None and len(Mx) > max_target_idx else (Mx if Mx is not None else None)
        Wx_target = Wx[target_indices] if Wx is not None and len(Wx) > max_target_idx else (Wx if Wx is not None else None)
        
        # Transform factors to target observations only
        X_forecast_std = Z_forecast @ C_target.T  # (horizon x len(target))
        X_forecast = X_forecast_std * Wx_target + Mx_target  # (horizon x len(target))
        
        # Validate forecast results are finite
        if np.any(~np.isfinite(X_forecast)):
            nan_count = np.sum(~np.isfinite(X_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in forecast. "
                f"Possible numerical instability. "
                f"Please check model parameters and data quality."
            )
        
        # Ensure X_forecast is numpy array (handles torch inputs if present)
        if hasattr(X_forecast, "cpu") and hasattr(X_forecast, "numpy") and not isinstance(X_forecast, np.ndarray):
            X_forecast = X_forecast.cpu().numpy()
        X_forecast = np.asarray(X_forecast, dtype=np.float32)
        
        # Validate forecast values are within reasonable bounds (only for target series now)
        if Wx_target is not None and Mx_target is not None and len(Wx_target) > 0 and len(Mx_target) > 0:
            # Check each target series individually
            extreme_threshold_std = _EM_CONFIG.extreme_forecast_threshold
            for i in range(X_forecast.shape[1] if X_forecast.ndim > 1 else 1):
                if i < len(Wx_target) and i < len(Mx_target) and Wx_target[i] > 0:
                    series_forecast = X_forecast[:, i] if X_forecast.ndim > 1 else X_forecast
                    series_mean = Mx_target[i]
                    series_std = Wx_target[i]
                    # Calculate how many standard deviations each forecast is from the mean
                    abs_deviations = np.abs(series_forecast - series_mean) / series_std
                    max_deviation = np.max(abs_deviations) if len(abs_deviations) > 0 else 0.0
                    if max_deviation > extreme_threshold_std:
                        extreme_count = np.sum(abs_deviations > extreme_threshold_std)
                        _logger.warning(
                            f"DFM prediction: Extreme forecast for target series {i} "
                            f"(max deviation: {max_deviation:.1f} std devs). "
                            f"Possible numerical instability."
                        )
        
        if return_factors and np.any(~np.isfinite(Z_forecast)):
            nan_count = np.sum(~np.isfinite(Z_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in factor forecast. "
                f"Possible numerical instability in factor dynamics. "
                f"Please check model parameters and training convergence."
            )
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    def _update_factor_state_dfm(
        self,
        X_recent_std: np.ndarray,
        result: DFMResult,
        kalman_filter: Optional[Any] = None
    ) -> np.ndarray:
        """Update factor state using pykalman.
        
        Overrides base class method to use pykalman instead of PyTorch KalmanFilter.
        
        Parameters
        ----------
        X_recent_std : np.ndarray
            Standardized recent data (T x N)
        result : DFMResult
            Model result containing parameters
        kalman_filter : Any, optional
            Ignored (kept for compatibility with base class signature)
            
        Returns
        -------
        np.ndarray
            Updated last factor state (m,)
        """
        # Get parameters from result
        A = result.A
        C = result.C
        Q = result.Q
        R = result.R
        Z_0 = result.Z_0
        V_0 = result.V_0
        
        # Handle missing data
        X_masked = np.ma.masked_invalid(X_recent_std)
        
        # Create DFM Kalman filter
        kalman_update = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z_0,
            initial_state_covariance=V_0
        )
        
        # Run Kalman smoother (expects T x N)
        smoothed_state_means, _ = kalman_update.smooth(X_masked)
        
        # Return last smoothed state
        return smoothed_state_means[-1, :]  # (m,)
    
    def update(
        self,
        X_std: np.ndarray,
        *,
        history: Optional[int] = None,
        kalman_filter: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> 'DFM':
        """Update factor state with standardized data.
        
        This method permanently updates the last factor state (result.Z[-1, :])
        using the provided standardized data. Users should handle all preprocessing
        (masking, imputation, standardization) before calling this method.
        
        Parameters
        ----------
        X_std : np.ndarray
            Standardized data array (T x N), where T is number of time periods
            and N is number of series. Data should already be standardized using
            result.Mx and result.Wx.
        history : int, optional
            Number of recent periods to use for factor state update. If None, uses
            all provided data (default). If specified (e.g., 60), uses only the most
            recent N periods. Initial state (Z_0, V_0) is always estimated from
            full training data, but the update uses only recent history for efficiency.
        kalman_filter : Any, optional
            Kalman filter instance. If None, uses default or model's kalman filter.
            
        Returns
        -------
        DFM
            Self for method chaining
            
        Examples
        --------
        >>> # Update state with new data, then predict
        >>> model.update(X_std).predict(horizon=1)
        >>> # Or update with only recent 12 periods
        >>> model.update(X_std, history=12)
        >>> forecast = model.predict(horizon=6)
        """
        self._check_trained()
        
        # Optionally replace scaler (e.g., if refit on new regime)
        if scaler is not None:
            self.scaler = scaler
        
        result = self.result  # Use property which ensures non-None after _check_trained()
        
        # Validate input shape
        if not isinstance(X_std, np.ndarray):
            X_std = np.asarray(X_std)
        if X_std.ndim != 2:
            raise ValueError(
                f"DFM update(): X_std must be 2D array (T x N), "
                f"got shape {X_std.shape}"
            )
        
        # Handle NaN/Inf values
        X_std = np.where(np.isfinite(X_std), X_std, np.nan)
        
        # Filter to recent history if specified
        # Note: Initial state (Z_0, V_0) from result is estimated from full training data,
        # but we use only recent history for the update step
        if history is not None and history > 0:
            if X_std.shape[0] > history:
                X_recent = X_std[-history:, :]
                _logger.debug(
                    f"DFM update(): Using {history} most recent periods out of {X_std.shape[0]} total periods"
                )
            else:
                X_recent = X_std
                _logger.debug(
                    f"DFM update(): history={history} specified but data has only {X_std.shape[0]} periods, using all data"
                )
        else:
            X_recent = X_std
        
        # Update factor state using Kalman filter directly on standardized data
        Z_last_updated = self._update_factor_state_dfm(
            X_recent, result, kalman_filter=None  # Use pykalman in override
        )
        
        # Update result.Z[-1, :] permanently
        if Z_last_updated is not None:
            result.Z[-1, :] = Z_last_updated
        else:
            _logger.warning(
                f"DFM update(): Failed to update factor state, "
                f"keeping current state"
            )
            
        return self
    
    def _resolve_target_indices(self, target: Optional[List[str]], result: Any) -> List[int]:
        """Resolve target series names to indices.
        
        Returns
        -------
        List[int]
            Target series indices
        """
        if target is None:
            try:
                data_module = self._get_datamodule()
                target = getattr(data_module, 'target_series', None)
            except (ValueError, AttributeError):
                target = None
        
        if target is None or len(target) == 0:
            raise ValueError(
                "DFM prediction failed: target is None but no target_series found in DataModule. "
                "Please specify target=['series_id'] or ensure DataModule has target_series set."
            )
        
        series_ids = self._config.get_series_ids() if self._config is not None else getattr(result, 'series_ids', None)
        if series_ids is None:
            raise ValueError(
                "DFM prediction failed: target specified but cannot determine series IDs. "
                "Please ensure config is loaded or result contains series_ids."
            )
        
        target_indices = []
        for tgt_id in target:
            if tgt_id in series_ids:
                target_indices.append(series_ids.index(tgt_id))
            else:
                _logger.warning(f"DFM prediction: target series '{tgt_id}' not found in series_ids. Available: {series_ids}")
        
        if len(target_indices) == 0:
            raise ValueError(f"DFM prediction failed: none of the specified target series found. Target: {target}, Available: {series_ids}")
        
        return target_indices
    
    def _validate_forecast(
        self,
        X_forecast: np.ndarray,
        Z_forecast: np.ndarray,
        Wx_target: Optional[np.ndarray],
        Mx_target: Optional[np.ndarray],
        return_factors: bool
    ) -> np.ndarray:
        """Validate forecast results and convert to NumPy array.
        
        Returns
        -------
        np.ndarray
            Validated and converted forecast
        """
        if np.any(~np.isfinite(X_forecast)):
            nan_count = np.sum(~np.isfinite(X_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in forecast. "
                f"Possible numerical instability. Please check model parameters and data quality."
            )
        
        if hasattr(X_forecast, "cpu") and hasattr(X_forecast, "numpy") and not isinstance(X_forecast, np.ndarray):
            X_forecast = X_forecast.cpu().numpy()
        X_forecast = np.asarray(X_forecast, dtype=np.float32)
        
        # Check for extreme values
        if Wx_target is not None and Mx_target is not None and len(Wx_target) > 0 and len(Mx_target) > 0:
            extreme_threshold = _EM_CONFIG.extreme_forecast_threshold
            for i in range(X_forecast.shape[1] if X_forecast.ndim > 1 else 1):
                if i < len(Wx_target) and i < len(Mx_target) and Wx_target[i] > 0:
                    series_forecast = X_forecast[:, i] if X_forecast.ndim > 1 else X_forecast
                    abs_deviations = np.abs(series_forecast - Mx_target[i]) / Wx_target[i]
                    max_deviation = np.max(abs_deviations) if len(abs_deviations) > 0 else 0.0
                    if max_deviation > extreme_threshold:
                        _logger.warning(
                            f"DFM prediction: Extreme forecast for target series {i} "
                            f"(max deviation: {max_deviation:.1f} std devs). Possible numerical instability."
                        )
        
        if return_factors and np.any(~np.isfinite(Z_forecast)):
            nan_count = np.sum(~np.isfinite(Z_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in factor forecast. "
                f"Possible numerical instability in factor dynamics. "
                f"Please check model parameters and training convergence."
            )
        
        return X_forecast
    
    @property
    def result(self) -> DFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        # Check if trained and extract result from training state if needed
        self._check_trained()
        if self._result is None:
            # Generate result from training state if not already computed
            self._result = self.get_result()
        if not isinstance(self._result, DFMResult):
            raise RuntimeError(f"Expected DFMResult but got {type(self._result)}")
        return self._result
    
    
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        super().reset()
        return self

