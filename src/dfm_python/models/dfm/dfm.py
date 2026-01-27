"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
DFM inherits from BaseFactorModel since all calculations are performed in NumPy using pykalman.
"""

# Standard library imports
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# NumPy-based Kalman filter (pykalman) - now a required dependency
from ...ssm.kalman import DFMKalmanFilter

# Local imports
from ...config import (
    DFMConfig,
    ConfigSource,
    DFMResult,
)
from ...config.schema.params import DFMModelState
# get_agg_structure and get_slower_freq_tent_weights are now used in dfm.mixed_frequency and dfm.initialization modules
from ...config.constants import (
    DEFAULT_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_ZERO_VALUE,
    FREQUENCY_HIERARCHY,
    DEFAULT_HIERARCHY_VALUE,
    MAX_STD_UNSCALED,
    MIN_STD_SCALED,
)
from ...logger import get_logger
from ..base import BaseFactorModel
from ...dataset.dfm_dataset import DFMDataset
from ...utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    ConfigurationError,
    DataError,
    DataValidationError,
    PredictionError,
    NumericalError
)
from ...utils.validation import check_condition, has_shape_with_min_dims
from ...numeric.validator import validate_dfm_initialization

# Import EM config from functional module
from ...functional.em import _DEFAULT_EM_CONFIG as _EM_CONFIG

# Import initialization and mixed-frequency functions from dfm submodule
from .initialization import (
    initialize_parameters,
)
from .mixed_freq import (
    setup_mixed_frequency_params,
)
from ...logger.dfm_logger import log_blocks_diagnostics
from ...numeric.builder import build_dfm_structure, build_dfm_blocks
from ...numeric.validator import validate_column_order

_logger = get_logger(__name__)


class DFM(BaseFactorModel):
    """Linear Dynamic Factor Model using EM algorithm with NumPy and pykalman."""
    
    def __init__(
        self,
        dataset: DFMDataset,
        config: Optional[DFMConfig] = None,
        scaler: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Initialize DFM instance.
        
        At initialization, variables and time_index are extracted from DFMDataset:
        - `dataset.variables`: DataFrame with numeric columns (used for training)
        - `dataset.time_index`: TimeIndex object (used for time alignment)
        
        Parameters
        ----------
        dataset : DFMDataset
            DFMDataset instance (required). Variables and time_index are extracted during fit().
        config : DFMConfig, optional
            DFM configuration. If None, config will be created from kwargs using DFMConfig.from_dict().
            All DFMConfig parameters can be passed directly as kwargs when config=None.
        scaler : StandardScaler, RobustScaler, MinMaxScaler, or None, optional
            Sklearn scaler instance for data standardization. If provided, it's used directly via isinstance() check.
            If None, a scaler will be created during fit() based on config.scaler or defaults to StandardScaler.
            If provided, it will be fitted during fit() if not already fitted.
        **kwargs : Any
            If config is None: All kwargs are used to create DFMConfig via DFMConfig.from_dict(kwargs).
            If config is provided: kwargs are used as parameter overrides (keys should match DFMConfig attribute names).
            Example: `DFM(dataset=ds, blocks={...}, threshold=1e-6)` creates config from kwargs.
            Example: `DFM(dataset=ds, config=cfg, threshold=1e-6)` overrides config.threshold.
            Precedence: kwargs > config > defaults.
            
        Returns
        -------
        None
            Initializes DFM instance in-place.
            
        Raises
        ------
        ConfigurationError
            If config validation fails or required parameters are missing.
        ValueError
            If mixed_freq=True and tent_weights are not specified in config.
            
        Examples
        --------
        >>> # Create config from kwargs
        >>> model = DFM(
        ...     dataset=my_dataset,
        ...     blocks={"global": {"num_factors": 1, "series": ["GDP", "CPI"]}},
        ...     threshold=1e-6,
        ...     max_iter=100
        ... )
        >>> 
        >>> # Or pass existing config
        >>> model = DFM(
        ...     dataset=my_dataset,
        ...     config=my_config,
        ...     threshold=1e-6  # Override config.threshold
        ... )
        """
        super().__init__()
        
        # Dataset is required
        if not isinstance(dataset, DFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DFMDataset, got {type(dataset).__name__}"
            )
        self._dataset = dataset
        
        # Create config from kwargs if not provided
        if config is None:
            if not kwargs:
                raise ConfigurationError(
                    "DFM: Either config or kwargs must be provided",
                    details="Provide a DFMConfig object, or pass all config parameters as kwargs (e.g., blocks, threshold, max_iter, etc.)"
                )
            # Remove dataset and scaler from kwargs (they're not part of DFMConfig)
            config_kwargs = {k: v for k, v in kwargs.items() if k not in ('dataset', 'scaler')}
            if not config_kwargs:
                raise ConfigurationError(
                    "DFM: No config parameters provided in kwargs",
                    details="Provide DFMConfig parameters like blocks, threshold, max_iter, etc."
                )
            # Create config from kwargs
            self._config = DFMConfig.from_dict(config_kwargs)
            # Clear kwargs since they were used to create config
            kwargs = {}
        else:
            self._config = config
        
        # Apply kwargs overrides to config (if any)
        # This allows runtime parameter overrides while keeping config as source of truth
        if kwargs:
            for key, value in kwargs.items():
                # Skip dataset and scaler (already handled)
                if key in ('dataset', 'scaler'):
                    continue
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                    _logger.debug(f"DFM: Overrode config.{key} with {value}")
                else:
                    _logger.warning(f"DFM: Unknown parameter '{key}' in kwargs, ignoring")
        
        # Mixed frequency: auto-detected from Dataset or config during fit()
        self._mixed_freq: Optional[bool] = None  # Internal property, auto-detected
        
        # Cache for smoothed factors from last EM iteration (avoids recomputation during save())
        self._cached_smoothed_factors: Optional[np.ndarray] = None
        
        # Mixed frequency parameters (set during fit)
        self._constraint_matrix = None  # R_mat: constraint matrix for tent kernel aggregation
        self._constraint_vector = None  # q: constraint vector for tent kernel aggregation
        self._n_slower_freq = 0  # Number of slower-frequency series
        self._tent_weights_dict = None
        self._frequencies = None
        self._idio_indicator = None  # i_idio: indicator for idiosyncratic components
        
        # Build model structure from config
        # Allow "minimal" configs that omit frequency by late-binding to dataset columns.
        # If config.frequency is missing, get_blocks_array(columns=...) will default all series to clock frequency.
        dataset_columns = list(self._dataset.variables.columns) if self._dataset.variables is not None else None
        self.blocks, self.r, self.num_factors, self.p = build_dfm_structure(self._config, columns=dataset_columns)
        
        # Create empty training state with structure parameters only
        # State-space parameters (A, C, Q, R, Z_0, V_0) will be set during fit()
        self.training_state = DFMModelState(
            num_factors=self.num_factors,
            r=self.r,
            p=self.p,
            blocks=self.blocks,
            mixed_freq=None,  # Set during fit()
            constraint_matrix=None,
            constraint_vector=None,
            n_slower_freq=0,
            n_clock_freq=None,
            tent_weights_dict=None,
            frequencies=None,
            idio_indicator=None,
            max_lag_size=None,
            A=None,
            C=None,
            Q=None,
            R=None,
            Z_0=None,
            V_0=None
        )
        
        # Training state
        self.data_processed: Optional[np.ndarray] = None
        self.scaler: Optional[Any] = scaler  # Sklearn scaler for data standardization and inverse transformation
    
    @property
    def A(self) -> Optional[np.ndarray]:
        """Transition matrix (m x m) - VAR dynamics for factors."""
        return self.training_state.A if self.training_state else None
    
    @property
    def C(self) -> Optional[np.ndarray]:
        """Observation matrix (N x m) - factor loadings."""
        return self.training_state.C if self.training_state else None
    
    @property
    def Q(self) -> Optional[np.ndarray]:
        """Process noise covariance (m x m) - innovation covariance."""
        return self.training_state.Q if self.training_state else None
    
    @property
    def R(self) -> Optional[np.ndarray]:
        """Observation noise covariance (N x N) - typically diagonal."""
        return self.training_state.R if self.training_state else None
    
    @property
    def Z_0(self) -> Optional[np.ndarray]:
        """Initial state mean (m,) - initial factor values."""
        return self.training_state.Z_0 if self.training_state else None
    
    @property
    def V_0(self) -> Optional[np.ndarray]:
        """Initial state covariance (m x m) - initial uncertainty."""
        return self.training_state.V_0 if self.training_state else None
    
    def _check_parameters_initialized(self) -> None:
        """Check if model parameters are initialized (required for prediction).
        
        Raises
        ------
        ModelNotInitializedError
            If parameters are not initialized
        """
        from ...numeric.validator import validate_parameters_initialized
        validate_parameters_initialized(
            {
                'A': self.A, 'C': self.C, 'Q': self.Q,
                'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
            },
            model_name=self.__class__.__name__
        )
    
    def _update_parameters(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
                          R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> None:
        """Update model parameters from NumPy arrays.
        
        Parameters
        ----------
        A, C, Q, R, Z_0, V_0 : np.ndarray
            Parameter arrays
        """
        # Update training_state directly (single source of truth)
        self.training_state.A = np.asarray(A, dtype=DEFAULT_DTYPE) if A is not None else None
        self.training_state.C = np.asarray(C, dtype=DEFAULT_DTYPE) if C is not None else None
        self.training_state.Q = np.asarray(Q, dtype=DEFAULT_DTYPE) if Q is not None else None
        self.training_state.R = np.asarray(R, dtype=DEFAULT_DTYPE) if R is not None else None
        self.training_state.Z_0 = np.asarray(Z_0, dtype=DEFAULT_DTYPE) if Z_0 is not None else None
        self.training_state.V_0 = np.asarray(V_0, dtype=DEFAULT_DTYPE) if V_0 is not None else None
    
    
    def fit(
        self,
        X: Optional[Union[np.ndarray, Any]] = None,
        checkpoint_callback: Optional[Any] = None
    ) -> DFMModelState:
        """Fit model using EM algorithm (wrapper around pykalman).
        
        Uses pykalman for E-step (Kalman filter/smoother) and custom M-step
        that preserves block structure and mixed-frequency constraints.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Standardized data (T x N). If None, data is extracted from dataset.
        checkpoint_callback : callable, optional
            Callback function for checkpointing during training.
            
        Returns
        -------
        DFMModelState
            Complete model state including structure, mixed-frequency parameters, and fitted state-space parameters
        """

        # Bug fix 1.3: Clear cached smoothed factors on retrain
        self._cached_smoothed_factors = None
        self._result = None  # Also clear result cache
        
        # Clear all caches for fresh training run (ensures no stale data from previous runs)
        if self._config is not None and hasattr(self._config, '_cached_blocks'):
            self._config._cached_blocks = None
        
        # Bug fix 1.1: Use X parameter if provided, otherwise extract from dataset
        # Extract data from DFMDataset (variables and time_index are extracted at initialization)
        # Dataset provides: variables (DataFrame) and time_index (TimeIndex)
        dataset = self._dataset
        if X is not None:
            # Use provided X - convert to numpy if needed
            if isinstance(X, np.ndarray):
                X_np = X
                # Get columns from dataset for validation
                data_df = dataset.variables
                columns = list(data_df.columns)
                if X_np.shape[1] != len(columns):
                    raise DataValidationError(
                        f"Provided X has {X_np.shape[1]} columns but dataset has {len(columns)} columns. "
                        f"Column count must match.",
                        details=f"X shape: {X_np.shape}, dataset columns: {columns[:5]}..."
                    )
            else:
                # Convert DataFrame to numpy
                X_np = np.asarray(X)
                if hasattr(X, 'columns'):
                    columns = list(X.columns)
                else:
                    data_df = dataset.variables
                    columns = list(data_df.columns)
        else:
            # Extract from dataset (original behavior)
            data_df = dataset.variables  # Extract variables DataFrame
            X_np = data_df.values  # Convert to numpy array
            columns = list(data_df.columns)

        # Get clock frequency from config
        clock = getattr(self._config, 'clock', DEFAULT_CLOCK_FREQUENCY)
        
        # Bug fix 2.2: Store column names for validation (column order validation happens after standardization)
        # Store column names for later validation in update()
        self._training_columns = columns
        
        # Bug fix 1.2: Check if scaler is already fitted before fit_transform
        # Standardize data if scaler is available
        if self.scaler is not None:
            # Check if scaler is already fitted (has mean_ or scale_ attributes)
            is_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
            if is_fitted:
                # Already fitted - only transform
                X_np_std = self.scaler.transform(X_np)
            else:
                # Not fitted - fit and transform
                X_np_std = self.scaler.fit_transform(X_np)
        else:
            X_np_std = X_np
        
        # GUARDRAIL: Validate data scaling after preprocessing
        # Principle 4: Make preprocessing a contract, not a suggestion
        # Unscaled data (e.g., raw FX levels) causes numerical instability and slow runs
        # Check AFTER scaler is applied (if provided) to catch cases where scaler wasn't applied
        data_std = np.nanstd(X_np_std, axis=0)
        unscaled_series = np.where((data_std > MAX_STD_UNSCALED) | (data_std < MIN_STD_SCALED))[0]
        if len(unscaled_series) > 0:
            unscaled_names = [columns[i] for i in unscaled_series[:5]]  # Show first 5
            raise DataValidationError(
                f"Input data appears unscaled after preprocessing. Series with problematic std: {unscaled_names}... "
                f"(std range: [{np.min(data_std):.2e}, {np.max(data_std):.2e}]). "
                f"Please apply a scaler (e.g., StandardScaler) before fitting the model. "
                f"Unscaled data causes numerical instability, excessive stabilization, and slow convergence.",
                details=f"Expected std range: [{MIN_STD_SCALED:.2e}, {MAX_STD_UNSCALED:.2e}]. "
                       f"Found {len(unscaled_series)}/{len(columns)} series outside this range. "
                       f"Scaler provided: {self.scaler is not None}"
            )
        
        # Build blocks array to match actual data dimensions
        N_actual = X_np_std.shape[1]
        self.blocks = build_dfm_blocks(self.blocks, self._config, columns, N_actual)
        log_blocks_diagnostics(self.blocks, columns, N_actual)
        
        # Setup mixed-frequency parameters
        mf_params = setup_mixed_frequency_params(self._config, clock, columns, N_actual)
        
        # CRITICAL: Column ordering validation is now enforced in setup_mixed_frequency_params()
        # It will raise DataValidationError if [clock series | slower series] ordering is violated
        # This is a hard requirement for correct mixed-frequency handling

        self._mixed_freq        = mf_params['mixed_freq']
        self._constraint_matrix = mf_params['R_mat']
        self._constraint_vector = mf_params['q']
        self._n_slower_freq     = mf_params['n_slower_freq']
        self._n_clock_freq      = mf_params['n_clock_freq']
        self._tent_weights_dict = mf_params['tent_weights_dict']
        self._frequencies       = mf_params['frequencies_np']
        self._idio_indicator    = mf_params['idio_indicator']
        self._max_lag_size      = max(self.p + 1, mf_params['tent_kernel_size'])
        
        # Initialize parameters (required for EM algorithm)
        _logger.info("Initializing DFM parameters...")
        _logger.info(f"  Data: {X_np_std.shape[0]} time steps × {X_np_std.shape[1]} series")
        _logger.info(f"  Blocks: {self.blocks.shape[1]}, Factors: {self.num_factors}, "
                    f"Mixed freq: {self._mixed_freq}")
        A_np, C_np, Q_np, R_np, Z_0_np, V_0_np = initialize_parameters(
            X_np_std, self.r, self.p, self.blocks, 
            self._constraint_matrix, self._constraint_vector, self._n_slower_freq, 
            self._idio_indicator, clock, self._tent_weights_dict
        )
        
        self.data_processed = X_np_std
        
        # Validate numerical stability before proceeding
        validate_dfm_initialization(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        self._update_parameters(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        
        _logger.info(f"Initialization complete: state_dim={self.A.shape[0]}, obs_dim={self.C.shape[0]}, "
                    f"factors={self.num_factors}, max_lag={self._max_lag_size}, mixed_freq={self._mixed_freq}")
        
        # Update training_state with mixed-frequency parameters before passing to EM
        self.training_state.mixed_freq = self._mixed_freq
        self.training_state.constraint_matrix = self._constraint_matrix
        self.training_state.constraint_vector = self._constraint_vector
        self.training_state.n_slower_freq = self._n_slower_freq
        self.training_state.n_clock_freq = self._n_clock_freq
        self.training_state.tent_weights_dict = self._tent_weights_dict
        self.training_state.frequencies = self._frequencies
        self.training_state.idio_indicator = self._idio_indicator
        self.training_state.max_lag_size = self._max_lag_size
        
        initial_state = self.training_state
        
        # Create EMConfig from DFMConfig (uses consolidated parameters)
        em_config = self._config.to_em_config() if self._config is not None else _EM_CONFIG
        
        # Run EM algorithm using run_em_algorithm() directly
        from ...functional.em import run_em_algorithm
        try:
            final_state, training_metadata = run_em_algorithm(
                X=X_np_std,  # Use standardized data
                initial_state=initial_state,
                max_iter=self._config.max_iter,
                threshold=self._config.threshold,
                config=em_config,
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
        
        # Update training_state with all parameters from final_state
        self.training_state = final_state
        
        # Store training metadata separately (not in state-space params)
        self._training_loglik = training_metadata['loglik']
        self._training_num_iter = training_metadata['num_iter']
        self._training_converged = training_metadata['converged']
        
        # Cache smoothed factors from last EM iteration (avoids recomputation during save())
        if 'smoothed_factors' in training_metadata and training_metadata['smoothed_factors'] is not None:
            self._cached_smoothed_factors = training_metadata['smoothed_factors']
        
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
        
        Uses cached factors from last EM iteration if available (avoids expensive recomputation).
        Otherwise, computes them using filter_and_smooth().
        
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
        
        # Use cached smoothed factors from last EM iteration if available
        # This avoids expensive recomputation during save() (saves 100-300s for large datasets)
        if self._cached_smoothed_factors is not None:
            _logger.debug("Using cached smoothed factors from last EM iteration (skipping recomputation)")
            return self._cached_smoothed_factors
        
        # Fallback: compute smoothed factors (should rarely happen if EM completed successfully)
        _logger.debug("Computing smoothed factors (cache not available)")
        kalman_final = self._create_kalman_filter()
        y_masked = np.ma.masked_invalid(self.data_processed)
        # Use filter_and_smooth() with compute_loglik=False to save time (log-likelihood already known)
        # This prevents SVD convergence failures with ill-conditioned covariance matrices
        smoothed_state_means, _, _, _ = kalman_final.filter_and_smooth(y_masked, compute_loglik=False)
        
        return smoothed_state_means
    
    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        # Return cached result if available (avoids expensive recomputation)
        if self._result is not None:
            return self._result
        
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
        
        # Get training metadata from instance attributes
        converged = getattr(self, '_training_converged', False)
        num_iter = getattr(self, '_training_num_iter', 0)
        loglik = getattr(self, '_training_loglik', 0.0)
        
        result = DFMResult(
            x_sm=x_sm, Z=Z, C=C, R=R, A=A, Q=Q,
            target_scaler=None,  # DFM uses model.scaler, not target_scaler
            Z_0=Z_0, V_0=V_0, r=self.r, p=self.p,
            converged=converged,
            num_iter=num_iter,
            loglik=loglik
        )
        
        # Cache result to avoid recomputation
        self._result = result
        return result
    
    def _get_last_state(self, result: DFMResult, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """Get last state from result.Z with safe fallback.
        
        Safely extracts the last row from result.Z, falling back to result.Z_0
        if result.Z is empty or invalid.
        
        Parameters
        ----------
        result : DFMResult
            Model result containing Z and Z_0
        fallback : np.ndarray, optional
            Fallback state if result.Z is invalid. If None, uses result.Z_0.
        
        Returns
        -------
        np.ndarray
            Last state vector (1D array)
        """
        if fallback is None:
            fallback = result.Z_0
        
        if has_shape_with_min_dims(result.Z, min_dims=1) and result.Z.shape[0] > 0:
            return result.Z[-1, :]
        return fallback
    
    
    def update(self, data: Union[np.ndarray, Any], rescale: bool = False, retrain: bool = False, **kwargs) -> None:
        """Update model state with new observations.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations (T_new x N). Must be preprocessed but NOT standardized.
        rescale : bool, default False
            If True, fit scaler on new data and transform. If False, only transform.
        retrain : bool, default False
            If True, run full EM. If False, only update state via Kalman filter.
        **kwargs
            Additional parameters for retraining:
            - max_iter : int, default 1
                Number of EM iterations when retrain=True
            - threshold : float, optional
                Convergence threshold for EM
            
        Notes
        -----
        - Data must be preprocessed (transformations applied) but NOT standardized
        - Columns must be in the same order as training data
        """
        # Validate column order and format if data is DataFrame
        validate_column_order(data, self.scaler)
        
        # Validate and convert data
        from ..numeric.validator import validate_and_convert_update_data
        data_new = validate_and_convert_update_data(
            data, 
            self.data_processed, 
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        
        # Scale data
        if self.scaler is not None:
            if rescale:
                self.scaler.fit(data_new)
            data_new = self.scaler.transform(data_new)
        
        if retrain:
            # Run full EM
            X_combined = np.vstack([self.data_processed, data_new])
            self.data_processed = X_combined
            
            # Update training_state with current parameters
            initial_state = self.training_state
            
            # Create EMConfig
            em_config = self._config.to_em_config() if self._config is not None else _EM_CONFIG
            
            # Get max_iter and threshold from kwargs or defaults
            max_iter = kwargs.get('max_iter', 1)
            threshold = kwargs.get('threshold', self._config.threshold)
            
            # Run EM
            from ...functional.em import run_em_algorithm
            final_state, training_metadata = run_em_algorithm(
                X=X_combined,
                initial_state=initial_state,
                max_iter=max_iter,
                threshold=threshold,
                config=em_config,
                checkpoint_callback=None
            )
            
            # Update training_state
            self.training_state = final_state
            
            # Update training metadata
            self._training_loglik = training_metadata['loglik']
            self._training_num_iter = training_metadata['num_iter']
            self._training_converged = training_metadata['converged']
            
            # CRITICAL: Invalidate cached smoothed factors on retrain
            # Cached factors become stale after parameter updates
            self._result = None
            self._cached_smoothed_factors = None
            _logger.debug("Invalidated cached smoothed factors due to retrain")
        else:
            # Only update state via Kalman filter
            result = self._ensure_result()
            
            # Get last smoothed state as initial state for new data
            Z_last = self._get_last_state(result)
            
            # Bug fix 1.4: Use last filtered covariance, not initial covariance V_0
            # V_0 is initial state covariance, not the covariance of the last filtered state
            # We need to run filter once to get the last filtered covariance
            kalman_temp = self._create_kalman_filter()
            y_masked_temp = np.ma.masked_invalid(self.data_processed)
            # Run filter to get filtered covariances
            filtered_states, filtered_covariances = kalman_temp.filter(y_masked_temp)
            # Get last filtered covariance (T x m x m) -> (m x m)
            if filtered_covariances.shape[0] > 0:
                V_last = filtered_covariances[-1, :, :]
            else:
                # Fallback to initial covariance if no filtered states
                V_last = result.V_0
                _logger.warning(
                    "No filtered covariances available, using initial covariance V_0. "
                    "This may underestimate uncertainty in update()."
                )
            
            # Create Kalman filter
            kalman_new = self._create_kalman_filter(
                initial_state_mean=Z_last,
                initial_state_covariance=V_last
            )
            
            # Run filter and smooth on new data
            y_masked = np.ma.masked_invalid(data_new)
            
            # Handle single timestep case (pykalman quirk)
            if hasattr(data_new, "shape") and len(data_new.shape) == 2 and int(data_new.shape[0]) == 1:
                pk = getattr(kalman_new, "_pykalman", None)
                if pk is None:
                    raise ModelNotTrainedError("Kalman filter not initialized for update()")
                obs = np.ma.masked_invalid(data_new[0])
                next_mean, _ = pk.filter_update(
                    filtered_state_mean=Z_last,
                    filtered_state_covariance=V_last,
                    observation=obs,
                )
                Z_new = np.asarray(next_mean, dtype=DEFAULT_DTYPE)[np.newaxis, :]
            else:
                Z_new, _, _, _ = kalman_new.filter_and_smooth(y_masked, compute_loglik=False)
            
            # Update model state
            result.Z = np.vstack([result.Z, Z_new])
            self.data_processed = np.vstack([self.data_processed, data_new])
            result.x_sm = result.Z @ result.C.T
            
            # CRITICAL: Invalidate cached smoothed factors after update
            # Cached factors are now stale because result.Z was mutated
            self._cached_smoothed_factors = None
            self._result = result  # Update cached result with new Z
            _logger.debug("Invalidated cached smoothed factors after update() - result.Z was mutated")
    
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
        data: Optional[Union[np.ndarray, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame, optional
            New preprocessed observations (T_new x N). If None, uses latest training data.
            Data must be preprocessed but NOT standardized.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_forecast, Z_forecast) where:
            - X_forecast: (horizon x N) forecasted series
            - Z_forecast: (horizon x m) forecasted factors
        """
        # Validate model is trained
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() first"
        )
        
        self._check_parameters_initialized()
        result = self._ensure_result()
        
        check_condition(
            result.Z is not None,
            ModelNotTrainedError,
            "DFM prediction failed: result.Z is not available"
        )
        
        # Validate and resolve horizon
        from ...numeric.validator import validate_horizon
        from ...config.constants import DEFAULT_FORECAST_HORIZON
        if horizon is None:
            if self._config is not None:
                from ...utils.misc import compute_default_horizon
                try:
                    horizon = compute_default_horizon(self._config)
                except (AttributeError, ValueError):
                    horizon = DEFAULT_FORECAST_HORIZON
            else:
                horizon = DEFAULT_FORECAST_HORIZON
        horizon = validate_horizon(horizon)
        A = result.A
        C = result.C
        
        # Determine initial factor state
        if data is not None:
            # Validate and convert data
            from ...numeric.validator import validate_and_convert_update_data
            data_new = validate_and_convert_update_data(
                data,
                self.data_processed,
                dtype=DEFAULT_DTYPE,
                model_name=self.__class__.__name__
            )
            
            # Always apply scaler transform
            if self.scaler is not None:
                data_new = self.scaler.transform(data_new)
            
            # Get last smoothed state from training as initial state
            Z_initial = self._get_last_state(result)
            V_initial = result.V_0
            
            # Create Kalman filter and run forward pass
            kalman_filter = self._create_kalman_filter(
                initial_state_mean=Z_initial,
                initial_state_covariance=V_initial
            )
            
            y_masked = np.ma.masked_invalid(data_new)
            filtered_states, _ = kalman_filter.filter(y_masked)
            
            # Extract last filtered state
            if has_shape_with_min_dims(filtered_states, min_dims=1) and filtered_states.shape[0] > 0:
                Z_last = filtered_states[-1, :]
            else:
                Z_last = Z_initial
        else:
            # Use latest training state
            fallback = np.zeros(result.A.shape[0], dtype=DEFAULT_DTYPE) if result.A is not None else None
            Z_last = self._get_last_state(result, fallback=fallback)
        
        # Validate
        from ...numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(Z_last, name="factor state Z_last")
        validate_no_nan_inf(A, name="transition matrix A")
        validate_no_nan_inf(C, name="observation matrix C")
        
        # Forecast factors forward
        from ...numeric.estimator import forecast_ar1_factors
        Z_forecast = forecast_ar1_factors(Z_last, A, horizon, dtype=DEFAULT_DTYPE)
        
        # Transform to series
        X_forecast_std = Z_forecast @ C.T
        
        # Log extreme predictions
        max_abs_value = np.abs(X_forecast_std).max()
        if max_abs_value > 5.0:
            _logger.warning(
                f"DFM predictions in standardized space are extreme (max abs: {max_abs_value:.2f}). "
                f"This indicates model instability or poor training."
            )
        
        # Unstandardize forecasts
        if self.scaler is not None:
            X_forecast = self.scaler.inverse_transform(X_forecast_std)
        else:
            X_forecast = X_forecast_std
        
        # Validate
        from ...config.types import to_numpy
        X_forecast = to_numpy(X_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(X_forecast, name="forecast X_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        
        return X_forecast, Z_forecast
    
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
        
        Parameters
        ----------
        path : str or Path
            Path to save the model checkpoint file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare result
        result = self._result if self._result is not None else (self.get_result() if self.training_state is not None else None)
        
        # Extract dataset metadata
        dataset_metadata = None
        if hasattr(self, '_dataset') and self._dataset is not None:
            dataset = self._dataset
            processed_columns = getattr(dataset, '_processed_columns', None) or getattr(dataset, 'colnames', None)
            dataset_metadata = {
                '_processed_columns': processed_columns,
                'colnames': getattr(dataset, 'colnames', None),
                'time_idx': getattr(dataset, 'time_idx', None),
            }
        
        checkpoint = {
            'model_state': self.training_state if self.training_state is not None else DFMModelState.from_model(self),
            'result': result,
            'config': self._config,
            'data_processed': self.data_processed,
            'scaler': self.scaler,
            'dataset_metadata': dataset_metadata,
            'training_loglik': getattr(self, '_training_loglik', None),
            'training_num_iter': getattr(self, '_training_num_iter', None),
            'training_converged': getattr(self, '_training_converged', None),
        }
        
        temp_path = path.with_suffix('.pkl.tmp')
        
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                import os
                os.fsync(f.fileno())
            
            temp_path.replace(path)
            
            # Verify saved file
            try:
                with open(path, 'rb') as f:
                    pickle.load(f)
            except Exception as e:
                _logger.error(f"Failed to verify saved checkpoint: {e}")
                try:
                    path.unlink()
                except Exception:
                    pass
                raise RuntimeError(f"Checkpoint verification failed: {e}") from e
            
            _logger.info(f"DFM model saved to {path}")
            
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            _logger.error(f"Failed to save DFM model to {path}: {e}")
            raise
    
    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[DFMConfig] = None, dataset: Optional[DFMDataset] = None) -> 'DFM':
        """Load DFM model from checkpoint file.
        
        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file
        config : DFMConfig, optional
            Configuration (if None, loaded from checkpoint)
        dataset : DFMDataset, optional
            DFMDataset instance (required)
            
        Returns
        -------
        DFM
            Loaded DFM model instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        # Try joblib first, fallback to pickle
        try:
            import joblib
            checkpoint = joblib.load(path)
        except Exception:
            try:
                with open(path, 'rb') as f:
                    checkpoint = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to load checkpoint from {path}. "
                    f"The file may be corrupted. Error: {e}"
                ) from e
        
        if config is None:
            config = checkpoint.get('config')
        
        if dataset is None:
            raise ModelNotInitializedError(
                "DFM.load(): dataset is required. "
                "Please provide a DFMDataset instance when loading a model."
            )
        
        # Get model state
        model_state = checkpoint.get('model_state')
        if model_state is None:
            raise RuntimeError(f"Checkpoint missing 'model_state'. Invalid checkpoint format.")
        
        if isinstance(model_state, dict):
            model_state = DFMModelState(**{k: v for k, v in model_state.items() if v is not None})
        
        # Create model instance
        model = cls(config=config, dataset=dataset)
        
        # Apply model state
        model_state.apply_to_model(model)
        model.training_state = model_state
        
        # Restore state
        model.scaler = checkpoint.get('scaler')
        model.data_processed = checkpoint.get('data_processed')
        model._training_loglik = checkpoint.get('training_loglik')
        model._training_num_iter = checkpoint.get('training_num_iter')
        model._training_converged = checkpoint.get('training_converged')
        
        if checkpoint.get('result') is not None:
            model._result = checkpoint['result']
        
        # Restore checkpoint metadata
        if hasattr(model, '_checkpoint_metadata'):
            model._checkpoint_metadata = {'dataset_metadata': checkpoint.get('dataset_metadata')}
        
        _logger.info(f"DFM model loaded from {path}")
        return model
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        super().reset()
        return self

