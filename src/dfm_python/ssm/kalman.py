"""Kalman filter wrapper for DFM using pykalman."""

from typing import Tuple, Optional
import time as time_module
import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
# Bug fix 3.1: Using private pykalman APIs is risky but documented
# These are private, undocumented functions that may change across versions
# Consider migrating to public API if pykalman provides it
from pykalman.standard import _filter, _smooth, _smooth_pair

from ..logger import get_logger
from ..utils.errors import ModelNotInitializedError, NumericalError
from ..config.types import FloatArray
from ..numeric.stability import ensure_symmetric
from ..config.constants import (
    MAX_CONDITION_NUMBER_SMOOTHER,
    MAX_CONDITION_NUMBER_INIT,
    MAX_STABILIZATION_AMOUNT,
    MAX_STABILIZATION_AMOUNT_INIT,
)

_logger = get_logger(__name__)


class DFMKalmanFilter:
    """Wrapper around pykalman for DFM E-step. Uses pykalman for filter/smooth, custom M-step."""
    
    def __init__(
        self,
        transition_matrices: Optional[FloatArray] = None,
        observation_matrices: Optional[FloatArray] = None,
        transition_covariance: Optional[FloatArray] = None,
        observation_covariance: Optional[FloatArray] = None,
        initial_state_mean: Optional[FloatArray] = None,
        initial_state_covariance: Optional[FloatArray] = None
    ) -> None:
        self._pykalman = None
        # Cache for smoothed factors from last EM iteration (avoids recomputation during save())
        # Bug fix 3.2: Cache is unsafe if parameters change after E-step (damping, clipping)
        # Cache should be invalidated when parameters are updated
        self._cached_smoothed_factors: Optional[np.ndarray] = None
        # IMPORTANT:
        # Always go through update_parameters() so covariance stabilization is applied consistently.
        if all(p is not None for p in [
            transition_matrices, observation_matrices,
            transition_covariance, observation_covariance,
            initial_state_mean, initial_state_covariance
        ]):
            self.update_parameters(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance
            )
    
    def update_parameters(
        self,
        transition_matrices: FloatArray,
        observation_matrices: FloatArray,
        transition_covariance: FloatArray,
        observation_covariance: FloatArray,
        initial_state_mean: FloatArray,
        initial_state_covariance: FloatArray,
        apply_stabilization: bool = True
    ) -> None:
        """Update filter parameters.
        
        Parameters
        ----------
        transition_matrices : np.ndarray
            Transition matrix A (m x m)
        observation_matrices : np.ndarray
            Observation matrix C (N x m)
        transition_covariance : np.ndarray
            Process noise covariance Q (m x m)
        observation_covariance : np.ndarray
            Observation noise covariance R (N x N)
        initial_state_mean : np.ndarray
            Initial state mean Z_0 (m,)
        initial_state_covariance : np.ndarray
            Initial state covariance V_0 (m x m)
        apply_stabilization : bool, default True
            If True, apply diagonal regularization to covariances for numerical stability.
            If False, use raw covariances (may fail on ill-conditioned matrices).
            NOTE: Stabilization biases E-step moments. For strict EM, set to False and handle
            numerical issues through priors or better initialization.
        """
        # Keep everything in float64 for numerical stability in large state spaces.
        transition_matrices = np.asarray(transition_matrices, dtype=np.float64)
        observation_matrices = np.asarray(observation_matrices, dtype=np.float64)
        transition_covariance = np.asarray(transition_covariance, dtype=np.float64)
        observation_covariance = np.asarray(observation_covariance, dtype=np.float64)
        initial_state_mean = np.asarray(initial_state_mean, dtype=np.float64)
        initial_state_covariance = np.asarray(initial_state_covariance, dtype=np.float64)

        # CRITICAL: Stabilization biases E-step moments
        # When apply_stabilization=True, we compute p(z_t | y_{1:T}, P_t + εI) instead of p(z_t | y_{1:T}, P_t)
        # This means E-step moments are biased, and M-step optimizes a regularized surrogate Q-function
        # This is acceptable for practical EM but breaks strict EM theory
        # Store stabilization amount for potential bias correction
        self._stabilization_applied = False
        self._stabilization_amount = 0.0
        
        # Fast PSD check: use Cholesky decomposition which is faster and fails if not PSD
        # For very large matrices (>100), skip check and apply minimal regularization
        def is_psd_fast(cov, tol=1e-8):
            """Fast check if covariance matrix is positive semi-definite using Cholesky."""
            try:
                # Cholesky fails if matrix is not PSD (faster than eigendecomposition)
                # Add small regularization to handle numerical issues
                np.linalg.cholesky(cov + np.eye(cov.shape[0], dtype=cov.dtype) * tol)
                return True
            except np.linalg.LinAlgError:
                return False
            except Exception:
                return False
        
        if not apply_stabilization:
            # Use raw covariances (may fail on ill-conditioned matrices)
            # This is for strict EM where numerical issues should be handled through priors
            pass
        else:
            # Bug fix 1.1: Conditional stabilization - only apply if PSD check fails
            # Unconditional diagonal loading violates Kalman consistency and breaks EM
            # Stabilization should be minimal and only when necessary
            # NOTE: PSD check is expensive (O(m³)), so we use a fast heuristic for large matrices
            from ..config.constants import MIN_EIGENVALUE
            reg = MIN_EIGENVALUE * 10  # 1e-5
            
            # Fast diagonal loading: O(m²) instead of O(m³) eigendecomposition
            # For large matrices, apply minimal regularization unconditionally (faster)
            # For small matrices, check PSD first
            transition_covariance = ensure_symmetric(transition_covariance)
            m_trans = transition_covariance.shape[0]
            if m_trans > 100:
                # Large matrix: apply minimal regularization unconditionally (faster than PSD check)
                transition_covariance = transition_covariance + np.eye(m_trans, dtype=transition_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
            elif not is_psd_fast(transition_covariance):
                # Small matrix: check PSD first, then regularize if needed
                transition_covariance = transition_covariance + np.eye(m_trans, dtype=transition_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
                
                # GUARDRAIL: Fail if stabilization exceeds threshold
                if self._stabilization_amount > MAX_STABILIZATION_AMOUNT:
                    raise NumericalError(
                        f"Stabilization amount ({self._stabilization_amount:.2e}) exceeds maximum allowed "
                        f"({MAX_STABILIZATION_AMOUNT:.2e}). This indicates severe numerical instability. "
                        f"Data likely unscaled or model misconfigured. Please apply a scaler before fitting.",
                        details=f"Stabilization applied to transition_covariance: {self._stabilization_amount:.2e}"
                    )
            
            observation_covariance = ensure_symmetric(observation_covariance)
            N_obs = observation_covariance.shape[0]
            if N_obs > 100:
                # Large matrix: apply minimal regularization unconditionally
                observation_covariance = observation_covariance + np.eye(N_obs, dtype=observation_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
            elif not is_psd_fast(observation_covariance):
                # Small matrix: check PSD first
                observation_covariance = observation_covariance + np.eye(N_obs, dtype=observation_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
                
                # GUARDRAIL: Fail if stabilization exceeds threshold
                if self._stabilization_amount > MAX_STABILIZATION_AMOUNT:
                    raise NumericalError(
                        f"Stabilization amount ({self._stabilization_amount:.2e}) exceeds maximum allowed "
                        f"({MAX_STABILIZATION_AMOUNT:.2e}). This indicates severe numerical instability. "
                        f"Data likely unscaled or model misconfigured. Please apply a scaler before fitting.",
                        details=f"Stabilization applied to observation_covariance: {self._stabilization_amount:.2e}"
                    )
            
            initial_state_covariance = ensure_symmetric(initial_state_covariance)
            m_init = initial_state_covariance.shape[0]
            if m_init > 100:
                # Large matrix: apply minimal regularization unconditionally
                initial_state_covariance = initial_state_covariance + np.eye(m_init, dtype=initial_state_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
            elif not is_psd_fast(initial_state_covariance):
                # Small matrix: check PSD first
                initial_state_covariance = initial_state_covariance + np.eye(m_init, dtype=initial_state_covariance.dtype) * reg
                self._stabilization_applied = True
                self._stabilization_amount = max(self._stabilization_amount, reg)
                
                # GUARDRAIL: Fail if stabilization exceeds threshold
                if self._stabilization_amount > MAX_STABILIZATION_AMOUNT:
                    raise NumericalError(
                        f"Stabilization amount ({self._stabilization_amount:.2e}) exceeds maximum allowed "
                        f"({MAX_STABILIZATION_AMOUNT:.2e}). This indicates severe numerical instability. "
                        f"Data likely unscaled or model misconfigured. Please apply a scaler before fitting.",
                        details=f"Stabilization applied to initial_state_covariance: {self._stabilization_amount:.2e}"
                    )
        
        if self._pykalman is None:
            self._pykalman = PyKalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance
            )
        else:
            self._pykalman.transition_matrices = transition_matrices
            self._pykalman.observation_matrices = observation_matrices
            self._pykalman.transition_covariance = transition_covariance
            self._pykalman.observation_covariance = observation_covariance
            self._pykalman.initial_state_mean = initial_state_mean
            self._pykalman.initial_state_covariance = initial_state_covariance
            
            # Bug fix 3.2: Invalidate cached smoothed factors when parameters change
            # Parameters have changed, so cached factors are no longer valid
            self._cached_smoothed_factors = None
    
    def filter(self, observations: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Run Kalman filter (forward pass).
        
        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) or masked array
            
        Returns
        -------
        filtered_state_means : np.ndarray
            Filtered state means (T x m)
        filtered_state_covariances : np.ndarray
            Filtered state covariances (T x m x m)
        """
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        
        return self._pykalman.filter(observations)
    
    def _stabilize_covariance_matrices(
        self,
        covariances: np.ndarray,
        regularization: Optional[float] = None
    ) -> np.ndarray:
        """Apply symmetrization and diagonal regularization to covariance matrices.
        
        This stabilization prevents SVD convergence failures in the Kalman smoother
        when covariance matrices are ill-conditioned. The operation is O(T × m²) which
        is much cheaper than full eigendecomposition (O(T × m³)).
        
        Parameters
        ----------
        covariances : np.ndarray
            Covariance matrices array of shape (T, m, m) where T is number of timesteps
            and m is state dimension
        regularization : float, optional
            Regularization value to add to diagonal. If None, uses default based on
            MIN_EIGENVALUE constant.
            
        Returns
        -------
        np.ndarray
            Stabilized covariance matrices (same shape as input, modified in-place)
        """
        from ..config.constants import MIN_EIGENVALUE
        
        if regularization is None:
            regularization = max(1e-6, MIN_EIGENVALUE * 100)  # 1e-4 for stability
        
        # Fast diagonal loading: just add regularization to diagonal (O(m²) per matrix)
        # Symmetrize and add diagonal regularization without expensive eigendecomposition
        # Use in-place diagonal modification for efficiency (avoids creating identity matrix in loop)
        for t in range(len(covariances)):
            cov = covariances[t]
            # Symmetrize and add regularization to diagonal (cheap: O(m²))
            cov = ensure_symmetric(cov)
            np.fill_diagonal(cov, np.diagonal(cov) + regularization)
            covariances[t] = cov
        
        return covariances
    
    def filter_and_smooth(
        self,
        observations: FloatArray,
        compute_loglik: bool = True
    ) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
        """Run filter and smooth with numerical stabilization.
        
        This is the **recommended method** for computing smoothed states. It applies
        automatic stabilization to covariance matrices before smoothing, preventing
        SVD convergence failures that can occur with ill-conditioned matrices.
        
        Unlike the deprecated `smooth()` method, this method:
        - Applies symmetrization and diagonal regularization to covariance matrices
        - Prevents SVD convergence failures in pykalman's internal smoother
        - Returns all necessary outputs (smoothed states, covariances, cross-covariances, log-likelihood)
        
        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) or masked array
        compute_loglik : bool, default True
            If True, compute log-likelihood (expensive, O(T * m^3)).
            If False, return -inf as placeholder (saves significant time when log-likelihood not needed).
            
        Returns
        -------
        smoothed_state_means : np.ndarray
            Smoothed state means (T x m)
        smoothed_state_covariances : np.ndarray
            Smoothed state covariances (T x m x m)
        sigma_pair_smooth : np.ndarray
            Lag-1 cross-covariances (T-1 x m x m)
        loglik : float
            Log-likelihood of observations (or -inf if compute_loglik=False)
        """
        # #region agent log
        import json
        e_step_start_time = time_module.time()
        # #endregion
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )

        # Get filtered states first (needed for smoother)
        transition_offsets = getattr(self._pykalman, 'transition_offsets', None)
        observation_offsets = getattr(self._pykalman, 'observation_offsets', None)
        
        # Filter step timing
        filter_start = time_module.time()
        T = observations.shape[0] if hasattr(observations, 'shape') else len(observations)
        m = self._pykalman.transition_matrices.shape[0] if self._pykalman.transition_matrices is not None else 0
        N = self._pykalman.observation_matrices.shape[0] if self._pykalman.observation_matrices is not None else 0
        
        _logger.info(f"    Filter: Processing {T} timesteps (state_dim={m}, obs_dim={N})...")
        
        def run_filter():
            # Use internal functions for standard KalmanFilter (more efficient, gets predicted states)
            # Bug fix 1.3: Ensure offsets have correct shapes
            # transition_offsets: (m,) - state dimension
            # observation_offsets: (N,) - observation dimension
            m = self._pykalman.transition_matrices.shape[0]
            N = self._pykalman.observation_matrices.shape[0]
            
            transition_offsets_final = (
                transition_offsets if transition_offsets is not None 
                else np.zeros(m, dtype=np.float64)
            )
            observation_offsets_final = (
                observation_offsets if observation_offsets is not None 
                else np.zeros(N, dtype=np.float64)
            )
            
            # Validate shapes
            if transition_offsets_final.shape != (m,):
                raise ValueError(
                    f"transition_offsets shape mismatch: expected ({m},), got {transition_offsets_final.shape}"
                )
            if observation_offsets_final.shape != (N,):
                raise ValueError(
                    f"observation_offsets shape mismatch: expected ({N},), got {observation_offsets_final.shape}"
                )
            
            return _filter(
                self._pykalman.transition_matrices,
                self._pykalman.observation_matrices,
                self._pykalman.transition_covariance,
                self._pykalman.observation_covariance,
                transition_offsets_final,
                observation_offsets_final,
                self._pykalman.initial_state_mean,
                self._pykalman.initial_state_covariance,
                observations
            )
        
        # Run filter
        predicted_state_means, predicted_state_covariances, _, filtered_state_means, filtered_state_covariances = run_filter()
        
        # Bug fix 1.1 & 1.2: We'll create stabilized copies inside run_smooth() and save them
        # for use in _smooth_pair. The stabilized versions must be used consistently.

        filter_time = time_module.time() - filter_start
        _logger.info(f"    Filter: Completed in {filter_time:.2f}s ({filter_time/T*1000:.2f}ms/timestep)")
        # Store for later use in smooth timing log
        self._last_filter_time = filter_time
        
        # Smooth to get smoothed states (also O(T × m³) - can be slow)
        _logger.info(f"    Smooth: Processing {T} timesteps (state_dim={m})...")
        smooth_start = time_module.time()
        
        def run_smooth():
            # Lightweight stabilization: add small diagonal regularization to prevent SVD failures
            # This is much cheaper than full eigendecomposition (O(m²) vs O(m³))
            from ..config.constants import MIN_EIGENVALUE
            base_regularization = max(1e-6, MIN_EIGENVALUE * 100)  # 1e-4 for stability
            
            # Bug fix 2.2: Use filtered covariance condition number, not predicted
            # Predicted covariance includes dynamics (A P A' + Q) which can legitimately have high condition
            # Filtered covariance reflects actual uncertainty and is better indicator of numerical issues
            # Adaptive regularization: check condition numbers and increase if needed
            # Investment dataset has larger state dimension (m=193) and worse conditioning (~204k)
            # Production has m=135 and conditioning ~131k
            # If condition number is very high (>1e5), increase regularization
            # GUARDRAIL: Numerical health check before expensive smoothing operation
            # Principle 2: Refuse to smooth if state is already toxic
            try:
                # Sample a few covariance matrices to check condition
                # Use filtered covariances (not predicted) as they better reflect numerical issues
                sample_indices = [0, len(filtered_state_covariances) // 2, len(filtered_state_covariances) - 1]
                sample_indices = [i for i in sample_indices if i < len(filtered_state_covariances)]
                max_cond = max([np.linalg.cond(filtered_state_covariances[i]) for i in sample_indices])
                
                # CRITICAL: Fail fast if condition number is too large
                # Use more lenient threshold (MAX_CONDITION_NUMBER_INIT) and warn if exceeded
                # Only error if it exceeds the stricter threshold (MAX_CONDITION_NUMBER_SMOOTHER)
                if max_cond > MAX_CONDITION_NUMBER_SMOOTHER:
                    raise NumericalError(
                        f"State covariance condition number too large ({max_cond:.2e} > {MAX_CONDITION_NUMBER_SMOOTHER:.2e}). "
                        f"Refusing to smooth. Data likely unscaled or ill-conditioned. "
                        f"Please apply a scaler (e.g., StandardScaler) before fitting the model.",
                        details=f"Condition number check failed before smoothing. "
                               f"Sample indices checked: {sample_indices}, max condition: {max_cond:.2e}"
                    )
                elif max_cond > MAX_CONDITION_NUMBER_INIT:
                    # High but not fatal - log warning and proceed with stabilization
                    _logger.warning(
                        f"    Smooth: High condition number detected ({max_cond:.2e} > {MAX_CONDITION_NUMBER_INIT:.2e}), "
                        f"but below fatal threshold ({MAX_CONDITION_NUMBER_SMOOTHER:.2e}). "
                        f"Proceeding with increased stabilization."
                    )
                
                # Adaptive regularization: scale up if condition number is very high
                # Investment dataset has condition ~204k, production ~131k
                # For very high condition numbers (>1e5), use more aggressive regularization
                if max_cond > 1e5:
                    # More aggressive scaling: for condition ~200k, use ~4-5x regularization
                    # This ensures SVD convergence even for very ill-conditioned matrices
                    adaptive_factor = min(10.0, max(2.0, max_cond / 5e4))  # More aggressive: 200k/50k = 4x
                    regularization = base_regularization * adaptive_factor
                    _logger.warning(
                        f"    Smooth: High condition number detected (max_cond={max_cond:.2e}), "
                        f"increasing regularization from {base_regularization:.2e} to {regularization:.2e} "
                        f"(factor={adaptive_factor:.2f})"
                    )
                elif max_cond > 5e4:
                    # Moderate increase for moderately ill-conditioned matrices
                    adaptive_factor = min(3.0, max(1.5, max_cond / 1e5))
                    regularization = base_regularization * adaptive_factor
                    _logger.info(
                        f"    Smooth: Moderate condition number (max_cond={max_cond:.2e}), "
                        f"increasing regularization to {regularization:.2e} (factor={adaptive_factor:.2f})"
                    )
                else:
                    regularization = base_regularization
            except Exception as e:
                # Fallback to base regularization if condition check fails
                _logger.warning(f"    Smooth: Condition check failed ({e}), using base regularization")
                regularization = base_regularization

            _logger.info(f"    Smooth: Stabilizing {len(predicted_state_covariances)} covariance matrices "
                        f"(regularization={regularization:.2e})")
            
            # CRITICAL: Stabilization before smoothing biases E-step moments
            # We compute p(z_t | y_{1:T}, P_t + εI) instead of p(z_t | y_{1:T}, P_t)
            # This means E-step is optimizing a regularized surrogate, not the true Q-function
            # Store stabilization amount for documentation/debugging
            self._smoothing_stabilization = regularization
            self._stabilization_applied = True
            self._stabilization_amount = max(self._stabilization_amount, regularization)
            
            # Bug fix 1.2 & 2.3: Copy before modifying to avoid mutating pykalman's internal arrays
            # In-place mutation breaks Kalman algebra and cross-covariance identities
            # Create copies to preserve original pykalman arrays
            predicted_state_covariances_stab = predicted_state_covariances.copy()
            filtered_state_covariances_stab = filtered_state_covariances.copy()
            
            # Apply stabilization to copies (not original pykalman arrays)
            # NOTE: This biases the smoother output - smoothed states are computed with regularized covariances
            self._stabilize_covariance_matrices(predicted_state_covariances_stab, regularization)
            self._stabilize_covariance_matrices(filtered_state_covariances_stab, regularization)
            
            if regularization > 1e-4:
                _logger.warning(
                    f"    Smooth: Large stabilization ({regularization:.2e}) biases E-step moments. "
                    f"This breaks strict EM - algorithm is now 'stabilized generalized EM'."
                )
            
            _logger.info("    Smooth: Covariance matrices stabilized, starting smoother")
            
            # Use internal functions for standard KalmanFilter
            # Use stabilized copies for smoothing (to prevent SVD failures)
            smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains = _smooth(
                self._pykalman.transition_matrices,
                filtered_state_means,
                filtered_state_covariances_stab,
                predicted_state_means,
                predicted_state_covariances_stab,
            )
            # Return smoothed results along with stabilized covariances for _smooth_pair
            return (smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains, 
                    filtered_state_covariances_stab, predicted_state_covariances_stab)
        
        # Run smooth
        _logger.info("    Smooth: Starting smoother execution...")
        smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains, filtered_state_covariances_stab, predicted_state_covariances_stab = run_smooth()
        
        smooth_time = time_module.time() - smooth_start
        _logger.info(f"    Smooth: Completed in {smooth_time:.2f}s ({smooth_time/T*1000:.2f}ms/timestep)")
        # #region agent log
        import json
        with open('/data/nowcasting-kr/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'D',
                'location': 'kalman.py:510',
                'message': 'Kalman smoother timing',
                'data': {
                    'T': T,
                    'm': m,
                    'smooth_time': smooth_time,
                    'filter_time': getattr(self, '_last_filter_time', 0.0),
                    'total_e_step_time': time_module.time() - e_step_start_time
                },
                'timestamp': int(time_module.time() * 1000)
            }) + '\n')
        # #endregion
        
        # Compute lag-1 cross-covariances (needed for M-step)
        _logger.info(f"    Smooth-pair: Computing cross-covariances...")
        smooth_pair_start = time_module.time()
        
        # Bug fix 1.4: _smooth_pair uses mismatched covariance lineage
        # _smooth_pair signature is (smoothed_state_covariances, kalman_smoothing_gain)
        # It uses exact Kalman identities that assume consistent covariance lineage.
        # However, if smoothing used stabilized filtered/predicted covariances (apply_stabilization=True),
        # then smoothed_state_covariances come from stabilized covariances, but _smooth_pair assumes
        # exact Kalman identities. This produces internally inconsistent VVsmooth relative to EZ.
        # This contaminates:
        # - VAR estimation (uses VVsmooth in EZZ_FB)
        # - Q updates (uses VVsmooth in process noise estimation)
        # - Block updates (uses VVsmooth in transition matrix updates)
        # When stabilization is active, VVsmooth should be interpreted as approximate.
        if self._stabilization_applied:
            _logger.debug(
                f"_smooth_pair called with stabilized covariances (stabilization={self._stabilization_amount:.2e}). "
                f"VVsmooth will be internally inconsistent with EZ due to covariance lineage mismatch."
            )
        sigma_pair_smooth = _smooth_pair(smoothed_state_covariances, kalman_smoothing_gains)
        
        smooth_pair_time = time_module.time() - smooth_pair_start
        _logger.info(f"    Smooth-pair: Completed in {smooth_pair_time:.2f}s")
        
        # Compute log-likelihood (optional, expensive operation)
        if compute_loglik:
            _logger.info(f"    Log-likelihood: Computing...")
            loglik_start = time_module.time()
            # #region agent log
            import json
            with open('/data/nowcasting-kr/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'kalman.py:540',
                    'message': 'Before loglik computation',
                    'data': {
                        'T': T,
                        'm': m,
                        'stabilization_applied': self._stabilization_applied,
                        'stabilization_amount': float(self._stabilization_amount) if hasattr(self, '_stabilization_amount') else 0.0,
                        'observation_cov_shape': self._pykalman.observation_covariance.shape if self._pykalman else None,
                        'transition_cov_shape': self._pykalman.transition_covariance.shape if self._pykalman else None
                    },
                    'timestamp': int(time_module.time() * 1000)
                }) + '\n')
            # #endregion
            try:
                # CRITICAL: pykalman.loglikelihood() doesn't handle masked arrays correctly
                # Convert masked array to regular array with NaNs (pykalman handles NaNs for missing data)
                if isinstance(observations, np.ma.MaskedArray):
                    observations_for_loglik = np.asarray(observations.filled(np.nan))
                else:
                    observations_for_loglik = observations
                
                # Bug fix 1.3: Log-likelihood uses unstabilized pykalman internal covariances
                # but smoothing used stabilized covariances. This creates model inconsistency:
                # - E-step moments: computed with stabilized covariances
                # - Log-likelihood: computed with unstabilized covariances
                # This breaks even generalized EM logic. When stabilization is active,
                # log-likelihood should be interpreted as diagnostic only, not used for convergence.
                # For strict EM, set apply_stabilization=False and handle numerical issues through priors.
                if self._stabilization_applied:
                    _logger.warning(
                        f"Log-likelihood computed with unstabilized covariances (stabilization={self._stabilization_amount:.2e} was applied). "
                        f"This creates model inconsistency: E-step used stabilized covariances, but loglik uses unstabilized. "
                        f"Log-likelihood should be interpreted as diagnostic only, not for convergence checks."
                    )
                
                loglik = self._pykalman.loglikelihood(observations_for_loglik)
                # Bug fix 3.3: Zero log-likelihood is not inherently pathological
                # It depends on scaling, normalization, and constant offsets
                # Only treat non-finite as an error
                if not np.isfinite(loglik):
                    _logger.error(f"DFMKalmanFilter: Log-likelihood is not finite: {loglik}. This indicates numerical instability.")
                    loglik = float('-inf')
                # Note: loglik == 0.0 is valid (depends on scaling/normalization)
                # Removed incorrect warning about zero log-likelihood
            except (ValueError, RuntimeError, AttributeError) as e:
                _logger.error(f"DFMKalmanFilter: Failed to compute log-likelihood: {e}. Using -inf (will break convergence checks).")
                _logger.debug(f"DFMKalmanFilter: Full exception traceback for loglikelihood computation failure:", exc_info=True)
                loglik = float('-inf')  # Use -inf instead of 0.0 (0.0 would break convergence checks)
            loglik_time = time_module.time() - loglik_start
            _logger.info(f"    Log-likelihood: Completed in {loglik_time:.2f}s, value={loglik:.2e}")
        else:
            # Skip log-likelihood computation (saves significant time when not needed, e.g., during save())
            loglik = float('-inf')
            loglik_time = 0.0
        
        # Log detailed timing breakdown (for debugging/performance analysis)
        total_e_step_time = filter_time + smooth_time + smooth_pair_time + loglik_time
        if total_e_step_time > 5.0:  # Only log if E-step takes significant time
            loglik_msg = f"Loglik={loglik_time:.2f}s ({100*loglik_time/total_e_step_time:.1f}%)" if compute_loglik else "Loglik=skipped"
            _logger.debug(f"E-step breakdown: Filter={filter_time:.2f}s ({100*filter_time/total_e_step_time:.1f}%), "
                        f"Smooth={smooth_time:.2f}s ({100*smooth_time/total_e_step_time:.1f}%), "
                        f"Pair={smooth_pair_time:.2f}s ({100*smooth_pair_time/total_e_step_time:.1f}%), "
                        f"{loglik_msg}")
        
        return smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth, loglik
    
