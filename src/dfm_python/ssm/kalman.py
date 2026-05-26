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
from ..numeric.stability import ensure_symmetric, cap_max_eigenval
from ..config.constants import (
    MAX_CONDITION_NUMBER_SMOOTHER,
    MAX_CONDITION_NUMBER_INIT,
    MAX_STABILIZATION_AMOUNT,
    MAX_STABILIZATION_AMOUNT_INIT,
    MAX_EIGENVALUE,
    VAR_STABILITY_THRESHOLD,
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
            
            # Option 4: Cap V_0 maximum eigenvalue to prevent overflow in prediction step
            # V_0 is used to compute P_pred = A @ V_0 @ A.T + Q, which can overflow if V_0 is too large
            # This matches the stabilization we do in the EM step
            try:
                initial_state_covariance = cap_max_eigenval(
                    initial_state_covariance,
                    max_eigenval=MAX_EIGENVALUE,  # Same cap as Q
                    symmetric=True,
                    warn=False  # Don't warn here (too verbose, already logged in EM step)
                )
            except (np.linalg.LinAlgError, ValueError):
                # Eigenvalue capping failed - use fallback
                _logger.warning(
                    f"V_0 eigenvalue capping failed in update_parameters. Using scaled identity fallback."
                )
                initial_state_covariance = create_scaled_identity(m_init, MAX_EIGENVALUE, dtype=initial_state_covariance.dtype)
        
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

    # ------------------------------------------------------------------
    # Streaming single-step Kalman filter
    # ------------------------------------------------------------------

    def init_streaming(self) -> None:
        """Initialize streaming state from the stored initial conditions.

        Call once before the first filter_step(). Resets the internal
        state to (Z_0, V_0).
        """
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        self._stream_state = np.asarray(
            self._pykalman.initial_state_mean, dtype=np.float64
        ).copy()
        self._stream_cov = np.asarray(
            self._pykalman.initial_state_covariance, dtype=np.float64
        ).copy()

    def filter_step(
        self, y_t: FloatArray, *, reencode: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Incremental Kalman predict + update for a single observation.

        By default (reencode=True), the state is re-derived from the
        observation via the full predict→update cycle. This ensures the
        streaming state never drifts from the data.

        Set reencode=False to run predict-only (no observation update),
        which produces a pure forecast from the dynamics.

        Parameters
        ----------
        y_t : np.ndarray
            Single observation vector, shape (N,). Ignored when
            reencode=False.
        reencode : bool, default True
            If True (default), incorporate y_t into the state via the
            Kalman update step. If False, only advance the state via
            the transition (predict-only, no observation correction).

        Returns
        -------
        state : np.ndarray
            Updated (or predicted) state mean, shape (m,)
        cov : np.ndarray
            Updated (or predicted) state covariance, shape (m, m)
        """
        if not hasattr(self, '_stream_state'):
            raise ModelNotInitializedError(
                "Streaming state not initialized. Call init_streaming() first."
            )
        kf = self._pykalman
        A = np.asarray(kf.transition_matrices, dtype=np.float64)
        C = np.asarray(kf.observation_matrices, dtype=np.float64)
        Q = np.asarray(kf.transition_covariance, dtype=np.float64)
        R = np.asarray(kf.observation_covariance, dtype=np.float64)

        # Predict
        f_pred = A @ self._stream_state
        P_pred = A @ self._stream_cov @ A.T + Q

        if not reencode:
            self._stream_state = f_pred
            self._stream_cov = ensure_symmetric(P_pred)
            return self._stream_state.copy(), self._stream_cov.copy()

        y_t = np.asarray(y_t, dtype=np.float64).ravel()

        # Innovation
        innov = y_t - C @ f_pred
        S = C @ P_pred @ C.T + R

        # Kalman gain (use solve instead of inv for stability)
        K = np.linalg.solve(S.T, (P_pred @ C.T).T).T

        # Update (Joseph form for numerical stability)
        self._stream_state = f_pred + K @ innov
        IKC = np.eye(A.shape[0]) - K @ C
        self._stream_cov = IKC @ P_pred @ IKC.T + K @ R @ K.T
        self._stream_cov = ensure_symmetric(self._stream_cov)

        return self._stream_state.copy(), self._stream_cov.copy()

    @property
    def streaming_state(self) -> Optional[np.ndarray]:
        """Current streaming state mean, or None if not initialized."""
        return getattr(self, '_stream_state', None)

    @property
    def streaming_cov(self) -> Optional[np.ndarray]:
        """Current streaming state covariance, or None if not initialized."""
        return getattr(self, '_stream_cov', None)
    
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
            
            # Validate parameters for NaN/Inf before calling _filter (prevents cryptic errors)
            A = self._pykalman.transition_matrices
            C = self._pykalman.observation_matrices
            Q = self._pykalman.transition_covariance
            R = self._pykalman.observation_covariance
            Z_0 = self._pykalman.initial_state_mean
            V_0 = self._pykalman.initial_state_covariance
            
            # Check for non-finite values in parameters
            param_checks = {
                'A (transition)': A,
                'C (observation)': C,
                'Q (process noise)': Q,
                'R (observation noise)': R,
                'Z_0 (initial state mean)': Z_0,
                'V_0 (initial state cov)': V_0,
            }
            non_finite_params = []
            for name, param in param_checks.items():
                if param is not None and not np.isfinite(param).all():
                    n_inf = np.sum(np.isinf(param)) if param is not None else 0
                    n_nan = np.sum(np.isnan(param)) if param is not None else 0
                    non_finite_params.append(f"{name}: {n_inf} Inf, {n_nan} NaN")
            
            if non_finite_params:
                # Try to recover by stabilizing parameters
                _logger.warning(
                    f"    Filter: Non-finite values detected in parameters: {', '.join(non_finite_params)}. "
                    f"Attempting to stabilize before filtering..."
                )
                # Stabilize all covariance matrices
                if Q is not None and not np.isfinite(Q).all():
                    Q = ensure_symmetric(Q)
                    Q = np.where(np.isfinite(Q), Q, 0.0)
                    Q = cap_max_eigenval(Q, max_eigenval=MAX_EIGENVALUE, symmetric=True, warn=False)
                if R is not None and not np.isfinite(R).all():
                    R = ensure_symmetric(R)
                    R = np.where(np.isfinite(R), R, np.diag(np.diag(R)))  # Preserve diagonal structure
                    # Ensure R diagonal is positive
                    R_diag = np.diag(R)
                    R_diag = np.where(np.isfinite(R_diag) & (R_diag > 0), R_diag, 1.0)  # Fallback to 1.0
                    np.fill_diagonal(R, R_diag)
                    R = cap_max_eigenval(R, max_eigenval=MAX_EIGENVALUE, symmetric=True, warn=False)
                if V_0 is not None and not np.isfinite(V_0).all():
                    V_0 = ensure_symmetric(V_0)
                    V_0 = np.where(np.isfinite(V_0), V_0, 0.0)
                    V_0 = cap_max_eigenval(V_0, max_eigenval=MAX_EIGENVALUE / 10, symmetric=True, warn=False)
                if A is not None and not np.isfinite(A).all():
                    A = np.where(np.isfinite(A), A, 0.0)
                if C is not None and not np.isfinite(C).all():
                    C = np.where(np.isfinite(C), C, 0.0)
                if Z_0 is not None and not np.isfinite(Z_0).all():
                    Z_0 = np.where(np.isfinite(Z_0), Z_0, 0.0)
            
            # Option 4: Try filter with current parameters, retry with more aggressive stabilization if overflow
            try:
                return _filter(
                    A,
                    C,
                    Q,
                    R,
                    transition_offsets_final,
                    observation_offsets_final,
                    Z_0,
                    V_0,
                    observations
                )
            except (ValueError, FloatingPointError, OverflowError) as e:
                if "infs or NaNs" in str(e) or "overflow" in str(e).lower():
                    _logger.warning(
                        f"    Filter: Overflow detected in _filter ({type(e).__name__}: {e}). "
                        f"Retrying with more aggressively stabilized parameters..."
                    )
                    # Retry with more aggressively stabilized parameters
                    V_0_retry = V_0.copy() if V_0 is not None else self._pykalman.initial_state_covariance.copy()
                    V_0_retry = ensure_symmetric(V_0_retry)
                    V_0_retry = np.where(np.isfinite(V_0_retry), V_0_retry, 0.0)
                    V_0_retry = cap_max_eigenval(V_0_retry, max_eigenval=MAX_EIGENVALUE / 10, symmetric=True, warn=False)
                    
                    Q_retry = Q.copy() if Q is not None else self._pykalman.transition_covariance.copy()
                    Q_retry = ensure_symmetric(Q_retry)
                    Q_retry = np.where(np.isfinite(Q_retry), Q_retry, 0.0)
                    Q_retry = cap_max_eigenval(Q_retry, max_eigenval=MAX_EIGENVALUE, symmetric=True, warn=False)
                    
                    # Also stabilize R (critical: R appears in predicted_observation_covariance = C P C^T + R)
                    R_retry = R.copy() if R is not None else self._pykalman.observation_covariance.copy()
                    R_retry = ensure_symmetric(R_retry)
                    R_retry = np.where(np.isfinite(R_retry), R_retry, np.diag(np.diag(R_retry)))
                    # Ensure R diagonal is positive and finite
                    R_diag = np.diag(R_retry)
                    R_diag = np.where(np.isfinite(R_diag) & (R_diag > 0), R_diag, 1.0)
                    np.fill_diagonal(R_retry, R_diag)
                    R_retry = cap_max_eigenval(R_retry, max_eigenval=MAX_EIGENVALUE, symmetric=True, warn=False)
                    
                    # Also stabilize A if it has eigenvalues > 1 (causes unbounded growth in prediction step)
                    # P_{t|t-1} = A P_{t-1|t-1} A^T + Q grows unbounded if A has eigenvalues > 1
                    A_retry = A.copy() if A is not None else self._pykalman.transition_matrices.copy()
                    if A_retry is not None:
                        A_retry = np.where(np.isfinite(A_retry), A_retry, 0.0)
                        # Check if A has eigenvalues > 1 (unstable)
                        try:
                            eigvals = np.linalg.eigvals(A_retry)
                            max_eig = np.max(np.abs(eigvals))  # Use absolute value for complex eigenvalues
                            if max_eig > 0.99:  # Close to or above 1.0
                                _logger.warning(
                                    f"    Filter: A has max |eigenvalue| {max_eig:.2e} > 0.99, capping to 0.99 to prevent unbounded growth"
                                )
                                A_retry = cap_max_eigenval(A_retry, max_eigenval=0.99, symmetric=False, warn=False)
                        except (np.linalg.LinAlgError, ValueError):
                            # If eigendecomposition fails, just ensure finite and scale down
                            _logger.warning("    Filter: Could not compute A eigenvalues, scaling A by 0.9 as safety measure")
                            A_retry = A_retry * 0.9
                    else:
                        A_retry = A
                    
                    try:
                        return _filter(
                            A_retry,
                            C,
                            Q_retry,
                            R_retry,
                            transition_offsets_final,
                            observation_offsets_final,
                            Z_0,
                            V_0_retry,
                            observations
                        )
                    except (ValueError, FloatingPointError, OverflowError) as e2:
                        # Even retry failed - parameters are too unstable
                        _logger.error(
                            f"    Filter: Retry also failed with {type(e2).__name__}: {e2}. "
                            f"Parameters are too unstable to filter. This indicates severe numerical issues."
                        )
                        raise NumericalError(
                            "Kalman filter failed even after parameter stabilization. "
                            "Model parameters contain extreme values that cannot be filtered. "
                            "This typically indicates: (1) model is too complex for the data, "
                            "(2) data scaling issues, (3) numerical instability in EM convergence.",
                            details=f"Original error: {type(e).__name__}: {e}, Retry error: {type(e2).__name__}: {e2}"
                        ) from e2
                else:
                    raise
        
        # Run filter
        predicted_state_means, predicted_state_covariances, _, filtered_state_means, filtered_state_covariances = run_filter()
        
        # Option 4: Stabilize filtered and predicted state covariances to prevent overflow
        # The overflow occurs because P_filtered grows unbounded during filtering, causing
        # A @ P_filtered @ A.T + Q to overflow in the NEXT timestep's prediction step.
        # By capping maximum eigenvalues of both filtered and predicted covariances,
        # we prevent this growth and ensure numerical stability.
        # This is similar to how we stabilize V_0 and Q - we need to bound all covariances.
        _logger.debug(f"    Filter: Stabilizing {len(filtered_state_covariances)} filtered and {len(predicted_state_covariances)} predicted covariance matrices...")
        n_filtered_stab = 0
        n_predicted_stab = 0
        n_predicted_recovered = 0
        
        # Stabilize filtered covariances (used in next timestep's prediction: A @ P_filtered @ A.T + Q)
        # CRITICAL: This prevents overflow in the NEXT filter run's prediction step
        for t in range(len(filtered_state_covariances)):
            P_filtered_t = filtered_state_covariances[t]
            # Check for non-finite values (shouldn't happen, but handle gracefully)
            if np.any(~np.isfinite(P_filtered_t)):
                _logger.warning(
                    f"    Filter: filtered_state_covariances[{t}] contains non-finite values. Recovering..."
                )
                P_filtered_t = np.where(np.isfinite(P_filtered_t), P_filtered_t, 0.0)
                P_filtered_t = ensure_symmetric(P_filtered_t)
            
            # Cap maximum eigenvalue to prevent overflow in prediction step
            P_filtered_t_stab = cap_max_eigenval(
                P_filtered_t,
                max_eigenval=MAX_EIGENVALUE,  # Same cap as Q
                symmetric=True,
                warn=False  # Don't warn for each timestep (too verbose)
            )
            if not np.array_equal(P_filtered_t, P_filtered_t_stab):
                n_filtered_stab += 1
                filtered_state_covariances[t] = P_filtered_t_stab
        
        # Stabilize predicted covariances (used in correction step, can overflow if too large)
        # CRITICAL: Overflow may have already occurred inside _filter, so we need to recover
        from ..numeric.stability import create_scaled_identity
        for t in range(len(predicted_state_covariances)):
            P_pred_t = predicted_state_covariances[t]
            # Check for non-finite values first (overflow already occurred inside _filter)
            if np.any(~np.isfinite(P_pred_t)):
                n_predicted_recovered += 1
                if n_predicted_recovered <= 5:  # Only log first few to avoid spam
                    _logger.warning(
                        f"    Filter: predicted_state_covariances[{t}] contains non-finite values "
                        f"(Inf: {np.sum(np.isinf(P_pred_t))}, NaN: {np.sum(np.isnan(P_pred_t))}). "
                        f"Overflow occurred during prediction step. Recovering..."
                    )
                # Use safe fallback: identity matrix scaled by MAX_EIGENVALUE
                # This ensures we have a valid covariance matrix for correction step
                P_pred_t = create_scaled_identity(P_pred_t.shape[0], MAX_EIGENVALUE, dtype=P_pred_t.dtype)
                predicted_state_covariances[t] = P_pred_t
                continue  # Skip eigenvalue capping since we already replaced with safe value
            
            # Cap maximum eigenvalue to prevent overflow in correction step
            try:
                P_pred_t_stab = cap_max_eigenval(
                    P_pred_t,
                    max_eigenval=MAX_EIGENVALUE,  # Same cap as Q
                    symmetric=True,
                    warn=False
                )
                if not np.array_equal(P_pred_t, P_pred_t_stab):
                    n_predicted_stab += 1
                    predicted_state_covariances[t] = P_pred_t_stab
            except (np.linalg.LinAlgError, ValueError):
                # Eigenvalue capping failed (matrix too ill-conditioned)
                # Use safe fallback
                P_pred_t = create_scaled_identity(P_pred_t.shape[0], MAX_EIGENVALUE, dtype=P_pred_t.dtype)
                predicted_state_covariances[t] = P_pred_t
                n_predicted_recovered += 1
        
        if n_predicted_recovered > 0:
            _logger.error(
                f"    Filter: Recovered {n_predicted_recovered}/{len(predicted_state_covariances)} predicted covariances "
                f"from overflow. This indicates severe numerical instability. "
                f"Consider: (1) Better initialization, (2) More aggressive V_0/Q stabilization, "
                f"(3) Data scaling issues."
            )
        if n_filtered_stab > 0 or n_predicted_stab > 0:
            _logger.warning(
                f"    Filter: Stabilized {n_filtered_stab}/{len(filtered_state_covariances)} filtered and "
                f"{n_predicted_stab}/{len(predicted_state_covariances)} predicted covariance matrices "
                f"(capped max eigenvalue to {MAX_EIGENVALUE:.2e}). This prevents overflow in prediction/correction steps."
            )
        
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
                               f"Sample time indices checked (t): {sample_indices}, max condition: {max_cond:.2e}"
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
                _logger.warning(f"    Smooth: Condition check failed ({e}); interpreting indices as time (t). Using base regularization")
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
    
