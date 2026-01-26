"""Kalman filter wrapper for DFM using pykalman."""

from typing import Tuple, Optional
import time as time_module
import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair

from ..logger import get_logger
from ..utils.errors import ModelNotInitializedError
from ..config.types import FloatArray
from ..numeric.stability import ensure_symmetric

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
        initial_state_covariance: FloatArray
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
        """
        # Keep everything in float64 for numerical stability in large state spaces.
        transition_matrices = np.asarray(transition_matrices, dtype=np.float64)
        observation_matrices = np.asarray(observation_matrices, dtype=np.float64)
        transition_covariance = np.asarray(transition_covariance, dtype=np.float64)
        observation_covariance = np.asarray(observation_covariance, dtype=np.float64)
        initial_state_mean = np.asarray(initial_state_mean, dtype=np.float64)
        initial_state_covariance = np.asarray(initial_state_covariance, dtype=np.float64)

        # Lightweight stabilization: add small diagonal regularization (O(m²) operation).
        # Used for high-frequency operations (EM iterations) where speed is critical.
        from ..config.constants import MIN_EIGENVALUE
        reg = MIN_EIGENVALUE * 10  # 1e-5
        
        # Fast diagonal loading: O(m²) instead of O(m³) eigendecomposition
        transition_covariance = ensure_symmetric(transition_covariance)
        transition_covariance = transition_covariance + np.eye(
            transition_covariance.shape[0], dtype=transition_covariance.dtype
        ) * reg
        
        observation_covariance = ensure_symmetric(observation_covariance)
        observation_covariance = observation_covariance + np.eye(
            observation_covariance.shape[0], dtype=observation_covariance.dtype
        ) * reg
        
        initial_state_covariance = ensure_symmetric(initial_state_covariance)
        initial_state_covariance = initial_state_covariance + np.eye(
            initial_state_covariance.shape[0], dtype=initial_state_covariance.dtype
        ) * reg
        
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
            If True, compute log-likelihood (expensive, O(T × m³)).
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
            return _filter(
                self._pykalman.transition_matrices,
                self._pykalman.observation_matrices,
                self._pykalman.transition_covariance,
                self._pykalman.observation_covariance,
                transition_offsets if transition_offsets is not None else np.zeros(self._pykalman.transition_matrices.shape[0]),
                observation_offsets if observation_offsets is not None else np.zeros(self._pykalman.observation_matrices.shape[0]),
                self._pykalman.initial_state_mean,
                self._pykalman.initial_state_covariance,
                observations
            )
        
        # Run filter
        predicted_state_means, predicted_state_covariances, _, filtered_state_means, filtered_state_covariances = run_filter()

        filter_time = time_module.time() - filter_start
        _logger.info(f"    Filter: Completed in {filter_time:.2f}s ({filter_time/T*1000:.2f}ms/timestep)")
        
        # Smooth to get smoothed states (also O(T × m³) - can be slow)
        _logger.info(f"    Smooth: Processing {T} timesteps (state_dim={m})...")
        smooth_start = time_module.time()
        
        def run_smooth():
            # Lightweight stabilization: add small diagonal regularization to prevent SVD failures
            # This is much cheaper than full eigendecomposition (O(m²) vs O(m³))
            from ..config.constants import MIN_EIGENVALUE
            base_regularization = max(1e-6, MIN_EIGENVALUE * 100)  # 1e-4 for stability
            
            # Adaptive regularization: check condition numbers and increase if needed
            # Investment dataset has larger state dimension (m=193) and worse conditioning (~204k)
            # Production has m=135 and conditioning ~131k
            # If condition number is very high (>1e5), increase regularization
            try:
                # Sample a few covariance matrices to check condition
                sample_indices = [0, len(predicted_state_covariances) // 2, len(predicted_state_covariances) - 1]
                sample_indices = [i for i in sample_indices if i < len(predicted_state_covariances)]
                max_cond = max([np.linalg.cond(predicted_state_covariances[i]) for i in sample_indices])
                
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
            
            # Apply stabilization to both predicted and filtered covariance matrices
            self._stabilize_covariance_matrices(predicted_state_covariances, regularization)
            self._stabilize_covariance_matrices(filtered_state_covariances, regularization)
            
            _logger.info("    Smooth: Covariance matrices stabilized, starting smoother")
            
            # Use internal functions for standard KalmanFilter
            return _smooth(
                self._pykalman.transition_matrices,
                filtered_state_means,
                filtered_state_covariances,
                predicted_state_means,
                predicted_state_covariances,
            )
        
        # Run smooth
        _logger.info("    Smooth: Starting smoother execution...")
        smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains = run_smooth()
        
        smooth_time = time_module.time() - smooth_start
        _logger.info(f"    Smooth: Completed in {smooth_time:.2f}s ({smooth_time/T*1000:.2f}ms/timestep)")
        
        # Compute lag-1 cross-covariances (needed for M-step)
        _logger.info(f"    Smooth-pair: Computing cross-covariances...")
        smooth_pair_start = time_module.time()
        # Use internal function for standard KalmanFilter
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
                
                loglik = self._pykalman.loglikelihood(observations_for_loglik)
                # Validate: log-likelihood should be finite and not exactly 0.0 (which indicates a bug)
                if not np.isfinite(loglik):
                    _logger.error(f"DFMKalmanFilter: Log-likelihood is not finite: {loglik}. This indicates numerical instability.")
                    loglik = float('-inf')
                elif loglik == 0.0:
                    _logger.warning(f"DFMKalmanFilter: Log-likelihood is exactly 0.0. This may indicate a bug in pykalman with masked arrays. "
                                  f"Trying with explicit NaN conversion...")
                    # Try again with explicit NaN conversion
                    obs_nan = np.where(np.ma.getmaskarray(observations) if isinstance(observations, np.ma.MaskedArray) else np.isnan(observations), np.nan, observations)
                    loglik = self._pykalman.loglikelihood(obs_nan)
                    if loglik == 0.0:
                        _logger.error(f"DFMKalmanFilter: Log-likelihood still 0.0 after NaN conversion. This is likely a pykalman bug.")
                        loglik = float('-inf')
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
    
