"""Kalman filter wrapper for DFM using pykalman."""

from typing import Tuple, Optional, Dict, Any, Callable
import time as time_module
import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
from pykalman.sqrt import CholeskyKalmanFilter as PyCholeskyKalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair

from ..logger import get_logger, log_em_iteration, log_convergence
from ..utils.errors import ModelNotInitializedError
from ..utils.misc import get_config_attr
from ..config.types import FloatArray
from ..config.constants import DEFAULT_MIN_DELTA, DEFAULT_ZERO_VALUE
from ..numeric.stability import ensure_symmetric, ensure_covariance_stable, ensure_process_noise_stable

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
        initial_state_covariance: Optional[FloatArray] = None,
        use_cholesky: bool = False
    ) -> None:
        self._use_cholesky = use_cholesky
        self._pykalman = None
        # IMPORTANT:
        # Always go through update_parameters() so covariance stabilization is applied consistently.
        # Previously, the constructor path would pass raw covariances to pykalman (bypassing
        # symmetrization/regularization), which can later crash Cholesky-based smooth().
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
                initial_state_covariance=initial_state_covariance,
                # Strict PD is only needed when using Cholesky-based routines; doing this once
                # on initialization is cheap relative to EM and prevents save-time crashes.
                strict_pd=use_cholesky,
            )
    
    def update_parameters(
        self,
        transition_matrices: FloatArray,
        observation_matrices: FloatArray,
        transition_covariance: FloatArray,
        observation_covariance: FloatArray,
        initial_state_mean: FloatArray,
        initial_state_covariance: FloatArray,
        strict_pd: bool = False,
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
        strict_pd : bool, default False
            If True, apply O(m^3) stabilization to guarantee (near) positive definiteness.
            This is mainly required for Cholesky-based filtering/smoothing.
        """
        # Keep everything in float64 for numerical stability in large state spaces.
        # (Downstream pykalman / SciPy Cholesky is sensitive to tiny negative eigenvalues.)
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

        # Strict stabilization: guarantee (near) PD when needed.
        # We do this only when explicitly requested (e.g., Cholesky-based filter/smoother).
        if strict_pd:
            # Ensure Q (process noise) is bounded and PD; keep float64.
            transition_covariance = ensure_process_noise_stable(
                transition_covariance,
                min_eigenval=max(1e-10, float(MIN_EIGENVALUE)),
                warn=True,
                dtype=np.float64,
            )
            # Ensure R and V0 are PD; keep float64.
            observation_covariance = ensure_covariance_stable(
                observation_covariance,
                min_eigenval=max(1e-10, float(MIN_EIGENVALUE)),
            ).astype(np.float64)
            initial_state_covariance = ensure_covariance_stable(
                initial_state_covariance,
                min_eigenval=max(1e-10, float(MIN_EIGENVALUE)),
            ).astype(np.float64)
        
        if self._pykalman is None:
            filter_class = PyCholeskyKalmanFilter if self._use_cholesky else PyKalmanFilter
            self._pykalman = filter_class(
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
    
    def smooth(self, observations: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Run Kalman smoother."""
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        
        return self._pykalman.smooth(observations)
    
    def loglikelihood(self, observations: FloatArray) -> float:
        """Compute log-likelihood of observations."""
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        
        return self._pykalman.loglikelihood(observations)
    
    def filter_and_smooth(
        self,
        observations: FloatArray
    ) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
        """Run filter and smooth, return smoothed states, covariances, cross-covariances, and log-likelihood."""
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
            if self._use_cholesky:
                # Use public API for CholeskyKalmanFilter
                filtered_state_means, filtered_state_covariances = self._pykalman.filter(observations)
                # Approximate predicted states (needed for smooth_pair calculation)
                predicted_state_means = filtered_state_means.copy()
                predicted_state_covariances = filtered_state_covariances.copy()
                return predicted_state_means, predicted_state_covariances, None, filtered_state_means, filtered_state_covariances
            else:
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
        
        # Log filter results summary
        _logger.info(f"    Filter: Results - predicted_state_means shape: {predicted_state_means.shape}, "
                    f"predicted_state_covariances shape: {predicted_state_covariances.shape}")
        
        # Smooth to get smoothed states (also O(T × m³) - can be slow)
        _logger.info(f"    Smooth: Processing {T} timesteps (state_dim={m})...")
        smooth_start = time_module.time()
        
        def run_smooth():
            if self._use_cholesky:
                _logger.info("    Smooth: Using CholeskyKalmanFilter smooth() method")
                # Use public API for CholeskyKalmanFilter
                smoothed_state_means, smoothed_state_covariances = self._pykalman.smooth(observations)
                # Approximate kalman_smoothing_gains (needed for smooth_pair)
                kalman_smoothing_gains = np.array([np.eye(m) for _ in range(T)])
                return smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains
            else:
                # Lightweight stabilization: add small diagonal regularization to prevent SVD failures
                # This is much cheaper than full eigendecomposition (O(m²) vs O(m³))
                from ..config.constants import MIN_EIGENVALUE
                regularization = max(1e-6, MIN_EIGENVALUE * 100)  # 1e-4 for stability
                
                _logger.info(f"    Smooth: Stabilizing {len(predicted_state_covariances)} covariance matrices "
                            f"(regularization={regularization:.2e})")
                
                # Fast diagonal loading: just add regularization to diagonal (O(m²) per matrix)
                # Symmetrize and add diagonal regularization without expensive eigendecomposition
                # Use in-place diagonal modification for efficiency (avoids creating identity matrix in loop)
                for t in range(len(predicted_state_covariances)):
                    cov = predicted_state_covariances[t]
                    # Symmetrize and add regularization to diagonal (cheap: O(m²))
                    cov = ensure_symmetric(cov)
                    np.fill_diagonal(cov, np.diagonal(cov) + regularization)
                    predicted_state_covariances[t] = cov
                
                # Same for filtered_state_covariances
                for t in range(len(filtered_state_covariances)):
                    cov = filtered_state_covariances[t]
                    cov = ensure_symmetric(cov)
                    np.fill_diagonal(cov, np.diagonal(cov) + regularization)
                    filtered_state_covariances[t] = cov
                
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
        _logger.debug(f"    Smooth: Results - smoothed_state_means shape: {smoothed_state_means.shape}, "
                     f"smoothed_state_covariances shape: {smoothed_state_covariances.shape}")
        
        # Compute lag-1 cross-covariances (needed for M-step)
        _logger.info(f"    Smooth-pair: Computing cross-covariances...")
        smooth_pair_start = time_module.time()
        if self._use_cholesky:
            # For CholeskyKalmanFilter, compute cross-covariances directly from smoothed results
            # Simplified computation: V_{t,t-1} ≈ A @ V_{t-1,t-1} (approximation)
            A = self._pykalman.transition_matrices
            sigma_pair_smooth = np.zeros((T-1, m, m), dtype=smoothed_state_covariances.dtype)
            for t in range(T-1):
                sigma_pair_smooth[t] = A @ smoothed_state_covariances[t]
        else:
            # Use internal function for standard KalmanFilter
            sigma_pair_smooth = _smooth_pair(smoothed_state_covariances, kalman_smoothing_gains)
        smooth_pair_time = time_module.time() - smooth_pair_start
        _logger.info(f"    Smooth-pair: Completed in {smooth_pair_time:.2f}s")
        
        # Compute log-likelihood
        _logger.info(f"    Log-likelihood: Computing...")
        loglik_start = time_module.time()
        try:
            loglik = self._pykalman.loglikelihood(observations)
            # Validate: log-likelihood should be finite
            if not np.isfinite(loglik):
                _logger.error(f"DFMKalmanFilter: Log-likelihood is not finite: {loglik}. This indicates numerical instability.")
                loglik = float('-inf')
        except (ValueError, RuntimeError, AttributeError) as e:
            _logger.error(f"DFMKalmanFilter: Failed to compute log-likelihood: {e}. Using -inf (will break convergence checks).")
            _logger.debug(f"DFMKalmanFilter: Full exception traceback for loglikelihood computation failure:", exc_info=True)
            loglik = float('-inf')  # Use -inf instead of 0.0 (0.0 would break convergence checks)
        loglik_time = time_module.time() - loglik_start
        _logger.info(f"    Log-likelihood: Completed in {loglik_time:.2f}s, value={loglik:.2e}")
        
        # Log detailed timing breakdown (for debugging/performance analysis)
        total_e_step_time = filter_time + smooth_time + smooth_pair_time + loglik_time
        if total_e_step_time > 5.0:  # Only log if E-step takes significant time
            _logger.debug(f"E-step breakdown: Filter={filter_time:.2f}s ({100*filter_time/total_e_step_time:.1f}%), "
                        f"Smooth={smooth_time:.2f}s ({100*smooth_time/total_e_step_time:.1f}%), "
                        f"Pair={smooth_pair_time:.2f}s ({100*smooth_pair_time/total_e_step_time:.1f}%), "
                        f"Loglik={loglik_time:.2f}s ({100*loglik_time/total_e_step_time:.1f}%)")
        
        return smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth, loglik
    
    def em(
        self,
        X: FloatArray,
        initial_params: Dict[str, FloatArray],
        max_iter: int = 200,
        threshold: float = 1e-4,
        blocks: Optional[FloatArray] = None,
        r: Optional[FloatArray] = None,
        p: Optional[int] = None,
        p_plus_one: Optional[int] = None,
        R_mat: Optional[FloatArray] = None,
        q: Optional[FloatArray] = None,
        n_clock_freq: Optional[int] = None,
        n_slower_freq: Optional[int] = None,
        idio_indicator: Optional[FloatArray] = None,
        tent_weights_dict: Optional[Dict[str, FloatArray]] = None,
        config: Optional[Any] = None,
        checkpoint_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Run full EM algorithm until convergence with custom M-step.
        
        This method orchestrates the full EM algorithm, using pykalman for the E-step
        and custom constrained M-step updates that preserve block structure, mixed-frequency
        constraints, and idiosyncratic component structure.
        
        Parameters
        ----------
        X : np.ndarray
            Data array (T x N)
        initial_params : dict
            Initial parameters with keys: 'A', 'C', 'Q', 'R', 'Z_0', 'V_0'
        max_iter : int, default 200
            Maximum number of EM iterations
        threshold : float, default 1e-4
            Convergence threshold (relative change in log-likelihood)
        blocks : np.ndarray, optional
            Block structure array (N x n_blocks). If provided, uses blocked updates.
        r : np.ndarray, optional
            Number of factors per block (n_blocks,). Required if blocks is provided.
        p : int, optional
            VAR lag order. Required if blocks is provided.
        p_plus_one : int, optional
            p + 1 (state dimension per factor). Required if blocks is provided.
        R_mat : np.ndarray, optional
            Tent kernel constraint matrix. Required for mixed-frequency data.
        q : np.ndarray, optional
            Tent kernel constraint vector. Required for mixed-frequency data.
        n_clock_freq : int, optional
            Number of clock-frequency series. Required if blocks is provided.
        n_slower_freq : int, optional
            Number of slower-frequency series. Required for mixed-frequency data.
        idio_indicator : np.ndarray, optional
            Idiosyncratic component indicator (N,). Required if blocks is provided.
        tent_weights_dict : dict, optional
            Dictionary mapping frequency pairs to tent weights.
        config : EMConfig, optional
            EM configuration. If None, uses defaults from functional.em.
            
        Returns
        -------
        dict
            Final state with keys:
            - 'A', 'C', 'Q', 'R', 'Z_0', 'V_0': Updated parameters
            - 'loglik': Final log-likelihood
            - 'num_iter': Number of iterations completed
            - 'converged': Whether convergence was achieved
            - 'change': Final relative change in log-likelihood
        """
        # Import here to avoid circular dependency
        from ..functional.em import em_step, EMConfig, _DEFAULT_EM_CONFIG
        from ..config.schema.block import BlockStructure
        
        if config is None:
            config = _DEFAULT_EM_CONFIG
        
        # Initialize parameters
        A = initial_params['A']
        C = initial_params['C']
        Q = initial_params['Q']
        R = initial_params['R']
        Z_0 = initial_params['Z_0']
        V_0 = initial_params['V_0']
        
        # Update filter with initial parameters
        self.update_parameters(A, C, Q, R, Z_0, V_0)
        
        # Initialize state
        previous_loglik = float('-inf')
        num_iter = 0
        converged = False
        loglik = float('-inf')
        change = DEFAULT_ZERO_VALUE
        
        # Track timing for progress estimation
        em_start_time = time_module.time()
        iteration_times = []
        
        # Create BlockStructure if block parameters are provided
        block_structure = None
        if blocks is not None and r is not None and p is not None and p_plus_one is not None and n_clock_freq is not None and idio_indicator is not None:
            block_structure = BlockStructure(
                blocks=blocks,
                r=r,
                p=p,
                p_plus_one=p_plus_one,
                n_clock_freq=n_clock_freq,
                idio_indicator=idio_indicator,
                R_mat=R_mat,
                q=q,
                n_slower_freq=n_slower_freq,
                tent_weights_dict=tent_weights_dict
            )
            # Cache starts empty (None) by default, will be computed once on first M-step and reused across iterations
        
        # Start EM loop - log initialization using custom training logger
        _logger.info(f"Starting EM algorithm: max_iter={max_iter}, threshold={threshold:.2e}")
        if block_structure is not None:
            total_factors = int(np.sum(r)) if r is not None else 0
            _logger.info(f"  Block structure: {blocks.shape[1] if blocks is not None else 0} blocks, {total_factors} factors")
        if R_mat is not None:
            _logger.info(f"  Mixed-frequency: tent kernel constraints enabled (R_mat shape: {R_mat.shape})")
        
        # EM loop
        while num_iter < max_iter and not converged:
            iter_start_time = time_module.time()
            
            # Log iteration start for first few iterations to show progress
            if num_iter < 3:
                _logger.info(f"Starting iteration {num_iter + 1}/{max_iter}...")
            
            try:
                # E-step + M-step
                A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, _ = em_step(
                    X, A, C, Q, R, Z_0, V_0, kalman_filter=self, config=config,
                    block_structure=block_structure, num_iter=num_iter
                )
            except Exception as e:
                _logger.error(f"EM step failed at iteration {num_iter + 1}: {e}", exc_info=True)
                _logger.error(f"  Current parameters shapes - A: {A.shape}, C: {C.shape}, Q: {Q.shape}, R: {R.shape}")
                _logger.error(f"  Data shape: {X.shape}, Block structure: {block_structure is not None}")
                raise RuntimeError(f"EM algorithm failed at iteration {num_iter + 1}: {e}") from e
            
            # Check for NaN/Inf (early stopping)
            if not all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                       for p in [A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik]):
                _logger.error(f"EM: NaN/Inf detected at iteration {num_iter + 1}, stopping")
                _logger.error(f"  Parameter shapes - A: {A_new.shape}, C: {C_new.shape}, Q: {Q_new.shape}, R: {R_new.shape}")
                _logger.error(f"  Loglik: {loglik}, isfinite: {np.isfinite(loglik) if isinstance(loglik, (int, float, np.number)) else 'N/A'}")
                break
            
            # Check for extremely negative log-likelihood (numerical instability indicator)
            if num_iter == 0 and loglik < -1e10:
                _logger.warning(f"Extremely negative initial log-likelihood: {loglik:.2e}. "
                              f"This may indicate numerical instability or data scaling issues.")
                _logger.warning(f"  Check: data scaling, initialization parameters, or covariance matrix conditioning.")
            
            # Update parameters
            A, C, Q, R, Z_0, V_0 = A_new, C_new, Q_new, R_new, Z_0_new, V_0_new
            self.update_parameters(A, C, Q, R, Z_0, V_0)
            
            # Check convergence using relative change in log-likelihood
            min_iterations = get_config_attr(config, 'min_iterations_for_convergence_check', 1)
            
            if num_iter >= min_iterations:
                if previous_loglik != float('-inf') and np.isfinite(previous_loglik) and abs(previous_loglik) > 1e-10:
                    # Relative change: |(loglik - previous_loglik) / previous_loglik|
                    # For negative log-likelihoods, we compute relative change based on magnitude
                    change = abs((loglik - previous_loglik) / previous_loglik)
                else:
                    # Fallback to absolute change if previous_loglik is invalid or too small
                    change = abs(loglik - previous_loglik) if np.isfinite(loglik) and np.isfinite(previous_loglik) else float('inf')
                converged = change < threshold
            else:
                # Before min_iterations, just track absolute change
                change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') and np.isfinite(loglik) and np.isfinite(previous_loglik) else DEFAULT_ZERO_VALUE
            
            previous_loglik = loglik
            num_iter += 1
            
            # Track iteration time
            iter_time = time_module.time() - iter_start_time
            iteration_times.append(iter_time)
            # Keep only last 5 iteration times for averaging
            if len(iteration_times) > 5:
                iteration_times.pop(0)
            avg_iter_time = sum(iteration_times) / len(iteration_times)
            
            # Calculate progress and time estimates
            elapsed_time = time_module.time() - em_start_time
            progress_pct = (num_iter / max_iter) * 100
            
            # Estimate remaining time based on average iteration time
            remaining_iters = max_iter - num_iter if not converged else 0
            estimated_remaining = avg_iter_time * remaining_iters if remaining_iters > 0 else 0
            
            # Format time estimates
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.0f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    return f"{hours}h{minutes}m"
            
            # Log progress using custom training logger
            # Changed default to 1 to log every iteration for better convergence monitoring
            progress_interval = get_config_attr(config, 'progress_log_interval', 1)
            # Log every iteration (progress_interval=1), or if converged
            should_log = (num_iter % progress_interval == 0) or converged
            
            if should_log:
                # Build progress message
                progress_msg = f"[{progress_pct:.1f}%]"
                if elapsed_time > 0:
                    elapsed_str = format_time(elapsed_time)
                    if remaining_iters > 0 and len(iteration_times) >= 2:
                        remaining_str = format_time(estimated_remaining)
                        progress_msg += f" Elapsed: {elapsed_str}, Est. remaining: {remaining_str}"
                    else:
                        progress_msg += f" Elapsed: {elapsed_str}"
                
                # Use custom training logger convenience function
                log_em_iteration(
                    iteration=num_iter,
                    loglik=loglik,
                    delta=change if change > 0 else None,
                    max_iter=max_iter,
                    converged="✓" if converged else ""
                )
                
                # Log additional progress info
                if avg_iter_time > 0:
                    _logger.info(f"  Progress: {progress_msg}, Iteration time: {iter_time:.1f}s (avg: {avg_iter_time:.1f}s)")
            
            # Periodic checkpoint saving (every 5 iterations, overriding previous)
            if checkpoint_callback is not None and num_iter % 5 == 0 and num_iter > 0:
                try:
                    current_state = {
                        'A': A,
                        'C': C,
                        'Q': Q,
                        'R': R,
                        'Z_0': Z_0,
                        'V_0': V_0,
                        'loglik': loglik,
                        'num_iter': num_iter,
                        'converged': converged,
                        'change': change
                    }
                    _logger.info(f"Saving checkpoint at iteration {num_iter}...")
                    checkpoint_callback(num_iter, current_state)
                except Exception as e:
                    _logger.warning(f"Checkpoint callback failed at iteration {num_iter}: {e}", exc_info=True)
        
        # Final status using custom training logger
        log_convergence(
            converged=converged,
            num_iter=num_iter,
            final_loglik=loglik if np.isfinite(loglik) else None,
            reason="converged" if converged else f"max_iterations_reached (change: {change:.2e})",
            model_type="dfm"
        )
        
        return {
            'A': A,
            'C': C,
            'Q': Q,
            'R': R,
            'Z_0': Z_0,
            'V_0': V_0,
            'loglik': loglik,
            'num_iter': num_iter,
            'converged': converged,
            'change': change
        }
