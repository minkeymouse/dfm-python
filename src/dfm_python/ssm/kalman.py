"""Kalman filter implementation for DFM using pykalman.

This module provides DFMKalmanFilter, a wrapper around pykalman for DFM models.
All DFM-related Kalman filtering uses this class instead of PyTorch-based implementations.

**Note**: This wrapper provides both E-step (Kalman filter/smoother) and full EM algorithm.
We do NOT use pykalman's built-in `em()` method because it doesn't handle:
- Block structure preservation
- Mixed-frequency constraints (tent kernel aggregation)
- Idiosyncratic component structure

Instead, we use pykalman for E-step and implement custom M-step with `em()` method.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair

from ..logger import get_logger
from ..utils.errors import ModelNotInitializedError
from ..utils.helper import get_config_attr
from ..config.types import FloatArray
from ..config.constants import DEFAULT_MIN_DELTA, DEFAULT_ZERO_VALUE

_logger = get_logger(__name__)


class DFMKalmanFilter:
    """Kalman filter wrapper for DFM using pykalman with custom EM algorithm.
    
    This class provides a clean interface to pykalman for both E-step and full EM algorithm.
    It handles parameter updates and provides filter/smooth operations.
    
    **Why not use pykalman's `kf.em()` directly?**
    
    pykalman's built-in EM does unconstrained parameter updates that would:
    1. Destroy block structure (factors organized in blocks)
    2. Break mixed-frequency constraints (tent kernel aggregation for quarterly series)
    3. Ignore idiosyncratic component structure
    
    **Our approach:**
    - Use pykalman for E-step: `filter()`, `smooth()`, `filter_and_smooth()`
    - Use custom M-step: `em()` method implements constrained updates that preserve structure
    
    Parameters
    ----------
    transition_matrices : np.ndarray, optional
        Transition matrix A (m x m)
    observation_matrices : np.ndarray, optional
        Observation matrix C (N x m)
    transition_covariance : np.ndarray, optional
        Process noise covariance Q (m x m)
    observation_covariance : np.ndarray, optional
        Observation noise covariance R (N x N)
    initial_state_mean : np.ndarray, optional
        Initial state mean Z_0 (m,)
    initial_state_covariance : np.ndarray, optional
        Initial state covariance V_0 (m x m)
    """
    
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
        if all(p is not None for p in [
            transition_matrices, observation_matrices,
            transition_covariance, observation_covariance,
            initial_state_mean, initial_state_covariance
        ]):
            self._pykalman = PyKalmanFilter(
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
    
    def smooth(self, observations: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Run Kalman smoother.
        
        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) or masked array
            
        Returns
        -------
        smoothed_state_means : np.ndarray
            Smoothed state means (T x m)
        smoothed_state_covariances : np.ndarray
            Smoothed state covariances (T x m x m)
        """
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        
        return self._pykalman.smooth(observations)
    
    def loglikelihood(self, observations: FloatArray) -> float:
        """Compute log-likelihood of observations.
        
        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) or masked array
            
        Returns
        -------
        float
            Log-likelihood value
        """
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
        """Run filter and smooth, returning all necessary outputs for EM step.
        
        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) or masked array
            
        Returns
        -------
        smoothed_state_means : np.ndarray
            Smoothed state means (T x m)
        smoothed_state_covariances : np.ndarray
            Smoothed state covariances (T x m x m)
        sigma_pair_smooth : np.ndarray
            Lag-1 cross-covariances (T x m x m)
        loglik : float
            Log-likelihood value
        """
        if self._pykalman is None:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. "
                "Call update_parameters() first."
            )
        
        # Get filtered states first (needed for smoother)
        transition_offsets = getattr(self._pykalman, 'transition_offsets', None)
        observation_offsets = getattr(self._pykalman, 'observation_offsets', None)
        
        (
            predicted_state_means,
            predicted_state_covariances,
            _,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
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
        
        # Smooth to get smoothed states
        smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains = _smooth(
            self._pykalman.transition_matrices,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances,
        )
        
        # Compute lag-1 cross-covariances (needed for M-step)
        sigma_pair_smooth = _smooth_pair(smoothed_state_covariances, kalman_smoothing_gains)
        
        # Compute log-likelihood
        try:
            loglik = self._pykalman.loglikelihood(observations)
        except (ValueError, RuntimeError, AttributeError) as e:
            _logger.warning(f"DFMKalmanFilter: Failed to compute log-likelihood: {e}. Using {DEFAULT_ZERO_VALUE}.")
            loglik = DEFAULT_ZERO_VALUE
        
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
        config: Optional[Any] = None
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
        
        # EM loop
        while num_iter < max_iter and not converged:
            # E-step + M-step
            A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, _ = em_step(
                X, A, C, Q, R, Z_0, V_0, kalman_filter=self, config=config,
                block_structure=block_structure
            )
            
            # Check for NaN/Inf (early stopping)
            if not all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                       for p in [A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik]):
                _logger.error(f"EM: NaN/Inf at iteration {num_iter + 1}, stopping")
                break
            
            # Update parameters
            A, C, Q, R, Z_0, V_0 = A_new, C_new, Q_new, R_new, Z_0_new, V_0_new
            self.update_parameters(A, C, Q, R, Z_0, V_0)
            
            # Check convergence (relative change in log-likelihood)
            min_iterations = get_config_attr(config, 'min_iterations_for_convergence_check', 1)
            small_loglik_threshold = get_config_attr(config, 'small_loglik_threshold', DEFAULT_MIN_DELTA)
            
            if num_iter >= min_iterations:
                if abs(previous_loglik) < small_loglik_threshold:
                    change = abs(loglik - previous_loglik)
                else:
                    change = abs((loglik - previous_loglik) / previous_loglik) if previous_loglik != DEFAULT_ZERO_VALUE else abs(loglik - previous_loglik)
                converged = change < threshold
            else:
                change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else DEFAULT_ZERO_VALUE
            
            previous_loglik = loglik
            num_iter += 1
            
            # Log progress
            progress_interval = get_config_attr(config, 'progress_log_interval', 10)
            if num_iter % progress_interval == 0 or num_iter == 1:
                status = " ✓" if converged else ""
                _logger.info(f"EM iteration {num_iter}/{max_iter}: loglik={loglik:.4f}, change={change:.2e}{status}")
        
        # Final status
        if converged:
            _logger.info(f"✓ EM converged after {num_iter} iterations (loglik: {loglik:.6f})")
        else:
            _logger.warning(f"⚠ EM stopped after {num_iter} iterations (loglik: {loglik:.6f}, change: {change:.2e})")
        
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
