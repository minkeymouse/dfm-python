"""Kalman filter implementation for DFM using pykalman.

This module provides DFMKalmanFilter, a wrapper around pykalman for DFM models.
All DFM-related Kalman filtering uses this class instead of PyTorch-based implementations.
"""

import numpy as np
from typing import Tuple, Optional
from pykalman import KalmanFilter as PyKalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair

from ..logger import get_logger

_logger = get_logger(__name__)


class DFMKalmanFilter:
    """Kalman filter wrapper for DFM using pykalman.
    
    This class provides a clean interface to pykalman for DFM models,
    handling parameter updates and providing filter/smooth operations.
    
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
        transition_matrices: Optional[np.ndarray] = None,
        observation_matrices: Optional[np.ndarray] = None,
        transition_covariance: Optional[np.ndarray] = None,
        observation_covariance: Optional[np.ndarray] = None,
        initial_state_mean: Optional[np.ndarray] = None,
        initial_state_covariance: Optional[np.ndarray] = None
    ):
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
        transition_matrices: np.ndarray,
        observation_matrices: np.ndarray,
        transition_covariance: np.ndarray,
        observation_covariance: np.ndarray,
        initial_state_mean: np.ndarray,
        initial_state_covariance: np.ndarray
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
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            raise ValueError("DFMKalmanFilter: Parameters not initialized. Call update_parameters() first.")
        
        return self._pykalman.filter(observations)
    
    def smooth(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            raise ValueError("DFMKalmanFilter: Parameters not initialized. Call update_parameters() first.")
        
        return self._pykalman.smooth(observations)
    
    def loglikelihood(self, observations: np.ndarray) -> float:
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
            raise ValueError("DFMKalmanFilter: Parameters not initialized. Call update_parameters() first.")
        
        return self._pykalman.loglikelihood(observations)
    
    def filter_and_smooth(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
            raise ValueError("DFMKalmanFilter: Parameters not initialized. Call update_parameters() first.")
        
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
        except Exception as e:
            _logger.warning(f"DFMKalmanFilter: Failed to compute log-likelihood: {e}. Using 0.0.")
            loglik = 0.0
        
        return smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth, loglik
