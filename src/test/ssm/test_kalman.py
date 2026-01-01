"""Tests for ssm.kalman module."""

import pytest
import numpy as np
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.numeric.stability import create_scaled_identity
from dfm_python.config.constants import DEFAULT_PROCESS_NOISE, DEFAULT_TRANSITION_COEF, DEFAULT_IDENTITY_SCALE
from dfm_python.utils.errors import ModelNotInitializedError


class TestDFMKalmanFilter:
    """Test suite for DFMKalmanFilter."""
    
    def test_dfm_kalman_filter_initialization(self):
        """Test DFMKalmanFilter can be initialized."""
        # Test initialization without parameters (lazy initialization)
        kf = DFMKalmanFilter()
        assert kf is not None
        assert kf._pykalman is None
        
        # Test initialization with all parameters
        m, n = 3, 5  # state dim, observation dim
        A = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf2 = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        assert kf2 is not None
        assert kf2._pykalman is not None
    
    def test_dfm_kalman_filter_predict(self):
        """Test DFMKalmanFilter prediction step."""
        # Setup filter with parameters
        m, n = 2, 3
        A = create_scaled_identity(m, DEFAULT_TRANSITION_COEF, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # Test filter() method
        T = 10
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        
        assert filtered_means is not None
        assert filtered_covs is not None
        assert filtered_means.shape == (T, m)
        assert filtered_covs.shape == (T, m, m)
        
        # Test that filter() raises error when not initialized
        kf_uninit = DFMKalmanFilter()
        with pytest.raises(ModelNotInitializedError, match="parameters not initialized"):
            kf_uninit.filter(observations)
    
    def test_dfm_kalman_filter_update(self):
        """Test DFMKalmanFilter update step."""
        # Test update_parameters() method
        m, n = 2, 3
        A = create_scaled_identity(m, DEFAULT_TRANSITION_COEF, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf = DFMKalmanFilter()
        assert kf._pykalman is None
        
        # Update parameters
        kf.update_parameters(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        assert kf._pykalman is not None
        
        # Verify parameters were set correctly
        assert np.allclose(kf._pykalman.transition_matrices, A)
        assert np.allclose(kf._pykalman.observation_matrices, C)
        assert np.allclose(kf._pykalman.transition_covariance, Q)
        assert np.allclose(kf._pykalman.observation_covariance, R)
        assert np.allclose(kf._pykalman.initial_state_mean, Z0)
        assert np.allclose(kf._pykalman.initial_state_covariance, V0)
        
        # Test that filter works after update
        T = 5
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        assert filtered_means.shape == (T, m)
        assert filtered_covs.shape == (T, m, m)

