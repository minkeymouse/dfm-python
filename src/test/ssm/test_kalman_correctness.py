"""Comprehensive correctness tests for Kalman filter.

These tests verify mathematical correctness of Kalman filter operations.
"""

import pytest
import numpy as np
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.numeric.stability import create_scaled_identity
from dfm_python.config.constants import DEFAULT_PROCESS_NOISE, DEFAULT_TRANSITION_COEF, DEFAULT_IDENTITY_SCALE


class TestKalmanMathematicalCorrectness:
    """Mathematical correctness tests for Kalman filter."""
    
    def test_covariance_symmetry_after_filter(self):
        """Test 1.1: Covariance matrices must be symmetric after filtering."""
        m, n = 3, 5
        T = 20
        
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        
        # Test symmetry: ||P_t - P_t^T|| < ε
        epsilon = 1e-6
        for t in range(min(10, T)):
            P_t = filtered_covs[t]
            asymmetry = np.linalg.norm(P_t - P_t.T)
            assert asymmetry < epsilon, f"Filtered covariance not symmetric at t={t}: ||P - P^T|| = {asymmetry}"
    
    def test_covariance_symmetry_after_smooth(self):
        """Test 1.1: Covariance matrices must be symmetric after smoothing."""
        m, n = 3, 5
        T = 20
        
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        observations = np.random.randn(T, n)
        smoothed_means, smoothed_covs, cross_covs, loglik = kf.filter_and_smooth(observations)
        
        # Test symmetry: ||P_t - P_t^T|| < ε
        epsilon = 1e-6
        for t in range(min(10, T)):
            P_t = smoothed_covs[t]
            asymmetry = np.linalg.norm(P_t - P_t.T)
            assert asymmetry < epsilon, f"Smoothed covariance not symmetric at t={t}: ||P - P^T|| = {asymmetry}"
    
    def test_cross_covariance_structure(self):
        """Test 1.1: Cross-covariances must have correct structure."""
        m, n = 3, 5
        T = 20
        
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        observations = np.random.randn(T, n)
        smoothed_means, smoothed_covs, cross_covs, loglik = kf.filter_and_smooth(observations)
        
        # Verify cross_covs shape and finiteness
        assert cross_covs.shape == (T, m, m), f"Cross-cov shape wrong: {cross_covs.shape}, expected ({T}, {m}, {m})"
        assert np.all(np.isfinite(cross_covs)), "Cross-covariances contain non-finite values"
        
        # Verify cross-covs are symmetric (they should be)
        for t in range(min(5, T)):
            cross_t = cross_covs[t]
            asymmetry = np.linalg.norm(cross_t - cross_t.T)
            assert asymmetry < 1e-6, f"Cross-cov not symmetric at t={t}"
    
    def test_psd_after_filter(self):
        """Test 4.1: Filtered covariances must be PSD."""
        m, n = 3, 5
        T = 20
        
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        
        # Check PSD: min(eig(P_t)) ≥ -ε
        epsilon = 1e-8
        for t in range(min(10, T)):
            P_t = filtered_covs[t]
            eigvals = np.linalg.eigvalsh(P_t)
            min_eig = np.min(eigvals)
            assert min_eig >= -epsilon, f"Filtered covariance not PSD at t={t}: min(eig) = {min_eig}"
    
    def test_psd_after_smooth(self):
        """Test 4.1: Smoothed covariances must be PSD."""
        m, n = 3, 5
        T = 20
        
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        observations = np.random.randn(T, n)
        smoothed_means, smoothed_covs, cross_covs, loglik = kf.filter_and_smooth(observations)
        
        # Check PSD: min(eig(P_t)) ≥ -ε
        epsilon = 1e-8
        for t in range(min(10, T)):
            P_t = smoothed_covs[t]
            eigvals = np.linalg.eigvalsh(P_t)
            min_eig = np.min(eigvals)
            assert min_eig >= -epsilon, f"Smoothed covariance not PSD at t={t}: min(eig) = {min_eig}"
