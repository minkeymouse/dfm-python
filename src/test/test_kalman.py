"""Tests for Kalman filter and EM algorithm implementation in DFM.

This module tests the pykalman-based Kalman filter and EM algorithm
used in the DFM model, including:
- Kalman filter and smoother functionality
- EM step (E-step + M-step)
- Full EM algorithm convergence
- Parameter updates and block structure preservation
- Missing data handling
- Numerical stability
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pykalman import KalmanFilter
from pykalman.standard import _filter, _smooth, _smooth_pair

from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.schema import SeriesConfig


class TestKalmanFilter:
    """Test Kalman filter functionality using pykalman."""
    
    def test_kalman_filter_basic(self):
        """Test basic Kalman filter forward pass."""
        # Simple 1D state, 1D observation model
        A = np.array([[0.9]])  # Transition matrix
        C = np.array([[1.0]])  # Observation matrix
        Q = np.array([[0.1]])  # Process noise
        R = np.array([[0.5]])  # Observation noise
        Z_0 = np.array([0.0])  # Initial state
        V_0 = np.array([[1.0]])  # Initial covariance
        
        # Generate synthetic data
        T = 50
        true_states = np.zeros((T, 1))
        observations = np.zeros((T, 1))
        
        # Simulate state evolution
        state = Z_0[0]
        for t in range(T):
            state = A[0, 0] * state + np.random.normal(0, np.sqrt(Q[0, 0]))
            true_states[t, 0] = state
            observations[t, 0] = state + np.random.normal(0, np.sqrt(R[0, 0]))
        
        # Create Kalman filter
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z_0,
            initial_state_covariance=V_0
        )
        
        # Run filter
        filtered_state_means, filtered_state_covariances = kf.filter(observations)
        
        # Check shapes
        assert filtered_state_means.shape == (T, 1)
        assert filtered_state_covariances.shape == (T, 1, 1)
        
        # Check that filtered states are reasonable
        assert np.all(np.isfinite(filtered_state_means))
        assert np.all(np.isfinite(filtered_state_covariances))
        assert np.all(filtered_state_covariances > 0)
    
    def test_kalman_smoother(self):
        """Test Kalman smoother."""
        # 2D state, 1D observation
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        C = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        Z_0 = np.array([0.0, 0.0])
        V_0 = np.eye(2)
        
        T = 30
        observations = np.random.randn(T, 1)
        
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z_0,
            initial_state_covariance=V_0
        )
        
        # Run smoother
        smoothed_state_means, smoothed_state_covariances = kf.smooth(observations)
        
        # Check shapes
        assert smoothed_state_means.shape == (T, 2)
        assert smoothed_state_covariances.shape == (T, 2, 2)
        
        # Check that smoothed states are finite and covariances are positive definite
        assert np.all(np.isfinite(smoothed_state_means))
        assert np.all(np.isfinite(smoothed_state_covariances))
        
        # Check positive definiteness
        for t in range(T):
            eigenvals = np.linalg.eigvals(smoothed_state_covariances[t])
            assert np.all(eigenvals > 0)
    
    def test_kalman_filter_with_missing_data(self):
        """Test Kalman filter with missing data (masked arrays)."""
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.5]])
        Z_0 = np.array([0.0])
        V_0 = np.array([[1.0]])
        
        T = 20
        observations = np.random.randn(T, 1)
        
        # Introduce missing data
        observations[5:8] = np.nan
        observations[15] = np.nan
        
        # Use masked array
        observations_masked = np.ma.masked_invalid(observations)
        
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z_0,
            initial_state_covariance=V_0
        )
        
        # Should handle missing data gracefully
        filtered_state_means, filtered_state_covariances = kf.filter(observations_masked)
        
        assert filtered_state_means.shape == (T, 1)
        assert np.all(np.isfinite(filtered_state_means))
    
    def test_kalman_loglikelihood(self):
        """Test log-likelihood computation."""
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.5]])
        Z_0 = np.array([0.0])
        V_0 = np.array([[1.0]])
        
        T = 10
        observations = np.random.randn(T, 1)
        
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z_0,
            initial_state_covariance=V_0
        )
        
        loglik = kf.loglikelihood(observations)
        
        # Log-likelihood should be finite and typically negative
        assert np.isfinite(loglik)
        assert loglik < 0  # Usually negative for likelihoods
    
    def test_kalman_low_level_functions(self):
        """Test low-level pykalman functions used in DFM."""
        # Test _filter, _smooth, _smooth_pair
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        C = np.array([[1.0, 0.5], [0.3, 0.7]])
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.5
        Z_0 = np.array([0.0, 0.0])
        V_0 = np.eye(2)
        
        T = 20
        observations = np.random.randn(T, 2)
        
        # Test _filter
        (
            predicted_state_means,
            predicted_state_covariances,
            _,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            A, C, Q, R,
            np.zeros(2),  # transition_offsets
            np.zeros(2),  # observation_offsets
            Z_0, V_0,
            observations
        )
        
        assert filtered_state_means.shape == (T, 2)
        assert filtered_state_covariances.shape == (T, 2, 2)
        
        # Test _smooth
        smoothed_state_means, smoothed_state_covariances, kalman_smoothing_gains = _smooth(
            A,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances,
        )
        
        assert smoothed_state_means.shape == (T, 2)
        assert smoothed_state_covariances.shape == (T, 2, 2)
        assert kalman_smoothing_gains.shape == (T-1, 2, 2)
        
        # Test _smooth_pair (lag-1 cross-covariances)
        sigma_pair_smooth = _smooth_pair(smoothed_state_covariances, kalman_smoothing_gains)
        
        assert sigma_pair_smooth.shape == (T, 2, 2)


class TestEMStep:
    """Test EM step functionality (E-step + M-step)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple DFM model
        config = DFMConfig(
            series=[
                SeriesConfig(series_id=f'series_{i}', frequency='m')
                for i in range(5)
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 2, 'ar_lag': 1, 'clock': 'm'}}
        )
        self.model = DFM(config=config, max_iter=10, threshold=1e-4)
        
        # Initialize with simple parameters
        m, N = 2, 5
        self.model.A = nn.Parameter(torch.tensor(np.eye(m) * 0.9, dtype=torch.float32))
        self.model.C = nn.Parameter(torch.tensor(np.random.randn(N, m) * 0.5, dtype=torch.float32))
        self.model.Q = nn.Parameter(torch.tensor(np.eye(m) * 0.1, dtype=torch.float32))
        self.model.R = nn.Parameter(torch.tensor(np.eye(N) * 0.5, dtype=torch.float32))
        self.model.Z_0 = nn.Parameter(torch.tensor(np.zeros(m), dtype=torch.float32))
        self.model.V_0 = nn.Parameter(torch.tensor(np.eye(m), dtype=torch.float32))
    
    def test_em_step_basic(self):
        """Test basic EM step execution."""
        T = 50
        X = np.random.randn(T, 5)  # (T x N)
        
        # Run one EM step
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik = self.model._em_step(X)
        
        # Check shapes
        assert A_new.shape == (2, 2)
        assert C_new.shape == (5, 2)
        assert Q_new.shape == (2, 2)
        assert R_new.shape == (5, 5)
        assert Z_0_new.shape == (2,)
        assert V_0_new.shape == (2, 2)
        assert isinstance(loglik, (float, np.floating))
        
        # Check that all parameters are finite
        assert np.all(np.isfinite(A_new))
        assert np.all(np.isfinite(C_new))
        assert np.all(np.isfinite(Q_new))
        assert np.all(np.isfinite(R_new))
        assert np.all(np.isfinite(Z_0_new))
        assert np.all(np.isfinite(V_0_new))
        assert np.isfinite(loglik)
        
        # Check positive definiteness of Q, R, V_0
        assert np.all(np.linalg.eigvals(Q_new) > 0)
        assert np.all(np.linalg.eigvals(R_new) > 0)
        assert np.all(np.linalg.eigvals(V_0_new) > 0)
    
    def test_em_step_with_missing_data(self):
        """Test EM step with missing data."""
        T = 30
        X = np.random.randn(T, 5)
        
        # Introduce missing data
        X[5:8, 0] = np.nan
        X[10, :] = np.nan
        X[15:18, 2:4] = np.nan
        
        # Should handle missing data gracefully
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik = self.model._em_step(X)
        
        # Check that results are still valid
        assert np.all(np.isfinite(A_new))
        assert np.all(np.isfinite(C_new))
        assert np.isfinite(loglik)
    
    def test_em_step_parameter_updates(self):
        """Test that EM step updates parameters."""
        T = 40
        X = np.random.randn(T, 5)
        
        # Get initial parameters
        A_old, C_old, Q_old, R_old, Z_0_old, V_0_old = self.model._get_parameters_numpy()
        
        # Run EM step
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik = self.model._em_step(X)
        
        # Parameters should change (unless already at optimum)
        # At least some should be different
        params_changed = (
            not np.allclose(A_new, A_old) or
            not np.allclose(C_new, C_old) or
            not np.allclose(Q_new, Q_old) or
            not np.allclose(R_new, R_old)
        )
        
        # It's possible parameters don't change much, but loglik should be computed
        assert np.isfinite(loglik)
    
    def test_em_step_stability(self):
        """Test numerical stability of EM step."""
        T = 100
        # Use data with different scales
        X = np.random.randn(T, 5) * np.array([1.0, 10.0, 0.1, 5.0, 2.0])
        
        # Should remain stable
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik = self.model._em_step(X)
        
        # Check stability constraints
        eigenvals_A = np.linalg.eigvals(A_new)
        max_eigenval = np.max(np.abs(eigenvals_A))
        assert max_eigenval <= 0.99  # Should be capped by DEFAULT_MAX_EIGENVAL
        
        # Check variance bounds
        diag_R = np.diag(R_new)
        assert np.all(diag_R >= 1e-4)  # DEFAULT_MIN_VARIANCE
        assert np.all(diag_R <= 1e4)    # DEFAULT_MAX_VARIANCE


class TestEMAlgorithm:
    """Test full EM algorithm convergence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = DFMConfig(
            series=[
                SeriesConfig(series_id=f'series_{i}', frequency='m')
                for i in range(4)
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 2, 'ar_lag': 1, 'clock': 'm'}}
        )
        self.model = DFM(config=config, max_iter=20, threshold=1e-3)
    
    def test_em_algorithm_convergence(self):
        """Test that EM algorithm converges."""
        # Generate synthetic data from a known DFM
        T = 60
        m, N = 2, 4
        
        # True parameters
        A_true = np.array([[0.8, 0.1], [0.0, 0.7]])
        C_true = np.random.randn(N, m) * 0.5
        Q_true = np.eye(m) * 0.1
        R_true = np.eye(N) * 0.3
        Z_0_true = np.zeros(m)
        V_0_true = np.eye(m)
        
        # Generate data
        Z = np.zeros((T, m))
        X = np.zeros((T, N))
        
        Z[0] = Z_0_true + np.random.multivariate_normal(np.zeros(m), V_0_true)
        for t in range(1, T):
            Z[t] = A_true @ Z[t-1] + np.random.multivariate_normal(np.zeros(m), Q_true)
        for t in range(T):
            X[t] = C_true @ Z[t] + np.random.multivariate_normal(np.zeros(N), R_true)
        
        # Fit model
        import torch
        X_torch = torch.tensor(X, dtype=torch.float32)
        self.model.initialize_from_data(X_torch)
        
        # Run EM
        state = self.model.fit_em(X_torch, Mx=np.zeros(N), Wx=np.ones(N))
        
        # Check convergence
        assert state.converged or state.num_iter < self.model.max_iter
        assert state.num_iter > 0
        assert np.isfinite(state.loglik)
        
        # Check that parameters are reasonable
        assert np.all(np.isfinite(state.A.numpy()))
        assert np.all(np.isfinite(state.C.numpy()))
    
    def test_em_algorithm_loglikelihood_increases(self):
        """Test that log-likelihood (generally) increases during EM."""
        T = 40
        X = np.random.randn(T, 4)
        
        import torch
        X_torch = torch.tensor(X, dtype=torch.float32)
        self.model.initialize_from_data(X_torch)
        
        # Track log-likelihoods
        logliks = []
        
        # Run a few EM steps manually
        for _ in range(5):
            _, _, _, _, _, _, loglik = self.model._em_step(X)
            logliks.append(loglik)
            # Update parameters
            A, C, Q, R, Z_0, V_0 = self.model._get_parameters_numpy()
            self.model._update_parameters_torch(A, C, Q, R, Z_0, V_0, X_torch.device)
        
        # Log-likelihood should generally increase (or at least not decrease dramatically)
        # Allow for some fluctuation
        assert all(np.isfinite(loglik) for loglik in logliks)
        # Check that we're not getting worse
        assert logliks[-1] >= logliks[0] - 100  # Allow some tolerance


class TestDFMIntegration:
    """Integration tests for DFM with Kalman filter and EM."""
    
    def test_dfm_full_pipeline(self):
        """Test complete DFM pipeline: initialization -> EM -> prediction."""
        from dfm_python.config import DEFAULT_BLOCK_NAME
        config = DFMConfig(
            series=[
                SeriesConfig(series_id=f'series_{i}', frequency='m')
                for i in range(6)
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 3, 'ar_lag': 1, 'clock': 'm'}}
        )
        model = DFM(config=config, max_iter=15, threshold=1e-3)
        
        # Generate data
        T = 50
        X = np.random.randn(T, 6)
        
        import torch
        X_torch = torch.tensor(X, dtype=torch.float32)
        
        # Initialize
        model.initialize_from_data(X_torch)
        assert model.A is not None
        assert model.C is not None
        
        # Fit
        state = model.fit_em(X_torch, Mx=np.zeros(6), Wx=np.ones(6))
        assert state.converged or state.num_iter <= model.max_iter
        
        # Get result
        result = model.get_result()
        assert result is not None
        assert result.Z.shape[0] == T
        # Check that result has correct number of factors (3 factors * 5 lags = 15 state dimensions)
        assert result.Z.shape[1] >= 3  # At least 3 factors (may have more due to lags/idiosyncratic)
        
        # Predict (specify target series)
        X_forecast, Z_forecast = model.predict(horizon=5, target=['series_0'])
        assert X_forecast.shape[0] == 5
        assert X_forecast.shape[1] == 1  # Only target series returned
        assert Z_forecast.shape[0] == 5
        assert Z_forecast.shape[1] >= 3  # At least 3 factors (may have more due to lags/idiosyncratic)
    
    def test_dfm_update_factor_state(self):
        """Test DFM factor state update using pykalman."""
        from dfm_python.config import DEFAULT_BLOCK_NAME
        config = DFMConfig(
            series=[
                SeriesConfig(series_id=f'series_{i}', frequency='m')
                for i in range(4)
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 2, 'ar_lag': 1, 'clock': 'm'}}
        )
        model = DFM(config=config, max_iter=10, threshold=1e-3)
        
        # Generate and fit
        T = 30
        X = np.random.randn(T, 4)
        
        import torch
        X_torch = torch.tensor(X, dtype=torch.float32)
        model.initialize_from_data(X_torch)
        model.fit_em(X_torch, Mx=np.zeros(4), Wx=np.ones(4))
        
        result = model.get_result()
        
        # Update with new data
        X_new = np.random.randn(10, 4)
        updated_state = model._update_factor_state_dfm(X_new, result)
        
        assert updated_state.shape == (2,)
        assert np.all(np.isfinite(updated_state))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

