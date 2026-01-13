"""Tests for functional.dfm_block module."""

import pytest
import numpy as np
from dfm_python.functional.dfm_block import (
    build_slower_freq_observation_matrix,
    build_slower_freq_idiosyncratic_chain,
    build_lag_matrix,
)
from dfm_python.numeric.tent import (
    generate_R_mat,
    get_slower_freq_tent_weights,
    generate_tent_weights,
)
from dfm_python.config.constants import DEFAULT_DTYPE


class TestDFMBlockFunctions:
    """Test suite for DFM block building functions."""
    
    def test_build_slower_freq_observation_matrix(self):
        """Test building slower frequency observation matrix."""
        N = 5
        n_clock_freq = 2
        n_slower_freq = 3
        tent_weights = np.array([1, 2, 3, 2, 1], dtype=DEFAULT_DTYPE)
        
        result = build_slower_freq_observation_matrix(N, n_clock_freq, n_slower_freq, tent_weights)
        tent_kernel_size = len(tent_weights)
        expected_cols = tent_kernel_size * n_slower_freq
        assert result.shape == (N, expected_cols)
        # First n_clock_freq rows should be zeros
        assert np.allclose(result[:n_clock_freq, :], 0.0)
    
    def test_build_lag_matrix(self):
        """Test building lag matrix."""
        T = 10
        num_factors = 2
        tent_kernel_size = 5
        p = 1
        factors = np.random.randn(T, num_factors).astype(DEFAULT_DTYPE)
        
        result = build_lag_matrix(factors, T, num_factors, tent_kernel_size, p)
        # Should return array of shape (T, num_factors * max(p + 1, tent_kernel_size))
        # The function uses max(p + 1, tent_kernel_size) as the number of lags
        num_lags = max(p + 1, tent_kernel_size)
        expected_cols = num_factors * num_lags
        assert result.shape == (T, expected_cols), \
            f"Expected shape ({T}, {expected_cols}), got {result.shape}"


class TestTentWeightsGeneration:
    """Test suite for tent weight generation and usage."""
    
    @pytest.mark.parametrize("tent_kernel_size", [3, 5, 7, 9])
    def test_generate_tent_weights_symmetric(self, tent_kernel_size):
        """Test that generate_tent_weights creates symmetric tent weights."""
        weights = generate_tent_weights(tent_kernel_size, tent_type='symmetric')
        
        assert weights is not None
        assert len(weights) == tent_kernel_size
        assert np.all(weights > 0), "All weights should be positive"
        
        # Should be symmetric (for odd-length)
        if tent_kernel_size % 2 == 1:
            mid = tent_kernel_size // 2
            first_half = weights[:mid]
            second_half = weights[mid+1:]
            assert np.array_equal(first_half, second_half[::-1]), "Weights should be symmetric"
    
    @pytest.mark.parametrize("tent_kernel_size", [3, 5, 7])
    def test_generate_R_mat_for_tent_weights(self, tent_kernel_size):
        """Test that generate_R_mat works correctly for generated tent weights."""
        weights = generate_tent_weights(tent_kernel_size, tent_type='symmetric')
        R_mat, q = generate_R_mat(weights)
        
        # R_mat should have correct shape
        assert R_mat.shape == (tent_kernel_size - 1, tent_kernel_size)
        assert q.shape == (tent_kernel_size - 1,)
        assert np.allclose(q, 0.0), "q should be all zeros"
        
        # Verify MATLAB pattern
        for i in range(tent_kernel_size - 1):
            assert np.abs(R_mat[i, 0] - weights[i + 1]) < 1e-6
            assert np.abs(R_mat[i, i + 1] + 1.0) < 1e-6
        
        # Verify constraint satisfaction
        loadings = weights * 1.0
        constraint_result = R_mat @ loadings
        assert np.allclose(constraint_result, q, atol=1e-6)
    
    @pytest.mark.parametrize("tent_kernel_size", [3, 5, 7])
    def test_build_slower_freq_observation_matrix(self, tent_kernel_size):
        """Test building slower frequency observation matrix with generated tent weights."""
        weights = generate_tent_weights(tent_kernel_size, tent_type='symmetric')
        
        N = 10
        n_clock_freq = 5
        n_slower_freq = 5
        
        result = build_slower_freq_observation_matrix(
            N, n_clock_freq, n_slower_freq, weights
        )
        
        expected_cols = tent_kernel_size * n_slower_freq
        assert result.shape == (N, expected_cols)
        assert np.allclose(result[:n_clock_freq, :], 0.0)
        assert not np.allclose(result[n_clock_freq:, :], 0.0)
    
    @pytest.mark.parametrize("tent_kernel_size", [3, 5])
    def test_build_slower_freq_idiosyncratic_chain(self, tent_kernel_size):
        """Test building slower frequency idiosyncratic chain with generated tent weights."""
        n_slower_freq = 3
        rho0 = 0.5
        sig_e = np.array([0.1, 0.2, 0.3], dtype=DEFAULT_DTYPE)
        
        BQ, SQ, initViQ = build_slower_freq_idiosyncratic_chain(
            n_slower_freq, tent_kernel_size, rho0, sig_e
        )
        
        expected_state_dim = tent_kernel_size * n_slower_freq
        assert BQ.shape == (expected_state_dim, expected_state_dim)
        assert SQ.shape == (expected_state_dim, expected_state_dim)
        assert initViQ.shape == (expected_state_dim, expected_state_dim)
    
    def test_get_slower_freq_tent_weights(self):
        """Test that get_slower_freq_tent_weights generates symmetric weights."""
        # Test with 3-period (production)
        weights_3 = get_slower_freq_tent_weights('m', 'w', tent_kernel_size=3)
        assert len(weights_3) == 3
        assert weights_3[0] == 1
        assert weights_3[-1] == 1
        
        # Test with 5-period (investment)
        weights_5 = get_slower_freq_tent_weights('m', 'w', tent_kernel_size=5)
        assert len(weights_5) == 5
        assert weights_5[0] == 1
        assert weights_5[-1] == 1
        assert weights_5[2] == 3  # Peak in middle

