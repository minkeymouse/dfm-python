"""Tests for functional.dfm_block module."""

import pytest
import numpy as np
from dfm_python.functional.dfm_block import (
    build_slower_freq_observation_matrix,
    build_slower_freq_idiosyncratic_chain,
    build_lag_matrix,
)
from dfm_python.numeric.tent import (
    get_tent_weights,
    generate_R_mat,
    get_slower_freq_tent_weights,
)
from dfm_python.config.constants import TENT_WEIGHTS_LOOKUP, DEFAULT_DTYPE


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


class TestFrequencyPairs:
    """Test suite for all frequency pairs in TENT_WEIGHTS_LOOKUP."""
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_get_tent_weights_for_all_pairs(self, slower_freq, faster_freq):
        """Test that get_tent_weights returns correct weights for all pairs."""
        expected_weights = TENT_WEIGHTS_LOOKUP[(slower_freq, faster_freq)]
        result = get_tent_weights(slower_freq, faster_freq)
        
        assert result is not None, f"get_tent_weights returned None for ({slower_freq}, {faster_freq})"
        assert np.array_equal(result, expected_weights), \
            f"Weights mismatch for ({slower_freq}, {faster_freq})"
        assert len(result) > 0, f"Empty weights for ({slower_freq}, {faster_freq})"
        assert np.all(result > 0), f"Non-positive weights for ({slower_freq}, {faster_freq})"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_tent_weights_properties(self, slower_freq, faster_freq):
        """Test that tent weights have correct properties (symmetric, positive, etc.)."""
        weights = get_tent_weights(slower_freq, faster_freq)
        
        # All weights should be positive
        assert np.all(weights > 0), f"Non-positive weights for ({slower_freq}, {faster_freq})"
        
        # Weights should be symmetric (first half should equal reversed second half)
        if len(weights) % 2 == 1:
            mid = len(weights) // 2
            first_half = weights[:mid]
            second_half = weights[mid+1:]
            assert np.array_equal(first_half, second_half[::-1]), \
                f"Weights not symmetric for ({slower_freq}, {faster_freq})"
        
        # Peak should be in the middle (for odd-length weights)
        if len(weights) % 2 == 1:
            mid = len(weights) // 2
            peak = weights[mid]
            assert peak == np.max(weights), \
                f"Peak not in middle for ({slower_freq}, {faster_freq})"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_generate_R_mat_for_all_pairs(self, slower_freq, faster_freq):
        """Test that generate_R_mat works correctly for all frequency pairs."""
        weights = get_tent_weights(slower_freq, faster_freq)
        R_mat, q = generate_R_mat(weights)
        
        # R_mat should have correct shape
        tent_kernel_size = len(weights)
        assert R_mat.shape == (tent_kernel_size - 1, tent_kernel_size), \
            f"R_mat shape incorrect for ({slower_freq}, {faster_freq}): expected ({tent_kernel_size - 1}, {tent_kernel_size}), got {R_mat.shape}"
        
        # q should have correct shape
        assert q.shape == (tent_kernel_size - 1,), \
            f"q shape incorrect for ({slower_freq}, {faster_freq}): expected ({tent_kernel_size - 1},), got {q.shape}"
        
        # q should be all zeros (constraint vector)
        assert np.allclose(q, 0.0), \
            f"q should be all zeros for ({slower_freq}, {faster_freq})"
        
        # R_mat enforces constraint: tent_weights[i+1]*c0 - 1*c(i+1) = 0
        # This means c(i+1) = tent_weights[i+1] * c0 (loadings are proportional to tent weights)
        # Test with loadings that satisfy this constraint: c = weights * scale
        scale = 1.0
        loadings_satisfying_constraint = weights * scale  # c[i] = weights[i] * c0
        constraint_result = R_mat @ loadings_satisfying_constraint
        assert np.allclose(constraint_result, q, atol=1e-6), \
            f"Constraint not satisfied for loadings = weights*scale for ({slower_freq}, {faster_freq}): {constraint_result}"
        
        # Test that R_mat matches MATLAB pattern:
        # - First column has tent_weights[i+1] (not w1)
        # - Diagonal has -1 (not -tent_weights[i+1])
        for i in range(tent_kernel_size - 1):
            assert np.abs(R_mat[i, 0] - weights[i + 1]) < 1e-6, \
                f"R_mat[{i}, 0] should be {weights[i+1]} (tent_weights[{i+1}]) for ({slower_freq}, {faster_freq}), got {R_mat[i, 0]}"
            assert np.abs(R_mat[i, i + 1] + 1.0) < 1e-6, \
                f"R_mat[{i}, {i+1}] should be -1 for ({slower_freq}, {faster_freq}), got {R_mat[i, i+1]}"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_build_slower_freq_observation_matrix_for_all_pairs(self, slower_freq, faster_freq):
        """Test that build_slower_freq_observation_matrix works for all frequency pairs."""
        weights = get_tent_weights(slower_freq, faster_freq)
        tent_kernel_size = len(weights)
        
        N = 10
        n_clock_freq = 5
        n_slower_freq = 5
        
        result = build_slower_freq_observation_matrix(
            N, n_clock_freq, n_slower_freq, weights
        )
        
        # Check shape
        expected_cols = tent_kernel_size * n_slower_freq
        assert result.shape == (N, expected_cols), \
            f"Shape mismatch for ({slower_freq}, {faster_freq}): expected ({N}, {expected_cols}), got {result.shape}"
        
        # First n_clock_freq rows should be zeros
        assert np.allclose(result[:n_clock_freq, :], 0.0), \
            f"First {n_clock_freq} rows not zero for ({slower_freq}, {faster_freq})"
        
        # Slower-frequency rows should have non-zero values
        assert not np.allclose(result[n_clock_freq:, :], 0.0), \
            f"Slower-frequency rows all zero for ({slower_freq}, {faster_freq})"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_build_slower_freq_idiosyncratic_chain_for_all_pairs(self, slower_freq, faster_freq):
        """Test that build_slower_freq_idiosyncratic_chain works for all frequency pairs."""
        weights = get_tent_weights(slower_freq, faster_freq)
        tent_kernel_size = len(weights)
        
        n_slower_freq = 3
        rho0 = 0.5
        sig_e = np.array([0.1, 0.2, 0.3], dtype=DEFAULT_DTYPE)
        
        BQ, SQ, initViQ = build_slower_freq_idiosyncratic_chain(
            n_slower_freq, tent_kernel_size, rho0, sig_e
        )
        
        # Check shapes
        expected_state_dim = tent_kernel_size * n_slower_freq
        assert BQ.shape == (expected_state_dim, expected_state_dim), \
            f"BQ shape incorrect for ({slower_freq}, {faster_freq}): expected ({expected_state_dim}, {expected_state_dim}), got {BQ.shape}"
        assert SQ.shape == (expected_state_dim, expected_state_dim), \
            f"SQ shape incorrect for ({slower_freq}, {faster_freq})"
        assert initViQ.shape == (expected_state_dim, expected_state_dim), \
            f"initViQ shape incorrect for ({slower_freq}, {faster_freq})"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_get_slower_freq_tent_weights_for_all_pairs(self, slower_freq, faster_freq):
        """Test that get_slower_freq_tent_weights works for all frequency pairs."""
        weights = get_tent_weights(slower_freq, faster_freq)
        tent_kernel_size = len(weights)
        
        result = get_slower_freq_tent_weights(slower_freq, faster_freq, tent_kernel_size)
        
        assert result is not None, \
            f"get_slower_freq_tent_weights returned None for ({slower_freq}, {faster_freq})"
        assert len(result) == tent_kernel_size, \
            f"Length mismatch for ({slower_freq}, {faster_freq}): expected {tent_kernel_size}, got {len(result)}"
        assert np.array_equal(result, weights), \
            f"Weights mismatch for ({slower_freq}, {faster_freq})"
    
    def test_invalid_frequency_pair(self):
        """Test that invalid frequency pairs return None."""
        result = get_tent_weights('invalid', 'freq')
        assert result is None, "Invalid frequency pair should return None"
        
        result = get_tent_weights('m', 'invalid')
        assert result is None, "Invalid frequency pair should return None"
    
    def test_all_pairs_have_valid_weights(self):
        """Test that all pairs in TENT_WEIGHTS_LOOKUP have valid weights."""
        for (slower_freq, faster_freq), weights in TENT_WEIGHTS_LOOKUP.items():
            assert len(weights) > 0, f"Empty weights for ({slower_freq}, {faster_freq})"
            assert np.all(weights > 0), f"Non-positive weights for ({slower_freq}, {faster_freq})"
            assert isinstance(weights, np.ndarray), \
                f"Weights not numpy array for ({slower_freq}, {faster_freq})"
    
    def test_frequency_pair_consistency(self):
        """Test that frequency pairs are consistent (slower < faster in hierarchy)."""
        from dfm_python.config.constants import FREQUENCY_HIERARCHY
        
        for slower_freq, faster_freq in TENT_WEIGHTS_LOOKUP.keys():
            slower_hierarchy = FREQUENCY_HIERARCHY.get(slower_freq, 0)
            faster_hierarchy = FREQUENCY_HIERARCHY.get(faster_freq, 0)
            
            assert slower_hierarchy > faster_hierarchy, \
                f"Hierarchy mismatch for ({slower_freq}, {faster_freq}): " \
                f"slower_freq hierarchy ({slower_hierarchy}) should be > faster_freq hierarchy ({faster_hierarchy})"

