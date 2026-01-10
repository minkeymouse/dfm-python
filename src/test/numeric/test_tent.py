"""Tests for numeric.tent module."""

import pytest
import numpy as np
from dfm_python.numeric.tent import (
    generate_tent_weights,
    generate_R_mat,
    get_tent_weights,
    get_agg_structure,
)
from dfm_python.config.constants import TENT_WEIGHTS_LOOKUP


class TestTentKernel:
    """Test suite for tent kernel."""
    
    def test_generate_tent_weights(self):
        """Test tent weights generation."""
        # Test symmetric tent with odd number of periods
        weights_odd = generate_tent_weights(5, tent_type='symmetric')
        assert weights_odd is not None
        assert isinstance(weights_odd, np.ndarray)
        assert len(weights_odd) == 5
        assert weights_odd[0] == 1  # First element should be 1
        assert weights_odd[-1] == 1  # Last element should be 1
        assert weights_odd[2] == 3  # Middle element should be peak
        
        # Test symmetric tent with even number of periods
        weights_even = generate_tent_weights(6, tent_type='symmetric')
        assert len(weights_even) == 6
        assert weights_even[0] == 1
        assert weights_even[-1] == 1
        
        # Test linear tent
        weights_linear = generate_tent_weights(5, tent_type='linear')
        assert len(weights_linear) == 5
        assert weights_linear[0] == 1  # Should start at 1
    
    def test_get_tent_weights(self):
        """Test getting tent weights for frequency pairs."""
        # Test known frequency pair from TENT_WEIGHTS_LOOKUP
        weights = get_tent_weights('q', 'm')  # Quarterly to monthly
        assert weights is not None
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 5  # Should match TENT_WEIGHTS_LOOKUP[('q', 'm')]
        expected = TENT_WEIGHTS_LOOKUP[('q', 'm')]
        assert np.array_equal(weights, expected)
        
        # Test another known pair
        weights2 = get_tent_weights('a', 'm')  # Annual to monthly
        assert weights2 is not None
        assert isinstance(weights2, np.ndarray)
        expected2 = TENT_WEIGHTS_LOOKUP[('a', 'm')]
        assert np.array_equal(weights2, expected2)
        
        # Test invalid frequency pair (should return None or raise)
        weights_invalid = get_tent_weights('invalid', 'm')
        # Function may return None for invalid pairs - check implementation behavior
        # This test verifies the function exists and can be called
    
    def test_get_agg_structure(self):
        """Test aggregation structure computation."""
        # Test with valid frequency pair
        # get_agg_structure may require config or specific parameters
        # This test verifies the function exists and can be called
        # Full test would require proper DFMConfig setup
        assert callable(get_agg_structure)
        # Function signature may require config - test basic callability
        # More comprehensive test would require full DFMConfig setup
    
    def test_generate_R_mat_matches_matlab(self):
        """Test that generate_R_mat matches FRBNY MATLAB pattern."""
        # Test with [1, 2, 3, 2, 1] tent weights (monthly->quarterly)
        weights = np.array([1, 2, 3, 2, 1])
        R_mat, q = generate_R_mat(weights)
        
        # Expected MATLAB pattern from dfm.m lines 85-88
        # R_mat = [  2 -1  0  0  0;...   % w[1]*c0 - 1*c1 = 0 → c1 = 2*c0
        #           3  0 -1  0  0;...   % w[2]*c0 - 1*c2 = 0 → c2 = 3*c0
        #           2  0  0 -1  0;...   % w[3]*c0 - 1*c3 = 0 → c3 = 2*c0
        #           1  0  0  0 -1];     % w[4]*c0 - 1*c4 = 0 → c4 = 1*c0
        expected = np.array([
            [2, -1,  0,  0,  0],
            [3,  0, -1,  0,  0],
            [2,  0,  0, -1,  0],
            [1,  0,  0,  0, -1]
        ])
        
        assert np.array_equal(R_mat, expected), \
            f"R_mat does not match MATLAB pattern.\nExpected:\n{expected}\nGot:\n{R_mat}"
        
        # Verify constraint vector is all zeros
        assert np.allclose(q, 0.0), f"q should be all zeros, got {q}"
        
        # Verify constraint satisfaction: loadings proportional to weights should satisfy constraint
        scale = 1.5
        loadings = weights * scale  # c[i] = weights[i] * c0
        constraint_result = R_mat @ loadings
        assert np.allclose(constraint_result, q, atol=1e-6), \
            f"Constraint not satisfied for loadings = weights*scale: {constraint_result}"
    
    @pytest.mark.parametrize("slower_freq,faster_freq", list(TENT_WEIGHTS_LOOKUP.keys()))
    def test_generate_R_mat_for_all_tent_weights(self, slower_freq, faster_freq):
        """Test generate_R_mat for all tent weights in TENT_WEIGHTS_LOOKUP."""
        weights = get_tent_weights(slower_freq, faster_freq)
        if weights is None:
            pytest.skip(f"No tent weights for ({slower_freq}, {faster_freq})")
        
        R_mat, q = generate_R_mat(weights)
        tent_kernel_size = len(weights)
        
        # Verify shape
        assert R_mat.shape == (tent_kernel_size - 1, tent_kernel_size), \
            f"R_mat shape incorrect for ({slower_freq}, {faster_freq})"
        assert q.shape == (tent_kernel_size - 1,), \
            f"q shape incorrect for ({slower_freq}, {faster_freq})"
        
        # Verify q is all zeros
        assert np.allclose(q, 0.0), \
            f"q should be all zeros for ({slower_freq}, {faster_freq})"
        
        # Verify MATLAB pattern: R_mat[i, 0] = tent_weights[i+1], R_mat[i, i+1] = -1
        for i in range(tent_kernel_size - 1):
            assert np.abs(R_mat[i, 0] - weights[i + 1]) < 1e-6, \
                f"R_mat[{i}, 0] should be {weights[i+1]} for ({slower_freq}, {faster_freq})"
            assert np.abs(R_mat[i, i + 1] + 1.0) < 1e-6, \
                f"R_mat[{i}, {i+1}] should be -1 for ({slower_freq}, {faster_freq})"
        
        # Verify constraint satisfaction
        scale = 1.0
        loadings = weights * scale
        constraint_result = R_mat @ loadings
        assert np.allclose(constraint_result, q, atol=1e-6), \
            f"Constraint not satisfied for ({slower_freq}, {faster_freq}): {constraint_result}"

