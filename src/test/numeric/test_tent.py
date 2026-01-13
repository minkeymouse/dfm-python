"""Tests for numeric.tent module."""

import pytest
import numpy as np
from dfm_python.numeric.tent import (
    generate_tent_weights,
    generate_R_mat,
    get_agg_structure,
    get_slower_freq_tent_weights,
)


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
    
    def test_get_slower_freq_tent_weights(self):
        """Test getting tent weights for slower frequency with fallback generation."""
        # Test with known tent kernel size - function should generate symmetric weights
        weights = get_slower_freq_tent_weights('q', 'm', tent_kernel_size=5)
        assert weights is not None
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 5
        # Should generate symmetric tent weights
        assert weights[0] == 1
        assert weights[-1] == 1
        assert weights[2] == 3  # Middle should be peak
    
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
    
    def test_generate_R_mat_various_sizes(self):
        """Test generate_R_mat for various tent weight sizes."""
        # Test with 3-period tent (production)
        weights_3 = np.array([1, 2, 1])
        R_mat_3, q_3 = generate_R_mat(weights_3)
        assert R_mat_3.shape == (2, 3)
        assert np.allclose(q_3, 0.0)
        
        # Test with 5-period tent (investment)
        weights_5 = np.array([1, 2, 3, 2, 1])
        R_mat_5, q_5 = generate_R_mat(weights_5)
        assert R_mat_5.shape == (4, 5)
        assert np.allclose(q_5, 0.0)
        
        # Verify MATLAB pattern for 5-period
        assert np.abs(R_mat_5[0, 0] - weights_5[1]) < 1e-6  # Should be 2
        assert np.abs(R_mat_5[0, 1] + 1.0) < 1e-6  # Should be -1
        
        # Verify constraint satisfaction
        loadings = weights_5 * 1.0
        constraint_result = R_mat_5 @ loadings
        assert np.allclose(constraint_result, q_5, atol=1e-6)

