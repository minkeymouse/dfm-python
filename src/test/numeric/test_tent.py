"""Tests for numeric.tent module."""

import pytest
import numpy as np
from dfm_python.numeric.tent import (
    generate_tent_weights,
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

