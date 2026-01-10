"""Tests for numeric.builder module."""

import pytest
import numpy as np
from dfm_python.config.constants import DEFAULT_DTYPE


class TestBuilder:
    """Test suite for numeric builder utilities."""
    
    def test_build_observation_matrix_factor_order_one(self):
        """Test build_observation_matrix with factor_order=1."""
        import numpy as np
        from dfm_python.numeric.builder import build_observation_matrix
        
        N = 3
        m = 2
        C = np.eye(N, m, dtype=DEFAULT_DTYPE)
        H = build_observation_matrix(C, factor_order=1, N=N)
        # Should be [C, I] where I is N x N identity
        assert H.shape == (N, m + N)
        assert np.allclose(H[:, :m], C)
        assert np.allclose(H[:, m:], np.eye(N))
    
    # VAR(2) tests removed - factors now always use AR(1) dynamics (simplified)
    # build_observation_matrix now only supports factor_order=1 (AR(1))

