"""Tests for numeric.estimator module."""

import pytest
import numpy as np
from dfm_python.config.constants import DEFAULT_DTYPE


class TestEstimator:
    """Test suite for numeric estimator utilities."""
    
    def test_estimate_ar1_unified_basic(self):
        """Test estimate_ar1_unified with basic input."""
        import numpy as np
        from dfm_python.numeric.estimator import estimate_ar1_unified
        
        T = 100
        N = 2
        y = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        x = y[:-1]  # Lagged values (T-1 x N)
        y_current = y[1:]  # Current values (T-1 x N) to match x
        
        A_diag, Q_diag = estimate_ar1_unified(y_current, x=x)
        # Returns diagonal arrays, not matrices
        assert A_diag.shape == (N,)
        assert Q_diag.shape == (N,)
        # Should be 1D arrays
        assert A_diag.ndim == 1
        assert Q_diag.ndim == 1
    
    def test_estimate_var_unified_basic(self):
        """Test estimate_var_unified with basic input."""
        import numpy as np
        from dfm_python.numeric.estimator import estimate_var_unified
        
        T = 100
        N = 2
        y = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        x = y[:-1]  # Lagged values (T-1 x N)
        y_current = y[1:]  # Current values (T-1 x N) to match x
        
        A, Q = estimate_var_unified(y_current, x)
        # Verify return types and basic properties
        assert A.ndim == 2
        assert Q.ndim == 2
        # A should have m rows where m = y_current.shape[1] = N
        assert A.shape[0] == N
        # Q should be square matrix (implementation may vary, but should be 2D)
        assert Q.shape[0] == Q.shape[1]
    
    def test_estimate_variance_unified_basic(self):
        """Test estimate_variance_unified with basic input."""
        import numpy as np
        from dfm_python.numeric.estimator import estimate_variance_unified
        
        T = 100
        N = 3
        residuals = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        
        R = estimate_variance_unified(residuals=residuals)
        # Returns diagonal matrix (N x N), not vector
        assert R.shape == (N, N)
        # Should be 2D matrix
        assert R.ndim == 2
        # Diagonal elements should be non-negative
        assert np.all(np.diag(R) >= 0)

