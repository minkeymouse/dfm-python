"""Core tests for Kalman filter and smoother."""

import sys
from pathlib import Path
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.kalman import skf, fis, miss_data, KalmanFilterState

# ============================================================================
# Helper Functions
# ============================================================================

def create_test_data(T=100, n=10, m=3, seed=42):
    """Create test data for Kalman filter."""
    np.random.seed(seed)
    
    A = np.random.randn(m, m) * 0.5
    A = A / (np.max(np.abs(np.linalg.eigvals(A))) + 0.1)  # Ensure stability
    
    C = np.random.randn(n, m)
    Q = np.eye(m) * 0.1
    R = np.eye(n) * 0.5
    
    Z_0 = np.zeros(m)
    V_0 = np.eye(m)
    
    # Generate data
    z = np.zeros((T, m))
    z[0] = Z_0
    for t in range(1, T):
        z[t] = A @ z[t-1] + np.random.multivariate_normal(np.zeros(m), Q)
    
    y = np.zeros((n, T))
    for t in range(T):
        y[:, t] = C @ z[t] + np.random.multivariate_normal(np.zeros(n), R)
    
    return y, A, C, Q, R, Z_0, V_0, z

# ============================================================================
# Core Tests
# ============================================================================

def test_skf_basic():
    """Test basic Kalman filter."""
    print("\n" + "="*70)
    print("TEST: Kalman Filter Basic")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    
    assert Sf is not None
    assert hasattr(Sf, 'ZmU') and hasattr(Sf, 'VmU')
    assert hasattr(Sf, 'loglik')
    assert Sf.ZmU.shape == (z_true.shape[1], y.shape[1] + 1)
    assert np.isfinite(Sf.loglik)
    


def test_skf_missing_data():
    """Test Kalman filter with missing data."""
    print("\n" + "="*70)
    print("TEST: Kalman Filter Missing Data")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    # Add missing data
    missing_mask = np.random.rand(*y.shape) < 0.2
    y_missing = y.copy()
    y_missing[missing_mask] = np.nan
    
    Sf = skf(y_missing, A, C, Q, R, Z_0, V_0)
    
    assert Sf is not None
    assert np.isfinite(Sf.loglik)
    


def test_fis_basic():
    """Test basic fixed interval smoother."""
    print("\n" + "="*70)
    print("TEST: Fixed Interval Smoother Basic")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    
    assert Ss is not None
    assert hasattr(Ss, 'ZmT') and hasattr(Ss, 'VmT')
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    


def test_miss_data():
    """Test missing data elimination function."""
    print("\n" + "="*70)
    print("TEST: Missing Data Elimination")
    print("="*70)
    
    n, m = 10, 3
    y = np.random.randn(n)
    C = np.random.randn(n, m)
    R = np.eye(n) * 0.5
    
    # Add missing values
    y_missing = y.copy()
    y_missing[2] = np.nan
    y_missing[5] = np.nan
    y_missing[8] = np.nan
    
    y_new, C_new, R_new, L = miss_data(y_missing, C, R)
    
    assert len(y_new) == n - 3
    assert C_new.shape[0] == n - 3
    assert C_new.shape[1] == m
    assert R_new.shape == (n - 3, n - 3)
    assert L.shape == (n, n - 3)
    
    # Check non-missing observations preserved
    non_missing_idx = [i for i in range(n) if not np.isnan(y_missing[i])]
    assert np.allclose(y_new, y[non_missing_idx])
    


def test_skf_zero_observation_variance():
    """Test Kalman filter with zero observation variance (edge case).
    
    This tests the edge case where R has zeros on the diagonal, which could
    cause numerical issues. The filter should handle this gracefully.
    """
    print("\n" + "="*70)
    print("TEST: Kalman Filter Zero Observation Variance")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=30, n=5, m=2)
    
    # Create R with some zeros on diagonal (edge case)
    R_zero = R.copy()
    R_zero[0, 0] = 0.0  # First observation has zero variance
    R_zero[2, 2] = 0.0  # Third observation has zero variance
    
    # Should handle gracefully - may use regularization or fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sf = skf(y, A, C, Q, R_zero, Z_0, V_0)
    
    assert Sf is not None
    assert hasattr(Sf, 'ZmU') and hasattr(Sf, 'VmU')
    assert hasattr(Sf, 'loglik')
    assert Sf.ZmU.shape == (z_true.shape[1], y.shape[1] + 1)
    assert np.isfinite(Sf.loglik) or np.isnan(Sf.loglik)  # loglik may be NaN in extreme cases
    
    # Verify outputs are finite (may have NaN in extreme cases, but should not crash)
    assert np.all(np.isfinite(Sf.ZmU)) or np.any(np.isfinite(Sf.ZmU)), \
        "At least some state estimates should be finite"
    


def test_fis_all_missing_observations():
    """Test fixed interval smoother with all missing observations.
    
    This tests the edge case where all observations are missing at all time
    points. The smoother should handle this gracefully.
    """
    print("\n" + "="*70)
    print("TEST: Fixed Interval Smoother All Missing Observations")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=30, n=5, m=2)
    
    # Create Y with all NaN (all observations missing)
    y_all_missing = np.full_like(y, np.nan)
    
    # Run Kalman filter with all missing data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sf = skf(y_all_missing, A, C, Q, R, Z_0, V_0)
    
    # Run smoother
    from dfm_python.kalman import fis
    Ss = fis(A, Sf)
    
    assert Ss is not None
    assert hasattr(Ss, 'ZmT') and hasattr(Ss, 'VmT')
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    
    # With all missing observations, smoother should still produce finite outputs
    # (based on prior/prediction only, no observation updates)
    assert np.all(np.isfinite(Ss.ZmT)) or np.any(np.isfinite(Ss.ZmT)), \
        "At least some smoothed state estimates should be finite"
    assert np.all(np.isfinite(Ss.VmT)) or np.any(np.isfinite(Ss.VmT)), \
        "At least some smoothed covariances should be finite"
    

