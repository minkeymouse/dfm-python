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
    
    print("✓ Kalman filter basic test passed")
    return True


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
    
    print("✓ Kalman filter missing data test passed")
    return True


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
    
    print("✓ Fixed interval smoother basic test passed")
    return True


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
    
    print("✓ Missing data elimination test passed")
    return True

# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all Kalman filter tests."""
    print("\n" + "="*70)
    print("KALMAN FILTER TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('skf_basic', test_skf_basic),
        ('skf_missing_data', test_skf_missing_data),
        ('fis_basic', test_fis_basic),
        ('miss_data', test_miss_data),
    ]
    
    for name, func in test_funcs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[name] = func()
            print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("="*70)
    
    return results


if __name__ == '__main__':
    run_all_tests()
