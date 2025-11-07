"""Consolidated tests for Kalman filter and smoother."""

import sys
from pathlib import Path
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.kalman import skf, fis, miss_data, KalmanFilterState

# ============================================================================
# Helper Functions
# ============================================================================

def create_test_data(T=100, n=10, m=3, seed=42):
    """Create test data for Kalman filter."""
    np.random.seed(seed)
    
    # State space model: y_t = C * z_t + e_t, z_t = A * z_{t-1} + w_t
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
# Kalman Filter Tests
# ============================================================================

def test_skf_basic():
    """Test basic Kalman filter."""
    print("\n" + "="*70)
    print("TEST: Kalman Filter Basic")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    
    assert Sf is not None
    assert hasattr(Sf, 'ZmU')
    assert hasattr(Sf, 'VmU')
    assert hasattr(Sf, 'loglik')
    
    assert Sf.ZmU.shape == (z_true.shape[1], y.shape[1] + 1)
    assert np.isfinite(Sf.loglik)
    
    print(f"✓ Kalman filter completed")
    print(f"  State shape: {Sf.ZmU.shape}")
    print(f"  Log-likelihood: {Sf.loglik:.4f}")
    
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
    
    print(f"✓ Kalman filter with missing data completed")
    print(f"  Missing rate: {np.sum(missing_mask) / missing_mask.size * 100:.1f}%")
    
    return True

def test_skf_dimensions():
    """Test Kalman filter dimension consistency."""
    print("\n" + "="*70)
    print("TEST: Kalman Filter Dimensions")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=60, n=8, m=3)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    
    n, T = y.shape
    m = A.shape[0]
    
    assert Sf.ZmU.shape == (m, T + 1)
    assert Sf.VmU.shape[0] == m
    assert Sf.VmU.shape[1] == m
    assert Sf.VmU.shape[2] == T + 1
    
    print("✓ Dimension consistency verified")
    return True

# ============================================================================
# Fixed Interval Smoother Tests
# ============================================================================

def test_fis_basic():
    """Test basic fixed interval smoother."""
    print("\n" + "="*70)
    print("TEST: Fixed Interval Smoother Basic")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    
    assert Ss is not None
    assert hasattr(Ss, 'ZmT')
    assert hasattr(Ss, 'VmT')
    
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    
    print("✓ Fixed interval smoother completed")
    return True

def test_fis_smoothness():
    """Test that smoothed estimates are smoother than filtered."""
    print("\n" + "="*70)
    print("TEST: Smoother Smoothness")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    
    # Smoothed estimates should generally have lower variance
    filtered_var = np.var(Sf.ZmU[:, 1:], axis=1)
    smoothed_var = np.var(Ss.ZmT[:, 1:], axis=1)
    
    # Check that smoothed variance is less than or equal to filtered
    assert np.all(smoothed_var <= filtered_var + 1e-6)
    
    print("✓ Smoothness property verified")
    return True

def test_fis_dimensions():
    """Test fixed interval smoother dimensions."""
    print("\n" + "="*70)
    print("TEST: Smoother Dimensions")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=60, n=8, m=3)
    
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    
    print("✓ Dimension consistency verified")
    return True

# ============================================================================
# Missing Data Utilities
# ============================================================================

def test_miss_data():
    """Test missing data elimination function."""
    print("\n" + "="*70)
    print("TEST: Missing Data Elimination")
    print("="*70)
    
    n, m = 10, 3
    y = np.random.randn(n)
    C = np.random.randn(n, m)
    R = np.eye(n) * 0.5
    
    # Add some missing values
    y_missing = y.copy()
    y_missing[2] = np.nan
    y_missing[5] = np.nan
    y_missing[8] = np.nan
    
    y_new, C_new, R_new, obs_idx = miss_data(y_missing, C, R)
    
    assert len(y_new) == n - 3
    assert C_new.shape[0] == n - 3
    assert C_new.shape[1] == m
    assert R_new.shape == (n - 3, n - 3)
    assert len(obs_idx) == n - 3
    
    # Check that non-missing observations are preserved
    non_missing_idx = [i for i in range(n) if not np.isnan(y_missing[i])]
    assert np.allclose(y_new, y[non_missing_idx])
    
    print("✓ Missing data elimination verified")
    return True

# ============================================================================
# Integration Tests
# ============================================================================

def test_kalman_smoother_integration():
    """Test integration of filter and smoother."""
    print("\n" + "="*70)
    print("TEST: Filter-Smoother Integration")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=100, n=10, m=4)
    
    # Run filter
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    
    # Run smoother
    Ss = fis(A, Sf)
    
    # Check consistency
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    
    # Smoothed should use all information (backward pass)
    # So smoothed estimates at T should equal filtered
    assert np.allclose(Ss.ZmT[:, -1], Sf.ZmU[:, -1], atol=1e-6)
    
    print("✓ Filter-smoother integration verified")
    return True

def test_kalman_stability():
    """Test numerical stability."""
    print("\n" + "="*70)
    print("TEST: Numerical Stability")
    print("="*70)
    
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=200, n=15, m=5)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sf = skf(y, A, C, Q, R, Z_0, V_0)
        Ss = fis(A, Sf)
    
    assert np.all(np.isfinite(Sf.ZmU))
    assert np.all(np.isfinite(Sf.VmU))
    assert np.all(np.isfinite(Ss.ZmT))
    assert np.all(np.isfinite(Ss.VmT))
    assert np.isfinite(Sf.loglik)
    
    print("✓ Numerical stability verified")
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
        ('skf_dimensions', test_skf_dimensions),
        ('fis_basic', test_fis_basic),
        ('fis_smoothness', test_fis_smoothness),
        ('fis_dimensions', test_fis_dimensions),
        ('miss_data', test_miss_data),
        ('integration', test_kalman_smoother_integration),
        ('stability', test_kalman_stability),
    ]
    
    for name, func in test_funcs:
        try:
            results[name] = func()
            print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    passed = sum(1 for v in results.values() if v is not None and v is not False)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("="*70)
    
    return results

if __name__ == '__main__':
    run_all_tests()



