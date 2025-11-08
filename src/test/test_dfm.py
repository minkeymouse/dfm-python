"""Core tests for DFM estimation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import dfm, DFMResult, em_step, init_conditions
from dfm_python.data_loader import load_config
from dfm_python.utils.data_utils import rem_nans_spline
from dfm_python.config import DFMConfig, SeriesConfig

# ============================================================================
# Core Tests
# ============================================================================

def test_em_step_basic():
    """Test basic EM step functionality."""
    print("\n" + "="*70)
    print("TEST: EM Step Basic")
    print("="*70)
    
    T, N = 80, 8
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
        y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, None
    )
    
    assert C_new is not None and R_new is not None
    assert A_new is not None and np.isfinite(loglik)
    assert C_new.shape == (N, A.shape[0])
    assert R_new.shape == (N, N)
    
    print("✓ EM step basic test passed")


def test_init_conditions_basic():
    """Test basic initial conditions."""
    print("\n" + "="*70)
    print("TEST: Init Conditions Basic")
    print("="*70)
    
    T, N = 100, 10
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    
    blocks = np.zeros((N, 2), dtype=int)
    blocks[:, 0] = 1
    r = np.ones(2)
    p = 1
    
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    m = A.shape[0]
    assert A.shape == (m, m)
    assert C.shape == (N, m)
    assert Q.shape == (m, m)
    assert R.shape == (N, N)
    assert Z_0.shape == (m,)
    assert V_0.shape == (m, m)
    assert not np.any(np.isnan(A))
    assert not np.any(np.isnan(C))
    
    print("✓ Init conditions basic test passed")


def test_dfm_quick():
    """Quick DFM test with synthetic data."""
    print("\n" + "="*70)
    print("TEST: DFM Quick")
    print("="*70)
    
    T, N = 50, 10
    np.random.seed(42)
    
    # Generate synthetic data
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    
    # Add missing values
    missing_mask = np.random.rand(T, N) < 0.1
    X[missing_mask] = np.nan
    
    # Create config
    block_names = ['Global']
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=[1]
        ))
    
    config = DFMConfig(series=series_list, block_names=block_names)
    
    # Run DFM
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    # Verify structure
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ DFM quick test passed")


def test_multi_block_different_factors():
    """Test multi-block with different factor counts (bug fix verification)."""
    print("\n" + "="*70)
    print("TEST: Multi-Block Different Factors")
    print("="*70)
    
    np.random.seed(42)
    T, N = 100, 15
    
    x = np.random.randn(T, N)
    
    # 3 blocks with different factor counts
    blocks = np.zeros((N, 3), dtype=int)
    blocks[0:5, 0] = 1
    blocks[5:10, 1] = 1
    blocks[10:15, 2] = 1
    blocks[:, 0] = 1  # All load on global
    
    r = np.array([3, 2, 2])
    p = 1
    
    opt_nan = {'method': 2, 'k': 3}
    nQ = 0
    i_idio = np.ones(N)
    R_mat = None
    q = None
    
    # Should not raise dimension mismatch error
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    assert A is not None and C is not None
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(C))
    
    print("✓ Multi-block different factors test passed")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all DFM tests."""
    print("\n" + "="*70)
    print("DFM TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('em_step_basic', test_em_step_basic),
        ('init_conditions_basic', test_init_conditions_basic),
        ('dfm_quick', test_dfm_quick),
        ('multi_block_different_factors', test_multi_block_different_factors),
    ]
    
    for name, func in test_funcs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                func()
            results[name] = True
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
