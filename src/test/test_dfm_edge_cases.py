"""Edge case tests for DFM estimation."""

import sys
from pathlib import Path
import numpy as np
import pytest
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import dfm, init_conditions, em_step
from dfm_python.config import DFMConfig, SeriesConfig
from dfm_python.utils.data_utils import rem_nans_spline


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_data():
    """Test with empty data array."""
    print("\n" + "="*70)
    print("TEST: Empty Data")
    print("="*70)
    
    X = np.array([]).reshape(0, 0)
    
    block_names = ['Global']
    series_list = [SeriesConfig(
        series_id="TEST_01",
        series_name="Test Series",
        frequency='m',
        transformation='lin',
        blocks=[1]
    )]
    config = DFMConfig(series=series_list, block_names=block_names)
    
    with pytest.raises((ValueError, IndexError)):
        dfm(X, config, max_iter=10)


def test_single_series():
    """Test with single time series."""
    print("\n" + "="*70)
    print("TEST: Single Series")
    print("="*70)
    
    T = 50
    np.random.seed(42)
    X = np.random.randn(T, 1)
    
    block_names = ['Global']
    series_list = [SeriesConfig(
        series_id="TEST_01",
        series_name="Test Series",
        frequency='m',
        transformation='lin',
        blocks=[1]
    )]
    config = DFMConfig(series=series_list, block_names=block_names)
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, 1)
    assert Res.Z.shape[0] == T
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Single series test passed")


def test_all_nan_data():
    """Test with all NaN data."""
    print("\n" + "="*70)
    print("TEST: All NaN Data")
    print("="*70)
    
    T, N = 50, 5
    X = np.full((T, N), np.nan)
    
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
    
    # Should handle gracefully or raise informative error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Res = dfm(X, config, threshold=1e-3, max_iter=10)
            # If it succeeds, verify structure
            assert Res.x_sm.shape == (T, N)
        except (ValueError, RuntimeError) as e:
            # Acceptable - all NaN data is invalid
            assert "nan" in str(e).lower() or "missing" in str(e).lower() or "data" in str(e).lower()
    
    print("✓ All NaN data test passed")


def test_high_missing_data():
    """Test with very high percentage of missing data."""
    print("\n" + "="*70)
    print("TEST: High Missing Data (80%)")
    print("="*70)
    
    T, N = 100, 10
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Make 80% missing
    missing_mask = np.random.rand(T, N) < 0.8
    X[missing_mask] = np.nan
    
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Res = dfm(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))
    
    print("✓ High missing data test passed")


def test_very_small_dataset():
    """Test with very small dataset (minimal dimensions)."""
    print("\n" + "="*70)
    print("TEST: Very Small Dataset")
    print("="*70)
    
    T, N = 10, 3
    np.random.seed(42)
    X = np.random.randn(T, N)
    
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
    
    Res = dfm(X, config, threshold=1e-3, max_iter=20)
    
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    
    print("✓ Very small dataset test passed")


def test_single_block():
    """Test with single block."""
    print("\n" + "="*70)
    print("TEST: Single Block")
    print("="*70)
    
    T, N = 50, 8
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    block_names = ['Block1']
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
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Single block test passed")


def test_extreme_values():
    """Test with extreme values (very large, very small)."""
    print("\n" + "="*70)
    print("TEST: Extreme Values")
    print("="*70)
    
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Add extreme values
    X[0, 0] = 1e10
    X[1, 0] = -1e10
    X[2, 1] = 1e-10
    X[3, 1] = -1e-10
    
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Res = dfm(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.all(np.isfinite(Res.Z))
    
    print("✓ Extreme values test passed")


def test_inf_values():
    """Test with Inf values."""
    print("\n" + "="*70)
    print("TEST: Inf Values")
    print("="*70)
    
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Add Inf values
    X[0, 0] = np.inf
    X[1, 0] = -np.inf
    
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Should convert Inf to NaN and handle
        Res = dfm(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.all(np.isfinite(Res.Z))
    
    print("✓ Inf values test passed")


def test_constant_series():
    """Test with constant (zero variance) series."""
    print("\n" + "="*70)
    print("TEST: Constant Series")
    print("="*70)
    
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Make one series constant
    X[:, 0] = 5.0
    
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Res = dfm(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))
    
    print("✓ Constant series test passed")


def test_mixed_frequencies():
    """Test with mixed frequencies (monthly and quarterly)."""
    print("\n" + "="*70)
    print("TEST: Mixed Frequencies")
    print("="*70)
    
    T, N = 60, 8
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    block_names = ['Global']
    series_list = []
    for i in range(N):
        freq = 'm' if i < 5 else 'q'
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency=freq,
            transformation='lin',
            blocks=[1]
        ))
    config = DFMConfig(series=series_list, block_names=block_names, clock='m')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Res = dfm(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))
    
    print("✓ Mixed frequencies test passed")


def test_no_convergence():
    """Test with very strict convergence threshold."""
    print("\n" + "="*70)
    print("TEST: No Convergence (Strict Threshold)")
    print("="*70)
    
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
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
    
    # Very strict threshold, should hit max_iter
    Res = dfm(X, config, threshold=1e-10, max_iter=10)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ No convergence test passed")


def test_multiple_blocks_different_sizes():
    """Test with multiple blocks of different sizes."""
    print("\n" + "="*70)
    print("TEST: Multiple Blocks Different Sizes")
    print("="*70)
    
    T, N = 100, 15
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    block_names = ['Block1', 'Block2', 'Block3']
    series_list = []
    for i in range(N):
        # All series must load on first block (global), then can load on additional blocks
        block_idx = (i % 2) + 1  # Block 1 or 2 (0-indexed: 1 or 2)
        blocks = [1, 0, 0]  # Always load on first block
        blocks[block_idx] = 1  # Also load on one additional block
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=blocks
        ))
    config = DFMConfig(series=series_list, block_names=block_names)
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Multiple blocks different sizes test passed")


def test_init_conditions_edge_cases():
    """Test init_conditions with edge cases."""
    print("\n" + "="*70)
    print("TEST: Init Conditions Edge Cases")
    print("="*70)
    
    T, N = 30, 5
    np.random.seed(42)
    x = np.random.randn(T, N)
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Test with all NaN
    x_all_nan = np.full((T, N), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            A, C, Q, R, Z_0, V_0 = init_conditions(
                x_all_nan, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
            )
            # If succeeds, verify structure
            assert A is not None
        except (ValueError, RuntimeError):
            # Acceptable
            pass
    
    # Test with single series
    x_single = x[:, 0:1]
    blocks_single = np.ones((1, 1), dtype=int)
    i_idio_single = np.ones(1)
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x_single, r, p, blocks_single, opt_nan, R_mat, q, nQ, i_idio_single
    )
    assert A is not None
    
    print("✓ Init conditions edge cases test passed")


def test_em_step_edge_cases():
    """Test em_step with edge cases."""
    print("\n" + "="*70)
    print("TEST: EM Step Edge Cases")
    print("="*70)
    
    T, N = 30, 5
    np.random.seed(42)
    x = np.random.randn(T, N)
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    from dfm_python.dfm import init_conditions
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    
    # Test with valid inputs
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
        y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, None
    )
    
    assert np.isfinite(loglik)
    assert C_new.shape == C.shape
    assert R_new.shape == R.shape
    
    print("✓ EM step edge cases test passed")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_edge_case_tests():
    """Run all edge case tests."""
    print("\n" + "="*70)
    print("DFM EDGE CASE TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('single_series', test_single_series),
        ('all_nan_data', test_all_nan_data),
        ('high_missing_data', test_high_missing_data),
        ('very_small_dataset', test_very_small_dataset),
        ('single_block', test_single_block),
        ('extreme_values', test_extreme_values),
        ('inf_values', test_inf_values),
        ('constant_series', test_constant_series),
        ('mixed_frequencies', test_mixed_frequencies),
        ('no_convergence', test_no_convergence),
        ('multiple_blocks_different_sizes', test_multiple_blocks_different_sizes),
        ('init_conditions_edge_cases', test_init_conditions_edge_cases),
        ('em_step_edge_cases', test_em_step_edge_cases),
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
    print(f"SUMMARY: {passed}/{total} edge case tests passed")
    print("="*70)
    
    return results


if __name__ == '__main__':
    run_all_edge_case_tests()

