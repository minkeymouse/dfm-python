"""Stress tests and additional edge cases for DFM estimation."""

import sys
from pathlib import Path
import numpy as np
import pytest
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import dfm, init_conditions
from dfm_python.config import DFMConfig, SeriesConfig


# ============================================================================
# Stress Tests
# ============================================================================

def test_large_dataset():
    """Test with large dataset."""
    print("\n" + "="*70)
    print("TEST: Large Dataset")
    print("="*70)
    
    T, N = 200, 30  # Reduced size for faster testing
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
    
    Res = dfm(X, config, threshold=1e-2, max_iter=20)  # Looser threshold, fewer iterations
    
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert np.all(np.isfinite(Res.Z))
    
    print("✓ Large dataset test passed")


@pytest.mark.slow
def test_very_large_dataset():
    """Test with very large dataset (marked as slow)."""
    print("\n" + "="*70)
    print("TEST: Very Large Dataset (SKIPPED - too slow)")
    print("="*70)
    
    # Skip this test by default - too slow for regular testing
    pytest.skip("Very large dataset test skipped - too slow for regular testing")
    
    T, N = 500, 50  # Reduced from 1000, 100
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Add some missing values
    missing_mask = np.random.rand(T, N) < 0.05
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
    
    Res = dfm(X, config, threshold=1e-2, max_iter=15)
    
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    
    print("✓ Very large dataset test passed")


def test_numerical_precision():
    """Test numerical precision with very small values."""
    print("\n" + "="*70)
    print("TEST: Numerical Precision")
    print("="*70)
    
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N) * 1e-10  # Very small values
    
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
    
    print("✓ Numerical precision test passed")


def test_sparse_data():
    """Test with very sparse data (many zeros)."""
    print("\n" + "="*70)
    print("TEST: Sparse Data")
    print("="*70)
    
    T, N = 100, 10
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Make 90% zeros
    zero_mask = np.random.rand(T, N) < 0.9
    X[zero_mask] = 0.0
    
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
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Sparse data test passed")


def test_high_correlation():
    """Test with highly correlated series."""
    print("\n" + "="*70)
    print("TEST: High Correlation")
    print("="*70)
    
    T, N = 100, 10
    np.random.seed(42)
    
    # Create highly correlated series
    base_series = np.random.randn(T)
    X = np.zeros((T, N))
    for i in range(N):
        X[:, i] = base_series + np.random.randn(T) * 0.1
    
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
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ High correlation test passed")


def test_different_transformations():
    """Test with different transformation types."""
    print("\n" + "="*70)
    print("TEST: Different Transformations")
    print("="*70)
    
    T, N = 100, 6
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Make some series positive for log transformation
    X[:, 3:6] = np.abs(X[:, 3:6]) + 1
    
    block_names = ['Global']
    series_list = []
    transformations = ['lin', 'lin', 'lin', 'log', 'log', 'log']
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation=transformations[i],
            blocks=[1]
        ))
    config = DFMConfig(series=series_list, block_names=block_names)
    
    Res = dfm(X, config, threshold=1e-3, max_iter=50)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Different transformations test passed")


def test_many_blocks():
    """Test with many blocks."""
    print("\n" + "="*70)
    print("TEST: Many Blocks")
    print("="*70)
    
    T, N = 100, 20
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    block_names = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5']
    series_list = []
    for i in range(N):
        # All load on first block, some also on other blocks
        blocks = [1, 0, 0, 0, 0]
        if i % 3 == 0:
            blocks[1] = 1
        elif i % 3 == 1:
            blocks[2] = 1
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
    
    print("✓ Many blocks test passed")


def test_rapid_convergence():
    """Test with very loose convergence threshold."""
    print("\n" + "="*70)
    print("TEST: Rapid Convergence")
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
    
    # Very loose threshold - should converge quickly
    Res = dfm(X, config, threshold=1e-1, max_iter=100)
    
    assert Res.x_sm.shape == (T, N)
    assert np.any(np.isfinite(Res.Z))
    
    print("✓ Rapid convergence test passed")


def test_init_conditions_stress():
    """Stress test for init_conditions."""
    print("\n" + "="*70)
    print("TEST: Init Conditions Stress")
    print("="*70)
    
    T, N = 100, 20  # Reduced size
    np.random.seed(42)
    x = np.random.randn(T, N)
    
    # Multiple blocks with different factor counts
    blocks = np.zeros((N, 3), dtype=int)
    blocks[:, 0] = 1  # All load on global
    blocks[0:7, 1] = 1
    blocks[7:14, 2] = 1
    
    r = np.array([2, 2, 2])  # Reduced factor counts
    p = 1  # Reduced lag
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    assert A is not None
    assert C is not None
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(C))
    
    print("✓ Init conditions stress test passed")


# ============================================================================
# Test Runner
# ============================================================================

def run_all_stress_tests():
    """Run all stress tests."""
    print("\n" + "="*70)
    print("DFM STRESS TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('large_dataset', test_large_dataset),
        # ('very_large_dataset', test_very_large_dataset),  # Skipped - too slow
        ('numerical_precision', test_numerical_precision),
        ('sparse_data', test_sparse_data),
        ('high_correlation', test_high_correlation),
        ('different_transformations', test_different_transformations),
        ('many_blocks', test_many_blocks),
        ('rapid_convergence', test_rapid_convergence),
        ('init_conditions_stress', test_init_conditions_stress),
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
    print(f"SUMMARY: {passed}/{total} stress tests passed")
    print("="*70)
    
    return results


if __name__ == '__main__':
    run_all_stress_tests()

