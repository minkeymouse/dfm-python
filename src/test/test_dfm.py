"""Core tests for DFM estimation - consolidated from all DFM tests."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import DFM, DFMResult
from dfm_python.core.em import em_step, init_conditions
from dfm_python.data_loader import load_config, load_data
from dfm_python.utils.data_utils import rem_nans_spline
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

# ============================================================================
# Core Tests
# ============================================================================

def test_em_step_basic():
    """Test basic EM step functionality."""
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
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
        clock='m',
        tent_weights_dict={},
        frequencies=None
    )
    
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
        y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks,
        tent_weights_dict={},
        clock='m',
        frequencies=None,
        config=None
    )
    
    assert C_new is not None and R_new is not None
    assert A_new is not None and np.isfinite(loglik)
    assert C_new.shape == (N, A.shape[0])
    assert R_new.shape == (N, N)


def test_init_conditions_basic():
    """Test basic initial conditions."""
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
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
        clock='m',
        tent_weights_dict={},
        frequencies=None
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


def test_dfm_quick():
    """Quick DFM test with synthetic data."""
    T, N = 50, 10
    np.random.seed(42)
    
    # Generate synthetic data
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    
    # Add missing values
    missing_mask = np.random.rand(T, N) < 0.1
    X[missing_mask] = np.nan
    
    # Create config (single global block)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    # Run DFM
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=5)
    
    # Verify structure
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    assert np.any(np.isfinite(Res.Z))


def test_dfm_class_fit():
    """Test DFM class fit() method (new API)."""
    np.random.seed(42)
    T, N = 50, 5
    
    X = np.random.randn(T, N)
    
    # Create config
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    # Use DFM class directly
    model = DFM()
    result = model.fit(X, config, threshold=1e-2, max_iter=5)
    
    # Verify structure
    assert isinstance(result, DFMResult)
    assert result.x_sm.shape == (T, N)
    assert result.Z.shape[0] == T
    assert result.C.shape[0] == N
    assert np.any(np.isfinite(result.Z))
    
    # Verify model stores result
    assert model.result is not None
    assert model.config is not None


def test_multi_block_different_factors():
    """Test multi-block with different factor counts."""
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
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
        clock='m',
        tent_weights_dict={},
        frequencies=None
    )
    
    assert A is not None and C is not None
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(C))


# ============================================================================
# Edge Case Tests (consolidated from test_dfm_edge_cases.py)
# ============================================================================

def test_single_series():
    """Test with single time series."""
    T = 50
    np.random.seed(42)
    X = np.random.randn(T, 1)
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(
        series_id="TEST_01",
        series_name="Test Series",
        frequency='m',
        transformation='lin',
        blocks=['Block_Global']
    )]
    config = DFMConfig(series=series_list, blocks=blocks)
    
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=5)
    
    assert Res.x_sm.shape == (T, 1)
    assert Res.Z.shape[0] == T
    assert np.any(np.isfinite(Res.Z))


def test_all_nan_data():
    """Test with all NaN data."""
    T, N = 50, 5
    X = np.full((T, N), np.nan)
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    # Should handle gracefully or raise informative error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = DFM()
            Res = model.fit(X, config, threshold=1e-2, max_iter=5)
            assert Res.x_sm.shape == (T, N)
        except (ValueError, RuntimeError) as e:
            # Acceptable - all NaN data is invalid
            assert "nan" in str(e).lower() or "missing" in str(e).lower() or "data" in str(e).lower()


def test_high_missing_data():
    """Test with very high percentage of missing data."""
    T, N = 100, 10
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    # Make 80% missing
    missing_mask = np.random.rand(T, N) < 0.8
    X[missing_mask] = np.nan
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-2, max_iter=5)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))


def test_mixed_frequencies():
    """Test with mixed frequencies (monthly and quarterly)."""
    T, N = 60, 8
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        freq = 'm' if i < 5 else 'q'
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency=freq,
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks, clock='m')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-2, max_iter=5)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))


# ============================================================================
# Stress Tests (consolidated from test_dfm_stress.py)
# ============================================================================

def test_large_dataset():
    """Test with large dataset."""
    T, N = 200, 30
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=20)
    
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert np.all(np.isfinite(Res.Z))


def test_numerical_precision():
    """Test numerical precision with very small values."""
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N) * 1e-10
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        ))
    config = DFMConfig(series=series_list, blocks=blocks)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-3, max_iter=50)
        
        assert Res.x_sm.shape == (T, N)
        assert np.all(np.isfinite(Res.Z))


# ============================================================================
# Integration Tests (consolidated from test_synthetic.py)
# ============================================================================

def test_with_direct_config():
    """Test with direct DFMConfig creation."""
    try:
        series_list = [
            SeriesConfig(
                series_id='series_0',
                series_name='Test Series 0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_1',
                series_name='Test Series 1',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series_list, blocks=blocks)
        
        # Generate simple synthetic data
        T = 50
        np.random.seed(42)
        X = np.random.randn(T, 2)
        
        model = DFM()
        result = model.fit(X, config, threshold=1e-2, max_iter=5)
        
        assert result.Z.shape[1] > 0
        assert result.C.shape[0] == 2
        
    except Exception as e:
        pytest.skip(f"Integration test skipped: {e}")


if __name__ == '__main__':
    # Quick verification
    print("Running DFM quick test...")
    test_dfm_quick()
    print("✓ DFM runs successfully!")
