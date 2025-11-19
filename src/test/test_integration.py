"""Integration tests for factor extraction, stability, and edge cases."""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import pytest
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.api import DFM
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig, Params
import dfm_python as dfm


# ============================================================================
# Factor Extraction Tests
# ============================================================================

def test_factor_not_near_zero():
    """Test that factors are not near-zero after training."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Test data files not found")
    
    params = Params(max_iter=2, threshold=1e-3)
    import polars as pl
    spec_df = pl.read_csv(spec_file)
    dfm.from_spec_df(spec_df, params=params)
    dfm.load_data(data_file, sample_start='2015-01-01', sample_end='2022-12-31')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfm.train()
    
    result = dfm.get_result()
    assert result is not None
    
    Z = result.Z
    Q = result.Q
    C = result.C
    
    if Z.shape[1] > 0:
        factor0 = Z[:, 0]
        assert np.std(factor0) > 1e-10
        assert np.abs(np.mean(factor0)) < 1e6 or np.std(factor0) > 1e-10
    
    if Q.shape[0] > 0:
        assert Q[0, 0] >= 1e-8
    
    if C.shape[1] > 0:
        loadings_factor0 = C[:, 0]
        max_loading = np.abs(loadings_factor0).max()
        assert max_loading > 1e-6


def test_factor_innovation_variance():
    """Test that innovation variances are above minimum threshold."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Test data files not found")
    
    params = Params(max_iter=2, threshold=1e-3)
    import polars as pl
    spec_df = pl.read_csv(spec_file)
    dfm.from_spec_df(spec_df, params=params)
    dfm.load_data(data_file, sample_start='2020-01-01', sample_end='2022-12-31')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfm.train()
    
    result = dfm.get_result()
    assert result is not None
    
    Q = result.Q
    Q_diag = np.diag(Q)
    assert np.all(Q_diag >= 1e-8)


# ============================================================================
# Stability Tests
# ============================================================================

def test_stability_guards():
    """Test that numerical stability guards are enforced."""
    T, N = 80, 8
    np.random.seed(42)
    x = np.random.randn(T, N)
    
    series = [
        SeriesConfig(series_id=f's{i}', frequency='m', transformation='lin', blocks=[1])
        for i in range(N)
    ]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series, blocks=blocks, augment_idio=True, idio_min_var=1e-8)
    
    model = DFM()
    result = model.fit(x, config, max_iter=2, threshold=1e-3)
    
    Q_diag = np.diag(result.Q)
    assert np.all(Q_diag >= 1e-8)
    
    R_diag = np.diag(result.R)
    assert np.all(R_diag >= 1e-8)
    
    eigenvals_A = np.linalg.eigvals(result.A)
    max_eig_A = np.max(np.abs(eigenvals_A))
    assert max_eig_A < 1.0
    
    assert np.isfinite(result.loglik)


# ============================================================================
# Edge Cases
# ============================================================================

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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = DFM()
            Res = model.fit(X, config, threshold=1e-2, max_iter=2)
            assert Res is not None
            assert Res.x_sm.shape == (T, N)
            assert isinstance(Res.converged, bool)
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["nan", "missing", "data", "insufficient"])


def test_high_missing_data():
    """Test with very high percentage of missing data."""
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
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
        Res = model.fit(X, config, threshold=1e-2, max_iter=2)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))

