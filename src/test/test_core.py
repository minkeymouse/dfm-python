"""Core tests for DFM estimation, EM algorithm, and numeric utilities."""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.api import DFM
from dfm_python.dfm import DFMResult
from dfm_python.core.em import em_step, init_conditions, em_converged
from dfm_python.data import load_data, rem_nans_spline
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.core.numeric import (
    _compute_covariance_safe,
    _compute_variance_safe,
    _ensure_innovation_variance_minimum,
    _ensure_real_and_symmetric,
    _ensure_covariance_stable,
)

# ============================================================================
# EM Algorithm Tests
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
    
    from dfm_python.core.em import EMStepParams
    idio_chain_lengths = np.zeros(N)
    em_params = EMStepParams(
        y=y, A=A, C=C, Q=Q, R=R, Z_0=Z_0, V_0=V_0,
        r=r, p=p, R_mat=R_mat, q=q, nQ=nQ, i_idio=i_idio, blocks=blocks,
        tent_weights_dict={}, clock='m', frequencies=None,
        idio_chain_lengths=idio_chain_lengths, config=None
    )
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_params)
    
    assert C_new is not None and R_new is not None
    assert A_new is not None and np.isfinite(loglik)
    assert C_new.shape == (N, A_new.shape[0])
    assert R_new.shape == (N, N)


def test_init_conditions_basic():
    """Test basic initial conditions."""
    T, N = 50, 5
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def test_em_converged():
    """Test EM convergence detection logic."""
    threshold = 1e-4
    
    # Test case 1: Convergence when relative change < threshold
    loglik_current = 100.0
    loglik_previous = 100.01
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert isinstance(converged, bool)
    assert isinstance(decreased, bool)
    
    # Test case 2: Clear convergence (very small change)
    loglik_current = 100.0
    loglik_previous = 100.00001
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged
    assert not decreased
    
    # Test case 3: No convergence (large change)
    loglik_current = 100.0
    loglik_previous = 50.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert not converged
    assert not decreased
    
    # Test case 4: Likelihood decrease detection
    loglik_current = 50.0
    loglik_previous = 100.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold, check_decreased=True)
    assert not converged
    assert decreased


# ============================================================================
# DFM Estimation Tests
# ============================================================================

def test_dfm_quick():
    """Quick DFM test with synthetic data."""
    T, N = 50, 10
    np.random.seed(42)
    
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    
    missing_mask = np.random.rand(T, N) < 0.1
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
    
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=5)
    
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    assert np.any(np.isfinite(Res.Z))


def test_dfm_class_fit():
    """Test DFM class fit() method."""
    np.random.seed(42)
    T, N = 50, 5
    
    X = np.random.randn(T, N)
    
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
    
    model = DFM()
    result = model.fit(X, config, threshold=1e-2, max_iter=5)
    
    assert isinstance(result, DFMResult)
    assert result.x_sm.shape == (T, N)
    assert result.Z.shape[0] == T
    assert result.C.shape[0] == N
    assert np.any(np.isfinite(result.Z))
    assert model.result is not None
    assert model.config is not None


def test_mixed_frequencies():
    """Test with mixed frequencies (monthly and quarterly)."""
    T, N = 40, 5
    np.random.seed(42)
    X = np.random.randn(T, N)
    
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        freq = 'm' if i < 2 else 'q'
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
        Res = model.fit(X, config, threshold=1e-2, max_iter=2)
        
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))


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


# ============================================================================
# Numeric Utilities Tests
# ============================================================================

def test_compute_covariance_safe_basic():
    """Test basic covariance computation."""
    np.random.seed(42)
    T, N = 50, 5
    data = np.random.randn(T, N)
    
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=False)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)


def test_compute_covariance_safe_pairwise_complete():
    """Test pairwise complete covariance computation."""
    np.random.seed(42)
    T, N = 50, 5
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.2
    data[missing_mask] = np.nan
    
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)


def test_ensure_innovation_variance_minimum():
    """Test innovation variance minimum enforcement."""
    np.random.seed(42)
    m = 5
    Q = np.random.randn(m, m)
    Q = Q @ Q.T
    Q[0, 0] = 0.0
    
    Q_fixed = _ensure_innovation_variance_minimum(Q, min_variance=1e-8)
    
    assert Q_fixed.shape == (m, m)
    assert np.all(np.diag(Q_fixed) >= 1e-8)
    assert Q_fixed[0, 0] >= 1e-8


def test_ensure_real_and_symmetric():
    """Test real and symmetric enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m) + 1j * np.random.randn(m, m) * 1e-10
    
    M_fixed = _ensure_real_and_symmetric(M)
    
    assert np.isrealobj(M_fixed)
    assert np.allclose(M_fixed, M_fixed.T)


def test_ensure_covariance_stable():
    """Test covariance stability enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m)
    M = M @ M.T
    eigenvals = np.linalg.eigvalsh(M)
    M = M - np.eye(m) * (np.min(eigenvals) + 0.1)
    
    M_fixed = _ensure_covariance_stable(M, min_eigenval=1e-8)
    
    eigenvals_fixed = np.linalg.eigvalsh(M_fixed)
    assert np.all(eigenvals_fixed >= 1e-8 * 0.99)

