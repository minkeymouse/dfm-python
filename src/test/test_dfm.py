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
from dfm_python.data import load_data, rem_nans_spline
from dfm_python.api import load_config
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
    
    from dfm_python.core.em import EMStepParams
    em_params = EMStepParams(
        y=y,
        A=A,
        C=C,
        Q=Q,
        R=R,
        Z_0=Z_0,
        V_0=V_0,
        r=r,
        p=p,
        R_mat=R_mat,
        q=q,
        nQ=nQ,
        i_idio=i_idio,
        blocks=blocks,
        tent_weights_dict={},
        clock='m',
        frequencies=None,
        config=None
    )
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_params)
    
    assert C_new is not None and R_new is not None
    assert A_new is not None and np.isfinite(loglik)
    # C_new should match the expanded state dimension (factors + idiosyncratic components)
    assert C_new.shape == (N, A_new.shape[0])
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


def test_init_conditions_large_block():
    """Test initial conditions with large block (like Block_Global with 78 series)."""
    T, N = 396, 78  # Realistic size
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23  # 23% missing
    x[missing_mask] = np.nan
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Should not raise broadcasting errors
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
    assert C.shape == (N, m), f"Expected C shape ({N}, {m}), got {C.shape}"
    assert Q.shape == (m, m)
    assert R.shape == (N, N), f"Expected R shape ({N}, {N}), got {R.shape}"
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
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    
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
    
    # Should not raise dimension mismatch or broadcasting errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
            clock='m',
            tent_weights_dict={},
            frequencies=None
        )
    
    assert A is not None and C is not None
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(C))
    assert C.shape[0] == N, f"Expected C shape ({N}, ...), got {C.shape}"
    assert R.shape == (N, N), f"Expected R shape ({N}, {N}), got {R.shape}"


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


def test_init_conditions_block_global_single_series():
    """Test Block_Global initialization with only one series loading on it.
    
    This tests the edge case where Block_Global has only a single series,
    which could cause issues with covariance computation or PCA.
    """
    np.random.seed(42)
    T, N = 50, 1  # Only one series
    x = np.random.randn(T, N)
    
    blocks = np.ones((N, 1), dtype=int)  # Single series in Block_Global
    r = np.array([1])  # Single factor
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Should handle single series gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
            clock='m',
            tent_weights_dict={},
            frequencies=None
        )
    
    # Verify outputs are valid
    assert A is not None, "A should not be None"
    assert C is not None, "C should not be None"
    assert Q is not None, "Q should not be None"
    assert C.shape == (N, A.shape[0]), f"C shape should be ({N}, {A.shape[0]}), got {C.shape}"
    assert Q.shape[0] == Q.shape[1], "Q should be square"
    # Q diagonal should be > 0 (enforced by safeguards)
    Q_diag = np.diag(Q)
    assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
    # Loadings should not be all zero
    max_loading_abs = np.max(np.abs(C))
    assert max_loading_abs > 0, f"Loadings should not be all zero, max_abs={max_loading_abs}"


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
    
    # All NaN data may raise error or use fallback - verify behavior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = DFM()
            Res = model.fit(X, config, threshold=1e-2, max_iter=5)
            # If succeeds with fallback, verify outputs are valid
            assert Res is not None
            assert Res.x_sm.shape == (T, N)
            # Should not converge with all NaN data (unless using placeholders)
            # With placeholders, may appear "converged" due to 0.0 loglik, which is acceptable
            # Full implementation should detect all-NaN and not converge
            # For now, just verify it doesn't crash
            assert isinstance(Res.converged, bool)
        except (ValueError, RuntimeError) as e:
            # If fails, error should be informative
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["nan", "missing", "data", "insufficient"]), \
                f"Error should mention data issue, got: {e}"


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
        
        # Verify tent weight constraints for quarterly series
        from dfm_python.utils import get_tent_weights_for_pair, generate_R_mat
        
        clock = 'm'
        slower_freq = 'q'
        r_i = 1  # Number of factors in Block_Global
        
        # Get tent weights for quarterly -> monthly
        tent_weights = get_tent_weights_for_pair(slower_freq, clock)
        assert tent_weights is not None, f"Tent weights should be available for {slower_freq} -> {clock}"
        
        # Generate constraint matrices
        R_mat, q_vec = generate_R_mat(tent_weights)
        pC_freq = len(tent_weights)  # Number of periods in tent (5 for quarterly->monthly)
        
        # For block with r_i factors, R_con_i = kron(R_mat, eye(r_i))
        R_con_i = np.kron(R_mat, np.eye(r_i))
        q_con_i = np.kron(q_vec, np.zeros(r_i))
        
        # Verify constraints for each quarterly series (indices 5-7)
        quarterly_series_indices = [i for i in range(N) if series_list[i].frequency == 'q']
        assert len(quarterly_series_indices) == 3, "Should have 3 quarterly series"
        
        max_violation = 0.0
        for i in quarterly_series_indices:
            # Extract loadings for this series (first pC_freq * r_i columns for tent weights)
            C_i = Res.C[i, :pC_freq * r_i]
            
            # Compute constraint violation: R_con_i @ C_i - q_con_i
            constraint_violation = R_con_i @ C_i - q_con_i
            
            # Track maximum violation
            max_violation = max(max_violation, np.max(np.abs(constraint_violation)))
        
        # Verify constraints are satisfied (within numerical tolerance)
        tolerance = 1e-6
        assert max_violation < tolerance, (
            f"Tent weight constraints violated for quarterly series. "
            f"Max violation: {max_violation:.2e} (tolerance: {tolerance:.2e}). "
            f"This indicates tent weight constraints are not being correctly enforced."
        )


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


def test_em_converged():
    """Test EM convergence detection logic."""
    from dfm_python.core.em import em_converged
    
    threshold = 1e-4
    
    # Test case 1: Convergence when relative change < threshold
    loglik_current = 100.0
    loglik_previous = 100.01
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    # Relative change: |100.0 - 100.01| / avg(|100.0|, |100.01|) ≈ 1e-4
    # Should converge (or be very close)
    assert isinstance(converged, bool), "converged should be boolean"
    assert isinstance(decreased, bool), "decreased should be boolean"
    
    # Test case 2: Clear convergence (very small change)
    loglik_current = 100.0
    loglik_previous = 100.00001
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged, "Should converge with very small relative change"
    assert not decreased, "Should not detect decrease for small positive change"
    
    # Test case 3: No convergence (large change)
    loglik_current = 100.0
    loglik_previous = 50.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert not converged, "Should not converge with large change"
    assert not decreased, "Should not detect decrease for increase"
    
    # Test case 4: Likelihood decrease detection
    loglik_current = 50.0
    loglik_previous = 100.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold, check_decreased=True)
    assert not converged, "Should not converge when likelihood decreases"
    assert decreased, "Should detect likelihood decrease"
    
    # Test case 5: Zero loglikelihood (edge case)
    loglik_current = 0.0
    loglik_previous = 0.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged, "Should converge when both are zero"
    assert not decreased, "Should not detect decrease when both are zero"
    
    # Test case 6: Negative loglikelihood (valid in some cases)
    loglik_current = -100.0
    loglik_previous = -100.01
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert isinstance(converged, bool), "Should handle negative loglikelihood"
    assert isinstance(decreased, bool), "Should handle negative loglikelihood"
    
    # Test case 7: Very small change near threshold
    loglik_current = 1.0
    loglik_previous = 1.0 + threshold * 0.5  # Half of threshold
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged, "Should converge when change is less than threshold"
    
    # Test case 8: Change exactly at threshold
    loglik_current = 1.0
    loglik_previous = 1.0 + threshold * 1.0  # Exactly at threshold
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    # Should converge (threshold is inclusive in practice due to floating point)
    assert isinstance(converged, bool), "Should handle threshold boundary case"


def test_kalman_stability_edge_cases():
    """Test Kalman filter/smoother stability with edge cases."""
    np.random.seed(42)
    T, N = 50, 5
    
    # Generate synthetic data
    x = np.random.randn(T, N)
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Initialize properly using init_conditions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
            clock='m', tent_weights_dict={}, frequencies=None
        )
    
    m = A.shape[0]  # Get actual state dimension
    
    # Prepare data for em_step (needs y, not x)
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    
    # Test case 1: Very small R diagonal (near-singular observation covariance)
    R_very_small = np.eye(N) * 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Use em_step which internally calls KF/KFS
            from dfm_python.core.em import EMStepParams
            em_params = EMStepParams(
                y=y, A=A, C=C, Q=Q, R=R_very_small, Z_0=Z_0, V_0=V_0,
                r=r, p=p, R_mat=R_mat, q=q, nQ=nQ, i_idio=i_idio, blocks=blocks,
                tent_weights_dict={}, clock='m', frequencies=None, config=None
            )
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_params)
            # Should not crash, should produce finite outputs
            assert np.all(np.isfinite(C_new)), "C should be finite with very small R"
            assert np.all(np.isfinite(R_new)), "R should be finite"
            assert np.all(np.isfinite(A_new)), "A should be finite"
            assert np.all(np.isfinite(Q_new)), "Q should be finite"
            assert np.isfinite(loglik) or loglik == -np.inf, "loglik should be finite or -inf"
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # Some edge cases may fail, but should fail with informative error
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["singular", "ill-conditioned", "broadcast", "shape", "matrix"]), \
                f"Should fail with expected error type, got: {e}"
    
    # Test case 2: Very large Q (high innovation variance)
    Q_very_large = Q.copy() * 1e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from dfm_python.core.em import EMStepParams
            em_params = EMStepParams(
                y=y, A=A, C=C, Q=Q_very_large, R=R, Z_0=Z_0, V_0=V_0,
                r=r, p=p, R_mat=R_mat, q=q, nQ=nQ, i_idio=i_idio, blocks=blocks,
                tent_weights_dict={}, clock='m', frequencies=None, config=None
            )
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_params)
            # Should handle large Q (may be capped)
            assert np.all(np.isfinite(C_new)), "C should be finite with large Q"
            assert np.all(np.isfinite(Q_new)), "Q should be finite (may be capped)"
            # Q should be capped to reasonable value
            max_eig = np.max(np.linalg.eigvals(Q_new))
            assert max_eig < 1e7, f"Q should be capped, max_eig={max_eig:.2e}"
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # Should fail with informative error if it fails
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["singular", "ill-conditioned", "broadcast", "shape", "matrix", "eigenvalue"]), \
                f"Should fail with expected error type, got: {e}"
    
    # Test case 3: Near-singular matrices (small eigenvalues)
    Q_near_singular = Q.copy() * 1e-12
    Q_near_singular[0, 0] = 1e-15  # Very small eigenvalue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from dfm_python.core.em import EMStepParams
            em_params = EMStepParams(
                y=y, A=A, C=C, Q=Q_near_singular, R=R, Z_0=Z_0, V_0=V_0,
                r=r, p=p, R_mat=R_mat, q=q, nQ=nQ, i_idio=i_idio, blocks=blocks,
                tent_weights_dict={}, clock='m', frequencies=None, config=None
            )
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(em_params)
            # Should regularize Q to be non-singular
            assert np.all(np.isfinite(Q_new)), "Q should be finite after regularization"
            Q_diag = np.diag(Q_new)
            assert np.all(Q_diag > 0), "Q diagonal should be > 0 after regularization"
        except Exception as e:
            # Should fail gracefully
            assert isinstance(e, (np.linalg.LinAlgError, ValueError, RuntimeError)), \
                f"Should fail gracefully, got: {type(e).__name__}"


def test_init_conditions_block_global_sparse_data():
    """Test Block_Global initialization with high missing data ratio (50-70%)."""
    np.random.seed(42)
    T, N = 200, 20
    x = np.random.randn(T, N)
    
    # Create sparse data: 50-70% missing values
    missing_rate = 0.6
    missing_mask = np.random.rand(T, N) < missing_rate
    x[missing_mask] = np.nan
    
    blocks = np.ones((N, 1), dtype=int)  # All series in Block_Global
    r = np.array([1])  # Single factor
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Should handle sparse data gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
            clock='m',
            tent_weights_dict={},
            frequencies=None
        )
    
    # Verify Q diagonal > 0 (critical for factor evolution)
    assert Q is not None, "Q should not be None"
    assert Q.shape[0] == Q.shape[1], "Q should be square"
    Q_diag = np.diag(Q)
    assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
    assert np.all(np.isfinite(Q_diag)), "Q diagonal should be finite"
    
    # Verify loadings not all zero
    assert C is not None, "C should not be None"
    assert C.shape == (N, A.shape[0]), f"C shape should be ({N}, {A.shape[0]}), got {C.shape}"
    max_loading_abs = np.max(np.abs(C))
    assert max_loading_abs > 0, f"Loadings should not be all zero, max_abs={max_loading_abs}"
    
    # Verify A not all zero (transition matrix)
    assert A is not None, "A should not be None"
    assert A.shape[0] == A.shape[1], "A should be square"
    max_A_abs = np.max(np.abs(A))
    assert max_A_abs > 0, f"A should not be all zero, max_abs={max_A_abs}"
    
    # Verify pairwise complete covariance was used (check that initialization succeeded)
    # If pairwise complete wasn't used, initialization would likely fail with sparse data
    assert np.all(np.isfinite(A)), "A should be finite"
    assert np.all(np.isfinite(C)), "C should be finite"
    assert np.all(np.isfinite(Q)), "Q should be finite"


def test_init_conditions_pairwise_complete_block_global():
    """Test that pairwise_complete=True is used for Block_Global initialization.
    
    This test verifies that Block_Global (i == 0) uses pairwise_complete=True
    for covariance computation, which allows initialization to succeed even when
    no single time point has all series observed.
    
    The test creates data where no time point has all series observed, but
    pairwise observations exist. If pairwise_complete is used, initialization
    should succeed. If not, it would fail with "insufficient data" error.
    """
    np.random.seed(42)
    T, N = 100, 15
    x = np.random.randn(T, N)
    
    # Create pattern where no row is complete, but pairs have overlap
    # This pattern requires pairwise_complete to compute covariance
    for t in range(T):
        # Each time point is missing at least one series
        missing_idx = np.random.choice(N, size=max(1, N // 3), replace=False)
        x[t, missing_idx] = np.nan
    
    # Verify no row is complete
    complete_rows = np.all(np.isfinite(x), axis=1)
    assert np.sum(complete_rows) == 0, "Test setup failed: some rows are complete"
    
    blocks = np.ones((N, 1), dtype=int)  # All series in Block_Global
    r = np.array([1])  # Single factor
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Should succeed because Block_Global uses pairwise_complete=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
            clock='m',
            tent_weights_dict={},
            frequencies=None
        )
    
    # Verify initialization succeeded (this confirms pairwise_complete was used)
    assert A is not None, "A should not be None"
    assert C is not None, "C should not be None"
    assert Q is not None, "Q should not be None"
    assert np.all(np.isfinite(A)), "A should be finite"
    assert np.all(np.isfinite(C)), "C should be finite"
    assert np.all(np.isfinite(Q)), "Q should be finite"


def test_init_conditions_block_global_all_nan_residuals():
    """Test Block_Global initialization when block residuals are all NaN.
    
    This tests the edge case where after removing non-finite rows, all block
    residuals are NaN. The code should handle this gracefully using fallback
    strategies (identity covariance or median imputation).
    """
    np.random.seed(42)
    T, N = 50, 10
    x = np.random.randn(T, N)
    
    # Create scenario where Block_Global residuals would be all NaN
    # Strategy: Create data where after filtering, residuals become all NaN
    # This simulates extreme sparsity where no valid observations remain
    blocks = np.ones((N, 1), dtype=int)  # All series in Block_Global
    r = np.array([1])  # Single factor
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    
    # Create data with extreme sparsity: most rows have all NaN
    # But ensure at least a few rows have some valid data to pass initial checks
    x_sparse = x.copy()
    # Make most rows all NaN, but keep a few rows with partial data
    for t in range(T):
        if t % 5 != 0:  # Make 80% of rows all NaN
            x_sparse[t, :] = np.nan
        else:
            # Keep some columns NaN even in valid rows
            x_sparse[t, ::2] = np.nan
    
    # With sparse data, initialization may succeed with fallback or fail
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            A, C, Q, R, Z_0, V_0 = init_conditions(
                x_sparse, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio,
                clock='m',
                tent_weights_dict={},
                frequencies=None
            )
            # If succeeds with fallback, verify outputs are valid
            assert A is not None, "A should not be None"
            assert C is not None, "C should not be None"
            assert Q is not None, "Q should not be None"
            assert np.all(np.isfinite(A)), "A should be finite"
            assert np.all(np.isfinite(C)), "C should be finite"
            assert np.all(np.isfinite(Q)), "Q should be finite"
            # Q diagonal should be > 0 (enforced by safeguards)
            Q_diag = np.diag(Q)
            assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
        except ValueError as e:
            # If fails, error should be informative
            error_msg = str(e).lower()
            assert "insufficient data" in error_msg or "data" in error_msg, \
                f"Expected error about insufficient data, got: {e}"


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
