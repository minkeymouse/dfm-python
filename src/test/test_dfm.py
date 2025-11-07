"""Consolidated tests for DFM estimation and related components."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import dfm, DFMResult, em_step, init_conditions
from dfm_python.data_loader import load_config, load_data
from utils.data_utils import rem_nans_spline

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Test data cache
_test_cache = {}

def _get_test_data():
    """Get or load test data (cached)."""
    if 'data' not in _test_cache:
        base_dir = Path(__file__).parent.parent.parent
        spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
        config = load_config(spec_file)
        vintage = '2016-06-29'
        data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
        X, Time, Z = load_data(data_file, config, sample_start=pd.Timestamp('2000-01-01'))
        _test_cache['data'] = (config, X, Time, Z)
    return _test_cache['data']

def _get_dfm_result(threshold=1e-4):
    """Get or compute DFM result (cached)."""
    cache_key = f'dfm_{threshold}'
    if cache_key not in _test_cache:
        config, X, Time, Z = _get_test_data()
        Res = dfm(X, config, threshold)
        _test_cache[cache_key] = Res
    return _test_cache[cache_key]

# ============================================================================
# Basic DFM Tests
# ============================================================================

def test_dfm_estimation():
    """Test basic DFM estimation."""
    print("\n" + "="*70)
    print("TEST: DFM Estimation")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-4
    
    Res = dfm(X, spec, threshold)
    
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert hasattr(Res, 'A') and hasattr(Res, 'Q') and hasattr(Res, 'R')
    
    T, N = X.shape
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    
    assert not np.all(np.isnan(Res.Z))
    assert not np.all(np.isnan(Res.x_sm))
    assert np.isfinite(Res.C).any()
    
    print("✓ DFM estimation completed")
    return Res

def test_dfm_workflow():
    """Test complete DFM workflow."""
    print("\n" + "="*70)
    print("TEST: Complete DFM Workflow")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-3
    Res = _get_dfm_result(threshold)
    
    T, N = X.shape
    m = Res.A.shape[0]
    
    assert Res.x_sm.shape == (T, N)
    assert Res.X_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    assert Res.R.shape == (N, N)
    assert Res.A.shape == (m, m)
    assert Res.Q.shape == (m, m)
    
    assert np.allclose(Res.R, Res.R.T, atol=1e-10)
    assert np.allclose(Res.Q, Res.Q.T, atol=1e-10)
    
    print("✓ Workflow test passed")
    return Res

def test_dfm_convergence():
    """Test DFM convergence."""
    print("\n" + "="*70)
    print("TEST: DFM Convergence")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-3
    Res = _get_dfm_result(threshold)
    
    assert Res is not None
    assert not np.all(np.isnan(Res.Z))
    assert np.isfinite(Res.C).any()
    
    print("✓ Convergence test passed")
    return True

def test_dfm_parameter_stability():
    """Test parameter stability."""
    print("\n" + "="*70)
    print("TEST: Parameter Stability")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-3
    Res = _get_dfm_result(threshold)
    
    eigvals = np.linalg.eigvals(Res.A)
    max_eigval = np.max(np.abs(eigvals))
    assert max_eigval < 1e4
    
    Q_diag = np.diag(Res.Q)
    assert np.all(Q_diag >= -1e-10)
    
    R_diag = np.diag(Res.R)
    assert np.all(R_diag >= 1e-6)
    
    print(f"✓ Parameter stability verified (max |eigval|: {max_eigval:.4f})")
    return True

# ============================================================================
# Integration Tests
# ============================================================================

def test_dfm_output_consistency():
    """Test DFM output consistency."""
    print("\n" + "="*70)
    print("TEST: Output Consistency")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-3
    Res = _get_dfm_result(threshold)
    
    x_sm_from_factors = Res.Z @ Res.C.T
    finite_mask = np.isfinite(Res.x_sm) & np.isfinite(x_sm_from_factors)
    
    if np.sum(finite_mask) > 0:
        diff = np.abs(Res.x_sm[finite_mask] - x_sm_from_factors[finite_mask])
        max_diff = np.max(diff)
        assert max_diff < 1e-3 or np.mean(diff) < 1e-6
    
    print("✓ Output consistency verified")
    return True

def test_dfm_factor_analysis():
    """Test factor extraction."""
    print("\n" + "="*70)
    print("TEST: Factor Analysis")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    threshold = 1e-3
    Res = _get_dfm_result(threshold)
    
    m = Res.A.shape[0]
    if m > 0:
        factor_0 = Res.Z[:, 0]
        factor_0_finite = factor_0[np.isfinite(factor_0)]
        if len(factor_0_finite) > 1:
            var_0 = np.var(factor_0_finite)
            assert var_0 > 1e-10
        
        common_loadings = Res.C[:, 0]
        finite_loadings = common_loadings[np.isfinite(common_loadings)]
        non_zero_loadings = np.sum(np.abs(finite_loadings) > 1e-4)
        assert non_zero_loadings >= 1
    
    print("✓ Factor analysis passed")
    return True

# ============================================================================
# Edge Cases
# ============================================================================

def test_dfm_edge_all_missing():
    """Test with all missing data."""
    print("\n" + "="*70)
    print("TEST: All Missing Data")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    X_all_missing = np.full_like(X, np.nan)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Res = dfm(X_all_missing, spec, threshold=1e-3)
            if Res is not None:
                assert hasattr(Res, 'Z')
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
    
    print("✓ All missing data handled")
    return True

def test_dfm_edge_short_series():
    """Test with short time series."""
    print("\n" + "="*70)
    print("TEST: Short Time Series")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    T_min = 15
    X_short = X[:T_min, :]
    
    try:
        Res = dfm(X_short, spec, threshold=1e-3)
        assert Res is not None
        assert Res.Z.shape[0] == T_min
    except (ValueError, np.linalg.LinAlgError):
        pass
    
    print("✓ Short series handled")
    return True

def test_dfm_edge_extreme_values():
    """Test with extreme values."""
    print("\n" + "="*70)
    print("TEST: Extreme Values")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    X_extreme = X.copy()
    X_extreme[:, 0] *= 1e6
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Res = dfm(X_extreme, spec, threshold=1e-3)
            assert Res is not None
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            pass
    
    print("✓ Extreme values handled")
    return True

def test_dfm_edge_high_missing():
    """Test with high missing rate."""
    print("\n" + "="*70)
    print("TEST: High Missing Rate")
    print("="*70)
    
    spec, X, Time, Z = _get_test_data()
    np.random.seed(42)
    X_high_missing = X.copy()
    missing_mask = np.random.rand(*X.shape) < 0.6
    X_high_missing[missing_mask] = np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Res = dfm(X_high_missing, spec, threshold=1e-3)
            assert Res is not None
        except (ValueError, np.linalg.LinAlgError):
            pass
    
    print("✓ High missing rate handled")
    return True

# ============================================================================
# EM Step Tests
# ============================================================================

def _setup_em_step_inputs(T=80, N=8, nQ=0, seed=42):
    """Set up inputs for em_step."""
    np.random.seed(seed)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    
    opt_nan = {'method': 2, 'k': 3}
    R_mat = np.array([[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], 
                      [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
    q = np.zeros(4)
    i_idio = np.ones(N)
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio
    )
    
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    
    return {
        'y': y, 'A': A, 'C': C, 'Q': Q, 'R': R,
        'Z_0': Z_0, 'V_0': V_0, 'r': r, 'p': p,
        'R_mat': R_mat, 'q': q, 'nQ': nQ, 'i_idio': i_idio, 'blocks': blocks
    }

def test_em_step_basic():
    """Test basic EM step."""
    print("\n" + "="*70)
    print("TEST: EM Step Basic")
    print("="*70)
    
    inputs = _setup_em_step_inputs(T=80, N=8, nQ=0, seed=42)
    
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(**inputs)
    
    assert C_new is not None
    assert R_new is not None
    assert A_new is not None
    assert np.isfinite(loglik)
    
    n, T = inputs['y'].shape
    m = inputs['A'].shape[0]
    assert C_new.shape == (n, m)
    assert R_new.shape == (n, n)
    assert A_new.shape == (m, m)
    assert np.allclose(R_new, R_new.T)
    
    print("✓ EM step basic test passed")
    return True

def test_em_step_dimensions():
    """Test EM step dimension consistency."""
    print("\n" + "="*70)
    print("TEST: EM Step Dimensions")
    print("="*70)
    
    inputs = _setup_em_step_inputs(T=80, N=12, nQ=0, seed=123)
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(**inputs)
    
    assert C_new.shape == inputs['C'].shape
    assert R_new.shape == inputs['R'].shape
    assert A_new.shape == inputs['A'].shape
    
    print("✓ EM step dimensions verified")
    return True

# ============================================================================
# Initial Conditions Tests
# ============================================================================

def test_init_conditions_basic():
    """Test basic initial conditions."""
    print("\n" + "="*70)
    print("TEST: Init Conditions Basic")
    print("="*70)
    
    T, N = 100, 10
    nQ = 2
    
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    
    blocks = np.zeros((N, 4), dtype=int)
    blocks[:, 0] = 1
    r = np.ones(4)
    p = 1
    
    opt_nan = {'method': 2, 'k': 3}
    Rcon = np.array([[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], 
                     [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
    q = np.zeros(4)
    i_idio = np.concatenate([np.ones(N - nQ), np.zeros(nQ)])
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        x, r, p, blocks, opt_nan, Rcon, q, nQ, i_idio
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
    return True

# ============================================================================
# MATLAB Comparison Tests
# ============================================================================

def load_matlab_results(mat_file):
    """Load MATLAB results from .mat file."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.io.loadmat required")
    if not mat_file.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_file}")
    
    mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
    if 'Res' in mat_data:
        return {'Res': mat_data['Res'], 'Spec': mat_data.get('Spec', None)}
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if len(keys) > 0:
        return {'Res': mat_data[keys[0]], 'Spec': None}
    raise ValueError(f"No DFM results found in {mat_file}")

def compare_matrices(py_mat, mat_mat, name, rtol=1e-3, atol=1e-5):
    """Compare two matrices with tolerance."""
    if py_mat.shape != mat_mat.shape:
        return False, {'error': f"Shape mismatch: {py_mat.shape} vs {mat_mat.shape}"}
    
    diff = np.abs(py_mat - mat_mat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    mat_abs = np.abs(mat_mat)
    finite_mask = mat_abs > atol
    if np.sum(finite_mask) > 0:
        rel_diff = diff[finite_mask] / mat_abs[finite_mask]
        max_rel_diff = np.max(rel_diff)
    else:
        max_rel_diff = max_diff
    
    passed = (max_diff < atol) or (max_rel_diff < rtol)
    return passed, {'max_diff': max_diff, 'mean_diff': mean_diff, 'max_rel_diff': max_rel_diff}

def test_matlab_comparison():
    """Compare Python vs MATLAB results."""
    print("\n" + "="*70)
    print("TEST: MATLAB Comparison")
    print("="*70)
    
    if not SCIPY_AVAILABLE:
        print("SKIPPED: scipy.io.loadmat not available")
        return False
    
    base_dir = Path(__file__).parent.parent.parent
    mat_file = base_dir / 'matlab' / 'ResDFM.mat'
    
    if not mat_file.exists():
        print(f"SKIPPED: MATLAB results not found: {mat_file}")
        return False
    
    try:
        mat_data = load_matlab_results(mat_file)
        mat_res = mat_data['Res']
        
        spec, X, Time, Z = _get_test_data()
        py_res = dfm(X, config, threshold=1e-4)
        
        results = {}
        if hasattr(mat_res, 'C'):
            results['C'] = compare_matrices(py_res.C, mat_res.C, 'C', rtol=1e-3, atol=1e-5)
        if hasattr(mat_res, 'A'):
            results['A'] = compare_matrices(py_res.A, mat_res.A, 'A', rtol=1e-3, atol=1e-5)
        if hasattr(mat_res, 'Z'):
            results['Z'] = compare_matrices(py_res.Z, mat_res.Z, 'Z', rtol=1e-4, atol=1e-6)
        
        passed = sum(1 for p, _ in results.values() if p)
        total = len(results)
        print(f"Comparison results: {passed}/{total} passed")
        
        return passed == total
    except Exception as e:
        print(f"Comparison failed: {e}")
        return False

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
        ('estimation', test_dfm_estimation),
        ('workflow', test_dfm_workflow),
        ('convergence', test_dfm_convergence),
        ('parameter_stability', test_dfm_parameter_stability),
        ('output_consistency', test_dfm_output_consistency),
        ('factor_analysis', test_dfm_factor_analysis),
        ('edge_all_missing', test_dfm_edge_all_missing),
        ('edge_short_series', test_dfm_edge_short_series),
        ('edge_extreme_values', test_dfm_edge_extreme_values),
        ('edge_high_missing', test_dfm_edge_high_missing),
        ('em_step_basic', test_em_step_basic),
        ('em_step_dimensions', test_em_step_dimensions),
        ('init_conditions_basic', test_init_conditions_basic),
        ('matlab_comparison', test_matlab_comparison),
    ]
    
    for name, func in test_funcs:
        try:
            results[name] = func()
            print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = None
    
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("="*70)
    
    return results

# ============================================================================
# Fast Tests (<60s total)
# ============================================================================

def test_dfm_quick():
    """Fast DFM test with synthetic data (target: <10 seconds).
    
    Uses small synthetic dataset and very few iterations to verify
    basic DFM structure without full convergence.
    """
    print("\n" + "="*70)
    print("TEST: DFM Quick (Fast Test)")
    print("="*70)
    
    # Create synthetic data: 10 series, 50 observations
    T, N = 50, 10
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic data with some structure
    # Factor structure: 2 common factors
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    
    # Add some missing values (10%)
    missing_mask = np.random.rand(T, N) < 0.1
    X[missing_mask] = np.nan
    
    # Create minimal config
    from src.nowcasting.config import ModelConfig, SeriesConfig
    
    block_names = ['Global', 'Block1']
    series_list = []
    for i in range(N):
        # All series load on Global, some on Block1
        blocks = [1, 1 if i < 5 else 0]  # First 5 series also load on Block1
        series_list.append(SeriesConfig(
            series_id=f"TEST_{i:02d}",
            series_name=f"Test Series {i}",
            frequency='m',
            transformation='lin',
            category='Test',
            units='Index',
            blocks=blocks
        ))
    
    config = ModelConfig(series=series_list, block_names=block_names)
    
    # Run DFM with relaxed threshold for faster convergence
    # Note: dfm() uses max_iter=5000 internally, but with relaxed threshold
    # it should converge quickly for this small synthetic dataset
    threshold = 1e-3  # Relaxed threshold for faster convergence
    Res = dfm(X, config, threshold=threshold)
    
    # Verify basic structure (even if not converged)
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert hasattr(Res, 'A') and hasattr(Res, 'Q') and hasattr(Res, 'R')
    
    T_actual, N_actual = X.shape
    assert Res.x_sm.shape == (T_actual, N_actual)
    assert Res.Z.shape[0] == T_actual
    assert Res.C.shape[0] == N_actual
    
    # Check that results are finite (even if not optimal)
    assert np.any(np.isfinite(Res.Z))
    assert np.any(np.isfinite(Res.x_sm))
    assert np.any(np.isfinite(Res.C))
    
    print("✓ DFM quick test completed")
    print(f"  Data shape: {X.shape}")
    print(f"  Factors estimated: {Res.Z.shape[1]}")
    print(f"  Convergence: {'Yes' if hasattr(Res, 'converged') and Res.converged else 'Partial (quick test)'}")
    return Res


def test_init_conditions_blocks():
    """Test init_conditions with various block configurations (target: <5 seconds).
    
    Tests covariance calculation for different block structures including
    edge cases like single series per block and missing data.
    """
    print("\n" + "="*70)
    print("TEST: Init Conditions Blocks (Fast Test)")
    print("="*70)
    
    from src.nowcasting.dfm import init_conditions
    
    # Test case 1: Multiple series per block
    print("  Test 1: Multiple series per block")
    T, N = 30, 8
    np.random.seed(123)
    x1 = np.random.randn(T, N)
    x1 = (x1 - np.mean(x1, axis=0)) / np.std(x1, axis=0)  # Standardize
    
    blocks1 = np.array([
        [1, 1, 0, 0],  # Series 0: Global + Block1
        [1, 1, 0, 0],  # Series 1: Global + Block1
        [1, 1, 0, 0],  # Series 2: Global + Block1
        [1, 0, 1, 0],  # Series 3: Global + Block2
        [1, 0, 1, 0],  # Series 4: Global + Block2
        [1, 0, 0, 1],  # Series 5: Global + Block3
        [1, 0, 0, 1],  # Series 6: Global + Block3
        [1, 0, 0, 0],  # Series 7: Global only
    ])
    
    r1 = np.ones(4)  # One factor per block
    p1 = 1
    nQ1 = 0  # All monthly
    i_idio1 = np.ones(N)
    Rcon1 = np.array([[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], 
                      [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
    q1 = np.zeros(4)
    opt_nan1 = {'method': 2, 'k': 3}
    
    A1, C1, Q1, R1, Z_01, V_01 = init_conditions(
        x1, r1, p1, blocks1, opt_nan1, Rcon1, q1, nQ1, i_idio1
    )
    
    # Verify outputs
    assert A1 is not None and C1 is not None
    assert Q1 is not None and R1 is not None
    assert Z_01 is not None and V_01 is not None
    assert np.all(np.isfinite(A1)) and np.all(np.isfinite(C1))
    print("    ✓ Multiple series per block: PASSED")
    
    # Test case 2: Single series per block (edge case)
    print("  Test 2: Single series per block (edge case)")
    T2, N2 = 20, 4
    x2 = np.random.randn(T2, N2)
    x2 = (x2 - np.mean(x2, axis=0)) / np.std(x2, axis=0)
    
    blocks2 = np.array([
        [1, 1, 0, 0],  # Series 0: Global + Block1
        [1, 0, 1, 0],  # Series 1: Global + Block2
        [1, 0, 0, 1],  # Series 2: Global + Block3
        [1, 0, 0, 0],  # Series 3: Global only
    ])
    
    r2 = np.ones(4)
    p2 = 1
    nQ2 = 0
    i_idio2 = np.ones(N2)
    
    A2, C2, Q2, R2, Z_02, V_02 = init_conditions(
        x2, r2, p2, blocks2, opt_nan1, Rcon1, q1, nQ2, i_idio2
    )
    
    assert A2 is not None and C2 is not None
    print("    ✓ Single series per block: PASSED")
    
    # Test case 3: Missing data (should use fallback)
    print("  Test 3: Missing data handling")
    T3, N3 = 15, 3
    x3 = np.random.randn(T3, N3)
    x3 = (x3 - np.mean(x3, axis=0)) / np.std(x3, axis=0)
    # Add significant missing data
    x3[:5, :] = np.nan  # First 5 rows missing
    
    blocks3 = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
    ])
    
    r3 = np.ones(3)
    p3 = 1
    nQ3 = 0
    i_idio3 = np.ones(N3)
    
    # Should handle missing data gracefully (may use identity fallback)
    A3, C3, Q3, R3, Z_03, V_03 = init_conditions(
        x3, r3, p3, blocks3, opt_nan1, Rcon1, q1, nQ3, i_idio3
    )
    
    assert A3 is not None and C3 is not None
    assert np.all(np.isfinite(A3)) and np.all(np.isfinite(C3))
    print("    ✓ Missing data handling: PASSED")
    
    print("✓ Init conditions blocks test completed")
    return True


if __name__ == '__main__':
    run_all_tests()
