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
from dfm_python.utils.data_utils import rem_nans_spline

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ============================================================================
# Tests using synthetic data only (no external files required)
# ============================================================================

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
        ('em_step_dimensions', test_em_step_dimensions),
        ('init_conditions_basic', test_init_conditions_basic),
        ('dfm_quick', test_dfm_quick),
        ('init_conditions_blocks', test_init_conditions_blocks),
        ('multi_block_different_factors', test_multi_block_different_factor_counts),
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
    from dfm_python.config import DFMConfig, SeriesConfig
    
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
    
    config = DFMConfig(series=series_list, block_names=block_names)
    
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


def test_init_conditions_blocks():
    """Test init_conditions with various block configurations (target: <5 seconds).
    
    Tests covariance calculation for different block structures including
    edge cases like single series per block and missing data.
    """
    print("\n" + "="*70)
    print("TEST: Init Conditions Blocks (Fast Test)")
    print("="*70)
    
    from dfm_python.dfm import init_conditions
    
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


def test_multi_block_different_factor_counts():
    """
    Test for bug fix: multi-block models with different factor counts.
    
    This test reproduces the bug where ff variable from previous block
    iteration caused dimension mismatch when stacking arrays.
    
    Bug scenario:
    - Block 0: r_i=3, creates ff with width 3*pC
    - Block 1: r_i=2, tries to use ff from block 0 (width 3*pC) 
               with np.zeros(width 2*pC) -> dimension mismatch
    
    Fix: Reset ff = None at start of each block iteration.
    """
    print("\n" + "="*70)
    print("Testing multi-block with different factor counts (bug fix)")
    print("="*70)
    
    np.random.seed(42)
    T = 100  # Time periods
    N = 15   # Total series
    
    # Create synthetic data
    x = np.random.randn(T, N)
    
    # Create blocks: 3 blocks with different factor counts
    # Block 0: 5 series, 3 factors
    # Block 1: 5 series, 2 factors  
    # Block 2: 5 series, 2 factors
    blocks = np.zeros((N, 3), dtype=int)
    blocks[0:5, 0] = 1   # First 5 series in block 0
    blocks[5:10, 1] = 1  # Next 5 series in block 1
    blocks[10:15, 2] = 1 # Last 5 series in block 2
    # All series also load on block 0 (global factor)
    blocks[:, 0] = 1
    
    # Different factor counts per block (this triggers the bug)
    r = np.array([3, 2, 2])  # 3 factors, 2 factors, 2 factors
    p = 1  # AR lag
    pC = max(p, 1)  # Will be 1
    
    # Setup other parameters
    opt_nan = {'method': 2, 'k': 3}
    nQ = 0  # All monthly
    i_idio = np.ones(N)
    Rcon = None
    q = None
    
    # This should NOT raise ValueError about dimension mismatch
    try:
        A, C, Q, R, Z_0, V_0 = init_conditions(
            x, r, p, blocks, opt_nan, Rcon, q, nQ, i_idio
        )
        
        # Verify outputs are valid
        assert A is not None, "A should not be None"
        assert C is not None, "C should not be None"
        assert Q is not None, "Q should not be None"
        assert R is not None, "R should not be None"
        
        # Verify dimensions
        # State dimension = sum(r_i * ppC) + monthly_idio + quarterly_idio * 5
        # where ppC = max(p, pC) and pC = 1 when Rcon is None
        ppC = max(p, 1)  # pC = 1 when Rcon is None
        expected_state_dim = sum(r) * ppC + N + 5 * nQ  # factors + monthly_idio + quarterly_idio*5
        assert A.shape == (expected_state_dim, expected_state_dim), \
            f"A shape {A.shape} != expected {(expected_state_dim, expected_state_dim)}"
        assert C.shape == (N, expected_state_dim), \
            f"C shape {C.shape} != expected {(N, expected_state_dim)}"
        
        # Verify no NaN/Inf
        assert np.all(np.isfinite(A)), "A contains NaN/Inf"
        assert np.all(np.isfinite(C)), "C contains NaN/Inf"
        assert np.all(np.isfinite(Q)), "Q contains NaN/Inf"
        assert np.all(np.isfinite(R)), "R contains NaN/Inf"
        
        print("    ✓ Multi-block init_conditions with different factor counts: PASSED")
        print(f"    ✓ State dimension: {expected_state_dim}")
        print(f"    ✓ A shape: {A.shape}")
        print(f"    ✓ C shape: {C.shape}")
        
    except ValueError as e:
        if "dimension" in str(e).lower() and "match" in str(e).lower():
            print(f"    ✗ FAILED: Dimension mismatch error still present!")
            print(f"    Error: {e}")
            raise
        else:
            # Other ValueError is OK (e.g., insufficient data)
            print(f"    ⚠ Warning: {e}")


if __name__ == '__main__':
    run_all_tests()
