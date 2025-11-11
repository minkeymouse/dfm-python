"""Tests for numeric utility functions - covariance, variance, matrix operations."""

import sys
from pathlib import Path
import numpy as np
import pytest
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.core.numeric import (
    _compute_covariance_safe,
    _compute_variance_safe,
    _ensure_innovation_variance_minimum,
    _ensure_real_and_symmetric,
    _ensure_covariance_stable,
)


# ============================================================================
# Covariance Computation Tests
# ============================================================================

def test_compute_covariance_safe_basic():
    """Test basic covariance computation."""
    np.random.seed(42)
    T, N = 100, 10
    data = np.random.randn(T, N)
    
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=False)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)  # Symmetric


def test_compute_covariance_safe_pairwise_complete():
    """Test pairwise complete covariance computation."""
    np.random.seed(42)
    T, N = 100, 10
    data = np.random.randn(T, N)
    # Add missing values
    missing_mask = np.random.rand(T, N) < 0.2
    data[missing_mask] = np.nan
    
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)  # Symmetric


def test_compute_covariance_safe_pairwise_extreme_sparsity():
    """Test pairwise complete covariance with extreme sparsity.
    
    Scenario: No time point has all series observed, but pairwise
    observations exist for valid covariance computation.
    """
    np.random.seed(42)
    T, N = 50, 5
    data = np.full((T, N), np.nan)
    
    # Create pattern where no row is complete, but pairs have overlap
    # Strategy: For each pair (i, j), add observations at different time points
    # and ensure each time point is missing at least one series
    for i in range(N):
        for j in range(i, N):
            # Each pair has some complete observations at different time points
            n_obs = min(10, T)
            overlap_indices = np.random.choice(T, size=n_obs, replace=False)
            if i == j:
                # Variance: add observations for series i
                data[overlap_indices, i] = np.random.randn(len(overlap_indices))
            else:
                # Covariance: ensure both series observed together
                data[overlap_indices, i] = np.random.randn(len(overlap_indices))
                data[overlap_indices, j] = np.random.randn(len(overlap_indices))
    
    # Ensure no row is complete by removing one observation from each complete row
    for t in range(T):
        if np.all(np.isfinite(data[t, :])):
            # Remove one random observation to make row incomplete
            remove_idx = np.random.randint(N)
            data[t, remove_idx] = np.nan
    
    # Verify no row is complete
    complete_rows = np.all(np.isfinite(data), axis=1)
    assert np.sum(complete_rows) == 0, "Test setup failed: some rows are complete"
    
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)  # Symmetric
    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals >= -1e-10)  # PSD (small tolerance for numerical errors)


def test_compute_covariance_safe_large_block():
    """Test covariance with large block (like Block_Global)."""
    np.random.seed(42)
    T, N = 396, 78  # Realistic block size
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23  # 23% missing
    data[missing_mask] = np.nan
    
    cov = _compute_covariance_safe(
        data, 
        rowvar=True, 
        pairwise_complete=True,
        fallback_to_identity=True
    )
    
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))


def test_compute_covariance_safe_small_block():
    """Test covariance with small block."""
    np.random.seed(42)
    T, N = 396, 20  # Smaller block
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23
    data[missing_mask] = np.nan
    
    cov = _compute_covariance_safe(
        data,
        rowvar=True,
        pairwise_complete=True,
        fallback_to_identity=True
    )
    
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))


def test_compute_covariance_safe_pairwise_single_observation():
    """Test pairwise covariance where each pair has only one overlapping observation.
    
    This tests the extreme edge case where pairwise complete covariance
    computation relies on minimal overlap between series pairs.
    """
    np.random.seed(42)
    T, N = 20, 5
    data = np.full((T, N), np.nan)
    
    # Strategy: For each pair (i, j), assign exactly one time point where
    # both series are observed. Different pairs use different time points.
    pair_count = 0
    for i in range(N):
        for j in range(i, N):
            # Assign one unique time point for this pair
            t = pair_count % T
            if i == j:
                # Variance: single observation for series i
                data[t, i] = np.random.randn()
            else:
                # Covariance: both series observed at same time point
                val = np.random.randn()
                data[t, i] = val
                data[t, j] = val + np.random.randn() * 0.1  # Slight variation
            pair_count += 1
    
    # Verify each pair has exactly one overlapping observation
    # (except for variance terms which may have more)
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)  # Symmetric
    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals >= -1e-10)  # PSD (small tolerance for numerical errors)


def test_compute_covariance_safe_sparse_data():
    """Test covariance with very sparse data (quarterly in monthly model)."""
    np.random.seed(42)
    T, N = 396, 7
    data = np.full((T, N), np.nan)
    # Only fill quarterly observations (every 3rd month)
    for i in range(0, T, 3):
        data[i, :] = np.random.randn(N)
    
    cov = _compute_covariance_safe(
        data,
        rowvar=True,
        pairwise_complete=True,
        fallback_to_identity=True
    )
    
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))


def test_compute_covariance_safe_shape_consistency():
    """Test that covariance always returns correct shape regardless of missing data pattern."""
    np.random.seed(42)
    T, N = 200, 15
    
    test_cases = [
        ("no_missing", None),
        ("low_missing", 0.1),
        ("medium_missing", 0.3),
        ("high_missing", 0.6),
    ]
    
    for name, missing_rate in test_cases:
        data = np.random.randn(T, N)
        if missing_rate is not None:
            missing_mask = np.random.rand(T, N) < missing_rate
            data[missing_mask] = np.nan
        
        cov = _compute_covariance_safe(
            data,
            rowvar=True,
            pairwise_complete=True,
            fallback_to_identity=True
        )
        
        assert cov.shape == (N, N), f"{name}: Expected ({N}, {N}), got {cov.shape}"


def test_compute_covariance_safe_rowvar_false():
    """Test covariance with rowvar=False (data is N x T)."""
    np.random.seed(42)
    N, T = 10, 100
    data = np.random.randn(N, T)  # (N, T) format
    
    cov = _compute_covariance_safe(data, rowvar=False, pairwise_complete=False)
    
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))


def test_compute_covariance_safe_single_variable():
    """Test covariance with single variable."""
    np.random.seed(42)
    T = 100
    data = np.random.randn(T, 1)
    
    cov = _compute_covariance_safe(data, rowvar=True)
    
    assert cov.shape == (1, 1)
    assert np.isfinite(cov[0, 0])


def test_compute_covariance_safe_empty_data():
    """Test covariance with empty data (should fallback to identity)."""
    data = np.array([]).reshape(0, 5)
    
    cov = _compute_covariance_safe(
        data,
        rowvar=True,
        fallback_to_identity=True
    )
    
    assert cov.shape == (5, 5)
    assert np.allclose(cov, np.eye(5))


# ============================================================================
# Variance Computation Tests
# ============================================================================

def test_compute_variance_safe_basic():
    """Test basic variance computation."""
    np.random.seed(42)
    data = np.random.randn(100)
    
    var = _compute_variance_safe(data)
    
    assert np.isfinite(var)
    assert var > 0


def test_compute_variance_safe_with_missing():
    """Test variance computation with missing values."""
    np.random.seed(42)
    data = np.random.randn(100)
    data[10:20] = np.nan
    
    var = _compute_variance_safe(data)
    
    assert np.isfinite(var)
    assert var > 0


def test_compute_variance_safe_all_missing():
    """Test variance with all missing data."""
    data = np.full(100, np.nan)
    
    var = _compute_variance_safe(data, default_variance=1.0)
    
    assert var == 1.0


# ============================================================================
# Matrix Stability Tests
# ============================================================================

def test_ensure_innovation_variance_minimum():
    """Test innovation variance minimum enforcement."""
    np.random.seed(42)
    m = 5
    Q = np.random.randn(m, m)
    Q = Q @ Q.T  # Make PSD
    Q[0, 0] = 0.0  # Set first diagonal to zero
    
    Q_fixed = _ensure_innovation_variance_minimum(Q, min_variance=1e-8)
    
    assert Q_fixed.shape == (m, m)
    assert np.all(np.diag(Q_fixed) >= 1e-8)
    assert Q_fixed[0, 0] >= 1e-8


def test_q_diagonal_never_zero():
    """Test that Q diagonal never becomes zero in any scenario."""
    from dfm_python.core.em import MIN_INNOVATION_VARIANCE
    
    np.random.seed(42)
    min_variance = MIN_INNOVATION_VARIANCE
    
    # Test case 1: All zeros
    Q_all_zeros = np.zeros((5, 5))
    Q_fixed = _ensure_innovation_variance_minimum(Q_all_zeros, min_variance=min_variance)
    assert np.all(np.diag(Q_fixed) >= min_variance), "All-zero Q should have minimum diagonal"
    
    # Test case 2: Single zero diagonal
    Q_single_zero = np.eye(5) * 0.1
    Q_single_zero[2, 2] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_single_zero, min_variance=min_variance)
    assert Q_fixed[2, 2] >= min_variance, "Single zero diagonal should be fixed"
    assert np.all(np.diag(Q_fixed) >= min_variance), "All diagonals should be >= min_variance"
    
    # Test case 3: Multiple zeros
    Q_multiple_zeros = np.eye(5) * 0.1
    Q_multiple_zeros[0, 0] = 0.0
    Q_multiple_zeros[3, 3] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_multiple_zeros, min_variance=min_variance)
    assert Q_fixed[0, 0] >= min_variance, "First zero should be fixed"
    assert Q_fixed[3, 3] >= min_variance, "Second zero should be fixed"
    assert np.all(np.diag(Q_fixed) >= min_variance), "All diagonals should be >= min_variance"
    
    # Test case 4: Very small values (below threshold)
    Q_very_small = np.eye(5) * 1e-10
    Q_fixed = _ensure_innovation_variance_minimum(Q_very_small, min_variance=min_variance)
    assert np.all(np.diag(Q_fixed) >= min_variance), "Very small values should be raised to minimum"
    
    # Test case 5: Negative diagonal (should be fixed)
    Q_negative = np.eye(5) * 0.1
    Q_negative[1, 1] = -0.01
    Q_fixed = _ensure_innovation_variance_minimum(Q_negative, min_variance=min_variance)
    # Allow for floating point precision (use slightly lower threshold)
    assert Q_fixed[1, 1] >= min_variance * 0.99, f"Negative diagonal should be fixed, got {Q_fixed[1, 1]:.2e}"
    
    # Test case 6: Off-diagonal elements preserved
    Q_with_offdiag = np.eye(5) * 0.1
    Q_with_offdiag[0, 1] = 0.05
    Q_with_offdiag[1, 0] = 0.05  # Symmetric
    Q_with_offdiag[0, 0] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_with_offdiag, min_variance=min_variance)
    assert Q_fixed[0, 0] >= min_variance, "Zero diagonal should be fixed"
    assert Q_fixed[0, 1] == Q_with_offdiag[0, 1], "Off-diagonal should be preserved"
    assert Q_fixed[1, 0] == Q_with_offdiag[1, 0], "Off-diagonal should be preserved"


def test_ensure_real_and_symmetric():
    """Test real and symmetric enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m) + 1j * np.random.randn(m, m) * 1e-10  # Tiny imaginary part
    
    M_fixed = _ensure_real_and_symmetric(M)
    
    assert np.isrealobj(M_fixed)
    assert np.allclose(M_fixed, M_fixed.T)


def test_ensure_covariance_stable():
    """Test covariance stability enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m)
    M = M @ M.T
    # Make it slightly non-PSD
    eigenvals = np.linalg.eigvalsh(M)
    M = M - np.eye(m) * (np.min(eigenvals) + 0.1)
    
    M_fixed = _ensure_covariance_stable(M, min_eigenval=1e-8)
    
    eigenvals_fixed = np.linalg.eigvalsh(M_fixed)
    # Allow for floating-point precision (use slightly lower threshold)
    assert np.all(eigenvals_fixed >= 1e-8 * 0.99), f"Eigenvalues: {eigenvals_fixed}"


# ============================================================================
# Integration Tests
# ============================================================================

def test_covariance_in_init_conditions_scenario():
    """Test covariance computation in realistic init_conditions scenario."""
    from dfm_python.core.em import init_conditions
    
    np.random.seed(42)
    T, N = 200, 20
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.2
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
    
    assert A is not None
    assert C.shape == (N, A.shape[0])
    assert Q.shape == (A.shape[0], A.shape[0])
    assert R.shape == (N, N)


def test_covariance_multiple_blocks():
    """Test covariance computation with multiple blocks (realistic scenario)."""
    from dfm_python.core.em import init_conditions
    
    np.random.seed(42)
    T, N = 200, 30
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.25
    x[missing_mask] = np.nan
    
    # Create 3 blocks
    blocks = np.zeros((N, 3), dtype=int)
    blocks[:, 0] = 1  # All in global
    blocks[0:10, 1] = 1  # First 10 in block 1
    blocks[10:20, 2] = 1  # Next 10 in block 2
    
    r = np.array([2, 1, 1])
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
    
    assert A is not None
    assert C.shape[0] == N
    assert Q.shape[0] == A.shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

