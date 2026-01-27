"""Tests for DFM initialization functions.

This module tests the initialization logic, especially:
- Sparse monthly data handling in mixed-frequency setups
- Constrained OLS fallback to unconstrained OLS
- Imputation for initialization
- Edge cases and numerical stability
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

# Import initialization functions
from dfm_python.models.dfm.initialization import (
    impute_for_init,
    initialize_block_loadings,
    initialize_parameters,
    initialize_clock_freq_idio,
    initialize_observation_noise,
)
from dfm_python.config.constants import (
    DEFAULT_DTYPE,
    DEFAULT_REGULARIZATION,
    MIN_EIGENVALUE,
)
from dfm_python.models.dfm.tent import generate_R_mat


class TestImputeForInit:
    """Test imputation function for initialization."""
    
    def test_impute_no_missing(self):
        """Test imputation with no missing values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = impute_for_init(data)
        np.testing.assert_array_equal(result, data)
    
    def test_impute_forward_fill(self):
        """Test forward fill imputation."""
        data = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = impute_for_init(data)
        assert not np.any(np.isnan(result))
        assert result[0] == 1.0
        assert result[1] == 1.0  # Forward filled
        assert result[2] == 1.0  # Forward filled
        assert result[3] == 4.0
        assert result[4] == 4.0  # Forward filled
    
    def test_impute_backward_fill(self):
        """Test backward fill for leading NaNs."""
        data = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = impute_for_init(data)
        assert not np.any(np.isnan(result))
        assert result[0] == 3.0  # Backward filled
        assert result[1] == 3.0  # Backward filled
    
    def test_impute_all_nan(self):
        """Test imputation when all values are NaN."""
        data = np.array([np.nan, np.nan, np.nan])
        result = impute_for_init(data)
        assert not np.any(np.isnan(result))
        assert np.all(result == 0.0)  # Should fallback to 0.0
    
    def test_impute_mean_fallback(self):
        """Test mean fallback for remaining NaNs."""
        data = np.array([1.0, 2.0, np.nan, np.nan])
        result = impute_for_init(data)
        assert not np.any(np.isnan(result))
        # Should use mean (1.5) for remaining NaNs
        assert np.allclose(result[2:], 1.5)


class TestInitializeBlockLoadings:
    """Test block loadings initialization, especially sparse monthly data."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        T = 200
        n_clock = 3
        n_slower = 2
        
        # Clock frequency data (all present)
        clock_data = np.random.randn(T, n_clock)
        
        # Slower frequency data (sparse - only ~15% observations)
        slower_data = np.full((T, n_slower), np.nan)
        monthly_indices = np.arange(0, T, step=4)[:int(T * 0.15)]
        slower_data[monthly_indices, :] = np.random.randn(len(monthly_indices), n_slower)
        
        data_for_extraction = np.hstack([clock_data, slower_data])
        data_with_nans = data_for_extraction.copy()
        
        return data_for_extraction, data_with_nans, n_clock, n_slower, T
    
    @pytest.fixture
    def tent_kernel_setup(self):
        """Setup tent kernel constraints."""
        tent_weights = np.array([1.0, 2.0, 1.0])  # Size 3
        R_mat, q = generate_R_mat(tent_weights)
        return R_mat, q, len(tent_weights)
    
    def test_initialize_block_loadings_sparse_monthly(self, sample_data, tent_kernel_setup):
        """Test that sparse monthly data initializes correctly with fallback."""
        data_for_extraction, data_with_nans, n_clock, n_slower, T = sample_data
        R_mat, q, tent_kernel_size = tent_kernel_setup
        
        clock_freq_indices = np.arange(n_clock)
        slower_freq_indices = np.arange(n_clock, n_clock + n_slower)
        num_factors = 2
        N = n_clock + n_slower
        max_lag_size = max(2, tent_kernel_size)  # p+1 or tent_kernel_size
        
        # Call initialization
        C_i, factors = initialize_block_loadings(
            data_for_extraction=data_for_extraction,
            data_with_nans=data_with_nans,
            clock_freq_indices=clock_freq_indices,
            slower_freq_indices=slower_freq_indices,
            num_factors=num_factors,
            tent_kernel_size=tent_kernel_size,
            R_mat=R_mat,
            q=q,
            N=N,
            max_lag_size=max_lag_size,
            matrix_regularization=DEFAULT_REGULARIZATION,
            dtype=DEFAULT_DTYPE,
            impute_func=impute_for_init
        )
        
        # Verify output shapes
        assert C_i.shape == (N, num_factors * max_lag_size)
        assert factors.shape == (T, num_factors)
        
        # Verify clock frequency series have loadings (PCA-based)
        for idx in clock_freq_indices:
            clock_loadings = C_i[idx, :num_factors]
            assert np.any(np.abs(clock_loadings) > 1e-10), \
                f"Clock frequency series {idx} should have non-zero loadings"
        
        # Verify slower frequency series have loadings (even if sparse)
        # With fallback, they should have non-zero loadings (not all skipped)
        for idx in slower_freq_indices:
            slower_loadings = C_i[idx, :num_factors * tent_kernel_size]
            # Should have at least some non-zero loadings (fallback ensures this)
            assert np.any(np.abs(slower_loadings) > 1e-10), \
                f"Slower frequency series {idx} should have non-zero loadings (fallback should work)"
        
        # Verify no NaN/Inf in loadings
        assert np.all(np.isfinite(C_i)), "C_i should not contain NaN/Inf"
        assert np.all(np.isfinite(factors)), "Factors should not contain NaN/Inf"
    
    def test_initialize_block_loadings_insufficient_data(self, sample_data, tent_kernel_setup):
        """Test initialization with extremely sparse data (insufficient for regression)."""
        data_for_extraction, data_with_nans, n_clock, n_slower, T = sample_data
        R_mat, q, tent_kernel_size = tent_kernel_setup
        
        # Make slower frequency data even sparser (only 2 observations)
        data_with_nans[:, n_clock:] = np.nan
        data_with_nans[10, n_clock] = 1.0
        data_with_nans[50, n_clock] = 2.0
        
        clock_freq_indices = np.arange(n_clock)
        slower_freq_indices = np.arange(n_clock, n_clock + n_slower)
        num_factors = 2
        N = n_clock + n_slower
        max_lag_size = max(2, tent_kernel_size)
        
        # Should still complete (may skip series with insufficient data)
        C_i, factors = initialize_block_loadings(
            data_for_extraction=data_for_extraction,
            data_with_nans=data_with_nans,
            clock_freq_indices=clock_freq_indices,
            slower_freq_indices=slower_freq_indices,
            num_factors=num_factors,
            tent_kernel_size=tent_kernel_size,
            R_mat=R_mat,
            q=q,
            N=N,
            max_lag_size=max_lag_size,
            matrix_regularization=DEFAULT_REGULARIZATION,
            dtype=DEFAULT_DTYPE,
            impute_func=impute_for_init
        )
        
        # Should still produce valid output
        assert C_i.shape == (N, num_factors * max_lag_size)
        assert factors.shape == (T, num_factors)
        assert np.all(np.isfinite(C_i))
        assert np.all(np.isfinite(factors))


class TestInitializeParameters:
    """Test main initialization function."""
    
    @pytest.fixture
    def mixed_freq_data(self):
        """Create mixed-frequency data for testing."""
        np.random.seed(42)
        T = 200
        n_clock = 3
        n_slower = 2
        
        # Clock frequency (weekly) - all present
        clock_data = np.random.randn(T, n_clock)
        
        # Slower frequency (monthly) - sparse (~15% present)
        slower_data = np.full((T, n_slower), np.nan)
        monthly_indices = np.arange(0, T, step=4)[:int(T * 0.15)]
        slower_data[monthly_indices, :] = np.random.randn(len(monthly_indices), n_slower)
        
        x = np.hstack([clock_data, slower_data])
        
        # Blocks: all series in one block
        blocks = np.ones((n_clock + n_slower, 1), dtype=int)
        r = np.array([2.0])  # 2 factors
        
        # Tent kernel setup
        tent_weights = np.array([1.0, 2.0, 1.0])
        R_mat, q = generate_R_mat(tent_weights)
        
        return x, blocks, r, R_mat, q, n_slower, tent_weights
    
    def test_initialize_parameters_mixed_freq(self, mixed_freq_data):
        """Test parameter initialization with mixed-frequency sparse data."""
        x, blocks, r, R_mat, q, n_slower, tent_weights = mixed_freq_data
        
        A, C, Q, R, Z_0, V_0 = initialize_parameters(
            x=x,
            r=r,
            p=1,  # VAR(1)
            blocks=blocks,
            R_mat=R_mat,
            q=q,
            n_slower_freq=n_slower,
            idio_indicator=None,
            clock='w',
            tent_weights_dict={'m:w': tent_weights}
        )
        
        # Verify shapes
        T, N = x.shape
        n_factors = int(r[0])
        tent_kernel_size = len(tent_weights)
        max_lag_size = max(2, tent_kernel_size)  # p+1 or tent_kernel_size
        
        # State dimension: factors + clock idio + slower idio
        # Factors: n_factors * max_lag_size
        # Clock idio: n_clock (one per series)
        # Slower idio: n_slower * tent_kernel_size
        n_clock = N - n_slower
        expected_state_dim = n_factors * max_lag_size + n_clock + n_slower * tent_kernel_size
        
        assert A.shape == (expected_state_dim, expected_state_dim), \
            f"A shape {A.shape} != expected ({expected_state_dim}, {expected_state_dim})"
        assert C.shape == (N, expected_state_dim), \
            f"C shape {C.shape} != expected ({N}, {expected_state_dim})"
        assert Q.shape == A.shape
        assert R.shape == (N, N)
        assert Z_0.shape == (expected_state_dim,)
        assert V_0.shape == A.shape
        
        # Verify no NaN/Inf
        assert np.all(np.isfinite(A)), "A should not contain NaN/Inf"
        assert np.all(np.isfinite(C)), "C should not contain NaN/Inf"
        assert np.all(np.isfinite(Q)), "Q should not contain NaN/Inf"
        assert np.all(np.isfinite(R)), "R should not contain NaN/Inf"
        assert np.all(np.isfinite(Z_0)), "Z_0 should not contain NaN/Inf"
        assert np.all(np.isfinite(V_0)), "V_0 should not contain NaN/Inf"
        
        # Verify slower frequency series have non-zero loadings (fallback should work)
        slower_indices = np.arange(N - n_slower, N)
        for idx in slower_indices:
            slower_loadings = C[idx, :n_factors * tent_kernel_size]
            # With fallback, should have at least some non-zero loadings
            assert np.any(np.abs(slower_loadings) > 1e-10), \
                f"Slower frequency series {idx} should have non-zero loadings"
    
    def test_initialize_parameters_clock_freq_only(self):
        """Test initialization with clock frequency only (no mixed frequency)."""
        np.random.seed(42)
        T = 100
        N = 5
        
        x = np.random.randn(T, N)
        blocks = np.ones((N, 1), dtype=int)
        r = np.array([2.0])
        
        A, C, Q, R, Z_0, V_0 = initialize_parameters(
            x=x,
            r=r,
            p=1,
            blocks=blocks,
            R_mat=None,
            q=None,
            n_slower_freq=0,
            idio_indicator=None,
            clock='w',
            tent_weights_dict=None
        )
        
        # Verify shapes (no tent kernel, simpler structure)
        n_factors = int(r[0])
        expected_state_dim = n_factors * 2 + N  # factors * (p+1) + idio
        
        assert A.shape == (expected_state_dim, expected_state_dim)
        assert C.shape == (N, expected_state_dim)
        assert Q.shape == A.shape
        assert R.shape == (N, N)
        
        # Verify no NaN/Inf
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(Q))
        assert np.all(np.isfinite(R))


class TestInitializeClockFreqIdio:
    """Test clock frequency idiosyncratic initialization."""
    
    def test_initialize_clock_freq_idio_basic(self):
        """Test basic clock frequency idiosyncratic initialization."""
        np.random.seed(42)
        T = 100
        n_clock = 3
        
        res = np.random.randn(T, n_clock)
        data_with_nans = res.copy()
        
        BM, SM, initViM = initialize_clock_freq_idio(
            res=res,
            data_with_nans=data_with_nans,
            n_clock_freq=n_clock,
            idio_indicator=None,
            T=T,
            dtype=DEFAULT_DTYPE
        )
        
        # Verify shapes
        assert BM.shape == (n_clock, n_clock)
        assert SM.shape == (n_clock, n_clock)
        assert initViM.shape == (n_clock, n_clock)
        
        # Verify no NaN/Inf
        assert np.all(np.isfinite(BM))
        assert np.all(np.isfinite(SM))
        assert np.all(np.isfinite(initViM))
        
        # BM should be diagonal (AR(1) per series)
        assert np.allclose(BM, np.diag(np.diag(BM)))


class TestInitializeObservationNoise:
    """Test observation noise initialization."""
    
    def test_initialize_observation_noise_basic(self):
        """Test basic observation noise initialization."""
        np.random.seed(42)
        T = 100
        N = 5
        
        data_with_nans = np.random.randn(T, N)
        
        R = initialize_observation_noise(
            data_with_nans=data_with_nans,
            N=N,
            idio_indicator=None,
            n_clock_freq=N,
            dtype=DEFAULT_DTYPE
        )
        
        # Verify shape
        assert R.shape == (N, N)
        
        # Should be diagonal
        assert np.allclose(R, np.diag(np.diag(R)))
        
        # Verify no NaN/Inf
        assert np.all(np.isfinite(R))
        
        # Diagonal should be positive
        assert np.all(np.diag(R) > 0)
    
    def test_initialize_observation_noise_with_missing(self):
        """Test observation noise initialization with missing values."""
        np.random.seed(42)
        T = 100
        N = 5
        
        data_with_nans = np.random.randn(T, N)
        # Add some missing values
        data_with_nans[10:20, 0] = np.nan
        data_with_nans[30:40, 2] = np.nan
        
        R = initialize_observation_noise(
            data_with_nans=data_with_nans,
            N=N,
            idio_indicator=None,
            n_clock_freq=N,
            dtype=DEFAULT_DTYPE
        )
        
        # Should still produce valid output
        assert R.shape == (N, N)
        assert np.all(np.isfinite(R))
        assert np.all(np.diag(R) > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
