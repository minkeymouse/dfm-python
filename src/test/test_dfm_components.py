"""Tests for DFM refactored components.

Tests cover:
- Numeric utilities (estimator, validator, analytic, stability)
- Functional utilities (dfm_block, em)
- SSM components (kalman, companion, structural)
"""

import pytest
import numpy as np
import torch
from typing import Optional

from dfm_python.numeric.estimator import (
    estimate_ar,
    estimate_var1,
    estimate_var2,
    estimate_idio_dynamics,
    estimate_idio_params,
    estimate_state_space_params
)
from dfm_python.numeric.validator import (
    validate_ar_order,
    validate_ma_order,
    validate_data_shape,
    validate_no_nan_inf,
    validate_model_components,
    validate_companion_stability,
    validate_forecast_inputs
)
from dfm_python.utils.errors import (
    ConfigurationError,
    DataValidationError,
    NumericalStabilityError
)
from dfm_python.numeric.analytic import (
    safe_divide,
    compute_var_safe,
    compute_cov_safe,
    compute_forecast_metrics,
    convergence_checker
)
from dfm_python.numeric.stability import (
    ensure_symmetric,
    clean_matrix,
    cap_max_eigenval,
    ensure_positive_definite,
    safe_determinant
)
from dfm_python.functional.dfm_block import initialize_block_loadings
from dfm_python.functional.em import em_step, run_em_algorithm
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.ssm.companion import CompanionSSM, MACompanionSSM
from dfm_python.ssm.structural import StructuralIdentificationSSM


class TestNumericEstimator:
    """Test numeric estimation functions."""
    
    def test_estimate_var1(self):
        """Test VAR(1) estimation."""
        T, r = 100, 3
        # Generate VAR(1) data
        A_true = np.random.randn(r, r) * 0.3
        factors = np.zeros((T, r))
        for t in range(1, T):
            factors[t] = A_true @ factors[t-1] + np.random.randn(r) * 0.1
        
        A_est, Q_est = estimate_var1(factors)
        
        assert A_est.shape == (r, r)
        assert Q_est.shape == (r, r)
        # Q should be positive semi-definite
        eigenvals = np.linalg.eigvals(Q_est)
        assert np.all(eigenvals.real >= -1e-8)
    
    def test_estimate_var2(self):
        """Test VAR(2) estimation."""
        T, r = 100, 3
        factors = np.random.randn(T, r)
        
        A_est, Q_est = estimate_var2(factors)
        
        # A_est is (r x 2r), split into A1 and A2
        assert A_est.shape == (r, 2 * r)
        assert Q_est.shape == (r, r)
    
    def test_estimate_ar(self):
        """Test AR estimation from expectations."""
        r = 3
        # estimate_ar requires EZZ_FB and EZZ_BB (expectations from EM algorithm)
        EZZ_FB = np.random.randn(r, r) * 0.1  # Forward-backward expectation
        EZZ_BB = np.eye(r) + np.random.randn(r, r) * 0.1  # Backward-backward expectation (must be PSD)
        EZZ_BB = EZZ_BB @ EZZ_BB.T  # Make it positive definite
        
        A_est, Q_est = estimate_ar(EZZ_FB, EZZ_BB)
        
        # A_est is diagonal AR coefficients
        assert A_est.shape == (r,)
        assert Q_est is None  # Q_diag is not computed (returns None)


class TestNumericValidator:
    """Test numeric validation functions."""
    
    def test_validate_ar_order(self):
        """Test AR order validation."""
        assert validate_ar_order(1) == 1
        assert validate_ar_order(2) == 2
        
        with pytest.raises(ConfigurationError):
            validate_ar_order(0)
        with pytest.raises(ConfigurationError):
            validate_ar_order(-1)
    
    def test_validate_ma_order(self):
        """Test MA order validation."""
        assert validate_ma_order(0) == 0
        assert validate_ma_order(1) == 1
        
        with pytest.raises(ConfigurationError):
            validate_ma_order(-1)
    
    def test_validate_data_shape(self):
        """Test data shape validation."""
        X = np.random.randn(100, 5)
        validate_data_shape(X, min_dims=2, max_dims=2, min_size=1)
        
        with pytest.raises(DataValidationError):
            validate_data_shape(X, min_dims=3)
    
    def test_validate_no_nan_inf(self):
        """Test NaN/Inf validation."""
        X = np.random.randn(100, 5)
        validate_no_nan_inf(X, name="test_data")
        
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(DataValidationError):
            validate_no_nan_inf(X_nan, name="test_data")
    
    def test_validate_companion_stability(self):
        """Test companion matrix stability validation."""
        # Stable matrix (eigenvalues < 1.0)
        A_stable = np.random.randn(3, 3) * 0.3
        is_stable, max_eigenval = validate_companion_stability(
            A_stable, model_name="test", name="test_matrix"
        )
        assert is_stable or max_eigenval < 1.0
        
        # Unstable matrix (eigenvalues >= 1.0)
        A_unstable = np.eye(3) * 1.1
        with pytest.raises(NumericalStabilityError):
            validate_companion_stability(
                A_unstable, model_name="test", name="test_matrix", threshold=1.0
            )


class TestNumericAnalytic:
    """Test numeric analytic functions."""
    
    def test_safe_divide(self):
        """Test safe division."""
        result = safe_divide(10.0, 2.0)
        assert result == 5.0
        
        result = safe_divide(10.0, 0.0)
        assert result == 0.0  # Should return 0 for division by zero
    
    def test_compute_var_safe(self):
        """Test safe variance computation."""
        X = np.random.randn(100, 5)
        var = compute_var_safe(X.flatten())  # Function expects 1D array
        assert isinstance(var, (float, np.floating))
        assert var >= 0
    
    def test_compute_cov_safe(self):
        """Test safe covariance computation."""
        X = np.random.randn(100, 5)
        cov = compute_cov_safe(X)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)  # Should be symmetric
    
    def test_compute_forecast_metrics(self):
        """Test forecast metrics computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = compute_forecast_metrics(y_true, y_pred)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['rmse'] > 0


class TestNumericStability:
    """Test numeric stability functions."""
    
    def test_ensure_symmetric(self):
        """Test matrix symmetry enforcement."""
        A = np.random.randn(3, 3)
        A_sym = ensure_symmetric(A)
        assert np.allclose(A_sym, A_sym.T)
    
    def test_cap_max_eigenval(self):
        """Test eigenvalue capping."""
        A = np.random.randn(3, 3) * 2.0  # May have large eigenvalues
        A_capped = cap_max_eigenval(A, max_eigenval=0.99)
        eigenvals = np.linalg.eigvals(A_capped)
        max_eigenval = np.max(np.abs(eigenvals))
        assert max_eigenval <= 0.99 + 1e-6  # Allow small floating point tolerance
    
    def test_ensure_positive_definite(self):
        """Test positive definiteness enforcement."""
        A = np.random.randn(3, 3)
        A = A @ A.T  # Make symmetric
        A_pd = ensure_positive_definite(A, min_eigenval=1e-6)
        eigenvals = np.linalg.eigvals(A_pd)
        assert np.all(eigenvals.real >= 1e-6)
    
    def test_safe_determinant(self):
        """Test safe determinant computation."""
        A = np.random.randn(3, 3)
        det = safe_determinant(A)
        assert isinstance(det, (float, np.floating))
        assert np.isfinite(det)


class TestFunctionalDFMBlock:
    """Test DFM block functional utilities."""
    
    def test_initialize_block_loadings(self):
        """Test block loadings initialization."""
        T, N = 100, 10
        data = np.random.randn(T, N)
        clock_freq_indices = np.arange(N)
        slower_freq_indices = np.array([], dtype=int)
        tent_kernel_size = 1
        max_lag_size = 1
        
        C, factors = initialize_block_loadings(
            data_for_extraction=data,
            data_with_nans=data,
            clock_freq_indices=clock_freq_indices,
            slower_freq_indices=slower_freq_indices,
            num_factors=3,
            tent_kernel_size=tent_kernel_size,
            R_mat=None,
            q=None,
            N=N,
            max_lag_size=max_lag_size
        )
        
        assert C.shape[0] == N  # Number of series
        assert C.shape[1] == 3 * max_lag_size  # Number of factors * max_lag_size
        assert factors.shape[0] == T  # Time dimension
        assert factors.shape[1] == 3  # Number of factors


class TestSSMComponents:
    """Test SSM components."""
    
    def test_companion_ssm_initialization(self):
        """Test CompanionSSM initialization."""
        ssm = CompanionSSM(
            n_vars=5,
            lag_order=1,
            n_kernels=1,
            kernel_init='normal',
            norm_order=1
        )
        assert ssm.n_vars == 5
        assert ssm.order == 1
    
    def test_companion_ssm_forward(self):
        """Test CompanionSSM forward pass."""
        ssm = CompanionSSM(n_vars=3, lag_order=1)
        x = torch.randn(1, 10, 3)  # (batch, time, vars) - 3D tensor required
        output = ssm(x)
        assert output.shape == (1, 10, 3)
    
    def test_ma_companion_ssm_initialization(self):
        """Test MACompanionSSM initialization."""
        ssm = MACompanionSSM(
            n_vars=5,
            ma_order=1,
            n_kernels=1,
            kernel_init='normal',
            norm_order=1
        )
        assert ssm.n_vars == 5
        assert ssm.order == 1
    
    def test_structural_identification_initialization(self):
        """Test StructuralIdentificationSSM initialization."""
        ssm = StructuralIdentificationSSM(
            n_vars=5,
            lag_order=1,
            method='cholesky'
        )
        assert ssm.n_vars == 5
        assert ssm.lag_order == 1


class TestKDFMRefactored:
    """Test KDFM after refactoring (no kdfm_components)."""
    
    def test_kdfm_direct_ssm_usage(self):
        """Test that KDFM uses CompanionSSM directly (no wrapper classes)."""
        from dfm_python.models import KDFM
        import torch
        
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(50, 5)
        model.initialize_from_data(X)
        
        # Should have direct CompanionSSM references (not ARStage/MAStage)
        assert model.companion_ar is not None
        assert isinstance(model.companion_ar, CompanionSSM)
        assert model.companion_ma is None  # ma_order=0
        
        # Should not have _ar_stage or _ma_stage attributes
        assert not hasattr(model, '_ar_stage')
        assert not hasattr(model, '_ma_stage')
        assert not hasattr(model, '_training_step_handler')
    
    def test_kdfm_forward_pass_direct(self):
        """Test KDFM forward pass uses SSMs directly."""
        from dfm_python.models import KDFM
        import torch
        
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(50, 5)
        model.initialize_from_data(X)
        
        # Forward pass should work
        output = model.forward(X)
        assert output.shape == X.shape
    
    def test_kdfm_training_step_inline(self):
        """Test that KDFM training step uses inlined loss computation."""
        from dfm_python.models import KDFM
        import torch
        
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(50, 5)
        model.initialize_from_data(X)
        
        # Training step should work
        batch = X.unsqueeze(0)  # (1, T, N)
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0

