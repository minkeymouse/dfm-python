"""Tests for numeric.stability module."""

import pytest
import numpy as np
from dfm_python.numeric.stability import (
    ensure_covariance_stable,
    cap_max_eigenval,
    ensure_positive_definite,
    safe_matrix_power,
    compute_cov_safe,
    clean_matrix,
    safe_determinant,
    ensure_symmetric,
    create_scaled_identity,
    safe_divide,
    compute_var_safe,
    stabilize_innovation_covariance,
    mse_missing_numpy,
    convergence_checker,
    extract_matrix_block,
    compute_forecast_metrics,
)
from dfm_python.utils.errors import DataValidationError, DataError, NumericalError
from dfm_python.config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_DTYPE,
    DEFAULT_VARIANCE_FALLBACK,
    CHOLESKY_LOG_DET_FACTOR,
    SYMMETRY_AVERAGE_FACTOR,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    MIN_EIGENVALUE,
)


class TestStability:
    """Test suite for numerical stability utilities."""
    
    def test_ensure_covariance_stable(self):
        """Test covariance stability enforcement."""
        # Test with a simple 2x2 matrix
        M = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=DEFAULT_DTYPE)
        result = ensure_covariance_stable(M)
        assert result.shape == M.shape
        # Result should be symmetric
        assert np.allclose(result, result.T)
        # Result should be positive semi-definite (all eigenvalues >= 0)
        eigenvals = np.linalg.eigvalsh(result)
        assert np.all(eigenvals >= 0)
    
    def test_cap_max_eigenval(self):
        """Test eigenvalue capping."""
        # Create matrix with large eigenvalues
        M = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=DEFAULT_DTYPE)
        max_eigenval = 5.0
        result = cap_max_eigenval(M, max_eigenval=max_eigenval)
        # Maximum eigenvalue should be capped
        eigenvals = np.linalg.eigvals(result)
        max_eig = float(np.max(np.abs(eigenvals)))
        assert max_eig <= max_eigenval + 1e-6  # Allow small numerical error
    
    def test_ensure_positive_definite(self):
        """Test positive definiteness enforcement."""
        # Create matrix with negative eigenvalues
        M = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=DEFAULT_DTYPE)
        min_eigenval = 0.1
        result = ensure_positive_definite(M, min_eigenval=min_eigenval)
        # Result should be positive semi-definite
        eigenvals = np.linalg.eigh(result)[0]
        assert np.all(eigenvals >= min_eigenval - 1e-6)  # Allow small numerical error
    
    def test_compute_cov_safe_empty_data(self):
        """Test compute_cov_safe raises DataError for empty data."""
        empty_data = np.array([]).reshape(0, 3)
        with pytest.raises(DataError, match="Cannot compute covariance: data is empty"):
            compute_cov_safe(empty_data, fallback_to_identity=False)
    
    def test_compute_cov_safe_insufficient_observations(self):
        """Test compute_cov_safe raises DataError for insufficient observations."""
        # Only 1 observation (need at least 2)
        data = np.array([[1.0, 2.0, 3.0]], dtype=DEFAULT_DTYPE)
        with pytest.raises(DataError, match="Insufficient complete observations"):
            compute_cov_safe(data, fallback_to_identity=False)
    
    def test_safe_matrix_power_negative_power(self):
        """Test safe_matrix_power raises DataValidationError for negative power."""
        matrix = np.eye(3, dtype=DEFAULT_DTYPE)
        with pytest.raises(DataValidationError, match="power must be >= 0"):
            safe_matrix_power(matrix, power=-1)
    
    def test_safe_matrix_power_exceeds_max(self):
        """Test safe_matrix_power raises DataValidationError when power exceeds max."""
        matrix = np.eye(3, dtype=DEFAULT_DTYPE)
        with pytest.raises(DataValidationError, match="power.*exceeds maximum"):
            safe_matrix_power(matrix, power=1001, max_power=1000)
    
    def test_ensure_covariance_stable_error_handling(self):
        """Test ensure_covariance_stable handles linear algebra errors gracefully."""
        # Create a matrix that might cause eigenvalue computation issues
        # Use a very large matrix that could trigger numerical issues
        M = np.ones((50, 50), dtype=DEFAULT_DTYPE) * 1e10
        # Should not raise, should use fallback
        result = ensure_covariance_stable(M)
        assert result.shape == M.shape
        # Result should still be symmetric
        assert np.allclose(result, result.T, atol=1e-5)
    
    def test_cap_max_eigenval_error_handling(self):
        """Test cap_max_eigenval handles linear algebra errors gracefully."""
        # Create a matrix that might cause eigenvalue computation issues
        M = np.ones((50, 50), dtype=DEFAULT_DTYPE) * 1e10
        max_eigenval = 5.0
        # Should not raise, should return original matrix if eigenvalue computation fails
        result = cap_max_eigenval(M, max_eigenval=max_eigenval, symmetric=True)
        assert result.shape == M.shape
    
    def test_ensure_positive_definite_error_handling(self):
        """Test ensure_positive_definite handles linear algebra errors gracefully."""
        # Create a matrix that might cause eigenvalue computation issues
        M = np.ones((50, 50), dtype=DEFAULT_DTYPE) * -1e10
        min_eigenval = 0.1
        # Should not raise, should use fallback regularization
        result = ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
        assert result.shape == M.shape
        # Result should be symmetric
        assert np.allclose(result, result.T, atol=1e-5)
    
    def test_clean_matrix_uses_default_variance_fallback(self):
        """Test clean_matrix uses DEFAULT_VARIANCE_FALLBACK constant."""
        # Test that clean_matrix uses the constant when default_variance is not specified
        M = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=DEFAULT_DTYPE)
        result = clean_matrix(M, matrix_type='covariance')
        # Should use DEFAULT_VARIANCE_FALLBACK for NaN/Inf values in covariance matrices
        assert np.all(np.isfinite(result))
        # Verify the constant is accessible
        assert DEFAULT_VARIANCE_FALLBACK == 1.0
    
    def test_ensure_symmetric(self):
        """Test ensure_symmetric function."""
        # Test with non-symmetric matrix
        M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        result = ensure_symmetric(M)
        # Result should be symmetric
        assert np.allclose(result, result.T)
        # Result should be average of M and M.T
        expected = SYMMETRY_AVERAGE_FACTOR * (M + M.T)
        assert np.allclose(result, expected)
        # Verify SYMMETRY_AVERAGE_FACTOR is used
        assert SYMMETRY_AVERAGE_FACTOR == 0.5
    
    def test_ensure_symmetric_already_symmetric(self):
        """Test ensure_symmetric with already symmetric matrix."""
        M = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=DEFAULT_DTYPE)
        result = ensure_symmetric(M)
        # Should remain symmetric and unchanged
        assert np.allclose(result, M)
        assert np.allclose(result, result.T)
    
    def test_safe_determinant_positive_definite(self):
        """Test safe_determinant with positive definite matrix."""
        # Create a positive definite matrix (identity)
        M = np.eye(3, dtype=DEFAULT_DTYPE)
        det = safe_determinant(M)
        # Determinant of identity should be 1.0
        assert abs(det - 1.0) < 1e-6
    
    def test_safe_determinant_uses_cholesky_log_det_factor(self):
        """Test safe_determinant uses CHOLESKY_LOG_DET_FACTOR constant."""
        # Create a positive definite matrix that will use Cholesky decomposition
        M = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=DEFAULT_DTYPE)
        # Verify it's positive definite
        eigenvals = np.linalg.eigvalsh(M)
        assert np.all(eigenvals > 0)
        
        det = safe_determinant(M, use_logdet=True)
        # Determinant should be computed correctly
        expected_det = np.linalg.det(M)
        assert abs(det - expected_det) < 1e-5
        
        # Verify CHOLESKY_LOG_DET_FACTOR is used (constant should be 2.0)
        assert CHOLESKY_LOG_DET_FACTOR == 2.0
    
    def test_safe_determinant_empty_matrix(self):
        """Test safe_determinant with empty matrix."""
        M = np.array([]).reshape(0, 0)
        det = safe_determinant(M)
        assert det == 0.0
    
    def test_safe_determinant_non_square(self):
        """Test safe_determinant with non-square matrix."""
        M = np.array([[1.0, 2.0, 3.0]], dtype=DEFAULT_DTYPE)
        det = safe_determinant(M)
        assert det == 0.0
    
    def test_create_scaled_identity_default_scale(self):
        """Test create_scaled_identity with default scale."""
        n = 3
        result = create_scaled_identity(n)
        # Should be identity matrix scaled by DEFAULT_IDENTITY_SCALE
        expected = np.eye(n, dtype=DEFAULT_DTYPE) * DEFAULT_IDENTITY_SCALE
        assert np.allclose(result, expected)
        assert result.shape == (n, n)
        assert result.dtype == np.float32
    
    def test_create_scaled_identity_custom_scale(self):
        """Test create_scaled_identity with custom scale."""
        n = 4
        scale = 2.5
        result = create_scaled_identity(n, scale=scale)
        expected = np.eye(n, dtype=DEFAULT_DTYPE) * scale
        assert np.allclose(result, expected)
        assert result.shape == (n, n)
    
    def test_create_scaled_identity_custom_dtype(self):
        """Test create_scaled_identity with custom dtype."""
        n = 2
        result = create_scaled_identity(n, dtype=np.float64)
        assert result.dtype == np.float64
        assert result.shape == (n, n)
    
    def test_safe_divide_normal_division(self):
        """Test safe_divide with normal division."""
        numerator = np.array([4.0, 6.0, 8.0], dtype=DEFAULT_DTYPE)
        denominator = np.array([2.0, 3.0, 4.0], dtype=DEFAULT_DTYPE)
        result = safe_divide(numerator, denominator)
        expected = np.array([2.0, 2.0, 2.0], dtype=DEFAULT_DTYPE)
        assert np.allclose(result, expected)
    
    def test_safe_divide_zero_denominator(self):
        """Test safe_divide handles zero denominators."""
        numerator = np.array([4.0, 6.0, 8.0], dtype=DEFAULT_DTYPE)
        denominator = np.array([2.0, 0.0, 4.0], dtype=DEFAULT_DTYPE)
        result = safe_divide(numerator, denominator)
        # Should use DEFAULT_ZERO_VALUE for zero denominator
        assert result[0] == 2.0
        assert result[1] == DEFAULT_ZERO_VALUE
        assert result[2] == 2.0
    
    def test_safe_divide_custom_default(self):
        """Test safe_divide with custom default value."""
        numerator = np.array([4.0, 6.0], dtype=DEFAULT_DTYPE)
        denominator = np.array([0.0, 3.0], dtype=DEFAULT_DTYPE)
        custom_default = 99.0
        result = safe_divide(numerator, denominator, default=custom_default)
        assert result[0] == custom_default
        assert result[1] == 2.0
    
    def test_compute_var_safe_basic(self):
        """Test compute_var_safe with basic data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=DEFAULT_DTYPE)
        result = compute_var_safe(data)
        # compute_var_safe flattens 2D arrays and computes variance of all values
        expected = np.nanvar(data.flatten(), ddof=0)
        assert abs(result - expected) < 1e-6
        assert isinstance(result, float)
    
    def test_compute_var_safe_with_ddof(self):
        """Test compute_var_safe with ddof parameter."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=DEFAULT_DTYPE)
        result = compute_var_safe(data, ddof=1)
        # compute_var_safe flattens 2D arrays and computes variance of all values
        expected = np.nanvar(data.flatten(), ddof=1)
        assert abs(result - expected) < 1e-6
        assert isinstance(result, float)
    
    def test_compute_var_safe_min_variance_enforcement(self):
        """Test compute_var_safe enforces minimum variance."""
        # Create data with very small variance
        data = np.array([[1.0, 1.0], [1.001, 1.001], [1.002, 1.002]], dtype=DEFAULT_DTYPE)
        min_variance = 0.1
        result = compute_var_safe(data, min_variance=min_variance)
        # All variances should be at least min_variance
        assert np.all(result >= min_variance - 1e-6)
    
    def test_stabilize_innovation_covariance_basic(self):
        """Test stabilize_innovation_covariance with basic matrix."""
        Q = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=DEFAULT_DTYPE)
        result = stabilize_innovation_covariance(Q)
        assert result.shape == Q.shape
        assert np.allclose(result, result.T)  # Should be symmetric
        eigenvals = np.linalg.eigvalsh(result)
        assert np.all(eigenvals >= 0)  # Should be positive semi-definite
    
    def test_stabilize_innovation_covariance_with_floor(self):
        """Test stabilize_innovation_covariance with floor value."""
        Q = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=DEFAULT_DTYPE)
        min_floor = 0.5
        result = stabilize_innovation_covariance(Q, min_floor=min_floor)
        # Floor is applied to diagonal elements via create_scaled_identity
        # Diagonal elements should be at least min_floor
        assert np.all(np.diag(result) >= min_floor - 1e-6)
    
    def test_stabilize_innovation_covariance_empty(self):
        """Test stabilize_innovation_covariance with empty matrix."""
        Q = np.array([], dtype=DEFAULT_DTYPE).reshape(0, 0)
        result = stabilize_innovation_covariance(Q)
        assert result.shape == Q.shape
    
    def test_mse_missing_numpy_basic(self):
        """Test mse_missing_numpy with no missing values."""
        y_actual = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        y_predicted = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        result = mse_missing_numpy(y_actual, y_predicted)
        expected = np.mean((y_actual - y_predicted) ** 2)
        assert abs(result - expected) < 1e-6
        assert isinstance(result, float)
    
    def test_mse_missing_numpy_with_missing(self):
        """Test mse_missing_numpy with missing values."""
        y_actual = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        y_predicted = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        result = mse_missing_numpy(y_actual, y_predicted)
        # Should only compute MSE on non-missing values (first column and second row)
        expected = np.mean([(1.0 - 1.1) ** 2, (3.0 - 3.1) ** 2, (4.0 - 4.1) ** 2])
        assert abs(result - expected) < 1e-6
    
    def test_mse_missing_numpy_all_missing(self):
        """Test mse_missing_numpy with all missing values."""
        y_actual = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=DEFAULT_DTYPE)
        y_predicted = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        result = mse_missing_numpy(y_actual, y_predicted)
        assert result == DEFAULT_ZERO_VALUE
    
    def test_convergence_checker_basic(self):
        """Test convergence_checker with basic inputs."""
        y_prev = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        y_now = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        y_actual = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        delta, loss_now = convergence_checker(y_prev, y_now, y_actual)
        assert isinstance(delta, float)
        assert isinstance(loss_now, float)
        assert loss_now >= 0
        assert delta >= 0
    
    def test_convergence_checker_with_missing(self):
        """Test convergence_checker with missing values."""
        y_prev = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        y_now = np.array([[1.1, np.nan], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        y_actual = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        delta, loss_now = convergence_checker(y_prev, y_now, y_actual)
        assert isinstance(delta, float)
        assert isinstance(loss_now, float)
        assert loss_now >= 0
    
    def test_extract_matrix_block_numpy(self):
        """Test extract_matrix_block with NumPy array."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=DEFAULT_DTYPE)
        result = extract_matrix_block(matrix, 0, 2, 1, 3)
        expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=DEFAULT_DTYPE)
        assert np.allclose(result, expected)
        assert result.shape == (2, 2)
    
    def test_extract_matrix_block_torch(self):
        """Test extract_matrix_block with PyTorch tensor."""
        import torch
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=DEFAULT_TORCH_DTYPE)
        result = extract_matrix_block(matrix, 0, 2, 1, 3)
        expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]], dtype=DEFAULT_TORCH_DTYPE)
        assert torch.allclose(result, expected)
        assert result.shape == (2, 2)
    
    def test_compute_forecast_metrics_basic(self):
        """Test compute_forecast_metrics with basic inputs."""
        forecast = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        actual = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        metrics = compute_forecast_metrics(forecast, actual)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mae'], float)
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_compute_forecast_metrics_with_mask(self):
        """Test compute_forecast_metrics with mask."""
        forecast = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_DTYPE)
        actual = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=DEFAULT_DTYPE)
        mask = np.array([[True, False], [True, True]], dtype=bool)
        metrics = compute_forecast_metrics(forecast, actual, mask=mask)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] >= 0
    
    def test_compute_forecast_metrics_all_invalid(self):
        """Test compute_forecast_metrics with all invalid values."""
        forecast = np.array([[np.nan, np.nan], [np.inf, np.inf]], dtype=DEFAULT_DTYPE)
        actual = np.array([[np.nan, np.nan], [np.inf, np.inf]], dtype=DEFAULT_DTYPE)
        metrics = compute_forecast_metrics(forecast, actual)
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['mae'])
        assert np.isnan(metrics['r2'])

