"""Tests for utils.common module.

NOTE: utils.common module was removed (2026-01-06) as over-engineered.
These tests are kept for reference but marked as skipped since the module no longer exists.
Functions were replaced with direct operations or utilities from config.types.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from dfm_python.utils.errors import DataValidationError
from dfm_python.config.constants import DEFAULT_ZERO_VALUE, MAX_EIGENVALUE, DEFAULT_TORCH_DTYPE

# Skip all tests - utils.common module was removed
pytestmark = pytest.mark.skip(reason="utils.common module was removed (2026-01-06)")


class TestEnsureTensor:
    """Test suite for ensure_tensor function."""
    
    def test_ensure_tensor_from_numpy(self):
        """Test ensure_tensor converts numpy array to tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = ensure_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        # Note: numpy arrays default to float64, torch.from_numpy preserves dtype
        # Use float64 for comparison to match numpy default
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    
    def test_ensure_tensor_from_tensor(self):
        """Test ensure_tensor returns tensor unchanged."""
        tensor_in = torch.tensor([1.0, 2.0, 3.0])
        tensor_out = ensure_tensor(tensor_in)
        assert tensor_out is tensor_in
    
    def test_ensure_tensor_from_list(self):
        """Test ensure_tensor converts list to tensor."""
        lst = [1.0, 2.0, 3.0]
        tensor = ensure_tensor(lst)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_ensure_tensor_from_scalar(self):
        """Test ensure_tensor converts scalar to tensor."""
        scalar = 3.14
        tensor = ensure_tensor(scalar)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([3.14]))
    
    def test_ensure_tensor_with_device(self):
        """Test ensure_tensor moves tensor to specified device."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = ensure_tensor(arr, device='cpu')
        assert tensor.device.type == 'cpu'
    
    def test_ensure_tensor_with_dtype(self):
        """Test ensure_tensor converts to specified dtype."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor = ensure_tensor(arr, dtype=DEFAULT_TORCH_DTYPE)
        assert tensor.dtype == DEFAULT_TORCH_DTYPE
    
    def test_ensure_tensor_with_requires_grad(self):
        """Test ensure_tensor sets requires_grad."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = ensure_tensor(arr, requires_grad=True)
        assert tensor.requires_grad
    
    def test_ensure_tensor_invalid_type(self):
        """Test ensure_tensor raises error for invalid type."""
        with pytest.raises(DataValidationError, match="Cannot convert"):
            ensure_tensor("invalid")


class TestEnsureNumpy:
    """Test suite for ensure_numpy function."""
    
    def test_ensure_numpy_from_numpy(self):
        """Test ensure_numpy returns numpy array unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
    
    def test_ensure_numpy_from_tensor(self):
        """Test ensure_numpy converts tensor to numpy array."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        arr = ensure_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))
    
    def test_ensure_numpy_from_list(self):
        """Test ensure_numpy converts list to numpy array."""
        lst = [1.0, 2.0, 3.0]
        arr = ensure_numpy(lst)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))
    
    def test_ensure_numpy_from_scalar(self):
        """Test ensure_numpy converts scalar to numpy array."""
        scalar = 3.14
        arr = ensure_numpy(scalar)
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([3.14]))
    
    def test_ensure_numpy_with_dtype(self):
        """Test ensure_numpy converts to specified dtype."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        arr = ensure_numpy(tensor, dtype=np.float32)
        assert arr.dtype == np.float32
    
    def test_ensure_numpy_invalid_type(self):
        """Test ensure_numpy raises error for invalid type."""
        with pytest.raises(DataValidationError, match="Cannot convert"):
            ensure_numpy("invalid")


class TestValidateMatrixShape:
    """Test suite for validate_matrix_shape function."""
    
    def test_validate_matrix_shape_correct(self):
        """Test validate_matrix_shape passes for correct shape."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        validate_matrix_shape(matrix, (2, 2))
        # Should not raise
    
    def test_validate_matrix_shape_wildcard(self):
        """Test validate_matrix_shape with wildcard dimension."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        validate_matrix_shape(matrix, (-1, 2))  # First dimension can be any
        # Should not raise
    
    def test_validate_matrix_shape_wrong_dimensions(self):
        """Test validate_matrix_shape raises error for wrong number of dimensions."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(DataValidationError, match="dimensions"):
            validate_matrix_shape(matrix, (2, 2, 2))
    
    def test_validate_matrix_shape_wrong_size(self):
        """Test validate_matrix_shape raises error for wrong size."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(DataValidationError, match="dimension"):
            validate_matrix_shape(matrix, (3, 2))
    
    def test_validate_matrix_shape_tensor(self):
        """Test validate_matrix_shape works with tensors."""
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        validate_matrix_shape(matrix, (2, 2))
        # Should not raise
    
    def test_validate_matrix_shape_invalid_type(self):
        """Test validate_matrix_shape raises error for invalid type."""
        with pytest.raises(DataValidationError, match="must be numpy array or torch Tensor"):
            validate_matrix_shape([1, 2, 3], (3,))


class TestLogTensorStats:
    """Test suite for log_tensor_stats function."""
    
    def test_log_tensor_stats_basic(self):
        """Test log_tensor_stats logs tensor statistics."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Should not raise - just logs
        log_tensor_stats(tensor, "test_tensor")
    
    def test_log_tensor_stats_with_custom_logger(self):
        """Test log_tensor_stats works with custom logger."""
        import logging
        logger = logging.getLogger("test")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        # Should not raise
        log_tensor_stats(tensor, "test_tensor", logger=logger)


class TestSanitizeArray:
    """Test suite for sanitize_array function."""
    
    def test_sanitize_array_with_nan(self):
        """Test sanitize_array replaces NaN with default value."""
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr)
        assert np.all(np.isfinite(result))
        assert result[0] == 1.0
        assert result[1] == DEFAULT_ZERO_VALUE
        assert result[2] == 3.0
    
    def test_sanitize_array_with_inf(self):
        """Test sanitize_array replaces infinity with default value."""
        arr = np.array([1.0, np.inf, 3.0])
        result = sanitize_array(arr)
        assert np.all(np.isfinite(result))
        assert result[0] == 1.0
        assert result[1] == MAX_EIGENVALUE
        assert result[2] == 3.0
    
    def test_sanitize_array_with_neg_inf(self):
        """Test sanitize_array replaces negative infinity."""
        arr = np.array([1.0, -np.inf, 3.0])
        result = sanitize_array(arr)
        assert np.all(np.isfinite(result))
        assert result[0] == 1.0
        assert result[1] == -MAX_EIGENVALUE
        assert result[2] == 3.0
    
    def test_sanitize_array_with_all_issues(self):
        """Test sanitize_array handles NaN, Inf, and -Inf together."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        result = sanitize_array(arr)
        assert np.all(np.isfinite(result))
        assert result[0] == 1.0
        assert result[1] == DEFAULT_ZERO_VALUE
        assert result[2] == MAX_EIGENVALUE
        assert result[3] == -MAX_EIGENVALUE
        assert result[4] == 2.0
    
    def test_sanitize_array_with_custom_nan_value(self):
        """Test sanitize_array uses custom nan_value parameter."""
        arr = np.array([1.0, np.nan, 3.0])
        custom_nan = 42.0
        result = sanitize_array(arr, nan_value=custom_nan)
        assert result[1] == custom_nan
    
    def test_sanitize_array_with_custom_inf_value(self):
        """Test sanitize_array uses custom inf_value parameter."""
        arr = np.array([1.0, np.inf, -np.inf, 3.0])
        custom_inf = 100.0
        result = sanitize_array(arr, inf_value=custom_inf)
        assert result[0] == 1.0
        assert result[1] == custom_inf  # posinf uses inf_value
        assert result[2] == -custom_inf  # neginf uses -inf_value
        assert result[3] == 3.0
    
    def test_sanitize_array_with_finite_array(self):
        """Test sanitize_array returns finite array unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = sanitize_array(arr)
        assert np.array_equal(result, arr)
    
    def test_sanitize_array_2d_array(self):
        """Test sanitize_array works with 2D arrays."""
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        result = sanitize_array(arr)
        assert np.all(np.isfinite(result))
        assert result[0, 0] == 1.0
        assert result[0, 1] == DEFAULT_ZERO_VALUE
        assert result[1, 0] == MAX_EIGENVALUE
        assert result[1, 1] == 2.0
    
    def test_sanitize_array_preserves_shape(self):
        """Test sanitize_array preserves array shape."""
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        result = sanitize_array(arr)
        assert result.shape == arr.shape
    
    def test_sanitize_array_preserves_dtype(self):
        """Test sanitize_array preserves array dtype."""
        arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result = sanitize_array(arr)
        assert result.dtype == np.float32


class TestSelectColumnsByPrefix:
    """Test suite for select_columns_by_prefix function."""
    
    def test_select_columns_by_prefix_basic(self):
        """Test select_columns_by_prefix selects columns correctly."""
        df = pd.DataFrame({
            "D1": [1, 2], "D2": [3, 4],
            "E1": [5, 6], "E2": [7, 8],
            "I1": [9, 10]
        })
        result = select_columns_by_prefix(df, ["D", "E"], count_per_prefix=2)
        assert result == ["D1", "D2", "E1", "E2"]
    
    def test_select_columns_by_prefix_partial_match(self):
        """Test select_columns_by_prefix handles missing columns."""
        df = pd.DataFrame({
            "D1": [1, 2], "D2": [3, 4],
            "E1": [5, 6],
            "I1": [9, 10]
        })
        result = select_columns_by_prefix(df, ["D", "E"], count_per_prefix=2)
        assert result == ["D1", "D2", "E1"]  # E2 missing, not included
    
    def test_select_columns_by_prefix_custom_count(self):
        """Test select_columns_by_prefix with custom count_per_prefix."""
        df = pd.DataFrame({
            "D1": [1, 2], "D2": [3, 4], "D3": [5, 6],
            "E1": [7, 8]
        })
        result = select_columns_by_prefix(df, ["D"], count_per_prefix=2)
        assert result == ["D1", "D2"]  # D3 not included with count=2
    
    def test_select_columns_by_prefix_empty_prefixes(self):
        """Test select_columns_by_prefix with empty prefix list."""
        df = pd.DataFrame({"D1": [1, 2], "E1": [3, 4]})
        result = select_columns_by_prefix(df, [], count_per_prefix=2)
        assert result == []
    
    def test_select_columns_by_prefix_no_matches(self):
        """Test select_columns_by_prefix with no matching columns."""
        df = pd.DataFrame({"X1": [1, 2], "Y1": [3, 4]})
        result = select_columns_by_prefix(df, ["D", "E"], count_per_prefix=2)
        assert result == []
    
    def test_select_columns_by_prefix_multiple_prefixes(self):
        """Test select_columns_by_prefix with multiple prefixes."""
        df = pd.DataFrame({
            "D1": [1], "D2": [2],
            "E1": [3], "E2": [4],
            "I1": [5], "I2": [6],
            "M1": [7]
        })
        result = select_columns_by_prefix(df, ["D", "E", "I"], count_per_prefix=2)
        assert result == ["D1", "D2", "E1", "E2", "I1", "I2"]
    
    def test_select_columns_by_prefix_non_dataframe(self):
        """Test select_columns_by_prefix with object without .columns attribute."""
        obj = {"D1": [1, 2], "D2": [3, 4]}
        result = select_columns_by_prefix(obj, ["D"], count_per_prefix=2)
        assert result == []  # No .columns attribute, returns empty list
    
    def test_select_columns_by_prefix_zero_count(self):
        """Test select_columns_by_prefix with count_per_prefix=0."""
        df = pd.DataFrame({"D1": [1, 2], "E1": [3, 4]})
        result = select_columns_by_prefix(df, ["D", "E"], count_per_prefix=0)
        assert result == []  # No columns selected with count=0


class TestComputeScaleStats:
    """Test suite for compute_scale_stats function."""
    
    def test_compute_scale_stats_numpy_array(self):
        """Test compute_scale_stats with numpy array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val, std_val = compute_scale_stats(arr)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - 3.0) < 1e-6  # Mean should be 3.0
        assert abs(std_val - np.std(arr)) < 1e-6  # Std should match numpy std
    
    def test_compute_scale_stats_tensor(self):
        """Test compute_scale_stats with torch Tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val, std_val = compute_scale_stats(tensor)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - 3.0) < 1e-6  # Mean should be 3.0
        assert abs(std_val - tensor.std().item()) < 1e-6  # Std should match tensor std
    
    def test_compute_scale_stats_2d_array(self):
        """Test compute_scale_stats with 2D numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean_val, std_val = compute_scale_stats(arr)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - np.mean(arr)) < 1e-6
        assert abs(std_val - np.std(arr)) < 1e-6
    
    def test_compute_scale_stats_2d_tensor(self):
        """Test compute_scale_stats with 2D torch Tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean_val, std_val = compute_scale_stats(tensor)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - tensor.mean().item()) < 1e-6
        assert abs(std_val - tensor.std().item()) < 1e-6
    
    def test_compute_scale_stats_standardized_data(self):
        """Test compute_scale_stats with standardized data (mean≈0, std≈1)."""
        # Generate standardized data
        arr = np.random.randn(100)
        mean_val, std_val = compute_scale_stats(arr)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        # For standardized data, mean should be close to 0, std close to 1
        assert abs(mean_val) < 0.5  # Mean should be close to 0
        assert 0.5 < std_val < 1.5  # Std should be close to 1
    
    def test_compute_scale_stats_constant_array(self):
        """Test compute_scale_stats with constant array (std=0)."""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        mean_val, std_val = compute_scale_stats(arr)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - 5.0) < 1e-6
        assert abs(std_val - 0.0) < 1e-6  # Std should be 0 for constant array
    
    def test_compute_scale_stats_single_element(self):
        """Test compute_scale_stats with single element array."""
        arr = np.array([42.0])
        mean_val, std_val = compute_scale_stats(arr)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert abs(mean_val - 42.0) < 1e-6
        # Single element: numpy std returns 0.0 (not NaN)
        assert std_val == 0.0


class TestNormalizeToMatchScale:
    """Test suite for normalize_to_match_scale function."""
    
    def test_normalize_to_match_scale_no_normalization_needed(self):
        """Test normalize_to_match_scale returns unchanged when scales match."""
        prediction = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Similar scale
        normalized, ratio, was_normalized = normalize_to_match_scale(prediction, target)
        assert not was_normalized
        assert np.allclose(normalized, prediction)
        assert 0.5 < ratio < 2.0  # Ratio should be reasonable
    
    def test_normalize_to_match_scale_normalization_applied(self):
        """Test normalize_to_match_scale normalizes when scale mismatch detected."""
        prediction = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # Large scale
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Small scale
        normalized, ratio, was_normalized = normalize_to_match_scale(prediction, target)
        assert was_normalized
        # Normalized prediction should have similar scale to target
        norm_mean, norm_std = compute_scale_stats(normalized)
        target_mean, target_std = compute_scale_stats(target)
        assert abs(norm_std - target_std) < 0.1  # Scales should match after normalization
    
    def test_normalize_to_match_scale_tensor_input(self):
        """Test normalize_to_match_scale works with torch tensors."""
        from dfm_python.config.constants import DEFAULT_SCALE_RATIO_MAX
        # Use extreme scale difference to ensure normalization is triggered
        prediction = torch.tensor([100.0, 200.0, 300.0])  # Large scale
        target = torch.tensor([0.1, 0.2, 0.3])  # Small scale (ratio > DEFAULT_SCALE_RATIO_MAX)
        normalized, ratio, was_normalized = normalize_to_match_scale(prediction, target)
        assert isinstance(normalized, torch.Tensor) or isinstance(normalized, np.ndarray)
        assert was_normalized  # Extreme scale difference, should normalize
    
    def test_normalize_to_match_scale_zero_std_raises_when_requested(self):
        """Test normalize_to_match_scale raises error on zero std when raise_on_zero_std=True."""
        prediction = np.array([5.0, 5.0, 5.0])  # Constant array, std=0
        target = np.array([1.0, 2.0, 3.0])
        from dfm_python.utils.errors import DataError
        with pytest.raises(DataError, match="zero std"):
            normalize_to_match_scale(prediction, target, raise_on_zero_std=True)
    
    def test_normalize_to_match_scale_zero_std_returns_unchanged_when_not_raised(self):
        """Test normalize_to_match_scale returns unchanged on zero std when raise_on_zero_std=False."""
        prediction = np.array([5.0, 5.0, 5.0])  # Constant array, std=0
        target = np.array([1.0, 2.0, 3.0])
        normalized, ratio, was_normalized = normalize_to_match_scale(prediction, target, raise_on_zero_std=False)
        assert not was_normalized
        assert np.allclose(normalized, prediction)
    
    def test_normalize_to_match_scale_2d_arrays(self):
        """Test normalize_to_match_scale works with 2D arrays."""
        from dfm_python.config.constants import DEFAULT_SCALE_RATIO_MAX
        # Use extreme scale difference to ensure normalization is triggered
        prediction = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])  # Large scale
        target = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Small scale (ratio > DEFAULT_SCALE_RATIO_MAX)
        normalized, ratio, was_normalized = normalize_to_match_scale(prediction, target)
        assert was_normalized  # Extreme scale difference, should normalize
        assert normalized.shape == prediction.shape

