"""Tests for utils.common module."""

import pytest
import numpy as np
import torch
import pandas as pd
from dfm_python.utils.common import ensure_tensor, ensure_numpy, validate_matrix_shape, log_tensor_stats, sanitize_array, select_columns_by_prefix
from dfm_python.utils.errors import DataValidationError
from dfm_python.config.constants import DEFAULT_ZERO_VALUE, MAX_EIGENVALUE


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
        tensor = ensure_tensor(arr, dtype=torch.float32)
        assert tensor.dtype == torch.float32
    
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

