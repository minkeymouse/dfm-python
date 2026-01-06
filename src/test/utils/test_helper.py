"""Tests for utils.helper module."""

import pytest
import numpy as np
from dfm_python.utils.helper import handle_linear_algebra_error
from dfm_python.utils.misc import get_config_attr
from dfm_python.utils.errors import ConfigValidationError


class TestHandleLinearAlgebraError:
    """Test suite for handle_linear_algebra_error."""
    
    def test_successful_operation(self):
        """Test successful operation without error."""
        A = np.eye(2)
        b = np.array([1.0, 1.0])
        # Function signature has *args after keyword args, which is unusual
        # Workaround: pass args as keyword arguments to avoid syntax issues
        # The function will pass them to the operation
        def solve_wrapper(a, b_val):
            return np.linalg.solve(a, b_val)
        result = handle_linear_algebra_error(
            solve_wrapper,
            "matrix solve",
            fallback_value=np.eye(2),
            a=A,
            b_val=b
        )
        assert result is not None
    
    def test_fallback_value(self):
        """Test fallback value on error."""
        def failing_operation(*args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")
        
        fallback = np.eye(2)
        result = handle_linear_algebra_error(
            failing_operation,
            "failing operation",
            fallback_value=fallback
        )
        np.testing.assert_array_equal(result, fallback)
    
    def test_fallback_function(self):
        """Test fallback function on error."""
        def failing_operation(*args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")
        
        def fallback_func(*args, **kwargs):
            return np.eye(2)
        
        result = handle_linear_algebra_error(
            failing_operation,
            "failing operation",
            fallback_func=fallback_func
        )
        assert result is not None
        assert result.shape == (2, 2)
    
    def test_valueerror_handling(self):
        """Test ValueError is also caught and handled."""
        def failing_operation(*args, **kwargs):
            raise ValueError("Invalid matrix")
        
        fallback = np.eye(2)
        result = handle_linear_algebra_error(
            failing_operation,
            "failing operation",
            fallback_value=fallback
        )
        np.testing.assert_array_equal(result, fallback)
    
    def test_fallback_func_with_args(self):
        """Test fallback function receives args and kwargs."""
        def failing_operation(a, b_val):
            raise np.linalg.LinAlgError("Singular matrix")
        
        def fallback_func(a, b_val):
            # Fallback should receive the same arguments
            return a + b_val
        
        A = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = handle_linear_algebra_error(
            failing_operation,
            "failing operation",
            fallback_func=fallback_func,
            a=A,
            b_val=b
        )
        expected = A + b
        np.testing.assert_array_equal(result, expected)
    
    def test_both_fallback_value_and_func(self):
        """Test that fallback_func takes precedence when both provided."""
        def failing_operation(*args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")
        
        def fallback_func(*args, **kwargs):
            return np.ones((2, 2))
        
        fallback_value = np.eye(2)
        result = handle_linear_algebra_error(
            failing_operation,
            "failing operation",
            fallback_value=fallback_value,
            fallback_func=fallback_func
        )
        # fallback_func should be used first (checked before fallback_value)
        np.testing.assert_array_equal(result, np.ones((2, 2)))
    
    def test_no_fallback_provided(self):
        """Test that error is re-raised when no fallback provided."""
        def failing_operation(*args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")
        
        with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
            handle_linear_algebra_error(
                failing_operation,
                "failing operation"
            )


class TestGetConfigAttr:
    """Test suite for get_config_attr."""
    
    def test_get_existing_attr(self):
        """Test getting existing attribute."""
        class Config:
            def __init__(self):
                self.num_factors = 3
        
        config = Config()
        value = get_config_attr(config, 'num_factors', default=2)
        assert value == 3
    
    def test_get_missing_attr_with_default(self):
        """Test getting missing attribute with default."""
        class Config:
            pass
        
        config = Config()
        value = get_config_attr(config, 'missing_attr', default=5)
        assert value == 5
    
    def test_get_required_attr_missing(self):
        """Test getting required attribute that is missing."""
        class Config:
            pass
        
        config = Config()
        with pytest.raises(ConfigValidationError):
            get_config_attr(config, 'required_attr', required=True)
    
    def test_none_config_with_default(self):
        """Test getting attribute from None config."""
        value = get_config_attr(None, 'attr', default=10)
        assert value == 10
    
    def test_none_config_with_required(self):
        """Test getting required attribute from None config raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Config is None"):
            get_config_attr(None, 'required_attr', required=True)



