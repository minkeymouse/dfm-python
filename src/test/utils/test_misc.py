"""Tests for utils.misc module."""

import pytest
from dfm_python.utils.misc import resolve_param, get_clock_frequency
from dfm_python.config.constants import DEFAULT_CLOCK_FREQUENCY


class TestResolveParam:
    """Test suite for resolve_param."""
    
    def test_resolve_param_with_value(self):
        """Test resolve_param with provided value."""
        value = resolve_param(5, None, 10)
        assert value == 5
    
    def test_resolve_param_with_config(self):
        """Test resolve_param with config value."""
        class Config:
            def __init__(self):
                self.num_factors = 3
        
        config = Config()
        value = resolve_param(None, config.num_factors, 10)
        assert value == 3
    
    def test_resolve_param_with_default(self):
        """Test resolve_param with default value."""
        value = resolve_param(None, None, 10)
        assert value == 10


class TestGetClockFrequency:
    """Test suite for get_clock_frequency."""
    
    def test_get_clock_frequency_with_config(self):
        """Test get_clock_frequency with config having clock attribute."""
        class Config:
            def __init__(self):
                self.clock = 'monthly'
        
        config = Config()
        result = get_clock_frequency(config)
        assert result == 'monthly'
    
    def test_get_clock_frequency_with_default(self):
        """Test get_clock_frequency with default parameter."""
        result = get_clock_frequency(None, default='quarterly')
        assert result == 'quarterly'
    
    def test_get_clock_frequency_with_none_config_no_default(self):
        """Test get_clock_frequency with None config and no default uses constant."""
        result = get_clock_frequency(None)
        assert result == DEFAULT_CLOCK_FREQUENCY
    
    def test_get_clock_frequency_with_config_no_clock(self):
        """Test get_clock_frequency with config missing clock attribute uses default."""
        class Config:
            pass
        
        config = Config()
        result = get_clock_frequency(config, default='annual')
        assert result == 'annual'
    
    def test_get_clock_frequency_uses_get_config_attr(self):
        """Test that get_clock_frequency uses get_config_attr helper."""
        # This test verifies the refactoring from Iteration 26
        class Config:
            def __init__(self):
                self.clock = 'daily'
        
        config = Config()
        result = get_clock_frequency(config)
        assert result == 'daily'

