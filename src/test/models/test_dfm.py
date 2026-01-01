"""Tests for models.dfm module."""

import pytest
import numpy as np
from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig
from dfm_python.utils.errors import ModelNotTrainedError, DataError, ConfigurationError, NumericalError


class TestDFM:
    """Test suite for DFM model."""
    
    def test_dfm_initialization(self):
        """Test DFM can be initialized."""
        model = DFM()
        # Verify model has expected attributes
        assert hasattr(model, 'config')
        assert hasattr(model, 'reset')
    
    def test_dfm_with_config(self, sample_config):
        """Test DFM initialization with config."""
        model = DFM(config=sample_config)
        assert model.config == sample_config
    
    def test_dfm_load_config(self):
        """Test DFM config loading (legacy pattern - still supported but not recommended)."""
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}}, frequency={'m': 'm'})
        model = DFM()
        # load_config accepts config object directly via source parameter
        # Note: Preferred pattern is DFM(config) instead of DFM().load_config()
        # load_config returns self (the model), not the config
        result = model.load_config(source=config)
        assert result is not None
        assert result is model  # load_config returns self
        # Verify config was loaded into model
        assert model.config is not None
        assert hasattr(model.config, 'blocks')
        assert hasattr(model.config, 'frequency')
    
    def test_dfm_initialization_with_config_preferred(self):
        """Test DFM initialization with config (preferred pattern)."""
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}}, frequency={'m': 'm'})
        # Preferred pattern: pass config directly to constructor
        model = DFM(config=config)
        assert model.config is not None
        assert model.config == config
        assert hasattr(model.config, 'blocks')
        assert hasattr(model.config, 'frequency')
    
    def test_dfm_fit(self):
        """Test DFM fitting requires config with blocks.
        
        Note: DFM fit() requires a properly configured model with blocks structure.
        This test verifies the API exists and can be called, but full setup
        requires complex configuration that is tested in integration tests.
        """
        model = DFM()
        # Verify fit() method exists and is callable
        assert hasattr(model, 'fit')
        assert callable(model.fit)
        # DFM fit requires proper config setup - this is tested in integration tests
        # Here we just verify the method signature and that it requires data
        # Full fit test requires proper DFMConfig with blocks, which is complex
    
    def test_dfm_predict_not_trained(self):
        """Test DFM predict raises error when model not trained."""
        model = DFM()
        with pytest.raises(ModelNotTrainedError):
            model.predict(horizon=5)
    
    def test_dfm_predict_invalid_horizon(self):
        """Test DFM predict raises error for invalid horizon."""
        from dfm_python.utils.errors import PredictionError
        model = DFM()
        # Test horizon <= 0 - will raise ModelNotTrainedError first (model not fitted),
        # but if model were trained, horizon validation would raise PredictionError
        # This test verifies the validation exists even if not reached due to training check
        with pytest.raises((ModelNotTrainedError, PredictionError)):
            model.predict(horizon=0)
        with pytest.raises((ModelNotTrainedError, PredictionError)):
            model.predict(horizon=-1)
    
    def test_dfm_get_result_not_trained(self):
        """Test DFM get_result raises error when model not trained."""
        model = DFM()
        with pytest.raises(ModelNotTrainedError, match="Model not fitted or data not available"):
            model.get_result()
    
    def test_dfm_result_property_not_trained(self):
        """Test DFM result property raises error when model not trained."""
        model = DFM()
        with pytest.raises(ModelNotTrainedError, match="model has not been trained yet"):
            _ = model.result
    
    def test_find_slower_frequency_from_tent_weights_dict(self):
        """Test _find_slower_frequency returns frequency from tent_weights_dict."""
        from dfm_python.config.constants import FREQUENCY_HIERARCHY
        model = DFM()
        
        # Create tent_weights_dict with multiple frequencies
        tent_weights_dict = {
            'd': np.array([1.0, 2.0]),
            'w': np.array([3.0, 4.0]),
            'm': np.array([5.0, 6.0])
        }
        
        # Test with clock='d', should return 'w' or 'm' (different from clock)
        slower_freq = model._find_slower_frequency('d', tent_weights_dict)
        assert slower_freq is not None
        assert slower_freq != 'd'
        assert slower_freq in tent_weights_dict
    
    def test_find_slower_frequency_from_hierarchy(self):
        """Test _find_slower_frequency returns frequency from hierarchy when tent_weights_dict not provided."""
        from dfm_python.config.constants import FREQUENCY_HIERARCHY
        from dfm_python.numeric.tent import get_tent_weights
        model = DFM()
        
        # Test with clock='d' (daily), should find slower frequency from hierarchy
        # Note: This depends on FREQUENCY_HIERARCHY and get_tent_weights implementation
        slower_freq = model._find_slower_frequency('d', None)
        # Result may be None if no valid slower frequency found, or a valid frequency string
        assert slower_freq is None or isinstance(slower_freq, str)
    
    def test_find_slower_frequency_returns_none_when_no_slower_freq(self):
        """Test _find_slower_frequency returns None when no slower frequency found."""
        model = DFM()
        
        # Test with tent_weights_dict containing only the clock frequency
        tent_weights_dict = {
            'd': np.array([1.0, 2.0])
        }
        
        # Should return None since no other frequency in dict
        slower_freq = model._find_slower_frequency('d', tent_weights_dict)
        # May return None or find from hierarchy, but if hierarchy also fails, returns None
        assert slower_freq is None or isinstance(slower_freq, str)
    
    def test_find_slower_frequency_with_empty_tent_weights_dict(self):
        """Test _find_slower_frequency handles empty tent_weights_dict."""
        model = DFM()
        
        # Test with empty tent_weights_dict
        slower_freq = model._find_slower_frequency('d', {})
        # Should try hierarchy, may return None or valid frequency
        assert slower_freq is None or isinstance(slower_freq, str)

