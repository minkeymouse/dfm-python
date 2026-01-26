"""Tests for models.dfm module."""

import pytest
import numpy as np
import pandas as pd
from dfm_python.models.dfm import DFM
from dfm_python.models.dfm.mixed_freq import find_slower_frequency
from dfm_python.config import DFMConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.utils.errors import ModelNotTrainedError, ConfigurationError
from dfm_python.config.constants import DEFAULT_DTYPE


class TestDFM:
    """Test suite for DFM model."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        T = 50
        n_series = 3
        data = pd.DataFrame(
            np.random.randn(T, n_series),
            columns=['series_0', 'series_1', 'series_2']
        )
        config = DFMConfig(
            blocks={'block1': {'num_factors': 1, 'series': ['series_0', 'series_1', 'series_2']}},
            frequency={'w': ['series_0', 'series_1', 'series_2']},
            clock='w'
        )
        return DFMDataset(config=config, data=data)
    
    def test_dfm_initialization_requires_dataset(self, sample_dataset):
        """Test DFM requires dataset in __init__."""
        config = sample_dataset.config
        
        model = DFM(dataset=sample_dataset, config=config)
        assert model._dataset is sample_dataset
        assert model._config is not None
        
        with pytest.raises((ModelNotInitializedError, ConfigurationError)):
            DFM()  # Missing dataset
    
    def test_dfm_fit(self, sample_dataset):
        """Test DFM fit method."""
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        
        state = model.fit()
        assert state is not None
        assert hasattr(state, 'A')
        assert hasattr(state, 'C')
    
    def test_dfm_predict_not_trained(self, sample_dataset):
        """Test DFM predict raises error when model not trained."""
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        with pytest.raises(ModelNotTrainedError):
            model.predict(horizon=5)
    
    def test_dfm_predict_with_data(self, sample_dataset):
        """Test DFM predict with data parameter."""
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        model.fit()
        
        X_forecast, Z_forecast = model.predict(horizon=5)
        assert X_forecast.shape[0] == 5
        assert Z_forecast.shape[0] == 5
    
    def test_dfm_get_result_not_trained(self, sample_dataset):
        """Test DFM get_result raises error when model not trained."""
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        with pytest.raises(ModelNotTrainedError):
            model.get_result()
    
    def test_find_slower_frequency_from_tent_weights_dict(self):
        """Test find_slower_frequency returns frequency from tent_weights_dict."""
        tent_weights_dict = {
            'd': np.array([1.0, 2.0]),
            'w': np.array([3.0, 4.0]),
            'm': np.array([5.0, 6.0])
        }
        
        slower_freq = find_slower_frequency('d', tent_weights_dict)
        assert slower_freq is not None
        assert slower_freq != 'd'
        assert slower_freq in tent_weights_dict
    
    def test_find_slower_frequency_from_hierarchy(self):
        """Test find_slower_frequency returns frequency from hierarchy."""
        slower_freq = find_slower_frequency('d', None)
        assert slower_freq is None or isinstance(slower_freq, str)
