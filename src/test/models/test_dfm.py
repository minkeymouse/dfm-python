"""Tests for models.dfm module."""

import pytest
import numpy as np
import pandas as pd
from dfm_python.models.dfm import DFM
from dfm_python.models.dfm.mixed_freq import find_slower_frequency
from dfm_python.config import DFMConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.utils.errors import ModelNotTrainedError, ConfigurationError, ModelNotInitializedError
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
            frequency={'series_0': 'w', 'series_1': 'w', 'series_2': 'w'},
            clock='w'
        )
        data['date'] = pd.date_range(start='2020-01-01', periods=T, freq='W')
        return DFMDataset(config=config, data=data, time_index='date')
    
    def test_dfm_initialization_requires_dataset(self, sample_dataset):
        """Test DFM requires dataset in __init__."""
        config = sample_dataset.config
        
        model = DFM(dataset=sample_dataset, config=config)
        assert model._dataset is sample_dataset
        assert model._config is not None
        
        with pytest.raises((TypeError, ModelNotInitializedError, ConfigurationError)):
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
        with pytest.raises((ModelNotTrainedError, ModelNotInitializedError)):
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
    
    @pytest.fixture
    def mixed_freq_dataset_sparse_monthly(self):
        """Create mixed-frequency dataset with sparse monthly data (expected pattern).
        
        In mixed-frequency setups, monthly series on weekly clock should have
        many missing values (~80-90% missing). This is expected behavior, not an error.
        """
        np.random.seed(42)
        T = 200  # 200 weeks
        n_weekly = 3
        n_monthly = 2
        
        # Create weekly data (all observations present)
        dates = pd.date_range(start='2020-01-01', periods=T, freq='W')
        weekly_data = np.random.randn(T, n_weekly)
        
        # Create monthly data with sparse observations (only ~15-20% of weeks have data)
        # Monthly data typically appears at month-end weeks
        monthly_data = np.full((T, n_monthly), np.nan)
        monthly_indices = np.arange(0, T, step=4)  # Roughly monthly (every 4 weeks)
        monthly_indices = monthly_indices[:int(T * 0.15)]  # ~15% of weeks have monthly data
        monthly_data[monthly_indices, :] = np.random.randn(len(monthly_indices), n_monthly)
        
        # Combine: weekly first, then monthly (required ordering)
        all_data = np.hstack([weekly_data, monthly_data])
        columns = [f'weekly_{i}' for i in range(n_weekly)] + [f'monthly_{i}' for i in range(n_monthly)]
        
        data = pd.DataFrame(all_data, columns=columns, index=dates)
        data = data.reset_index()
        data.rename(columns={'index': 'date'}, inplace=True)
        
        config = DFMConfig(
            blocks={
                'Block_Global': {
                    'num_factors': 2,
                    'series': columns
                }
            },
            frequency={
                'w': [f'weekly_{i}' for i in range(n_weekly)],
                'm': [f'monthly_{i}' for i in range(n_monthly)]
            },
            clock='w',
            tent_weights={'m:w': [1, 2, 1]}  # Tent kernel for monthly-to-weekly aggregation
        )
        
        return DFMDataset(config=config, data=data, time_index='date')
    
    def test_mixed_freq_initialization_sparse_monthly(self, mixed_freq_dataset_sparse_monthly):
        """Test that mixed-frequency initialization handles sparse monthly data correctly.
        
        Monthly series on weekly clock should have many missing values (expected).
        Initialization should:
        1. Use imputation for regression if insufficient observations
        2. Still align tent kernel factors with actual monthly observations
        3. Initialize loadings (not all zeros) even with sparse data
        4. Complete initialization without errors
        """
        dataset = mixed_freq_dataset_sparse_monthly
        
        # Verify data has expected sparsity
        monthly_cols = [col for col in dataset.variables.columns if col.startswith('monthly_')]
        for col in monthly_cols:
            missing_ratio = dataset.variables[col].isna().sum() / len(dataset.variables)
            assert missing_ratio > 0.7, f"Monthly series {col} should have >70% missing (got {missing_ratio:.1%})"
        
        # Create and fit model
        model = DFM(dataset=dataset, config=dataset.config)
        
        # Fit should complete without errors
        state = model.fit()
        
        # Verify state is initialized
        assert state is not None
        assert hasattr(state, 'A')
        assert hasattr(state, 'C')
        assert hasattr(state, 'Q')
        assert hasattr(state, 'R')
        
        # Verify C matrix has non-zero loadings for monthly series
        # (even if sparse, initialization should produce some loadings)
        monthly_indices = [i for i, col in enumerate(dataset.variables.columns) if col.startswith('monthly_')]
        for idx in monthly_indices:
            monthly_loadings = state.C[idx, :]
            # At least some loadings should be non-zero (not all skipped)
            # Note: if all loadings are zero, it means initialization failed for that series
            assert np.any(np.abs(monthly_loadings) > 1e-10), \
                f"Monthly series {idx} has all-zero loadings - initialization may have failed"
        
        # Verify matrices have correct shapes
        assert state.A.shape[0] == state.A.shape[1], "A should be square"
        assert state.C.shape[0] == len(dataset.variables.columns), "C rows should match number of series"
        assert state.Q.shape == state.A.shape, "Q should match A shape"
        assert state.R.shape[0] == len(dataset.variables.columns), "R should match number of series"
        
        # Verify no NaN or Inf in initialized matrices
        assert np.all(np.isfinite(state.A)), "A should not contain NaN/Inf"
        assert np.all(np.isfinite(state.C)), "C should not contain NaN/Inf"
        assert np.all(np.isfinite(state.Q)), "Q should not contain NaN/Inf"
        assert np.all(np.isfinite(state.R)), "R should not contain NaN/Inf"
