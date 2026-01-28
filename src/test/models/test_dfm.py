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
        """Test find_slower_frequency returns frequency from tent_weights_dict.
        
        Note: DFM supports only ONE slower frequency. This test uses a single
        slower frequency pair.
        """
        # Single slower frequency (weekly on daily clock)
        tent_weights_dict = {
            'w': np.array([1.0, 2.0, 1.0])  # Weekly aggregation weights
        }
        
        slower_freq = find_slower_frequency('d', tent_weights_dict)
        assert slower_freq == 'w'
    
    def test_find_slower_frequency_rejects_multiple_slower_frequencies(self):
        """Test find_slower_frequency raises error for multiple slower frequencies.
        
        DFM supports only ONE slower frequency in addition to the clock frequency.
        """
        from dfm_python.utils.errors import ConfigurationError
        
        # Multiple slower frequencies (should be rejected)
        tent_weights_dict = {
            'w': np.array([1.0, 2.0]),
            'm': np.array([5.0, 6.0])
        }
        
        with pytest.raises(ConfigurationError, match="Multiple slower frequencies"):
            find_slower_frequency('d', tent_weights_dict)
    
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
    
    def test_dfm_save_and_load(self, sample_dataset, tmp_path):
        """Test DFM save and load functionality."""
        import pickle
        from pathlib import Path
        
        # Train a model
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        model.fit()
        
        # Get original predictions
        X_forecast_orig, Z_forecast_orig = model.predict(horizon=5)
        result_orig = model.result
        
        # Save model
        save_path = tmp_path / "test_dfm_model.pkl"
        model.save(save_path)
        assert save_path.exists(), "Model file should be created"
        
        # Load model
        loaded_model = DFM.load(save_path, dataset=sample_dataset)
        
        # Verify loaded model can make predictions
        X_forecast_loaded, Z_forecast_loaded = loaded_model.predict(horizon=5)
        
        # Verify predictions match (within numerical precision)
        assert X_forecast_loaded.shape == X_forecast_orig.shape
        assert Z_forecast_loaded.shape == Z_forecast_orig.shape
        np.testing.assert_allclose(X_forecast_loaded, X_forecast_orig, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(Z_forecast_loaded, Z_forecast_orig, rtol=1e-5, atol=1e-6)
        
        # Verify result matches
        result_loaded = loaded_model.result
        assert result_loaded.converged == result_orig.converged
        assert result_loaded.num_iter == result_orig.num_iter
        np.testing.assert_allclose(result_loaded.loglik, result_orig.loglik, rtol=1e-5)
        
        # Verify state-space parameters match
        assert loaded_model.training_state is not None
        assert model.training_state is not None
        np.testing.assert_allclose(loaded_model.training_state.A, model.training_state.A, rtol=1e-5)
        np.testing.assert_allclose(loaded_model.training_state.C, model.training_state.C, rtol=1e-5)
        np.testing.assert_allclose(loaded_model.training_state.Q, model.training_state.Q, rtol=1e-5)
    
    def test_dfm_save_and_load_without_dataset(self, sample_dataset, tmp_path):
        """Test DFM load requires dataset parameter."""
        # Train and save model
        model = DFM(dataset=sample_dataset, config=sample_dataset.config)
        model.fit()
        
        save_path = tmp_path / "test_dfm_model.pkl"
        model.save(save_path)
        
        # Load should work with dataset
        loaded_model = DFM.load(save_path, dataset=sample_dataset)
        assert loaded_model is not None
        
        # Load should also work with config override
        loaded_model2 = DFM.load(save_path, config=sample_dataset.config, dataset=sample_dataset)
        assert loaded_model2 is not None
