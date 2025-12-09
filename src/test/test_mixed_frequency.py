"""Tests for mixed-frequency functionality in DFM.

Tests verify that mixed_freq parameter correctly handles:
- Tent kernel usage when mixed_freq=True
- Unified frequency assumption when mixed_freq=False
- Error handling for unsupported frequency pairs
- Clock frequency overrides
"""

import pytest
import numpy as np
import torch
import pandas as pd
from datetime import datetime

from dfm_python.models import DFM
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.utils import (
    TENT_WEIGHTS_LOOKUP, 
    FREQUENCY_HIERARCHY,
    get_agg_structure,
    get_tent_weights
)
from dfm_python import DFMDataModule, DFMTrainer
from dfm_python.utils.time import TimeIndex, parse_timestamp


class TestMixedFrequencyConfig:
    """Test mixed frequency configuration and validation."""
    
    @pytest.fixture
    def monthly_quarterly_config(self):
        """Config with monthly and quarterly series."""
        return DFMConfig(
            series=[
                SeriesConfig(series_id='m1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='m2', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='q1', frequency='q', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}},
            clock='m'
        )
    
    @pytest.fixture
    def monthly_weekly_config(self):
        """Config with monthly and weekly series (clock='w')."""
        return DFMConfig(
            series=[
                SeriesConfig(series_id='w1', frequency='w', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='w2', frequency='w', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='m1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'w'}},
            clock='w'
        )
    
    def test_tent_weights_lookup_exists(self):
        """Test that TENT_WEIGHTS_LOOKUP contains expected pairs."""
        assert ('q', 'm') in TENT_WEIGHTS_LOOKUP
        assert ('m', 'w') in TENT_WEIGHTS_LOOKUP
        assert ('q', 'w') in TENT_WEIGHTS_LOOKUP
        
        # Verify weights are numpy arrays
        assert isinstance(TENT_WEIGHTS_LOOKUP[('q', 'm')], np.ndarray)
        assert len(TENT_WEIGHTS_LOOKUP[('q', 'm')]) == 5  # [1, 2, 3, 2, 1]
    
    def test_get_agg_structure_monthly_quarterly(self, monthly_quarterly_config):
        """Test get_agg_structure with monthly clock and quarterly series."""
        agg_structure = get_agg_structure(monthly_quarterly_config, clock='m')
        
        # Should have tent weights for quarterly
        assert 'q' in agg_structure['tent_weights']
        assert len(agg_structure['tent_weights']['q']) == 5
        
        # Should have structure for (q, m) pair
        assert ('q', 'm') in agg_structure['structures']
    
    def test_get_agg_structure_weekly_monthly(self, monthly_weekly_config):
        """Test get_agg_structure with weekly clock and monthly series."""
        agg_structure = get_agg_structure(monthly_weekly_config, clock='w')
        
        # Should have tent weights for monthly
        assert 'm' in agg_structure['tent_weights']
        assert len(agg_structure['tent_weights']['m']) == 5
        
        # Should have structure for (m, w) pair
        assert ('m', 'w') in agg_structure['structures']
    
    def test_get_tent_weights(self):
        """Test get_tent_weights function."""
        # Valid pair
        weights = get_tent_weights('q', 'm')
        assert weights is not None
        assert np.array_equal(weights, np.array([1, 2, 3, 2, 1]))
        
        # Invalid pair
        weights_invalid = get_tent_weights('d', 'm')
        assert weights_invalid is None


class TestMixedFrequencyDFM:
    """Test DFM with mixed_freq parameter."""
    
    @pytest.fixture
    def monthly_quarterly_config(self):
        """Config with monthly and quarterly series."""
        return DFMConfig(
            series=[
                SeriesConfig(series_id='m1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='m2', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='q1', frequency='q', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}},
            clock='m'
        )
    
    @pytest.fixture
    def monthly_quarterly_data(self):
        """Generate synthetic monthly and quarterly data."""
        np.random.seed(42)
        T = 60  # 5 years of monthly data
        
        # Monthly series (full data)
        m1 = np.random.randn(T).cumsum()
        m2 = np.random.randn(T).cumsum()
        
        # Quarterly series (every 3rd month, starting from month 2)
        q1 = np.full(T, np.nan)
        q1[2::3] = np.random.randn(len(q1[2::3])).cumsum()
        
        # Combine into array (T x N)
        X = np.column_stack([m1, m2, q1])
        
        # Create time index
        dates = pd.date_range('2020-01-01', periods=T, freq='ME')
        time_index = TimeIndex([parse_timestamp(d.strftime('%Y-%m-%d')) for d in dates])
        
        df = pd.DataFrame(X, columns=['m1', 'm2', 'q1'], index=dates)
        return df, time_index
    
    def test_dfm_mixed_freq_true(self, monthly_quarterly_config, monthly_quarterly_data):
        """Test DFM with mixed_freq=True."""
        df, time_index = monthly_quarterly_data
        
        model = DFM(
            config=monthly_quarterly_config,
            mixed_freq=True,
            max_iter=10,  # Short for testing
            threshold=1e-3
        )
        
        # Should not raise error
        assert model.mixed_freq is True
        
        # Initialize from data should work
        X_tensor = torch.tensor(df.values, dtype=torch.float32)
        model.initialize_from_data(X_tensor)
        
        # Check that mixed frequency parameters are set
        assert model._em_nQ == 1  # One quarterly series
        assert model._em_tent_weights_dict is not None
        assert 'q' in model._em_tent_weights_dict
    
    def test_dfm_mixed_freq_false(self, monthly_quarterly_config, monthly_quarterly_data):
        """Test DFM with mixed_freq=False (unified frequency)."""
        df, time_index = monthly_quarterly_data
        
        model = DFM(
            config=monthly_quarterly_config,
            mixed_freq=False,
            max_iter=10,
            threshold=1e-3
        )
        
        assert model.mixed_freq is False
        
        # Initialize from data should work
        X_tensor = torch.tensor(df.values, dtype=torch.float32)
        model.initialize_from_data(X_tensor)
        
        # Check that mixed frequency parameters are not set (unified frequency)
        assert model._em_nQ == 0
        assert model._em_tent_weights_dict is None or len(model._em_tent_weights_dict) == 0
    
    def test_dfm_mixed_freq_unsupported_pair(self):
        """Test that unsupported frequency pairs raise ValueError."""
        # Create config with unsupported frequency pair (e.g., annual -> weekly)
        # This pair is not in TENT_WEIGHTS_LOOKUP
        config = DFMConfig(
            series=[
                SeriesConfig(series_id='w1', frequency='w', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='a1', frequency='a', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'w'}},
            clock='w'
        )
        
        model = DFM(config=config, mixed_freq=True)
        
        # Create dummy data (weekly frequency, ~2 years)
        np.random.seed(42)
        T = 104  # ~2 years of weekly data
        w1 = np.random.randn(T).cumsum()
        a1 = np.full(T, np.nan)
        a1[51] = np.random.randn()  # One annual observation
        
        X = np.column_stack([w1, a1])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Should raise ValueError because 'a' -> 'w' is not in TENT_WEIGHTS_LOOKUP
        with pytest.raises(ValueError, match=r".*not in TENT_WEIGHTS_LOOKUP.*"):
            model.initialize_from_data(X_tensor)
    
    def test_dfm_weekly_clock_monthly_series(self):
        """Test DFM with clock='w' and monthly series."""
        config = DFMConfig(
            series=[
                SeriesConfig(series_id='w1', frequency='w', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='w2', frequency='w', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='m1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'w'}},
            clock='w'
        )
        
        # Generate weekly data (52 weeks per year, ~5 years)
        np.random.seed(42)
        T = 260  # ~5 years of weekly data
        
        w1 = np.random.randn(T).cumsum()
        w2 = np.random.randn(T).cumsum()
        
        # Monthly series (every ~4 weeks)
        m1 = np.full(T, np.nan)
        m1[3::4] = np.random.randn(len(m1[3::4])).cumsum()
        
        X = np.column_stack([w1, w2, m1])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        model = DFM(
            config=config,
            mixed_freq=True,
            max_iter=10,
            threshold=1e-3
        )
        
        # Should work with weekly clock and monthly series
        model.initialize_from_data(X_tensor)
        
        # Check that tent weights are set for monthly
        assert model._em_tent_weights_dict is not None
        assert 'm' in model._em_tent_weights_dict
        assert model._em_nQ == 1  # One monthly series (slower than weekly clock)


class TestMixedFrequencyTraining:
    """Test training DFM with mixed frequency data."""
    
    @pytest.fixture
    def simple_mixed_config(self):
        """Simple config for training tests."""
        return DFMConfig(
            series=[
                SeriesConfig(series_id='m1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
                SeriesConfig(series_id='q1', frequency='q', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            ],
            blocks={DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': 'm'}},
            clock='m'
        )
    
    @pytest.fixture
    def simple_mixed_data(self):
        """Simple synthetic mixed frequency data."""
        np.random.seed(42)
        T = 36  # 3 years of monthly data
        
        m1 = np.random.randn(T).cumsum()
        q1 = np.full(T, np.nan)
        q1[2::3] = np.random.randn(len(q1[2::3])).cumsum()
        
        X = np.column_stack([m1, q1])
        dates = pd.date_range('2020-01-01', periods=T, freq='ME')
        df = pd.DataFrame(X, columns=['m1', 'q1'], index=dates)
        
        return df
    
    def test_training_mixed_freq_true(self, simple_mixed_config, simple_mixed_data):
        """Test training with mixed_freq=True."""
        model = DFM(
            config=simple_mixed_config,
            mixed_freq=True,
            max_iter=5,  # Very short for testing
            threshold=1e-2
        )
        
        # Create data module
        dm = DFMDataModule(config=simple_mixed_config, data=simple_mixed_data)
        dm.setup()
        
        # Create trainer
        trainer = DFMTrainer(max_epochs=5)
        
        # Should train without error
        trainer.fit(model, dm)
        
        # Check that model was trained
        assert model.training_state is not None
        assert hasattr(model.training_state, 'converged')
    
    def test_training_mixed_freq_false(self, simple_mixed_config, simple_mixed_data):
        """Test training with mixed_freq=False."""
        model = DFM(
            config=simple_mixed_config,
            mixed_freq=False,
            max_iter=5,
            threshold=1e-2
        )
        
        # Create data module
        dm = DFMDataModule(config=simple_mixed_config, data=simple_mixed_data)
        dm.setup()
        
        # Create trainer
        trainer = DFMTrainer(max_epochs=5)
        
        # Should train without error (treats all as monthly)
        trainer.fit(model, dm)
        
        # Check that model was trained
        assert model.training_state is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

