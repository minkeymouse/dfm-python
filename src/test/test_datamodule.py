"""Tests for DataModule implementations (DFMDataModule and DDFMDataModule).

Tests cover:
- DFMDataModule: Linear DFM data loading with full sequences
- DDFMDataModule: Deep DDFM data loading with windowed sequences
- Target series handling (not preprocessed)
- Preprocessed data handling
- Pipeline statistics extraction
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional

from dfm_python import DFMDataModule, DDFMDataModule
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.data import DFMDataset, DDFMDataset
from dfm_python.utils.time import TimeIndex, parse_timestamp
from test_helpers import (
    get_test_data_path,
    get_test_config_path,
    load_sample_data_from_csv,
)


def create_simple_config() -> DFMConfig:
    """Create a simple DFM config for testing."""
    return DFMConfig(
        series=[
            SeriesConfig(series_id='series_0', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            SeriesConfig(series_id='series_1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            SeriesConfig(series_id='series_2', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            SeriesConfig(series_id='series_3', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
            SeriesConfig(series_id='series_4', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME]),
        ],
        blocks={DEFAULT_BLOCK_NAME: {'factors': 2, 'ar_lag': 1, 'clock': 'm'}}
    )


class TestDFMDataModule:
    """Test DFMDataModule for linear DFM."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple DFM config for testing."""
        return create_simple_config()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        T, N = 100, 5
        data = np.random.randn(T, N)
        return pd.DataFrame(data, columns=pd.Index([f'series_{i}' for i in range(N)]))
    
    def test_dfm_datamodule_initialization(self, simple_config, sample_data):
        """Test DFMDataModule initialization."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data
        )
        assert dm.config is not None
        assert dm.data is not None
        assert dm.pipeline is None  # Default: passthrough
    
    def test_dfm_datamodule_setup(self, simple_config, sample_data):
        """Test DFMDataModule setup."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data
        )
        dm.setup()
        
        assert dm.train_dataset is not None
        assert isinstance(dm.train_dataset, DFMDataset)
        assert dm.data_processed is not None
        assert isinstance(dm.data_processed, torch.Tensor)
        assert dm.Mx is not None
        assert dm.Wx is not None
    
    def test_dfm_datamodule_get_processed_data(self, simple_config, sample_data):
        """Test getting processed data."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data
        )
        dm.setup()
        
        data = dm.get_processed_data()
        assert isinstance(data, torch.Tensor)
        assert data.shape[0] == len(sample_data)
        assert data.shape[1] == len(sample_data.columns)
    
    def test_dfm_datamodule_get_std_params(self, simple_config, sample_data):
        """Test getting standardization parameters."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data
        )
        dm.setup()
        
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        assert len(Mx) == len(sample_data.columns)
        assert len(Wx) == len(sample_data.columns)
    
    def test_dfm_datamodule_val_split(self, simple_config, sample_data):
        """Test validation split."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data,
            val_split=0.2
        )
        dm.setup()
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 1  # DFMDataset always returns 1 (full sequence)
        assert len(dm.val_dataset) == 1
    
    def test_dfm_datamodule_dataloader(self, simple_config, sample_data):
        """Test DataLoader creation."""
        dm = DFMDataModule(
            config=simple_config,
            data=sample_data,
            batch_size=1
        )
        dm.setup()
        
        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 1
        
        # DFM uses full sequence, so batch should contain full data
        for batch in train_loader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape[0] == 1  # Single batch
            break


class TestDDFMDataModule:
    """Test DDFMDataModule for Deep DDFM."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple DDFM config for testing."""
        config = create_simple_config()
        # Convert to DDFMConfig if needed
        if isinstance(config, DFMConfig):
            # DDFMConfig doesn't use blocks, so we can use DFMConfig
            pass
        return config
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        T, N = 200, 5
        data = np.random.randn(T, N)
        return pd.DataFrame(data, columns=pd.Index([f'series_{i}' for i in range(N)]))
    
    def test_ddfm_datamodule_initialization(self, simple_config, sample_data):
        """Test DDFMDataModule initialization."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data
        )
        assert dm.config is not None
        assert dm.data is not None
        assert dm.target_series == []
        assert dm.target_scaler is None
        assert dm.preprocessed is False
        assert dm.window_size == 100
        assert dm.batch_size == 100
    
    def test_ddfm_datamodule_setup(self, simple_config, sample_data):
        """Test DDFMDataModule setup."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data
        )
        dm.setup()
        
        assert dm.train_dataset is not None
        assert isinstance(dm.train_dataset, DDFMDataset)
        assert dm.data_processed is not None
        assert isinstance(dm.data_processed, torch.Tensor)
        assert dm.Mx is not None
        assert dm.Wx is not None
    
    def test_ddfm_datamodule_with_target_series(self, simple_config, sample_data):
        """Test DDFMDataModule with target series (not preprocessed)."""
        # Add target series
        sample_data['target'] = np.random.randn(len(sample_data))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            target_series=['target']
        )
        dm.setup()
        
        # Target should not be preprocessed (Mx=0, Wx=1)
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        target_idx = list(sample_data.columns).index('target')
        assert Mx[target_idx] == 0.0  # Target not preprocessed
        assert Wx[target_idx] == 1.0  # Target not preprocessed
    
    def test_ddfm_datamodule_target_scaler_standard(self, simple_config, sample_data):
        """Test DDFMDataModule with target_scaler='standard'."""
        sample_data['target'] = np.random.randn(len(sample_data))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            target_series=['target'],
            target_scaler='standard'
        )
        dm.setup()
        
        # Target should be scaled with StandardScaler
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        target_idx = list(sample_data.columns).index('target')
        # Mx should be mean of target (not 0)
        # Wx should be std of target (not 1)
        assert Mx[target_idx] != 0.0  # Target was scaled
        assert Wx[target_idx] != 1.0  # Target was scaled
    
    def test_ddfm_datamodule_target_scaler_robust(self, simple_config, sample_data):
        """Test DDFMDataModule with target_scaler='robust'."""
        sample_data['target'] = np.random.randn(len(sample_data))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            target_series=['target'],
            target_scaler='robust'
        )
        dm.setup()
        
        # Target should be scaled with RobustScaler
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        target_idx = list(sample_data.columns).index('target')
        # RobustScaler uses median, so Mx might be different
        assert dm.target_scaler is not None
    
    def test_ddfm_datamodule_preprocessed_true(self, simple_config, sample_data):
        """Test DDFMDataModule with preprocessed=True."""
        # Simulate preprocessed data (mean=0, std=1)
        sample_data_preprocessed = (sample_data - sample_data.mean()) / sample_data.std()
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data_preprocessed,
            preprocessed=True
        )
        dm.setup()
        
        # Data should be used as-is (no additional preprocessing)
        assert dm.data_processed is not None
        assert dm.train_dataset is not None
    
    def test_ddfm_datamodule_preprocessed_with_pipeline(self, simple_config, sample_data):
        """Test DDFMDataModule with preprocessed=True and fitted pipeline."""
        from sklearn.preprocessing import StandardScaler
        from sktime.transformations.compose import TransformerPipeline
        
        # Create and fit pipeline
        pipeline = TransformerPipeline([
            ('scaler', StandardScaler())
        ])
        pipeline.fit(sample_data)
        
        # Preprocess data
        sample_data_preprocessed = pd.DataFrame(
            pipeline.transform(sample_data),
            columns=sample_data.columns,
            index=sample_data.index
        )
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data_preprocessed,
            pipeline=pipeline,  # Already fitted
            preprocessed=True
        )
        dm.setup()
        
        # Pipeline should not be fit again (already fitted)
        # Statistics should be extracted from pipeline
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
    
    def test_ddfm_datamodule_window_size(self, simple_config, sample_data):
        """Test DDFMDataModule with custom window_size."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            window_size=50,
            stride=1
        )
        dm.setup()
        
        # Dataset should use window_size=50
        assert dm.train_dataset is not None
        assert dm.train_dataset.window_size == 50
        assert len(dm.train_dataset) > 1  # Multiple windows
    
    def test_ddfm_datamodule_val_split(self, simple_config, sample_data):
        """Test DDFMDataModule validation split."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            val_split=0.2
        )
        dm.setup()
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
    
    def test_ddfm_datamodule_dataloader(self, simple_config, sample_data):
        """Test DDFMDataModule DataLoader creation."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            batch_size=32
        )
        dm.setup()
        
        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 32
        
        # DDFM uses windowed sequences, so batches should contain windows
        for batch in train_loader:
            x, target = batch
            assert isinstance(x, torch.Tensor)
            assert isinstance(target, torch.Tensor)
            assert x.shape == target.shape
            assert x.shape[0] <= 32  # Batch size
            break
    
    def test_ddfm_datamodule_get_raw_data(self, simple_config, sample_data):
        """Test getting raw data."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data
        )
        dm.setup()
        
        raw_data = dm.get_raw_data()
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) == len(sample_data)
        assert list(raw_data.columns) == list(sample_data.columns)
    
    def test_ddfm_datamodule_get_target_indices(self, simple_config, sample_data):
        """Test getting target indices."""
        sample_data['target1'] = np.random.randn(len(sample_data))
        sample_data['target2'] = np.random.randn(len(sample_data))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            target_series=['target1', 'target2']
        )
        dm.setup()
        
        target_indices = dm.get_target_indices()
        assert len(target_indices) == 2
        assert all(isinstance(idx, int) for idx in target_indices)
    
    def test_ddfm_datamodule_is_data_preprocessed(self, simple_config, sample_data):
        """Test is_data_preprocessed method."""
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            preprocessed=False
        )
        assert dm.is_data_preprocessed() is False
        
        dm2 = DDFMDataModule(
            config=simple_config,
            data=sample_data,
            preprocessed=True
        )
        assert dm2.is_data_preprocessed() is True


class TestDataModuleTargetHandling:
    """Test target series handling in DDFMDataModule."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple config."""
        return create_simple_config()
    
    @pytest.fixture
    def sample_data_with_target(self):
        """Create sample data with target."""
        np.random.seed(42)
        T, N = 200, 5
        data = np.random.randn(T, N)
        df = pd.DataFrame(data, columns=pd.Index([f'feature_{i}' for i in range(N)]))
        df['target'] = np.random.randn(T)
        return df
    
    def test_target_not_in_pipeline(self, simple_config, sample_data_with_target):
        """Test that target is not preprocessed by feature pipeline."""
        from sklearn.preprocessing import StandardScaler
        from sktime.transformations.compose import TransformerPipeline
        
        # Create pipeline
        pipeline = TransformerPipeline([
            ('scaler', StandardScaler())
        ])
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data_with_target,
            pipeline=pipeline,
            target_series=['target'],
            preprocessed=False
        )
        dm.setup()
        
        # Feature columns should be preprocessed
        # Target should remain raw
        processed_data = dm.get_processed_data()
        raw_data = dm.get_raw_data()
        
        # Target column index
        target_idx = list(sample_data_with_target.columns).index('target')
        
        # Target should not be scaled (check that it's close to raw)
        # Note: This is a simplified check - in practice, target might have different scale
        target_processed = processed_data[:, target_idx].numpy()
        target_raw_series = raw_data['target']
        target_raw = target_raw_series.values if hasattr(target_raw_series, 'values') else np.asarray(target_raw_series)
        
        # If target was preprocessed, mean would be ~0 and std would be ~1
        # Since target is not preprocessed, mean and std should match raw data
        target_raw_array = np.asarray(target_raw)
        target_raw_mean = float(np.mean(target_raw_array))
        target_raw_std = float(np.std(target_raw_array))
        assert np.abs(target_processed.mean() - target_raw_mean) < 0.1
        assert np.abs(target_processed.std() - target_raw_std) < 0.1
    
    def test_multiple_targets(self, simple_config, sample_data_with_target):
        """Test multiple target series."""
        sample_data_with_target['target2'] = np.random.randn(len(sample_data_with_target))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=sample_data_with_target,
            target_series=['target', 'target2']
        )
        dm.setup()
        
        target_indices = dm.get_target_indices()
        assert len(target_indices) == 2
        
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        # Both targets should have Mx=0, Wx=1 (not preprocessed)
        for idx in target_indices:
            assert Mx[idx] == 0.0
            assert Wx[idx] == 1.0


class TestDataModuleEdgeCases:
    """Test edge cases for DataModules."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple config."""
        return create_simple_config()
    
    def test_empty_target_series(self, simple_config):
        """Test with empty target_series list."""
        data = pd.DataFrame(np.random.randn(100, 5), columns=pd.Index([f'series_{i}' for i in range(5)]))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=data,
            target_series=[]  # Empty list
        )
        dm.setup()
        
        assert len(dm.get_target_indices()) == 0
        assert dm.target_Mx is not None
        assert len(dm.target_Mx) == 0
    
    def test_target_series_not_in_data(self, simple_config):
        """Test with target_series not in data columns."""
        data = pd.DataFrame(np.random.randn(100, 5), columns=pd.Index([f'series_{i}' for i in range(5)]))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=data,
            target_series=['nonexistent']  # Not in data
        )
        dm.setup()
        
        # Should handle gracefully
        assert len(dm.get_target_indices()) == 0
    
    def test_all_columns_are_targets(self, simple_config):
        """Test when all columns are targets."""
        data = pd.DataFrame(np.random.randn(100, 3), columns=pd.Index(['target1', 'target2', 'target3']))
        
        dm = DDFMDataModule(
            config=simple_config,
            data=data,
            target_series=['target1', 'target2', 'target3']
        )
        dm.setup()
        
        # All columns are targets, so no features
        # All should have Mx=0, Wx=1
        Mx, Wx = dm.get_std_params()
        assert Mx is not None
        assert Wx is not None
        assert np.allclose(Mx, 0.0)
        assert np.allclose(Wx, 1.0)
    
    def test_missing_data_handling(self, simple_config):
        """Test handling of missing data (NaN)."""
        data = pd.DataFrame(np.random.randn(100, 5), columns=pd.Index([f'series_{i}' for i in range(5)]))
        # Add some NaN values
        data.iloc[10:20, 0] = np.nan
        
        dm = DFMDataModule(
            config=simple_config,
            data=data
        )
        # Should not raise error (DFM can handle NaN)
        dm.setup()
        
        assert dm.data_processed is not None

