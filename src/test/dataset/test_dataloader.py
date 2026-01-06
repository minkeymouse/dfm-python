"""Tests for dataset.dataloader module."""

import pytest
from torch.utils.data import DataLoader
from dfm_python.dataset.ddfm_dataset import DDFMDataset


class TestDataLoader:
    """Test suite for DataLoader."""
    
    def test_dataloader_initialization(self, sample_data):
        """Test DataLoader can be initialized with DDFMDataset."""
        # Use current DDFMDataset API: data (DataFrame), time_idx, target_series, target_scaler
        target_series = list(sample_data.columns)
        dataset = DDFMDataset(
            data=sample_data,
            time_idx=sample_data.index.name or 'index',
            target_series=target_series,
            target_scaler=None
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        assert dataloader is not None
        assert dataloader.batch_size == 4
        assert dataloader.dataset == dataset

