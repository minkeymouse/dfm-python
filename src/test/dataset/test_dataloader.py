"""Tests for dataset.dataloader module."""

import pytest
from torch.utils.data import DataLoader
from dfm_python.dataset.dataset import DDFMDataset
import numpy as np


class TestDataLoader:
    """Test suite for DataLoader."""
    
    def test_dataloader_initialization(self, sample_data):
        """Test DataLoader can be initialized with DDFMDataset."""
        data_array = sample_data.values.astype(np.float32)
        dataset = DDFMDataset(data_array, window_size=10)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        assert dataloader is not None
        assert dataloader.batch_size == 4
        assert dataloader.dataset == dataset

