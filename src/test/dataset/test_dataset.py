"""Tests for dataset.dataset module."""

import pytest
from dfm_python.dataset.dataset import DDFMDataset, KDFMDataset


class TestDDFMDataset:
    """Test suite for DDFMDataset."""
    
    def test_ddfm_dataset_initialization(self, sample_data):
        """Test DDFMDataset can be initialized."""
        import numpy as np
        data_array = sample_data.values.astype(np.float32)
        dataset = DDFMDataset(data_array, window_size=10)
        assert dataset is not None
        assert dataset.window_size == 10
        assert dataset.T == len(sample_data)
        assert dataset.N == len(sample_data.columns)
    
    def test_ddfm_dataset_len(self, sample_data):
        """Test DDFMDataset length."""
        import numpy as np
        data_array = sample_data.values.astype(np.float32)
        dataset = DDFMDataset(data_array, window_size=10, stride=1)
        expected_len = (len(sample_data) - 10) // 1 + 1
        assert len(dataset) == expected_len
        assert len(dataset) == dataset.n_samples
    
    def test_ddfm_dataset_getitem(self, sample_data):
        """Test DDFMDataset indexing."""
        import numpy as np
        import torch
        data_array = sample_data.values.astype(np.float32)
        dataset = DDFMDataset(data_array, window_size=10, stride=1)
        x, target = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert x.shape == (10, len(sample_data.columns))
        assert target.shape == x.shape


class TestKDFMDataset:
    """Test suite for KDFMDataset."""
    
    def test_kdfm_dataset_initialization(self, sample_data):
        """Test KDFMDataset can be initialized."""
        import numpy as np
        data_array = sample_data.values.astype(np.float32)
        dataset = KDFMDataset(data_array)
        assert dataset is not None
        assert dataset.T == len(sample_data)
        assert dataset.N == len(sample_data.columns)

