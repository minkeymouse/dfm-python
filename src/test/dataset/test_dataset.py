"""Tests for dataset module."""

import pytest
import numpy as np
import torch
from dfm_python.dataset.ddfm_dataset import DDFMDataset, AutoencoderDataset
from dfm_python.dataset.kdfm_dataset import KDFMDataset
from dfm_python.config import DFMConfig


class TestDDFMDataset:
    """Test suite for DDFMDataset."""
    
    def test_ddfm_dataset_initialization_with_config(self, sample_data, sample_config):
        """Test DDFMDataset can be initialized with config."""
        dataset = DDFMDataset(config=sample_config, data=sample_data)
        assert dataset is not None
        assert hasattr(dataset, 'data_processed')
        assert dataset.data_processed is not None
    
    def test_ddfm_dataset_get_processed_data(self, sample_data, sample_config):
        """Test DDFMDataset get_processed_data returns tensor."""
        dataset = DDFMDataset(config=sample_config, data=sample_data)
        data = dataset.get_processed_data()
        assert isinstance(data, torch.Tensor)
        assert data.shape[1] == len(sample_data.columns)
    
    def test_ddfm_dataset_build_inputs_no_lags(self, sample_data, sample_config):
        """Test build_inputs with lags_input=0."""
        dataset = DDFMDataset(config=sample_config, data=sample_data)
        data_tensor = dataset.get_processed_data()
        inputs = dataset.build_inputs(data_tensor, lags_input=0)
        assert inputs.shape == data_tensor.shape
    
    def test_ddfm_dataset_build_inputs_with_lags(self, sample_data, sample_config):
        """Test build_inputs with lags_input=1."""
        dataset = DDFMDataset(config=sample_config, data=sample_data)
        data_tensor = dataset.get_processed_data()
        inputs = dataset.build_inputs(data_tensor, lags_input=1)
        T, N = data_tensor.shape
        expected_shape = (T - 1, N * 2)  # [y_t, y_{t-1}]
        assert inputs.shape == expected_shape


class TestAutoencoderDataset:
    """Test suite for AutoencoderDataset."""
    
    def test_autoencoder_dataset_initialization(self):
        """Test AutoencoderDataset can be initialized."""
        T, N_input, N = 10, 8, 5
        x_corrupted = torch.randn(T, N_input)
        y_clean = torch.randn(T, N)
        mask = torch.ones(T, N, dtype=torch.bool)
        
        dataset = AutoencoderDataset(x_corrupted, y_clean, mask)
        assert len(dataset) == T
    
    def test_autoencoder_dataset_getitem(self):
        """Test AutoencoderDataset indexing."""
        T, N_input, N = 10, 8, 5
        x_corrupted = torch.randn(T, N_input)
        y_clean = torch.randn(T, N)
        mask = torch.ones(T, N, dtype=torch.bool)
        
        dataset = AutoencoderDataset(x_corrupted, y_clean, mask)
        x, y, m = dataset[0]
        assert x.shape == (N_input,)
        assert y.shape == (N,)
        assert m.shape == (N,)


class TestKDFMDataset:
    """Test suite for KDFMDataset."""
    
    def test_kdfm_dataset_initialization_with_config(self, sample_data, sample_config):
        """Test KDFMDataset can be initialized with config."""
        dataset = KDFMDataset(config=sample_config, data=sample_data)
        assert dataset is not None
        assert hasattr(dataset, 'data_processed')
    
    def test_kdfm_dataset_get_processed_data(self, sample_data, sample_config):
        """Test KDFMDataset get_processed_data returns tensor."""
        dataset = KDFMDataset(config=sample_config, data=sample_data)
        data = dataset.get_processed_data()
        assert isinstance(data, torch.Tensor)
