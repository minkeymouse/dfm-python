"""Tests for dataset module."""

import pytest
import numpy as np
import pandas as pd
import torch
from dfm_python.dataset.ddfm_dataset import DDFMDataset, AutoencoderDataset
from dfm_python.config import DFMConfig


class TestDDFMDataset:
    """Test suite for DDFMDataset."""
    
    def test_ddfm_dataset_initialization_with_config(self, sample_data, sample_config):
        """Test DDFMDataset can be initialized with required parameters."""
        # Extract parameters from config or use defaults
        target_series = list(sample_data.columns)  # Use all columns as targets
        dataset = DDFMDataset(
            data=sample_data,
            time_idx=sample_data.index.name or 'time',
            target_series=target_series,
            target_scaler=None  # No scaling for test
        )
        assert dataset is not None
        assert hasattr(dataset, 'data')
        assert dataset.data is not None
        assert hasattr(dataset, 'y')
        assert dataset.y is not None
    
    def test_ddfm_dataset_get_processed_data(self, sample_data, sample_config):
        """Test DDFMDataset data attributes are accessible."""
        target_series = list(sample_data.columns)
        dataset = DDFMDataset(
            data=sample_data,
            time_idx=sample_data.index.name or 'time',
            target_series=target_series,
            target_scaler=None
        )
        # DDFMDataset stores processed data as DataFrame in 'data' and numpy array in 'y'
        assert hasattr(dataset, 'data')
        assert isinstance(dataset.data, pd.DataFrame)
        assert dataset.data.shape[1] == len(sample_data.columns)
        assert hasattr(dataset, 'y')
        assert isinstance(dataset.y, np.ndarray)
    
class TestAutoencoderDataset:
    """Test suite for AutoencoderDataset."""
    
    def test_autoencoder_dataset_initialization(self):
        """Test AutoencoderDataset can be initialized."""
        T, N_input, N = 10, 8, 5
        x_corrupted = torch.randn(T, N_input)
        y_clean = torch.randn(T, N)
        
        # AutoencoderDataset takes (X, y_corrupted, y_clean) - no mask parameter
        dataset = AutoencoderDataset(X=None, y_corrupted=x_corrupted, y_clean=y_clean)
        assert len(dataset) == T
    
    def test_autoencoder_dataset_getitem(self):
        """Test AutoencoderDataset indexing."""
        T, N_input, N = 10, 8, 5
        x_corrupted = torch.randn(T, N_input)
        y_clean = torch.randn(T, N)
        
        # AutoencoderDataset takes (X, y_corrupted, y_clean) - no mask parameter
        dataset = AutoencoderDataset(X=None, y_corrupted=x_corrupted, y_clean=y_clean)
        full_input, y = dataset[0]
        assert full_input.shape == (N_input,)
        assert y.shape == (N,)
