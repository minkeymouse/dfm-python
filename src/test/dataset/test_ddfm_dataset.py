"""Tests for dataset.ddfm_dataset module."""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from dfm_python.dataset.ddfm_dataset import DDFMDataset, AutoencoderDataset
from dfm_python.config.constants import DEFAULT_TORCH_DTYPE


class TestDDFMDataset:
    """Test suite for DDFMDataset."""
    
    def test_ddfm_dataset_initialization(self):
        """Test DDFMDataset can be initialized."""
        data = pd.DataFrame(np.random.randn(10, 5))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        assert dataset is not None
        assert dataset.data is not None
        assert len(dataset.target_series) == 5
    
    def test_ddfm_dataset_with_scaler(self):
        """Test DDFMDataset scales target series when scaler provided."""
        data = pd.DataFrame(np.random.randn(10, 5))
        scaler = StandardScaler()
        scaler.fit(data.values)
        
        dataset = DDFMDataset(data=data, time_idx='index', scaler=scaler)
        
        assert dataset.scaler is not None
        # Data should be scaled
        assert np.allclose(dataset.y.mean(axis=0), 0, atol=0.1)
        assert np.allclose(dataset.y.std(axis=0), 1, atol=0.1)
    
    def test_ddfm_dataset_with_covariates(self):
        """Test DDFMDataset handles covariates correctly."""
        data = pd.DataFrame(np.random.randn(10, 5), columns=['cov1', 'cov2', 'target1', 'target2', 'target3'])
        dataset = DDFMDataset(
            data=data,
            time_idx='index',
            covariates=['cov1', 'cov2']
        )
        
        assert len(dataset.covariates) == 2
        assert len(dataset.target_series) == 3
        assert dataset.X.shape[1] == 2  # Features
        assert dataset.y.shape[1] == 3  # Targets
    
    def test_ddfm_dataset_all_columns_are_targets(self):
        """Test DDFMDataset when all columns are targets."""
        data = pd.DataFrame(np.random.randn(10, 5))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        assert dataset.all_columns_are_targets
        assert dataset.X.shape[1] == 0
        assert dataset.y.shape[1] == 5
    
    def test_ddfm_dataset_target_indices(self):
        """Test DDFMDataset target_indices property."""
        data = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
        dataset = DDFMDataset(data=data, time_idx='index', covariates=['a', 'b'])
        
        target_indices = dataset.target_indices
        assert len(target_indices) == 3
        assert all(idx in [0, 1, 2, 3, 4] for idx in target_indices)
    
    def test_ddfm_dataset_from_dataset(self):
        """Test DDFMDataset.from_dataset creates new dataset with same config."""
        data1 = pd.DataFrame(np.random.randn(10, 5))
        scaler = StandardScaler()
        scaler.fit(data1.values)
        dataset1 = DDFMDataset(data=data1, time_idx='index', scaler=scaler, covariates=['col0'])
        
        data2 = pd.DataFrame(np.random.randn(5, 5))
        dataset2 = DDFMDataset.from_dataset(data2, dataset1)
        
        assert dataset2.time_idx == dataset1.time_idx
        assert dataset2.covariates == dataset1.covariates
        assert dataset2.scaler is dataset1.scaler
    
    def test_ddfm_dataset_create_autoencoder_dataset(self):
        """Test DDFMDataset.create_autoencoder_dataset creates AutoencoderDataset."""
        data = pd.DataFrame(np.random.randn(10, 5))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.randn(10, 2, device=device) if not dataset.all_columns_are_targets else None
        y_tmp = torch.randn(10, 5, device=device)
        y_actual = torch.randn(10, 5, device=device)
        eps_draw = torch.randn(10, 5, device=device)
        
        ae_dataset = dataset.create_autoencoder_dataset(X, y_tmp, y_actual, eps_draw)
        
        assert isinstance(ae_dataset, AutoencoderDataset)
        assert ae_dataset.y_corrupted.shape == (10, 5)
        assert ae_dataset.y_clean.shape == (10, 5)
    
    def test_ddfm_dataset_create_pretrain_dataset(self):
        """Test DDFMDataset.create_pretrain_dataset creates pretrain dataset."""
        data = pd.DataFrame(np.random.randn(10, 5))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        ae_dataset = dataset.create_pretrain_dataset(data)
        
        assert isinstance(ae_dataset, AutoencoderDataset)
        # For pretrain, y_corrupted should equal y_clean
        assert torch.allclose(ae_dataset.y_corrupted, ae_dataset.y_clean)
    
    def test_ddfm_dataset_create_autoencoder_datasets_list(self):
        """Test DDFMDataset.create_autoencoder_datasets_list creates multiple datasets."""
        data = pd.DataFrame(np.random.randn(10, 5))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        n_mc_samples = 3
        mu_eps = np.zeros(5)
        std_eps = np.ones(5)
        rng = np.random.RandomState(42)
        
        datasets = dataset.create_autoencoder_datasets_list(
            n_mc_samples=n_mc_samples,
            mu_eps=mu_eps,
            std_eps=std_eps,
            X=pd.DataFrame() if dataset.all_columns_are_targets else pd.DataFrame(np.random.randn(10, 2)),
            y_tmp=data,
            y_actual=data.values,
            rng=rng
        )
        
        assert len(datasets) == n_mc_samples
        assert all(isinstance(ds, AutoencoderDataset) for ds in datasets)


class TestAutoencoderDataset:
    """Test suite for AutoencoderDataset."""
    
    def test_autoencoder_dataset_initialization(self):
        """Test AutoencoderDataset can be initialized."""
        T, num_targets = 10, 5
        y_corrupted = torch.randn(T, num_targets)
        y_clean = torch.randn(T, num_targets)
        
        dataset = AutoencoderDataset(X=None, y_corrupted=y_corrupted, y_clean=y_clean)
        
        assert dataset.X is None
        assert dataset.y_corrupted.shape == (T, num_targets)
        assert dataset.y_clean.shape == (T, num_targets)
        assert dataset.full_input.shape == (T, num_targets)
    
    def test_autoencoder_dataset_with_features(self):
        """Test AutoencoderDataset with features."""
        T, num_features, num_targets = 10, 3, 5
        X = torch.randn(T, num_features)
        y_corrupted = torch.randn(T, num_targets)
        y_clean = torch.randn(T, num_targets)
        
        dataset = AutoencoderDataset(X=X, y_corrupted=y_corrupted, y_clean=y_clean)
        
        assert dataset.X.shape == (T, num_features)
        assert dataset.full_input.shape == (T, num_features + num_targets)
    
    def test_autoencoder_dataset_len(self):
        """Test AutoencoderDataset __len__ method."""
        T = 10
        y_corrupted = torch.randn(T, 5)
        y_clean = torch.randn(T, 5)
        
        dataset = AutoencoderDataset(X=None, y_corrupted=y_corrupted, y_clean=y_clean)
        
        assert len(dataset) == T
