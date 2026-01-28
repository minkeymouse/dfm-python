"""Tests for dataset.ivdfm_dataset module."""

import pytest
import numpy as np
import pandas as pd
import torch
from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
from dfm_python.config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_IVDFM_SEQUENCE_LENGTH


class TestiVDFMDataset:
    """Test suite for iVDFMDataset."""
    
    def test_ivdfm_dataset_initialization_basic(self):
        """Test iVDFMDataset can be initialized with basic data."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        
        dataset = iVDFMDataset(data=data, sequence_length=10)
        
        assert dataset is not None
        assert dataset.data_dim == N
        assert dataset.context_dim == 1  # Default context_dim
        assert len(dataset) == T - 10 + 1
    
    def test_ivdfm_dataset_with_dataframe(self):
        """Test iVDFMDataset with DataFrame input."""
        np.random.seed(42)
        T, N = 50, 5
        data = pd.DataFrame(
            np.random.randn(T, N),
            columns=[f'series_{i}' for i in range(N)]
        )
        
        dataset = iVDFMDataset(data=data, sequence_length=10)
        
        assert dataset.data_dim == N
        assert len(dataset.target_series) == N
        assert len(dataset.context_columns) == 0
    
    def test_ivdfm_dataset_with_context_columns(self):
        """Test iVDFMDataset with context columns specified."""
        np.random.seed(42)
        T, N = 50, 5
        data = pd.DataFrame(
            np.random.randn(T, N),
            columns=['context1', 'context2', 'target1', 'target2', 'target3']
        )
        
        dataset = iVDFMDataset(
            data=data,
            sequence_length=10,
            context=['context1', 'context2']
        )
        
        assert dataset.data_dim == 3  # Only targets
        assert dataset.context_dim == 3  # time(1) + two context columns
        assert len(dataset.target_series) == 3
        assert len(dataset.context_columns) == 2
    
    def test_ivdfm_dataset_with_context_indices(self):
        """Test iVDFMDataset with context as array indices."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        
        dataset = iVDFMDataset(
            data=data,
            sequence_length=10,
            context=[0, 1]  # First two columns as context
        )
        
        assert dataset.data_dim == 3  # Remaining columns
        assert dataset.context_dim == 3  # time(1) + two context indices
    
    def test_ivdfm_dataset_context_type_time(self):
        """Test iVDFMDataset generates time-based context."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        
        # When context is None, time-based context is generated with context_dim
        dataset = iVDFMDataset(
            data=data,
            sequence_length=10,
            context=None,  # No context columns provided
            context_dim=2  # Generate time-based context with dimension 2
        )
        
        assert dataset.context_dim == 2
        context = dataset.context
        assert context.shape == (T, 2)
        # First column should be normalized time index
        assert np.allclose(context[0, 0], 0.0, atol=0.01)
        assert np.allclose(context[-1, 0], 1.0, atol=0.01)
    
    def test_ivdfm_dataset_context_type_regime(self):
        """Test iVDFMDataset with regime context from data columns."""
        np.random.seed(42)
        T, N = 50, 5
        # Create data with regime indicators as columns
        data = np.random.randn(T, N)
        # Add regime indicators (one-hot encoded) as additional columns
        regime = np.random.randint(0, 3, size=(T,))
        regime_onehot = np.eye(3)[regime]  # (T, 3)
        data_with_regime = np.hstack([data, regime_onehot])
        
        # Create DataFrame with regime columns
        df = pd.DataFrame(
            data_with_regime,
            columns=[f'target_{i}' for i in range(N)] + ['regime_0', 'regime_1', 'regime_2']
        )
        
        dataset = iVDFMDataset(
            data=df,
            sequence_length=10,
            context=['regime_0', 'regime_1', 'regime_2'],  # Extract regime from data
            context_dim=3
        )
        
        assert dataset.context_dim == 6  # time(3) + regime(3)
        context = dataset.context
        assert context.shape == (T, 6)
        # Regime part should be one-hot encoded (from data)
        regime_part = context[:, 3:]
        assert np.allclose(regime_part.sum(axis=1), 1.0)
    
    def test_ivdfm_dataset_context_custom(self):
        """Test iVDFMDataset with custom context from data columns."""
        np.random.seed(42)
        T, N = 50, 5
        context_data = np.random.randn(T, 3)
        
        # Create DataFrame with context columns
        df = pd.DataFrame(
            np.hstack([np.random.randn(T, N), context_data]),
            columns=[f'target_{i}' for i in range(N)] + ['context_0', 'context_1', 'context_2']
        )
        
        dataset = iVDFMDataset(
            data=df,
            sequence_length=10,
            context=['context_0', 'context_1', 'context_2']  # Extract context from data
        )
        
        assert dataset.context_dim == 4  # time(1) + 3 custom context cols
        np.testing.assert_array_almost_equal(dataset.context[:, 1:], context_data, decimal=5)
    
    def test_ivdfm_dataset_getitem(self):
        """Test iVDFMDataset __getitem__ returns correct sequences."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        sequence_length = 10
        
        dataset = iVDFMDataset(data=data, sequence_length=sequence_length)
        
        # Get first sequence
        y_seq, u_seq = dataset[0]
        
        assert isinstance(y_seq, torch.Tensor)
        assert isinstance(u_seq, torch.Tensor)
        assert y_seq.shape == (sequence_length, N)
        assert u_seq.shape == (sequence_length, dataset.context_dim)
        assert y_seq.dtype == DEFAULT_TORCH_DTYPE
        assert u_seq.dtype == DEFAULT_TORCH_DTYPE
    
    def test_ivdfm_dataset_len(self):
        """Test iVDFMDataset __len__ method."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        sequence_length = 10
        
        dataset = iVDFMDataset(data=data, sequence_length=sequence_length)
        
        expected_len = T - sequence_length + 1
        assert len(dataset) == expected_len
    
    def test_ivdfm_dataset_properties(self):
        """Test iVDFMDataset properties."""
        np.random.seed(42)
        T, N = 50, 5
        data = np.random.randn(T, N)
        
        dataset = iVDFMDataset(data=data, sequence_length=10, context_dim=2)
        
        assert dataset.data_dim == N
        assert dataset.context_dim == 2
        assert isinstance(dataset.context, np.ndarray)
        assert dataset.context.shape == (T, 2)
    
    def test_ivdfm_dataset_context_from_dataframe_columns(self):
        """Test extracting context from DataFrame columns."""
        np.random.seed(42)
        T = 50
        data = pd.DataFrame(
            np.random.randn(T, 4),
            columns=['target1', 'target2', 'context1', 'context2']
        )
        
        dataset = iVDFMDataset(
            data=data,
            sequence_length=10,
            context=['context1', 'context2']
        )
        
        assert dataset.data_dim == 2
        assert dataset.context_dim == 3  # time(1) + context(2)
        assert 'target1' in dataset.target_series
        assert 'target2' in dataset.target_series
        assert 'context1' in dataset.context_columns
        assert 'context2' in dataset.context_columns
    
    def test_ivdfm_dataset_context_custom_requires_data(self):
        """Test that providing context columns requires them to exist in data."""
        np.random.seed(42)
        T, N = 50, 5
        data = pd.DataFrame(
            np.random.randn(T, N),
            columns=[f'target_{i}' for i in range(N)]
        )
        
        # Providing context columns that don't exist should result in empty context
        # and fallback to time-based context generation
        dataset = iVDFMDataset(
            data=data,
            sequence_length=10,
            context=['nonexistent_col'],  # Column doesn't exist
            context_dim=1  # Will generate time-based context instead
        )
        
        # Should fallback to time-based context
        assert dataset.context_dim == 1
        assert len(dataset.context_columns) == 0  # No context columns found

    def test_ivdfm_dataset_with_covariates_excluded_from_targets(self):
        """Targets = variables - covariates - context - time_idx."""
        np.random.seed(42)
        T = 30
        df = pd.DataFrame(
            np.random.randn(T, 6),
            columns=["date", "y1", "y2", "x1", "x2", "aux1"]
        )
        df["date"] = pd.date_range("2020-01-01", periods=T, freq="D")

        dataset = iVDFMDataset(
            data=df,
            time_idx="date",
            covariates=["x1", "x2"],
            context=["aux1"],
            sequence_length=10,
            context_dim=1,
        )

        assert dataset.data_dim == 2
        assert set(dataset.target_series) == {"y1", "y2"}
        # Context (auxiliary u_t) = time + aux columns. Covariates are excluded from targets
        # but are NOT part of u_t.
        assert dataset.context_dim == 1 + 1  # time + aux
    
    def test_ivdfm_dataset_torch_tensor_input(self):
        """Test iVDFMDataset accepts torch.Tensor input."""
        np.random.seed(42)
        T, N = 50, 5
        data_tensor = torch.randn(T, N)
        
        dataset = iVDFMDataset(data=data_tensor, sequence_length=10)
        
        assert dataset.data_dim == N
        assert isinstance(dataset.data, np.ndarray)
    
    def test_ivdfm_dataset_minimum_sequence_length(self):
        """Test iVDFMDataset handles minimum sequence length."""
        np.random.seed(42)
        T, N = 20, 5
        data = np.random.randn(T, N)
        sequence_length = 15
        
        dataset = iVDFMDataset(data=data, sequence_length=sequence_length)
        
        assert len(dataset) == T - sequence_length + 1
        assert len(dataset) == 6
