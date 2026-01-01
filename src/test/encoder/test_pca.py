"""Tests for encoder.pca module."""

import pytest
import numpy as np
import torch
from dfm_python.encoder.pca import PCAEncoder
from dfm_python.utils.errors import ModelNotTrainedError


class TestPCAEncoder:
    """Test suite for PCAEncoder."""
    
    def test_pca_encoder_initialization(self):
        """Test PCAEncoder can be initialized."""
        encoder = PCAEncoder(n_components=2)
        assert encoder.n_components == 2
        assert encoder.eigenvectors is None
        assert encoder.eigenvalues is None
        assert encoder.cov_matrix is None
        assert encoder.mean_ is None
    
    def test_pca_encoder_initialization_with_block_idx(self):
        """Test PCAEncoder can be initialized with block_idx."""
        encoder = PCAEncoder(n_components=3, block_idx=0)
        assert encoder.n_components == 3
        assert encoder.block_idx == 0
    
    def test_pca_encoder_fit(self):
        """Test PCAEncoder fitting."""
        encoder = PCAEncoder(n_components=2)
        # Create sample data: 10 time steps, 5 variables
        X = np.random.randn(10, 5)
        encoder.fit(X)
        assert encoder.eigenvectors is not None
        assert encoder.eigenvalues is not None
        assert encoder.eigenvectors.shape == (5, 2)  # N x n_components
        assert encoder.eigenvalues.shape == (2,)  # n_components
        assert encoder.mean_ is not None
        assert encoder.mean_.shape == (1, 5)  # 1 x N
    
    def test_pca_encoder_fit_with_cov_matrix(self):
        """Test PCAEncoder fitting with precomputed covariance matrix."""
        encoder = PCAEncoder(n_components=2)
        # Create sample data and compute covariance
        X = np.random.randn(10, 5)
        cov_matrix = np.cov(X.T)
        encoder.fit(X, cov_matrix=cov_matrix)
        assert encoder.eigenvectors is not None
        assert encoder.eigenvalues is not None
        assert encoder.cov_matrix is not None
        assert encoder.eigenvectors.shape == (5, 2)
        assert encoder.eigenvalues.shape == (2,)
    
    def test_pca_encoder_fit_with_torch_tensor(self):
        """Test PCAEncoder fitting accepts torch tensors."""
        encoder = PCAEncoder(n_components=2)
        X = torch.randn(10, 5)
        encoder.fit(X)
        assert encoder.eigenvectors is not None
        assert encoder.eigenvalues is not None
        assert isinstance(encoder.eigenvectors, np.ndarray)
        assert isinstance(encoder.eigenvalues, np.ndarray)
    
    def test_pca_encoder_encode(self):
        """Test PCAEncoder transformation."""
        encoder = PCAEncoder(n_components=2)
        # Fit on training data
        X_train = np.random.randn(10, 5)
        encoder.fit(X_train)
        # Encode new data
        X_test = np.random.randn(8, 5)
        factors = encoder.encode(X_test)
        assert factors.shape == (8, 2)  # T x n_components
        assert isinstance(factors, np.ndarray)
    
    def test_pca_encoder_encode_with_torch_tensor(self):
        """Test PCAEncoder encode accepts torch tensors."""
        encoder = PCAEncoder(n_components=2)
        X_train = np.random.randn(10, 5)
        encoder.fit(X_train)
        X_test = torch.randn(8, 5)
        factors = encoder.encode(X_test)
        assert factors.shape == (8, 2)
        assert isinstance(factors, np.ndarray)  # Always returns NumPy
    
    def test_pca_encoder_encode_not_fitted(self):
        """Test PCAEncoder encode raises error when not fitted."""
        encoder = PCAEncoder(n_components=2)
        X = np.random.randn(10, 5)
        with pytest.raises(ModelNotTrainedError, match="must be fitted before encoding"):
            encoder.encode(X)
    
    def test_pca_encoder_fit_encode(self):
        """Test PCAEncoder fit_encode convenience method."""
        encoder = PCAEncoder(n_components=2)
        X = np.random.randn(10, 5)
        factors = encoder.fit_encode(X)
        assert factors.shape == (10, 2)
        assert encoder.eigenvectors is not None  # Should be fitted

