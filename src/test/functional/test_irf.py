"""Tests for functional.irf module."""

import pytest
import numpy as np
import torch
from dfm_python.functional.irf import compute_irf
from dfm_python.utils.errors import DataValidationError, ConfigurationError
from dfm_python.config.constants import DEFAULT_AR_COEF


class TestIRF:
    """Test suite for Impulse Response Functions."""
    
    def test_compute_irf_shape_validation(self):
        """Test IRF computation validates tensor shapes correctly."""
        K = 3  # Number of variables
        p = 1  # AR order
        q = 0  # MA order (no MA stage)
        horizon = 5
        
        # Create valid tensors
        A_ar = torch.eye(p * K, dtype=torch.float32)
        A_ma = torch.eye(K, dtype=torch.float32)  # Identity for q=0
        B = torch.randn(p * K, K, dtype=torch.float32)
        C = torch.randn(K, p * K, dtype=torch.float32)
        B_prime = torch.eye(K, dtype=torch.float32)  # Identity for q=0
        C_prime = torch.eye(K, dtype=torch.float32)  # Identity for q=0
        S = torch.randn(K, K, dtype=torch.float32)
        
        # Test with non-square A_ar (should raise DataValidationError)
        A_ar_wrong = torch.randn(p * K, p * K + 1, dtype=torch.float32)
        with pytest.raises(DataValidationError, match="A_ar.*must be square"):
            compute_irf(A_ar_wrong, A_ma, B, C, B_prime, C_prime, S, horizon)
        
        # Test with non-square A_ma (should raise DataValidationError)
        A_ma_wrong = torch.randn(K, K + 1, dtype=torch.float32)
        with pytest.raises(DataValidationError, match="A_ma.*must be square"):
            compute_irf(A_ar, A_ma_wrong, B, C, B_prime, C_prime, S, horizon)
        
        # Test with incompatible B shape (should raise DataValidationError)
        B_wrong = torch.randn(p * K + 1, K, dtype=torch.float32)
        with pytest.raises(DataValidationError, match="B.*shape.*incompatible"):
            compute_irf(A_ar, A_ma, B_wrong, C, B_prime, C_prime, S, horizon)
        
        # Test with incompatible C shape (should raise DataValidationError)
        # Note: C shape validation happens after B validation, so we need compatible B
        # C_wrong has shape (K+1, p*K) but should be (K, p*K)
        # K is determined from C.shape[0], so wrong C will cause K mismatch
        # Actually, K is determined from C.shape[0], so we need to test differently
        # Let's test with C that has wrong second dimension instead
        C_wrong_dim2 = torch.randn(K, p * K + 1, dtype=torch.float32)
        with pytest.raises(DataValidationError, match="C.*shape.*incompatible"):
            compute_irf(A_ar, A_ma, B, C_wrong_dim2, B_prime, C_prime, S, horizon)
        
        # Test with incompatible S shape (should raise DataValidationError)
        S_wrong = torch.randn(K, K + 1, dtype=torch.float32)
        with pytest.raises(DataValidationError, match="S.*must be square"):
            compute_irf(A_ar, A_ma, B, C, B_prime, C_prime, S_wrong, horizon)
    
    def test_compute_irf_horizon_validation(self):
        """Test IRF computation validates horizon parameter."""
        K = 3
        p = 1
        A_ar = torch.eye(p * K, dtype=torch.float32)
        A_ma = torch.eye(K, dtype=torch.float32)
        B = torch.randn(p * K, K, dtype=torch.float32)
        C = torch.randn(K, p * K, dtype=torch.float32)
        B_prime = torch.eye(K, dtype=torch.float32)
        C_prime = torch.eye(K, dtype=torch.float32)
        S = torch.randn(K, K, dtype=torch.float32)
        
        # Test with invalid horizon (should be validated by validate_irf_horizon)
        # Note: validate_irf_horizon raises ConfigurationError
        with pytest.raises(ConfigurationError, match="IRF horizon must be >= 1"):
            compute_irf(A_ar, A_ma, B, C, B_prime, C_prime, S, horizon=0)
        
        with pytest.raises(ConfigurationError, match="IRF horizon must be >= 1"):
            compute_irf(A_ar, A_ma, B, C, B_prime, C_prime, S, horizon=-1)
    
    def test_compute_irf_basic(self):
        """Test basic IRF computation with valid inputs."""
        K = 2  # Small number for testing
        p = 1
        q = 0
        horizon = 3
        
        # Create valid tensors
        A_ar = torch.eye(p * K, dtype=torch.float32) * DEFAULT_AR_COEF  # Stable eigenvalues
        A_ma = torch.eye(K, dtype=torch.float32)
        B = torch.randn(p * K, K, dtype=torch.float32)
        C = torch.randn(K, p * K, dtype=torch.float32)
        B_prime = torch.eye(K, dtype=torch.float32)
        C_prime = torch.eye(K, dtype=torch.float32)
        S = torch.randn(K, K, dtype=torch.float32)
        
        # Should not raise
        irf_reduced, irf_structural = compute_irf(
            A_ar, A_ma, B, C, B_prime, C_prime, S, horizon=horizon, structural=True
        )
        
        # Verify output shapes
        assert irf_reduced.shape == (horizon, K, K)
        assert irf_structural is not None
        assert irf_structural.shape == (horizon, K, K)
        
        # Verify output types
        assert isinstance(irf_reduced, np.ndarray)
        assert isinstance(irf_structural, np.ndarray)
    
    def test_compute_irf_reduced_only(self):
        """Test IRF computation with structural=False."""
        K = 2
        p = 1
        horizon = 3
        
        A_ar = torch.eye(p * K, dtype=torch.float32) * DEFAULT_AR_COEF
        A_ma = torch.eye(K, dtype=torch.float32)
        B = torch.randn(p * K, K, dtype=torch.float32)
        C = torch.randn(K, p * K, dtype=torch.float32)
        B_prime = torch.eye(K, dtype=torch.float32)
        C_prime = torch.eye(K, dtype=torch.float32)
        S = torch.randn(K, K, dtype=torch.float32)
        
        irf_reduced, irf_structural = compute_irf(
            A_ar, A_ma, B, C, B_prime, C_prime, S, horizon=horizon, structural=False
        )
        
        # Verify reduced-form IRF exists
        assert irf_reduced.shape == (horizon, K, K)
        # Structural IRF should be None when structural=False
        assert irf_structural is None

