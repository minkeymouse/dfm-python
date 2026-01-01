"""Tests for ssm.companion module."""

import pytest
import numpy as np
import torch
from dfm_python.ssm.companion import CompanionSSM, MACompanionSSM, CompanionSSMBase
from dfm_python.utils.errors import ConfigurationError, NumericalError


class TestCompanionSSM:
    """Test suite for companion SSM."""
    
    def test_companion_ssm_initialization(self):
        """Test CompanionSSM can be initialized."""
        # Test basic initialization with default parameters
        ssm = CompanionSSM(n_vars=3, lag_order=1)
        assert ssm is not None
        assert ssm.n_vars == 3
        assert ssm.order == 1
        assert ssm.latent_dim == 3  # order * n_vars = 1 * 3
        assert ssm.n_kernels == 1
        assert ssm.kernel_init == 'normal'
        assert ssm.norm_order == 1
        
        # Test initialization with custom parameters
        ssm2 = CompanionSSM(
            n_vars=5,
            lag_order=2,
            n_kernels=2,
            kernel_init='xavier',
            norm_order=2
        )
        assert ssm2.n_vars == 5
        assert ssm2.order == 2
        assert ssm2.latent_dim == 10  # order * n_vars = 2 * 5
        assert ssm2.n_kernels == 2
        assert ssm2.kernel_init == 'xavier'
        assert ssm2.norm_order == 2
        
        # Test that parameters are registered
        assert hasattr(ssm, 'a')  # VAR coefficient parameter
        assert hasattr(ssm, 'B')  # B matrix
        assert hasattr(ssm, 'C')  # C matrix
        assert hasattr(ssm, 'shift_matrix')  # Shift matrix buffer
    
    def test_companion_matrix_properties(self):
        """Test companion matrix properties."""
        # Test companion matrix shape
        ssm = CompanionSSM(n_vars=3, lag_order=2)
        A = ssm.get_companion_matrix()
        assert A is not None
        assert isinstance(A, torch.Tensor)
        assert A.shape == (1, 6, 6)  # (n_kernels=1, latent_dim=6, latent_dim=6)
        
        # Test companion matrix with multiple kernels
        ssm2 = CompanionSSM(n_vars=2, lag_order=1, n_kernels=3)
        A2 = ssm2.get_companion_matrix()
        assert A2.shape == (3, 2, 2)  # (n_kernels=3, latent_dim=2, latent_dim=2)
        
        # Test stability check
        is_stable, max_eig = ssm.check_stability()
        assert isinstance(is_stable, bool)
        assert isinstance(max_eig, float)
        assert max_eig >= 0.0  # Eigenvalue magnitude should be non-negative
        
        # Test stability check with custom threshold
        is_stable2, max_eig2 = ssm.check_stability(threshold=0.5)
        assert isinstance(is_stable2, bool)
        assert isinstance(max_eig2, float)
        
        # Test that stability check raises error for invalid threshold
        with pytest.raises(ConfigurationError, match="threshold must be > 0"):
            ssm.check_stability(threshold=0.0)
        with pytest.raises(ConfigurationError, match="threshold must be > 0"):
            ssm.check_stability(threshold=-1.0)
        
        # Test extract_coefficients returns correct shape
        coeffs = ssm.extract_coefficients()
        assert coeffs.shape == (2, 3, 3)  # (order=2, n_vars=3, n_vars=3)
        assert isinstance(coeffs, torch.Tensor)
    
    def test_create_identity_block(self):
        """Test _create_identity_block helper method."""
        ssm = CompanionSSM(n_vars=3, lag_order=1)
        identity_block = ssm._create_identity_block()
        
        # Check shape: (1, n_vars, n_vars) for broadcasting
        assert identity_block.shape == (1, 3, 3)
        assert isinstance(identity_block, torch.Tensor)
        
        # Check that it's an identity matrix
        identity_2d = identity_block.squeeze(0)  # Remove batch dimension
        expected = torch.eye(3)
        torch.testing.assert_close(identity_2d, expected)
        
        # Test with different n_vars
        ssm2 = CompanionSSM(n_vars=5, lag_order=1)
        identity_block2 = ssm2._create_identity_block()
        assert identity_block2.shape == (1, 5, 5)
        identity_2d2 = identity_block2.squeeze(0)
        expected2 = torch.eye(5)
        torch.testing.assert_close(identity_2d2, expected2)

