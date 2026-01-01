"""Tests for ssm.structural module."""

import pytest
import torch
import numpy as np
from dfm_python.ssm.structural import StructuralIdentificationSSM
from dfm_python.utils.errors import ConfigurationError


class TestStructuralIdentificationSSM:
    """Test suite for StructuralIdentificationSSM."""
    
    def test_structural_identification_initialization(self):
        """Test StructuralIdentificationSSM can be initialized."""
        # Test basic initialization with default parameters
        struct_id = StructuralIdentificationSSM(n_vars=3)
        assert struct_id is not None
        assert struct_id.n_vars == 3
        assert struct_id.lag_order == 1
        assert struct_id.method == 'cholesky'
        assert struct_id.align_with_latent_state is True
        assert struct_id.shock_dim == 3  # lag_order * n_vars = 1 * 3
        
        # Test initialization with custom parameters
        struct_id2 = StructuralIdentificationSSM(
            n_vars=4,
            lag_order=2,
            method='full',
            align_with_latent_state=False
        )
        assert struct_id2.n_vars == 4
        assert struct_id2.lag_order == 2
        assert struct_id2.method == 'full'
        assert struct_id2.align_with_latent_state is False
        assert struct_id2.shock_dim == 4  # n_vars when not aligned
    
    def test_structural_identification(self):
        """Test structural identification."""
        # Test get_structural_matrix() returns valid matrix
        struct_id = StructuralIdentificationSSM(n_vars=3, method='cholesky')
        S = struct_id.get_structural_matrix()
        assert S is not None
        assert isinstance(S, torch.Tensor)
        assert S.shape == (3, 3)  # n_vars x n_vars for cholesky with align_with_latent_state=True
        
        # Test forward() method transforms residuals
        residuals = torch.randn(10, 3)  # (T, n_vars)
        structural_shocks = struct_id(residuals)
        assert structural_shocks is not None
        assert isinstance(structural_shocks, torch.Tensor)
        assert structural_shocks.shape[0] == 10  # T dimension preserved
        assert structural_shocks.shape[1] == 3  # shock_dim = lag_order * n_vars = 1 * 3
        
        # Test with full method
        struct_id_full = StructuralIdentificationSSM(n_vars=2, method='full')
        S_full = struct_id_full.get_structural_matrix()
        assert S_full.shape == (2, 2)
        
        # Test with align_with_latent_state=False
        struct_id_no_align = StructuralIdentificationSSM(
            n_vars=3,
            align_with_latent_state=False
        )
        residuals2 = torch.randn(5, 3)
        shocks2 = struct_id_no_align(residuals2)
        assert shocks2.shape == (5, 3)  # shock_dim = n_vars when not aligned

