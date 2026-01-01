"""Tests for functional.krylov module."""

import pytest
import numpy as np
from dfm_python.functional.krylov import krylov, krylov_sequential


class TestKrylov:
    """Test suite for Krylov subspace methods."""
    
    def test_krylov_function(self):
        """Test krylov function."""
        import torch
        L = 5
        N = 3
        A = torch.eye(N, dtype=torch.float32)
        b = torch.ones(N, dtype=torch.float32)
        
        # Test basic krylov computation
        result = krylov(L, A, b)
        assert result.shape == (N, L)
        # For identity matrix, all columns should be ones
        assert torch.allclose(result, torch.ones(N, L))
    
    def test_krylov_sequential(self):
        """Test krylov_sequential function."""
        import torch
        L = 4
        N = 2
        A = torch.eye(N, dtype=torch.float32)
        b = torch.tensor([1.0, 2.0], dtype=torch.float32)
        
        # Test basic sequential krylov computation
        result = krylov_sequential(L, A, b)
        assert result.shape == (N, L)
        # For identity matrix, all columns should be the same as b
        expected = b.unsqueeze(-1).repeat(1, L)
        assert torch.allclose(result, expected)

