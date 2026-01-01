"""Tests for decoder.mlp module."""

import pytest
import torch
from dfm_python.decoder.mlp import MLPDecoder


class TestMLPDecoder:
    """Test suite for MLPDecoder."""
    
    def test_mlp_decoder_initialization(self):
        """Test MLPDecoder can be initialized."""
        input_dim = 3
        hidden_dims = [64, 32]
        output_dim = 10
        decoder = MLPDecoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        assert decoder is not None
    
    def test_mlp_decoder_forward(self):
        """Test MLPDecoder forward pass."""
        decoder = MLPDecoder(input_dim=3, hidden_dims=[64, 32], output_dim=10)
        x = torch.randn(5, 3)
        output = decoder(x)
        assert output.shape == (5, 10)

