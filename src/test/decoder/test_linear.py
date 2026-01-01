"""Tests for decoder.linear module."""

import pytest
import numpy as np
import torch
from dfm_python.decoder.linear import Decoder


class TestDecoder:
    """Test suite for linear Decoder."""
    
    def test_decoder_initialization(self):
        """Test Decoder can be initialized."""
        input_dim = 3
        output_dim = 10
        decoder = Decoder(input_dim=input_dim, output_dim=output_dim)
        assert decoder is not None
        assert decoder.decoder.in_features == input_dim
        assert decoder.decoder.out_features == output_dim
    
    def test_decoder_forward(self):
        """Test Decoder forward pass."""
        decoder = Decoder(input_dim=3, output_dim=10)
        x = torch.randn(5, 3)
        output = decoder(x)
        assert output.shape == (5, 10)
    
    def test_decoder_to_numpy(self):
        """Test extracting decoder weights and bias as numpy arrays."""
        decoder = Decoder(input_dim=3, output_dim=10)
        # Extract weight matrix as numpy
        weight_np = decoder.decoder.weight.detach().cpu().numpy()
        assert weight_np.shape == (10, 3)
        assert isinstance(weight_np, np.ndarray)
        # Extract bias as numpy (if bias is used)
        if decoder.decoder.bias is not None:
            bias_np = decoder.decoder.bias.detach().cpu().numpy()
            assert bias_np.shape == (10,)
            assert isinstance(bias_np, np.ndarray)

