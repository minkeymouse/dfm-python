"""Tests for decoder modules.

This module tests both linear and MLP decoders for DDFM.
"""

import pytest
import numpy as np
import torch

try:
    from dfm_python.decoder.linear import Decoder
    from dfm_python.decoder.mlp import MLPDecoder
    from dfm_python.encoder.autoencoder import extract_decoder_params, convert_decoder_to_numpy
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch required for decoder tests", allow_module_level=True)


class TestLinearDecoder:
    """Test linear decoder."""
    
    def test_linear_decoder_initialization(self):
        """Test linear decoder initialization."""
        input_dim = 3
        output_dim = 10
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        assert decoder is not None
        assert decoder.decoder is not None
        assert decoder.decoder.weight.shape == (output_dim, input_dim)
        assert decoder.decoder.bias is not None
    
    def test_linear_decoder_forward(self):
        """Test linear decoder forward pass."""
        input_dim = 3
        output_dim = 10
        batch_size = 5
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        factors = torch.randn(batch_size, input_dim)
        reconstructed = decoder(factors)
        
        assert reconstructed.shape == (batch_size, output_dim)
        assert torch.all(torch.isfinite(reconstructed))
    
    def test_linear_decoder_no_bias(self):
        """Test linear decoder without bias."""
        decoder = Decoder(
            input_dim=2,
            output_dim=5,
            use_bias=False
        )
        
        assert decoder.decoder.bias is None
        
        factors = torch.randn(3, 2)
        reconstructed = decoder(factors)
        assert reconstructed.shape == (3, 5)


class TestMLPDecoder:
    """Test MLP decoder."""
    
    def test_mlp_decoder_initialization_default(self):
        """Test MLP decoder initialization with default hidden layers."""
        input_dim = 3
        output_dim = 10
        
        decoder = MLPDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            activation='relu',
            use_batch_norm=False,
            use_bias=True
        )
        
        assert decoder is not None
        assert len(decoder.layers) == 1  # Default: single hidden layer
        assert decoder.layers[0].weight.shape == (output_dim, input_dim)
        assert decoder.output_layer is not None
    
    def test_mlp_decoder_initialization_custom(self):
        """Test MLP decoder initialization with custom hidden layers."""
        input_dim = 3
        output_dim = 10
        hidden_dims = [16, 8]
        
        decoder = MLPDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation='relu',
            use_batch_norm=False,
            use_bias=True
        )
        
        assert decoder is not None
        assert len(decoder.layers) == len(hidden_dims)
        assert decoder.layers[0].weight.shape == (hidden_dims[0], input_dim)
        assert decoder.layers[-1].weight.shape == (hidden_dims[-1], hidden_dims[-2] if len(hidden_dims) > 1 else input_dim)
        assert decoder.output_layer.weight.shape == (output_dim, hidden_dims[-1])
    
    def test_mlp_decoder_forward(self):
        """Test MLP decoder forward pass."""
        input_dim = 3
        output_dim = 10
        batch_size = 5
        
        decoder = MLPDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=False
        )
        
        factors = torch.randn(batch_size, input_dim)
        reconstructed = decoder(factors)
        
        assert reconstructed.shape == (batch_size, output_dim)
        assert torch.all(torch.isfinite(reconstructed))
    
    def test_mlp_decoder_activations(self):
        """Test MLP decoder with different activations."""
        input_dim = 2
        output_dim = 5
        
        for activation in ['relu', 'tanh', 'sigmoid']:
            decoder = MLPDecoder(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[8],
                activation=activation,
                use_batch_norm=False
            )
            
            factors = torch.randn(3, input_dim)
            reconstructed = decoder(factors)
            assert reconstructed.shape == (3, output_dim)
            assert torch.all(torch.isfinite(reconstructed))
    
    def test_mlp_decoder_batch_norm(self):
        """Test MLP decoder with batch normalization."""
        decoder = MLPDecoder(
            input_dim=3,
            output_dim=10,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=True,
            use_bias=True
        )
        
        assert decoder.use_batch_norm is True
        assert decoder.batch_norms is not None
        assert len(decoder.batch_norms) == len(decoder.layers)
        
        factors = torch.randn(5, 3)
        reconstructed = decoder(factors)
        assert reconstructed.shape == (5, 10)


class TestDecoderUtils:
    """Test decoder utility functions."""
    
    def test_extract_decoder_params_linear(self):
        """Test extract_decoder_params for linear decoder."""
        decoder = Decoder(
            input_dim=3,
            output_dim=10,
            use_bias=True
        )
        
        C, bias = extract_decoder_params(decoder)
        
        assert C.shape == (10, 3)
        assert bias.shape == (10,)
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(bias))
    
    def test_extract_decoder_params_mlp(self):
        """Test extract_decoder_params for MLP decoder."""
        decoder = MLPDecoder(
            input_dim=3,
            output_dim=10,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=False
        )
        
        C, bias = extract_decoder_params(decoder)
        
        # For MLP, C is from output_layer: (output_dim x last_hidden_dim)
        # But we expect (output_dim x input_dim) for consistency
        # Actually, MLP decoder's output_layer is (output_dim x last_hidden_dim)
        # So C shape will be (10, 16) not (10, 3)
        assert C.shape == (10, 16)  # output_dim x last_hidden_dim
        assert bias.shape == (10,)
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(bias))
    
    def test_convert_decoder_to_numpy_linear_var1(self):
        """Test convert_decoder_to_numpy for linear decoder VAR(1)."""
        decoder = Decoder(
            input_dim=2,
            output_dim=5,
            use_bias=True
        )
        
        bias, emission = convert_decoder_to_numpy(
            decoder,
            has_bias=True,
            factor_order=1
        )
        
        # VAR(1): emission = [C, I] where C is (N x m), I is (N x N)
        assert bias.shape == (5,)
        assert emission.shape == (5, 2 + 5)  # (N x (m + N))
        assert np.all(np.isfinite(emission))
    
    def test_convert_decoder_to_numpy_mlp_var1(self):
        """Test convert_decoder_to_numpy for MLP decoder VAR(1)."""
        decoder = MLPDecoder(
            input_dim=2,
            output_dim=5,
            hidden_dims=[8],
            activation='relu',
            use_batch_norm=False
        )
        
        bias, emission = convert_decoder_to_numpy(
            decoder,
            has_bias=True,
            factor_order=1
        )
        
        # For MLP, emission uses output_layer which is (N x last_hidden_dim)
        # So emission will be (N x (last_hidden_dim + N))
        assert bias.shape == (5,)
        assert emission.shape == (5, 8 + 5)  # (N x (last_hidden_dim + N))
        assert np.all(np.isfinite(emission))


class TestDecoderIntegration:
    """Test decoder integration with DDFM."""
    
    def test_linear_decoder_integration(self):
        """Test linear decoder with encoder."""
        from dfm_python.encoder.autoencoder import Encoder
        
        input_dim = 10
        num_factors = 3
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            output_dim=num_factors,
            activation='relu',
            use_batch_norm=False
        )
        
        decoder = Decoder(
            input_dim=num_factors,
            output_dim=input_dim,
            use_bias=True
        )
        
        batch_size = 5
        x = torch.randn(batch_size, input_dim)
        
        factors = encoder(x)
        reconstructed = decoder(factors)
        
        assert factors.shape == (batch_size, num_factors)
        assert reconstructed.shape == (batch_size, input_dim)
        assert torch.all(torch.isfinite(reconstructed))
    
    def test_mlp_decoder_integration(self):
        """Test MLP decoder with encoder."""
        from dfm_python.encoder.autoencoder import Encoder
        
        input_dim = 10
        num_factors = 3
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            output_dim=num_factors,
            activation='relu',
            use_batch_norm=False
        )
        
        decoder = MLPDecoder(
            input_dim=num_factors,
            output_dim=input_dim,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=False
        )
        
        batch_size = 5
        x = torch.randn(batch_size, input_dim)
        
        factors = encoder(x)
        reconstructed = decoder(factors)
        
        assert factors.shape == (batch_size, num_factors)
        assert reconstructed.shape == (batch_size, input_dim)
        assert torch.all(torch.isfinite(reconstructed))
    
    def test_decoder_training_step(self):
        """Test training step with both decoder types."""
        from dfm_python.encoder.autoencoder import Encoder
        
        input_dim = 10
        num_factors = 3
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            output_dim=num_factors,
            activation='relu',
            use_batch_norm=False
        )
        
        for decoder_type, decoder_class in [("linear", Decoder), ("mlp", MLPDecoder)]:
            if decoder_type == "linear":
                decoder = decoder_class(
                    input_dim=num_factors,
                    output_dim=input_dim,
                    use_bias=True
                )
            else:
                decoder = decoder_class(
                    input_dim=num_factors,
                    output_dim=input_dim,
                    hidden_dims=[16],
                    activation='relu',
                    use_batch_norm=False
                )
            
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=0.01
            )
            
            batch_size = 5
            x = torch.randn(batch_size, input_dim)
            
            encoder.train()
            decoder.train()
            
            optimizer.zero_grad()
            factors = encoder(x)
            reconstructed = decoder(factors)
            loss = torch.nn.functional.mse_loss(reconstructed, x)
            loss.backward()
            optimizer.step()
            
            assert torch.isfinite(loss)
            assert loss.item() > 0

