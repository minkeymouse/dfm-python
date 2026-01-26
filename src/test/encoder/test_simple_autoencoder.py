"""Tests for encoder.simple_autoencoder module."""

import pytest
import torch
from dfm_python.encoder.simple_autoencoder import Encoder, SimpleAutoencoder
from dfm_python.decoder.linear import LinearDecoder
from dfm_python.utils.errors import ConfigurationError, DataValidationError


class TestEncoder:
    """Test suite for Encoder network."""
    
    def test_encoder_initialization(self):
        """Test Encoder can be initialized."""
        input_dim = 10
        encoder_dims = [64, 32, 3]  # encoder_dims includes output_dim as last element
        encoder = Encoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims
        )
        assert encoder is not None
    
    def test_encoder_forward(self):
        """Test Encoder forward pass."""
        encoder = Encoder(input_dim=10, encoder_dims=[64, 32, 3])
        x = torch.randn(5, 10)
        output = encoder(x)
        assert output.shape == (5, 3)
    
    def test_encoder_invalid_activation(self):
        """Test Encoder raises ConfigurationError for invalid activation."""
        with pytest.raises(ConfigurationError, match="Unknown activation"):
            Encoder(
                input_dim=10,
                encoder_dims=[64, 32, 3],
                activation='invalid_activation'
            )


@pytest.mark.skip(reason="AutoencoderEncoder not implemented - variational_autoencoder.py is empty")
class TestAutoencoderEncoder:
    """Test suite for AutoencoderEncoder."""
    
    def test_autoencoder_encoder_initialization(self):
        """Test AutoencoderEncoder can be initialized."""
        pass
    
    def test_autoencoder_encoder_encode_2d(self):
        """Test AutoencoderEncoder encode method with 2D input."""
        pass
    
    def test_autoencoder_encoder_encode_3d(self):
        """Test AutoencoderEncoder encode method with 3D input."""
        pass
    
    def test_autoencoder_encoder_invalid_input_dimensions(self):
        """Test AutoencoderEncoder raises DataValidationError for invalid input dimensions."""
        pass


class TestDecoderExtraction:
    """Test suite for decoder parameter extraction functions."""
    
    @pytest.mark.skip(reason="_get_decoder_layer method was removed during code simplification")
    def test_get_decoder_layer_with_decoder_attribute(self):
        """Test _get_decoder_layer extracts Linear layer from decoder.decoder."""
        from dfm_python.encoder.simple_autoencoder import _get_decoder_layer
        
        # Create a decoder with 'decoder' attribute (Linear decoder)
        class LinearDecoder:
            def __init__(self):
                self.decoder = torch.nn.Linear(3, 10)
        
        decoder = LinearDecoder()
        layer = _get_decoder_layer(decoder)
        assert isinstance(layer, torch.nn.Linear)
        assert layer.in_features == 3
        assert layer.out_features == 10
    
    @pytest.mark.skip(reason="_get_decoder_layer method was removed during code simplification")
    def test_get_decoder_layer_with_output_layer_attribute(self):
        """Test _get_decoder_layer extracts Linear layer from decoder.output_layer."""
        from dfm_python.encoder.simple_autoencoder import _get_decoder_layer
        
        # Create a decoder with 'output_layer' attribute (MLP decoder)
        class MLPDecoder:
            def __init__(self):
                self.output_layer = torch.nn.Linear(3, 10)
        
        decoder = MLPDecoder()
        layer = _get_decoder_layer(decoder)
        assert isinstance(layer, torch.nn.Linear)
        assert layer.in_features == 3
        assert layer.out_features == 10
    
    @pytest.mark.skip(reason="_get_decoder_layer method was removed during code simplification")
    def test_get_decoder_layer_with_direct_linear(self):
        """Test _get_decoder_layer returns Linear layer directly."""
        from dfm_python.encoder.simple_autoencoder import _get_decoder_layer
        
        # Direct Linear layer
        linear_layer = torch.nn.Linear(3, 10)
        layer = _get_decoder_layer(linear_layer)
        assert layer is linear_layer
        assert isinstance(layer, torch.nn.Linear)
    
    @pytest.mark.skip(reason="_get_decoder_layer method was removed during code simplification")
    def test_get_decoder_layer_invalid_decoder(self):
        """Test _get_decoder_layer raises DataValidationError for invalid decoder."""
        from dfm_python.encoder.simple_autoencoder import _get_decoder_layer
        
        # Create a decoder-like object without required attributes
        class InvalidDecoder:
            pass
        
        invalid_decoder = InvalidDecoder()
        with pytest.raises(DataValidationError, match="decoder must have"):
            _get_decoder_layer(invalid_decoder)
    
    def test_extract_decoder_params_invalid_decoder(self):
        """Test extract_decoder_params raises DataValidationError for invalid decoder."""
        from dfm_python.encoder.simple_autoencoder import extract_decoder_params
        
        # Create a decoder-like object without required attributes
        class InvalidDecoder:
            pass
        
        invalid_decoder = InvalidDecoder()
        with pytest.raises(DataValidationError, match="decoder must have"):
            extract_decoder_params(invalid_decoder)
    
    def test_extract_decoder_params_success(self):
        """Test extract_decoder_params successfully extracts decoder parameters."""
        from dfm_python.encoder.simple_autoencoder import extract_decoder_params
        from dfm_python.decoder import LinearDecoder
        import numpy as np
        
        # Use real LinearDecoder (which now has extract_params method)
        decoder = LinearDecoder(3, 10)
        C, bias = extract_decoder_params(decoder)
        
        # Verify output shapes and values
        assert isinstance(C, np.ndarray)
        assert isinstance(bias, np.ndarray)
        assert C.shape == (10, 3)  # (out_features, in_features)
        assert bias.shape == (10,)
        # Verify values are not NaN (actual values depend on initialization)
        assert not np.any(np.isnan(C))
        assert not np.any(np.isnan(bias))
    
    @pytest.mark.skip(reason="convert_decoder_to_numpy method was removed during code simplification")
    def test_convert_decoder_to_numpy_success(self):
        """Test convert_decoder_to_numpy successfully converts decoder to numpy."""
        from dfm_python.encoder.simple_autoencoder import convert_decoder_to_numpy
        import numpy as np
        
        # Create a decoder with 'decoder' attribute
        class LinearDecoder:
            def __init__(self):
                self.decoder = torch.nn.Linear(3, 10)
                # Initialize weights for deterministic test
                torch.nn.init.ones_(self.decoder.weight)
                torch.nn.init.zeros_(self.decoder.bias)
        
        decoder = LinearDecoder()
        result = convert_decoder_to_numpy(decoder)
        
        # convert_decoder_to_numpy returns a tuple (bias, C)
        assert isinstance(result, tuple)
        assert len(result) == 2
        bias, C = result
        
        # Verify output shapes and types
        assert isinstance(bias, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert bias.shape == (10,)
        assert C.shape == (10, 13)  # C includes bias column and weight matrix
        # Verify values (zeros for bias, ones for weight)
        assert np.allclose(bias, 0.0)


# Tests for noise injection removed - SimpleAutoencoder does not support noise injection.
# These tests were removed to reduce clutter since the functionality is not implemented.

