"""Tests for encoder.simple_autoencoder module."""

import pytest
import torch
from dfm_python.encoder.simple_autoencoder import Encoder, Autoencoder
from dfm_python.encoder.variational_autoencoder import AutoencoderEncoder
from dfm_python.decoder.linear import Decoder
from dfm_python.utils.errors import ConfigurationError, DataValidationError


class TestEncoder:
    """Test suite for Encoder network."""
    
    def test_encoder_initialization(self):
        """Test Encoder can be initialized."""
        input_dim = 10
        hidden_dims = [64, 32]
        output_dim = 3
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        assert encoder is not None
    
    def test_encoder_forward(self):
        """Test Encoder forward pass."""
        encoder = Encoder(input_dim=10, hidden_dims=[64, 32], output_dim=3)
        x = torch.randn(5, 10)
        output = encoder(x)
        assert output.shape == (5, 3)
    
    def test_encoder_invalid_activation(self):
        """Test Encoder raises ConfigurationError for invalid activation."""
        with pytest.raises(ConfigurationError, match="Unknown activation"):
            Encoder(
                input_dim=10,
                hidden_dims=[64, 32],
                output_dim=3,
                activation='invalid_activation'
            )


class TestAutoencoderEncoder:
    """Test suite for AutoencoderEncoder."""
    
    def test_autoencoder_encoder_initialization(self):
        """Test AutoencoderEncoder can be initialized."""
        encoder = AutoencoderEncoder(
            input_dim=10,
            hidden_dims=[64, 32],
            n_components=3
        )
        assert encoder is not None
        assert encoder.n_components == 3
    
    def test_autoencoder_encoder_encode_2d(self):
        """Test AutoencoderEncoder encode method with 2D input."""
        encoder = AutoencoderEncoder(input_dim=10, hidden_dims=[64, 32], n_components=3)
        encoder.fit(torch.randn(5, 10))  # Mark as fitted
        x = torch.randn(5, 10)  # (T, N)
        output = encoder.encode(x)
        assert output.shape == (5, 3)
    
    def test_autoencoder_encoder_encode_3d(self):
        """Test AutoencoderEncoder encode method with 3D input."""
        encoder = AutoencoderEncoder(input_dim=10, hidden_dims=[64, 32], n_components=3)
        encoder.fit(torch.randn(5, 10))  # Mark as fitted
        x = torch.randn(2, 5, 10)  # (batch_size, T, N)
        output = encoder.encode(x)
        assert output.shape == (2, 5, 3)
    
    def test_autoencoder_encoder_invalid_input_dimensions(self):
        """Test AutoencoderEncoder raises DataValidationError for invalid input dimensions."""
        encoder = AutoencoderEncoder(input_dim=10, hidden_dims=[64, 32], n_components=3)
        encoder.fit(torch.randn(5, 10))  # Mark as fitted
        # Try with 1D input (should raise DataValidationError)
        x_1d = torch.randn(10)
        with pytest.raises(DataValidationError, match="Expected 2D or 3D input"):
            encoder.encode(x_1d)
        
        # Try with 4D input (should raise DataValidationError)
        x_4d = torch.randn(2, 3, 5, 10)
        with pytest.raises(DataValidationError, match="Expected 2D or 3D input"):
            encoder.encode(x_4d)


class TestDecoderExtraction:
    """Test suite for decoder parameter extraction functions."""
    
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
    
    def test_get_decoder_layer_with_direct_linear(self):
        """Test _get_decoder_layer returns Linear layer directly."""
        from dfm_python.encoder.simple_autoencoder import _get_decoder_layer
        
        # Direct Linear layer
        linear_layer = torch.nn.Linear(3, 10)
        layer = _get_decoder_layer(linear_layer)
        assert layer is linear_layer
        assert isinstance(layer, torch.nn.Linear)
    
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
        import numpy as np
        
        # Create a decoder with 'decoder' attribute
        class LinearDecoder:
            def __init__(self):
                self.decoder = torch.nn.Linear(3, 10)
                # Initialize weights for deterministic test
                torch.nn.init.ones_(self.decoder.weight)
                torch.nn.init.zeros_(self.decoder.bias)
        
        decoder = LinearDecoder()
        C, bias = extract_decoder_params(decoder)
        
        # Verify output shapes and values
        assert isinstance(C, np.ndarray)
        assert isinstance(bias, np.ndarray)
        assert C.shape == (10, 3)  # (out_features, in_features)
        assert bias.shape == (10,)
        # Verify values (ones for weight, zeros for bias)
        assert np.allclose(C, 1.0)
        assert np.allclose(bias, 0.0)
    
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


class TestAutoencoderNoiseInjection:
    """Test suite for Autoencoder noise injection functionality."""
    
    def test_autoencoder_without_noise_injection(self):
        """Test Autoencoder can be initialized without noise injection."""
        encoder = Encoder(input_dim=10, hidden_dims=[16], output_dim=3)
        decoder = Decoder(input_dim=3, output_dim=10)
        autoencoder = Autoencoder(encoder, decoder)
        
        assert autoencoder._num_series is None
        assert autoencoder._noise_samples is None
        assert autoencoder._generator is None
    
    def test_autoencoder_with_noise_injection(self):
        """Test Autoencoder can be initialized with noise injection."""
        encoder = Encoder(input_dim=10, hidden_dims=[16], output_dim=3)
        decoder = Decoder(input_dim=3, output_dim=10)
        Sigma_eps = torch.ones(10) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=10, Sigma_eps=Sigma_eps, seed=42)
        
        assert autoencoder._num_series == 10
        assert autoencoder.Sigma_eps is not None
        assert autoencoder.Sigma_eps.shape == (10,)
        assert autoencoder._generator is not None
    
    def test_autoencoder_generate_noise_samples(self):
        """Test generate_noise_samples pre-generates noise correctly."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 3
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        assert autoencoder._noise_samples is not None
        assert autoencoder._noise_samples.shape == (n_mc_samples, T, 5)
        # Verify noise has correct scale (approximately)
        assert torch.allclose(autoencoder._noise_samples.std(dim=1), Sigma_eps, atol=0.05)
    
    def test_autoencoder_inject_noise_without_samples(self):
        """Test inject_noise returns clean data when noise_samples not generated."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        autoencoder = Autoencoder(encoder, decoder, num_series=5)
        
        x = torch.ones(10, 5)
        x_corrupted, mask = autoencoder.inject_noise(x, training=True)
        
        assert torch.allclose(x_corrupted, x)
        assert mask.all()
    
    def test_autoencoder_inject_noise_subtracts_epsilon(self):
        """Test inject_noise subtracts epsilon following original DDFM pattern."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 2
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        x = torch.ones(10, 5)
        x_corrupted, mask = autoencoder.inject_noise(x, sample_idx=0, training=True)
        
        # Verify noise was subtracted (x_corrupted < x)
        assert (x_corrupted < x).any()
        assert mask.all()
        assert x_corrupted.shape == x.shape
    
    def test_autoencoder_inject_noise_with_batch_slicing(self):
        """Test inject_noise works with batch slicing (start_idx, end_idx)."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 2
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        x = torch.ones(5, 5)  # Smaller batch
        x_corrupted, mask = autoencoder.inject_noise(
            x, sample_idx=0, start_idx=0, end_idx=5, training=True
        )
        
        assert x_corrupted.shape == x.shape
        assert mask.all()
    
    def test_autoencoder_inject_noise_inference_mode(self):
        """Test inject_noise returns clean data in inference mode (training=False)."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 2
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        x = torch.ones(10, 5)
        x_corrupted, mask = autoencoder.inject_noise(x, sample_idx=0, training=False)
        
        assert torch.allclose(x_corrupted, x)
        assert mask.all()
    
    def test_autoencoder_inject_noise_requires_sample_idx(self):
        """Test inject_noise raises error when sample_idx not provided."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 2
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        x = torch.ones(10, 5)
        with pytest.raises(ValueError, match="sample_idx is required"):
            autoencoder.inject_noise(x, training=True)
    
    def test_autoencoder_update_sigma_eps(self):
        """Test update_Sigma_eps updates the Sigma_eps buffer."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps_old = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps_old, seed=42)
        
        Sigma_eps_new = torch.ones(5) * 0.2
        autoencoder.update_Sigma_eps(Sigma_eps_new)
        
        assert torch.allclose(autoencoder.Sigma_eps, Sigma_eps_new)
    
    def test_autoencoder_update_sigma_eps_scalar(self):
        """Test update_Sigma_eps accepts scalar input."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=torch.ones(5) * 0.1)
        
        autoencoder.update_Sigma_eps(torch.tensor(0.2))
        
        assert torch.allclose(autoencoder.Sigma_eps, torch.ones(5) * 0.2)
    
    def test_autoencoder_noise_injection_follows_ddfm_pattern(self):
        """Test noise injection follows original DDFM pattern: x_sim_den = x_sim_den - eps_draws."""
        encoder = Encoder(input_dim=5, hidden_dims=[8], output_dim=2)
        decoder = Decoder(input_dim=2, output_dim=5)
        Sigma_eps = torch.ones(5) * 0.1
        autoencoder = Autoencoder(encoder, decoder, num_series=5, Sigma_eps=Sigma_eps, seed=42)
        
        n_mc_samples = 2
        T = 10
        autoencoder.generate_noise_samples(n_mc_samples, T, device=torch.device('cpu'))
        
        x_clean = torch.ones(10, 5) * 2.0
        x_corrupted, _ = autoencoder.inject_noise(x_clean, sample_idx=0, training=True)
        
        # Verify: x_corrupted = x_clean - noise (subtraction, not addition)
        noise = autoencoder._noise_samples[0]
        expected = x_clean - noise
        assert torch.allclose(x_corrupted, expected, atol=1e-6)

