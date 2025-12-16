"""Tests for encoder and decoder modules.

This module tests the encoder/decoder components for DDFM:
- Encoder: Nonlinear encoder network
- AutoencoderEncoder: Wrapper for BaseEncoder interface
- Decoder: Linear decoder network
- Denoising trainer: MCMC-based training procedure
"""

import pytest
import numpy as np
import torch

try:
    from dfm_python.encoder.autoencoder import Encoder, AutoencoderEncoder, extract_decoder_params, convert_decoder_to_numpy
    from dfm_python.decoder.linear import Decoder
    from dfm_python.trainer.denoising import DDFMDenoisingTrainer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch required for encoder/decoder tests", allow_module_level=True)


class TestEncoder:
    """Test Encoder network."""
    
    def test_encoder_initialization(self):
        """Test Encoder initialization."""
        input_dim = 10
        hidden_dims = [64, 32]
        output_dim = 3
        activation = 'relu'
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            use_batch_norm=True
        )
        
        assert encoder is not None
        assert len(encoder.layers) == len(hidden_dims)
        assert encoder.use_batch_norm is True
        assert encoder.output_layer is not None
    
    def test_encoder_forward(self):
        """Test Encoder forward pass."""
        input_dim = 10
        hidden_dims = [64, 32]
        output_dim = 3
        batch_size = 5
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='relu',
            use_batch_norm=False  # Disable for simpler test
        )
        
        x = torch.randn(batch_size, input_dim)
        factors = encoder(x)
        
        assert factors.shape == (batch_size, output_dim)
        assert torch.all(torch.isfinite(factors))
    
    def test_encoder_activations(self):
        """Test different activation functions."""
        input_dim = 5
        hidden_dims = [16]
        output_dim = 2
        
        for activation in ['relu', 'tanh', 'sigmoid']:
            encoder = Encoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                use_batch_norm=False
            )
            
            x = torch.randn(3, input_dim)
            factors = encoder(x)
            assert factors.shape == (3, output_dim)
            assert torch.all(torch.isfinite(factors))


class TestAutoencoderEncoder:
    """Test AutoencoderEncoder wrapper."""
    
    def test_autoencoder_encoder_initialization(self):
        """Test AutoencoderEncoder initialization."""
        n_components = 3
        input_dim = 10
        hidden_dims = [32, 16]
        
        encoder = AutoencoderEncoder(
            n_components=n_components,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation='relu',
            use_batch_norm=False
        )
        
        assert encoder.n_components == n_components
        assert encoder.input_dim == input_dim
        assert encoder.hidden_dims == hidden_dims
        assert encoder.encoder_module is not None
    
    def test_autoencoder_encoder_fit(self):
        """Test AutoencoderEncoder fit method (no-op)."""
        encoder = AutoencoderEncoder(
            n_components=2,
            input_dim=5,
            hidden_dims=[16],
            activation='relu'
        )
        
        X = np.random.randn(20, 5)
        result = encoder.fit(X)
        
        assert result is encoder  # Should return self
        assert encoder._is_fitted is True
    
    def test_autoencoder_encoder_encode_2d(self):
        """Test AutoencoderEncoder encode with 2D input."""
        encoder = AutoencoderEncoder(
            n_components=2,
            input_dim=5,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=False
        )
        
        # Fit first (required for interface)
        X = np.random.randn(20, 5)
        encoder.fit(X)
        
        # Encode
        factors = encoder.encode(X)
        
        assert factors.shape == (20, 2)
        assert torch.all(torch.isfinite(factors))
    
    def test_autoencoder_encoder_encode_3d(self):
        """Test AutoencoderEncoder encode with 3D input."""
        encoder = AutoencoderEncoder(
            n_components=2,
            input_dim=5,
            hidden_dims=[16],
            activation='relu',
            use_batch_norm=False
        )
        
        # Fit first
        X = np.random.randn(20, 5)
        encoder.fit(X)
        
        # Encode with 3D input (batch_size, T, N)
        X_3d = np.random.randn(3, 10, 5)
        factors = encoder.encode(X_3d)
        
        assert factors.shape == (3, 10, 2)
        assert torch.all(torch.isfinite(factors))


class TestDecoder:
    """Test Decoder network."""
    
    def test_decoder_initialization(self):
        """Test Decoder initialization."""
        input_dim = 3  # num_factors
        output_dim = 10  # num_series
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        assert decoder is not None
        assert decoder.decoder is not None
        assert decoder.decoder.weight.shape == (output_dim, input_dim)
        assert decoder.decoder.bias is not None
    
    def test_decoder_forward(self):
        """Test Decoder forward pass."""
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
    
    def test_decoder_no_bias(self):
        """Test Decoder without bias."""
        decoder = Decoder(
            input_dim=2,
            output_dim=5,
            use_bias=False
        )
        
        assert decoder.decoder.bias is None
        
        factors = torch.randn(3, 2)
        reconstructed = decoder(factors)
        assert reconstructed.shape == (3, 5)


class TestDecoderUtils:
    """Test decoder utility functions."""
    
    def test_extract_decoder_params(self):
        """Test extract_decoder_params function."""
        input_dim = 3
        output_dim = 10
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        C, bias = extract_decoder_params(decoder)
        
        assert C.shape == (output_dim, input_dim)
        assert bias.shape == (output_dim,)
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(bias))
    
    def test_convert_decoder_to_numpy_var1(self):
        """Test convert_decoder_to_numpy for VAR(1)."""
        input_dim = 2
        output_dim = 5
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        bias, emission = convert_decoder_to_numpy(
            decoder,
            has_bias=True,
            factor_order=1
        )
        
        # VAR(1): emission = [C, I] where C is (N x m), I is (N x N)
        # So emission should be (N x (m + N)) = (5 x (2 + 5)) = (5 x 7)
        assert bias.shape == (output_dim,)
        assert emission.shape == (output_dim, input_dim + output_dim)
        assert np.all(np.isfinite(emission))
    
    def test_convert_decoder_to_numpy_var2(self):
        """Test convert_decoder_to_numpy for VAR(2)."""
        input_dim = 2
        output_dim = 5
        
        decoder = Decoder(
            input_dim=input_dim,
            output_dim=output_dim,
            use_bias=True
        )
        
        bias, emission = convert_decoder_to_numpy(
            decoder,
            has_bias=True,
            factor_order=2
        )
        
        # VAR(2): emission = [C, zeros, I] where C is (N x m), zeros is (N x m), I is (N x N)
        # So emission should be (N x (m + m + N)) = (5 x (2 + 2 + 5)) = (5 x 9)
        assert bias.shape == (output_dim,)
        assert emission.shape == (output_dim, 2 * input_dim + output_dim)
        assert np.all(np.isfinite(emission))


class TestDenoisingTrainer:
    """Test DDFMDenoisingTrainer."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a minimal DDFM model for testing."""
        from dfm_python.models.ddfm import DDFM
        
        model = DDFM(
            encoder_layers=[16, 8],
            num_factors=2,
            factor_order=1,
            epochs=5,
            max_iter=2,  # Minimal iterations for testing
            batch_size=8,
            learning_rate=0.01,
            use_idiosyncratic=False  # Disable for simpler test
        )
        
        # Initialize networks manually
        model.initialize_networks(5)  # 5 series
        
        return model
    
    def test_denoising_trainer_initialization(self, sample_model):
        """Test DDFMDenoisingTrainer initialization."""
        trainer = DDFMDenoisingTrainer(sample_model)
        
        assert trainer.model is sample_model
    
    def test_denoising_trainer_fit(self, sample_model):
        """Test DDFMDenoisingTrainer fit method."""
        trainer = DDFMDenoisingTrainer(sample_model)
        
        T, N = 20, 5
        X = torch.randn(T, N)
        x_clean = X.clone()
        missing_mask = np.zeros((T, N), dtype=bool)
        
        # Run training with minimal iterations
        state = trainer.fit(
            X=X,
            x_clean=x_clean,
            missing_mask=missing_mask,
            max_iter=2,
            tolerance=1e-3,
            disp=1
        )
        
        assert state is not None
        assert state.factors.shape == (T, sample_model.num_factors)
        assert state.prediction.shape == (T, N)
        assert isinstance(state.converged, bool)
        assert state.num_iter >= 1
        assert np.all(np.isfinite(state.factors))
        assert np.all(np.isfinite(state.prediction))
    
    def test_denoising_trainer_with_missing_data(self, sample_model):
        """Test DDFMDenoisingTrainer with missing data."""
        trainer = DDFMDenoisingTrainer(sample_model)
        
        T, N = 20, 5
        X = torch.randn(T, N)
        x_clean = X.clone()
        
        # Create missing mask (some missing values)
        missing_mask = np.zeros((T, N), dtype=bool)
        missing_mask[5:7, 2] = True  # Missing values in series 2, periods 5-6
        missing_mask[10:12, 0] = True  # Missing values in series 0, periods 10-11
        
        # Set missing values to NaN
        X_np = X.numpy()
        X_np[missing_mask] = np.nan
        X = torch.tensor(X_np)
        
        # Run training
        state = trainer.fit(
            X=X,
            x_clean=x_clean,
            missing_mask=missing_mask,
            max_iter=2,
            tolerance=1e-3,
            disp=1
        )
        
        assert state is not None
        assert state.factors.shape == (T, sample_model.num_factors)
        assert np.all(np.isfinite(state.factors))


class TestEncoderDecoderIntegration:
    """Test encoder-decoder integration."""
    
    def test_encoder_decoder_roundtrip(self):
        """Test that encoder-decoder can reconstruct input."""
        input_dim = 10
        hidden_dims = [32, 16]
        num_factors = 3
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_factors,
            activation='relu',
            use_batch_norm=False
        )
        
        decoder = Decoder(
            input_dim=num_factors,
            output_dim=input_dim,
            use_bias=True
        )
        
        # Create sample data
        batch_size = 5
        x = torch.randn(batch_size, input_dim)
        
        # Encode
        factors = encoder(x)
        assert factors.shape == (batch_size, num_factors)
        
        # Decode
        reconstructed = decoder(factors)
        assert reconstructed.shape == (batch_size, input_dim)
        
        # Check that reconstruction is finite
        assert torch.all(torch.isfinite(reconstructed))
        
        # Note: Without training, reconstruction won't be accurate,
        # but we can check that the shapes and values are correct
    
    def test_autoencoder_training_step(self):
        """Test a single training step of encoder-decoder."""
        input_dim = 10
        hidden_dims = [32, 16]
        num_factors = 3
        
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_factors,
            activation='relu',
            use_batch_norm=False
        )
        
        decoder = Decoder(
            input_dim=num_factors,
            output_dim=input_dim,
            use_bias=True
        )
        
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.01
        )
        
        # Training step
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
        
        # Check that loss is finite
        assert torch.isfinite(loss)
        assert loss.item() > 0

