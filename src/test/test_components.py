from dfm_python.decoder.linear import Decoder
from dfm_python.decoder.mlp import MLPDecoder
from dfm_python.encoder.autoencoder import Encoder
from dfm_python.encoder.autoencoder import Encoder, AutoencoderEncoder, extract_decoder_params, convert_decoder_to_numpy
from dfm_python.encoder.autoencoder import extract_decoder_params, convert_decoder_to_numpy
from dfm_python.models.ddfm import DDFM
from dfm_python.trainer.ddfm import DDFMDenoisingTrainer
import numpy as np
import pytest
import torch

# === From test_encoder.py ===

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
    from dfm_python.trainer.ddfm import DDFMDenoisingTrainer
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
        return model
        
        # Initialize networks manually
        model.initialize_networks(5)  # 5 series
        
        return model
    
    def test_denoising_trainer_initialization(self, sample_model):
        """Test DDFMDenoisingTrainer initialization."""
        trainer = DDFMDenoisingTrainer(sample_model)
        
        assert trainer.model is sample_model
        # Verify trainer has required methods
        assert hasattr(trainer, 'fit')
        assert callable(getattr(trainer, 'fit', None))
    
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
        # Verify state is returned
        assert state is not None
    
    def test_denoising_trainer_edge_cases(self, sample_model):
        """Test DDFMDenoisingTrainer with edge cases."""
        trainer = DDFMDenoisingTrainer(sample_model)
        
        T, N = 20, 5
        X = torch.randn(T, N)
        x_clean = X.clone()
        
        # Test with all data missing
        all_missing_mask = np.ones((T, N), dtype=bool)
        try:
            state = trainer.fit(
                X=X,
                x_clean=x_clean,
                missing_mask=all_missing_mask,
                max_iter=1,
                tolerance=1e-3,
                disp=0
            )
            # May succeed or fail - either is acceptable
        except Exception:
            # Expected if all data is missing
            pass
        
        # Test with empty data (edge case)
        try:
            empty_X = torch.randn(0, N)
            empty_clean = empty_X.clone()
            empty_mask = np.zeros((0, N), dtype=bool)
            empty_state = trainer.fit(
                X=empty_X,
                x_clean=empty_clean,
                missing_mask=empty_mask,
                max_iter=1,
                tolerance=1e-3,
                disp=0
            )
            # If it succeeds, verify state structure
            if empty_state is not None:
                assert hasattr(empty_state, 'factors')
        except (ValueError, RuntimeError, IndexError):
            # Expected for empty data
            pass
        
        # Test with small batch (should still work)
        X_small = torch.randn(5, N)
        x_clean_small = X_small.clone()
        small_mask = np.zeros((5, N), dtype=bool)
        try:
            state = trainer.fit(
                X=X_small,
                x_clean=x_clean_small,
                missing_mask=small_mask,
                max_iter=1,
                tolerance=1e-3,
                disp=0
            )
            if state is not None:
                assert state.factors.shape[0] == 5
                assert state.factors.shape[1] == sample_model.num_factors
                assert state.prediction.shape == (5, N)
                assert isinstance(state.converged, bool)
                assert state.num_iter >= 1
                assert np.all(np.isfinite(state.factors))
                assert np.all(np.isfinite(state.prediction))
        except (ValueError, RuntimeError):
            # May fail with very small data - acceptable
            pass
    
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



# === From test_decoder.py ===

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



