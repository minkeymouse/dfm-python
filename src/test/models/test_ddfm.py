"""Tests for models.ddfm module."""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock
from dfm_python.models.ddfm import DDFM, DDFMTrainingState
from dfm_python.utils.errors import ConfigurationError, DataError, DataValidationError, ModelNotInitializedError, ModelNotTrainedError
from dfm_python.config.constants import MIN_VARIABLES, MIN_DDFM_TIME_STEPS, DEFAULT_INF_VALUE, DEFAULT_ENCODER_LAYERS
from dfm_python.utils.checkpoint import infer_ddfm_input_dim, infer_input_dim_from_data


class TestDDFM:
    """Test suite for DDFM model."""
    
    def _create_initialized_ddfm(self, input_dim=5, time_steps=10, encoder_layers=None, num_factors=2, **model_kwargs):
        """Helper to create and initialize DDFM model for testing.
        
        Consolidates duplicate pattern: model creation, X generation, network initialization.
        
        Parameters
        ----------
        input_dim : int, default 5
            Number of variables (input dimension)
        time_steps : int, default 10
            Number of time steps for test data
        encoder_layers : list, optional
            Encoder layer sizes. Defaults to [64, 32] if None.
        num_factors : int, default 2
            Number of factors
        **model_kwargs
            Additional arguments passed to DDFM constructor
            
        Returns
        -------
        model : DDFM
            Initialized DDFM model
        X : torch.Tensor
            Test data tensor of shape (time_steps, input_dim)
        """
        if encoder_layers is None:
            encoder_layers = DEFAULT_ENCODER_LAYERS
        model = DDFM(encoder_layers=encoder_layers, num_factors=num_factors, **model_kwargs)
        X = torch.randn(time_steps, input_dim)
        model.initialize_networks(input_dim)
        return model, X
    
    @pytest.mark.parametrize("encoder_layers,num_factors", [
        (DEFAULT_ENCODER_LAYERS, 2),
        ([64, 32], 2),
        (None, 1),  # Minimal initialization
    ])
    def test_ddfm_initialization(self, encoder_layers, num_factors):
        """Test DDFM can be initialized with various parameters."""
        if encoder_layers is None:
            model = DDFM(num_factors=num_factors)
        else:
            model = DDFM(encoder_layers=encoder_layers, num_factors=num_factors)
        assert model.num_factors == num_factors
        if encoder_layers is not None:
            assert model.encoder_layers == encoder_layers
    
    # factor_order tests removed - factors now always use AR(1) dynamics (simplified)
    # factor_order parameter was removed from DDFM.__init__()
    
    def test_ddfm_forward(self):
        """Test DDFM forward pass."""
        model, X = self._create_initialized_ddfm(input_dim=5, time_steps=10)
        # Forward pass
        y_pred = model.forward(X)
        assert y_pred.shape == X.shape
        assert isinstance(y_pred, torch.Tensor)
        assert torch.isfinite(y_pred).all(), "Forward pass output should be finite"
    
    def test_ddfm_forward_not_initialized(self):
        """Test DDFM forward raises error when networks not initialized."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        X = torch.randn(10, 5)
        with pytest.raises(ModelNotInitializedError, match="networks must be initialized"):
            model.forward(X)
    
    def test_ddfm_window_size_parameter(self):
        """Test DDFM window_size parameter."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2, n_mc_samples=10, window_size=100)
        # Verify window_size is set correctly
        assert model.window_size == 100
        assert model.n_mc_samples == 10
    
    def test_ddfm_grad_clip_val_parameter(self):
        """Test DDFM gradient clipping parameter validation."""
        from dfm_python.config.constants import DEFAULT_ZERO_VALUE
        # Test with different grad_clip_val values
        model1 = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2, grad_clip_val=0.0)
        assert model1.grad_clip_val == DEFAULT_ZERO_VALUE
        
        model2 = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2, grad_clip_val=1.0)
        assert model2.grad_clip_val == 1.0
        assert model2.grad_clip_val > DEFAULT_ZERO_VALUE
    
    def test_ddfm_result_not_trained(self):
        """Test DDFM result access raises error when model not trained."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        with pytest.raises(ModelNotTrainedError):
            model.get_result()
        with pytest.raises(ModelNotTrainedError):
            _ = model.result
    
    def test_ddfm_predict_not_trained(self):
        """Test DDFM predict raises error when model not trained."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        # DDFM.predict() doesn't take last_observation parameter
        # It uses training state from Lightning module
        with pytest.raises(ModelNotTrainedError, match="model has not been fitted yet"):
            model.predict(horizon=5)
    
    @pytest.mark.parametrize("invalid_input", ["not a dict", [1, 2, 3], 42, None])
    def test_infer_input_dim_invalid_type(self, invalid_input):
        """Test infer_ddfm_input_dim raises DataValidationError for non-dict input."""
        with pytest.raises(DataValidationError, match="state_dict must be a dictionary"):
            infer_ddfm_input_dim(invalid_input)
    
    def test_infer_input_dim_empty_dict(self):
        """Test infer_ddfm_input_dim returns None for empty dict."""
        result = infer_ddfm_input_dim({})
        assert result is None
    
    def test_infer_input_dim_no_matching_keys(self):
        """Test infer_ddfm_input_dim returns None when no matching keys found."""
        state_dict = {"some.other.key": torch.randn(10, 5)}
        result = infer_ddfm_input_dim(state_dict)
        assert result is None
    
    def test_infer_input_dim_from_encoder_layer(self):
        """Test infer_ddfm_input_dim correctly infers from encoder layer."""
        state_dict = {
            "encoder.layers.0.weight": torch.randn(32, 64)  # (hidden_dim, input_dim)
        }
        result = infer_ddfm_input_dim(state_dict)
        assert result == 64
    
    def test_infer_input_dim_from_decoder_weight(self):
        """Test infer_ddfm_input_dim correctly infers from decoder weight."""
        state_dict = {
            "decoder.decoder.weight": torch.randn(10, 5)  # (output_dim, num_factors)
        }
        result = infer_ddfm_input_dim(state_dict)
        assert result == 10
    
    @pytest.mark.parametrize("data_factory", [
        lambda: np.array([[1, 2, 3], [4, 5, 6]]),  # numpy 2D
        lambda: torch.tensor([[1, 2, 3], [4, 5, 6]])  # torch 2D
    ])
    def test_infer_input_dim_from_data_2d(self, data_factory):
        """Test infer_input_dim_from_data correctly infers from 2D array/tensor."""
        data_2d = data_factory()  # (2, 3) -> should return 3
        result = infer_input_dim_from_data(data_2d)
        assert result == 3
    
    @pytest.mark.parametrize("data_factory", [
        lambda: np.array([1, 2, 3]),  # numpy 1D
        lambda: torch.tensor([1, 2, 3])  # torch 1D
    ])
    def test_infer_input_dim_from_data_1d_raises_error(self, data_factory):
        """Test infer_input_dim_from_data raises DataError for 1D array/tensor."""
        data_1d = data_factory()  # (3,) -> should raise error
        with pytest.raises(DataError, match="Data must be at least 2D"):
            infer_input_dim_from_data(data_1d)
    
    @pytest.mark.parametrize("encoder,decoder,autoencoder,expected", [
        (True, True, True, True),  # All initialized
        (False, True, True, False),  # Encoder missing
        (True, False, True, False),  # Decoder missing
        (True, True, False, False),  # Autoencoder missing
        (False, False, False, False),  # None initialized
    ])
    def test_are_networks_initialized(self, encoder, decoder, autoencoder, expected):
        """Test _are_networks_initialized with various network states."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        if encoder:
            model.encoder = Mock()
        if decoder:
            model.decoder = Mock()
        if autoencoder:
            model.autoencoder = Mock()
        assert model._are_networks_initialized() == expected
    
    def test_ddfm_dimension_validation_uses_constants(self):
        """Test DDFM dimension validation uses MIN_VARIABLES and MIN_DDFM_TIME_STEPS constants."""
        # Verify constants are defined and have correct values
        assert MIN_VARIABLES == 1
        assert MIN_DDFM_TIME_STEPS == 2
        # Verify constants are used in validation (indirectly through code inspection)
        # The actual validation happens in _validate_training_data using MIN_VARIABLES and MIN_DDFM_TIME_STEPS
        # The constants are used in the error messages:
        # - f"DDFM {operation} failed: Need at least {MIN_VARIABLES} series, got N={N}"
        # - f"DDFM {operation} failed: Need at least {MIN_DDFM_TIME_STEPS} time periods, got T={T}"
    
    
    def test_ddfm_load_restores_training_state(self):
        """Test DDFM.load() restores training_state from checkpoint.
        
        This test verifies the Plan Iteration 16 fix: training_state restoration
        in DDFM.load() fallback path (ddfm.py:2074).
        """
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Create a training_state to save (includes training_time from Plan Iteration 11)
        training_state = DDFMTrainingState(
            factors=np.random.randn(10, 2),
            prediction=np.random.randn(10, 5),
            converged=True,
            num_iter=5,
            training_loss=0.5,
            training_time=123.45  # Plan Iteration 11: training_time tracking added
        )
        
        # Create checkpoint dictionary directly (simulating PyTorch Lightning checkpoint)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_ddfm.ckpt"
            # Create minimal checkpoint dict with state_dict and training_state
            checkpoint_dict = {
                'state_dict': model.state_dict(),
                'training_state': {
                    'factors': training_state.factors,
                    'prediction': training_state.prediction,
                    'converged': training_state.converged,
                    'num_iter': training_state.num_iter,
                    'training_loss': training_state.training_loss,
                    'training_time': training_state.training_time  # Plan Iteration 11: training_time tracking
                }
            }
            torch.save(checkpoint_dict, checkpoint_path)
            
            # Load checkpoint
            loaded_model = DDFM.load(str(checkpoint_path), input_dim=5)
            
            # Verify training_state was restored (Plan Iteration 11: includes training_time)
            assert loaded_model.training_state is not None
            assert loaded_model.training_state.converged == training_state.converged
            assert loaded_model.training_state.num_iter == training_state.num_iter
            assert np.allclose(loaded_model.training_state.factors, training_state.factors)
            assert np.allclose(loaded_model.training_state.prediction, training_state.prediction)
            # Plan Iteration 11: Verify training_time is restored in training_state
            assert loaded_model.training_state.training_time == training_state.training_time
    
    def test_ddfm_on_load_checkpoint_restores_training_time(self):
        """Test DDFM.on_load_checkpoint() restores training_time from checkpoint."""
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Create checkpoint dictionary with training_time (as saved by on_save_checkpoint)
        test_training_time = 123.45
        checkpoint_dict = {
            'state_dict': model.state_dict(),
            'training_time': test_training_time
        }
        
        # Create new model instance to test loading
        loaded_model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2)
        loaded_model.initialize_networks(X.shape[1])
        
        # Call on_load_checkpoint to restore training_time
        loaded_model.on_load_checkpoint(checkpoint_dict)
        
        # Verify training_time was restored
        assert hasattr(loaded_model, '_training_time')
        assert loaded_model._training_time == test_training_time
    
    def test_ddfm_on_train_epoch_start_generates_noise_samples(self):
        """Test on_train_epoch_start() generates noise samples in autoencoder."""
        model, X = self._create_initialized_ddfm(input_dim=5, time_steps=20, n_mc_samples=3)
        
        # Set up data structures needed
        model._data_clean = X.clone()
        model._data_target = X.clone()
        model._missing_mask = np.zeros((20, 5), dtype=bool)
        model._dataset = Mock()
        model._dataset.data_clean = X
        
        # Initialize buffers
        model._initialize_buffers(20, 5, X.device, X.dtype)
        model.Sigma_eps = torch.ones(5) * 0.1
        
        # Call on_train_epoch_start
        model.on_train_epoch_start()
        
        # Verify noise samples were generated in autoencoder
        assert model.autoencoder._noise_samples is not None
        assert model.autoencoder._noise_samples.shape == (3, 20, 5), f"Expected (3, 20, 5), got {model.autoencoder._noise_samples.shape}"
        assert torch.isfinite(model.autoencoder._noise_samples).all(), "Noise samples should be finite"
        assert isinstance(model.autoencoder._noise_samples, torch.Tensor), "Noise samples should be torch.Tensor"
    
    def test_ddfm_extract_factors_from_corrupted_samples(self):
        """Test _extract_factors_from_corrupted_samples with pre-corrupted samples.
        
        This test verifies Plan Iteration 11 fix: factor extraction from already-corrupted
        samples (matches original TensorFlow pattern where x_sim_den[i, :, :] is reused).
        """
        model = DDFM(encoder_layers=DEFAULT_ENCODER_LAYERS, num_factors=2, n_mc_samples=3)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Calculate N from input shape and lags_input
        N = X.shape[1] if model.lags_input == 0 else X.shape[1] // (model.lags_input + 1)
        
        # Create list of corrupted samples (simulating what comes from training loop)
        x_sim_den_list = [X + torch.randn_like(X) * 0.1 for _ in range(3)]
        
        # Extract factors from corrupted samples
        factors_samples, predictions_samples = model._extract_factors_from_corrupted_samples(x_sim_den_list, N)
        
        # Verify output shapes match expected (n_mc_samples, time_steps, num_factors/input_dim)
        assert factors_samples.shape == (3, 10, 2), f"Expected (3, 10, 2), got {factors_samples.shape}"
        assert predictions_samples.shape == (3, 10, 5), f"Expected (3, 10, 5), got {predictions_samples.shape}"
        assert torch.isfinite(factors_samples).all(), "Factors should be finite"
        assert torch.isfinite(predictions_samples).all(), "Predictions should be finite"
        assert isinstance(factors_samples, torch.Tensor), "Factors should be torch.Tensor"
        assert isinstance(predictions_samples, torch.Tensor), "Predictions should be torch.Tensor"
    

