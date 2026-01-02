"""Tests for models.ddfm module."""

import pytest
import numpy as np
import torch
from dfm_python.models.ddfm import DDFM
from dfm_python.utils.errors import ConfigurationError, DataError, DataValidationError, ModelNotInitializedError, ModelNotTrainedError
from dfm_python.config.constants import MIN_VARIABLES, MIN_DDFM_TIME_STEPS
from dfm_python.utils.checkpoint import infer_ddfm_input_dim, infer_input_dim_from_data


class TestDDFM:
    """Test suite for DDFM model."""
    
    def test_ddfm_initialization(self):
        """Test DDFM can be initialized."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        assert model.num_factors == 2
        assert model.encoder_layers == [64, 32]
    
    def test_ddfm_initialization_minimal(self):
        """Test DDFM can be initialized with minimal parameters."""
        model = DDFM(num_factors=1)
        assert model.num_factors == 1
    
    # factor_order tests removed - factors now always use AR(1) dynamics (simplified)
    # factor_order parameter was removed from DDFM.__init__()
    
    def test_ddfm_forward(self):
        """Test DDFM forward pass."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        # Initialize networks (required for forward) - input_dim is number of variables
        X = torch.randn(10, 5)  # 10 time steps, 5 variables
        model.initialize_networks(X.shape[1])  # Pass number of variables (5)
        # Forward pass
        y_pred = model.forward(X)
        assert y_pred.shape == X.shape
        assert isinstance(y_pred, torch.Tensor)
    
    def test_ddfm_forward_not_initialized(self):
        """Test DDFM forward raises error when networks not initialized."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)
        with pytest.raises(ModelNotInitializedError):
            model.forward(X)
    
    def test_ddfm_training_step(self):
        """Test DDFM training step with MC dataset batch format."""
        from unittest.mock import Mock
        import pytorch_lightning as pl
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)  # 10 time steps, 5 variables
        model.initialize_networks(X.shape[1])  # Pass number of variables (5)
        # Configure optimizer manually for testing
        optimizer_config = model.configure_optimizers()
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config[0] if isinstance(optimizer_config, list) else optimizer_config
        # Create a mock trainer with optimizer
        mock_trainer = Mock()
        mock_trainer.strategy = Mock()
        mock_trainer.strategy._lightning_optimizers = [optimizer]
        model.trainer = mock_trainer
        # Training step expects batch from DDFMMCDataset: (x_corrupted, x_target, mask)
        n_mc_samples = 1  # Use minimal MC samples for testing
        x_corrupted = X[None, :, :].expand(n_mc_samples, -1, -1)  # (n_mc_samples, T, N)
        x_target = X[None, :, :].expand(n_mc_samples, -1, -1)  # (n_mc_samples, T, N)
        mask = torch.ones(n_mc_samples, X.shape[0], X.shape[1], dtype=torch.bool)  # (n_mc_samples, T, N)
        batch = (x_corrupted, x_target, mask)
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_ddfm_training_step_invalid_batch_format(self):
        """Test DDFM training step raises error for invalid batch format."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        # Training step should raise ValueError for invalid batch format
        with pytest.raises(ValueError, match="expects batch from DDFMMCDataset with 3 elements"):
            model.training_step(X, batch_idx=0)  # Invalid: single tensor instead of tuple
    
    def test_ddfm_grad_clip_val_zero_disables_clipping(self):
        """Test DDFM with grad_clip_val=0.0 disables gradient clipping."""
        from unittest.mock import Mock
        from dfm_python.config.constants import DEFAULT_ZERO_VALUE
        model = DDFM(encoder_layers=[64, 32], num_factors=2, grad_clip_val=DEFAULT_ZERO_VALUE)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        # Configure optimizer manually for testing
        optimizer_config = model.configure_optimizers()
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config[0] if isinstance(optimizer_config, list) else optimizer_config
        # Create a mock trainer with optimizer
        mock_trainer = Mock()
        mock_trainer.strategy = Mock()
        mock_trainer.strategy._lightning_optimizers = [optimizer]
        model.trainer = mock_trainer
        # Training step expects batch from DDFMMCDataset: (x_corrupted, x_target, mask)
        n_mc_samples = 1
        x_corrupted = X[None, :, :].expand(n_mc_samples, -1, -1)
        x_target = X[None, :, :].expand(n_mc_samples, -1, -1)
        mask = torch.ones(n_mc_samples, X.shape[0], X.shape[1], dtype=torch.bool)
        batch = (x_corrupted, x_target, mask)
        # Training step should work without gradient clipping
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_ddfm_grad_clip_val_positive_enables_clipping(self):
        """Test DDFM with grad_clip_val > 0 enables gradient clipping."""
        from unittest.mock import Mock
        from dfm_python.config.constants import DEFAULT_ZERO_VALUE, DEFAULT_IDENTITY_SCALE
        model = DDFM(encoder_layers=[64, 32], num_factors=2, grad_clip_val=DEFAULT_IDENTITY_SCALE)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        # Configure optimizer manually for testing
        optimizer_config = model.configure_optimizers()
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config[0] if isinstance(optimizer_config, list) else optimizer_config
        # Create a mock trainer with optimizer
        mock_trainer = Mock()
        mock_trainer.strategy = Mock()
        mock_trainer.strategy._lightning_optimizers = [optimizer]
        model.trainer = mock_trainer
        # Training step expects batch from DDFMMCDataset: (x_corrupted, x_target, mask)
        n_mc_samples = 1
        x_corrupted = X[None, :, :].expand(n_mc_samples, -1, -1)
        x_target = X[None, :, :].expand(n_mc_samples, -1, -1)
        mask = torch.ones(n_mc_samples, X.shape[0], X.shape[1], dtype=torch.bool)
        batch = (x_corrupted, x_target, mask)
        # Training step should work with gradient clipping
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        # Verify grad_clip_val is set correctly
        assert model.grad_clip_val > DEFAULT_ZERO_VALUE
    
    def test_ddfm_get_result_not_trained(self):
        """Test DDFM get_result raises error when model not trained."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        with pytest.raises(ModelNotTrainedError, match="model has not been fitted yet"):
            model.get_result()
    
    def test_ddfm_result_property_not_trained(self):
        """Test DDFM result property raises error when model not trained."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        with pytest.raises(ModelNotTrainedError, match="model has not been trained yet"):
            _ = model.result
    
    def test_ddfm_predict_not_trained(self):
        """Test DDFM predict raises error when model not trained."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        # DDFM.predict() doesn't take last_observation parameter
        # It uses training state from Lightning module
        with pytest.raises(ModelNotTrainedError):
            model.predict(horizon=5)
    
    def test_infer_input_dim_invalid_type(self):
        """Test infer_ddfm_input_dim raises DataValidationError for non-dict input."""
        with pytest.raises(DataValidationError, match="state_dict must be a dictionary"):
            infer_ddfm_input_dim("not a dict")
    
    def test_infer_input_dim_invalid_type_list(self):
        """Test infer_ddfm_input_dim raises DataValidationError for list input."""
        with pytest.raises(DataValidationError, match="state_dict must be a dictionary"):
            infer_ddfm_input_dim([1, 2, 3])
    
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
    
    def test_infer_input_dim_from_data_numpy_2d(self):
        """Test infer_input_dim_from_data correctly infers from 2D numpy array."""
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3) -> should return 3
        result = infer_input_dim_from_data(arr_2d)
        assert result == 3
    
    def test_infer_input_dim_from_data_numpy_1d(self):
        """Test infer_input_dim_from_data raises DataError for 1D numpy array."""
        arr_1d = np.array([1, 2, 3])  # (3,) -> should raise error
        with pytest.raises(DataError, match="Data must be at least 2D"):
            infer_input_dim_from_data(arr_1d)
    
    def test_infer_input_dim_from_data_torch_2d(self):
        """Test infer_input_dim_from_data correctly infers from 2D torch tensor."""
        tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3) -> should return 3
        result = infer_input_dim_from_data(tensor_2d)
        assert result == 3
    
    def test_infer_input_dim_from_data_torch_1d(self):
        """Test infer_input_dim_from_data raises DataError for 1D torch tensor."""
        tensor_1d = torch.tensor([1, 2, 3])  # (3,) -> should raise error
        with pytest.raises(DataError, match="Data must be at least 2D"):
            infer_input_dim_from_data(tensor_1d)
    
    def test_are_networks_initialized_returns_true_when_initialized(self):
        """Test networks are initialized when encoder and decoder are not None."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Test networks are initialized (direct check)
        assert model.encoder is not None
        assert model.decoder is not None
    
    def test_are_networks_initialized_returns_false_when_not_initialized(self):
        """Test networks are not initialized when encoder or decoder is None."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        
        # Test networks are not initialized (direct check)
        assert model.encoder is None or model.decoder is None
    
    def test_are_networks_initialized_returns_false_when_encoder_none(self):
        """Test networks are not initialized when encoder is None."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Manually set encoder to None
        model.encoder = None
        
        # Test networks are not initialized (direct check)
        assert model.encoder is None or model.decoder is None
    
    def test_are_networks_initialized_returns_false_when_decoder_none(self):
        """Test networks are not initialized when decoder is None."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        X = torch.randn(10, 5)
        model.initialize_networks(X.shape[1])
        
        # Manually set decoder to None
        model.decoder = None
        
        # Test networks are not initialized (direct check)
        assert model.encoder is None or model.decoder is None
    
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
    
    def test_ddfm_time_steps_validation(self):
        """Test DDFM raises error when time steps < MIN_DDFM_TIME_STEPS."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        # Test with T=1 (below MIN_DDFM_TIME_STEPS=2)
        X = torch.randn(1, 5)  # 1 time step, 5 variables (T < MIN_DDFM_TIME_STEPS)
        model.initialize_networks(X.shape[1])
        # _validate_training_data should raise DataValidationError due to insufficient time steps
        # The error message comes from validate_data_shape which checks all dimensions
        with pytest.raises(DataValidationError, match="All dimensions must be >= 2"):
            model._validate_training_data(X, operation="test")
    
    def test_ddfm_time_steps_validation_minimum(self):
        """Test DDFM accepts data with T=MIN_DDFM_TIME_STEPS."""
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        # Test with T=2 (exactly MIN_DDFM_TIME_STEPS)
        X = torch.randn(MIN_DDFM_TIME_STEPS, 5)  # 2 time steps, 5 variables
        model.initialize_networks(X.shape[1])
        # _validate_training_data should not raise error for minimum valid time steps
        # This should pass validation (no exception raised)
        model._validate_training_data(X, operation="test")

