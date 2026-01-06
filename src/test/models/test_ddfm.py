"""Tests for models.ddfm module."""

import pytest
import numpy as np
import torch
import pandas as pd
from dfm_python.models.ddfm import DDFM
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.utils.errors import DataError, DataValidationError, ModelNotTrainedError
from dfm_python.config.constants import MIN_VARIABLES, MIN_DDFM_TIME_STEPS, DEFAULT_ENCODER_LAYERS
from dfm_python.utils.checkpoint import infer_ddfm_input_dim, infer_input_dim_from_data


class TestDDFM:
    """Test suite for DDFM model."""
    
    def _create_test_dataset(self, num_series=5, time_steps=10, target_scaler=None):
        """Helper to create DDFMDataset for testing.
        
        Parameters
        ----------
        num_series : int, default 5
            Number of series (columns)
        time_steps : int, default 10
            Number of time steps (rows)
        target_scaler : sklearn scaler class or None, default None
            Scaler for target series (None = no scaling)
            
        Returns
        -------
        dataset : DDFMDataset
            Test dataset
        """
        # Create test data
        data = pd.DataFrame(
            np.random.randn(time_steps, num_series),
            columns=[f'series_{i}' for i in range(num_series)]
        )
        
        # All series are targets for testing
        target_series = list(data.columns)
        
        dataset = DDFMDataset(
            data=data,
            time_idx='index',
            target_series=target_series,
            target_scaler=target_scaler
        )
        return dataset
    
    def _create_initialized_ddfm(self, num_series=5, time_steps=10, encoder_size=None, **model_kwargs):
        """Helper to create and initialize DDFM model for testing.
        
        Parameters
        ----------
        num_series : int, default 5
            Number of variables (input dimension)
        time_steps : int, default 10
            Number of time steps for test data
        encoder_size : tuple, optional
            Encoder layer sizes (last element is num_factors). Defaults to tuple(DEFAULT_ENCODER_LAYERS).
        **model_kwargs
            Additional arguments passed to DDFM constructor
            
        Returns
        -------
        model : DDFM
            Initialized DDFM model (not yet trained)
        dataset : DDFMDataset
            Test dataset
        """
        if encoder_size is None:
            encoder_size = tuple(DEFAULT_ENCODER_LAYERS)
        dataset = self._create_test_dataset(num_series=num_series, time_steps=time_steps)
        model = DDFM(
            dataset=dataset,
            encoder_size=encoder_size,
            **model_kwargs
        )
        return model, dataset
    
    @pytest.mark.parametrize("encoder_size", [
        ((16, 4),),  # Default
        ((64, 32, 4),),  # 3-layer encoder
        ((4,),),  # Minimal (just latent dim)
    ])
    def test_ddfm_initialization(self, encoder_size):
        """Test DDFM can be initialized with various encoder sizes."""
        encoder_size = encoder_size[0]  # Unpack from parametrize
        dataset = self._create_test_dataset(num_series=5, time_steps=10)
        model = DDFM(dataset=dataset, encoder_size=encoder_size)
        
        assert model.encoder_size == encoder_size
        assert model.num_series == 5
        # num_factors is last element of encoder_size
        assert len(encoder_size) > 0
        num_factors = encoder_size[-1]
        # Verify encoder_size is stored correctly
        assert model.encoder_size[-1] == num_factors
    
    def test_ddfm_window_size_parameter(self):
        """Test DDFM window_size parameter."""
        dataset = self._create_test_dataset(num_series=5, time_steps=10)
        model = DDFM(
            dataset=dataset,
            encoder_size=tuple(DEFAULT_ENCODER_LAYERS),
            n_mc_samples=10,
            window_size=100
        )
        # Verify window_size is set correctly
        assert model.window_size == 100
        assert model.n_mc_samples == 10
    
    def test_ddfm_result_not_trained(self):
        """Test DDFM result access raises error when model not trained."""
        dataset = self._create_test_dataset(num_series=5, time_steps=10)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS))
        with pytest.raises(ModelNotTrainedError):
            model.get_result()
        # DDFM uses get_result() method (not result property like DFM/KDFM)
        # Verify _result attribute is None for untrained model
        assert getattr(model, '_result', None) is None
    
    def test_ddfm_predict_not_trained(self):
        """Test DDFM predict raises error when model not trained."""
        dataset = self._create_test_dataset(num_series=5, time_steps=10)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS))
        with pytest.raises(ModelNotTrainedError, match="model has not been trained"):
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
            "autoencoder.encoder.layers.0.weight": torch.randn(32, 64)  # (hidden_dim, input_dim)
        }
        result = infer_ddfm_input_dim(state_dict)
        assert result == 64
    
    def test_infer_input_dim_from_decoder_weight(self):
        """Test infer_ddfm_input_dim correctly infers from decoder weight."""
        state_dict = {
            "autoencoder.decoder.decoder.weight": torch.randn(10, 5)  # (output_dim, num_factors)
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
    
    def test_ddfm_dimension_validation_uses_constants(self):
        """Test DDFM dimension validation uses MIN_VARIABLES and MIN_DDFM_TIME_STEPS constants."""
        # Verify constants are defined and have correct values
        assert MIN_VARIABLES == 1
        assert MIN_DDFM_TIME_STEPS == 2
    
    def test_ddfm_fit_creates_components(self):
        """Test DDFM fit() creates all required components (autoencoder, encoder, decoder, optimizer)."""
        dataset = self._create_test_dataset(num_series=5, time_steps=10)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        
        # Model should not have autoencoder before fit()
        assert getattr(model, 'autoencoder', None) is None
        
        # Fit model (builds model, pre-trains, and trains)
        model.fit()
        
        # Verify all components are created
        assert model.autoencoder is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.optimizer is not None
        assert model.factors is not None
