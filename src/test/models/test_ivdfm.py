"""Tests for models.ivdfm module."""

import pytest
import numpy as np
import pandas as pd
import torch
from dfm_python.models.ivdfm import iVDFM
from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
from dfm_python.config.schema.model import iVDFMConfig
from dfm_python.utils.errors import ModelNotTrainedError, DataValidationError, ConfigurationError


class TestiVDFM:
    """Test suite for iVDFM model."""
    
    def _create_test_data(self, T=100, N=5, seed=42):
        """Helper to create test data."""
        np.random.seed(seed)
        return np.random.randn(T, N)
    
    def _create_test_dataset(self, T=100, N=5, context=None, seed=42, window=20):
        """Helper to create test dataset (iVDFM accepts dataset only for fit/update)."""
        data = self._create_test_data(T, N, seed)
        return iVDFMDataset(
            data=data,
            window=window,
            stride=1,
            time_context=1,
            context=context,
        )
    
    def test_ivdfm_initialization_basic(self):
        """Test iVDFM can be initialized with basic parameters."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        assert model.data_dim == 5
        assert model.latent_dim == 3
        assert model.context_dim == 1
        assert model.window == 20
        assert model.innovation_encoder is not None
        assert model.decoder is not None
        assert model.prior_network is not None
        assert model.ssm is not None
    
    def test_ivdfm_initialization_with_config(self):
        """Test iVDFM can be initialized with config."""
        config = iVDFMConfig(
            num_factors=3,
            window=20,
            time_context=1,
            encoder_hidden_dim=100,
            decoder_hidden_dim=100
        )
        
        model = iVDFM(
            data_dim=5,
            config=config
        )
        
        assert model.latent_dim == 3
        assert model.window == 20
        assert model.context_dim is None  # not in config; inferred at fit
    
    def test_ivdfm_initialization_requires_dimensions(self):
        """Test iVDFM can be initialized without dimensions (will be inferred during fit)."""
        # Dimensions can be None and will be inferred during fit
        model = iVDFM(num_factors=3)
        assert model.data_dim is None
        assert model.context_dim is None
        assert model.latent_dim == 3
    
    def test_ivdfm_forward(self):
        """Test iVDFM forward pass."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        batch_size, T, N = 2, 20, 5
        device = next(model.parameters()).device
        y_1T = torch.randn(batch_size, T, N, device=device)
        u_1T = torch.randn(batch_size, T, 1, device=device)
        
        outputs = model.forward(y_1T, u_1T)
        
        assert 'y_pred' in outputs
        assert 'eta' in outputs
        assert 'factors' in outputs
        assert outputs['y_pred'].shape == (batch_size, T, N)
        assert outputs['eta'].shape == (batch_size, T, 3)
        assert outputs['factors'].shape == (batch_size, T, 3)
    
    def test_ivdfm_elbo(self):
        """Test iVDFM ELBO computation."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        batch_size, T, N = 2, 20, 5
        device = next(model.parameters()).device
        y_1T = torch.randn(batch_size, T, N, device=device)
        u_1T = torch.randn(batch_size, T, 1, device=device)
        
        elbo, loss_dict = model.elbo(y_1T, u_1T, N=100)
        
        assert isinstance(elbo, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert 'reconstruction' in loss_dict
        assert 'kl' in loss_dict
    
    def test_ivdfm_fit_basic(self):
        """Test iVDFM fit method with basic data."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,  # Small for testing
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        assert model.training_state is not None
        assert model.factors is not None
        assert model.innovations is not None
        assert hasattr(model.training_state, 'num_iter')
    
    def test_ivdfm_fit_with_dataframe(self):
        """Test iVDFM fit with DataFrame input."""
        T, N = 100, 5
        data = pd.DataFrame(
            self._create_test_data(T, N),
            columns=[f'series_{i}' for i in range(N)]
        )
        dataset = iVDFMDataset(data=data, window=20, stride=1, time_context=1)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        assert model.training_state is not None
        assert model.data_dim == N
    
    def test_ivdfm_fit_with_context_columns(self):
        """Test iVDFM fit with context columns from DataFrame."""
        T = 100
        data = pd.DataFrame(
            np.random.randn(T, 4),
            columns=['target1', 'target2', 'context1', 'context2']
        )
        dataset = iVDFMDataset(
            data=data,
            window=20,
            stride=1,
            time_context=1,
            context=['context1', 'context2'],
        )

        config = iVDFMConfig(
            num_factors=3,
            window=20,
            context=['context1', 'context2'],
        )
        model = iVDFM(
            config=config,
            max_epochs=2,
            batch_size=8,
        )

        model.set_dataset(dataset)
        model.fit()
        
        assert model.data_dim == 2  # Only targets
        assert model.context_dim == 3  # time(1) + context columns(2)
    
    def test_ivdfm_fit_with_context_indices(self):
        """Test iVDFM fit with context as array indices."""
        T, N = 100, 5
        data = self._create_test_data(T, N)
        dataset = iVDFMDataset(
            data=data,
            window=20,
            stride=1,
            time_context=1,
            context=[0, 1],
        )

        config = iVDFMConfig(
            num_factors=3,
            window=20,
            context=[0, 1],
        )
        model = iVDFM(
            config=config,
            max_epochs=2,
            batch_size=8,
        )

        model.set_dataset(dataset)
        model.fit()
        
        assert model.data_dim == 3  # Remaining columns
        assert model.context_dim == 3  # time(1) + context indices(2)
    
    def test_ivdfm_predict_not_trained(self):
        """Test iVDFM predict raises error when not trained."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        with pytest.raises(ModelNotTrainedError):
            model.predict(horizon=5)
    
    def test_ivdfm_predict_basic(self):
        """Test iVDFM predict method."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        predictions = model.predict(horizon=5)
        
        assert predictions.shape == (5, N)
        assert not np.isnan(predictions).any()
    
    def test_ivdfm_predict_with_context_data(self):
        """Test iVDFM predict with custom auxiliary data."""
        T, N = 100, 5
        # Dataset with time_context=2 so model.context_dim becomes 2 and context_data (horizon, 2) matches
        dataset = iVDFMDataset(
            data=self._create_test_data(T, N),
            window=20,
            stride=1,
            time_context=2,
        )
        
        model = iVDFM(
            data_dim=N,
            num_factors=3,
            context_dim=2,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        horizon = 5
        context_future = np.random.randn(horizon, 2)
        predictions = model.predict(horizon=horizon, context_data=context_future)
        
        assert predictions.shape == (horizon, N)
    
    def test_ivdfm_update_not_trained(self):
        """Test iVDFM update raises error when not trained."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        new_data = self._create_test_data(20, 5)
        update_dataset = iVDFMDataset(data=new_data, window=min(20, len(new_data)), stride=1, time_context=1)
        
        with pytest.raises(ModelNotTrainedError):
            model.update(update_dataset)
    
    def test_ivdfm_update_basic(self):
        """Test iVDFM update method."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        initial_factors_shape = model.factors.shape
        
        new_data = self._create_test_data(30, N, seed=123)
        update_dataset = iVDFMDataset(data=new_data, window=min(20, len(new_data)), stride=1, time_context=1)
        model.update(update_dataset)
        
        # Factors should be updated (appended or replaced)
        assert model.factors is not None
        assert model.training_state is not None
    
    def test_ivdfm_get_result_not_trained(self):
        """Test iVDFM get_result raises error when not trained."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            window=20
        )
        
        with pytest.raises(ModelNotTrainedError):
            model.get_result()
    
    def test_ivdfm_get_result(self):
        """Test iVDFM get_result method."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        result = model.get_result()
        
        assert result is not None
        assert hasattr(result, 'factors')
        assert hasattr(result, 'innovations')
        assert hasattr(result, 'reconstructions')
        assert hasattr(result, 'training_elbo')
    
    def test_ivdfm_save_and_load(self, tmp_path):
        """Test iVDFM save and load functionality."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        model.set_dataset(dataset)
        model.fit()
        
        # Save model
        save_path = tmp_path / "test_ivdfm_model.pt"
        model.save(save_path)
        assert save_path.exists()
        
        # Load model
        loaded_model = iVDFM.load(save_path)
        
        assert loaded_model.latent_dim == model.latent_dim
        assert loaded_model.data_dim == model.data_dim
        assert loaded_model.context_dim == model.context_dim

        # Save weights-only (recommended format)
        weights_path = tmp_path / "test_ivdfm_weights.pt"
        model.save(weights_path, weights_only=True)
        assert weights_path.exists()

        # Load weights-only: must provide architecture args
        loaded_weights_model = iVDFM.load(
            weights_path,
            data_dim=N,
            num_factors=3,
            context_dim=1,
            window=20,
        )
        assert loaded_weights_model.latent_dim == model.latent_dim
        assert loaded_weights_model.data_dim == model.data_dim
        assert loaded_weights_model.context_dim == model.context_dim
    
    def test_ivdfm_different_factor_orders(self):
        """Test iVDFM with different factor orders."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        for factor_order in [1, 2, 3]:
            model = iVDFM(
                data_dim=N,
                num_factors=3,
                context_dim=1,
                window=20,
                factor_order=factor_order,
                max_epochs=2,
                batch_size=8
            )
            
            model.set_dataset(dataset)
            model.fit()
            
            assert model.factor_order == factor_order
            assert model.training_state is not None
    
    def test_ivdfm_different_innovation_distributions(self):
        """Test iVDFM with different innovation distributions."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        for dist in ['laplace', 'gaussian']:
            model = iVDFM(
                data_dim=N,
                num_factors=3,
                context_dim=1,
                window=20,
                innovation_distribution=dist,
                max_epochs=2,
                batch_size=8
            )
            
            model.set_dataset(dataset)
            model.fit()
            
            assert model.innovation_distribution == dist
            assert model.training_state is not None
    
    def test_ivdfm_data_dim_mismatch(self):
        """Test iVDFM raises error on data_dim mismatch."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=10,  # Mismatch with actual data
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        with pytest.raises(DataValidationError, match="data_dim mismatch"):
            model.set_dataset(dataset)
            model.fit()
    
    def test_ivdfm_full_workflow(self):
        """Test complete iVDFM workflow: fit -> predict -> update -> get_result."""
        T, N = 100, 5
        dataset = self._create_test_dataset(T=T, N=N)
        
        model = iVDFM(
            data_dim=N,  # Provide data_dim
            num_factors=3,
            context_dim=1,
            window=20,
            max_epochs=2,
            batch_size=8
        )
        
        # Fit
        model.set_dataset(dataset)
        model.fit()
        assert model.training_state is not None
        
        # Predict
        predictions = model.predict(horizon=5)
        assert predictions.shape == (5, N)
        
        # Update
        new_data = self._create_test_data(30, N, seed=123)
        update_dataset = iVDFMDataset(data=new_data, window=min(20, len(new_data)), stride=1, time_context=1)
        model.update(update_dataset)
        
        # Get result
        result = model.get_result()
        assert result is not None
        assert result.factors is not None
