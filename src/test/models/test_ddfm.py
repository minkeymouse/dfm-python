"""Tests for models.ddfm module."""

import warnings
import pytest
import numpy as np
import torch
import torch.nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from dfm_python.models.ddfm import DDFM
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.utils.errors import DataError, DataValidationError, ModelNotTrainedError, ModelNotInitializedError
from dfm_python.config.constants import MIN_VARIABLES, MIN_DDFM_TIME_STEPS, DEFAULT_ENCODER_LAYERS
from dfm_python.utils.checkpoint import infer_ddfm_input_dim, infer_input_dim_from_data
from dfm_python.numeric.statistic import diagnose_variance_collapse


class TestDDFM:
    """Test suite for DDFM model."""
    
    def _create_test_dataset(self, num_series=5, time_steps=10, target_scaler=None, random_seed=42):
        """Helper to create DDFMDataset for testing.
        
        Parameters
        ----------
        num_series : int, default 5
            Number of series (columns)
        time_steps : int, default 10
            Number of time steps (rows)
        target_scaler : sklearn scaler class or None, default None
            Scaler for target series (None = no scaling)
        random_seed : int, default 42
            Random seed for deterministic test data
            
        Returns
        -------
        dataset : DDFMDataset
            Test dataset
        """
        # Create test data with deterministic seed
        np.random.seed(random_seed)
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
        torch.manual_seed(42)
        state_dict = {"some.other.key": torch.randn(10, 5)}
        result = infer_ddfm_input_dim(state_dict)
        assert result is None
    
    def test_infer_input_dim_from_encoder_layer(self):
        """Test infer_ddfm_input_dim correctly infers from encoder layer."""
        torch.manual_seed(42)
        state_dict = {
            "autoencoder.encoder.layers.0.weight": torch.randn(32, 64)  # (hidden_dim, input_dim)
        }
        result = infer_ddfm_input_dim(state_dict)
        assert result == 64
    
    def test_infer_input_dim_from_decoder_weight(self):
        """Test infer_ddfm_input_dim correctly infers from decoder weight."""
        torch.manual_seed(42)
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
        """Test DDFM fit() creates all required components (autoencoder, encoder, decoder, optimizer, scheduler)."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        
        assert getattr(model, 'autoencoder', None) is None
        
        model.fit()
        
        assert model.autoencoder is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.optimizer is not None
        assert model.scheduler is not None
        assert model.factors is not None
        assert hasattr(model, '_training_time')
        assert model._training_time > 0
    
    def test_ddfm_decoder_helper_methods(self):
        """Test DDFM decoder helper methods (get_intermediate, get_last_linear_layer)."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        linear_layer = model.decoder.get_last_linear_layer()
        assert isinstance(linear_layer, torch.nn.Linear)
        assert linear_layer in [m for m in model.decoder.modules() if isinstance(m, torch.nn.Linear)]
        
        # Test get_intermediate (may return None for linear decoder)
        intermediate = model.decoder.get_intermediate()
        # For linear decoder, intermediate should be None
        if model.decoder_type == "linear":
            assert intermediate is None
        else:
            assert intermediate is not None
    
    def test_ddfm_variance_diagnostics_attributes(self):
        """Test DDFM variance diagnostics attributes exist after fit (prediction_std, factor_std)."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        assert hasattr(model, 'prediction_std')
        assert hasattr(model, 'factor_std')
    
    def test_ddfm_variance_diagnostics_standardization_detection(self):
        """Test DDFM variance diagnostics detects standardized vs non-standardized data."""
        np.random.seed(42)
        
        dataset_std = self._create_test_dataset(num_series=5, time_steps=50, target_scaler=StandardScaler())
        dataset_std.target_scaler.fit(dataset_std.data.values)
        dataset_std.data = pd.DataFrame(
            dataset_std.target_scaler.transform(dataset_std.data.values),
            columns=dataset_std.data.columns,
            index=dataset_std.data.index
        )
        model_std = DDFM(dataset=dataset_std, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model_std.fit()
        
        dataset_robust = self._create_test_dataset(num_series=5, time_steps=50, target_scaler=RobustScaler())
        model_robust = DDFM(dataset=dataset_robust, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model_robust.fit()
        
        np.random.seed(42)
        prediction_std = np.random.rand(50, 5) * 0.1
        prediction_mean = np.random.rand(50, 5)
        factors_mean = np.random.rand(50, 4)
        
        diagnostics_std = diagnose_variance_collapse(
            prediction_std=prediction_std,
            prediction_mean=prediction_mean,
            factors_mean=factors_mean,
            target_scaler=dataset_std.target_scaler
        )
        assert 'is_standardized' in diagnostics_std
        
        diagnostics_robust = diagnose_variance_collapse(
            prediction_std=prediction_std,
            prediction_mean=prediction_mean,
            factors_mean=factors_mean,
            target_scaler=dataset_robust.target_scaler
        )
        assert 'is_standardized' in diagnostics_robust
    
    def test_ddfm_variance_diagnostics_shape_validation(self):
        """Test DDFM variance diagnostics shape validation handles edge cases."""
        import warnings
        
        np.random.seed(42)
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        np.random.seed(42)
        prediction_mean = np.random.rand(50, 5)
        factors_mean = np.random.rand(50, 4)
        
        diagnostics_invalid = diagnose_variance_collapse(
            prediction_std=[[0.1] * 5] * 50,
            prediction_mean=prediction_mean,
            factors_mean=factors_mean,
            target_scaler=dataset.target_scaler
        )
        assert 'warnings' in diagnostics_invalid
        assert any('Invalid prediction_std type' in w for w in diagnostics_invalid['warnings'])
        
        np.random.seed(42)
        prediction_std_1d = np.random.rand(50)
        diagnostics_1d = diagnose_variance_collapse(
            prediction_std=prediction_std_1d,
            prediction_mean=prediction_mean,
            factors_mean=factors_mean,
            target_scaler=dataset.target_scaler
        )
        assert 'warnings' in diagnostics_1d
        assert any('1D prediction_std array' in w for w in diagnostics_1d['warnings'])
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            prediction_std_empty = np.array([]).reshape(0, 5)
            diagnostics_empty = diagnose_variance_collapse(
                prediction_std=prediction_std_empty,
                prediction_mean=prediction_mean,
                factors_mean=factors_mean,
                target_scaler=dataset.target_scaler
            )
        assert 'warnings' in diagnostics_empty
        assert any('Invalid prediction_std shape' in w for w in diagnostics_empty['warnings'])
        
        np.random.seed(42)
        prediction_std_valid = np.random.rand(50, 5) * 0.1
        diagnostics_valid = diagnose_variance_collapse(
            prediction_std=prediction_std_valid,
            prediction_mean=prediction_mean,
            factors_mean=factors_mean,
            target_scaler=dataset.target_scaler
        )
        assert 'prediction_std_mean' in diagnostics_valid
        assert 'variance_collapse_detected' in diagnostics_valid
    
    def test_ddfm_build_state_space(self):
        """Test DDFM build_state_space() creates state-space parameters."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        assert getattr(model, 'state_space_params', None) is None
        
        model.build_state_space()
        
        assert model.state_space_params is not None
        assert hasattr(model.state_space_params, 'F')
        assert hasattr(model.state_space_params, 'H')
        assert hasattr(model.state_space_params, 'Q')
        assert hasattr(model.state_space_params, 'R')
        assert hasattr(model.state_space_params, 'mu_0')
        assert hasattr(model.state_space_params, 'Sigma_0')
        
        assert model.state_space_params.F.shape[0] == model.state_space_params.F.shape[1]
        assert model.state_space_params.H.shape[1] == model.state_space_params.F.shape[0]
    
    def test_ddfm_predict_requires_state_space(self):
        """Test DDFM predict() requires build_state_space() to be called."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50, target_scaler=StandardScaler())
        dataset.target_scaler.fit(dataset.data.values)
        dataset.data = pd.DataFrame(
            dataset.target_scaler.transform(dataset.data.values),
            columns=dataset.data.columns,
            index=dataset.data.index
        )
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        assert not hasattr(model, 'state_space_params') or getattr(model, 'state_space_params', None) is None
        
        with pytest.raises(ModelNotInitializedError, match="state-space model has not been built"):
            model.predict(horizon=5)
        
        model.build_state_space()
        forecasts = model.predict(horizon=5)
        assert forecasts is not None
        assert isinstance(forecasts, (np.ndarray, tuple))
    
    def test_ddfm_get_result_requires_state_space(self):
        """Test DDFM get_result() requires build_state_space() to be called."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50, target_scaler=StandardScaler())
        dataset.target_scaler.fit(dataset.data.values)
        dataset.data = pd.DataFrame(
            dataset.target_scaler.transform(dataset.data.values),
            columns=dataset.data.columns,
            index=dataset.data.index
        )
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        
        assert not hasattr(model, 'state_space_params') or getattr(model, 'state_space_params', None) is None
        
        with pytest.raises(ModelNotInitializedError, match="state-space model has not been built"):
            model.get_result()
        
        model.build_state_space()
        result = model.get_result()
        assert result is not None
        assert hasattr(result, 'Z')
        assert hasattr(result, 'C')
        assert hasattr(result, 'A')
        assert hasattr(result, 'x_sm')
    
    def test_ddfm_full_workflow(self):
        """Test complete DDFM workflow: initialization -> fit -> build_state_space -> predict -> get_result."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50, target_scaler=StandardScaler())
        dataset.target_scaler.fit(dataset.data.values)
        dataset.data = pd.DataFrame(
            dataset.target_scaler.transform(dataset.data.values),
            columns=dataset.data.columns,
            index=dataset.data.index
        )
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=2)
        
        assert getattr(model, 'autoencoder', None) is None
        assert getattr(model, 'factors', None) is None
        
        model.fit()
        
        assert model.autoencoder is not None
        assert model.factors is not None
        assert model.factors.shape[-1] == DEFAULT_ENCODER_LAYERS[-1]
        
        model.build_state_space()
        
        assert hasattr(model, 'state_space_params')
        assert model.state_space_params is not None
        
        forecasts = model.predict(horizon=5)
        assert forecasts is not None
        
        result = model.get_result()
        assert result is not None
        assert result.Z.shape[-1] == DEFAULT_ENCODER_LAYERS[-1]
        assert result.converged is not None
        assert result.num_iter >= 0
    
    def test_ddfm_update_with_new_data(self):
        """Test DDFM update() method with new data."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), max_iter=1)
        model.fit()
        model.build_state_space()
        
        new_data = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'series_{i}' for i in range(5)]
        )
        new_dataset = DDFMDataset(
            data=new_data,
            time_idx='index',
            target_series=list(new_data.columns),
            target_scaler=dataset.target_scaler
        )
        
        old_factors = model.factors.copy()
        old_shape = old_factors.shape
        model.update(new_dataset)
        
        assert model.factors is not None
        if old_factors.ndim == 2:
            assert model.factors.shape[0] == old_shape[0] + 10
            assert model.factors.shape[1] == old_shape[1]
        else:
            assert model.factors.shape[1] == old_shape[1] + 10
            assert model.factors.shape[2] == old_shape[2]
    
    def test_ddfm_update_raises_error_if_not_trained(self):
        """Test DDFM update() raises error if model not trained."""
        dataset = self._create_test_dataset(num_series=5, time_steps=50)
        model = DDFM(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS))
        
        new_data = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'series_{i}' for i in range(5)]
        )
        new_dataset = DDFMDataset(
            data=new_data,
            time_idx='index',
            target_series=list(new_data.columns)
        )
        
        with pytest.raises(ModelNotTrainedError, match="model has not been trained"):
            model.update(new_dataset)
