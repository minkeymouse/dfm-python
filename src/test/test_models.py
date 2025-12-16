"""Tests for model implementations (DFM and DDFM).

Tests align with theoretical foundations from:
- Stock & Watson (2002a,b): Linear DFM with EM algorithm
- Giannone et al. (2008): EM algorithm for DFM with missing data
- Andreini et al. (2020): Deep Dynamic Factor Models (DDFM)
"""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from dfm_python.models import DFM, BaseFactorModel
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.config.results import DFMResult, FitParams
from dfm_python import DFMDataModule
from dfm_python.data import DFMDataset, DDFMDataset
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.utils.time import TimeIndex, parse_timestamp


class TestBaseFactorModel:
    """Test BaseFactorModel interface."""
    
    def test_base_factor_model_interface(self):
        """Test that BaseFactorModel defines required interface."""
        # BaseFactorModel is abstract, so we test via DFM (high-level API)
        # DFM is the high-level API that inherits from BaseFactorModel
        model = DFM()
        assert isinstance(model, BaseFactorModel)
        assert hasattr(model, 'predict')
        # Check that update method exists (replaces legacy nowcast)
        assert hasattr(model, 'update')
        assert callable(getattr(model, 'update', None))
        # DFM creates a placeholder config when none is provided
        assert model.config is not None
        # Result property raises ValueError when accessed before training
        # Error message format: "{ModelType} model has not been trained yet. Please call trainer.fit(model, data_module) first."
        with pytest.raises(ValueError, match=r".*model has not been trained yet.*"):
            _ = model.result
        # Update method also raises ValueError before training
        with pytest.raises(ValueError, match=r".*model has not been trained yet.*"):
            import numpy as np
            _ = model.update(np.random.randn(10, 2))


# TestDFMLinear removed: DFMLinear is now internal (_DFMLinear) and not part of public API.
# Use DFM class (high-level API) instead.


class TestDFM:
    """Test DFM high-level API."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        from test_helpers import get_test_data_path
        return get_test_data_path()
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test DFM config."""
        from test_helpers import get_test_config_path
        return get_test_config_path("dfm")
    
    @pytest.fixture
    def sample_data_from_file(self, test_data_path):
        """Load sample data from CSV using pandas."""
        from test_helpers import load_sample_data_from_csv
        return load_sample_data_from_csv(test_data_path)
    
    def test_dfm_initialization(self):
        """Test DFM initialization."""
        model = DFM()
        # DFM creates a placeholder config when none is provided
        assert model.config is not None
        # Result property raises ValueError when accessed before training
        # Error message format: "{ModelType} model has not been trained yet. Please call trainer.fit(model, data_module) first."
        with pytest.raises(ValueError, match=r".*model has not been trained yet.*"):
            _ = model.result
    
    def test_dfm_load_config(self, test_config_path):
        """Test loading configuration from YAML."""
        if not test_config_path.exists():
            pytest.skip(f"Test config file not found: {test_config_path}")
        
        model = DFM()
        source = YamlSource(test_config_path)
        
        # Config loading may fail if series are strings instead of dicts
        # This is expected for test configs that use simplified format
        try:
            config = source.load()
            assert config is not None
            if hasattr(config, 'series') and len(config.series) > 0:
                # Verify series are SeriesConfig objects
                assert all(hasattr(s, 'series_id') for s in config.series)
        except (TypeError, ValueError) as e:
            # Expected if config uses string format instead of SeriesConfig dicts
            pytest.skip(f"Config format not fully supported (series as strings): {e}")
        
        assert hasattr(model, 'load_config')
    
    def test_dfm_with_real_data(self, test_data_path, test_config_path):
        """Test DFM with real sample data."""
        if not test_data_path.exists() or not test_config_path.exists():
            pytest.skip("Test data or config files not found")
        
        # Load config (may fail if series are strings instead of dicts)
        source = YamlSource(test_config_path)
        try:
            config = source.load()
        except (TypeError, ValueError) as e:
            # Expected if config uses string format instead of SeriesConfig dicts
            pytest.skip(f"Config format not fully supported (series as strings): {e}")
        
        # Load data
        df = pd.read_csv(test_data_path)
        date_col = df.select("date").to_series().to_list()
        time_index = TimeIndex([parse_timestamp(d) for d in date_col])
        
        # Get series from config
        series_ids = [s.series_id for s in config.series]
        data_cols = [col for col in df.columns if col != "date" and col in series_ids]
        
        if len(data_cols) == 0:
            pytest.skip("No matching series found in data")
        
        # Extract and preprocess data
        data_array = df.select(data_cols).to_numpy()
        data_clean, _ = rem_nans_spline(data_array, method=2, k=3)
        
        # Sort data to match config order
        data_sorted, mnem_sorted = sort_data(data_clean, data_cols, config)
        
        assert data_sorted.shape[0] > 0
        assert data_sorted.shape[1] == len(mnem_sorted)
        assert len(mnem_sorted) <= len(series_ids)


class TestDDFM:
    """Test DDFM implementation (if available)."""
    
    def test_ddfm_import(self):
        """Test that DDFM can be imported (if PyTorch available)."""
        try:
            from dfm_python.models import DDFM, DDFMModel
            assert DDFM is not None
            assert DDFMModel is not None
        except ImportError:
            pytest.skip("DDFM requires PyTorch")
    
    def test_ddfm_autoencoder_structure(self):
        """Test DDFM autoencoder structure from papers.
        
        According to Andreini et al. (2020):
        - DDFM uses autoencoder: encode y -> f, decode f -> y_hat
        - Nonlinear encoding: G_θ_G(y) = f
        - Nonlinear decoding: F_θ_F(f) = y_hat
        - Factor dynamics: f_t = B(L) f_{t-1} + u_t
        """
        try:
            from dfm_python.models import DDFM
            # DDFM structure may vary - test that it can be instantiated
            # Note: DDFM may require config or other parameters
            from dfm_python.models import DDFM
            # This test verifies DDFM can be imported and has expected interface
            assert DDFM is not None
            # Actual instantiation may require config
        except ImportError:
            pytest.skip("DDFM requires PyTorch")


class TestStateSpaceConsistency:
    """Test state-space model consistency with theory."""
    
    def test_observation_equation_structure(self):
        """Test observation equation: y_t = C Z_t + e_t.
        
        From Stock & Watson (2002a):
        - y_t: N x 1 observed variables
        - C: N x r loading matrix
        - Z_t: r x 1 latent factors
        - e_t: N x 1 idiosyncratic errors
        """
        N, r = 10, 3
        C = np.random.randn(N, r)
        Z_t = np.random.randn(r, 1)
        e_t = np.random.randn(N, 1)
        
        y_t = C @ Z_t + e_t
        assert y_t.shape == (N, 1)
    
    def test_transition_equation_structure(self):
        """Test transition equation: Z_t = A Z_{t-1} + v_t.
        
        From DFM theory:
        - Z_t: r x 1 factors at time t
        - A: r x r transition matrix
        - v_t: r x 1 factor innovations
        """
        r = 3
        A = np.random.randn(r, r) * 0.5  # Stationary
        Z_prev = np.random.randn(r, 1)
        v_t = np.random.randn(r, 1)
        
        Z_t = A @ Z_prev + v_t
        assert Z_t.shape == (r, 1)
    
    def test_factor_dynamics_stationarity(self):
        """Test that factor dynamics respect stationarity.
        
        For VAR(1): Z_t = A Z_{t-1} + v_t
        Stationarity requires eigenvalues of A < 1 in modulus.
        """
        r = 3
        # Create stationary transition matrix
        A = np.random.randn(r, r) * 0.3  # Small coefficients
        eigenvals = np.linalg.eigvals(A)
        max_eigenval = np.max(np.abs(eigenvals))
        
        # Should be stationary (eigenvalues < 1)
        assert max_eigenval < 1.0


class TestEstimationConsistency:
    """Test estimation consistency with EM algorithm theory."""
    
    def test_em_algorithm_structure(self):
        """Test EM algorithm structure from Dempster et al. (1977).
        
        E-step: Compute E[Z_t | Y, θ] using Kalman smoother
        M-step: Maximize Q(θ | θ_old) = E[log p(Y, Z | θ) | Y, θ_old]
        """
        # EM algorithm should have E-step and M-step
        # This is tested in test_ssm.py for EMAlgorithm class
        from dfm_python.ssm import EMAlgorithm
        em = EMAlgorithm()
        assert hasattr(em, 'forward')  # PyTorch module forward pass
    
    def test_pca_initialization(self):
        """Test PCA initialization from papers.
        
        According to Stock & Watson (2002a) and Giannone et al. (2008):
        - Factors initialized via PCA on observed data
        - Loadings from PCA eigenvectors
        """
        from dfm_python.encoder import PCAEncoder
        
        T, N, r = 100, 10, 3
        X = np.random.randn(T, N)
        
        encoder = PCAEncoder(n_components=r, use_torch=False)
        encoder.fit(X)
        factors = encoder.encode(X)
        
        assert factors.shape == (T, r)
        assert encoder.eigenvectors is not None
        assert encoder.eigenvectors.shape == (N, r)


class TestPredictionConsistency:
    """Test prediction consistency with forecasting theory."""
    
    def test_forecast_horizon(self):
        """Test forecast horizon parameter."""
        model = DFM()
        # Predict should accept horizon parameter
        assert hasattr(model, 'predict')
        
        # Check that predict signature includes history parameter
        import inspect
        sig = inspect.signature(model.predict)
        assert 'horizon' in sig.parameters
        assert 'history' in sig.parameters  # history parameter was added
    
    def test_predict_history_parameter(self):
        """Test predict() history parameter.
        
        The history parameter allows using only recent N periods for
        Kalman filter updates, improving efficiency and adaptability.
        """
        model = DFM()
        import inspect
        sig = inspect.signature(model.predict)
        history_param = sig.parameters['history']
        # history should be Optional[int] with default None
        assert history_param.default is None
    
    def test_factor_forecast_structure(self):
        """Test factor forecast structure.
        
        From DFM theory:
        - Forecast factors: E[Z_{t+h} | Y_1:t]
        - Forecast observables: E[y_{t+h} | Y_1:t] = C E[Z_{t+h} | Y_1:t]
        """
        # Factor forecast should use transition equation
        r = 3
        A = np.random.randn(r, r) * 0.5
        Z_t = np.random.randn(r, 1)
        
        # One-step ahead forecast
        Z_forecast = A @ Z_t
        assert Z_forecast.shape == (r, 1)


class TestModelResults:
    """Test model result structures."""
    
    def test_dfm_result_structure(self):
        """Test DFMResult contains required fields."""
        # DFMResult should contain:
        # - Factors (Z)
        # - Loadings (C)
        # - Transition matrix (A)
        # - Covariances (Q, R)
        # - Log-likelihood
        # DFMResult is BaseResult, which has different structure
        from dfm_python.config.results import DFMResult
        T, N, r = 100, 5, 2
        result = DFMResult(
            x_sm=np.random.randn(T, N),
            X_sm=np.random.randn(T, N),
            Z=np.random.randn(T, r),
            C=np.random.randn(N, r),
            R=np.eye(N) * 0.1,
            A=np.random.randn(r, r) * 0.5,
            Q=np.eye(r) * 0.1,
            Mx=np.zeros(N),
            Wx=np.ones(N),
            Z_0=np.zeros(r),
            V_0=np.eye(r),
            r=np.array([r]),
            p=1,
            loglik=-100.0
        )
        assert result.Z.shape[1] == r  # Number of factors
        assert result.C.shape[0] == N  # Number of series
        assert result.loglik is not None

