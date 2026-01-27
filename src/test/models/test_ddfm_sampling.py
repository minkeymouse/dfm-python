"""Tests for models.ddfm.sampling module."""

import pytest
import numpy as np
import torch
import pandas as pd
from dfm_python.models.ddfm.sampling import (
    denoise_targets,
    train_on_mc_samples,
    extract_predictions_from_mc_samples,
    check_variance_collapse,
    run_mcmc_iteration
)
from dfm_python.dataset.ddfm_dataset import DDFMDataset, AutoencoderDataset
from dfm_python.encoder.simple_autoencoder import SimpleAutoencoder
from sklearn.preprocessing import StandardScaler


class TestDenoiseTargets:
    """Test suite for denoise_targets function."""
    
    def test_denoise_targets_basic(self):
        """Test denoise_targets denoises target series."""
        T, num_targets = 50, 5
        np.random.seed(42)
        
        # Create test data
        eps = np.random.randn(T, num_targets) * 0.1
        data_imputed = pd.DataFrame(np.random.randn(T, num_targets))
        data_denoised = data_imputed.copy()
        
        # Create dataset
        dataset = DDFMDataset(
            data=data_imputed,
            time_idx='index'
        )
        
        data_denoised_interpolated, Phi, mu_eps, std_eps = denoise_targets(
            eps=eps,
            data_imputed=data_imputed,
            data_denoised=data_denoised,
            dataset=dataset,
            lags_input=0,
            interpolation_method='linear',
            interpolation_limit=10,
            interpolation_limit_direction='both'
        )
        
        assert isinstance(data_denoised_interpolated, pd.DataFrame)
        assert Phi is not None
        assert mu_eps.shape == (num_targets,)
        assert std_eps.shape == (num_targets,)
        assert data_denoised_interpolated.shape == data_imputed.shape


class TestTrainOnMCSamples:
    """Test suite for train_on_mc_samples function."""
    
    def test_train_on_mc_samples(self):
        """Test train_on_mc_samples trains autoencoder on MC samples."""
        T, input_dim, output_dim, num_factors = 50, 10, 5, 4
        n_mc_samples = 3
        
        # Create autoencoder
        autoencoder = SimpleAutoencoder.build(
            input_dim=input_dim,
            encoder_size=[64, 32, num_factors],
            output_dim=output_dim
        )
        
        # Create MC datasets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder.to(device)
        
        autoencoder_datasets = []
        for _ in range(n_mc_samples):
            X = torch.randn(T, input_dim - output_dim, device=device) if input_dim > output_dim else None
            y_corrupted = torch.randn(T, output_dim, device=device)
            y_clean = torch.randn(T, output_dim, device=device)
            autoencoder_datasets.append(AutoencoderDataset(X=X, y_corrupted=y_corrupted, y_clean=y_clean))
        
        # Create optimizer
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)
        
        # Train (should not raise)
        train_on_mc_samples(
            autoencoder=autoencoder,
            autoencoder_datasets=autoencoder_datasets,
            window_size=10,
            learning_rate=0.001,
            optimizer_type='Adam',
            optimizer=optimizer,
            scheduler=scheduler,
            target_indices=None,
            device=device
        )


class TestExtractPredictionsFromMCSamples:
    """Test suite for extract_predictions_from_mc_samples function."""
    
    def test_extract_predictions_from_mc_samples(self):
        """Test extract_predictions_from_mc_samples extracts factors and predictions."""
        T, num_factors, num_targets = 50, 4, 5
        n_mc_samples = 3
        
        # Create encoder and decoder
        autoencoder = SimpleAutoencoder.build(
            input_dim=10,
            encoder_size=[64, 32, num_factors],
            output_dim=num_targets
        )
        encoder = autoencoder.encoder
        decoder = autoencoder.decoder
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder.to(device)
        
        # Create MC datasets
        autoencoder_datasets = []
        for _ in range(n_mc_samples):
            X = torch.randn(T, 5, device=device)
            y_corrupted = torch.randn(T, num_targets, device=device)
            y_clean = torch.randn(T, num_targets, device=device)
            autoencoder_datasets.append(AutoencoderDataset(X=X, y_corrupted=y_corrupted, y_clean=y_clean))
        
        def extract_target_predictions(y_pred_full_tensor):
            return y_pred_full_tensor
        
        factors, y_pred, y_pred_full, y_pred_std, factor_std, factors_mean = extract_predictions_from_mc_samples(
            encoder=encoder,
            decoder=decoder,
            autoencoder_datasets=autoencoder_datasets,
            n_mc_samples=n_mc_samples,
            extract_target_predictions=extract_target_predictions,
            device=device
        )
        
        assert factors.shape == (n_mc_samples, T, num_factors)
        assert y_pred.shape == (T, num_targets)
        assert y_pred_full.shape == (T, num_targets)
        assert y_pred_std.shape == (T, num_targets)
        assert factor_std.shape == (T, num_factors)
        assert factors_mean.shape == (T, num_factors)


class TestCheckVarianceCollapse:
    """Test suite for check_variance_collapse function."""
    
    def test_check_variance_collapse_no_collapse(self):
        """Test check_variance_collapse when no collapse detected."""
        T, num_targets, num_factors = 50, 5, 4
        
        autoencoder = SimpleAutoencoder.build(
            input_dim=10,
            encoder_size=[64, 32, num_factors],
            output_dim=num_targets
        )
        encoder = autoencoder.encoder
        decoder = autoencoder.decoder
        
        np.random.seed(42)
        y_pred_std = np.random.rand(T, num_targets) * 0.5  # Good variance
        y_pred_full = np.random.randn(T, num_targets)
        factors_mean = np.random.randn(T, num_factors)
        y_actual = np.random.randn(T, num_targets)
        
        result = check_variance_collapse(
            y_pred_std=y_pred_std,
            y_pred_full=y_pred_full,
            factors_mean=factors_mean,
            y_actual=y_actual,
            target_scaler=None,
            encoder=encoder,
            decoder=decoder,
            factors_std=np.random.rand(T, num_factors) * 0.5,
            num_iter=10,
            disp=10
        )
        
        # Should return diagnostics dict
        assert result is not None
        assert isinstance(result, dict)
        assert 'variance_collapse_detected' in result


class TestRunMCMCIteration:
    """Test suite for run_mcmc_iteration function."""
    
    def test_run_mcmc_iteration_basic(self):
        """Test run_mcmc_iteration runs complete MCMC step."""
        T, num_targets, num_factors = 50, 5, 4
        n_mc_samples = 3
        
        # Create dataset
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(T, num_targets))
        dataset = DDFMDataset(data=data, time_idx='index')
        
        # Create autoencoder
        autoencoder = SimpleAutoencoder.build(
            input_dim=num_targets,
            encoder_size=[64, 32, num_factors],
            output_dim=num_targets
        )
        encoder = autoencoder.encoder
        decoder = autoencoder.decoder
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder.to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)
        
        # Setup data
        eps = np.random.randn(T, num_targets) * 0.1
        data_imputed = data.copy()
        data_denoised = data.copy()
        y_actual = data.values
        
        def extract_target_predictions(y_pred_full_tensor):
            return y_pred_full_tensor
        
        rng = np.random.RandomState(42)
        
        factors, y_pred, y_pred_full, y_pred_std, factor_std, data_denoised_interpolated, autoencoder_datasets = run_mcmc_iteration(
            eps=eps,
            data_imputed=data_imputed,
            data_denoised=data_denoised,
            dataset=dataset,
            encoder=encoder,
            decoder=decoder,
            autoencoder=autoencoder,
            y_actual=y_actual,
            lags_input=0,
            n_mc_samples=n_mc_samples,
            window_size=10,
            learning_rate=0.001,
            optimizer_type='Adam',
            optimizer=optimizer,
            scheduler=scheduler,
            extract_target_predictions=extract_target_predictions,
            interpolation_method='linear',
            interpolation_limit=10,
            interpolation_limit_direction='both',
            target_scaler=None,
            num_iter=0,
            disp=10,
            device=device,
            rng=rng
        )
        
        assert factors.shape == (n_mc_samples, T, num_factors)
        assert y_pred.shape == (T, num_targets)
        assert y_pred_full.shape == (T, num_targets)
        assert y_pred_std.shape == (T, num_targets)
        assert factor_std.shape == (T, num_factors)
        assert isinstance(data_denoised_interpolated, pd.DataFrame)
        assert len(autoencoder_datasets) == n_mc_samples
