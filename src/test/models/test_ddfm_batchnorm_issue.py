"""Test for DDFM BatchNorm running_var collapse issue.

This test verifies the BatchNorm running_var collapse problem identified during
DDFM pre-training, where running_var becomes too small (0.008920 vs expected ~1.0),
causing predictions to be amplified by ~10x.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dfm_python import DDFM, DDFMDataset
from dfm_python.encoder.simple_autoencoder import SimpleAutoencoder
from dfm_python.config.constants import DEFAULT_TORCH_DTYPE


@pytest.fixture
def exchange_rate_data():
    """Load exchange rate data for testing."""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "DDFM" / "data" / "exchange_rate.csv",
        Path(__file__).parent.parent.parent.parent.parent / "DDFM" / "data" / "exchange_rate.csv",
        Path(__file__).parent.parent.parent.parent.parent.parent / "DDFM" / "data" / "exchange_rate.csv",
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        # Create synthetic data for testing
        np.random.seed(3)
        dates = pd.date_range('1990-01-01', periods=7588, freq='D')
        data = pd.DataFrame(
            np.random.randn(7588, 8),
            index=dates,
            columns=[f'series_{i}' for i in range(8)]
        )
        df_scaled = (data - data.mean()) / data.std()
        return df_scaled
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Standardize data
    df_scaled = (df - df.mean()) / df.std()
    return df_scaled


@pytest.fixture
def ddfm_model(exchange_rate_data):
    """Create DDFM model for testing."""
    dataset = DDFMDataset(
        data=exchange_rate_data,
        time_idx='index',
        target_series=list(exchange_rate_data.columns),
        target_scaler=None
    )
    
    model = DDFM(
        dataset=dataset,
        encoder_size=(16, 4),
        decoder_type='linear',
        seed=3,
        batch_norm=True,
        activation='relu',
        learning_rate=0.005,
        optimizer='Adam',
        n_mc_samples=10,
        window_size=100,
        max_iter=2,
        tolerance=0.0005,
        disp=1
    )
    return model


class TestBatchNormRunningVarCollapse:
    """Test BatchNorm running_var collapse during pre-training."""
    
    def test_batchnorm_initial_state(self, ddfm_model):
        """Test BatchNorm initial state is correct."""
        # Build model
        ddfm_model.autoencoder = SimpleAutoencoder.build(
            input_dim=ddfm_model.input_dim,
            encoder_size=ddfm_model.encoder_size,
            decoder_size=ddfm_model.decoder_size,
            decoder_type=ddfm_model.decoder_type,
            output_dim=ddfm_model.output_dim,
            activation=ddfm_model.activation,
            use_batch_norm=ddfm_model.batch_norm,
            seed=ddfm_model.initializer_seed
        )
        ddfm_model.encoder = ddfm_model.autoencoder.encoder
        ddfm_model.decoder = ddfm_model.autoencoder.decoder
        ddfm_model.autoencoder.to(ddfm_model.device)
        
        # Check initial BatchNorm state
        for name, module in ddfm_model.encoder.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert module.running_var.mean().item() == pytest.approx(1.0, abs=0.1), \
                    f"Initial running_var should be ~1.0, got {module.running_var.mean().item()}"
                assert module.running_mean.mean().item() == pytest.approx(0.0, abs=0.1), \
                    f"Initial running_mean should be ~0.0, got {module.running_mean.mean().item()}"
    
    def test_batchnorm_collapse_during_pretraining(self, ddfm_model, exchange_rate_data):
        """Test that BatchNorm running_var collapses during pre-training."""
        # Build model
        ddfm_model.autoencoder = SimpleAutoencoder.build(
            input_dim=ddfm_model.input_dim,
            encoder_size=ddfm_model.encoder_size,
            decoder_size=ddfm_model.decoder_size,
            decoder_type=ddfm_model.decoder_type,
            output_dim=ddfm_model.output_dim,
            activation=ddfm_model.activation,
            use_batch_norm=ddfm_model.batch_norm,
            seed=ddfm_model.initializer_seed
        )
        ddfm_model.encoder = ddfm_model.autoencoder.encoder
        ddfm_model.decoder = ddfm_model.autoencoder.decoder
        ddfm_model.autoencoder.to(ddfm_model.device)
        
        # Get initial running_var
        initial_running_var = None
        for name, module in ddfm_model.encoder.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                initial_running_var = module.running_var.mean().item()
                break
        
        assert initial_running_var is not None, "BatchNorm layer not found"
        assert initial_running_var == pytest.approx(1.0, abs=0.1), \
            f"Initial running_var should be ~1.0, got {initial_running_var}"
        
        # Prepare pre-training data
        if ddfm_model._dataset.target_nan_ratio > ddfm_model.min_target_interporate_ratio:
            data_pre_train = ddfm_model._interpolate_dataframe(ddfm_model._dataset.data)
        else:
            data_pre_train = ddfm_model._dataset.data
        
        pre_train_dataset = ddfm_model._dataset.create_pretrain_dataset(
            data_pre_train, device=ddfm_model.device
        )
        
        # Run pre-training
        ddfm_model._build_optimizer()
        ddfm_model.autoencoder.fit(
            dataset=pre_train_dataset,
            epochs=ddfm_model.max_epoch_pre_train,
            batch_size=ddfm_model.window_size,
            learning_rate=ddfm_model.learning_rate,
            optimizer_type=ddfm_model.optimizer_type,
            optimizer=ddfm_model.optimizer,
            scheduler=None,
            target_indices=None
        )
        
        # Check final running_var
        final_running_var = None
        for name, module in ddfm_model.encoder.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                final_running_var = module.running_var.mean().item()
                break
        
        assert final_running_var is not None, "BatchNorm layer not found"
        
        # Verify collapse: running_var should be much smaller than initial
        assert final_running_var < 0.1, \
            f"running_var collapsed to {final_running_var}, expected < 0.1"
        
        # Verify it's too small (should be ~1.0 for standardized input)
        assert final_running_var < 0.5, \
            f"running_var is too small: {final_running_var}, expected ~1.0 for standardized input"
    
    def test_batchnorm_amplification_effect(self, ddfm_model, exchange_rate_data):
        """Test that collapsed BatchNorm amplifies predictions."""
        # Build and pre-train model
        ddfm_model.autoencoder = SimpleAutoencoder.build(
            input_dim=ddfm_model.input_dim,
            encoder_size=ddfm_model.encoder_size,
            decoder_size=ddfm_model.decoder_size,
            decoder_type=ddfm_model.decoder_type,
            output_dim=ddfm_model.output_dim,
            activation=ddfm_model.activation,
            use_batch_norm=ddfm_model.batch_norm,
            seed=ddfm_model.initializer_seed
        )
        ddfm_model.encoder = ddfm_model.autoencoder.encoder
        ddfm_model.decoder = ddfm_model.autoencoder.decoder
        ddfm_model.autoencoder.to(ddfm_model.device)
        
        # Prepare pre-training data
        if ddfm_model._dataset.target_nan_ratio > ddfm_model.min_target_interporate_ratio:
            data_pre_train = ddfm_model._interpolate_dataframe(ddfm_model._dataset.data)
        else:
            data_pre_train = ddfm_model._dataset.data
        
        pre_train_dataset = ddfm_model._dataset.create_pretrain_dataset(
            data_pre_train, device=ddfm_model.device
        )
        
        # Run pre-training
        ddfm_model._build_optimizer()
        ddfm_model.autoencoder.fit(
            dataset=pre_train_dataset,
            epochs=ddfm_model.max_epoch_pre_train,
            batch_size=ddfm_model.window_size,
            learning_rate=ddfm_model.learning_rate,
            optimizer_type=ddfm_model.optimizer_type,
            optimizer=ddfm_model.optimizer,
            scheduler=None,
            target_indices=None
        )
        
        # Get BatchNorm running_var
        running_var = None
        for name, module in ddfm_model.encoder.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                running_var = module.running_var.mean().item()
                break
        
        assert running_var is not None, "BatchNorm layer not found"
        
        # Calculate expected amplification
        # BatchNorm: output = (input - mean) / sqrt(running_var + eps)
        # If running_var is small, output is amplified
        eps = 0.001
        expected_amplification = 1.0 / np.sqrt(running_var + eps)
        
        # Test with standardized input
        test_input = torch.randn(100, 8, device=ddfm_model.device) * 1.0
        
        ddfm_model.autoencoder.eval()
        with torch.no_grad():
            # Get encoder output before BatchNorm
            x = test_input
            for i, layer in enumerate(ddfm_model.encoder.layers):
                x = layer(x)
                if i < len(ddfm_model.encoder.batch_norms):
                    bn = ddfm_model.encoder.batch_norms[i]
                    x_before_bn = x.clone()
                    x = bn(x)
                    x_after_bn = x.clone()
                    
                    # Calculate actual amplification
                    input_std = x_before_bn.std().item()
                    output_std = x_after_bn.std().item()
                    actual_amplification = output_std / input_std if input_std > 0 else 0
                    
                    # Verify amplification is significant
                    assert actual_amplification > 5.0, \
                        f"BatchNorm amplification is {actual_amplification}x, expected > 5.0x " \
                        f"(running_var={running_var:.6f}, expected_amp={expected_amplification:.2f}x)"
                    break


class TestPredictionJump:
    """Test prediction jump from pre-training to MCMC initialization."""
    
    def test_prediction_jump_after_pretraining(self, ddfm_model, exchange_rate_data):
        """Test that predictions jump dramatically after pre-training."""
        # Build model
        ddfm_model.autoencoder = SimpleAutoencoder.build(
            input_dim=ddfm_model.input_dim,
            encoder_size=ddfm_model.encoder_size,
            decoder_size=ddfm_model.decoder_size,
            decoder_type=ddfm_model.decoder_type,
            output_dim=ddfm_model.output_dim,
            activation=ddfm_model.activation,
            use_batch_norm=ddfm_model.batch_norm,
            seed=ddfm_model.initializer_seed
        )
        ddfm_model.encoder = ddfm_model.autoencoder.encoder
        ddfm_model.decoder = ddfm_model.autoencoder.decoder
        ddfm_model.autoencoder.to(ddfm_model.device)
        
        # Prepare pre-training data
        if ddfm_model._dataset.target_nan_ratio > ddfm_model.min_target_interporate_ratio:
            data_pre_train = ddfm_model._interpolate_dataframe(ddfm_model._dataset.data)
        else:
            data_pre_train = ddfm_model._dataset.data
        
        pre_train_dataset = ddfm_model._dataset.create_pretrain_dataset(
            data_pre_train, device=ddfm_model.device
        )
        
        # Run pre-training
        ddfm_model._build_optimizer()
        ddfm_model.autoencoder.fit(
            dataset=pre_train_dataset,
            epochs=ddfm_model.max_epoch_pre_train,
            batch_size=ddfm_model.window_size,
            learning_rate=ddfm_model.learning_rate,
            optimizer_type=ddfm_model.optimizer_type,
            optimizer=ddfm_model.optimizer,
            scheduler=None,
            target_indices=None
        )
        
        # Check prediction on pre-training data
        ddfm_model.autoencoder.eval()
        with torch.no_grad():
            pred_pretrain = ddfm_model.autoencoder(pre_train_dataset.full_input[:100])
            pred_pretrain_mean = pred_pretrain.mean().item()
            pred_pretrain_std = pred_pretrain.std().item()
        
        # Initialize MCMC state
        ddfm_model._build_optimizer()
        ddfm_model.data = ddfm_model._dataset.data.copy()
        ddfm_model.data_denoised = ddfm_model.data.copy()
        ddfm_model.missing_mask = ddfm_model.data.isna().values
        ddfm_model.target_indices = ddfm_model._dataset.target_indices
        if not ddfm_model._dataset.all_columns_are_targets:
            ddfm_model._target_col_tensor = torch.tensor(
                ddfm_model.target_indices, device=ddfm_model.device, dtype=torch.long
            )
        ddfm_model.rng = np.random.RandomState(ddfm_model.initializer_seed)
        
        # Interpolate
        ddfm_model.data_denoised_interpolated = ddfm_model._interpolate_dataframe(
            ddfm_model.data_denoised
        )
        
        # Make initial MCMC prediction
        from dfm_python.config.types import to_tensor
        mcmc_input = to_tensor(
            ddfm_model.data_denoised_interpolated.values,
            dtype=DEFAULT_TORCH_DTYPE,
            device=ddfm_model.device
        )
        
        ddfm_model.autoencoder.eval()
        with torch.no_grad():
            pred_mcmc = ddfm_model.autoencoder(mcmc_input)
            pred_mcmc_mean = pred_mcmc.mean().item()
            pred_mcmc_std = pred_mcmc.std().item()
        
        # Verify inputs are identical
        assert abs(pre_train_dataset.full_input.mean().item() - mcmc_input.mean().item()) < 0.01, \
            "Pre-training and MCMC inputs should have same mean"
        assert abs(pre_train_dataset.full_input.std().item() - mcmc_input.std().item()) < 0.01, \
            "Pre-training and MCMC inputs should have same std"
        
        # Verify prediction jump (this is the problem we're testing)
        prediction_jump = abs(pred_mcmc_mean - pred_pretrain_mean)
        assert prediction_jump > 1.0, \
            f"Prediction should jump significantly, got jump={prediction_jump:.6f}, " \
            f"pretrain_mean={pred_pretrain_mean:.6f}, mcmc_mean={pred_mcmc_mean:.6f}"


class TestStandardization:
    """Test that standardization is correct throughout the pipeline."""
    
    def test_standardization_consistency(self, ddfm_model, exchange_rate_data):
        """Test that standardization is consistent throughout."""
        # Check dataset data
        assert abs(ddfm_model._dataset.data.mean().values[0]) < 0.01, \
            "Dataset data should be standardized (mean ~0)"
        assert abs(ddfm_model._dataset.data.std().values[0] - 1.0) < 0.01, \
            "Dataset data should be standardized (std ~1)"
        
        # Check pre-training data
        if ddfm_model._dataset.target_nan_ratio > ddfm_model.min_target_interporate_ratio:
            data_pre_train = ddfm_model._interpolate_dataframe(ddfm_model._dataset.data)
        else:
            data_pre_train = ddfm_model._dataset.data
        
        assert abs(data_pre_train.mean().values[0]) < 0.01, \
            "Pre-training data should be standardized (mean ~0)"
        assert abs(data_pre_train.std().values[0] - 1.0) < 0.01, \
            "Pre-training data should be standardized (std ~1)"
        
        # Check MCMC initialization data
        ddfm_model.data = ddfm_model._dataset.data.copy()
        ddfm_model.data_denoised = ddfm_model.data.copy()
        ddfm_model.data_denoised_interpolated = ddfm_model._interpolate_dataframe(
            ddfm_model.data_denoised
        )
        
        assert abs(ddfm_model.data_denoised_interpolated.mean().values[0]) < 0.01, \
            "MCMC data should be standardized (mean ~0)"
        assert abs(ddfm_model.data_denoised_interpolated.std().values[0] - 1.0) < 0.01, \
            "MCMC data should be standardized (std ~1)"
        
        # Check that pre-training and MCMC inputs are identical
        pre_train_dataset = ddfm_model._dataset.create_pretrain_dataset(
            data_pre_train, device=ddfm_model.device
        )
        from dfm_python.config.types import to_tensor
        mcmc_input = to_tensor(
            ddfm_model.data_denoised_interpolated.values,
            dtype=DEFAULT_TORCH_DTYPE,
            device=ddfm_model.device
        )
        
        assert abs(pre_train_dataset.full_input.mean().item() - mcmc_input.mean().item()) < 0.01, \
            "Pre-training and MCMC inputs should have same mean"
        assert abs(pre_train_dataset.full_input.std().item() - mcmc_input.std().item()) < 0.01, \
            "Pre-training and MCMC inputs should have same std"

