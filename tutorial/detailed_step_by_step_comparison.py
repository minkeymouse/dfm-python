"""Detailed step-by-step comparison of PyTorch DDFM to identify divergence points.

This script captures all intermediate values at each step and compares them
to identify where numbers diverge greatly.
"""

import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm import DDFM
from dfm_python.config.constants import DEFAULT_SEED, DEFAULT_DDFM_LEARNING_RATE
from dfm_python.config.types import to_numpy, to_tensor
from dfm_python.numeric.estimator import get_idio
from dfm_python.numeric.stability import convergence_checker

def print_value(name, value, max_print=3):
    """Print value with detailed statistics."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        if isinstance(value, pd.DataFrame):
            value = value.values
        print(f"Shape: {value.shape}")
        print(f"Mean: {np.mean(value):.8f}, Std: {np.std(value):.8f}")
        print(f"Min: {np.min(value):.8f}, Max: {np.max(value):.8f}")
        print(f"First row: {value[0]}")
        if value.size <= 100:
            print(f"Full data:\n{value}")
    elif isinstance(value, dict):
        for k, v in value.items():
            print(f"{k}:")
            print_value(f"  {k}", v, max_print)
    elif isinstance(value, torch.Tensor):
        value_np = to_numpy(value)
        print_value(name, value_np, max_print)
    else:
        print(f"Value: {value}")

def compare_values(name, val1, val2, threshold=1e-3):
    """Compare two values and report differences."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name}")
    print(f"{'='*80}")
    
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        if val1.shape != val2.shape:
            print(f"⚠️  SHAPE MISMATCH: {val1.shape} vs {val2.shape}")
            return True
        
        diff = np.abs(val1 - val2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = max_diff / (np.abs(val2).max() + 1e-10)
        
        print(f"Val1 - Mean: {np.mean(val1):.8f}, Std: {np.std(val1):.8f}")
        print(f"Val2 - Mean: {np.mean(val2):.8f}, Std: {np.std(val2):.8f}")
        print(f"Difference - Max: {max_diff:.8f}, Mean: {mean_diff:.8f}, Rel: {rel_diff:.8f}")
        
        if max_diff > threshold:
            print(f"⚠️  LARGE DIFFERENCE! Max diff: {max_diff:.8f} (threshold: {threshold})")
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   Max diff at index {max_idx}:")
            print(f"   Val1: {val1[max_idx]:.8f}")
            print(f"   Val2: {val2[max_idx]:.8f}")
            print(f"   Val1 first row: {val1[0]}")
            print(f"   Val2 first row: {val2[0]}")
            return True
        else:
            print(f"✓ Values match within threshold")
            return False
    else:
        diff = abs(val1 - val2) if isinstance(val1, (int, float)) else None
        print(f"Val1: {val1}")
        print(f"Val2: {val2}")
        if diff is not None:
            print(f"Difference: {diff:.8f}")
            if diff > threshold:
                print(f"⚠️  LARGE DIFFERENCE! Diff: {diff:.8f}")
                return True
        return False

def main():
    print("="*80)
    print("DETAILED STEP-BY-STEP DDFM COMPARISON")
    print("="*80)
    
    # Load exchange rate data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "dfm-python" / "data" / "exchange_rate.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_processed = df.dropna(how='all')
    df_processed = df_processed.ffill().bfill()  # Use new API
    
    # Scale data
    mean_z = df_processed.mean().values
    sigma_z = df_processed.std().values
    df_scaled = (df_processed - mean_z) / sigma_z
    
    print(f"\nData shape: {df_scaled.shape}")
    print(f"Data mean: {df_scaled.values.mean():.8f}, std: {df_scaled.values.std():.8f}")
    
    # Use smaller subset for detailed analysis
    df_subset = df_scaled.iloc[:200].copy()
    
    # Create dataset
    target_series = list(df_subset.columns)
    dataset = DDFMDataset(df_subset, target_series=target_series, time_idx=None)
    
    # Create model
    model = DDFM(
        dataset=dataset,
        encoder_size=(16, 4),
        decoder_type='linear',
        seed=3,
        learning_rate=0.005,
        optimizer='Adam',
        n_mc_samples=3,  # Fewer for faster testing
        max_epoch_pre_train=50,  # Fewer for faster testing
        max_iter=3,  # Just 3 iterations for detailed analysis
        window_size=100,
        tolerance=0.0005,
        disp=1
    )
    
    # Build autoencoder (normally done in fit(), but we need it for step-by-step)
    from dfm_python.encoder.simple_autoencoder import SimpleAutoencoder
    model.autoencoder = SimpleAutoencoder.build(
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        encoder_size=model.encoder_size,
        decoder_size=model.decoder_size,
        decoder_type=model.decoder_type,
        activation=model.activation,
        seed=model.initializer_seed
    )
    model.encoder = model.autoencoder.encoder
    model.decoder = model.autoencoder.decoder
    model.autoencoder.to(model.device)
    model._build_optimizer()
    
    # Store all intermediate values
    intermediate_values = {}
    
    print("\n" + "="*80)
    print("STEP 1: PRE-TRAINING")
    print("="*80)
    
    # Pre-training data
    data_pre_train = dataset.create_pretrain_dataset(dataset.data, device=model.device)
    print_value("Pre-training y_clean", data_pre_train.y_clean)
    intermediate_values['pretrain_y_clean'] = to_numpy(data_pre_train.y_clean)
    
    # Pre-train
    model.autoencoder.train()
    model.autoencoder.fit(
        dataset=data_pre_train,
        epochs=model.max_epoch_pre_train,
        batch_size=model.window_size,
        learning_rate=model.learning_rate,
        optimizer_type=model.optimizer_type,
        optimizer=model.optimizer,
        scheduler=None,
        target_indices=None
    )
    
    # Check BatchNorm after pre-training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print(f"\nBatchNorm after pre-training:")
        print(f"  running_mean (first 3): {bn.running_mean.data[:3].cpu().numpy()}")
        print(f"  running_var (first 3): {bn.running_var.data[:3].cpu().numpy()}")
        intermediate_values['pretrain_bn_running_mean'] = bn.running_mean.data.cpu().numpy()
        intermediate_values['pretrain_bn_running_var'] = bn.running_var.data.cpu().numpy()
    
    print("\n" + "="*80)
    print("STEP 2: INITIAL PREDICTION")
    print("="*80)
    
    # Initialize data
    model.data = model._dataset.data.copy()
    model.missing_mask = model.data.isna().values
    model.target_indices = model._dataset.target_indices
    model.rng = np.random.RandomState(model.initializer_seed)
    
    # Initial prediction
    data_original_tensor = to_tensor(model.data.values, dtype=torch.float32, device=model.device)
    print_value("Input to initial prediction", model.data.values)
    intermediate_values['initial_pred_input'] = model.data.values.copy()
    
    model.autoencoder.train()  # Use training mode
    with torch.no_grad():
        initial_pred_tensor = model.autoencoder.forward(data_original_tensor)
        initial_pred = to_numpy(initial_pred_tensor)
    
    print_value("Initial prediction", initial_pred)
    intermediate_values['initial_prediction'] = initial_pred.copy()
    
    # Initialize MCMC state
    model.data_denoised = model.data.copy()
    model.data_denoised_interpolated = model._interpolate_dataframe(model.data_denoised)
    model.data_imputed = model.data_denoised_interpolated.copy()
    model.y_actual = model._dataset.data.values
    
    # Update imputed and eps
    if model.missing_mask.any():
        model.data_imputed.values[model.missing_mask] = initial_pred[model.missing_mask]
    model.eps = model.data.values - initial_pred
    model.eps = model.eps[:, model.target_indices]
    
    print_value("Initial eps", model.eps)
    intermediate_values['initial_eps'] = model.eps.copy()
    
    print("\n" + "="*80)
    print("STEP 3: FIRST MCMC ITERATION")
    print("="*80)
    
    # Get idio distribution
    Phi, mu_eps, std_eps = get_idio(model.eps, model._dataset.observed_y)
    print_value("Idio Phi", Phi)
    print_value("Idio mu_eps", mu_eps)
    print_value("Idio std_eps", std_eps)
    intermediate_values['idio_Phi'] = Phi.copy()
    intermediate_values['idio_mu_eps'] = mu_eps.copy()
    intermediate_values['idio_std_eps'] = std_eps.copy()
    
    # Denoising step
    eps_expanded = np.zeros((model.eps.shape[0], model.num_series))
    eps_expanded[:, model.target_indices] = model.eps
    model.data_denoised.values[1:] = model.data_imputed.values[1:] - eps_expanded[:-1, :] @ Phi
    model.data_denoised.values[0] = model.data_imputed.values[0]
    
    print_value("Denoised data (first 5 rows)", model.data_denoised.values[:5])
    intermediate_values['denoised_data'] = model.data_denoised.values.copy()
    
    # Create MC samples
    X_features_df, y_tmp = model._dataset.split_features_and_targets(model.data_denoised)
    X_features = pd.DataFrame() if X_features_df is None else X_features_df
    
    print_value("y_tmp (for corruption)", y_tmp.values if hasattr(y_tmp, 'values') else y_tmp)
    intermediate_values['y_tmp'] = y_tmp.values.copy() if hasattr(y_tmp, 'values') else y_tmp.copy()
    
    autoencoder_datasets = model._dataset.create_autoencoder_datasets_list(
        n_mc_samples=model.n_mc_samples,
        mu_eps=mu_eps,
        std_eps=std_eps,
        X=X_features,
        y_tmp=y_tmp,
        y_actual=model.y_actual,
        rng=model.rng,
        device=model.device
    )
    
    print_value("MC Sample 0 - y_corrupted", to_numpy(autoencoder_datasets[0].y_corrupted))
    intermediate_values['mc0_y_corrupted'] = to_numpy(autoencoder_datasets[0].y_corrupted).copy()
    
    # Train on MC samples
    model.autoencoder.train()
    for i, ae_dataset in enumerate(autoencoder_datasets):
        model.autoencoder.fit(
            dataset=ae_dataset,
            epochs=1,
            batch_size=model.window_size,
            learning_rate=model.learning_rate,
            optimizer_type=model.optimizer_type,
            optimizer=model.optimizer,
            scheduler=None,
            target_indices=None
        )
    
    # Extract factors
    with torch.no_grad():
        factors_list = [model.encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
        factors_tensor = torch.stack(factors_list, dim=0)
        factors = to_numpy(factors_tensor)
    
    print_value("Factors (MC averaged)", factors.mean(axis=0))
    intermediate_values['factors'] = factors.copy()
    
    # Compute predictions
    with torch.no_grad():
        predictions_list = [model.decoder(f) for f in factors_list]
        predictions_tensor = torch.stack(predictions_list, dim=0)
        prediction_iter = to_numpy(predictions_tensor.mean(dim=0))
    
    print_value("Prediction after iteration 0", prediction_iter)
    intermediate_values['prediction_iter0'] = prediction_iter.copy()
    
    # Compute loss
    delta, loss = convergence_checker(initial_pred, prediction_iter, model.y_actual)
    print(f"\nLoss after iteration 0: {loss:.8f}, Delta: {delta:.8f}")
    intermediate_values['loss_iter0'] = loss
    intermediate_values['delta_iter0'] = delta
    
    # Save intermediate values for comparison
    output_file = project_root / "dfm-python" / "tutorial" / "pytorch_intermediate_values.pkl"
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(intermediate_values, f)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Initial prediction mean: {np.mean(initial_pred):.8f}")
    print(f"Prediction iter0 mean: {np.mean(prediction_iter):.8f}")
    print(f"Target mean: {np.mean(model.y_actual):.8f}")
    print(f"Loss iter0: {loss:.8f}")
    print(f"\nIntermediate values saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

