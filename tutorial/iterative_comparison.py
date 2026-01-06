"""Iterative comparison to identify divergence points across iterations."""

import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm import DDFM
from dfm_python.config.types import to_numpy, to_tensor
from dfm_python.numeric.estimator import get_idio
from dfm_python.numeric.stability import convergence_checker

def print_comparison(name, val1, val2=None, threshold=1e-3):
    """Print value(s) with statistics."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    if isinstance(val1, np.ndarray):
        print(f"Shape: {val1.shape}")
        print(f"Mean: {np.mean(val1):.8f}, Std: {np.std(val1):.8f}")
        print(f"Min: {np.min(val1):.8f}, Max: {np.max(val1):.8f}")
        if len(val1.shape) == 2 and val1.shape[0] > 0:
            print(f"First row: {val1[0]}")
            if val1.shape[0] > 1:
                print(f"Second row: {val1[1]}")
        if val2 is not None:
            diff = np.abs(val1 - val2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rel_diff = max_diff / (np.abs(val2).max() + 1e-10)
            print(f"\nComparison with reference:")
            print(f"  Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}, Rel diff: {rel_diff:.8f}")
            if max_diff > threshold:
                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  ⚠️  LARGE DIFFERENCE at index {max_idx}:")
                print(f"     Val1: {val1[max_idx]:.8f}, Val2: {val2[max_idx]:.8f}")
    else:
        print(f"Value: {val1}")
        if val2 is not None:
            diff = abs(val1 - val2)
            print(f"Difference: {diff:.8f}")
            if diff > threshold:
                print(f"  ⚠️  LARGE DIFFERENCE!")

def main():
    print("="*80)
    print("ITERATIVE COMPARISON - FULL PIPELINE")
    print("="*80)
    
    # Load exchange rate data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "dfm-python" / "data" / "exchange_rate.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_processed = df.dropna(how='all')
    df_processed = df_processed.ffill().bfill()
    
    # Scale data
    mean_z = df_processed.mean().values
    sigma_z = df_processed.std().values
    df_scaled = (df_processed - mean_z) / sigma_z
    
    # Use full dataset for realistic comparison
    print(f"\nData shape: {df_scaled.shape}")
    print(f"Data mean: {df_scaled.values.mean():.8f}, std: {df_scaled.values.std():.8f}")
    
    # Create dataset and model
    target_series = list(df_scaled.columns)
    dataset = DDFMDataset(df_scaled, target_series=target_series, time_idx=None)
    
    model = DDFM(
        dataset=dataset,
        encoder_size=(16, 4),
        decoder_type='linear',
        seed=3,
        learning_rate=0.005,
        optimizer='Adam',
        n_mc_samples=10,
        max_epoch_pre_train=200,
        max_iter=5,  # Just 5 iterations for detailed analysis
        window_size=100,
        tolerance=0.0005,
        disp=1
    )
    
    # Build autoencoder manually for step-by-step analysis
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
    
    # STEP 1: Pre-training
    print("\n" + "="*80)
    print("STEP 1: PRE-TRAINING")
    print("="*80)
    
    data_pre_train = dataset.create_pretrain_dataset(dataset.data, device=model.device)
    model.autoencoder.train()
    model.autoencoder.fit(
        dataset=data_pre_train,
        epochs=model.max_epoch_pre_train,
        batch_size=model.window_size,
        learning_rate=model.learning_rate,
        optimizer_type=model.optimizer_type,
        optimizer=model.optimizer,
        scheduler=None,  # No scheduler during pre-training
        target_indices=None
    )
    
    # Check BatchNorm after pre-training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print_comparison("BatchNorm after pre-training", {
            'running_mean': bn.running_mean.data.cpu().numpy()[:3],
            'running_var': bn.running_var.data.cpu().numpy()[:3],
        })
    
    # STEP 2: Initial prediction
    print("\n" + "="*80)
    print("STEP 2: INITIAL PREDICTION")
    print("="*80)
    
    model.data = model._dataset.data.copy()
    model.missing_mask = model.data.isna().values
    model.target_indices = model._dataset.target_indices
    model.rng = np.random.RandomState(model.initializer_seed)
    
    data_original_tensor = to_tensor(model.data.values, dtype=torch.float32, device=model.device)
    model.autoencoder.train()
    with torch.no_grad():
        initial_pred_tensor = model.autoencoder.forward(data_original_tensor)
        initial_pred = to_numpy(initial_pred_tensor)
    
    print_comparison("Initial prediction", initial_pred)
    print_comparison("Input data", model.data.values)
    
    # Initialize MCMC state
    model.data_denoised = model.data.copy()
    model.data_denoised_interpolated = model._interpolate_dataframe(model.data_denoised)
    model.data_imputed = model.data_denoised_interpolated.copy()
    lags_input = getattr(model, 'lags_input', 0)
    if model._dataset.all_columns_are_targets:
        model.y_actual = model.data.values[lags_input:]
    else:
        model.y_actual = model.data.values[lags_input:, model.target_indices]
    
    if model.missing_mask.any():
        model.data_imputed.values[model.missing_mask] = initial_pred[model.missing_mask]
    model.eps = model.data_imputed.values - initial_pred
    model.eps = model.eps[:, model.target_indices]
    
    print_comparison("Initial eps", model.eps)
    print_comparison("y_actual", model.y_actual)
    
    # Initialize previous predictions
    prediction_prev_iter_full = initial_pred.copy()
    prediction_prev_iter = initial_pred[:, model.target_indices] if not model._dataset.all_columns_are_targets else initial_pred.copy()
    
    # STEP 3: MCMC iterations
    print("\n" + "="*80)
    print("STEP 3: MCMC ITERATIONS")
    print("="*80)
    
    for iter_num in range(model.max_iter):
        print(f"\n--- Iteration {iter_num + 1} ---")
        
        # Get idio distribution
        Phi, mu_eps, std_eps = get_idio(model.eps, model._dataset.observed_y)
        print_comparison(f"Iter {iter_num+1} - Idio Phi (mean)", np.mean(Phi))
        print_comparison(f"Iter {iter_num+1} - Idio mu_eps (mean)", np.mean(mu_eps))
        print_comparison(f"Iter {iter_num+1} - Idio std_eps (mean)", np.mean(std_eps))
        
        # Denoising
        eps_expanded = np.zeros((model.eps.shape[0], model.num_series))
        eps_expanded[:, model.target_indices] = model.eps
        data_before_denoising = model.data_imputed.values.copy()
        model.data_denoised.values[1:] = model.data_imputed.values[1:] - eps_expanded[:-1, :] @ Phi
        model.data_denoised.values[0] = model.data_imputed.values[0]
        
        print_comparison(f"Iter {iter_num+1} - Denoised data (mean)", np.mean(model.data_denoised.values))
        
        # Create MC samples
        X_features_df, y_tmp = model._dataset.split_features_and_targets(model.data_denoised)
        X_features = pd.DataFrame() if X_features_df is None else X_features_df
        
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
        
        print_comparison(f"Iter {iter_num+1} - MC0 y_corrupted (mean)", to_numpy(autoencoder_datasets[0].y_corrupted).mean())
        
        # Train on MC samples
        model.autoencoder.train()
        target_indices = model._target_col_tensor if not model._dataset.all_columns_are_targets else None
        for ae_dataset in autoencoder_datasets:
            model.autoencoder.fit(
                dataset=ae_dataset,
                epochs=1,
                batch_size=model.window_size,
                learning_rate=model.learning_rate,
                optimizer_type=model.optimizer_type,
                optimizer=model.optimizer,
                scheduler=model.scheduler,
                target_indices=target_indices
            )
        
        # Step scheduler after all MC samples
        if model.scheduler is not None:
            model.scheduler.step()
            current_lr = model.optimizer.param_groups[0]['lr']
            print(f"  Learning rate after scheduler step: {current_lr:.8f}")
        
        # Extract factors and predictions
        with torch.no_grad():
            factors_list = [model.encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
            factors_tensor = torch.stack(factors_list, dim=0)
            factors = to_numpy(factors_tensor)
            
            predictions_list = [model.decoder(f) for f in factors_list]
            predictions_tensor = torch.stack(predictions_list, dim=0)
            prediction_iter_full = to_numpy(predictions_tensor.mean(dim=0))
            prediction_iter_target = prediction_iter_full if model._dataset.all_columns_are_targets else prediction_iter_full[:, model.target_indices]
        
        print_comparison(f"Iter {iter_num+1} - Factors (mean)", factors.mean(axis=0).mean())
        print_comparison(f"Iter {iter_num+1} - Prediction (mean)", prediction_iter_full.mean())
        
        # Compute loss
        delta, loss = convergence_checker(prediction_prev_iter, prediction_iter_target, model.y_actual)
        print(f"\n  Iter {iter_num+1} - Loss: {loss:.8f}, Delta: {delta:.8f}")
        print(f"  Prediction mean: {prediction_iter_full.mean():.8f}")
        print(f"  Target mean: {model.y_actual.mean():.8f}")
        print(f"  Prediction std: {prediction_iter_full.std():.8f}")
        print(f"  Target std: {model.y_actual.std():.8f}")
        
        # Update imputed data/eps
        if model.missing_mask.any():
            model.data_imputed.values[model.missing_mask] = prediction_iter_full[model.missing_mask]
        model.eps = model.data_imputed.values - prediction_iter_full
        model.eps = model.eps[:, model.target_indices]
        
        print_comparison(f"Iter {iter_num+1} - Eps (mean)", model.eps.mean())
        
        # Update previous predictions
        prediction_prev_iter_full = prediction_iter_full.copy()
        prediction_prev_iter = prediction_iter_target.copy()
        
        # Check convergence
        if delta < model.tolerance:
            print(f"\n✓ Convergence achieved at iteration {iter_num + 1}")
            break
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Final loss: {loss:.8f}")
    print(f"Final prediction mean: {prediction_iter_full.mean():.8f}")
    print(f"Target mean: {model.y_actual.mean():.8f}")

if __name__ == "__main__":
    main()

