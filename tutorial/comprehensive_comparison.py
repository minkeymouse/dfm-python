"""Comprehensive step-by-step comparison to identify divergence points.

This script captures detailed intermediate values at each step and compares
them to identify where numbers diverge greatly.
"""

import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm import DDFM
from dfm_python.config.types import to_numpy, to_tensor
from dfm_python.numeric.estimator import get_idio
from dfm_python.numeric.stability import convergence_checker

def capture_step(name, values_dict, step_data):
    """Capture step data for comparison."""
    step_data[name] = {}
    for key, value in values_dict.items():
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            if isinstance(value, pd.DataFrame):
                value = value.values
            step_data[name][key] = {
                'shape': value.shape,
                'mean': float(np.mean(value)),
                'std': float(np.std(value)),
                'min': float(np.min(value)),
                'max': float(np.max(value)),
                'first_row': value[0].tolist() if len(value.shape) > 1 else float(value[0]),
                'first_3_rows': value[:3].tolist() if len(value.shape) > 1 and len(value) >= 3 else None
            }
        elif isinstance(value, torch.Tensor):
            value_np = to_numpy(value)
            step_data[name][key] = {
                'shape': value_np.shape,
                'mean': float(np.mean(value_np)),
                'std': float(np.std(value_np)),
                'min': float(np.min(value_np)),
                'max': float(np.max(value_np)),
                'first_row': value_np[0].tolist() if len(value_np.shape) > 1 else float(value_np[0]),
            }
        else:
            step_data[name][key] = value
    return step_data

def print_step_summary(name, step_data):
    """Print summary of a step."""
    print(f"\n{'='*80}")
    print(f"STEP: {name}")
    print(f"{'='*80}")
    if name in step_data:
        for key, value in step_data[name].items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"  {key}: mean={value['mean']:.8f}, std={value['std']:.8f}, shape={value['shape']}")
            else:
                print(f"  {key}: {value}")

def main():
    print("="*80)
    print("COMPREHENSIVE STEP-BY-STEP COMPARISON")
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
    
    # Use first 200 rows for detailed analysis
    df_subset = df_scaled.iloc[:200].copy()
    
    print(f"\nData shape: {df_subset.shape}")
    print(f"Data mean: {df_subset.values.mean():.8f}, std: {df_subset.values.std():.8f}")
    
    # Create dataset and model
    target_series = list(df_subset.columns)
    dataset = DDFMDataset(df_subset, target_series=target_series, time_idx=None)
    
    model = DDFM(
        dataset=dataset,
        encoder_size=(16, 4),
        decoder_type='linear',
        seed=3,
        learning_rate=0.005,
        optimizer='Adam',
        n_mc_samples=3,
        max_epoch_pre_train=50,
        max_iter=2,  # Just 2 iterations for detailed analysis
        window_size=100,
        tolerance=0.0005,
        disp=1
    )
    
    # Build autoencoder
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
    
    # Store all step data
    all_steps = {}
    
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
        scheduler=None,
        target_indices=None
    )
    
    # Check BatchNorm after pre-training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        all_steps = capture_step("1_pre_training", {
            'bn_running_mean': bn.running_mean.data.cpu(),
            'bn_running_var': bn.running_var.data.cpu(),
        }, all_steps)
    
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
    
    all_steps = capture_step("2_initial_prediction", {
        'input_data': model.data.values,
        'prediction': initial_pred,
        'prediction_mean': np.mean(initial_pred),
        'prediction_std': np.std(initial_pred),
    }, all_steps)
    
    # Initialize MCMC state
    model.data_denoised = model.data.copy()
    model.data_denoised_interpolated = model._interpolate_dataframe(model.data_denoised)
    model.data_imputed = model.data_denoised_interpolated.copy()
    model.y_actual = model._dataset.data.values
    
    if model.missing_mask.any():
        model.data_imputed.values[model.missing_mask] = initial_pred[model.missing_mask]
    model.eps = model.data_imputed.values - initial_pred
    model.eps = model.eps[:, model.target_indices]
    
    all_steps = capture_step("2_initial_eps", {
        'eps': model.eps,
        'eps_mean': np.mean(model.eps),
        'eps_std': np.std(model.eps),
        'data_imputed': model.data_imputed.values,
    }, all_steps)
    
    print_step_summary("2_initial_prediction", all_steps)
    print_step_summary("2_initial_eps", all_steps)
    
    # STEP 3: First MCMC iteration
    print("\n" + "="*80)
    print("STEP 3: FIRST MCMC ITERATION")
    print("="*80)
    
    # Get idio distribution
    Phi, mu_eps, std_eps = get_idio(model.eps, model._dataset.observed_y)
    all_steps = capture_step("3_idio_distribution", {
        'Phi': Phi,
        'mu_eps': mu_eps,
        'std_eps': std_eps,
    }, all_steps)
    
    # Denoising
    eps_expanded = np.zeros((model.eps.shape[0], model.num_series))
    eps_expanded[:, model.target_indices] = model.eps
    data_before_denoising = model.data_imputed.values.copy()
    model.data_denoised.values[1:] = model.data_imputed.values[1:] - eps_expanded[:-1, :] @ Phi
    model.data_denoised.values[0] = model.data_imputed.values[0]
    
    all_steps = capture_step("3_denoising", {
        'data_before': data_before_denoising,
        'data_after': model.data_denoised.values,
        'Phi': Phi,
        'eps_expanded': eps_expanded,
    }, all_steps)
    
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
    
    all_steps = capture_step("3_mc_samples", {
        'y_tmp': y_tmp.values if hasattr(y_tmp, 'values') else y_tmp,
        'mc0_y_corrupted': to_numpy(autoencoder_datasets[0].y_corrupted),
        'mc0_y_clean': to_numpy(autoencoder_datasets[0].y_clean),
        'mc0_full_input': to_numpy(autoencoder_datasets[0].full_input),
    }, all_steps)
    
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
    
    # Extract factors and predictions
    with torch.no_grad():
        factors_list = [model.encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
        factors_tensor = torch.stack(factors_list, dim=0)
        factors = to_numpy(factors_tensor)
        
        predictions_list = [model.decoder(f) for f in factors_list]
        predictions_tensor = torch.stack(predictions_list, dim=0)
        prediction_iter = to_numpy(predictions_tensor.mean(dim=0))
    
    all_steps = capture_step("3_factors_and_predictions", {
        'factors': factors.mean(axis=0),  # Average over MC samples
        'factors_all_mc': factors,
        'prediction': prediction_iter,
        'prediction_mean': np.mean(prediction_iter),
        'prediction_std': np.std(prediction_iter),
    }, all_steps)
    
    # Compute loss
    delta, loss = convergence_checker(initial_pred, prediction_iter, model.y_actual)
    all_steps = capture_step("3_loss", {
        'loss': loss,
        'delta': delta,
        'initial_pred_mean': np.mean(initial_pred),
        'prediction_mean': np.mean(prediction_iter),
        'target_mean': np.mean(model.y_actual),
    }, all_steps)
    
    print_step_summary("3_idio_distribution", all_steps)
    print_step_summary("3_denoising", all_steps)
    print_step_summary("3_mc_samples", all_steps)
    print_step_summary("3_factors_and_predictions", all_steps)
    print_step_summary("3_loss", all_steps)
    
    # Save all steps
    output_file = project_root / "dfm-python" / "tutorial" / "comprehensive_steps.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_steps, f)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Initial prediction mean: {np.mean(initial_pred):.8f}")
    print(f"Prediction iter0 mean: {np.mean(prediction_iter):.8f}")
    print(f"Target mean: {np.mean(model.y_actual):.8f}")
    print(f"Loss iter0: {loss:.8f}")
    print(f"\nAll step data saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print key values for comparison
    print("\nKEY VALUES FOR COMPARISON:")
    print(f"  Initial prediction - Mean: {all_steps['2_initial_prediction']['prediction_mean']:.8f}")
    print(f"  Initial eps - Mean: {all_steps['2_initial_eps']['eps_mean']:.8f}, Std: {all_steps['2_initial_eps']['eps']['std']:.8f}")
    mu_eps_val = all_steps['3_idio_distribution']['mu_eps']
    std_eps_val = all_steps['3_idio_distribution']['std_eps']
    if isinstance(mu_eps_val, dict):
        print(f"  Idio mu_eps - Mean: {mu_eps_val['mean']:.8f}")
        print(f"  Idio std_eps - Mean: {std_eps_val['mean']:.8f}")
    else:
        print(f"  Idio mu_eps - Mean: {np.mean(mu_eps_val):.8f}")
        print(f"  Idio std_eps - Mean: {np.mean(std_eps_val):.8f}")
    print(f"  Denoised data - Mean: {all_steps['3_denoising']['data_after']['mean']:.8f}")
    print(f"  MC0 y_corrupted - Mean: {all_steps['3_mc_samples']['mc0_y_corrupted']['mean']:.8f}")
    print(f"  Factors - Mean: {all_steps['3_factors_and_predictions']['factors']['mean']:.8f}")
    print(f"  Prediction iter0 - Mean: {all_steps['3_factors_and_predictions']['prediction_mean']:.8f}")
    print(f"  Loss iter0: {all_steps['3_loss']['loss']:.8f}")

if __name__ == "__main__":
    main()

