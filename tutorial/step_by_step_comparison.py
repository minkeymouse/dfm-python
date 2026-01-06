"""Step-by-step comparison of PyTorch DDFM with sample data.

This script runs the DDFM algorithm step-by-step and compares intermediate
values to identify where results diverge from expected behavior.
"""

import numpy as np
import pandas as pd
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm import DDFM
from dfm_python.config.constants import DEFAULT_SEED, DEFAULT_DDFM_LEARNING_RATE
from dfm_python.config.types import to_numpy

def create_sample_data(n_samples=50, n_series=5, seed=3):
    """Create simple sample data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate simple correlated time series
    factors = np.random.randn(n_samples, 2)  # 2 factors
    loadings = np.random.randn(2, n_series)  # Loading matrix
    noise = 0.1 * np.random.randn(n_samples, n_series)
    
    data = factors @ loadings + noise
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f'series_{i}' for i in range(n_series)])
    
    return df

def print_step(step_name, data, max_print=5):
    """Print step information."""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        if isinstance(data, pd.DataFrame):
            data = data.values
        print(f"Shape: {data.shape}")
        print(f"Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
        print(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
        if data.size <= max_print * max_print:
            print(f"Data:\n{data}")
        else:
            print(f"First {max_print} rows:\n{data[:max_print]}")
    elif isinstance(data, torch.Tensor):
        data_np = to_numpy(data)
        print(f"Shape: {data_np.shape}")
        print(f"Mean: {np.mean(data_np):.6f}, Std: {np.std(data_np):.6f}")
        print(f"Min: {np.min(data_np):.6f}, Max: {np.max(data_np):.6f}")
    else:
        print(f"Value: {data}")

def main():
    print("="*80)
    print("STEP-BY-STEP DDFM COMPARISON")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(n_samples=50, n_series=5, seed=3)
    print_step("1. Original Data", data)
    
    # Create dataset (need time_idx for DDFMDataset - use None to use index)
    time_idx_dates = pd.date_range(start='2000-01-01', periods=len(data), freq='D')
    data.index = time_idx_dates
    # time_idx should be None (use index) or a column name string
    dataset = DDFMDataset(data, target_series=list(data.columns), time_idx=None)
    print_step("2. Dataset Creation", dataset.data)
    print(f"Target shape: {dataset.target_shape}")
    print(f"Feature shape: {dataset.feature_shape}")
    print(f"All columns are targets: {dataset.all_columns_are_targets}")
    
    # Create model
    model = DDFM(
        dataset=dataset,
        encoder_size=(8, 2),  # Smaller for testing
        decoder_type='linear',
        seed=3,
        learning_rate=DEFAULT_DDFM_LEARNING_RATE,
        n_mc_samples=3,  # Fewer MC samples for testing
        max_epoch_pre_train=10,  # Fewer epochs for testing
        max_iter=5,  # Fewer iterations for testing
        window_size=50,
    )
    
    print_step("3. Model Created", None)
    print(f"Input dim: {model.input_dim}, Output dim: {model.output_dim}")
    
    # Build model (this creates autoencoder) - need to call fit() which builds it
    # For step-by-step, we'll manually build it
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
    model.autoencoder.to(model.device)  # Move to device
    model._build_optimizer()
    print(f"Autoencoder built: {model.autoencoder is not None}")
    
    # Pre-training
    print("\n" + "="*80)
    print("PRE-TRAINING")
    print("="*80)
    
    # Get pre-training data
    data_pre_train = dataset.create_pretrain_dataset(dataset.data, device=model.device)
    print_step("4. Pre-training Data (y_clean)", data_pre_train.y_clean)
    if data_pre_train.X is not None:
        print_step("4. Pre-training Data (X)", data_pre_train.X)
    
    # Pre-train
    model.autoencoder.train()
    # Pre-train with smaller batch size for testing
    batch_size_pre = min(model.window_size, len(data_pre_train))
    model.autoencoder.fit(
        dataset=data_pre_train,
        epochs=model.max_epoch_pre_train,
        batch_size=batch_size_pre,
        learning_rate=model.learning_rate,
        optimizer_type=model.optimizer_type,
        optimizer=model.optimizer,
        scheduler=None,
        target_indices=None
    )
    
    # Check BatchNorm stats after pre-training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print(f"\nBatchNorm after pre-training:")
        print(f"  running_mean: {bn.running_mean.data[:3].cpu().numpy()}")
        print(f"  running_var: {bn.running_var.data[:3].cpu().numpy()}")
        print(f"  training: {bn.training}")
    
    # Initialize data and attributes (matching fit() method)
    model.data = model._dataset.data.copy()
    model.missing_mask = model.data.isna().values
    model.target_indices = model._dataset.target_indices
    if not model._dataset.all_columns_are_targets:
        model._target_col_tensor = torch.tensor(model.target_indices, device=model.device, dtype=torch.long)
    model.rng = np.random.RandomState(model.initializer_seed)
    
    # Initial prediction
    print("\n" + "="*80)
    print("INITIAL PREDICTION")
    print("="*80)
    
    model.autoencoder.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(model.data.values).to(
            dtype=torch.float32, device=model.device
        )
        initial_pred = model.autoencoder.forward(data_tensor)
        initial_pred_np = to_numpy(initial_pred)
    
    print_step("5. Initial Prediction", initial_pred_np)
    
    # Initialize MCMC state
    model.data_denoised = model.data.copy()
    model.data_denoised_interpolated = model._interpolate_dataframe(model.data_denoised)
    model.data_imputed = model.data_denoised_interpolated.copy()
    
    # Set y_actual
    model.y_actual = model._dataset.data.values
    
    # Update imputed and eps
    model.missing_mask = model.data.isna().values
    model.target_indices = model._dataset.target_indices
    if model.missing_mask.any():
        model.data_imputed.values[model.missing_mask] = initial_pred_np[model.missing_mask]
    model.eps = model.data.values - initial_pred_np
    model.eps = model.eps[:, model.target_indices]
    
    print_step("6. Initial Eps", model.eps)
    print(f"Eps mean: {np.mean(model.eps):.6f}, std: {np.std(model.eps):.6f}")
    
    # MCMC Loop - First iteration
    print("\n" + "="*80)
    print("MCMC ITERATION 0")
    print("="*80)
    
    from dfm_python.numeric.estimator import get_idio
    
    # Get idio distr
    Phi, mu_eps, std_eps = get_idio(model.eps, model._dataset.observed_y)
    print_step("7. Idio Distribution", {
        'Phi': Phi,
        'mu_eps': mu_eps,
        'std_eps': std_eps
    })
    
    # Denoising step
    eps_expanded = np.zeros((model.eps.shape[0], model.num_series))
    eps_expanded[:, model.target_indices] = model.eps
    model.data_denoised.values[1:] = model.data_imputed.values[1:] - eps_expanded[:-1, :] @ Phi
    print_step("8. Denoised Data", model.data_denoised.values)
    
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
    
    print_step("9. MC Sample 0 (y_corrupted)", to_numpy(autoencoder_datasets[0].y_corrupted))
    print_step("9. MC Sample 0 (y_clean)", to_numpy(autoencoder_datasets[0].y_clean))
    
    # Train on first MC sample
    print("\n" + "="*80)
    print("TRAINING ON MC SAMPLE 0")
    print("="*80)
    
    model.autoencoder.train()
    print(f"Model training mode: {model.autoencoder.training}")
    
    # Check BatchNorm before training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print(f"\nBatchNorm before training:")
        print(f"  running_mean: {bn.running_mean.data[:3].cpu().numpy()}")
        print(f"  running_var: {bn.running_var.data[:3].cpu().numpy()}")
        print(f"  training: {bn.training}")
    
    # Get initial weights
    initial_weights = model.autoencoder.encoder.layers[0].weight.data.clone()
    
    model.autoencoder.fit(
        dataset=autoencoder_datasets[0],
        epochs=1,
        batch_size=model.window_size,
        learning_rate=model.learning_rate,
        optimizer_type=model.optimizer_type,
        optimizer=model.optimizer,
        scheduler=None,
        target_indices=None
    )
    
    # Check weights after training
    final_weights = model.autoencoder.encoder.layers[0].weight.data.clone()
    weight_change = torch.norm(final_weights - initial_weights).item()
    print(f"\nWeight change (L2 norm): {weight_change:.6f}")
    
    # Check BatchNorm after training
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print(f"\nBatchNorm after training:")
        print(f"  running_mean: {bn.running_mean.data[:3].cpu().numpy()}")
        print(f"  running_var: {bn.running_var.data[:3].cpu().numpy()}")
        print(f"  training: {bn.training}")
    
    # Extract factors (in training mode)
    print("\n" + "="*80)
    print("FACTOR EXTRACTION")
    print("="*80)
    
    print(f"Model training mode: {model.autoencoder.training}")
    
    with torch.no_grad():
        factors = model.encoder(autoencoder_datasets[0].full_input)
        factors_np = to_numpy(factors)
    
    print_step("10. Factors", factors_np)
    
    # Check BatchNorm during factor extraction
    if hasattr(model.autoencoder.encoder, 'batch_norms') and len(model.autoencoder.encoder.batch_norms) > 0:
        bn = model.autoencoder.encoder.batch_norms[0]
        print(f"\nBatchNorm during factor extraction:")
        print(f"  running_mean: {bn.running_mean.data[:3].cpu().numpy()}")
        print(f"  running_var: {bn.running_var.data[:3].cpu().numpy()}")
        print(f"  training: {bn.training}")
        print(f"  Using batch stats: {bn.training}")
    
    # Compute prediction
    with torch.no_grad():
        prediction = model.decoder(factors)
        prediction_np = to_numpy(prediction)
    
    print_step("11. Prediction", prediction_np)
    
    # Compute loss
    from dfm_python.numeric.stability import convergence_checker
    
    # For first iteration, use initial prediction as previous
    delta, loss = convergence_checker(initial_pred_np, prediction_np, model.y_actual)
    print(f"\nLoss: {loss:.6f}, Delta: {delta:.6f}")
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Initial prediction mean: {np.mean(initial_pred_np):.6f}")
    print(f"Final prediction mean: {np.mean(prediction_np):.6f}")
    print(f"Target mean: {np.mean(model.y_actual):.6f}")
    print(f"Loss: {loss:.6f}")

if __name__ == "__main__":
    main()

