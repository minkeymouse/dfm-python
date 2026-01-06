#!/usr/bin/env python3
"""Run PyTorch DDFM with comprehensive diagnostics for comparison with original."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import time
import torch

# Add dfm-python to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "dfm-python" / "src"))

from dfm_python import DDFM, DDFMDataset

def collect_diagnostics(model, iter_num, prediction_target=None, eps=None, gradients=None):
    """Collect comprehensive diagnostic metrics at each iteration."""
    diagnostics = {
        'iter': iter_num,
        'loss': getattr(model, 'loss_now', None),
        'eps_mean': float(np.mean(eps)) if eps is not None else None,
        'eps_std': float(np.std(eps)) if eps is not None else None,
        'eps_shape': eps.shape if eps is not None else None,
        'eps_min': float(np.min(eps)) if eps is not None else None,
        'eps_max': float(np.max(eps)) if eps is not None else None,
        'factors_mean': float(np.mean(model.factors)) if model.factors is not None else None,
        'factors_std': float(np.std(model.factors)) if model.factors is not None else None,
        'factors_shape': model.factors.shape if model.factors is not None else None,
        'factors_min': float(np.min(model.factors)) if model.factors is not None else None,
        'factors_max': float(np.max(model.factors)) if model.factors is not None else None,
    }
    
    # Collect model weights
    if model.autoencoder is not None:
        try:
            encoder_weights = []
            decoder_weights = []
            
            # Encoder weights
            for name, param in model.encoder.named_parameters():
                if param.requires_grad:
                    weight_data = param.data.cpu().numpy()
                    encoder_weights.append({
                        'layer_name': name,
                        'weight_mean': float(np.mean(weight_data)),
                        'weight_std': float(np.std(weight_data)),
                        'weight_min': float(np.min(weight_data)),
                        'weight_max': float(np.max(weight_data)),
                        'weight_shape': list(weight_data.shape),
                    })
                    # Collect gradients if available
                    if param.grad is not None:
                        grad_data = param.grad.cpu().numpy()
                        encoder_weights[-1].update({
                            'grad_mean': float(np.mean(grad_data)),
                            'grad_std': float(np.std(grad_data)),
                            'grad_norm': float(torch.norm(param.grad).item()),
                        })
            
            # Decoder weights
            for name, param in model.decoder.named_parameters():
                if param.requires_grad:
                    weight_data = param.data.cpu().numpy()
                    decoder_weights.append({
                        'layer_name': name,
                        'weight_mean': float(np.mean(weight_data)),
                        'weight_std': float(np.std(weight_data)),
                        'weight_min': float(np.min(weight_data)),
                        'weight_max': float(np.max(weight_data)),
                        'weight_shape': list(weight_data.shape),
                    })
                    # Collect gradients if available
                    if param.grad is not None:
                        grad_data = param.grad.cpu().numpy()
                        decoder_weights[-1].update({
                            'grad_mean': float(np.mean(grad_data)),
                            'grad_std': float(np.std(grad_data)),
                            'grad_norm': float(torch.norm(param.grad).item()),
                        })
            
            diagnostics['encoder_weights'] = encoder_weights
            diagnostics['decoder_weights'] = decoder_weights
        except Exception as e:
            diagnostics['weight_error'] = str(e)
    
    # Collect data statistics
    if hasattr(model, 'X_imputed'):
        diagnostics['X_imputed_mean'] = float(np.mean(model.X_imputed.values))
        diagnostics['X_imputed_std'] = float(np.std(model.X_imputed.values))
    if hasattr(model, 'X_denoised_interpolated'):
        diagnostics['X_denoised_interpolated_mean'] = float(np.mean(model.X_denoised_interpolated.values))
        diagnostics['X_denoised_interpolated_std'] = float(np.std(model.X_denoised_interpolated.values))
    if hasattr(model, 'y_actual'):
        diagnostics['y_actual_mean'] = float(np.mean(model.y_actual))
        diagnostics['y_actual_std'] = float(np.std(model.y_actual))
        diagnostics['y_actual_shape'] = model.y_actual.shape
    
    if prediction_target is not None:
        diagnostics['prediction_mean'] = float(np.mean(prediction_target))
        diagnostics['prediction_std'] = float(np.std(prediction_target))
        diagnostics['prediction_shape'] = prediction_target.shape
        diagnostics['prediction_min'] = float(np.min(prediction_target))
        diagnostics['prediction_max'] = float(np.max(prediction_target))
        # Prediction statistics per series
        if prediction_target.ndim == 2:
            diagnostics['prediction_per_series_mean'] = [float(x) for x in np.mean(prediction_target, axis=0)]
            diagnostics['prediction_per_series_std'] = [float(x) for x in np.std(prediction_target, axis=0)]
    
    # Collect optimizer state if available
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            lr = model.optimizer.param_groups[0]['lr']
            diagnostics['learning_rate'] = float(lr)
        except:
            pass
    
    # Collect scheduler state if available
    if hasattr(model, 'scheduler') and model.scheduler is not None:
        try:
            diagnostics['scheduler_last_epoch'] = getattr(model.scheduler, 'last_epoch', None)
        except:
            pass
    
    return diagnostics

def main():
    # Load exchange rate data
    data_path = project_root / "dfm-python" / "tutorial" / "data" / "exchange_rate.csv"
    if not data_path.exists():
        data_path = project_root / "DDFM" / "data" / "exchange_rate.csv"
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Standardize data (matching original TensorFlow approach)
    print("\n" + "="*60)
    print("Preprocessing Data")
    print("="*60)
    data_mean = df.mean()
    data_std = df.std(ddof=1)  # Match pandas default
    df_scaled = (df - data_mean) / data_std
    print(f"Scaled data mean (should be ~0): {df_scaled.mean().values[:3]}")
    print(f"Scaled data std (should be ~1): {df_scaled.std().values[:3]}")
    
    # Create dataset
    print("\n" + "="*60)
    print("Creating Dataset")
    print("="*60)
    dataset = DDFMDataset(
        data=df_scaled,
        time_idx='index',  # Use index as time_idx
        target_series=list(df.columns),  # All series are targets
        target_scaler=None  # Already scaled
    )
    print(f"Dataset created: data shape={dataset.data.shape}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating DDFM Model")
    print("="*60)
    model = DDFM(
        dataset=dataset,
        encoder_size=(16, 4),
        decoder_type="linear",
        seed=3,
        activation='relu',
        learning_rate=0.005,
        optimizer='Adam',
        n_mc_samples=10,
        window_size=100,
        max_iter=200,
        tolerance=0.0005,
        disp=10
    )
    
    # Collect diagnostics
    all_diagnostics = []
    
    # Note: The diagnostics tutorial uses fit() which consolidates build_model(),
    # pre_train(), and train() into one method. For detailed diagnostics during
    # training, you would need to modify the fit() method or use a custom training loop.
    # For this tutorial, we'll use fit() and collect diagnostics after training.
    
    print("\n[1-3] Training model (builds model, pre-trains, and trains)...")
    model.fit()
    
    # Collect diagnostics after training
    final_diag = collect_diagnostics(model, -1)
    final_diag['stage'] = 'after_training'
    all_diagnostics.append(final_diag)
    print(f"   Training completed: encoder={model.encoder}, decoder={model.decoder}")
    
    # Note: fit() now handles build_model(), pre_train(), and train() in one method.
    # For per-iteration diagnostics, you would need to modify the fit() method
    # or create a custom training loop. This simplified version collects diagnostics
    # only after training completes.
    
    # Save diagnostics
    results_dir = project_root / "dfm-python" / "tutorial" / "results_diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics_file = results_dir / f"pytorch_diagnostics_{timestamp}.pkl"
    
    results = {
        'diagnostics': all_diagnostics,
        'final_loss': getattr(model, 'loss_now', None),
        'num_iterations': getattr(model, '_num_iter', 0),
        'converged': getattr(model, '_converged', False),
        'data_shape': df.shape,
        'columns': list(df.columns),
        'timestamp': timestamp
    }
    
    with open(diagnostics_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"Diagnostics saved to: {diagnostics_file}")
    final_loss = getattr(model, 'loss_now', None)
    num_iter = getattr(model, '_num_iter', 0)
    converged = getattr(model, '_converged', False)
    print(f"Final loss: {final_loss:.6f}" if final_loss is not None else "Final loss: N/A")
    print(f"Iterations: {num_iter}")
    print(f"Converged: {converged}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary of key metrics:")
    final_diag = all_diagnostics[-1] if all_diagnostics else {}
    if 'prediction_mean' in final_diag and final_diag['prediction_mean'] is not None:
        print(f"  Final prediction: mean={final_diag['prediction_mean']:.6f}, std={final_diag.get('prediction_std', 0):.6f}")
    if final_loss is not None:
        print(f"  Final loss: {final_loss:.6f}")
    if 'eps_mean' in final_diag and final_diag['eps_mean'] is not None:
        print(f"  Final eps: mean={final_diag['eps_mean']:.6f}, std={final_diag.get('eps_std', 0):.6f}")
    if 'factors_mean' in final_diag and final_diag['factors_mean'] is not None:
        print(f"  Final factors: mean={final_diag['factors_mean']:.6f}, std={final_diag.get('factors_std', 0):.6f}")
    
    return results

if __name__ == "__main__":
    main()

