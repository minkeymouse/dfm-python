"""Side-by-side comparison of PyTorch and TensorFlow DDFM implementations.

This script runs both implementations with identical data and compares
intermediate values at each step to identify where divergence occurs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dfm_python import DDFM, DDFMDataset
from dfm_python.config import DDFMConfig, YamlSource, make_config_source
from dfm_python.config.types import to_numpy
import torch

def compare_values(name, pytorch_val, tensorflow_val=None, threshold=1e-2):
    """Compare two values and report differences."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name}")
    print(f"{'='*80}")
    
    if isinstance(pytorch_val, (np.ndarray, pd.DataFrame)):
        if isinstance(pytorch_val, pd.DataFrame):
            pytorch_val = pytorch_val.values
        print(f"PyTorch Shape: {pytorch_val.shape}")
        print(f"PyTorch Mean: {np.mean(pytorch_val):.6f}, Std: {np.std(pytorch_val):.6f}")
        print(f"PyTorch Min: {np.min(pytorch_val):.6f}, Max: {np.max(pytorch_val):.6f}")
        
        if tensorflow_val is not None:
            if isinstance(tensorflow_val, pd.DataFrame):
                tensorflow_val = tensorflow_val.values
            print(f"TensorFlow Shape: {tensorflow_val.shape}")
            print(f"TensorFlow Mean: {np.mean(tensorflow_val):.6f}, Std: {np.std(tensorflow_val):.6f}")
            print(f"TensorFlow Min: {np.min(tensorflow_val):.6f}, Max: {np.max(tensorflow_val):.6f}")
            
            if pytorch_val.shape == tensorflow_val.shape:
                diff = np.abs(pytorch_val - tensorflow_val)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                rel_diff = max_diff / (np.abs(tensorflow_val).max() + 1e-10)
                print(f"Difference - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}, Rel: {rel_diff:.6f}")
                
                if max_diff > threshold:
                    print(f"⚠️  LARGE DIFFERENCE! Max diff: {max_diff:.6f} (threshold: {threshold})")
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    print(f"   Max diff at index {max_idx}:")
                    print(f"   PyTorch: {pytorch_val[max_idx]:.6f}")
                    print(f"   TensorFlow: {tensorflow_val[max_idx]:.6f}")
                    return True
    else:
        print(f"PyTorch: {pytorch_val}")
        if tensorflow_val is not None:
            print(f"TensorFlow: {tensorflow_val}")
            diff = abs(pytorch_val - tensorflow_val)
            print(f"Difference: {diff:.6f}")
            if diff > threshold:
                print(f"⚠️  LARGE DIFFERENCE! Diff: {diff:.6f}")
                return True
    
    return False

def run_pytorch_ddfm(data, config_dict):
    """Run PyTorch DDFM and capture intermediate values."""
    print("\n" + "="*80)
    print("RUNNING PYTORCH DDFM")
    print("="*80)
    
    # Create dataset
    target_series = list(data.columns)
    dataset = DDFMDataset(data, target_series=target_series, time_idx=None)
    
    # Create model
    model = DDFM(dataset=dataset, **config_dict)
    
    # Capture values before fit
    data_before = model._dataset.data.values.copy()
    
    # Run fit
    model.fit()
    
    # Capture values after fit
    result = model.get_result()
    
    return {
        'data_before': data_before,
        'data_after': model.data.values,
        'factors': result.factors,
        'predictions': result.predictions,
        'loss': model.loss_now,
        'num_iter': model._num_iter,
        'y_actual': model.y_actual if hasattr(model, 'y_actual') else None,
        'eps': model.eps if hasattr(model, 'eps') else None,
    }

def load_tensorflow_results():
    """Load TensorFlow DDFM results from saved diagnostics."""
    # Try to load from comparison script output
    tf_results_path = project_root / "DDFM" / "exchange_rate_comparison_results.pkl"
    if tf_results_path.exists():
        with open(tf_results_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    print("="*80)
    print("SIDE-BY-SIDE COMPARISON: PyTorch vs TensorFlow DDFM")
    print("="*80)
    
    # Load data (use smaller subset for faster testing)
    data_path = project_root / "data" / "exchange_rate.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_processed = df.dropna(how='all')
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Use first 500 rows for faster testing
    df_subset = df_processed.iloc[:500].copy()
    
    # Scale data (matching TensorFlow)
    mean_z = df_subset.mean().values
    sigma_z = df_subset.std().values
    df_scaled = (df_subset - mean_z) / sigma_z
    
    print(f"\nData shape: {df_scaled.shape}")
    print(f"Data mean: {df_scaled.values.mean():.6f}, std: {df_scaled.values.std():.6f}")
    
    # Config for testing (smaller for speed)
    config_dict = {
        'encoder_size': (8, 2),  # Smaller for testing
        'decoder_type': 'linear',
        'activation': 'relu',
        'learning_rate': 0.005,
        'optimizer': 'Adam',
        'n_mc_samples': 3,  # Fewer MC samples
        'window_size': 100,
        'max_iter': 5,  # Fewer iterations
        'max_epoch_pre_train': 10,  # Fewer pre-train epochs
        'tolerance': 0.0005,
        'disp': 1,
        'seed': 3
    }
    
    # Run PyTorch
    pytorch_results = run_pytorch_ddfm(df_scaled, config_dict)
    
    # Try to load TensorFlow results
    tf_results = load_tensorflow_results()
    
    # Compare
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80)
    
    divergences = []
    
    # Compare data
    if tf_results and 'data_before' in tf_results:
        if compare_values("Input Data", pytorch_results['data_before'], tf_results['data_before']):
            divergences.append("Input Data")
    
    # Compare factors
    if tf_results and 'factors' in tf_results:
        if compare_values("Factors", pytorch_results['factors'], tf_results['factors']):
            divergences.append("Factors")
    
    # Compare predictions
    if tf_results and 'predictions' in tf_results:
        if compare_values("Predictions", pytorch_results['predictions'], tf_results['predictions']):
            divergences.append("Predictions")
    
    # Compare loss
    if tf_results and 'loss' in tf_results:
        if compare_values("Loss", pytorch_results['loss'], tf_results['loss'], threshold=0.1):
            divergences.append("Loss")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"PyTorch Loss: {pytorch_results['loss']:.6f}")
    print(f"PyTorch Iterations: {pytorch_results['num_iter']}")
    if tf_results:
        print(f"TensorFlow Loss: {tf_results.get('loss', 'N/A')}")
        print(f"TensorFlow Iterations: {tf_results.get('num_iter', 'N/A')}")
    
    if divergences:
        print(f"\n⚠️  Divergences found in: {', '.join(divergences)}")
    else:
        print("\n✓ No major divergences found")
    
    print("\nNext steps:")
    print("1. Run TensorFlow DDFM with same data and save diagnostics")
    print("2. Compare intermediate values at each MCMC iteration")
    print("3. Compare BatchNorm statistics")
    print("4. Compare optimizer states")

if __name__ == "__main__":
    main()

