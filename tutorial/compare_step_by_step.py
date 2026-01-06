"""Step-by-step comparison of PyTorch DDFM with TensorFlow.

This script runs the DDFM algorithm and compares intermediate values
at each step to identify where results diverge.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dfm_python import DDFM, DDFMDataset
from dfm_python.config import DDFMConfig, YamlSource, make_config_source
from dfm_python.config.types import to_numpy
import torch

def print_comparison(step_name, pytorch_val, tensorflow_val=None, threshold=1e-3):
    """Print comparison between PyTorch and TensorFlow values."""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
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
                print(f"Difference - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
                
                if max_diff > threshold:
                    print(f"⚠️  LARGE DIFFERENCE DETECTED! Max diff: {max_diff:.6f}")
                    # Find where max difference occurs
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    print(f"   Max diff at index {max_idx}: PyTorch={pytorch_val[max_idx]:.6f}, TensorFlow={tensorflow_val[max_idx]:.6f}")
    else:
        print(f"PyTorch: {pytorch_val}")
        if tensorflow_val is not None:
            print(f"TensorFlow: {tensorflow_val}")
            diff = abs(pytorch_val - tensorflow_val)
            print(f"Difference: {diff:.6f}")
            if diff > threshold:
                print(f"⚠️  LARGE DIFFERENCE DETECTED! Diff: {diff:.6f}")

def main():
    print("="*80)
    print("STEP-BY-STEP COMPARISON: PyTorch vs TensorFlow DDFM")
    print("="*80)
    
    # Load data
    data_path = project_root / "data" / "exchange_rate.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df_processed = df.dropna(how='all')
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Use smaller subset for faster testing
    df_subset = df_processed.iloc[:200].copy()  # First 200 rows
    
    print(f"\nData shape: {df_subset.shape}")
    print(f"Data mean: {df_subset.values.mean():.6f}, std: {df_subset.values.std():.6f}")
    
    # Create dataset
    target_series = list(df_subset.columns)
    dataset = DDFMDataset(df_subset, target_series=target_series, time_idx=None)
    
    # Load config
    config_path = project_root / "config" / "ddfm_exchange.yaml"
    config_source = make_config_source(YamlSource(config_path))
    config = DDFMConfig.from_source(config_source)
    
    # Override for faster testing
    config.max_iter = 3
    config.max_epoch_pre_train = 10
    config.n_mc_samples = 2
    
    # Create model
    model = DDFM(dataset=dataset, **config.to_dict())
    
    # Run fit and capture intermediate values
    print("\n" + "="*80)
    print("RUNNING PYTORCH DDFM")
    print("="*80)
    
    # We'll need to modify fit() to return intermediate values, or use a callback
    # For now, let's run fit and check final results
    model.fit()
    
    # Get results
    result = model.get_result()
    factors = result.factors
    predictions = result.predictions
    
    print_comparison("Final Factors", factors)
    print_comparison("Final Predictions", predictions)
    
    # Check loss
    from dfm_python.numeric.stability import convergence_checker
    if hasattr(model, 'y_actual') and hasattr(model, 'data_imputed'):
        # Get final prediction
        final_pred = model.data_imputed.values
        delta, loss = convergence_checker(final_pred, final_pred, model.y_actual)
        print(f"\nFinal Loss: {loss:.6f}, Delta: {delta:.6f}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Run TensorFlow DDFM with same data")
    print("2. Compare intermediate values at each step")
    print("3. Identify where divergence occurs")

if __name__ == "__main__":
    main()

