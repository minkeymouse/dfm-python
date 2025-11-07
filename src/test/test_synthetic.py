#!/usr/bin/env python3
"""Test the dfm-python package with sample data."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path (go up two levels from src/test/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python import load_config, load_data, dfm

def test_with_yaml_config():
    """Test with YAML configuration."""
    print("=" * 60)
    print("Testing with YAML Configuration")
    print("=" * 60)
    
    try:
        # Load configuration from YAML (relative to project root)
        config_path = project_root / 'config' / 'example_config.yaml'
        config = load_config(str(config_path))
        print(f"✓ Loaded YAML config: {len(config.series)} series")
        print(f"  Block names: {config.block_names}")
        print(f"  Factors per block: {config.factors_per_block}")
        
        # Load data (relative to project root)
        data_path = project_root / 'data' / 'sample_data.csv'
        X, Time, Z = load_data(
            str(data_path),
            config,
            sample_start=pd.Timestamp('2000-01-01')
        )
        print(f"✓ Loaded data: shape {X.shape}")
        print(f"  Time range: {Time[0]} to {Time[-1]}")
        print(f"  Number of observations: {len(Time)}")
        
        # Estimate DFM
        print("\nEstimating DFM model...")
        result = dfm(X, config, threshold=1e-4, max_iter=1000)
        
        # Display results
        print(f"✓ DFM estimation completed")
        print(f"  Factors extracted: {result.Z.shape[1]}")
        print(f"  Factor loadings shape: {result.C.shape}")
        print(f"  Transition matrix shape: {result.A.shape}")
        print(f"  Number of series: {result.C.shape[0]}")
        print(f"  Number of time periods: {result.Z.shape[0]}")
        
        # Show first few factor values
        print(f"\n  First 5 factor values (first 3 factors):")
        print(f"  {result.Z[:5, :3]}")
        
        return True, result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_with_csv_config():
    """Test with CSV configuration."""
    print("\n" + "=" * 60)
    print("Testing with CSV Configuration")
    print("=" * 60)
    
    try:
        # Load configuration from CSV (relative to project root)
        config_path = project_root / 'spec' / 'example_spec.csv'
        config = load_config(str(config_path))
        print(f"✓ Loaded CSV config: {len(config.series)} series")
        print(f"  Block names: {config.block_names}")
        
        # Load data (relative to project root)
        data_path = project_root / 'data' / 'sample_data.csv'
        X, Time, Z = load_data(
            str(data_path),
            config,
            sample_start=pd.Timestamp('2000-01-01')
        )
        print(f"✓ Loaded data: shape {X.shape}")
        print(f"  Time range: {Time[0]} to {Time[-1]}")
        
        # Estimate DFM
        print("\nEstimating DFM model...")
        result = dfm(X, config, threshold=1e-4, max_iter=1000)
        
        # Display results
        print(f"✓ DFM estimation completed")
        print(f"  Factors extracted: {result.Z.shape[1]}")
        print(f"  Factor loadings shape: {result.C.shape}")
        print(f"  Number of series: {result.C.shape[0]}")
        print(f"  Number of time periods: {result.Z.shape[0]}")
        
        return True, result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests."""
    print("Testing dfm-python with sample data\n")
    
    # Test with YAML config
    success_yaml, result_yaml = test_with_yaml_config()
    
    # Test with CSV config
    success_csv, result_csv = test_with_csv_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"YAML config test: {'✓ PASSED' if success_yaml else '✗ FAILED'}")
    print(f"CSV config test:   {'✓ PASSED' if success_csv else '✗ FAILED'}")
    
    if success_yaml and success_csv:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

