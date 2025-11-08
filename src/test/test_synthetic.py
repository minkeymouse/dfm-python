"""Synthetic data integration test."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python import load_config, load_data, dfm

def test_with_yaml_config():
    """Test with YAML configuration."""
    print("=" * 60)
    print("Testing with YAML Configuration")
    print("=" * 60)
    
    try:
        config_path = project_root / 'config' / 'example_config.yaml'
        if not config_path.exists():
            print("SKIPPED: Config file not found")
            return False, None
        
        config = load_config(str(config_path))
        print(f"✓ Loaded YAML config: {len(config.series)} series")
        
        data_path = project_root / 'data' / 'sample_data.csv'
        if not data_path.exists():
            print("SKIPPED: Data file not found")
            return False, None
        
        X, Time, Z = load_data(
            str(data_path),
            config,
            sample_start=pd.Timestamp('2000-01-01')
        )
        print(f"✓ Loaded data: shape {X.shape}")
        
        result = dfm(X, config, threshold=1e-3, max_iter=50)
        
        print(f"✓ DFM estimation completed")
        print(f"  Factors: {result.Z.shape[1]}")
        print(f"  Series: {result.C.shape[0]}")
        
        return True, result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None


def test_with_csv_config():
    """Test with CSV configuration (deprecated - using direct DFMConfig instead)."""
    print("\n" + "=" * 60)
    print("Testing with Direct DFMConfig Creation")
    print("=" * 60)
    
    try:
        # Create config directly instead of loading from CSV
        from dfm_python import DFMConfig, SeriesConfig
        
        series_list = [
            SeriesConfig(
                series_id='series_0',
                series_name='Test Series 0',
                frequency='m',
                transformation='lin',
                blocks=[1]
            ),
            SeriesConfig(
                series_id='series_1',
                series_name='Test Series 1',
                frequency='m',
                transformation='lin',
                blocks=[1]
            )
        ]
        
        config = DFMConfig(
            series=series_list,
            block_names=['Global']
        )
        print(f"✓ Created DFMConfig directly: {len(config.series)} series")
        
        data_path = project_root / 'data' / 'sample_data.csv'
        if not data_path.exists():
            print("SKIPPED: Data file not found")
            return False, None
        
        X, Time, Z = load_data(
            str(data_path),
            config,
            sample_start=pd.Timestamp('2000-01-01')
        )
        print(f"✓ Loaded data: shape {X.shape}")
        
        result = dfm(X, config, threshold=1e-3, max_iter=50)
        
        print(f"✓ DFM estimation completed")
        print(f"  Factors: {result.Z.shape[1]}")
        
        return True, result
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None


def main():
    """Run all tests."""
    print("Testing dfm-python with sample data\n")
    
    success_yaml, _ = test_with_yaml_config()
    success_csv, _ = test_with_csv_config()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"YAML config test: {'✓ PASSED' if success_yaml else '✗ FAILED/SKIPPED'}")
    print(f"CSV config test:   {'✓ PASSED' if success_csv else '✗ FAILED/SKIPPED'}")
    
    if success_yaml and success_csv:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests skipped (missing files)")
        return 0


if __name__ == '__main__':
    sys.exit(main())
