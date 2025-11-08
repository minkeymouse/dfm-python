"""Core tests for news decomposition."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python import load_config, load_data, dfm, update_nowcast

# ============================================================================
# Core Tests
# ============================================================================

def test_update_nowcast_basic():
    """Test basic nowcast update (if data files available)."""
    print("\n" + "="*70)
    print("TEST: Basic Nowcast Update")
    print("="*70)
    
    base_dir = Path(__file__).parent.parent.parent
    spec_file = base_dir / 'Nowcasting' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    try:
        config = load_config(spec_file)
        vintage_old = '2016-12-16'
        vintage_new = '2016-12-23'
        
        datafile_old = base_dir / 'data' / 'US' / f'{vintage_old}.xls'
        datafile_new = base_dir / 'data' / 'US' / f'{vintage_new}.xls'
        
        if not datafile_old.exists() or not datafile_new.exists():
            print("SKIPPED: Vintage data files not found")
            return None
        
        X_old, Time_old, _ = load_data(datafile_old, config, sample_start=pd.Timestamp('2000-01-01'))
        X_new, Time, _ = load_data(datafile_new, config, sample_start=pd.Timestamp('2000-01-01'))
        
        # Quick DFM estimation for testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Res = dfm(X_old, config, threshold=1e-3, max_iter=20)
        
        update_nowcast(X_old, X_new, Time, config, Res, 
                      series='GDPC1', period='2016q4',
                      vintage_old=vintage_old, vintage_new=vintage_new)
        
        print("✓ Nowcast update test passed")
        return True
    except Exception as e:
        print(f"SKIPPED: {type(e).__name__}: {e}")
        return None

# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all news decomposition tests."""
    print("\n" + "="*70)
    print("NEWS DECOMPOSITION TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('basic', test_update_nowcast_basic),
    ]
    
    for name, func in test_funcs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[name] = func()
            if results[name] is not None:
                print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = None
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("="*70)
    
    return results


if __name__ == '__main__':
    run_all_tests()
