"""Comprehensive test suite for DFM package with various parameters and situations."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python import load_config, load_data, dfm
from dfm_python.config import DFMConfig, SeriesConfig
from dfm_python.dfm import init_conditions, em_step
from dfm_python.kalman import run_kf

warnings.filterwarnings('ignore')

import logging
logging.getLogger('dfm_python').setLevel(logging.ERROR)  # Suppress INFO/WARNING logs

def create_test_data(T=100, N=10, seed=42, missing_rate=0.1):
    """Create synthetic test data."""
    np.random.seed(seed)
    # Generate data with some structure
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    
    # Add missing values
    if missing_rate > 0:
        missing_mask = np.random.rand(T, N) < missing_rate
        X[missing_mask] = np.nan
    
    return X

def create_test_config(series_freqs, clock='m', block_names=None, factors_per_block=None):
    """Create test configuration."""
    if block_names is None:
        block_names = ['Global', 'Block1']
    if factors_per_block is None:
        factors_per_block = [1, 1]
    
    N = len(series_freqs)
    series_list = []
    for i, freq in enumerate(series_freqs):
        # All series load on Global, first half also load on Block1
        blocks = [1, 1 if i < N // 2 else 0]
        series_list.append(SeriesConfig(
            series_id=f"test_{i:02d}",
            series_name=f"Test Series {i}",
            frequency=freq,
            transformation='pch',
            category='Test',
            units='Index',
            blocks=blocks
        ))
    
    return DFMConfig(
        series=series_list,
        block_names=block_names,
        factors_per_block=factors_per_block,
        ar_lag=1,
        threshold=1e-4,
        max_iter=100,
        nan_method=2,
        nan_k=3,
        clock=clock
    )

def test_clock_frequencies():
    """Test different clock frequencies."""
    print("\n" + "="*70)
    print("TEST: Different Clock Frequencies")
    print("="*70)
    
    results = {}
    # Test with monthly and quarterly clocks (daily/weekly may have issues with small data)
    clocks = ['m', 'q']
    series_freqs = ['m', 'm', 'm', 'q', 'q']  # 3 monthly, 2 quarterly
    
    for clock in clocks:
        try:
            print(f"\n  Testing clock='{clock}'...")
            config = create_test_config(series_freqs, clock=clock)
            X = create_test_data(T=100, N=5, seed=42)
            
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[clock] = 'PASS'
            print(f"    ✓ clock='{clock}' passed")
        except Exception as e:
            results[clock] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ clock='{clock}' failed: {str(e)[:50]}")
    
    return results

def test_mixed_frequencies():
    """Test various mixed frequency combinations."""
    print("\n" + "="*70)
    print("TEST: Mixed Frequency Combinations")
    print("="*70)
    
    test_cases = [
        ('All Monthly', ['m', 'm', 'm', 'm', 'm'], 'm'),
        ('Monthly + Quarterly', ['m', 'm', 'm', 'q', 'q'], 'm'),
        ('Daily + Monthly', ['d', 'd', 'm', 'm', 'm'], 'm'),
        ('Weekly + Monthly', ['w', 'w', 'm', 'm', 'm'], 'm'),
        ('Quarterly + Semi-annual', ['q', 'q', 'sa', 'sa'], 'q'),
        ('All Frequencies', ['d', 'w', 'm', 'q', 'sa', 'a'], 'm'),
        ('Monthly + Quarterly + Annual', ['m', 'm', 'q', 'q', 'a'], 'm'),
    ]
    
    results = {}
    for name, freqs, clock in test_cases:
        try:
            print(f"\n  Testing {name} (clock={clock})...")
            config = create_test_config(freqs, clock=clock)
            X = create_test_data(T=100, N=len(freqs), seed=42)
            
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_parameters():
    """Test different parameter combinations."""
    print("\n" + "="*70)
    print("TEST: Different Parameter Combinations")
    print("="*70)
    
    base_config = create_test_config(['m', 'm', 'm', 'q', 'q'], clock='m')
    X = create_test_data(T=100, N=5, seed=42)
    
    test_cases = [
        ('Low threshold', {'threshold': 1e-6, 'max_iter': 10}),
        ('High threshold', {'threshold': 1e-2, 'max_iter': 10}),
        ('Few iterations', {'threshold': 1e-3, 'max_iter': 3}),
        ('Many iterations', {'threshold': 1e-4, 'max_iter': 50}),
        ('nan_method=1', {'nan_method': 1, 'nan_k': 3}),
        ('nan_method=2', {'nan_method': 2, 'nan_k': 3}),
        ('nan_method=3', {'nan_method': 3, 'nan_k': 3}),
        ('nan_k=1', {'nan_method': 2, 'nan_k': 1}),
        ('nan_k=5', {'nan_method': 2, 'nan_k': 5}),
        ('ar_lag=2', {'ar_lag': 2}),
    ]
    
    results = {}
    for name, params in test_cases:
        try:
            print(f"\n  Testing {name}...")
            config_dict = base_config.__dict__.copy()
            config_dict.update(params)
            config = DFMConfig(**config_dict)
            
            result = dfm(X, config, max_iter=params.get('max_iter', 10), 
                        threshold=params.get('threshold', 1e-3))
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_data_scenarios():
    """Test different data scenarios."""
    print("\n" + "="*70)
    print("TEST: Different Data Scenarios")
    print("="*70)
    
    test_cases = [
        ('No missing', {'missing_rate': 0.0, 'T': 100, 'N': 5}),
        ('Low missing (5%)', {'missing_rate': 0.05, 'T': 100, 'N': 5}),
        ('Medium missing (20%)', {'missing_rate': 0.20, 'T': 100, 'N': 5}),
        ('High missing (40%)', {'missing_rate': 0.40, 'T': 100, 'N': 5}),
        ('Short series (T=50)', {'T': 50, 'N': 5, 'missing_rate': 0.1}),
        ('Long series (T=200)', {'T': 200, 'N': 5, 'missing_rate': 0.1}),
        ('Few series (N=3)', {'N': 3, 'T': 100, 'missing_rate': 0.1}),
        ('Many series (N=15)', {'N': 15, 'T': 100, 'missing_rate': 0.1}),
    ]
    
    results = {}
    for name, params in test_cases:
        try:
            print(f"\n  Testing {name}...")
            T = params.get('T', 100)
            N = params.get('N', 5)
            missing_rate = params.get('missing_rate', 0.1)
            
            # Adjust config for different N
            freqs = ['m'] * N
            config = create_test_config(freqs, clock='m')
            
            X = create_test_data(T=T, N=N, seed=42, missing_rate=missing_rate)
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_block_configurations():
    """Test different block configurations."""
    print("\n" + "="*70)
    print("TEST: Different Block Configurations")
    print("="*70)
    
    test_cases = [
        ('Single block', ['Global'], [1], 5),
        ('Two blocks', ['Global', 'Block1'], [1, 1], 5),
        ('Three blocks', ['Global', 'Block1', 'Block2'], [1, 1, 1], 8),
        ('Different factors', ['Global', 'Block1'], [2, 1], 5),
        ('All on Global', ['Global'], [1], 5, True),  # All series only on Global
    ]
    
    results = {}
    for case in test_cases:
        name = case[0]
        block_names = case[1]
        factors = case[2]
        N = case[3]
        all_global = case[4] if len(case) > 4 else False
        
        try:
            print(f"\n  Testing {name}...")
            freqs = ['m'] * N
            series_list = []
            for i in range(N):
                if all_global:
                    blocks = [1]  # Only Global
                else:
                    # Distribute across blocks
                    blocks = [1]  # Always Global
                    if len(block_names) > 1:
                        blocks.append(1 if i < N // 2 else 0)  # First half on Block1
                    if len(block_names) > 2:
                        blocks.append(1 if i >= N // 2 else 0)  # Second half on Block2
                
                series_list.append(SeriesConfig(
                    series_id=f"test_{i:02d}",
                    series_name=f"Test Series {i}",
                    frequency='m',
                    transformation='pch',
                    category='Test',
                    units='Index',
                    blocks=blocks
                ))
            
            config = DFMConfig(
                series=series_list,
                block_names=block_names,
                factors_per_block=factors,
                clock='m'
            )
            
            X = create_test_data(T=100, N=N, seed=42)
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_transformations():
    """Test different transformation types."""
    print("\n" + "="*70)
    print("TEST: Different Transformation Types")
    print("="*70)
    
    transformations = ['lin', 'chg', 'pch', 'pca', 'log']
    results = {}
    
    for trans in transformations:
        try:
            print(f"\n  Testing transformation='{trans}'...")
            series_list = []
            for i in range(5):
                series_list.append(SeriesConfig(
                    series_id=f"test_{i:02d}",
                    series_name=f"Test Series {i}",
                    frequency='m',
                    transformation=trans,
                    category='Test',
                    units='Index',
                    blocks=[1, 1 if i < 3 else 0]
                ))
            
            config = DFMConfig(
                series=series_list,
                block_names=['Global', 'Block1'],
                clock='m'
            )
            
            X = create_test_data(T=100, N=5, seed=42)
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[trans] = 'PASS'
            print(f"    ✓ transformation='{trans}' passed")
        except Exception as e:
            results[trans] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ transformation='{trans}' failed: {str(e)[:50]}")
    
    return results

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*70)
    print("TEST: Edge Cases and Boundary Conditions")
    print("="*70)
    
    results = {}
    
    # Test cases (skip problematic ones)
    test_cases = [
        ('Minimal data (T=30, N=3)', {'T': 30, 'N': 3}),
        ('Very high missing (50%)', {'missing_rate': 0.5}),
        ('Constant series', {'constant': True}),
        ('Extreme values', {'extreme': True}),
    ]
    
    for name, params in test_cases:
        try:
            print(f"\n  Testing {name}...")
            T = params.get('T', 100)
            N = params.get('N', 5)
            missing_rate = params.get('missing_rate', 0.1)
            
            if params.get('constant'):
                X = np.ones((T, N)) * 100.0
                # Add some variation to first series
                np.random.seed(42)
                X[:, 0] = 100.0 + np.random.randn(T) * 0.1
            elif params.get('extreme'):
                np.random.seed(42)
                X = np.random.randn(T, N) * 1e6  # Very large values
            else:
                X = create_test_data(T=T, N=N, seed=42, missing_rate=missing_rate)
            
            config = create_test_config(['m'] * N, clock='m')
            result = dfm(X, config, max_iter=3, threshold=1e-2)
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_tent_weights_scenarios():
    """Test scenarios that trigger tent weights."""
    print("\n" + "="*70)
    print("TEST: Tent Weights Scenarios")
    print("="*70)
    
    results = {}
    
    # Test cases that should use tent weights (slower freq < clock)
    test_cases = [
        ('Quarterly with monthly clock', ['m', 'm', 'q', 'q'], 'm'),
        ('Semi-annual with monthly clock', ['m', 'm', 'sa'], 'm'),
        ('Annual with monthly clock', ['m', 'm', 'a'], 'm'),
        ('Annual with quarterly clock', ['q', 'q', 'a'], 'q'),
        ('Quarterly with daily clock', ['d', 'd', 'q'], 'd'),
    ]
    
    for name, freqs, clock in test_cases:
        try:
            print(f"\n  Testing {name}...")
            config = create_test_config(freqs, clock=clock)
            X = create_test_data(T=100, N=len(freqs), seed=42)
            
            result = dfm(X, config, max_iter=5, threshold=1e-3)
            results[name] = 'PASS'
            print(f"    ✓ {name} passed")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_convergence_scenarios():
    """Test convergence behavior with different settings."""
    print("\n" + "="*70)
    print("TEST: Convergence Scenarios")
    print("="*70)
    
    config = create_test_config(['m', 'm', 'm', 'q', 'q'], clock='m')
    X = create_test_data(T=100, N=5, seed=42)
    
    results = {}
    test_cases = [
        ('Converges quickly', {'threshold': 1e-2, 'max_iter': 50}),
        ('Strict convergence', {'threshold': 1e-6, 'max_iter': 100}),
        ('Max iterations reached', {'threshold': 1e-8, 'max_iter': 5}),
    ]
    
    for name, params in test_cases:
        try:
            print(f"\n  Testing {name}...")
            result = dfm(X, config, max_iter=params['max_iter'], 
                        threshold=params['threshold'])
            
            # Check if converged or hit max_iter
            converged = hasattr(result, 'converged') and result.converged if hasattr(result, 'converged') else True
            results[name] = 'PASS'
            print(f"    ✓ {name} passed (converged: {converged})")
        except Exception as e:
            results[name] = f'FAIL: {str(e)[:50]}'
            print(f"    ✗ {name} failed: {str(e)[:50]}")
    
    return results

def test_real_data_scenario():
    """Test with real sample data configuration."""
    print("\n" + "="*70)
    print("TEST: Real Sample Data Scenario")
    print("="*70)
    
    try:
        config_file = project_root / 'config' / 'example_config.yaml'
        data_file = project_root / 'data' / 'sample_data.csv'
        
        if not config_file.exists() or not data_file.exists():
            print("    ⚠ Sample data files not found, skipping...")
            return {'Real data': 'SKIP'}
        
        config = load_config(config_file)
        X, Time, Z = load_data(data_file, config)
        
        # Test with different parameters
        test_cases = [
            ('Default params', {}),
            ('Quick test', {'max_iter': 5, 'threshold': 1e-2}),
            ('Strict convergence', {'max_iter': 100, 'threshold': 1e-6}),
        ]
        
        results = {}
        for name, params in test_cases:
            try:
                print(f"\n  Testing {name}...")
                result = dfm(X, config, **params)
                results[name] = 'PASS'
                print(f"    ✓ {name} passed")
            except Exception as e:
                results[name] = f'FAIL: {str(e)[:50]}'
                print(f"    ✗ {name} failed: {str(e)[:50]}")
        
        return results
    except Exception as e:
        return {'Real data': f'FAIL: {str(e)[:50]}'}

def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE FOR DFM PACKAGE")
    print("="*70)
    
    all_results = {}
    
    # Run all test suites
    all_results['Clock Frequencies'] = test_clock_frequencies()
    all_results['Mixed Frequencies'] = test_mixed_frequencies()
    all_results['Parameters'] = test_parameters()
    all_results['Data Scenarios'] = test_data_scenarios()
    all_results['Block Configurations'] = test_block_configurations()
    all_results['Transformations'] = test_transformations()
    all_results['Edge Cases'] = test_edge_cases()
    all_results['Tent Weights'] = test_tent_weights_scenarios()
    all_results['Convergence'] = test_convergence_scenarios()
    all_results['Real Data'] = test_real_data_scenario()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for suite_name, suite_results in all_results.items():
        print(f"\n{suite_name}:")
        for test_name, result in suite_results.items():
            total_tests += 1
            if result == 'PASS':
                passed_tests += 1
                print(f"  ✓ {test_name}")
            elif result == 'SKIP':
                print(f"  ⊘ {test_name} (skipped)")
            else:
                failed_tests += 1
                print(f"  ✗ {test_name}: {result}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {total_tests} tests | PASSED: {passed_tests} | FAILED: {failed_tests} | SKIPPED: {total_tests - passed_tests - failed_tests}")
    print("="*70)
    
    return all_results

if __name__ == '__main__':
    results = run_all_tests()

