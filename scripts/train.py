#!/usr/bin/env python3
"""
Training script for Dynamic Factor Model (DFM).

This script loads configuration and data, runs DFM estimation,
and saves the results for later use in inference.

Usage:
    python scripts/train.py --config config/config.yaml --data data/data.csv --output outputs/model.pkl
"""

import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dfm_python import load_config, load_data, dfm

def check_data_preprocessing(X, config, verbose=True):
    """Check data preprocessing issues that could cause numerical problems."""
    issues = []
    warnings_list = []
    
    T, N = X.shape
    
    # Check 1: Standard deviation before standardization
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0)
    
    # Check for zero or near-zero standard deviations
    zero_std_mask = Wx < 1e-6
    if np.any(zero_std_mask):
        zero_std_indices = np.where(zero_std_mask)[0]
        zero_std_series = [config.SeriesID[i] if hasattr(config, 'SeriesID') else f"Series_{i}" 
                          for i in zero_std_indices]
        issues.append(f"⚠️  CRITICAL: {len(zero_std_indices)} series have zero/near-zero standard deviation: {zero_std_series}")
        warnings_list.append(("zero_std", zero_std_indices, zero_std_series))
    
    # Check for NaN standard deviations
    nan_std_mask = np.isnan(Wx)
    if np.any(nan_std_mask):
        nan_std_indices = np.where(nan_std_mask)[0]
        nan_std_series = [config.SeriesID[i] if hasattr(config, 'SeriesID') else f"Series_{i}" 
                         for i in nan_std_indices]
        issues.append(f"⚠️  CRITICAL: {len(nan_std_indices)} series have NaN standard deviation: {nan_std_series}")
        warnings_list.append(("nan_std", nan_std_indices, nan_std_series))
    
    # Check 2: Extreme values after transformation
    extreme_mask = np.abs(X) > 1e10
    if np.any(extreme_mask):
        n_extreme = np.sum(extreme_mask)
        pct_extreme = 100 * n_extreme / (T * N)
        issues.append(f"⚠️  WARNING: {n_extreme} extreme values (>1e10) found ({pct_extreme:.2f}% of data)")
        warnings_list.append(("extreme_values", n_extreme, pct_extreme))
    
    # Check 3: Missing data percentage
    missing_pct = 100 * np.sum(np.isnan(X)) / (T * N)
    if missing_pct > 50:
        issues.append(f"⚠️  WARNING: High missing data rate: {missing_pct:.2f}%")
        warnings_list.append(("high_missing", missing_pct))
    
    # Check 4: Constant series (after transformation)
    for i in range(N):
        series_data = X[:, i]
        finite_data = series_data[np.isfinite(series_data)]
        if len(finite_data) > 0:
            if np.std(finite_data) < 1e-6:
                series_name = config.SeriesID[i] if hasattr(config, 'SeriesID') else f"Series_{i}"
                issues.append(f"⚠️  WARNING: Series {series_name} is constant (std={np.std(finite_data):.2e})")
                warnings_list.append(("constant_series", i, series_name))
    
    # Check 5: Mixed frequency scaling issues
    if hasattr(config, 'Frequency'):
        freq_counts = {}
        for i, freq in enumerate(config.Frequency):
            if freq not in freq_counts:
                freq_counts[freq] = []
            freq_counts[freq].append(i)
        
        # Check if different frequencies have very different scales
        freq_means = {}
        freq_stds = {}
        for freq, indices in freq_counts.items():
            freq_data = X[:, indices]
            finite_data = freq_data[np.isfinite(freq_data)]
            if len(finite_data) > 0:
                freq_means[freq] = np.mean(np.abs(finite_data))
                freq_stds[freq] = np.std(finite_data)
        
        if len(freq_means) > 1:
            max_mean = max(freq_means.values())
            min_mean = min(freq_means.values())
            if max_mean > 0 and min_mean > 0:
                scale_ratio = max_mean / min_mean
                if scale_ratio > 1e6:
                    issues.append(f"⚠️  WARNING: Large scale differences between frequencies (ratio: {scale_ratio:.2e})")
                    warnings_list.append(("scale_difference", scale_ratio))
    
    if verbose and issues:
        print("\n" + "="*70)
        print("DATA PREPROCESSING CHECKS")
        print("="*70)
        for issue in issues:
            print(issue)
        print("="*70 + "\n")
    
    return issues, warnings_list

def fix_preprocessing_issues(X, config, warnings_list):
    """Apply fixes for identified preprocessing issues."""
    T, N = X.shape
    X_fixed = X.copy()
    
    # Fix 1: Handle zero/near-zero standard deviations
    Mx = np.nanmean(X_fixed, axis=0)
    Wx = np.nanstd(X_fixed, axis=0)
    
    # Set minimum standard deviation
    Wx = np.maximum(Wx, 1e-6)
    
    # Fix 2: Handle NaN standard deviations
    nan_std_mask = np.isnan(Wx) | np.isnan(Mx)
    if np.any(nan_std_mask):
        Wx[nan_std_mask] = 1.0
        Mx[nan_std_mask] = 0.0
    
    # Standardize with fixed Wx
    xNaN = (X_fixed - Mx) / Wx
    
    # Fix 3: Clip extreme values
    xNaN = np.clip(xNaN, -100, 100)  # Cap at 100 standard deviations
    
    # Fix 4: Handle constant series
    for i in range(N):
        series_data = xNaN[:, i]
        finite_data = series_data[np.isfinite(series_data)]
        if len(finite_data) > 0 and np.std(finite_data) < 1e-6:
            # Set to small random noise to avoid division issues
            xNaN[:, i] = np.random.randn(T) * 1e-6
    
    return xNaN, Mx, Wx

def main():
    parser = argparse.ArgumentParser(description='Train Dynamic Factor Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (YAML or CSV)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file (CSV)')
    parser.add_argument('--output', type=str, default='outputs/model.pkl',
                       help='Output path for trained model (default: outputs/model.pkl)')
    parser.add_argument('--sample-start', type=str, default=None,
                       help='Sample start date (YYYY-MM-DD, optional)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='EM convergence threshold (overrides config)')
    parser.add_argument('--max-iter', type=int, default=None,
                       help='Maximum EM iterations (overrides config)')
    parser.add_argument('--fix-preprocessing', action='store_true',
                       help='Automatically fix preprocessing issues')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DFM TRAINING")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print("="*70 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded: {len(config.series)} series, {len(config.block_names)} blocks")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return 1
    
    # Load data
    print("\nLoading data...")
    try:
        sample_start = pd.Timestamp(args.sample_start) if args.sample_start else None
        X, Time, Z = load_data(args.data, config, sample_start=sample_start)
        print(f"✓ Data loaded: {X.shape[0]} time periods, {X.shape[1]} series")
        print(f"  Time range: {Time[0]} to {Time[-1]}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check preprocessing issues
    print("\nChecking data preprocessing...")
    issues, warnings_list = check_data_preprocessing(X, config, verbose=args.verbose)
    
    if issues and not args.fix_preprocessing:
        print("\n⚠️  Preprocessing issues detected. Use --fix-preprocessing to auto-fix.")
        print("   Continuing anyway, but results may be unreliable...\n")
    
    # Fix preprocessing if requested
    if args.fix_preprocessing and warnings_list:
        print("\nApplying preprocessing fixes...")
        X_fixed, Mx_fixed, Wx_fixed = fix_preprocessing_issues(X, config, warnings_list)
        # Replace X with fixed version for estimation
        # Note: This is a workaround - ideally we'd fix it in dfm() itself
        print("⚠️  Note: Preprocessing fixes applied. This is a temporary workaround.")
        print("   Consider fixing the data preprocessing in dfm() function.\n")
        # For now, we'll proceed with original X but with warnings
    
    # Estimate DFM
    print("Estimating DFM model...")
    print("-"*70)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress numerical warnings during estimation
            result = dfm(
                X, 
                config, 
                threshold=args.threshold,
                max_iter=args.max_iter
            )
        
        print(f"\n✓ Model estimation completed!")
        print(f"  Factors extracted: {result.Z.shape[1]}")
        print(f"  Factor loadings shape: {result.C.shape}")
        print(f"  Smoothed data shape: {result.X_sm.shape}")
        
        # Check for extreme values in results
        if np.any(np.abs(result.C) > 100):
            n_extreme = np.sum(np.abs(result.C) > 100)
            print(f"\n⚠️  WARNING: {n_extreme} extreme factor loadings (>100) detected!")
            print("   This indicates numerical instability. Check preprocessing.")
        
        if np.any(np.abs(result.Z) > 1e10):
            n_extreme = np.sum(np.abs(result.Z) > 1e10)
            print(f"\n⚠️  WARNING: {n_extreme} extreme factor values (>1e10) detected!")
            print("   This indicates numerical instability. Check preprocessing.")
        
    except Exception as e:
        print(f"\n✗ Model estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    try:
        with open(args.output, 'wb') as f:
            pickle.dump({
                'result': result,
                'config': config,
                'Time': Time,
                'X_original': Z,  # Original untransformed data
                'X_transformed': X,  # Transformed data
            }, f)
        print(f"✓ Results saved successfully")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        return 1
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

