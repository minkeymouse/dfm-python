#!/usr/bin/env python3
"""
DFM Python - Basic Tutorial (Hydra-based)

This tutorial demonstrates the complete DFM workflow using Hydra for configuration:
1) Load config from Hydra (YAML files or convert from spec CSV)
2) Load data
3) Train the model
4) Predict and visualize
5) Save forecasts to CSV
6) Understand DFM components and results

This approach uses Hydra for all configuration management:
- YAML config files: config/default.yaml, config/series/*.yaml, config/blocks/*.yaml
- Spec CSV conversion: Use dfm.from_spec() to convert CSV to YAML, then load via Hydra
- CLI overrides: Override any parameter via CLI (e.g., max_iter=10 threshold=1e-4)

Run:
  # Using default YAML config
  python tutorial/basic_tutorial.py \\
    --config-path config \\
    --config-name default \\
    data_path=data/sample_data.csv

  # Convert spec CSV to YAML first, then use
  python -c "import dfm_python as dfm; dfm.from_spec('data/sample_spec.csv')"
  python tutorial/basic_tutorial.py \\
    --config-path config \\
    --config-name sample_spec \\
    data_path=data/sample_data.csv

  # With CLI overrides
  python tutorial/basic_tutorial.py \\
    --config-path config \\
    --config-name default \\
    data_path=data/sample_data.csv \\
    max_iter=10 \\
    threshold=1e-4 \\
    blocks.Block_Global.factors=2

For quick testing:
  python tutorial/basic_tutorial.py \\
    --config-path config \\
    --config-name default \\
    data_path=data/sample_data.csv \\
    max_iter=1
"""

from pathlib import Path
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("ERROR: Hydra is required. Install with: pip install hydra-core")
    sys.exit(1)

import numpy as np
import pickle
from datetime import datetime
import polars as pl
import dfm_python as dfm
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.core.diagnostics import (
    evaluate_factor_estimation,
    evaluate_loading_estimation,
)


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for Hydra-based DFM tutorial.
    
    This function demonstrates:
    1. Loading configuration from Hydra
    2. Loading data with path resolution
    3. Training the model
    4. Forecasting and visualization
    5. Understanding DFM results
    6. Model result storage and data view management
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all DFM parameters.
        Can be overridden via CLI arguments.
    """
    print("="*70)
    print("DFM Python - Basic Tutorial (Hydra-based)")
    print("="*70)
    
    # Get original working directory (Hydra changes cwd)
    original_cwd = Path(get_original_cwd())
    print(f"\nOriginal working directory: {original_cwd}")
    print(f"Current working directory: {Path.cwd()}")
    
    # 1) Load configuration from Hydra
    print(f"\n--- Loading configuration from Hydra ---")
    dfm.load_config(hydra=cfg)
    config = dfm.get_config()
    if config is None:
        raise ValueError("Configuration not loaded")
    
    print(f"✓ Config loaded:")
    print(f"  - Series: {len(config.series)}")
    print(f"  - Blocks: {len(config.block_names)} ({', '.join(config.block_names)})")
    print(f"  - Clock: {config.clock}")
    print(f"  - max_iter: {config.max_iter}, threshold: {config.threshold}")
    
    # 2) Load data
    print(f"\n--- Loading data ---")
    # Resolve data path relative to original cwd
    data_path = cfg.get('data_path', 'data/sample_data.csv')
    if not Path(data_path).is_absolute():
        data_path = original_cwd / data_path
    
    # Handle sample period if specified
    sample_start = cfg.get('sample_start', None)
    sample_end = cfg.get('sample_end', None)
    
    dfm.load_data(
        str(data_path),
        sample_start=sample_start,
        sample_end=sample_end
    )
    
    X = dfm.get_data()
    Time = dfm.get_time()
    if X is None or Time is None:
        raise ValueError("Data not loaded")
    
    print(f"✓ Data loaded:")
    print(f"  - Shape: {X.shape} (time periods × series)")
    from dfm_python.core.time import get_latest_time
    time_start = Time[0]
    time_end = get_latest_time(Time)
    print(f"  - Time range: {time_start} ~ {time_end}")
    print(f"  - Missing data ratio: {np.isnan(X).sum() / X.size * 100:.2f}%")
    
    # 3) Train
    print(f"\n--- Training model ---")
    # Use max_iter from config or override (sufficient for meaningful results)
    max_iter = cfg.get('max_iter', 10)  # Sufficient: 10 iterations for meaningful learning
    threshold = cfg.get('threshold', 1e-2)  # Relaxed threshold for quick convergence
    print(f"  Training with max_iter={max_iter}, threshold={threshold}")
    print(f"  Note: max_iter=1 is too small and results in poor model quality (Q matrix stuck at floor, negative explained variance)")
    
    dfm.train(max_iter=max_iter, threshold=threshold)
    result = dfm.get_result()
    if result is None:
        raise ValueError("Model training failed - no result available")
    
    print(f"✓ Trained:")
    print(f"  - Iterations: {result.num_iter}")
    print(f"  - Converged: {result.converged}")
    print(f"  - Factors: {result.Z.shape[1]} (state dimension: {result.Z.shape[1]})")
    
    # Format loglik for display
    loglik_str = f"{result.loglik:.2f}" if (
        hasattr(result, 'loglik') and 
        result.loglik is not None and 
        np.isfinite(result.loglik)
    ) else 'N/A'
    print(f"  - Final log-likelihood: {loglik_str}")
    
    # 3b) Fast reuse demo (save + load_pickle)
    print(f"\n--- Saving model payload for fast reuse ---")
    outputs_dir = original_cwd / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = outputs_dir / "basic_tutorial_model.pkl"
    payload = {
        'result': result,
        'config': config,
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"  ✓ Saved payload to: {pickle_path}")
    
    print(f"\n--- Fast reload demo via DFM.load_pickle() ---")
    fast_model = dfm.DFM()
    fast_model.load_pickle(
        pickle_path,
        data=X.copy(),
        time_index=Time
    )
    fast_X_forecast, fast_Z_forecast = fast_model.predict(horizon=3)
    print(f"  ✓ Reloaded model -> quick forecast shape: {fast_X_forecast.shape}")
    
    # ========================================================================
    # Model Quality Assessment: Comprehensive Metrics
    # ========================================================================
    print(f"\n" + "="*70)
    print("Model Quality Assessment: Comprehensive Metrics")
    print("="*70)
    
    # 1. Q Matrix (Innovation Variance) Analysis
    print(f"\n--- 1. Q Matrix (Innovation Variance) Analysis ---")
    Q_diag = np.diag(result.Q)
    Q_min = Q_diag.min()
    Q_max = Q_diag.max()
    Q_mean = Q_diag.mean()
    Q_std = Q_diag.std()
    Q_floor = 0.01  # Expected minimum value
    
    print(f"  Q diagonal statistics:")
    print(f"    - Min: {Q_min:.6f}")
    print(f"    - Max: {Q_max:.6f}")
    print(f"    - Mean: {Q_mean:.6f}")
    print(f"    - Std: {Q_std:.6f}")
    print(f"    - Floor (expected): {Q_floor:.6f}")
    
    # Check if Q values are diverse (not all stuck at floor)
    Q_at_floor = np.sum(np.abs(Q_diag - Q_floor) < 1e-6)
    Q_diversity_ratio = 1.0 - (Q_at_floor / len(Q_diag))
    print(f"  Q diversity check:")
    print(f"    - Values at floor: {Q_at_floor}/{len(Q_diag)}")
    print(f"    - Diversity ratio: {Q_diversity_ratio:.2%}")
    if Q_diversity_ratio < 0.1:
        print(f"    ⚠ WARNING: Most Q values are at floor - model may not have learned properly")
        print(f"      Consider increasing max_iter (current: {max_iter})")
    else:
        print(f"    ✓ Q values show good diversity")
    
    # Block-wise Q analysis
    if hasattr(result, 'r') and result.r is not None:
        print(f"  Q by block:")
        factor_idx = 0
        for block_idx, block_name in enumerate(config.block_names):
            r_i = int(result.r[block_idx])
            if r_i > 0:
                Q_block = Q_diag[factor_idx:factor_idx + r_i]
                print(f"    - {block_name} ({r_i} factors): "
                      f"min={Q_block.min():.6f}, max={Q_block.max():.6f}, mean={Q_block.mean():.6f}")
                factor_idx += r_i
    
    # 2. Factor Explained Variance (R Matrix Analysis)
    print(f"\n--- 2. Factor Explained Variance (R Matrix Analysis) ---")
    R_diag = np.diag(result.R)
    X_sm_var = np.nanvar(result.X_sm, axis=0)
    
    # Calculate explained variance per series
    # explained = 1 - (idio_var / total_var)
    # If idio_var > total_var, explained becomes negative (problem!)
    explained_variance = 1.0 - (R_diag / (R_diag + X_sm_var + 1e-10))
    
    print(f"  Idiosyncratic variance (R diagonal) statistics:")
    print(f"    - Min: {R_diag.min():.6f}")
    print(f"    - Max: {R_diag.max():.6f}")
    print(f"    - Mean: {R_diag.mean():.6f}")
    print(f"    - Median: {np.median(R_diag):.6f}")
    
    print(f"  Factor explained variance (1 - R[i,i] / Var(y[i])):")
    print(f"    - Mean: {explained_variance.mean():.2%}")
    print(f"    - Min: {explained_variance.min():.2%}")
    print(f"    - Max: {explained_variance.max():.2%}")
    print(f"    - Median: {np.median(explained_variance):.2%}")
    
    negative_explained = np.sum(explained_variance < 0)
    if negative_explained > 0:
        print(f"    ⚠ WARNING: {negative_explained} series have negative explained variance")
        print(f"      This indicates model fit issues - consider increasing max_iter")
    else:
        print(f"    ✓ All series have non-negative explained variance")
    
    # 3. A Matrix (Transition) Stability
    print(f"\n--- 3. A Matrix (Transition) Stability ---")
    A_eigenvals = np.linalg.eigvals(result.A)
    A_eigenvals_abs = np.abs(A_eigenvals)
    A_max_eigval = np.max(A_eigenvals_abs)
    
    print(f"  A matrix eigenvalues:")
    print(f"    - Max absolute eigenvalue: {A_max_eigval:.6f}")
    print(f"    - Min absolute eigenvalue: {np.min(A_eigenvals_abs):.6f}")
    print(f"    - Mean absolute eigenvalue: {np.mean(A_eigenvals_abs):.6f}")
    
    if A_max_eigval < 0.99:
        print(f"    ✓ All factors are stationary (max |eigenvalue| < 0.99)")
    elif A_max_eigval < 1.0:
        print(f"    ⚠ WARNING: Some factors are near unit root (max |eigenvalue| = {A_max_eigval:.6f})")
    else:
        print(f"    ⚠ WARNING: Some factors may be non-stationary (max |eigenvalue| >= 1.0)")
    
    # 4. C Matrix (Loadings) Analysis
    print(f"\n--- 4. C Matrix (Loadings) Analysis ---")
    C_abs = np.abs(result.C)
    C_norms = np.linalg.norm(result.C, axis=0)  # Norm of each factor column
    
    print(f"  C matrix statistics:")
    print(f"    - Mean absolute loading: {C_abs.mean():.6f}")
    print(f"    - Max absolute loading: {C_abs.max():.6f}")
    print(f"    - Min absolute loading: {C_abs.min():.6f}")
    print(f"    - Series with weak loadings (<0.1): {np.sum(C_abs.max(axis=1) < 0.1)}")
    
    print(f"  C column norms (factor scales):")
    print(f"    - Min: {C_norms.min():.6f}")
    print(f"    - Max: {C_norms.max():.6f}")
    print(f"    - Mean: {C_norms.mean():.6f}")
    
    # Check normalization (should be around 1.0 for clock-frequency factors)
    if np.allclose(C_norms, 1.0, atol=0.1):
        print(f"    ✓ C columns are normalized (norms ≈ 1.0)")
    else:
        print(f"    ⚠ C columns are not normalized (expected for tent-weighted factors)")
    
    # 5. Factor Time Series Statistics
    print(f"\n--- 5. Factor Time Series Statistics ---")
    num_factors = result.Z.shape[1]
    print(f"  Factor time series (Z) statistics:")
    for j in range(min(5, num_factors)):  # Show first 5 factors
        Z_j = result.Z[:, j]
        print(f"    Factor {j}: mean={Z_j.mean():.4f}, std={Z_j.std():.4f}, "
              f"min={Z_j.min():.4f}, max={Z_j.max():.4f}")
    
    # 6. Reconstruction Error
    print(f"\n--- 6. Reconstruction Error ---")
    # Reconstruct: X_recon = Z @ C.T
    X_recon = result.Z @ result.C.T
    # Compare with smoothed data (standardized)
    reconstruction_error = result.x_sm - X_recon
    reconstruction_rmse = np.sqrt(np.nanmean(reconstruction_error ** 2))
    reconstruction_mae = np.nanmean(np.abs(reconstruction_error))
    
    print(f"  Reconstruction error (x_sm - Z @ C.T):")
    print(f"    - RMSE: {reconstruction_rmse:.6f}")
    print(f"    - MAE: {reconstruction_mae:.6f}")
    print(f"    - Note: Small values indicate good factor representation")
    
    # 7. Numerical Stability
    print(f"\n--- 7. Numerical Stability ---")
    has_nan = (np.any(np.isnan(result.Z)) or 
               np.any(np.isnan(result.C)) or 
               np.any(np.isnan(result.A)) or
               np.any(np.isnan(result.Q)) or
               np.any(np.isnan(result.R)))
    has_inf = (np.any(np.isinf(result.Z)) or 
               np.any(np.isinf(result.C)) or 
               np.any(np.isinf(result.A)) or
               np.any(np.isinf(result.Q)) or
               np.any(np.isinf(result.R)))
    
    if has_nan:
        print(f"    ⚠ WARNING: NaN values detected in model matrices")
    else:
        print(f"    ✓ No NaN values detected")
    
    if has_inf:
        print(f"    ⚠ WARNING: Inf values detected in model matrices")
    else:
        print(f"    ✓ No Inf values detected")
    
    # 8. Overall Model Quality Summary
    print(f"\n--- 8. Overall Model Quality Summary ---")
    quality_issues = []
    quality_warnings = []
    
    if not result.converged:
        quality_issues.append("Model did not converge")
    if Q_diversity_ratio < 0.1:
        quality_warnings.append("Q matrix lacks diversity (most values at floor)")
    if negative_explained > 0:
        quality_warnings.append(f"{negative_explained} series have negative explained variance")
    if A_max_eigval >= 1.0:
        quality_warnings.append("A matrix has eigenvalues >= 1.0 (non-stationary)")
    if has_nan or has_inf:
        quality_issues.append("Numerical instability detected")
    
    if quality_issues:
        print(f"  ⚠ CRITICAL ISSUES:")
        for issue in quality_issues:
            print(f"    - {issue}")
    
    if quality_warnings:
        print(f"  ⚠ WARNINGS:")
        for warning in quality_warnings:
            print(f"    - {warning}")
        print(f"  💡 Recommendation: Increase max_iter (current: {max_iter}) or check data quality")
    
    if not quality_issues and not quality_warnings:
        print(f"  ✓ Model quality appears good")
    
    print(f"\n" + "="*70)
    
    # ------------------------------------------------------------------
    # Synthetic data test for factor/loading estimation accuracy (minimal for quick demo)
    # ------------------------------------------------------------------
    print(f"\n" + "="*70)
    print("Synthetic Data Test: Factor/Loading Estimation Accuracy (Minimal)")
    print("="*70)

    # Synthetic data generation using actual config structure
    # Use minimal size for quick demo
    np.random.seed(42)
    T_syn = 30  # Reduced from 100 for quick demo
    N_syn = min(len(config.series), 5)  # Limit to 5 series max for quick demo
    
    # Calculate number of factors from actual block structure
    num_factors_syn = sum(
        config.blocks[block_name].factors 
        for block_name in config.block_names
    )
    # Ensure at least 1 factor
    if num_factors_syn == 0:
        num_factors_syn = 1
    
    print(f"\nSynthetic data will match actual config structure:")
    print(f"  - Num series: {N_syn} (from config)")
    print(f"  - Num factors: {num_factors_syn} (from config blocks)")

    # Generate latent factors as AR(1) processes
    factors = np.zeros((T_syn + 1, num_factors_syn))
    # Use different AR coefficients for each factor
    ar_coeffs = np.linspace(0.8, 0.6, num_factors_syn) if num_factors_syn > 1 else np.array([0.8])
    innovation_std = 0.5
    for t in range(T_syn):
        factors[t + 1, :] = (
            ar_coeffs * factors[t, :] + np.random.randn(num_factors_syn) * innovation_std
        )

    # Generate loadings and observed data
    loadings = np.random.randn(N_syn, num_factors_syn) * 0.5 + 1.0
    common_part_all = factors[1 : T_syn + 1, :] @ loadings.T  # (T, N)

    idio_std = 0.3
    idio_rho = 0.5
    idio_innov = np.random.randn(T_syn, N_syn) * idio_std
    idio = np.zeros((T_syn, N_syn))
    idio[0, :] = idio_innov[0, :]
    for t in range(1, T_syn):
        idio[t, :] = idio_rho * idio[t - 1, :] + idio_innov[t, :]

    X_syn = common_part_all + idio

    # Small amount of missing data
    missing_mask = np.random.rand(T_syn, N_syn) < 0.05
    X_syn[missing_mask] = np.nan

    print("\nSynthetic data summary:")
    print(f"  - Shape: {X_syn.shape} (time × series)")
    print(f"  - Num factors (true): {num_factors_syn}")
    print(f"  - Missing ratio: {np.isnan(X_syn).sum() / X_syn.size * 100:.2f}%")

    # Build DFMConfig for synthetic data using actual config structure
    # Reuse actual series structure (frequency, transformation, blocks) but with synthetic IDs
    # Only use first N_syn series to match the data size
    series_syn = [
        SeriesConfig(
            series_id=f"syn_{s.series_id}",
            frequency=s.frequency,  # Use actual frequency
            transformation=s.transformation,  # Use actual transformation
            blocks=s.blocks,  # Use actual block membership
        )
        for s in config.series[:N_syn]  # Only use first N_syn series
    ]
    # Reuse actual block structure
    blocks_syn = {
        block_name: BlockConfig(
            factors=block_config.factors,  # Use actual number of factors per block
            ar_lag=block_config.ar_lag,  # Use actual AR lag
            clock=block_config.clock,  # Use actual clock frequency
        )
        for block_name, block_config in config.blocks.items()
    }
    config_syn = DFMConfig(series=series_syn, blocks=blocks_syn)

    # Fit model to synthetic data (sufficient iterations for meaningful results)
    print("\n--- Fitting model on synthetic data ---")
    model_syn = dfm.DFM()
    result_syn = model_syn.fit(
        X_syn,
        config_syn,
        max_iter=10,  # Sufficient: 10 iterations for meaningful learning
        threshold=1e-2,  # Relaxed threshold
    )
    print("✓ Synthetic model trained")
    print(f"  - Iterations: {result_syn.num_iter}")
    print(f"  - Converged: {result_syn.converged}")

    # Evaluate factors
    true_factors = factors[1 : T_syn + 1, :]
    est_factors = result_syn.Z[:, :num_factors_syn]
    factor_eval = evaluate_factor_estimation(
        true_factors,
        est_factors,
        use_procrustes=True,
    )

    print("\nFactor estimation accuracy:")
    print(f"  - Num factors compared: {factor_eval['num_factors']}")
    print(
        f"  - Correlation per factor: "
        f"{np.round(factor_eval['correlation_per_factor'], 4)}"
    )
    if factor_eval["overall_correlation"] is not None:
        print(
            f"  - Overall correlation (after Procrustes): "
            f"{factor_eval['overall_correlation']:.4f}"
        )

    # Evaluate loadings (aligned using rotation from factor comparison)
    true_loadings = loadings
    est_loadings = result_syn.C[:, :num_factors_syn]
    loading_eval = evaluate_loading_estimation(
        true_loadings,
        est_loadings,
        rotation_matrix=factor_eval["rotation_matrix"],
    )

    print("\nLoading estimation accuracy:")
    print(
        f"  - Correlation per factor: "
        f"{np.round(loading_eval['correlation_per_factor'], 4)}"
    )
    print(f"  - Overall RMSE (across series × factors): {loading_eval['overall_rmse']:.4f}")
    print(
        f"  - RMSE per series (first 5): "
        f"{np.round(loading_eval['rmse_per_series'][:5], 4)}"
    )
    
    # ========================================================================
    # Understanding DFMResult: What Each Component Means
    # ========================================================================
    print(f"\n" + "="*70)
    print("Understanding DFMResult: What Each Component Means")
    print("="*70)
    print("""
DFM (Dynamic Factor Model) decomposes observed time series into:
  1. Common factors (Z): Unobserved latent variables that drive multiple series
  2. Factor loadings (C): How much each series depends on each factor
  3. Idiosyncratic components: Series-specific noise

The model equations are:
  Observation: y_t = C * Z_t + e_t,  where e_t ~ N(0, R)
  State:       Z_t = A * Z_{{t-1}} + v_t,  where v_t ~ N(0, Q)

Where:
  - y_t: Observed data at time t (N series)
  - Z_t: Latent factors at time t (m factors)
  - C: Loading matrix (N x m) - how series relate to factors
  - A: Transition matrix (m x m) - how factors evolve over time
  - R: Observation error covariance (N x N) - idiosyncratic variances
  - Q: Factor innovation covariance (m x m) - factor shock variances
    """)
    
    print("\n--- DFMResult Components Explained ---")
    print(f"""
1. Z (Smoothed Factors) - Shape: {result.Z.shape}
   - What: Estimated latent factors over time
   - Meaning: Unobserved common drivers of your time series
   - Usage: 
     * result.Z[:, 0] = First factor (often the "common business cycle")
     * result.Z[:, 1] = Second factor (e.g., sector-specific factor)
     * Higher factors capture additional common variation
   - Example: If Z[:, 0] increases, it means the common factor is rising
   
2. C (Loading Matrix) - Shape: {result.C.shape}
   - What: How much each series depends on each factor
   - Meaning: C[i, j] = loading of series i on factor j
   - Usage:
     * result.C[0, :] = Loadings of first series on all factors
     * result.C[:, 0] = Loadings of all series on first factor
   - Interpretation:
     * Large |C[i, j]| = series i is strongly influenced by factor j
     * C[i, j] > 0 = series moves in same direction as factor
     * C[i, j] < 0 = series moves opposite to factor
   
3. A (Transition Matrix) - Shape: {result.A.shape}
   - What: Describes how factors evolve: Z_t = A * Z_{{t-1}} + shock
   - Meaning: A[i, j] = how factor i depends on lagged factor j
   - Usage: Forecast factors forward: Z_{{t+1}} = A @ Z_t
   - Interpretation:
     * Diagonal elements: AR(1) coefficients for each factor
     * Off-diagonal: Cross-factor dependencies
     * Eigenvalues < 1: Factors are stationary (stable)
   
4. Q (Factor Innovation Covariance) - Shape: {result.Q.shape}
   - What: Covariance of shocks to factors
   - Meaning: How volatile factor innovations are
   - Usage: Quantify uncertainty in factor forecasts
   - Interpretation:
     * Diagonal: Variance of each factor's shock
     * Off-diagonal: Co-movement of factor shocks
   
5. R (Observation Error Covariance) - Shape: {result.R.shape}
   - What: Covariance of observation errors (idiosyncratic components)
   - Meaning: Series-specific noise that factors don't explain
   - Usage: Measure how well factors explain each series
   - Interpretation:
     * Diagonal: Idiosyncratic variance for each series
     * Small R[i, i] = series i is well explained by factors
     * Large R[i, i] = series i has high series-specific noise
   
6. X_sm (Smoothed Data - Original Scale) - Shape: {result.X_sm.shape}
   - What: Kalman-smoothed estimates of observed series
   - Meaning: Best estimate of true values given all data
   - Usage: 
     * result.X_sm[:, i] = Smoothed version of series i
     * Useful for: Denoising, gap-filling, nowcasting
   - Note: X_sm = x_sm * Wx + Mx (unstandardized)
   
7. x_sm (Smoothed Data - Standardized) - Shape: {result.x_sm.shape}
   - What: Standardized version of X_sm (mean=0, std=1)
   - Meaning: Data after removing mean and scaling
   - Usage: For analysis where scale doesn't matter
   
8. Mx, Wx (Standardization Parameters) - Shapes: {result.Mx.shape}, {result.Wx.shape}
   - What: Mean and standard deviation used for standardization
   - Meaning: Mx[i] = mean of series i, Wx[i] = std of series i
   - Usage: Convert between standardized and original scale
   - Formula: x_standardized = (X - Mx) / Wx
   
9. Z_0, V_0 (Initial State) - Shapes: {result.Z_0.shape}, {result.V_0.shape}
   - What: Starting values for factors and their uncertainty
   - Meaning: Z_0 = initial factor values, V_0 = initial covariance
   - Usage: Usually not needed for analysis, but important for forecasting
   
10. r (Factors per Block) - Shape: {result.r.shape}
    - What: Number of factors in each block structure
    - Meaning: r[i] = number of factors in block i
    - Usage: Understand model structure
   
11. p (AR Lag Order) - Value: {result.p}
    - What: Number of lags in factor transition equation
    - Meaning: Typically p=1 (AR(1) dynamics)
    - Usage: Usually 1, higher values allow more complex dynamics
   
12. converged (Convergence Status) - Value: {result.converged}
    - What: Whether EM algorithm converged
    - Meaning: True = algorithm found stable solution
    - Usage: Check if training was successful
   
13. num_iter (Iterations) - Value: {result.num_iter}
    - What: Number of EM iterations performed
    - Meaning: How many times parameters were updated
    - Usage: Monitor training progress
   
14. loglik (Log-Likelihood) - Value: {loglik_str}
    - What: Final log-likelihood value
    - Meaning: Measure of model fit (higher is better)
    - Usage: Compare different models or parameter settings
    """)
    
    print("\n--- Practical Usage Examples ---")
    print("""
# Example 1: Extract the common factor (first factor)
common_factor = result.Z[:, 0]
print(f"Common factor range: [{common_factor.min():.2f}, {common_factor.max():.2f}]")

# Example 2: Find which series load most on first factor
loadings_on_factor1 = result.C[:, 0]
top_series_idx = np.argsort(np.abs(loadings_on_factor1))[-5:]  # Top 5
print("Top 5 series loading on factor 1:")
for idx in top_series_idx:
    print(f"  Series {idx}: loading = {loadings_on_factor1[idx]:.3f}")

# Example 3: Reconstruct a series from factors
series_idx = 0
reconstructed = result.Z @ result.C[series_idx, :].T
print(f"Reconstructed series {series_idx} from factors")

# Example 4: Check how well factors explain each series
# R[i, i] is the idiosyncratic variance (lower = better explained)
idiosyncratic_var = np.diag(result.R)
explained_variance = 1 - (idiosyncratic_var / (idiosyncratic_var + np.var(result.X_sm, axis=0)))
print(f"Variance explained by factors (per series):")
print(f"  Mean: {explained_variance.mean():.2%}")
print(f"  Min: {explained_variance.min():.2%}, Max: {explained_variance.max():.2%}")

# Example 5: Forecast factors forward
# Z_{t+h} = A^h @ Z_t
horizon = 12
Z_last = result.Z[-1, :]  # Last observed factor values
Z_forecast = np.zeros((horizon, len(Z_last)))
Z_forecast[0] = result.A @ Z_last
for h in range(1, horizon):
    Z_forecast[h] = result.A @ Z_forecast[h-1]
print(f"Forecasted factors for {horizon} periods ahead")

# Example 6: Convert smoothed data to polars DataFrame
if result.time_index is not None:
    smoothed_df = result.to_polars_smoothed()
    factors_df = result.to_polars_factors()
    print("Converted to polars DataFrames for easy analysis")
    """)

    # 4) Forecast
    print(f"\n--- Performing forecasts ---")
    forecast_horizon = cfg.get('forecast_horizon', None)
    if forecast_horizon is None:
        # Minimal horizon for quick demo (3 periods instead of full year)
        forecast_horizon = 3
        print(f"  Using minimal horizon: {forecast_horizon} periods (for quick demo)")
    pred_out = dfm.predict(horizon=forecast_horizon)
    
    if isinstance(pred_out, tuple):
        X_forecast, Z_forecast = pred_out
    else:
        X_forecast, Z_forecast = pred_out, None
    
    # Build forecast date index (generic based on clock frequency)
    from dfm_python.core.time import get_latest_time
    from dfm_python.core.timestamp import datetime_range, clock_to_datetime_freq, get_next_period_end
    last_date = get_latest_time(Time)
    # Get clock frequency and calculate next period end
    clock = config.clock if config else 'm'
    next_period = get_next_period_end(last_date, clock)
    datetime_freq = clock_to_datetime_freq(clock)
    forecast_dates = datetime_range(start=next_period, periods=forecast_horizon, freq=datetime_freq)
    
    print(f"✓ Forecast complete:")
    print(f"  - Horizon: {forecast_horizon} periods")
    print(f"  - X_forecast shape: {X_forecast.shape} (forecasted series)")
    print(f"    * X_forecast[t, i] = forecasted value of series i at time t")
    if Z_forecast is not None:
        print(f"  - Z_forecast shape: {Z_forecast.shape} (forecasted factors)")
        print(f"    * Z_forecast[t, j] = forecasted value of factor j at time t")
        print(f"    * Forecast method: Z_{{t+h}} = A^h @ Z_t (deterministic)")

    # 5) Visualize
    print(f"\n--- Visualizing factor forecast ---")
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dfm.plot(
        kind='factor',
        factor_index=0,
        forecast_horizon=forecast_horizon,
        save_path=str(output_dir / 'factor_forecast.png'),
        show=False
    )
    print(f"✓ Saved factor forecast plot: {output_dir / 'factor_forecast.png'}")

    # 6) Save forecasts
    print(f"\n--- Saving forecasts ---")
    from dfm_python.core.helpers import get_series_ids
    series_ids = get_series_ids(config) if config is not None else [
        f'series_{i}' for i in range(X_forecast.shape[1])
    ]
    # Create polars DataFrame
    forecast_dict = {'time': forecast_dates}
    for i, series_id in enumerate(series_ids):
        forecast_dict[series_id] = X_forecast[:, i].tolist()
    forecast_df = pl.DataFrame(forecast_dict)
    forecast_path = output_dir / 'forecasts.csv'
    forecast_df.write_csv(forecast_path)
    print(f"✓ Saved forecast CSV: {forecast_path}")

    # ========================================================================
    # 7) Additional Analysis: Understanding Your Results
    # ========================================================================
    print(f"\n" + "="*70)
    print("Additional Analysis: Understanding Your Results")
    print("="*70)
    
    # Factor analysis
    print("\n--- Factor Analysis ---")
    print(f"Number of factors: {result.num_factors()}")
    print(f"State dimension: {result.num_state()}")
    print(f"Number of series: {result.num_series()}")
    
    # Check factor stability
    A_eigenvals = np.linalg.eigvals(result.A).real
    print(f"\nFactor transition matrix (A) eigenvalues:")
    print(f"  Range: [{A_eigenvals.min():.4f}, {A_eigenvals.max():.4f}]")
    if np.all(np.abs(A_eigenvals) < 1):
        print("  ✓ All factors are stationary (stable)")
    else:
        print("  ⚠ Some factors may be non-stationary")
    
    # Factor loadings summary
    C_abs = np.abs(result.C)
    print(f"\nFactor loadings (C) summary:")
    print(f"  Mean absolute loading: {C_abs.mean():.4f}")
    print(f"  Max absolute loading: {C_abs.max():.4f}")
    print(f"  Series with weak loadings (<0.1): {np.sum(C_abs.max(axis=1) < 0.1)}")
    
    # Idiosyncratic variance
    R_diag = np.diag(result.R)
    print(f"\nIdiosyncratic variance (R diagonal):")
    print(f"  Mean: {R_diag.mean():.4f}")
    print(f"  Min: {R_diag.min():.4f}, Max: {R_diag.max():.4f}")
    print(f"  Lower values = series better explained by factors")
    
    # ========================================================================
    # 8) Model Result Storage (File-based)
    # ========================================================================
    print(f"\n" + "="*70)
    print("Model Result Storage (File-based)")
    print("="*70)
    
    model_results_dir = None
    try:
        from adapters import PickleModelResultSaver
        
        # Initialize file-based saver
        model_results_dir = output_dir / 'model_results'
        saver = PickleModelResultSaver(base_dir=str(model_results_dir))
        
        # Save model result
        result_id = saver.save_model_result(
            result=result,
            config=config,
            metadata={
                'tag': 'baseline',
                'max_iter': max_iter,
                'threshold': threshold,
                'tutorial': 'basic_tutorial'
            }
        )
        print(f"✓ Model result saved: {result_id}")
        print(f"  - Location: {model_results_dir / f'{result_id}.pkl'}")
        
        # List saved results
        saved_results = saver.list_model_results()
        print(f"  - Total saved results: {len(saved_results)}")
        
        # Load saved result (demonstration)
        loaded_result, loaded_config, loaded_metadata = saver.load_model_result(result_id)
        print(f"✓ Model result loaded successfully")
        print(f"  - Converged: {loaded_result.converged}")
        print(f"  - Tag: {loaded_metadata.get('tag', 'N/A')}")
        
    except ImportError:
        print("⚠ Adapters module not available (optional feature)")
    
    # ========================================================================
    # 9) Data View Management (File-based)
    # ========================================================================
    print(f"\n" + "="*70)
    print("Data View Management (File-based)")
    print("="*70)
    
    try:
        from adapters import BasicDataViewManager
        
        # Initialize data view manager with current data
        manager = BasicDataViewManager(data_source=(X, Time, dfm.get_original_data()))
        
        # Get data view at a specific date
        from dfm_python.core.time import get_latest_time
        view_date = get_latest_time(Time)
        if isinstance(view_date, datetime):
            view_date_str = view_date.strftime('%Y-%m-%d')
        else:
            view_date_str = str(view_date)
        
        X_view, Time_view, Z_view = manager.get_data_view(
            view_date=view_date_str,
            config=config
        )
        
        print(f"✓ Data view created for date: {view_date_str}")
        print(f"  - Original data shape: {X.shape}")
        print(f"  - View data shape: {X_view.shape}")
        print(f"  - Missing data in view: {np.isnan(X_view).sum() / X_view.size * 100:.2f}%")
        print(f"  - Note: Series with release_date > view_date are masked")
        
        # Demonstrate caching
        X_view2, _, _ = manager.get_data_view(view_date=view_date_str, config=config)
        print(f"✓ Data view cached (subsequent calls are faster)")
        
    except ImportError:
        print("⚠ Adapters module not available (optional feature)")
    
    # ========================================================================
    # 10) Spec CSV to YAML Conversion Workflow
    # ========================================================================
    print(f"\n" + "="*70)
    print("Spec CSV to YAML Conversion Workflow")
    print("="*70)
    print("""
If you have a spec CSV file, convert it to YAML first, then use Hydra:

Step 1: Convert CSV to YAML
  python -c "import dfm_python as dfm; dfm.from_spec('data/sample_spec.csv')"
  
  This creates:
    - config/series/sample_spec.yaml
    - config/blocks/sample_spec.yaml

Step 2: Create main config file (config/sample_spec.yaml)
  defaults:
    - series: sample_spec
    - blocks: sample_spec
    - _self_
  
  # Estimation parameters
  max_iter: 5000
  threshold: 1e-5
  clock: m
  # ... other parameters

Step 3: Use with Hydra
  python tutorial/basic_tutorial.py \\
    --config-path config \\
    --config-name sample_spec \\
    data_path=data/sample_data.csv

Note: Spec CSV is now only used for YAML conversion, not direct loading.
All configuration is managed through Hydra YAML files.
    """)
    
    # ========================================================================
    # 11) Nowcasting and Backtesting
    # ========================================================================
    print(f"\n" + "="*70)
    print("Nowcasting and Backtesting")
    print("="*70)
    
    try:
        from dfm_python import Nowcast
        from dfm_python.core.helpers import get_latest_time
        
        # Get Nowcast instance
        # The module-level DFM instance (_dfm_instance) has a nowcast property
        # that returns a cached Nowcast instance
        model_instance = dfm.DFM()  # This returns the singleton instance
        nowcast = model_instance.nowcast  # Access the nowcast property
        
        # Get target series and date for demonstration
        target_series = config.series[0].series_id if config.series else 'series_0'
        latest_date = get_latest_time(Time)
        
        print(f"\n--- Basic Nowcasting ---")
        print(f"Target series: {target_series}")
        print(f"Latest date: {latest_date}")
        
        # Simple nowcast calculation
        try:
            nowcast_value = nowcast(target_series, view_date=latest_date)
            print(f"✓ Nowcast value: {nowcast_value:.4f}")
            
            # Get full result with metadata
            nowcast_result = nowcast(
                target_series, 
                view_date=latest_date, 
                return_result=True
            )
            print(f"✓ Full nowcast result:")
            print(f"  - Nowcast value: {nowcast_result.nowcast_value:.4f}")
            if nowcast_result.data_availability:
                print(f"  - Data available: {nowcast_result.data_availability['n_available']} values")
                print(f"  - Data missing: {nowcast_result.data_availability['n_missing']} values")
        except Exception as e:
            print(f"⚠ Nowcast calculation failed: {e}")
        
        # News decomposition example
        print(f"\n--- News Decomposition ---")
        try:
            # Use dates from the data if available
            if len(Time) >= 2:
                # Get two different view dates
                view_date_old = Time[-10] if len(Time) >= 10 else Time[0]
                view_date_new = latest_date
                
                news = nowcast.decompose(
                    target_series=target_series,
                    target_period=latest_date,
                    view_date_old=view_date_old,
                    view_date_new=view_date_new
                )
                print(f"✓ News decomposition:")
                print(f"  - Old forecast: {news.y_old:.4f}")
                print(f"  - New forecast: {news.y_new:.4f}")
                print(f"  - Change: {news.change:.4f}")
                if len(news.top_contributors) > 0:
                    print(f"  - Top contributor: {news.top_contributors[0][0]} "
                          f"(impact: {news.top_contributors[0][1]:.4f})")
        except Exception as e:
            print(f"⚠ News decomposition failed: {e}")
        
        # Backtesting example
        print(f"\n--- Pseudo Real-Time Backtesting ---")
        try:
            # Perform a minimal backtest (2 steps for quick demo)
            backtest_result = nowcast.backtest(
                target_series=target_series,
                target_date=latest_date,
                backward_steps=2,  # Minimal: 2 steps for quick demo
                higher_freq=False,  # Use clock frequency
                include_actual=True
            )
            
            print(f"✓ Backtest complete:")
            print(f"  - Backward steps: {backtest_result.backward_steps}")
            print(f"  - Backward frequency: {backtest_result.backward_freq}")
            if backtest_result.overall_rmse is not None:
                print(f"  - Overall RMSE: {backtest_result.overall_rmse:.4f}")
                print(f"  - Overall MAE: {backtest_result.overall_mae:.4f}")
            print(f"  - Failed steps: {len(backtest_result.failed_steps)}")
            
            # Save backtest plot
            backtest_plot_path = output_dir / 'backtest_results.png'
            backtest_result.plot(save_path=str(backtest_plot_path), show=False)
            print(f"✓ Backtest plot saved: {backtest_plot_path}")
            
        except Exception as e:
            print(f"⚠ Backtesting failed: {e}")
            import traceback
            traceback.print_exc()
    
    except ImportError as e:
        print(f"⚠ Nowcasting features not available: {e}")
    
    # ========================================================================
    # 12) SQLite Storage (Optional - Future Feature)
    # ========================================================================
    print(f"\n" + "="*70)
    print("SQLite Storage (Optional - Future Feature)")
    print("="*70)
    
    print("Note: SQLite storage is planned for future releases.")
    print("  - File-based storage (PickleModelResultSaver) is fully functional")
    print("  - BasicDataViewManager provides in-memory data views with caching")
    print("  - For production use, file-based storage is recommended")
    print("  - SQLite integration will be available in future versions")
    
    print("\n" + "="*70)
    print("✓ Tutorial complete!")
    print("="*70)
    print(f"  - Output directory: {output_dir}")
    print(f"  - Generated files:")
    print(f"    * {output_dir / 'factor_forecast.png'}")
    print(f"    * {output_dir / 'forecasts.csv'}")
    if 'model_results_dir' in locals() and model_results_dir:
        print(f"    * {model_results_dir} (model results)")
    if 'backtest_plot_path' in locals():
        print(f"    * {backtest_plot_path} (backtest results)")
    print("\nKey Takeaways:")
    print("  - Use Hydra for all configuration management")
    print("  - YAML configs: config/default.yaml, config/series/*.yaml, config/blocks/*.yaml")
    print("  - Spec CSV: Convert to YAML first using dfm.from_spec()")
    print("  - CLI overrides: python script.py max_iter=10 threshold=1e-4")
    print("  - Nested overrides: blocks.Block_Global.factors=2")
    print("  - result.Z: Latent factors (common drivers)")
    print("  - result.C: How series relate to factors (loadings)")
    print("  - result.X_sm: Smoothed data (denoised observations)")
    print("  - result.A: How factors evolve over time")
    print("  - Use result.to_polars_factors() and to_polars_smoothed() for easy analysis")
    print("  - Model results can be saved/loaded using PickleModelResultSaver (file-based)")
    print("  - Data views can be managed using BasicDataViewManager (file-based)")
    print("  - Nowcasting: Use model.nowcast() for nowcast calculations and news decomposition")
    print("  - Backtesting: Use nowcast.backtest() for pseudo real-time evaluation")
    print("\nNext steps:")
    print("  - Try different config files: --config-name alternative")
    print("  - Override parameters via CLI: max_iter=20 threshold=1e-5")
    print("  - Use Hydra's composition features for complex configs")
    print("  - Convert spec CSV to YAML: dfm.from_spec('data/sample_spec.csv')")
    print("  - Analyze factors: result.Z[:, 0] for common factor")
    print("  - Check loadings: result.C[:, 0] to see which series drive factor 1")
    print("  - Save/load model results for later use")
    print("  - Use data views for pseudo real-time evaluation")
    print("  - Try nowcasting: nowcast = model.nowcast; value = nowcast('gdp')")
    print("  - Perform backtesting: result = nowcast.backtest('gdp', '2024Q4', backward_steps=20)")


if __name__ == "__main__":
    if not HYDRA_AVAILABLE:
        print("ERROR: Hydra is required for this tutorial.")
        print("Install with: pip install hydra-core")
        sys.exit(1)
    main()
