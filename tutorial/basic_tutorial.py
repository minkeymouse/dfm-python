#!/usr/bin/env python3
"""
DFM Python - Basic Tutorial (Spec + Params approach)

This tutorial demonstrates the Spec CSV + Params approach:
1) Load config from spec CSV with Params dataclass
2) Load data with a short window
3) Train the model
4) Predict and visualize
5) Save forecasts to CSV
6) Understand DFM components and results

This approach is ideal when:
- You have series definitions in a CSV file
- You want to control main settings programmatically via Params
- You don't need Hydra's advanced features
- You prefer a simple, straightforward workflow

Run:
  python tutorial/basic_tutorial.py \\
    --spec data/sample_spec.csv \\
    --data data/sample_data.csv \\
    --output outputs \\
    --sample-start 2021-01-01 --sample-end 2022-12-31 \\
    --max-iter 10 --forecast-horizon 12

For quick testing:
  python tutorial/basic_tutorial.py \\
    --spec data/sample_spec.csv \\
    --data data/sample_data.csv \\
    --max-iter 1
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import dfm_python as dfm
from dfm_python.config import Params, DFMConfig, SeriesConfig, BlockConfig
from dfm_python.core.diagnostics import (
    evaluate_factor_estimation,
    evaluate_loading_estimation,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DFM Python - Basic Tutorial (Spec + Params)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python tutorial/basic_tutorial.py --spec data/sample_spec.csv --data data/sample_data.csv

  # Custom parameters
  python tutorial/basic_tutorial.py \\
    --spec data/sample_spec.csv \\
    --data data/sample_data.csv \\
    --max-iter 10 \\
    --threshold 1e-4 \\
    --damping-factor 0.9
        """
    )
    parser.add_argument("--spec", type=str, required=True,
                       help="Path to spec CSV (series definitions). Required columns: series_id, series_name, frequency, transformation, category, units, plus Block_* columns.")
    parser.add_argument("--data", type=str, default="data/sample_data.csv",
                       help="Path to data CSV")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--sample-start", type=str, default="1990-01-01",
                       help="Sample start date (YYYY-MM-DD)")
    parser.add_argument("--sample-end", type=str, default="2022-12-31",
                       help="Sample end date (YYYY-MM-DD)")
    # Estimation parameters (exposed as CLI arguments)
    parser.add_argument("--max-iter", type=int, default=10,
                       help="Maximum EM iterations (default: 10)")
    parser.add_argument("--threshold", type=float, default=1e-5,
                       help="EM convergence threshold (default: 1e-5)")
    parser.add_argument("--nan-method", type=int, default=2,
                       help="Missing data handling method (1-5, default: 2 = spline)")
    parser.add_argument("--nan-k", type=int, default=3,
                       help="Spline parameter for NaN interpolation (default: 3)")
    parser.add_argument("--clock", type=str, default="m",
                       choices=['d', 'w', 'm', 'q', 'sa', 'a'],
                       help="Base frequency for latent factors (default: 'm' for monthly)")
    # Numerical stability parameters
    parser.add_argument("--regularization-scale", type=float, default=1e-5,
                       help="Regularization scale factor (default: 1e-5)")
    parser.add_argument("--damping-factor", type=float, default=0.8,
                       help="Damping factor when likelihood decreases (default: 0.8)")
    parser.add_argument("--forecast-horizon", type=int, default=12,
                       help="Forecast horizon (periods, default: 12)")
    return parser.parse_args()


def main() -> None:
    print("="*70)
    print("DFM Python - Basic Tutorial (Spec + Params)")
    print("="*70)
    args = parse_args()

    data_file = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Create Params object with all exposed parameters
    # This demonstrates all available parameters you can control
    params = Params(
        # Estimation parameters
        ar_lag=1,
        threshold=args.threshold,
        max_iter=args.max_iter,
        nan_method=args.nan_method,
        nan_k=args.nan_k,
        clock=args.clock,
        # Numerical stability - AR clipping
        clip_ar_coefficients=True,
        ar_clip_min=-0.99,
        ar_clip_max=0.99,
        warn_on_ar_clip=True,
        # Numerical stability - Data clipping
        clip_data_values=True,
        data_clip_threshold=100.0,
        warn_on_data_clip=True,
        # Numerical stability - Regularization
        use_regularization=True,
        regularization_scale=args.regularization_scale,
        min_eigenvalue=1e-8,
        max_eigenvalue=1e6,
        warn_on_regularization=True,
        # Numerical stability - Damped updates
        use_damped_updates=True,
        damping_factor=args.damping_factor,
        warn_on_damped_update=True,
    )
    print(f"\n✓ Params created:")
    print(f"  - max_iter={params.max_iter}, threshold={params.threshold}")
    print(f"  - clock={params.clock}, nan_method={params.nan_method}")
    print(f"  - regularization_scale={params.regularization_scale}")
    print(f"  - damping_factor={params.damping_factor}")

    # 2) Load configuration from spec CSV + Params
    # The spec CSV defines all series and their block memberships
    # The Params object provides main settings (threshold, max_iter, etc.)
    print(f"\n--- Loading configuration from spec CSV ---")
    dfm.from_spec(args.spec, params=params)
    config = dfm.get_config()
    if config is None:
        raise ValueError("Configuration not loaded")
    print(f"✓ Config loaded:")
    print(f"  - Series: {len(config.series)}")
    print(f"  - Blocks: {len(config.block_names)} ({', '.join(config.block_names)})")
    print(f"  - Clock: {config.clock}")
    print(f"  - max_iter: {config.max_iter}, threshold: {config.threshold}")

    # 3) Load data
    print(f"\n--- Loading data ---")
    dfm.load_data(str(data_file), sample_start=args.sample_start, sample_end=args.sample_end)
    X = dfm.get_data()
    Time = dfm.get_time()
    if X is None or Time is None:
        raise ValueError("Data not loaded")
    print(f"✓ Data loaded:")
    print(f"  - Shape: {X.shape} (time periods × series)")
    print(f"  - Time range: {Time.iloc[0] if hasattr(Time, 'iloc') else Time[0]} ~ {Time.iloc[-1] if hasattr(Time, 'iloc') else Time[-1]}")
    print(f"  - Missing data ratio: {pd.isna(X).sum().sum() / X.size * 100:.2f}%")

    # 4) Train
    print(f"\n--- Training model ---")
    dfm.train(max_iter=args.max_iter)  # Can override here or use config.max_iter
    result = dfm.get_result()
    if result is None:
        raise ValueError("Model training failed - no result available")
    print(f"✓ Trained:")
    print(f"  - Iterations: {result.num_iter}")
    print(f"  - Converged: {result.converged}")
    print(f"  - Factors: {result.Z.shape[1]} (state dimension: {result.Z.shape[1]})")
    if hasattr(result, 'loglik') and result.loglik is not None:
        loglik_val = result.loglik
        if loglik_val is not None and np.isfinite(loglik_val):
            print(f"  - Final log-likelihood: {loglik_val:.2f}")

    # ------------------------------------------------------------------
    # Synthetic data test for factor/loading estimation accuracy (always run)
    # ------------------------------------------------------------------
    print(f"\n" + "="*70)
    print("Synthetic Data Test: Factor/Loading Estimation Accuracy")
    print("="*70)

    # Synthetic data generation using actual config structure
    # Use the same number of series and block structure as the real data
    np.random.seed(42)
    T_syn = 100
    N_syn = len(config.series)  # Use actual number of series from config
    
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
    series_syn = [
        SeriesConfig(
            series_id=f"syn_{s.series_id}",
            frequency=s.frequency,  # Use actual frequency
            transformation=s.transformation,  # Use actual transformation
            blocks=s.blocks,  # Use actual block membership
        )
        for s in config.series
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

    # Fit model to synthetic data
    print("\n--- Fitting model on synthetic data ---")
    model_syn = dfm.DFM()
    result_syn = model_syn.fit(
        X_syn,
        config_syn,
        max_iter=max(50, args.max_iter),
        threshold=min(1e-4, args.threshold),
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
    
    # Format loglik for display
    loglik_str = f"{result.loglik:.2f}" if (hasattr(result, 'loglik') and result.loglik is not None and np.isfinite(result.loglik)) else 'N/A'
    
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

# Example 6: Convert smoothed data to pandas DataFrame
if result.time_index is not None:
    smoothed_df = result.to_pandas_smoothed()
    factors_df = result.to_pandas_factors()
    print("Converted to pandas DataFrames for easy analysis")
    """)

    # 5) Forecast
    print(f"\n--- Performing forecasts ---")
    pred_out = dfm.predict(args.forecast_horizon)
    if isinstance(pred_out, tuple):
        X_forecast, Z_forecast = pred_out
    else:
        X_forecast, Z_forecast = pred_out, None
    # Build forecast date index for saving
    last_date = pd.to_datetime(Time.iloc[-1] if hasattr(Time, 'iloc') else Time[-1])
    try:
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset('ME'),
            periods=args.forecast_horizon, freq='ME'
        )
    except Exception:
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=30),
            periods=args.forecast_horizon, freq='ME'
        )
    print(f"✓ Forecast complete:")
    print(f"  - Horizon: {args.forecast_horizon} periods")
    print(f"  - X_forecast shape: {X_forecast.shape} (forecasted series)")
    print(f"    * X_forecast[t, i] = forecasted value of series i at time t")
    if Z_forecast is not None:
        print(f"  - Z_forecast shape: {Z_forecast.shape} (forecasted factors)")
        print(f"    * Z_forecast[t, j] = forecasted value of factor j at time t")
        print(f"    * Forecast method: Z_{{t+h}} = A^h @ Z_t (deterministic)")

    # 6) Visualize
    print(f"\n--- Visualizing factor forecast ---")
    dfm.plot(
        kind='factor',
        factor_index=0,
        forecast_horizon=args.forecast_horizon,
        save_path=output_dir / 'factor_forecast.png',
        show=False
    )
    print(f"✓ Saved factor forecast plot: {output_dir / 'factor_forecast.png'}")

    # 7) Save forecasts
    print(f"\n--- Saving forecasts ---")
    series_ids = config.get_series_ids() if config is not None else [
        f'series_{i}' for i in range(X_forecast.shape[1])
    ]
    forecast_df = pd.DataFrame(
        X_forecast,
        index=forecast_dates,
        columns=series_ids
    )
    forecast_path = output_dir / 'forecasts.csv'
    forecast_df.to_csv(forecast_path)
    print(f"✓ Saved forecast CSV: {forecast_path}")

    # ========================================================================
    # 8) Additional Analysis: Understanding Your Results
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
    
    print("\n" + "="*70)
    print("✓ Tutorial complete!")
    print("="*70)
    print(f"  - Output directory: {output_dir}")
    print(f"  - Generated files:")
    print(f"    * {output_dir / 'factor_forecast.png'}")
    print(f"    * {output_dir / 'forecasts.csv'}")
    print("\nKey Takeaways:")
    print("  - result.Z: Latent factors (common drivers)")
    print("  - result.C: How series relate to factors (loadings)")
    print("  - result.X_sm: Smoothed data (denoised observations)")
    print("  - result.A: How factors evolve over time")
    print("  - Use result.to_pandas_factors() and to_pandas_smoothed() for easy analysis")
    print("\nNext steps:")
    print("  - Try different parameters: --max-iter 10 --threshold 1e-4")
    print("  - Adjust regularization: --regularization-scale 1e-6")
    print("  - Change damping: --damping-factor 0.9")
    print("  - Analyze factors: result.Z[:, 0] for common factor")
    print("  - Check loadings: result.C[:, 0] to see which series drive factor 1")
    print("  - See hydra_tutorial.py for Hydra-based configuration")


if __name__ == "__main__":
    main()
