#!/usr/bin/env python3
"""
DFM Python - Basic Tutorial (Spec + Params approach)

This tutorial demonstrates the Spec CSV + Params approach:
1) Load config from spec CSV with Params dataclass
2) Load data with a short window
3) Train quickly (max_iter=1 by default)
4) Predict and Plot
5) Save forecasts to CSV

This approach is ideal when:
- You have series definitions in a CSV file
- You want to control main settings programmatically via Params
- You don't need Hydra's advanced features

Run:
  python tutorial/basic_tutorial.py \\
    --spec data/sample_spec.csv \\
    --data data/sample_data.csv \\
    --output outputs \\
    --sample-start 2021-01-01 --sample-end 2022-12-31 \\
    --max-iter 1 --forecast-horizon 6
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import dfm_python as dfm
from dfm_python.config import Params

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
    parser.add_argument("--sample-start", type=str, default="2021-01-01",
                       help="Sample start date (YYYY-MM-DD)")
    parser.add_argument("--sample-end", type=str, default="2022-12-31",
                       help="Sample end date (YYYY-MM-DD)")
    # Estimation parameters (exposed as CLI arguments)
    parser.add_argument("--max-iter", type=int, default=1,
                       help="Maximum EM iterations (default: 1 for quick testing)")
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
    print(f"  - Time range: {Time[0]} ~ {Time[-1]}")
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

    # 5) Forecast
    print(f"\n--- Performing forecasts ---")
    pred_out = dfm.predict(args.forecast_horizon)
    if isinstance(pred_out, tuple):
        X_forecast, Z_forecast = pred_out
    else:
        X_forecast, Z_forecast = pred_out, None
    # Build forecast date index for saving
    last_date = pd.to_datetime(Time[-1])
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
    print(f"  - X_forecast shape: {X_forecast.shape}")
    if Z_forecast is not None:
        print(f"  - Z_forecast shape: {Z_forecast.shape}")

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

    print("\n" + "="*70)
    print("✓ Tutorial complete!")
    print("="*70)
    print(f"  - Output directory: {output_dir}")
    print(f"  - Generated files:")
    print(f"    * {output_dir / 'factor_forecast.png'}")
    print(f"    * {output_dir / 'forecasts.csv'}")
    print("\nNext steps:")
    print("  - Try different parameters: --max-iter 10 --threshold 1e-4")
    print("  - Adjust regularization: --regularization-scale 1e-6")
    print("  - Change damping: --damping-factor 0.9")
    print("  - See hydra_tutorial.py for Hydra-based configuration")


if __name__ == "__main__":
    main()
