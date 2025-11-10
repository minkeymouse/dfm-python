#!/usr/bin/env python3
"""
DFM Python - Hydra Tutorial

This tutorial demonstrates the Hydra-based configuration approach:
1) Use @hydra.main decorator for configuration management
2) Access config values explicitly via cfg.* (e.g., cfg.max_iter, cfg.blocks.*)
3) Override parameters via CLI (e.g., max_iter=10, blocks.Block_Global.factors=2)
4) Load data, train, predict, and plot

This approach is ideal when:
- You want Hydra's powerful configuration composition
- You need CLI overrides for experimentation
- You want to manage complex config hierarchies

Run:
  python tutorial/hydra_tutorial.py \\
    --config-path ../config \\
    --config-name default \\
    data_path=../data/sample_data.csv \\
    sample_start=2021-01-01 \\
    sample_end=2022-12-31 \\
    max_iter=1 \\
    forecast_horizon=6

Or with overrides:
  python tutorial/hydra_tutorial.py \\
    max_iter=10 \\
    threshold=1e-4 \\
    damping_factor=0.9 \\
    blocks.Block_Global.factors=2
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("ERROR: Hydra is not installed. Install with: pip install hydra-core")
    print("This tutorial requires Hydra for configuration management.")
    exit(1)

import dfm_python as dfm


@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra decorator.
    
    The @hydra.main decorator automatically:
    - Loads config from config_path/config_name
    - Parses CLI overrides
    - Changes working directory to outputs/ (Hydra default)
    - Provides cfg as a DictConfig with all merged values
    """
    print("="*70)
    print("DFM Python - Hydra Tutorial")
    print("="*70)
    
    # ========================================================================
    # 1) Access configuration values explicitly via cfg.*
    # ========================================================================
    print("\n--- Configuration (accessed via cfg.*) ---")
    print(f"✓ Estimation parameters:")
    print(f"  - max_iter: {cfg.max_iter}")
    print(f"  - threshold: {cfg.threshold}")
    print(f"  - nan_method: {cfg.nan_method}, nan_k: {cfg.nan_k}")
    print(f"  - clock: {cfg.clock}")
    print(f"  - ar_lag: {cfg.ar_lag}")
    
    print(f"\n✓ Numerical stability parameters:")
    print(f"  - regularization_scale: {cfg.regularization_scale}")
    print(f"  - damping_factor: {cfg.damping_factor}")
    print(f"  - clip_ar_coefficients: {cfg.clip_ar_coefficients}")
    print(f"  - clip_data_values: {cfg.clip_data_values}")
    print(f"  - use_regularization: {cfg.use_regularization}")
    print(f"  - use_damped_updates: {cfg.use_damped_updates}")
    
    # Access blocks explicitly
    print(f"\n✓ Block structure (accessed via cfg.blocks.*):")
    if hasattr(cfg, 'blocks') and cfg.blocks is not None:
        for block_name, block_cfg in cfg.blocks.items():
            # Handle both underscore and hyphen in block names
            factors = block_cfg.get('factors', 1)
            clock = block_cfg.get('clock', 'm')
            print(f"  - {block_name}: factors={factors}, clock={clock}")
    else:
        print("  - No blocks found in config")
    
    # Access series count (if available in config)
    if hasattr(cfg, 'series') and cfg.series is not None:
        if isinstance(cfg.series, dict):
            print(f"\n✓ Series: {len(cfg.series)} series defined")
        elif isinstance(cfg.series, list):
            print(f"\n✓ Series: {len(cfg.series)} series defined")
    
    # ========================================================================
    # 2) Load configuration into DFM
    # ========================================================================
    print(f"\n--- Loading DFM configuration from Hydra cfg ---")
    dfm.load_config(hydra=cfg)
    config = dfm.get_config()
    if config is None:
        raise ValueError("Configuration not loaded")
    print(f"✓ DFM config loaded:")
    print(f"  - Series: {len(config.series)}")
    print(f"  - Blocks: {len(config.block_names)} ({', '.join(config.block_names)})")
    print(f"  - Clock: {config.clock}")
    
    # ========================================================================
    # 3) Load data (resolve paths relative to original working directory)
    # ========================================================================
    print(f"\n--- Loading data ---")
    # Get original working directory (before Hydra changes it)
    original_cwd = Path(get_original_cwd())
    
    # Resolve data path (can be provided via CLI: data_path=../data/sample_data.csv)
    data_path = cfg.get('data_path', 'data/sample_data.csv')
    if not Path(data_path).is_absolute():
        data_path = original_cwd / data_path
    
    sample_start = cfg.get('sample_start', '2021-01-01')
    sample_end = cfg.get('sample_end', '2022-12-31')
    
    print(f"  - Data path: {data_path}")
    print(f"  - Sample window: {sample_start} to {sample_end}")
    
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
    print(f"  - Time range: {Time[0]} ~ {Time[-1]}")
    print(f"  - Missing data ratio: {pd.isna(X).sum().sum() / X.size * 100:.2f}%")
    
    # ========================================================================
    # 4) Train (use cfg.max_iter explicitly)
    # ========================================================================
    print(f"\n--- Training model ---")
    print(f"  - Using max_iter from cfg: {cfg.max_iter}")
    print(f"  - Using threshold from cfg: {cfg.threshold}")
    dfm.train(max_iter=cfg.max_iter)  # Explicitly use cfg.max_iter
    result = dfm.get_result()
    if result is None:
        raise ValueError("Model training failed - no result available")
    print(f"✓ Trained:")
    print(f"  - Iterations: {result.num_iter}")
    print(f"  - Converged: {result.converged}")
    print(f"  - Factors: {result.Z.shape[1]}")
    if hasattr(result, 'loglik') and result.loglik is not None:
        loglik_val = result.loglik
        if loglik_val is not None and np.isfinite(loglik_val):
            print(f"  - Final log-likelihood: {loglik_val:.2f}")
    
    # ========================================================================
    # 5) Forecast
    # ========================================================================
    print(f"\n--- Performing forecasts ---")
    forecast_horizon = cfg.get('forecast_horizon', 12)
    pred_out = dfm.predict(forecast_horizon)
    if isinstance(pred_out, tuple):
        X_forecast, Z_forecast = pred_out
    else:
        X_forecast, Z_forecast = pred_out, None
    
    # Build forecast date index
    last_date = pd.to_datetime(Time[-1])
    try:
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset('ME'),
            periods=forecast_horizon, freq='ME'
        )
    except Exception:
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=30),
            periods=forecast_horizon, freq='ME'
        )
    print(f"✓ Forecast complete:")
    print(f"  - Horizon: {forecast_horizon} periods")
    print(f"  - X_forecast shape: {X_forecast.shape}")
    if Z_forecast is not None:
        print(f"  - Z_forecast shape: {Z_forecast.shape}")
    
    # ========================================================================
    # 6) Visualize
    # ========================================================================
    print(f"\n--- Visualizing factor forecast ---")
    # Hydra changes working directory, so save to current directory
    plot_path = Path('factor_forecast.png')
    dfm.plot(
        kind='factor',
        factor_index=0,
        forecast_horizon=forecast_horizon,
        save_path=str(plot_path),
        show=False
    )
    print(f"✓ Saved factor forecast plot: {plot_path}")
    
    # ========================================================================
    # 7) Save forecasts
    # ========================================================================
    print(f"\n--- Saving forecasts ---")
    series_ids = config.get_series_ids() if config is not None else [
        f'series_{i}' for i in range(X_forecast.shape[1])
    ]
    forecast_df = pd.DataFrame(
        X_forecast,
        index=forecast_dates,
        columns=series_ids
    )
    forecast_path = Path('forecasts.csv')
    forecast_df.to_csv(forecast_path)
    print(f"✓ Saved forecast CSV: {forecast_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("✓ Tutorial complete!")
    print("="*70)
    print(f"  - Output directory: {Path.cwd()} (Hydra changes working directory)")
    print(f"  - Generated files:")
    print(f"    * {plot_path}")
    print(f"    * {forecast_path}")
    print("\nHydra CLI override examples:")
    print("  # Change max_iter:")
    print("  python tutorial/hydra_tutorial.py max_iter=10")
    print("  # Change threshold:")
    print("  python tutorial/hydra_tutorial.py threshold=1e-4")
    print("  # Change block factors:")
    print("  python tutorial/hydra_tutorial.py blocks.Block_Global.factors=2")
    print("  # Multiple overrides:")
    print("  python tutorial/hydra_tutorial.py max_iter=10 threshold=1e-4 damping_factor=0.9")
    print("\nSee basic_tutorial.py for Spec+Params approach (no Hydra required)")


if __name__ == "__main__":
    if not HYDRA_AVAILABLE:
        print("ERROR: Hydra is required for this tutorial.")
        exit(1)
    main()

