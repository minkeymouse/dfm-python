#!/usr/bin/env python3
"""
DFM Python - Hydra Tutorial

This tutorial demonstrates the Hydra decorator approach:
1) Use @hydra.main decorator for configuration
2) Load config from Hydra DictConfig
3) Load data and train
4) Predict and visualize

This approach is ideal when:
- You want to use Hydra's powerful configuration management
- You need CLI overrides and configuration composition
- You're working with complex configuration hierarchies

Run:
  python tutorial/hydra_tutorial.py \\
    --config-path config \\
    --config-name default \\
    data_path=data/sample_data.csv \\
    max_iter=10 \\
    threshold=1e-4

Or with overrides:
  python tutorial/hydra_tutorial.py \\
    max_iter=20 \\
    threshold=1e-5 \\
    blocks.Block_Global.factors=2
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
    print("ERROR: Hydra is not installed. Install with: pip install hydra-core")
    sys.exit(1)

import numpy as np
import pandas as pd
import dfm_python as dfm


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for Hydra-based DFM tutorial.
    
    This function demonstrates:
    1. Loading configuration from Hydra
    2. Loading data with path resolution
    3. Training the model
    4. Forecasting and visualization
    5. Accessing results
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all DFM parameters.
        Can be overridden via CLI arguments.
    """
    print("="*70)
    print("DFM Python - Hydra Tutorial")
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
    print(f"  - Time range: {Time[0]} ~ {Time[-1]}")
    print(f"  - Missing data ratio: {pd.isna(X).sum().sum() / X.size * 100:.2f}%")
    
    # 3) Train
    print(f"\n--- Training model ---")
    # Use max_iter from config or override
    max_iter = cfg.get('max_iter', config.max_iter)
    threshold = cfg.get('threshold', config.threshold)
    
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
    
    # 4) Forecast
    print(f"\n--- Performing forecasts ---")
    forecast_horizon = cfg.get('forecast_horizon', 12)
    pred_out = dfm.predict(horizon=forecast_horizon)
    
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
    print(f"  - X_forecast shape: {X_forecast.shape} (forecasted series)")
    if Z_forecast is not None:
        print(f"  - Z_forecast shape: {Z_forecast.shape} (forecasted factors)")
    
    # 5) Visualize
    print(f"\n--- Visualizing results ---")
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot factor forecast
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
    
    # 7) Additional analysis
    print(f"\n--- Model Analysis ---")
    print(f"Number of factors: {result.num_factors()}")
    print(f"State dimension: {result.num_state()}")
    print(f"Number of series: {result.num_series()}")
    
    # Factor stability check
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
    
    # Idiosyncratic variance
    R_diag = np.diag(result.R)
    print(f"\nIdiosyncratic variance (R diagonal):")
    print(f"  Mean: {R_diag.mean():.4f}")
    print(f"  Min: {R_diag.min():.4f}, Max: {R_diag.max():.4f}")
    
    print("\n" + "="*70)
    print("✓ Hydra tutorial complete!")
    print("="*70)
    print(f"  - Output directory: {output_dir}")
    print(f"  - Generated files:")
    print(f"    * {output_dir / 'factor_forecast.png'}")
    print(f"    * {output_dir / 'forecasts.csv'}")
    print("\nKey Takeaways:")
    print("  - Use Hydra for flexible configuration management")
    print("  - CLI overrides: python script.py max_iter=10 threshold=1e-4")
    print("  - Nested overrides: blocks.Block_Global.factors=2")
    print("  - Access results: result.Z, result.C, result.X_sm")
    print("\nNext steps:")
    print("  - Try different config files: --config-name alternative")
    print("  - Override parameters via CLI")
    print("  - Use Hydra's composition features for complex configs")
    print("  - See basic_tutorial.py for Spec CSV + Params approach")


if __name__ == "__main__":
    if not HYDRA_AVAILABLE:
        print("ERROR: Hydra is required for this tutorial.")
        print("Install with: pip install hydra-core")
        sys.exit(1)
    main()
