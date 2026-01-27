"""Tutorial: DFM for Finance Data

This tutorial demonstrates the complete workflow for training and prediction
using finance data with market_forward_excess_returns as the target variable.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from datetime import datetime
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="dfm_finance")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("DFM Tutorial: Finance Data")
    print("=" * 80)

    print("\n[Step 1] Loading data...")
    df = pd.read_csv(project_root / "data" / "finance.csv")
    print(f"   Data shape: {df.shape}")
    
    # Truncate to ~1000 timesteps for faster testing (use most recent data)
    # This reduces complexity from O(T × m³) while still providing enough data
    if len(df) > 1000:
        df = df.iloc[-1000:].copy()
        df = df.reset_index(drop=True)
        print(f"   Truncated to {len(df)} timesteps (most recent) for faster testing")

    # Build a daily date index (finance.csv only has date_id=0..T-1)
    if "date" not in df.columns:
        start_date = datetime(1980, 1, 1)
        df["date"] = pd.date_range(start=start_date, periods=len(df), freq="D")

    print("\n[Step 2] Creating Dataset...")
    config = DFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    
    # Filter data to only include series specified in blocks (reduces observation dimension)
    # Collect all unique series from all blocks
    all_block_series = set()
    if hasattr(config, 'blocks') and config.blocks:
        for block_name, block_config in config.blocks.items():
            if isinstance(block_config, dict) and 'series' in block_config:
                all_block_series.update(block_config['series'])
    
    # Keep only series in blocks + time index + any required metadata columns
    columns_to_keep = ['date'] + list(all_block_series)
    # Filter to only columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df_filtered = df[columns_to_keep].copy()
    print(f"   Filtered to {len(columns_to_keep) - 1} series (from {len(df.columns) - 1}) based on block configuration")
    
    dataset = DFMDataset(config=config, data=df_filtered, time_index="date")

    print("\n[Step 3] Training DFM model...")
    model = DFM(dataset=dataset, config=config, scaler=StandardScaler())
    model.fit()

    result = model.result
    print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")

    print("\n[Step 4] Making predictions...")
    X_forecast, Z_forecast = model.predict(horizon=6)
    print(f"   Forecast shape: {X_forecast.shape}")

    print("\n[Step 5] Saving model...")
    model_path = project_root / "models" / "dfm_finance.pkl"
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    print(f"   Model saved to: {model_path}")

    print("\n" + result.summary())


if __name__ == "__main__":
    main()
