"""Tutorial: DFM for Macro Data

This tutorial demonstrates the complete workflow for training, prediction, and nowcasting
using macro data with KOEQUIPTE as the target variable.

Target: KOEQUIPTE (Investment, Equipment, Estimation, SA)
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python import DFM, DFMDataModule, DFMTrainer
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.utils.time import TimeIndex, parse_timestamp

print("=" * 80)
print("DFM Tutorial: Macro Data")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[Step 1] Loading macro data...")
data_path = project_root / "data" / "macro.csv"
df = pd.read_csv(data_path)

print(f"   Data shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

# ============================================================================
# Step 2: Prepare Data
# ============================================================================
print("\n[Step 2] Preparing data...")

# Target variable
target_col = "KOEQUIPTE"

# Select a subset of series for faster execution
# Use a mix of different categories: employment, consumption, investment, etc.
selected_cols = [
    # Employment
    "KOEMPTOTO", "KOEMPWREP", "KOHWRWEMP",
    # Consumption
    "KOWRCCNSE", "KOWRCDURE", "KOWRCSEME",
    # Investment
    "KOIMPCONA", "KOIPALL.G",
    # Production
    "KOCONPRCF", "KOCPCOREF",
    # Target
    target_col
]

# Filter to only columns that exist in the data
selected_cols = [col for col in selected_cols if col in df.columns]

# Filter data
df_processed = df[selected_cols + ["date"]].copy()
print(f"   Selected {len(selected_cols)} series (including target)")
print(f"   Series: {selected_cols[:5]}...")

# Parse date column
df_processed["date"] = pd.to_datetime(df_processed["date"])
df_processed = df_processed.sort_values("date")

# Remove date column for processing
date_col = df_processed["date"].copy()
df_processed = df_processed.drop(columns=["date"])

# Remove rows with all NaN
df_processed = df_processed.dropna(how='all')
print(f"   Data shape after cleaning: {df_processed.shape}")

# ============================================================================
# Step 3: Create Configuration
# ============================================================================
print("\n[Step 3] Creating configuration...")

# Create series configs
series_configs = []
for col in selected_cols:
    if col == target_col:
        # Target series - use chg transformation (as per series config)
        series_configs.append(
            SeriesConfig(
                series_id=col,
                frequency="m",
                transformation="chg",  # As per KOEQUIPTE.yaml
                blocks=[DEFAULT_BLOCK_NAME]
            )
        )
    else:
        # Predictor series - use lin for simplicity
        series_configs.append(
            SeriesConfig(
                series_id=col,
                frequency="m",
                transformation="lin",
                blocks=[DEFAULT_BLOCK_NAME]
            )
        )

# Create blocks config
blocks_config = {
    DEFAULT_BLOCK_NAME: {
        "factors": 2,  # Small number for fast execution
        "ar_lag": 1,
        "clock": "m"
    }
}

# Create DFM config
config = DFMConfig(
    series=series_configs,
    blocks=blocks_config,
    max_iter=10,  # Small number for fast execution
    threshold=1e-4
)

print(f"   Number of series: {len(series_configs)}")
print(f"   Number of factors: {config.blocks[DEFAULT_BLOCK_NAME]['factors']}")
print(f"   Target series: {target_col}")

# ============================================================================
# Step 4: Create DataModule
# ============================================================================
print("\n[Step 4] Creating DataModule...")

# Create time index from date column
# Align with processed data (after dropping NaN rows)
valid_dates = date_col.iloc[:len(df_processed)].values
time_index = TimeIndex([parse_timestamp(str(d)) for d in valid_dates])

# Create DataModule
data_module = DFMDataModule(
    config=config,
    data=df_processed.values,
    time=time_index
)
data_module.setup()

print(f"   DataModule created successfully")
if hasattr(data_module, 'data_processed') and data_module.data_processed is not None:
    print(f"   Processed data shape: {data_module.data_processed.shape}")
else:
    print(f"   Data shape: {df_processed.shape}")
print(f"   Time range: {time_index[0]} to {time_index[-1]}")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n[Step 5] Training DFM model...")

model = DFM()
model._config = config  # Set config directly

trainer = DFMTrainer(max_epochs=10)  # Small number for fast execution
trainer.fit(model, data_module)

print("   Training completed!")

# ============================================================================
# Step 6: Prediction
# ============================================================================
print("\n[Step 6] Making predictions...")

X_forecast = None
Z_forecast = None
X_forecast_history = None
Z_forecast_history = None

try:
    # Predict with default horizon
    X_forecast, Z_forecast = model.predict(horizon=6)
    
    print(f"   Forecast shape: {X_forecast.shape}")
    print(f"   Factor forecast shape: {Z_forecast.shape}")
    
    # Find target index
    target_idx = selected_cols.index(target_col)
    print(f"   First forecast value (target {target_col}): {X_forecast[0, target_idx]:.6f}")
    
    # Predict with history parameter (using recent 60 periods)
    X_forecast_history, Z_forecast_history = model.predict(horizon=6, history=60)
    
    print(f"   Forecast with history shape: {X_forecast_history.shape}")
    print(f"   First forecast with history (target): {X_forecast_history[0, target_idx]:.6f}")
    
except ValueError as e:
    print(f"   Prediction failed: {e}")
    print("   Note: This may indicate numerical instability. Try:")
    print("   - Using more training iterations")
    print("   - Adjusting data transformations")
    print("   - Using different factor configurations")

# ============================================================================
# Step 7: Nowcasting
# ============================================================================
print("\n[Step 7] Nowcasting...")

try:
    # Get nowcast manager
    # Note: nowcast requires src.nowcasting module which may not be available in dfm-python
    # This is expected if using dfm-python standalone
    nowcast = model.nowcast
    
    # Calculate nowcast for target series
    # Use latest available date
    latest_date = time_index[-1]
    view_date = latest_date
    
    nowcast_value = nowcast(
        target_series=target_col,
        view_date=view_date,
        target_period=latest_date
    )
    
    print(f"   Nowcast value for {target_col}: {nowcast_value:.6f}")
    print(f"   View date: {view_date}")
    
except (ValueError, ImportError) as e:
    print(f"   Nowcasting skipped: {e}")
    print("   Note: Nowcasting requires src.nowcasting module from main project")

# ============================================================================
# Step 8: Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {len(selected_cols)} series")
print(f"✅ Model trained: {len(series_configs)} series, {config.blocks[DEFAULT_BLOCK_NAME]['factors']} factors")
if X_forecast is not None:
    print(f"✅ Predictions generated: {X_forecast.shape[0]} periods ahead")
else:
    print(f"⚠️  Predictions: Failed (see error message above)")
print(f"✅ Target series: {target_col}")
print("=" * 80)

