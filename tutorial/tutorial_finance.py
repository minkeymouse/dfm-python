"""Tutorial: DFM for Finance Data

This tutorial demonstrates the complete workflow for training, prediction, and nowcasting
using finance data with market_forward_excess_returns as the target variable.

Target: market_forward_excess_returns
Excluded: risk_free_rate, forward_returns
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
print("DFM Tutorial: Finance Data")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[Step 1] Loading finance data...")
data_path = project_root / "data" / "finance.csv"
df = pd.read_csv(data_path)

print(f"   Data shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

# ============================================================================
# Step 2: Prepare Data
# ============================================================================
print("\n[Step 2] Preparing data...")

# Exclude target and excluded variables from predictors
target_col = "market_forward_excess_returns"
exclude_cols = ["risk_free_rate", "forward_returns", "date_id"]

# Select a subset of series for faster execution
# Use first few series from each category: D, E, I, M, P, S, V
selected_cols = []
for prefix in ["D", "E", "I", "M", "P", "S", "V"]:
    for i in range(1, 6):  # Use first 5 from each category
        col = f"{prefix}{i}"
        if col in df.columns:
            selected_cols.append(col)

# Add target to selected columns
if target_col not in selected_cols:
    selected_cols.append(target_col)

# Filter data
df_processed = df[selected_cols].copy()
print(f"   Selected {len(selected_cols)} series (including target)")
print(f"   Excluded: {exclude_cols}")

# Remove rows with all NaN
df_processed = df_processed.dropna(how='all')

# Use only recent data for faster execution and to avoid date overflow
# Take last 500 periods (about 40 years of monthly data)
max_periods = 500
if len(df_processed) > max_periods:
    df_processed = df_processed.iloc[-max_periods:]
    print(f"   Using last {max_periods} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# ============================================================================
# Step 3: Create Configuration
# ============================================================================
print("\n[Step 3] Creating configuration...")

# Create series configs
series_configs = []
for col in selected_cols:
    if col == target_col:
        # Target series
        series_configs.append(
            SeriesConfig(
                series_id=col,
                frequency="m",  # Assuming monthly
                transformation="lin",
                blocks=[DEFAULT_BLOCK_NAME]
            )
        )
    else:
        # Predictor series
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

# Create time index (assuming monthly data)
# For finance data, date_id is an index, so we'll create a simple time index
# Use a recent start date to avoid overflow
n_periods = len(df_processed)
# Start from 1980 to ensure we don't hit overflow (500 months = ~42 years)
start_date = datetime(1980, 1, 1)
time_list = [
    (pd.Timestamp(start_date) + pd.DateOffset(months=i)).to_pydatetime()
    for i in range(n_periods)
]

time_index = TimeIndex(time_list)

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
    print(f"   First forecast values (target): {X_forecast[0, -1]:.6f}")
    
    # Predict with history parameter (using recent 60 periods)
    X_forecast_history, Z_forecast_history = model.predict(horizon=6, history=60)
    
    print(f"   Forecast with history shape: {X_forecast_history.shape}")
    print(f"   First forecast with history (target): {X_forecast_history[0, -1]:.6f}")
    
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

