"""Tutorial: DDFM for Macro Data

This tutorial demonstrates the complete workflow for training, prediction, and nowcasting
using macro data with KOEQUIPTE as the target variable.

Target: KOEQUIPTE (Investment, Equipment, Estimation, SA)

Nowcasting Pattern: model.update(X_std).predict(horizon=1)
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python import DDFM, DFMDataModule, DDFMTrainer
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.utils.time import TimeIndex, parse_timestamp

# sktime imports for preprocessing
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("DDFM Tutorial: Macro Data")
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
# Use fewer series for faster execution
selected_cols = [
    # Employment (reduced)
    "KOEMPTOTO", "KOHWRWEMP",
    # Consumption (reduced)
    "KOWRCCNSE", "KOWRCDURE",
    # Investment (reduced)
    "KOIMPCONA",
    # Production (reduced)
    "KOCONPRCF",
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

# Use only recent data for faster execution
# Take last 100 periods (further reduced for faster execution)
max_periods = 100
if len(df_processed) > max_periods:
    df_processed = df_processed.iloc[-max_periods:]
    date_col = date_col.iloc[-max_periods:]
    print(f"   Using last {max_periods} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# Check for missing values
missing_before = df_processed.isnull().sum().sum()
print(f"   Missing values before preprocessing: {missing_before} ({missing_before/df_processed.size*100:.1f}%)")

# ============================================================================
# Step 2.5: Create Preprocessing Pipeline with sktime
# ============================================================================
print("\n[Step 2.5] Creating preprocessing pipeline with sktime...")

# Simplified preprocessing: Apply difference to target series manually, then use unified pipeline
# This is faster than ColumnEnsembleTransformer for small datasets

# Apply difference transformation to target series manually (for chg transformation)
if target_col in df_processed.columns:
    target_idx = df_processed.columns.get_loc(target_col)
    target_series = df_processed[target_col].values
    # Apply first difference
    target_diff = np.diff(target_series, prepend=target_series[0])
    df_processed[target_col] = target_diff
    print(f"   Applied difference transformation to {target_col}")

# Create simplified preprocessing pipeline: Imputation → Scaling
# (Transformations already applied manually above)
preprocessing_pipeline = TransformerPipeline(
    steps=[
        ('impute_ffill', Imputer(method="ffill")),  # Forward fill missing values
        ('impute_bfill', Imputer(method="bfill")),  # Backward fill remaining NaNs
        ('scaler', StandardScaler())  # Unified scaling for all series
    ]
)

print("   Pipeline: Imputer(ffill) → Imputer(bfill) → StandardScaler")
print(f"   Transformations: {target_col} uses difference (chg), others use linear")
print("   Applying preprocessing pipeline...")

# Apply preprocessing
df_preprocessed = preprocessing_pipeline.fit_transform(df_processed)

# Ensure output is DataFrame
if isinstance(df_preprocessed, np.ndarray):
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=df_processed.columns, index=df_processed.index)
elif not isinstance(df_preprocessed, pd.DataFrame):
    df_preprocessed = pd.DataFrame(df_preprocessed)

missing_after = df_preprocessed.isnull().sum().sum()
print(f"   Missing values after preprocessing: {missing_after}")
print(f"   Preprocessed data shape: {df_preprocessed.shape}")

# Verify standardization
mean_vals = df_preprocessed.mean()
std_vals = df_preprocessed.std()
max_mean = float(mean_vals.abs().max())
max_std_dev = float((std_vals - 1.0).abs().max())
print(f"   Standardization check - Max |mean|: {max_mean:.6f} (should be ~0)")
print(f"   Standardization check - Max |std - 1|: {max_std_dev:.6f} (should be ~0)")

# Update df_processed to use preprocessed data
df_processed = df_preprocessed

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

# Create blocks config - VAR(1) only
blocks_config = {
    DEFAULT_BLOCK_NAME: {
        "factors": 1,  # Reduced to 1 for faster execution
        "ar_lag": 1,   # VAR(1) - first-order autoregressive
        "clock": "m"
    }
}

# Create DFM config
config = DFMConfig(
    series=series_configs,
    blocks=blocks_config
)

print(f"   Number of series: {len(series_configs)}")
print(f"   Number of factors: 1 (DDFM uses num_factors parameter)")
print(f"   Factor dynamics: VAR(1) (ar_lag=1)")
print(f"   Target series: {target_col}")

# ============================================================================
# Step 4: Create DataModule
# ============================================================================
print("\n[Step 4] Creating DataModule...")

# Create time index from date column
# Align with processed data (after dropping NaN rows)
valid_dates = date_col.iloc[:len(df_processed)].values
time_index = TimeIndex([parse_timestamp(str(d)) for d in valid_dates])

# Create DataModule with transformer
data_module = DFMDataModule(
    config=config,
    data=df_processed.values,
    time_index=time_index,
    pipeline=preprocessing_pipeline  # Pass the preprocessing pipeline
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
print("\n[Step 5] Training DDFM model...")

model = DDFM(
    encoder_layers=[32, 16],  # Reduced for faster execution
    num_factors=1,  # Reduced to 1 for faster execution
    factor_order=1,  # VAR(1) - first-order factor dynamics
    epochs=10,  # Reduced for faster execution
    max_iter=3,  # Reduced for faster execution
    batch_size=32,  # Reduced for faster execution
    learning_rate=0.005
)
model._config = config  # Set config directly

trainer = DDFMTrainer(max_epochs=1)  # Minimal epochs for faster execution
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
# Step 7: Nowcasting with update().predict() pattern
# ============================================================================
print("\n[Step 7] Nowcasting using update().predict() pattern...")

try:
    # Get the trained model's result for standardization parameters
    result = model.result
    Mx = result.Mx  # Mean for standardization
    Wx = result.Wx  # Standard deviation for standardization
    
    # Simulate new data for nowcasting (in practice, this would be real-time data)
    # Use the last few periods of training data as "new" data
    n_new_periods = 5
    X_new_raw = df_processed.iloc[-n_new_periods:].values
    
    # Standardize new data using the same parameters from training
    # Standardization: (X - Mx) / Wx
    X_new_std = (X_new_raw - Mx) / Wx
    
    # Handle any NaN values (missing data in new observations)
    X_new_std = np.where(np.isfinite(X_new_std), X_new_std, np.nan)
    
    print(f"   New data shape: {X_new_std.shape}")
    print(f"   Standardized new data (first row): {X_new_std[0, :5]}")
    
    # Update model state with new standardized data, then predict
    # Pattern: model.update(X_std).predict(horizon=1)
    X_nowcast, Z_nowcast = model.update(X_new_std).predict(horizon=1)
    
    # Extract nowcast for target series
    target_idx = selected_cols.index(target_col)
    nowcast_value = X_nowcast[0, target_idx]
    
    print(f"   Nowcast value for {target_col}: {nowcast_value:.6f}")
    print(f"   Nowcast uses VAR(1) factor dynamics")
    
    # Alternative: Update and predict separately
    model.update(X_new_std)
    X_nowcast2, Z_nowcast2 = model.predict(horizon=1)
    nowcast_value2 = X_nowcast2[0, target_idx]
    print(f"   Alternative pattern (separate calls): {nowcast_value2:.6f}")
    
except (ValueError, AttributeError, IndexError) as e:
    print(f"   Nowcasting failed: {e}")
    print("   Note: Ensure model is trained and data is properly standardized")

# ============================================================================
# Step 8: Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {len(selected_cols)} series")
print(f"✅ Model trained: {len(series_configs)} series, 1 factor, VAR(1) dynamics")
if X_forecast is not None:
    print(f"✅ Predictions generated: {X_forecast.shape[0]} periods ahead")
else:
    print(f"⚠️  Predictions: Failed (see error message above)")
print(f"✅ Nowcasting pattern: model.update(X_std).predict(horizon=1)")
print(f"✅ Target series: {target_col}")
print("=" * 80)
