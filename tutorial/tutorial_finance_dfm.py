"""Tutorial: DFM for Finance Data

This tutorial demonstrates the complete workflow for training, prediction, and nowcasting
using finance data with market_forward_excess_returns as the target variable.

Target: market_forward_excess_returns
Excluded: risk_free_rate, forward_returns

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
from dfm_python import DFM, DFMDataModule, DFMTrainer
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.utils.time import TimeIndex, parse_timestamp

# sktime imports for preprocessing
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer
from sklearn.preprocessing import StandardScaler

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
# Use first 2 series from each category: D, E, I, M, P, S, V (balanced for speed)
selected_cols = []
for prefix in ["D", "E", "I", "M", "P", "S", "V"]:
    for i in range(1, 3):  # Use first 2 from each category
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
# Take last 100 periods (further reduced for faster execution)
max_periods = 100
if len(df_processed) > max_periods:
    df_processed = df_processed.iloc[-max_periods:]
    print(f"   Using last {max_periods} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# Check for missing values
missing_before = df_processed.isnull().sum().sum()
print(f"   Missing values before preprocessing: {missing_before}")

# ============================================================================
# Step 2.5: Create Preprocessing Pipeline with sktime
# ============================================================================
print("\n[Step 2.5] Creating preprocessing pipeline with sktime...")

# Create preprocessing pipeline: Imputation → Scaling
# This pipeline will be fitted and used to preprocess data
# Then passed to DataModule for statistics extraction (Mx/Wx)
preprocessing_pipeline = TransformerPipeline(
    steps=[
        ('impute_ffill', Imputer(method="ffill")),  # Forward fill missing values
        ('impute_bfill', Imputer(method="bfill")),  # Backward fill remaining NaNs
        ('scaler', StandardScaler())  # Unified scaling for all series
    ]
)

print("   Pipeline: Imputer(ffill) → Imputer(bfill) → StandardScaler")
print("   Applying preprocessing pipeline...")

# Apply preprocessing (fit and transform)
# Data will be preprocessed before passing to DataModule
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
    blocks=blocks_config,
    max_iter=3,  # Further reduced for faster execution
    threshold=1e-2  # More relaxed threshold for faster convergence
)

print(f"   Number of series: {len(series_configs)}")
print(f"   Number of factors: {config.blocks[DEFAULT_BLOCK_NAME]['factors']}")
print(f"   Factor dynamics: VAR(1) (ar_lag=1)")
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

# Create DataModule with preprocessed data
# Since data is already preprocessed, use preprocessed=True
# Pipeline is already fitted, so it will only be used for statistics extraction
data_module = DFMDataModule(
    config=config,
    data=df_processed,  # Pass DataFrame directly (not .values)
    time_index=time_index,
    pipeline=preprocessing_pipeline,  # Already fitted pipeline
    preprocessed=True  # Data is already preprocessed
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

# Create DFM model
# Note: mixed_freq=False (default) since all series are monthly (unified frequency)
# Set mixed_freq=True if you have mixed frequencies (e.g., quarterly + monthly)
model = DFM(mixed_freq=False)
model._config = config  # Set config directly

trainer = DFMTrainer(max_epochs=1)  # Minimal epochs for faster execution
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
print(f"✅ Model trained: {len(series_configs)} series, {config.blocks[DEFAULT_BLOCK_NAME]['factors']} factors, VAR(1) dynamics")
if X_forecast is not None:
    print(f"✅ Predictions generated: {X_forecast.shape[0]} periods ahead")
else:
    print(f"⚠️  Predictions: Failed (see error message above)")
print(f"✅ Nowcasting pattern: model.update(X_std).predict(horizon=1)")
print(f"✅ Target series: {target_col}")
print("=" * 80)
