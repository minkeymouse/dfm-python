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
from dfm_python import DDFM, DDFMDataModule, DDFMTrainer
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.utils.time import TimeIndex

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

# Filter data (include date column for time index, but separate it early)
df_with_date = df[selected_cols + ["date"]].copy()
print(f"   Selected {len(selected_cols)} series (including target)")
print(f"   Series: {selected_cols[:5]}...")

# Parse and sort by date column
df_with_date["date"] = pd.to_datetime(df_with_date["date"])
df_with_date = df_with_date.sort_values("date")

# Extract date column separately before preprocessing
date_column = df_with_date["date"].copy()

# Remove date column from data for preprocessing
df_processed = df_with_date.drop(columns=["date"]).copy()

# Remove rows with all NaN
df_processed = df_processed.dropna(how='all')

# Use only recent data for faster execution
# Take last 100 periods (further reduced for faster execution)
max_periods = 100
if len(df_processed) > max_periods:
    df_processed = df_processed.iloc[-max_periods:]
    date_column = date_column.iloc[-max_periods:]
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

# Date column is already separated, so df_processed has no date column
# Use df_processed directly for preprocessing
df_for_preprocessing = df_processed

# Create simplified preprocessing pipeline: Imputation → Scaling
# (Transformations already applied manually above)
# This pipeline will be fitted and used to preprocess feature data
# Target series will be handled separately (not preprocessed by this pipeline)
preprocessing_pipeline = TransformerPipeline(
    steps=[
        ('impute_ffill', Imputer(method="ffill")),  # Forward fill missing values
        ('impute_bfill', Imputer(method="bfill")),  # Backward fill remaining NaNs
        ('scaler', StandardScaler())  # Unified scaling for all series
    ]
)

def _get_fitted_scaler(pipeline, data_frame):
    """Extract fitted scaler; if not fitted, fit it on provided data."""
    steps_attr = getattr(pipeline, "steps_", None)
    candidate = (steps_attr or pipeline.steps)[-1][1]
    if not hasattr(candidate, "n_features_in_"):
        candidate = candidate.fit(data_frame)
    return candidate

print("   Pipeline: Imputer(ffill) → Imputer(bfill) → StandardScaler")
print(f"   Transformations: {target_col} uses difference (chg), others use linear")
print("   Applying preprocessing pipeline...")

# Fit preprocessing pipeline on training data once (without date column)
fitted_pipeline = preprocessing_pipeline.clone().fit(df_for_preprocessing)
df_preprocessed = fitted_pipeline.transform(df_for_preprocessing)

# Ensure output is DataFrame
if isinstance(df_preprocessed, np.ndarray):
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=df_for_preprocessing.columns, index=df_for_preprocessing.index)
elif not isinstance(df_preprocessed, pd.DataFrame):
    df_preprocessed = pd.DataFrame(df_preprocessed)

# Add date column back for DataModule to extract
# Date column was separated earlier, so add it back now
df_preprocessed['date'] = date_column.values

# Ensure output is DataFrame
if isinstance(df_preprocessed, np.ndarray):
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=list(df_for_preprocessing.columns) + ['date'], index=df_for_preprocessing.index)
elif not isinstance(df_preprocessed, pd.DataFrame):
    df_preprocessed = pd.DataFrame(df_preprocessed)

missing_after = df_preprocessed.isnull().sum().sum()
print(f"   Missing values after preprocessing: {missing_after}")
print(f"   Preprocessed data shape: {df_preprocessed.shape}")

# Verify standardization (exclude date column if present)
df_for_check = df_preprocessed.drop(columns=['date']) if 'date' in df_preprocessed.columns else df_preprocessed
mean_vals = df_for_check.mean()
std_vals = df_for_check.std()
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
    # SeriesConfig only needs frequency (transformation and blocks removed)
    # Transformation is handled by preprocessing pipeline, not SeriesConfig
    series_configs.append(
        SeriesConfig(
            series_id=col,
            frequency="m"
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

# Create DDFMDataModule with preprocessed data
# Data must be preprocessed before passing to DataModule
# Target series are specified separately - they remain in raw form (not preprocessed)
# time_index_column='date' will extract time index from DataFrame and remove the column
data_module = DDFMDataModule(
    config=config,
    data=df_processed,  # Pass DataFrame directly (not .values) - already preprocessed
    time_index_column='date',  # Extract time index from 'date' column and exclude it from data
    target_series=[target_col]  # Specify target series
)
data_module.setup()

print(f"   DataModule created successfully")
if hasattr(data_module, 'data_processed') and data_module.data_processed is not None:
    print(f"   Processed data shape: {data_module.data_processed.shape}")
else:
    print(f"   Data shape: {df_processed.shape}")
if data_module.time_index is not None:
    print(f"   Time range: {data_module.time_index[0]} to {data_module.time_index[-1]}")

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
# Exclude date column for scaler extraction
df_for_scaler = df_processed.drop(columns=['date']) if 'date' in df_processed.columns else df_processed
scaler = _get_fitted_scaler(fitted_pipeline, df_for_scaler)

try:
    # Predict with default horizon (target=None uses DataModule's target_series)
    X_forecast, Z_forecast = model.predict(horizon=6)
    
    print(f"   Forecast shape: {X_forecast.shape} (target series only)")
    print(f"   Factor forecast shape: {Z_forecast.shape}")
    
    # X_forecast now contains only target series (no features)
    if X_forecast.shape[1] == 1:
        print(f"   First forecast value (target {target_col}): {X_forecast[0, 0]:.6f}")
    else:
        print(f"   First forecast values (targets): {X_forecast[0, :]}")
    
    # Note: Inverse-transform check removed - predict() now returns only target series,
    # so full-series scaler cannot be used for validation
    
    # Predict with history parameter (using recent 60 periods)
    X_forecast_history, Z_forecast_history = model.predict(horizon=6, history=60)
    
    print(f"   Forecast with history shape: {X_forecast_history.shape} (target series only)")
    if X_forecast_history.shape[1] == 1:
        print(f"   First forecast with history (target): {X_forecast_history[0, 0]:.6f}")
    else:
        print(f"   First forecast with history (targets): {X_forecast_history[0, :]}")
    # Note: Inverse-transform check removed - predict() now returns only target series
    
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
    # Exclude date column for nowcasting
    n_new_periods = 5
    df_new_raw = df_processed.drop(columns=['date']).iloc[-n_new_periods:] if 'date' in df_processed.columns else df_processed.iloc[-n_new_periods:]
    X_new_raw = df_new_raw.values
    
    # Option 1: Update scaler with new data (fit_transform on update batch)
    # Use columns without date for scaler
    data_cols = [col for col in df_processed.columns if col != 'date']
    update_pipeline = preprocessing_pipeline.clone().fit(
        pd.DataFrame(X_new_raw, columns=data_cols, index=df_processed.index[-n_new_periods:])
    )
    X_new_std = update_pipeline.transform(
        pd.DataFrame(X_new_raw, columns=data_cols, index=df_processed.index[-n_new_periods:])
    )
    update_scaler = _get_fitted_scaler(update_pipeline, pd.DataFrame(X_new_raw, columns=data_cols))
    
    # Handle any NaN values (missing data in new observations)
    X_new_std = np.where(np.isfinite(X_new_std), X_new_std, np.nan)
    
    print(f"   New data shape: {X_new_std.shape}")
    print(f"   Standardized new data (first row): {X_new_std[0, :5]}")
    
    # Update model state with new standardized data, then predict
    # Pattern: model.update(X_std).predict(horizon=1)
    # predict() returns only target series (target=None uses DataModule's target_series)
    X_nowcast, Z_nowcast = model.update(X_new_std).predict(horizon=1)
    
    # X_nowcast now contains only target series (no features)
    if X_nowcast.shape[1] == 1:
        nowcast_value = X_nowcast[0, 0]
    else:
        nowcast_value = X_nowcast[0, :]
    
    print(f"   Nowcast value for {target_col}: {nowcast_value:.6f}")
    print(f"   Nowcast uses VAR(1) factor dynamics")
    # Note: Inverse-transform check removed - predict() now returns only target series
    
    # Alternative: Update and predict separately
    model.update(X_new_std)
    X_nowcast2, Z_nowcast2 = model.predict(horizon=1)
    if X_nowcast2.shape[1] == 1:
        nowcast_value2 = X_nowcast2[0, 0]
    else:
        nowcast_value2 = X_nowcast2[0, :]
    print(f"   Alternative pattern (separate calls): {nowcast_value2:.6f}")
    # Note: Inverse-transform check removed - predict() now returns only target series
    
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
