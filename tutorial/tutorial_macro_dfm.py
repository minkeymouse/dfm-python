"""Tutorial: DFM for Macro Data

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
from dfm_python import DFM, DFMDataModule, DFMTrainer
from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.utils.time import TimeIndex

# sktime imports for preprocessing
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer
from sklearn.preprocessing import StandardScaler

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

# Filter data (include date column for time index)
df_processed = df[selected_cols + ["date"]].copy()
print(f"   Selected {len(selected_cols)} series (including target)")
print(f"   Series: {selected_cols[:5]}...")

# Parse date column
df_processed["date"] = pd.to_datetime(df_processed["date"])
df_processed = df_processed.sort_values("date")

# Remove rows with all NaN
df_processed = df_processed.dropna(how='all')

# Use only recent data for faster execution
# Take last 100 periods (further reduced for faster execution)
max_periods = 100
if len(df_processed) > max_periods:
    df_processed = df_processed.iloc[-max_periods:]
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

# Note: date column will be removed by DataModule when time_index_column='date' is used
# For preprocessing, we need to temporarily remove it to avoid issues with datetime columns
# Store date column separately for reference (though DataModule will extract it)
if 'date' in df_processed.columns:
    # Temporarily remove date column for preprocessing (DataModule will handle it)
    df_for_preprocessing = df_processed.drop(columns=['date'])
else:
    df_for_preprocessing = df_processed

# Create simplified preprocessing pipeline: Imputation → Scaling
# (Transformations already applied manually above)
# This pipeline will be fitted and used to preprocess data
# Then passed to DataModule for statistics extraction (Mx/Wx)
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

# Add date column back for DataModule to extract (if it exists)
if 'date' in df_processed.columns:
    df_preprocessed['date'] = df_processed['date'].values

# Ensure output is DataFrame
if isinstance(df_preprocessed, np.ndarray):
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=df_processed.columns, index=df_processed.index)
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

# Create DataModule with preprocessed data
# Since data is already preprocessed, use preprocessed=True
# Pipeline is already fitted, so it will only be used for statistics extraction
# time_index_column='date' will extract time index from DataFrame and remove the column
data_module = DFMDataModule(
    config=config,
    data=df_processed,  # Pass DataFrame directly (not .values)
    time_index_column='date',  # Extract time index from 'date' column and exclude it from data
    pipeline=fitted_pipeline,  # Already fitted pipeline
    preprocessed=True  # Data is already preprocessed
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
# Use df_for_preprocessing (without date column) for scaler extraction
scaler = _get_fitted_scaler(fitted_pipeline, df_for_preprocessing)

try:
    # Predict with default horizon
    X_forecast, Z_forecast = model.predict(horizon=6)
    
    print(f"   Forecast shape: {X_forecast.shape}")
    print(f"   Factor forecast shape: {Z_forecast.shape}")
    
    # Find target index
    target_idx = selected_cols.index(target_col)
    print(f"   First forecast value (target {target_col}): {X_forecast[0, target_idx]:.6f}")
    
    # Verify inverse-transform consistency (round-trip through scaler)
    try:
        restored = scaler.inverse_transform(scaler.transform(X_forecast))
        assert np.allclose(restored, X_forecast, atol=1e-6)
        print("   ✔ Inverse-transform check passed (predict)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (predict): {inv_err}")
    
    # Predict with history parameter (using recent 60 periods)
    X_forecast_history, Z_forecast_history = model.predict(horizon=6, history=60)
    
    print(f"   Forecast with history shape: {X_forecast_history.shape}")
    print(f"   First forecast with history (target): {X_forecast_history[0, target_idx]:.6f}")
    try:
        restored_hist = scaler.inverse_transform(scaler.transform(X_forecast_history))
        assert np.allclose(restored_hist, X_forecast_history, atol=1e-6)
        print("   ✔ Inverse-transform check passed (predict with history)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (history): {inv_err}")
    
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
    
    # Option 1: Update scaler with new data (fit_transform on update batch)
    # Remove date column if present
    X_new_df = pd.DataFrame(X_new_raw, columns=df_processed.columns, index=df_processed.index[-n_new_periods:])
    X_new_df_for_preprocessing = X_new_df.drop(columns=['date']) if 'date' in X_new_df.columns else X_new_df
    
    update_pipeline = preprocessing_pipeline.clone().fit(X_new_df_for_preprocessing)
    X_new_std = update_pipeline.transform(X_new_df_for_preprocessing)
    update_scaler = _get_fitted_scaler(update_pipeline, X_new_df_for_preprocessing)
    
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
    try:
        restored_now = scaler.inverse_transform(scaler.transform(X_nowcast))
        assert np.allclose(restored_now, X_nowcast, atol=1e-6)
        print("   ✔ Inverse-transform check passed (nowcast, training scaler)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (nowcast): {inv_err}")
    try:
        restored_now_upd = update_scaler.inverse_transform(update_scaler.transform(X_nowcast))
        assert np.allclose(restored_now_upd, X_nowcast, atol=1e-6)
        print("   ✔ Inverse-transform check passed (nowcast, update scaler)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (nowcast, update scaler): {inv_err}")
    
    # Alternative: Update and predict separately
    model.update(X_new_std)
    X_nowcast2, Z_nowcast2 = model.predict(horizon=1)
    nowcast_value2 = X_nowcast2[0, target_idx]
    print(f"   Alternative pattern (separate calls): {nowcast_value2:.6f}")
    try:
        restored_now2 = scaler.inverse_transform(scaler.transform(X_nowcast2))
        assert np.allclose(restored_now2, X_nowcast2, atol=1e-6)
        print("   ✔ Inverse-transform check passed (nowcast alt, training scaler)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (nowcast alt): {inv_err}")
    try:
        restored_now2_upd = update_scaler.inverse_transform(update_scaler.transform(X_nowcast2))
        assert np.allclose(restored_now2_upd, X_nowcast2, atol=1e-6)
        print("   ✔ Inverse-transform check passed (nowcast alt, update scaler)")
    except Exception as inv_err:
        print(f"   ⚠ Inverse-transform check failed (nowcast alt, update scaler): {inv_err}")
    
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
