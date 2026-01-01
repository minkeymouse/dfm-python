"""Tutorial: DDFM for Macro Data

This tutorial demonstrates the complete workflow for training, prediction
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
from dfm_python import DDFM, DDFMDataModule, DDFMTrainer
from dfm_python.config import DDFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_DDFM_LEARNING_RATE
from dfm_python.utils.misc import TimeIndex
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer

# Preprocessing pipeline helper (uses sktime internally)

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
# Take last TUTORIAL_MAX_PERIODS periods (reduced for faster execution)
if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]
    date_column = date_column.iloc[-TUTORIAL_MAX_PERIODS:]
    print(f"   Using last {TUTORIAL_MAX_PERIODS} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# Check for missing values
missing_before = df_processed.isnull().sum().sum()
print(f"   Missing values before preprocessing: {missing_before} ({missing_before/df_processed.size*100:.1f}%)")

# ============================================================================
# Step 2.5: Create Preprocessing Pipeline with sktime
# ============================================================================
print("\n[Step 2.5] Creating preprocessing pipeline with sktime...")

# Note: Target series will be kept raw (no differencing, no preprocessing pipeline)

# Date column is already separated, so df_processed has no date column
# Use df_processed directly for preprocessing
df_for_preprocessing = df_processed

# Separate X (features) and y (target)
X_cols = [col for col in selected_cols if col != target_col]
y_col = target_col

X = df_for_preprocessing[X_cols].copy()
y = df_for_preprocessing[[y_col]].copy()  # Keep y raw (no preprocessing)

print("   Separating features (X) and target (y)...")
print(f"   X (features): {len(X_cols)} series - will be preprocessed")
print(f"   y (target): 1 series ({y_col}) - kept raw (no preprocessing pipeline, no differencing)")

# Create preprocessing pipeline for X (features): Imputation → Scaling
X_pipeline = Pipeline(
    steps=[
        ('impute_ffill', Imputer(method="ffill")),
        ('impute_bfill', Imputer(method="bfill")),
        ('scaler', StandardScaler())
    ]
)

print("   Pipeline for X: Imputer(ffill) → Imputer(bfill) → StandardScaler")
print("   y (target): Raw series (no preprocessing pipeline)")
print("   Applying preprocessing pipeline to X only...")

# Fit and transform X (features)
X_pipeline.fit(X)
X_preprocessed = X_pipeline.transform(X)

# Create scaler for y (target) - fit but don't transform
# This scaler will be used for inverse transformation during prediction
y_scaler = StandardScaler()
y_scaler.fit(y)

# Ensure X output is DataFrame
if isinstance(X_preprocessed, np.ndarray):
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X_cols, index=X.index)
else:
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X_cols, index=X.index)

# Combine X (preprocessed) and y (raw) back together
df_preprocessed = pd.concat([X_preprocessed, y], axis=1)

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

# Create frequency dict (maps column names to frequencies)
# All series are monthly, so use 'm' for all
frequency_dict = {col: "m" for col in selected_cols}

# Create DDFM config (DDFM does not use blocks structure)
# DDFM uses num_factors directly, not blocks
config = DDFMConfig(
    frequency=frequency_dict,
    clock="m",  # Monthly clock frequency
    num_factors=1,  # Reduced to 1 for faster execution
    factor_order=1,  # VAR(1) - first-order autoregressive
    encoder_layers=[32, 16],  # Reduced for faster execution
    epochs=10,  # Reduced for faster execution
    learning_rate=DEFAULT_LEARNING_RATE,
    batch_size=DEFAULT_BATCH_SIZE,
    target_scaler=y_scaler  # Fitted scaler for target series inverse transformation
)

print(f"   Number of series: {len(selected_cols)}")
print(f"   Number of factors: {config.num_factors} (DDFM uses num_factors parameter)")
print(f"   Factor dynamics: VAR(1) (factor_order=1)")
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
    batch_size=DEFAULT_BATCH_SIZE,  # Reduced for faster execution
    learning_rate=DEFAULT_DDFM_LEARNING_RATE
)
model._config = config  # Set config directly

trainer = DDFMTrainer(max_epochs=1)  # Minimal epochs for faster execution
trainer.fit(model, data_module)

print("   Training completed!")

# ============================================================================
# Step 6: Prediction
# ============================================================================
print("\n[Step 6] Making predictions...")

# Predict with horizon=6 (uses target_series from DataModule)
X_forecast, Z_forecast = model.predict(horizon=6)

print(f"   Forecast shape: {X_forecast.shape}")
print(f"   Factor forecast shape: {Z_forecast.shape}")
print(f"   First forecast value (target): {X_forecast[0, 0]:.6f}")

# ============================================================================
# Step 7: Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {len(selected_cols)} series")
print(f"✅ Model trained: {len(selected_cols)} series, 1 factor, VAR(1) dynamics")
if X_forecast is not None:
    print(f"✅ Predictions generated: {X_forecast.shape[0]} periods ahead")
else:
    print(f"⚠️  Predictions: Failed (see error message above)")
print(f"✅ Target series: {target_col}")
print("=" * 80)
