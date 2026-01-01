"""Tutorial: DDFM for Finance Data

This tutorial demonstrates the complete workflow for training, prediction
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
from dfm_python import DDFM, DDFMDataModule, DDFMTrainer
from dfm_python.config import DDFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_DDFM_LEARNING_RATE
from dfm_python.utils.misc import TimeIndex
from dfm_python.utils.common import select_columns_by_prefix
from dfm_python.dataset.process import parse_timestamp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer

# Preprocessing pipeline helper (uses sktime internally)

print("=" * 80)
print("DDFM Tutorial: Finance Data")
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
selected_cols = select_columns_by_prefix(df, ["D", "E", "I", "M", "P", "S", "V"], count_per_prefix=2)

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
# Take last TUTORIAL_MAX_PERIODS periods (reduced for faster execution)
if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]
    print(f"   Using last {TUTORIAL_MAX_PERIODS} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# Check for missing values
missing_before = df_processed.isnull().sum().sum()
print(f"   Missing values before preprocessing: {missing_before}")

# ============================================================================
# Step 2.5: Create Preprocessing Pipeline
# ============================================================================
print("\n[Step 2.5] Creating preprocessing pipeline...")

# Separate X (features) and y (target)
X_cols = [col for col in selected_cols if col != target_col]
y_col = target_col

X = df_processed[X_cols].copy()
y = df_processed[[y_col]].copy()  # Keep y raw (no preprocessing)

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

# Combine X (preprocessed) and y (raw) back together
df_preprocessed = pd.concat([X_preprocessed, y], axis=1)

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

# Create time index (assuming monthly data)
# For finance data, date_id is an index, so we'll create a simple time index
# Use a recent start date to avoid overflow
n_periods = len(df_processed)
# Start from 1980 to ensure we don't hit overflow
start_date = datetime(1980, 1, 1)
time_list = [
    (pd.Timestamp(start_date) + pd.DateOffset(months=i)).to_pydatetime()
    for i in range(n_periods)
]

time_index = TimeIndex(time_list)

# Create DDFMDataModule with preprocessed data
# Data must be preprocessed before passing to DataModule
# Target series are specified separately - they remain in raw form (not preprocessed)
data_module = DDFMDataModule(
    config=config,
    data=df_processed,  # Pass DataFrame directly (not .values) - already preprocessed
    time_index=time_index,
    target_series=[target_col]  # Specify target series
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
