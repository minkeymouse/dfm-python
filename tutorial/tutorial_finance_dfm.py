"""Tutorial: DFM for Finance Data

This tutorial demonstrates the complete workflow for training and prediction
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
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS
from dfm_python.dataset.time import TimeIndex
from dfm_python.utils.misc import select_columns_by_prefix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer

# Preprocessing pipeline helper (uses sktime internally)

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

# Create blocks config - new format: {"block_name": {"num_factors": int, "series": [str]}}
# VAR(1) is specified globally via ar_lag parameter
blocks_config = {
    "Block_Global": {
        "num_factors": 1,  # Reduced to 1 for faster execution
        "series": selected_cols  # All series in one block
    }
}

# Create DFM config
config = DFMConfig(
    frequency=frequency_dict,
    blocks=blocks_config,
    clock="m",  # Monthly clock frequency
    target_scaler=y_scaler,  # Fitted scaler for target series inverse transformation
    # Note: ar_lag removed - factors always use AR(1) dynamics (simplified)
    max_iter=3,  # Further reduced for faster execution
    threshold=1e-2  # More relaxed threshold for faster convergence
)

print(f"   Number of series: {len(selected_cols)}")
print(f"   Number of factors: {config.blocks['Block_Global']['num_factors']}")
print(f"   Factor dynamics: AR(1) (always used, ar_lag parameter removed)")
print(f"   Target series: {target_col}")

# ============================================================================
# Step 4: Create Dataset
# ============================================================================
print("\n[Step 4] Creating Dataset...")

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

# Create Dataset with preprocessed data
# Data is already preprocessed, so pass it directly
dataset = DFMDataset(
    config=config,
    data=df_processed,  # Pass DataFrame directly (already preprocessed)
    time_index=time_index,
    target_series=[target_col]  # Specify target series
)
# Dataset initialization happens in __init__

print(f"   Dataset created successfully")
if hasattr(dataset, 'data_processed') and dataset.data_processed is not None:
    print(f"   Processed data shape: {dataset.data_processed.shape}")
else:
    print(f"   Data shape: {df_processed.shape}")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n[Step 5] Training DFM model...")

# Create DFM model with config
# Note: mixed_freq is auto-detected from config frequencies
# Mixed frequency will be automatically detected based on config frequencies
model = DFM(config)

# Get initialization parameters from datamodule
init_params = dataset.get_initialization_params()
X = init_params['X']
# Note: Mx/Wx removed - target_scaler is used instead for inverse transformation
# Target scaler is available via dataset.target_scaler

# Fit model directly (DFM uses fit() method)
# Pass dataset to extract all initialization parameters automatically
model.fit(X=X, dataset=dataset)

# Access results via property
result = model.result

print("   Training completed!")
print(f"   Converged: {result.converged}")
print(f"   Iterations: {result.num_iter}")
print(f"   Log-likelihood: {result.loglik:.4f}")

# ============================================================================
# Step 6: Prediction
# ============================================================================
print("\n[Step 6] Making predictions...")

# Predict with horizon=6 (uses target_series from Dataset)
X_forecast, Z_forecast = model.predict(horizon=6)

print(f"   Forecast shape: {X_forecast.shape}")
print(f"   Factor forecast shape: {Z_forecast.shape}")
print(f"   First forecast value (target): {X_forecast[0, 0]:.6f}")

# ============================================================================
# Step 7: Summary
# ============================================================================
print("\n" + result.summary())
