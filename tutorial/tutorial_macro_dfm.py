"""Tutorial: DFM for Macro Data

This tutorial demonstrates the complete workflow for training and prediction
using macro data with multiple target variables.

Targets: KOEQUIPTE (Investment), KOWRCCNSE (Consumption), KOIMPCONA (Imports)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Preprocessing pipeline helper (uses sktime internally)

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

# Target variables (multiple targets)
target_cols = ["KOEQUIPTE", "KOWRCCNSE", "KOIMPCONA"]  # Investment, Consumption, Imports

# Select a subset of series for faster execution
# Use fewer series for faster execution
selected_cols = [
    # Employment (reduced)
    "KOEMPTOTO", "KOHWRWEMP",
    # Consumption (reduced)
    "KOWRCDURE",  # Note: KOWRCCNSE is in target_cols, so don't duplicate
    # Investment (reduced) - KOIMPCONA is in target_cols, so don't duplicate
    # Production (reduced)
    "KOCONPRCF",
    # Targets (included in selected_cols)
] + target_cols

# Filter to only columns that exist in the data and remove duplicates
selected_cols = [col for col in selected_cols if col in df.columns]
# Remove duplicates while preserving order
seen = set()
selected_cols = [col for col in selected_cols if not (col in seen or seen.add(col))]

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
# Take last TUTORIAL_MAX_PERIODS periods (reduced for faster execution)
if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]
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

# Separate X (features) and y (target)
# Note: date column will be removed by Dataset when time_index_column='date' is used
if 'date' in df_processed.columns:
    df_for_preprocessing = df_processed.drop(columns=['date'])
else:
    df_for_preprocessing = df_processed

# Separate features (X) and target (y)
X_cols = [col for col in selected_cols if col not in target_cols]
y_cols = target_cols

X = df_for_preprocessing[X_cols].copy()
y = df_for_preprocessing[y_cols].copy()  # Keep y raw (no preprocessing)

print("   Separating features (X) and target (y)...")
print(f"   X (features): {len(X_cols)} series - will be preprocessed")
print(f"   y (target): {len(y_cols)} series - kept raw (no preprocessing pipeline)")

# Create preprocessing pipeline for X (features): Imputation → Scaling
X_pipeline = Pipeline(
    steps=[
        ('impute_ffill', Imputer(method="ffill")),
        ('impute_bfill', Imputer(method="bfill")),
        ('scaler', StandardScaler())
    ]
)

print("   Pipeline for X: Imputer(ffill) → Imputer(bfill) → StandardScaler")
print("   y (target): Raw series (no preprocessing pipeline, no differencing)")
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

# Add date column back for Dataset to extract (if it exists)
if 'date' in df_processed.columns:
    df_preprocessed['date'] = df_processed['date'].values

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
# Step 3: Load Configuration from YAML using Hydra
# ============================================================================
print("\n[Step 3] Loading configuration from YAML...")

# y_scaler is already created and fitted above (fitted on raw y, not transformed)
target_scaler = y_scaler

# Load config from YAML using Hydra
config_dir = project_root / "config"
config_path = config_dir / "dfm_macro.yaml"

# Initialize Hydra and load config
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="dfm_macro")
    
    # Convert to dict for easier manipulation
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Filter frequency dict to only include selected columns
    if 'frequency' in config_dict and config_dict['frequency']:
        config_dict['frequency'] = {k: v for k, v in config_dict['frequency'].items() if k in selected_cols}
    else:
        # Create frequency dict from selected columns (default to clock frequency)
        clock = config_dict.get('clock', 'w')
        config_dict['frequency'] = {col: clock for col in selected_cols}
    
    # Filter blocks to only include selected series
    if 'blocks' in config_dict and config_dict['blocks']:
        filtered_blocks = {}
        for block_name, block_data in config_dict['blocks'].items():
            block_series = [s for s in block_data.get('series', []) if s in selected_cols]
            if block_series:  # Only include blocks with at least one selected series
                filtered_blocks[block_name] = {
                    'num_factors': block_data.get('num_factors', 1),
                    'series': block_series
                }
        config_dict['blocks'] = filtered_blocks
    else:
        # Fallback: create a single block with all selected series
        config_dict['blocks'] = {
            "Block_Global": {
                "num_factors": 1,
                "series": selected_cols
            }
        }
    
    # Override max_iter and threshold for faster tutorial execution
    config_dict['max_iter'] = 3
    config_dict['threshold'] = 1e-2
    
    # Create DFMConfig from filtered dict
    config = DFMConfig.from_dict(config_dict)
    
    # Update target_scaler (must be set after creating config)
    config.target_scaler = target_scaler

print(f"   Config loaded from: {config_path}")
print(f"   Number of series: {len(selected_cols)}")
if config.blocks:
    first_block = list(config.blocks.values())[0]
    print(f"   Number of factors: {first_block['num_factors']}")
print(f"   Clock frequency: {config.clock}")
print(f"   Max iterations: {config.max_iter} (reduced for tutorial)")
print(f"   Target series: {', '.join(target_cols)} ({len(target_cols)} targets)")

# ============================================================================
# Step 4: Create Dataset
# ============================================================================
print("\n[Step 4] Creating Dataset...")

# Create Dataset with preprocessed data
# Data is already preprocessed, so pass it directly
# time_index='date' will extract time index from DataFrame and remove the column
dataset = DFMDataset(
    config=config,
    data=df_processed,  # Pass DataFrame directly (already preprocessed)
    time_index='date',  # Extract time index from 'date' column and exclude it from data
    target_series=target_cols  # Specify multiple target series
)
# Dataset initialization happens in __init__

print(f"   Dataset created successfully")
if hasattr(dataset, 'data_processed') and dataset.data_processed is not None:
    print(f"   Processed data shape: {dataset.data_processed.shape}")
else:
    print(f"   Data shape: {df_processed.shape}")
if dataset.time_index is not None:
    print(f"   Time range: {dataset.time_index[0]} to {dataset.time_index[-1]}")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n[Step 5] Training DFM model...")

# Create DFM model with config
# Note: mixed_freq is now auto-detected from Dataset
# Mixed frequency will be automatically detected based on config frequencies
model = DFM(config)

# Get initialization parameters from datamodule
init_params = dataset.get_initialization_params()
X = init_params['X']


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

print(f"   Forecast shape: {X_forecast.shape} (horizon x {len(target_cols)} targets)")
print(f"   Factor forecast shape: {Z_forecast.shape}")
print(f"   First period forecasts:")
for i, target_col in enumerate(target_cols):
    print(f"     {target_col}: {X_forecast[0, i]:.6f}")

# ============================================================================
# Step 7: Summary
# ============================================================================
print("\n" + result.summary())
