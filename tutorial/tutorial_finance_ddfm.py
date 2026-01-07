"""Tutorial: DDFM for Finance Data

This tutorial demonstrates the complete workflow for training, prediction
using finance data with market_forward_excess_returns as the target variable.

Target: market_forward_excess_returns
Excluded: risk_free_rate, forward_returns

Note: DDFM uses noise injection integrated into the Autoencoder class.
Noise is pre-sampled on GPU and injected by subtracting epsilon from clean data,
following the original DDFM pattern: y_t^(mc) = ỹ_t - ε_t^(mc).

"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python import DDFM, DDFMDataset
from dfm_python.config import DDFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS, DEFAULT_LEARNING_RATE, DEFAULT_DDFM_WINDOW_SIZE, DEFAULT_DDFM_LEARNING_RATE
from dfm_python.utils.misc import select_columns_by_prefix
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

# Standardize ALL data (matching original TensorFlow DDFM)
# Original TensorFlow: self.data = (data - self.mean_z) / self.sigma_z
print("\n[Step 2.6] Standardizing ALL data (matching original TensorFlow DDFM)...")
print("   Original TensorFlow: self.data = (data - self.mean_z) / self.sigma_z")
print("   All series (including target) must be standardized before passing to Dataset")

# Standardize all data (matching original TensorFlow DDFM)
mean_z = df_preprocessed.mean().values
sigma_z = df_preprocessed.std().values
df_standardized = (df_preprocessed - mean_z) / sigma_z

# Verify standardization
mean_vals = df_standardized.mean()
std_vals = df_standardized.std()
max_mean = float(mean_vals.abs().max())
max_std_dev = float((std_vals - 1.0).abs().max())
print(f"   Standardization check - Max |mean|: {max_mean:.6f} (should be ~0)")
print(f"   Standardization check - Max |std - 1|: {max_std_dev:.6f} (should be ~0)")

# Update df_preprocessed to use standardized data
df_preprocessed = df_standardized

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
# Note: factor_order is not a parameter - factors always use AR(1) dynamics
config = DDFMConfig(
    frequency=frequency_dict,
    clock="m",  # Monthly clock frequency
    num_factors=1,  # Reduced to 1 for faster execution
    encoder_layers=[32, 16],  # Reduced for faster execution
    n_mc_samples=10,  # Number of MC samples per MCMC iteration (reduced for faster execution)
    learning_rate=DEFAULT_LEARNING_RATE,
    window_size=DEFAULT_DDFM_WINDOW_SIZE,  # Window size (time-step batch size) for training
    target_scaler=y_scaler  # Fitted scaler for target series inverse transformation
)

print(f"   Number of series: {len(selected_cols)}")
print(f"   Number of factors: {config.num_factors} (DDFM uses num_factors parameter)")
print(f"   Factor dynamics: VAR(1) (always AR(1), not configurable)")
print(f"   MC samples per iteration: {config.n_mc_samples}")
print(f"   Target series: {target_col}")
print(f"   Noise injection: Integrated in Autoencoder (pre-sampled on GPU)")

# ============================================================================
# Step 4: Create Dataset
# ============================================================================
print("\n[Step 4] Creating Dataset...")

# Note: DDFMDataset uses time_idx='index' to use DataFrame index as time identifier
# No need to create separate TimeIndex object

# Create DDFMDataset with preprocessed data
# Data must be preprocessed before passing to Dataset
# Target series are specified separately - they remain in raw form (not preprocessed)
dataset = DDFMDataset(
    data=df_processed,  # Pass DataFrame directly (not .values) - already preprocessed
    time_idx='index',  # Use DataFrame index as time identifier
    target_series=[target_col],  # Specify target series
    target_scaler=y_scaler  # Fitted scaler for target series
)

print(f"   Dataset created successfully")
print(f"   Data shape: {dataset.data.shape}")
print(f"   Target series: {dataset.target_series}")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n[Step 5] Training DDFM model...")

# Create model with parameters from config
# Map config parameters to model parameters
encoder_layers = getattr(config, 'encoder_layers', [32, 1])
encoder_size = tuple(encoder_layers) if encoder_layers else (32, 1)
max_epoch = getattr(config, 'max_epoch', 3)  # Reduced for faster execution

model = DDFM(
    dataset=dataset,
    config=config,
    encoder_size=encoder_size,  # Convert list to tuple
    decoder_type="linear",
    activation=getattr(config, 'activation', 'relu'),
    learning_rate=getattr(config, 'learning_rate', DEFAULT_DDFM_LEARNING_RATE),
    optimizer='Adam',
    n_mc_samples=getattr(config, 'n_mc_samples', 10),
    window_size=getattr(config, 'window_size', DEFAULT_DDFM_WINDOW_SIZE),
    max_iter=max_epoch,  # Map max_epoch -> max_iter
    tolerance=getattr(config, 'tolerance', 0.0005),
    disp=getattr(config, 'disp', 10),
    seed=getattr(config, 'seed', None)
)

# Train model - fit() builds model, pre-trains, and trains in one method
print(f"   Starting training (max {model.max_iter} iterations)...")
model.fit()
print("   Training completed!")

# Build state-space model for prediction
print("\n[Step 5.5] Building state-space model...")
model.build_state_space()
print("   State-space model built successfully")

# ============================================================================
# Step 6: Prediction
# ============================================================================
print("\n[Step 6] Making predictions...")

# Predict with horizon=6 (uses target_series from Dataset)
# Note: predict() requires build_state_space() to be called first
X_forecast, Z_forecast = model.predict(horizon=6, return_series=True, return_factors=True)

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
