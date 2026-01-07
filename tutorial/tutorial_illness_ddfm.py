"""Tutorial: DDFM for Illness Data with Covariates

This tutorial demonstrates the complete workflow for training and prediction
using illness data with covariates (features that are not targets).

Target: ILI_TOTAL_CASES
Features (covariates): WEIGHTED_ILI_PCT, UNWEIGHTED_ILI_PCT, ILI_CASES_AGE_0_4,
                       ILI_CASES_AGE_5_24, NUM_REPORTING_PROVIDERS, TOTAL_OUTPATIENT_VISITS

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
from dfm_python import DDFM, DDFMDataset
from dfm_python.config import DDFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS, DEFAULT_DDFM_WINDOW_SIZE, DEFAULT_DDFM_LEARNING_RATE
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("DDFM Tutorial: Illness Data with Covariates")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[Step 1] Loading illness data...")

# Try multiple possible paths
data_paths = [
    project_root / "data" / "illness" / "national_illness.csv",
    project_root / "data" / "illness" / "illness.csv",
    project_root.parent / "data" / "illness" / "national_illness.csv",
    project_root.parent / "data" / "illness" / "illness.csv",
]

data_path = None
for path in data_paths:
    if path.exists():
        data_path = path
        break

if data_path is None:
    print("   ⚠️  Illness data file not found. Please ensure data file exists.")
    print("   Expected paths:")
    for path in data_paths:
        print(f"      - {path}")
    sys.exit(1)

print(f"   Loading from: {data_path}")
df = pd.read_csv(data_path)

# Parse date column if it exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')

print(f"   Data shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# Step 2: Prepare Data
# ============================================================================
print("\n[Step 2] Preparing data...")

# Define target and feature series based on config
target_col = "ILI_TOTAL_CASES"
feature_cols = [
    "WEIGHTED_ILI_PCT",
    "UNWEIGHTED_ILI_PCT",
    "ILI_CASES_AGE_0_4",
    "ILI_CASES_AGE_5_24",
    "NUM_REPORTING_PROVIDERS",
    "TOTAL_OUTPATIENT_VISITS"
]

# Filter to only columns that exist in the data
available_cols = list(df.columns)
target_col = target_col if target_col in available_cols else available_cols[0]  # Fallback to first column
feature_cols = [col for col in feature_cols if col in available_cols]

# If no feature columns found, use all except target
if len(feature_cols) == 0:
    feature_cols = [col for col in available_cols if col != target_col]

print(f"   Target series: {target_col}")
print(f"   Feature series ({len(feature_cols)}): {feature_cols[:3]}..." if len(feature_cols) > 3 else f"   Feature series ({len(feature_cols)}): {feature_cols}")

# Select relevant columns
selected_cols = feature_cols + [target_col]
df_processed = df[selected_cols].copy()

# Remove rows with all NaN
df_processed = df_processed.dropna(how='all')

# Use only recent data for faster execution
if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]
    print(f"   Using last {TUTORIAL_MAX_PERIODS} periods for faster execution")

print(f"   Data shape after cleaning: {df_processed.shape}")

# Check for missing values
missing_before = df_processed.isnull().sum().sum()
print(f"   Missing values before preprocessing: {missing_before}")

# Handle missing values with forward fill and backward fill
if missing_before > 0:
    print("   Handling missing values with forward fill and backward fill...")
    df_processed = df_processed.ffill().bfill()

missing_after = df_processed.isnull().sum().sum()
print(f"   Missing values after imputation: {missing_after}")

# ============================================================================
# Step 2.5: Preprocess Features and Target
# ============================================================================
print("\n[Step 2.5] Preprocessing features and target...")

# Separate X (features) and y (target)
X = df_processed[feature_cols].copy()
y = df_processed[[target_col]].copy()

print("   Separating features (X) and target (y)...")
print(f"   X (features): {len(feature_cols)} series - will be standardized")
print(f"   y (target): 1 series ({target_col}) - will be standardized separately")

# Standardize features (X)
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X.values)
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Standardize target (y) - fit scaler for inverse transformation during prediction
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values)
y_scaled = pd.DataFrame(y_scaled, index=y.index, columns=y.columns)

# Combine X (scaled) and y (scaled) back together
df_preprocessed = pd.concat([X_scaled, y_scaled], axis=1)

print(f"   Preprocessed data shape: {df_preprocessed.shape}")
print(f"   Features standardized: mean={X_scaled.mean().abs().max():.6f}, std={X_scaled.std().abs().max():.6f}")
print(f"   Target standardized: mean={y_scaled.mean().abs().max():.6f}, std={y_scaled.std().abs().max():.6f}")

# ============================================================================
# Step 3: Create Configuration
# ============================================================================
print("\n[Step 3] Creating configuration...")

# Create frequency dict (all series are weekly for illness data)
frequency_dict = {col: "w" for col in selected_cols}

# Create DDFM config
config = DDFMConfig(
    frequency=frequency_dict,
    clock="w",  # Weekly clock frequency
    num_factors=2,  # Reduced for faster execution
    encoder_layers=[16, 2],  # Reduced for faster execution
    n_mc_samples=10,
    learning_rate=DEFAULT_DDFM_LEARNING_RATE,
    window_size=DEFAULT_DDFM_WINDOW_SIZE,
    target_scaler=y_scaler  # Fitted scaler for target series inverse transformation
)

print(f"   Number of series: {len(selected_cols)}")
print(f"   Number of factors: {config.num_factors}")
print(f"   Factor dynamics: VAR(1) (always AR(1), not configurable)")
print(f"   MC samples per iteration: {config.n_mc_samples}")
print(f"   Target series: {target_col}")
print(f"   Feature series: {len(feature_cols)} covariates")

# ============================================================================
# Step 4: Create Dataset
# ============================================================================
print("\n[Step 4] Creating Dataset...")

# Create DDFMDataset with preprocessed data
# Data must be preprocessed before passing to Dataset
# Target series are specified separately
dataset = DDFMDataset(
    data=df_preprocessed,  # Pass DataFrame directly - already preprocessed
    time_idx='index',  # Use DataFrame index as time identifier
    target_series=[target_col],  # Specify target series only
    target_scaler=y_scaler  # Fitted scaler for target series
)

print(f"   Dataset created successfully")
print(f"   Data shape: {dataset.data.shape}")
print(f"   Target series: {dataset.target_series}")
print(f"   Feature series: {dataset.feature_columns}")
print(f"   All columns are targets: {dataset.all_columns_are_targets} (should be False)")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n[Step 5] Training DDFM model...")

# Create model with parameters from config
encoder_layers = getattr(config, 'encoder_layers', [16, 2])
encoder_size = tuple(encoder_layers) if encoder_layers else (16, 2)
max_epoch = getattr(config, 'max_epoch', 3)  # Reduced for faster execution

model = DDFM(
    dataset=dataset,
    config=config,
    encoder_size=encoder_size,
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
# Step 6: Extract Results
# ============================================================================
print("\n[Step 6] Extracting results...")

# Get result from model (requires build_state_space() to be called)
result = model.get_result()
factors = result.Z  # (T, num_factors) - averaged factors
print(f"   Factors shape: {factors.shape} (T x num_factors)")

# Get training information
final_loss = model.loss_now if hasattr(model, 'loss_now') and model.loss_now is not None else None
print(f"   Final training loss: {final_loss:.6f}" if final_loss is not None else "   Final training loss: N/A")

# ============================================================================
# Step 7: Prediction
# ============================================================================
print("\n[Step 7] Making predictions...")

# Predict with horizon=6 (uses target_series from Dataset)
# Note: predict() requires build_state_space() to be called first
try:
    X_forecast, Z_forecast = model.predict(horizon=6, return_series=True, return_factors=True)
    print(f"   Forecast series shape: {X_forecast.shape} (horizon x num_target_series)")
    print(f"   Forecast factors shape: {Z_forecast.shape} (horizon x num_factors)")
    print(f"   First forecast value: {X_forecast[0, 0]:.6f}")
except Exception as e:
    print(f"   Prediction failed: {e}")

# ============================================================================
# Step 8: Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {len(selected_cols)} series")
print(f"✅ Features (covariates): {len(feature_cols)} series")
print(f"✅ Target: {target_col}")
print(f"✅ Model trained: {len(selected_cols)} series, {config.num_factors} factors, VAR(1) dynamics")
if X_forecast is not None:
    print(f"✅ Predictions generated: {X_forecast.shape[0]} periods ahead")
else:
    print(f"⚠️  Predictions: Failed (see error message above)")
print("=" * 80)

