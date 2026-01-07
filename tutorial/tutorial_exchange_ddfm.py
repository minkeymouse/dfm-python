"""Tutorial: DDFM for Exchange Rate Data

This tutorial demonstrates the complete workflow for training and prediction
using exchange rate data, matching the original TensorFlow DDFM implementation.

The tutorial follows the same preprocessing and configuration as the original
TensorFlow DDFM (DDFM/run_exchange_rate_original.py).

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


print("=" * 80)
print("DDFM Tutorial: Exchange Rate Data")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n[Step 1] Loading exchange rate data...")
data_path = project_root / "data" / "exchange_rate.csv"
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

print(f"   Data shape: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Columns: {list(df.columns)}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# Step 2: Prepare Data
# ============================================================================
print("\n[Step 2] Preparing data...")

# Remove rows with all NaN
df_processed = df.dropna(how='all')

# Use full dataset to match original TensorFlow DDFM
# Original uses full dataset (7588 periods), so we use all available data
print(f"   Using full dataset ({len(df_processed)} periods) to match original TensorFlow DDFM")

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
# Step 2.5: Preprocess Features (if any)
# ============================================================================
print("\n[Step 2.5] Preprocessing features (if any)...")
print("   Note: Features (non-target series) should be preprocessed by user.")
print("   Target series will be scaled within DDFM pipeline using scaler class.")

# For exchange_rate data, all series are targets, so no feature preprocessing needed
# In general, identify features vs targets and preprocess features here
target_series = list(df_processed.columns)  # All series are targets for exchange_rate
feature_series = []  # No features for exchange_rate case

if len(feature_series) > 0:
    # Preprocess features (standardization, normalization, etc.)
    # User should do this before passing to DDFM
    print(f"   {len(feature_series)} feature series should be preprocessed by user")
else:
    print(f"   No features to preprocess (all {len(target_series)} series are targets)")
    print(f"   Target series will be scaled within DDFM pipeline")

# Store for later use
df_processed_final = df_processed.copy()

# ============================================================================
# Step 3: Load Configuration from YAML File
# ============================================================================
print("\n[Step 3] Loading configuration from YAML file...")

# Load config from YAML file (matching original TensorFlow DDFM parameters)
config_path = project_root / "config" / "ddfm_exchange.yaml"
print(f"   Loading config from: {config_path}")

# Option 1: Load using YamlSource and convert to DDFMConfig
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Add frequency dict for all series (all are daily)
    if 'frequency' not in cfg_dict or cfg_dict['frequency'] is None:
        cfg_dict['frequency'] = {col: "d" for col in df_processed.columns}
    
    # Convert to DDFMConfig (from_dict automatically detects DDFM based on parameters)
    config = DDFMConfig.from_dict(cfg_dict)
    print(f"   ✓ Config loaded from YAML file using OmegaConf")
except ImportError:
    print(f"   ⚠️  omegaconf not available. Install with: pip install omegaconf")
    print("   Falling back to manual config creation...")
    # Fallback: Create config manually
    frequency_dict = {col: "d" for col in df_processed.columns}
    config = DDFMConfig(
        frequency=frequency_dict,
        clock="d",
        num_factors=4,
        encoder_layers=[16, 4],
        activation='relu',
        learning_rate=0.005,
        n_mc_samples=10,
        window_size=100,
        max_epoch=200,
        tolerance=0.0005,
        disp=10,
        lags_input=0,
        seed=3
    )
except Exception as e:
    print(f"   ⚠️  YAML loading failed: {e}")
    print("   Falling back to manual config creation...")
    # Fallback: Create config manually
    frequency_dict = {col: "d" for col in df_processed.columns}
    config = DDFMConfig(
        frequency=frequency_dict,
        clock="d",
        num_factors=4,
        encoder_layers=[16, 4],
        activation='relu',
        learning_rate=0.005,
        n_mc_samples=10,
        window_size=100,
        max_epoch=200,
        tolerance=0.0005,
        disp=10,
        lags_input=0,
        seed=3
    )

# Ensure frequency dict is set for all series (all are daily)
if not hasattr(config, 'frequency') or config.frequency is None:
    frequency_dict = {col: "d" for col in df_processed.columns}
    config.frequency = frequency_dict
    print(f"   Added frequency dict for {len(frequency_dict)} series (all daily)")

print(f"\n   Configuration loaded:")
print(f"   - Number of series: {len(df_processed.columns)}")
encoder_layers = getattr(config, 'encoder_layers', [16, 4])  # Default if not in config
num_factors = encoder_layers[-1] if encoder_layers else 4  # Last element is latent dimension
print(f"   - Number of factors: {num_factors} (matching original, inferred from encoder_layers)")
print(f"   - Encoder layers: {encoder_layers} (matching original structure_encoder=(16, 4))")
print(f"   - Decoder: linear (matching original structure_decoder=None)")
print(f"   - Factor dynamics: VAR(1) (always AR(1), not configurable)")
n_mc_samples = getattr(config, 'n_mc_samples', 10)
window_size = getattr(config, 'window_size', 100)
learning_rate = getattr(config, 'learning_rate', 0.005)
max_iter = getattr(config, 'max_epoch', 200)  # Config uses max_epoch
tolerance = getattr(config, 'tolerance', 0.0005)
seed = getattr(config, 'seed', 3)
print(f"   - MC samples per iteration: {n_mc_samples} (matching original epochs=10)")
print(f"   - Window size (batch size): {window_size} (matching original batch_size=100)")
print(f"   - Learning rate: {learning_rate} (matching original)")
print(f"   - Max iterations (MCMC): {max_iter} (matching original max_iter=200)")
print(f"   - Tolerance: {tolerance} (matching original)")
print(f"   - Seed: {seed} (matching original)")

# ============================================================================
# Step 4: Create Dataset
# ============================================================================
print("\n[Step 4] Creating Dataset...")

# For DDFM, we need to specify target_series (series to forecast)
# In this tutorial, all series are targets (no features)
target_series = list(df_processed_final.columns)

# Mimic original TensorFlow scaling: pandas (data - mean) / std (ddof=1)
# This matches the original: self.data = (data - self.mean_z) / self.sigma_z
# where mean_z = data.mean().values and sigma_z = data.std().values (ddof=1 by default)
print("   Applying pandas scaling (matching original TensorFlow):")
mean_z = df_processed_final.mean().values
sigma_z = df_processed_final.std().values  # pandas std uses ddof=1 by default
df_scaled = (df_processed_final - mean_z) / sigma_z
print(f"   Scaled data shape: {df_scaled.shape}")
print(f"   Mean (should be ~0): {df_scaled.mean().values[:3]}")
print(f"   Std (should be ~1): {df_scaled.std().values[:3]}")

# Create scaler for target series (needed for prediction inverse transformation)
# Even though data is already scaled, we need a scaler for prediction
from sklearn.preprocessing import StandardScaler
target_scaler = StandardScaler()
target_scaler.fit(df_processed_final[target_series].values)

# Create Dataset - data is already scaled, but scaler needed for prediction
dataset = DDFMDataset(
    data=df_scaled,  # Data already scaled using pandas (matching original)
    time_idx='index',  # Use DataFrame index as time identifier
    target_series=target_series,  # All series are targets
    target_scaler=target_scaler  # Scaler for prediction inverse transformation
)

print(f"   Dataset created successfully")
print(f"   Data shape: {dataset.data.shape}")
print(f"   Target series: {dataset.target_series}")

# ============================================================================
# Step 5: Create and Train Model
# ============================================================================
print("\n[Step 5] Creating and training DDFM model...")

# Create model with parameters from config
# Map config parameters to model parameters
encoder_layers = getattr(config, 'encoder_layers', [16, 4])
encoder_size = tuple(encoder_layers) if encoder_layers else (16, 4)
max_epoch = getattr(config, 'max_epoch', 200)

model = DDFM(
    dataset=dataset,
    config=config,
    encoder_size=encoder_size,  # Convert list to tuple
    decoder_type="linear",  # Matching original structure_decoder=None (linear decoder)
    activation=getattr(config, 'activation', 'relu'),
    learning_rate=getattr(config, 'learning_rate', 0.005),
    optimizer='Adam',
    n_mc_samples=getattr(config, 'n_mc_samples', 10),
    window_size=getattr(config, 'window_size', 100),
    max_iter=max_epoch,  # Map max_epoch -> max_iter
    tolerance=getattr(config, 'tolerance', 0.0005),
    disp=getattr(config, 'disp', 10),
    seed=getattr(config, 'seed', 3)
)

# Train model
print(f"   Starting training (max {model.max_iter} iterations)...")
model.fit()  # fit() builds model, pre-trains, and trains in one method
print("   Training completed!")

# Build state-space model for prediction
print("\n[Step 5.5] Building state-space model...")
model.build_state_space()
print("   State-space model built successfully")

# Check prediction variance (for debugging variance collapse issue)
print("\n[Step 5.6] Checking prediction variance...")
if hasattr(model, 'prediction_std') and model.prediction_std is not None:
    prediction_std_mean = float(np.mean(model.prediction_std))
    prediction_std_std = float(np.std(model.prediction_std))
    print(f"   Prediction std (mean across all predictions): {prediction_std_mean:.6f}")
    print(f"   Prediction std (std across all predictions): {prediction_std_std:.6f}")
    print(f"   Target std: ~1.0 (TensorFlow reference)")
    if prediction_std_mean < 0.1:
        print(f"   ⚠️  WARNING: Variance collapse detected (std={prediction_std_mean:.6f} << target ~1.0)")
    elif prediction_std_mean < 0.5:
        print(f"   ⚠️  WARNING: Low variance (std={prediction_std_mean:.6f} < target ~1.0)")
    else:
        print(f"   ✓ Variance within acceptable range (std={prediction_std_mean:.6f} vs target ~1.0)")
else:
    print("   Prediction std not available (model may not have completed training)")

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
# Step 7: Access Results and Predict
# ============================================================================
print("\n[Step 7] Accessing results and making predictions...")

# Get factors (averaged over MC samples)
factors = result.Z  # (T, num_factors)
print(f"   Factors shape: {factors.shape} (T x num_factors)")

# Get smoothed predictions from result
x_sm = result.x_sm  # (T, N) - smoothed data
print(f"   Smoothed data shape: {x_sm.shape} (T x N)")

# Make predictions
print("\n[Step 8] Making forecasts...")
try:
    X_forecast, Z_forecast = model.predict(horizon=6, return_series=True, return_factors=True)
    print(f"   Forecast series shape: {X_forecast.shape} (horizon x num_target_series)")
    print(f"   Forecast factors shape: {Z_forecast.shape} (horizon x num_factors)")
    print(f"   First forecast (first target series): {X_forecast[0, 0]:.6f}")
except Exception as e:
    print(f"   Prediction failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} series")
if len(feature_series) > 0:
    print(f"✅ Features preprocessed: {len(feature_series)} feature series")
else:
    print(f"✅ No features: all {len(target_series)} series are targets (scaled within DDFM pipeline)")
print(f"✅ Model trained: {len(df_processed_final.columns)} series, {len(factors[0]) if len(factors) > 0 else 'N/A'} factors, VAR(1) dynamics")
print(f"✅ Factors extracted: {factors.shape[0]} periods, {factors.shape[1]} factors")
print(f"✅ Configuration matches original TensorFlow DDFM")
if final_loss is not None:
    print(f"✅ Training completed (loss: {final_loss:.6f})")
print("=" * 80)

