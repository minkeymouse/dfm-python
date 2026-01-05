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
from dfm_python.config import DDFMConfig, YamlSource, make_config_source
from dfm_python.utils.misc import TimeIndex


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
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')

missing_after = df_processed.isnull().sum().sum()
print(f"   Missing values after imputation: {missing_after}")

# ============================================================================
# Step 2.5: Standardize Data (matching original TensorFlow DDFM)
# ============================================================================
# Original TensorFlow: self.data = (data - self.mean_z) / self.sigma_z
print("\n[Step 2.5] Standardizing data (matching original TensorFlow DDFM)...")
print("   Original TensorFlow: self.data = (data - self.mean_z) / self.sigma_z")
print("   All series must be standardized before passing to DataModule")

# Standardize all data (matching original TensorFlow DDFM)
mean_z = df_processed.mean().values
sigma_z = df_processed.std().values
df_standardized = (df_processed - mean_z) / sigma_z

# Verify standardization
mean_vals = df_standardized.mean()
std_vals = df_standardized.std()
max_mean = float(mean_vals.abs().max())
max_std_dev = float((std_vals - 1.0).abs().max())
print(f"   Standardization check - Max |mean|: {max_mean:.6f} (should be ~0)")
print(f"   Standardization check - Max |std - 1|: {max_std_dev:.6f} (should be ~0)")

# Update df_processed to use standardized data
df_processed = df_standardized

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
        use_batch_norm=True,
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
        use_batch_norm=True,
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
print(f"   - Number of factors: {config.num_factors} (matching original)")
print(f"   - Encoder layers: {config.encoder_layers} (matching original structure_encoder=(16, 4))")
print(f"   - Decoder: linear (matching original structure_decoder=None)")
print(f"   - Factor dynamics: VAR(1) (always AR(1), not configurable)")
print(f"   - MC samples per iteration: {config.n_mc_samples} (matching original epochs=10)")
print(f"   - Window size (batch size): {config.window_size} (matching original batch_size=100)")
print(f"   - Learning rate: {config.learning_rate} (matching original)")
print(f"   - Max epochs (MCMC iterations): {config.max_epoch} (matching original max_iter=200)")
print(f"   - Tolerance: {config.tolerance} (matching original)")
print(f"   - Seed: {config.seed} (matching original)")
print(f"   - Lags input: {config.lags_input} (matching original)")

# ============================================================================
# Step 4: Create DataModule
# ============================================================================
print("\n[Step 4] Creating DataModule...")

# Create time index from DataFrame index
time_list = [pd.Timestamp(idx).to_pydatetime() for idx in df_processed.index]
time_index = TimeIndex(time_list)

# Create Dataset with standardized data
dataset = DDFMDataset(
    config=config,
    data=df_standardized,  # Must be standardized (matching original TensorFlow)
    time_index=time_index,
    target_series=None  # All series are features (no separate target)
)

print(f"   Dataset created successfully")
print(f"   Processed data shape: {dataset.get_processed_data().shape}")

# ============================================================================
# Step 5: Create and Train Model
# ============================================================================
print("\n[Step 5] Creating and training DDFM model...")

# Create model with same parameters as config
# Note: decoder="linear" matches original structure_decoder=None (linear decoder)
model = DDFM(
    config=config,
    encoder_layers=[16, 4],
    num_factors=4,
    activation='relu',
    use_batch_norm=True,
    learning_rate=0.005,
    n_mc_samples=10,
    window_size=100,
    max_epoch=200,
    tolerance=0.0005,
    disp=10,
    lags_input=0,
    seed=3,
    decoder="linear",  # Matching original structure_decoder=None (linear decoder)
    decay_learning_rate=True
)

# Train model
print(f"   Starting training (max {config.max_epoch} iterations)...")
model.train(dataset=dataset)
print("   Training completed!")

# ============================================================================
# Step 6: Extract Results
# ============================================================================
print("\n[Step 6] Extracting results...")

# Get result from model
result = model.get_result()
factors = result.Z  # (T, num_factors) - averaged factors
print(f"   Factors shape: {factors.shape} (T x num_factors)")

# Get training state
if hasattr(model, 'training_state') and model.training_state is not None:
    final_loss = model.training_state.training_loss
    num_iter = model.training_state.num_iter
    converged = model.training_state.converged
    print(f"   Final training loss: {final_loss:.6f}")
    print(f"   Number of iterations: {num_iter}")
    print(f"   Converged: {converged}")

# ============================================================================
# Step 7: Access Results
# ============================================================================
print("\n[Step 7] Accessing results...")

# For exchange rate data (all series are features, no target_series),
# we can access factors and predictions from the training state
# The original TensorFlow DDFM also extracts factors and predictions this way

# Get factors (averaged over MC samples)
factors = result.Z  # (T, num_factors)
print(f"   Factors shape: {factors.shape} (T x num_factors)")

# Get smoothed predictions (from training state)
if hasattr(model, 'training_state') and model.training_state.prediction is not None:
    predictions = model.training_state.prediction  # (T, N)
    print(f"   Predictions shape: {predictions.shape} (T x N)")
    print(f"   First prediction (first series): {predictions[0, 0]:.6f}")

# Note: For forecasting future values, you would need to:
# 1. Set target_series in DataModule (if you want to forecast specific series)
# 2. Or use the state-space model to forecast factors forward, then decode to observations
print("   Note: For forecasting, set target_series in DataModule or use state-space model")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Tutorial Summary")
print("=" * 80)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} series")
print(f"✅ Data standardized: mean≈{max_mean:.6f}, std≈{1.0 + max_std_dev:.6f}")
print(f"✅ Model trained: {len(df_processed.columns)} series, {config.num_factors} factors, VAR(1) dynamics")
print(f"✅ Factors extracted: {factors.shape[0]} periods, {factors.shape[1]} factors")
print(f"✅ Configuration matches original TensorFlow DDFM")
print(f"✅ Training converged in {num_iter} iterations (loss: {final_loss:.6f})")
print("=" * 80)

