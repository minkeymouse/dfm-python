"""Tutorial: DFM for Exchange Data

This tutorial demonstrates the complete workflow for training and prediction
using exchange rate data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig

print("=" * 80)
print("DFM Tutorial: Exchange Data")
print("=" * 80)

print("\n[Step 1] Loading exchange data...")
data_path = project_root / "data" / "exchange_rate.csv"
df = pd.read_csv(data_path)

print(f"   Data shape: {df.shape}")

print("\n[Step 2] Preparing data...")
if 'date' not in df.columns:
    df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

df_processed = df.dropna(how='all')
selected_cols = [col for col in df_processed.columns if col != 'date']

print("\n[Step 2.5] Imputing missing values...")
# Prepare data for imputation (exclude date column)
if 'date' in df_processed.columns:
    df_for_imputation = df_processed.drop(columns=['date']).copy()
else:
    df_for_imputation = df_processed.copy()

X = df_for_imputation[selected_cols].copy()

# Create imputation pipeline: forward fill -> backward fill
imputation_pipeline = Pipeline([
    ('impute_ffill', Imputer(method="ffill")),
    ('impute_bfill', Imputer(method="bfill"))
])

# Fit and transform (imputation only)
imputation_pipeline.fit(X)
X_imputed = imputation_pipeline.transform(X)

# Convert back to DataFrame if needed
if isinstance(X_imputed, np.ndarray):
    X_imputed = pd.DataFrame(X_imputed, columns=selected_cols, index=X.index)

# Reconstruct dataframe with imputed data
df_processed = X_imputed.copy()
if 'date' in df.columns:
    df_processed['date'] = df['date'].values

print("\n[Step 2.6] Creating scaler for DFM...")
print("   Note: Scaling is required for numerical stability in daily factor models.")
print("   The scaler will be passed to DFM and applied internally during fit().")
# Create scaler to pass to DFM (will be fitted during model.fit())
scaler = StandardScaler()

print("\n[Step 3] Loading configuration...")
config_path = project_root / "config" / "dfm_exchange.yaml"
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# Populate series and frequency from data
config_dict['frequency'] = {col: "d" for col in selected_cols}
config_dict['blocks']['Block_Global']['series'] = selected_cols
config_dict['blocks']['Block_Global']['num_factors'] = 2  # Override to 2 factors

config = DFMConfig.from_dict(config_dict)

print("\n[Step 4] Creating Dataset...")
dataset = DFMDataset(
    config=config,
    data=df_processed,
    time_index='date'
)

print("\n[Step 5] Training DFM model...")
model = DFM(dataset=dataset, config=config, scaler=scaler)
model.fit()

result = model.result
print(f"   Converged: {result.converged}")

print("\n[Step 6] Making predictions...")
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"   Forecast shape: {X_forecast.shape}")
