"""Tutorial: DFM for Finance Data

This tutorial demonstrates the complete workflow for training and prediction
using finance data with market_forward_excess_returns as the target variable.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer

print("=" * 80)
print("DFM Tutorial: Finance Data")
print("=" * 80)

print("\n[Step 1] Loading finance data...")
data_path = project_root / "data" / "finance.csv"
df = pd.read_csv(data_path)

print(f"   Data shape: {df.shape}")

print("\n[Step 2] Preparing data...")
target_col = "market_forward_excess_returns"
exclude_cols = ["risk_free_rate", "forward_returns"]

selected_cols = [col for col in df.columns if col not in exclude_cols and col != 'date_id']
if target_col not in selected_cols:
    selected_cols.append(target_col)

df_processed = df[selected_cols].copy()
df_processed = df_processed.dropna(how='all')

print(f"   Data shape: {df_processed.shape}")

print("\n[Step 2.5] Imputing missing values...")
X_cols = [col for col in selected_cols if col != target_col]
y_col = target_col

X = df_processed[X_cols].copy()
y = df_processed[[y_col]].copy()

# Create imputation pipeline: forward fill -> backward fill
imputation_pipeline = Pipeline([
    ('impute_ffill', Imputer(method="ffill")),
    ('impute_bfill', Imputer(method="bfill"))
])

# Fit and transform (imputation only)
imputation_pipeline.fit(X)
X_imputed = imputation_pipeline.transform(X)

if isinstance(X_imputed, np.ndarray):
    X_imputed = pd.DataFrame(X_imputed, columns=X_cols, index=X.index)

df_preprocessed = pd.concat([X_imputed, y], axis=1)

print("\n[Step 2.6] Creating scaler for DFM...")
print("   Note: Scaling is required for numerical stability.")
print("   The scaler will be passed to DFM and applied internally during fit().")
# Create scaler to pass to DFM (will be fitted during model.fit())
scaler = StandardScaler()

if 'date_id' in df.columns:
    df_preprocessed['date'] = pd.to_datetime(df['date_id'].iloc[-len(df_preprocessed):].values)
else:
    n_periods = len(df_preprocessed)
    start_date = datetime(1980, 1, 1)
    df_preprocessed['date'] = pd.date_range(start=start_date, periods=n_periods, freq='M')

print("\n[Step 3] Loading configuration...")
config_path = project_root / "config" / "dfm_finance.yaml"
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# Populate series and frequency from data
clock = config_dict.get('clock', 'm')
config_dict['frequency'] = {col: clock for col in selected_cols}
config_dict['blocks'] = {
    "Block_Global": {
        "num_factors": 2,
        "series": selected_cols
    }
}

config = DFMConfig.from_dict(config_dict)

print("\n[Step 4] Creating Dataset...")
dataset = DFMDataset(
    config=config,
    data=df_preprocessed,
    time_index='date'
)

print(f"   Dataset created: {dataset.variables.shape}")

print("\n[Step 5] Training DFM model...")
model = DFM(dataset=dataset, config=config, scaler=scaler)
model.fit()

result = model.result
print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")

print("\n[Step 6] Making predictions...")
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"   Forecast shape: {X_forecast.shape}")

print("\n" + result.summary())
