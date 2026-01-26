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
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS
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

if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]

print(f"   Data shape: {df_processed.shape}")

print("\n[Step 2.5] Creating preprocessing pipeline...")
X_cols = [col for col in selected_cols if col != target_col]
y_col = target_col

X = df_processed[X_cols].copy()
y = df_processed[[y_col]].copy()

X_pipeline = Pipeline([
    ('impute_ffill', Imputer(method="ffill")),
    ('impute_bfill', Imputer(method="bfill")),
    ('scaler', StandardScaler())
])

X_pipeline.fit(X)
X_preprocessed = X_pipeline.transform(X)

if isinstance(X_preprocessed, np.ndarray):
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X_cols, index=X.index)

df_preprocessed = pd.concat([X_preprocessed, y], axis=1)

if 'date_id' in df.columns:
    df_preprocessed['date'] = pd.to_datetime(df['date_id'].iloc[-len(df_preprocessed):].values)
else:
    n_periods = len(df_preprocessed)
    start_date = datetime(1980, 1, 1)
    df_preprocessed['date'] = pd.date_range(start=start_date, periods=n_periods, freq='M')

print("\n[Step 3] Creating configuration...")
frequency_dict = {col: "m" for col in selected_cols}
blocks_config = {
    "Block_Global": {
        "num_factors": 1,
        "series": selected_cols
    }
}

config = DFMConfig(
    frequency=frequency_dict,
    blocks=blocks_config,
    clock="m",
    max_iter=3,
    threshold=1e-2
)

print("\n[Step 4] Creating Dataset...")
dataset = DFMDataset(
    config=config,
    data=df_preprocessed,
    time_index='date'
)

print(f"   Dataset created: {dataset.variables.shape}")

print("\n[Step 5] Training DFM model...")
model = DFM(dataset=dataset, config=config)
model.fit()

result = model.result
print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")

print("\n[Step 6] Making predictions...")
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"   Forecast shape: {X_forecast.shape}")

print("\n" + result.summary())
