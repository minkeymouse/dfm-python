"""Tutorial: DFM for Macro Data

This tutorial demonstrates the complete workflow for training and prediction
using macro data with multiple target variables.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from dfm_python.config.constants import TUTORIAL_MAX_PERIODS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.impute import Imputer

print("=" * 80)
print("DFM Tutorial: Macro Data")
print("=" * 80)

print("\n[Step 1] Loading macro data...")
data_path = project_root / "data" / "macro.csv"
df = pd.read_csv(data_path)

print(f"   Data shape: {df.shape}")

print("\n[Step 2] Preparing data...")
target_cols = ["KOEQUIPTE", "KOWRCCNSE", "KOIMPCONA"]
selected_cols = [
    "KOEMPTOTO", "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF"
] + target_cols

selected_cols = [col for col in selected_cols if col in df.columns]

df_processed = df[selected_cols + ["date"]].copy() if 'date' in df.columns else df[selected_cols].copy()
df_processed = df_processed.dropna(how='all')

if len(df_processed) > TUTORIAL_MAX_PERIODS:
    df_processed = df_processed.iloc[-TUTORIAL_MAX_PERIODS:]

if 'date' in df_processed.columns:
    df_processed["date"] = pd.to_datetime(df_processed["date"])

print("\n[Step 2.5] Creating preprocessing pipeline...")
X_cols = [col for col in selected_cols if col not in target_cols]
y_cols = target_cols

if 'date' in df_processed.columns:
    df_for_preprocessing = df_processed.drop(columns=['date'])
else:
    df_for_preprocessing = df_processed

X = df_for_preprocessing[X_cols].copy()
y = df_for_preprocessing[y_cols].copy()

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

if 'date' in df_processed.columns:
    df_preprocessed['date'] = df_processed['date'].values

print("\n[Step 3] Loading configuration...")
config_path = project_root / "config" / "dfm_macro.yaml"
import yaml
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

clock = config_dict.get('clock', 'w')
config_dict['frequency'] = {col: clock for col in selected_cols}
config_dict['blocks'] = {
    "Block_Global": {
        "num_factors": 1,
        "series": selected_cols
    }
}
config_dict['max_iter'] = 3
config_dict['threshold'] = 1e-2

config = DFMConfig.from_dict(config_dict)

print("\n[Step 4] Creating Dataset...")
if 'date' not in df_preprocessed.columns:
    n_periods = len(df_preprocessed)
    df_preprocessed['date'] = pd.date_range(start='1985-01-01', periods=n_periods, freq='W')

dataset = DFMDataset(
    config=config,
    data=df_preprocessed,
    time_index='date'
)

print("\n[Step 5] Training DFM model...")
model = DFM(dataset=dataset, config=config)
model.fit()

result = model.result
print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")

print("\n[Step 6] Making predictions...")
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"   Forecast shape: {X_forecast.shape}")

print("\n" + result.summary())
