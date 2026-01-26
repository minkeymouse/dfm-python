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

print("\n[Step 3] Creating configuration...")
frequency_dict = {col: "d" for col in selected_cols}
blocks_config = {
    "Block_Global": {
        "num_factors": 1,
        "series": selected_cols
    }
}

config = DFMConfig(
    frequency=frequency_dict,
    blocks=blocks_config,
    clock="d",
    max_iter=3,
    threshold=1e-2
)

print("\n[Step 4] Creating Dataset...")
dataset = DFMDataset(
    config=config,
    data=df_processed,
    time_index='date'
)

print("\n[Step 5] Training DFM model...")
model = DFM(dataset=dataset, config=config)
model.fit()

result = model.result
print(f"   Converged: {result.converged}")

print("\n[Step 6] Making predictions...")
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"   Forecast shape: {X_forecast.shape}")
