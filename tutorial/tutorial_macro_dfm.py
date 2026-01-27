"""Tutorial: DFM for Macro Data

This tutorial demonstrates the complete workflow for training and prediction
using macro data with multiple target variables.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="dfm_macro")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("DFM Tutorial: Macro Data")
    print("=" * 80)

    print("\n[Step 1] Loading data...")
    df = pd.read_csv(project_root / "data" / "macro.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    print(f"   Data shape: {df.shape}")

    print("\n[Step 2] Creating Dataset...")
    config = DFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    dataset = DFMDataset(config=config, data=df, time_index='date')

    print("\n[Step 3] Training DFM model...")
    model = DFM(dataset=dataset, config=config, scaler=StandardScaler())
    model.fit()

    result = model.result
    print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")

    print("\n[Step 4] Making predictions...")
    X_forecast, Z_forecast = model.predict(horizon=6)
    print(f"   Forecast shape: {X_forecast.shape}")

    print("\n[Step 5] Saving model...")
    model_path = project_root / "models" / "dfm_macro.pkl"
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    print(f"   Model saved to: {model_path}")

    print("\n" + result.summary())

if __name__ == "__main__":
    main()
