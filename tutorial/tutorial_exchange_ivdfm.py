"""Tutorial: iVDFM for Exchange Rate Data

This tutorial demonstrates the complete workflow for training and prediction
using exchange rate data with iVDFM (Identifiable Variational Dynamic Factor Model).
"""

import sys
from pathlib import Path
import traceback
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from dfm_python.models.ivdfm.ivdfm import iVDFM
from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
from dfm_python.config.schema.model import iVDFMConfig
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="ivdfm_exchange")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("iVDFM Tutorial: Exchange Rate Data")
    print("=" * 80)
    
    start_time = datetime.now()
    model = None
    dataset = None
    config = None
    df = None
    
    try:
        print("\n[Step 1] Loading data...")
        try:
            df = pd.read_csv(project_root / "data" / "exchange_rate.csv", index_col=0, parse_dates=True)
            print(f"   Data shape: {df.shape}")
        except Exception as e:
            print("\n[FAILURE] Step 1: Loading data")
            traceback.print_exc()
            raise

        # Keep tutorial fast: use a small slice by default for quick verification
        # Remove this line to use full dataset
        # df = df.iloc[:300].copy()
        # print(f"   Using slice for tutorial speed: {df.shape}")

        try:
            print("\n[Step 2] Creating Dataset...")
            config = iVDFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            dataset = iVDFMDataset(
                data=df,
                time_idx=None,          # exchange data uses index; dataset will default to positional time
                covariates=None,        # no covariates for exchange
                sequence_length=config.sequence_length,
                context=None,           # auxiliary context: none
                context_dim=config.context_dim,  # time-only context from YAML
                scaler=config.scaler,  # Scaler string from config (applied to targets only, not context)
            )
            print(f"   Dataset created: target_length={dataset.target_length}, context_length={dataset.context_length}")
        except Exception as e:
            print("\n[FAILURE] Step 2: Creating Dataset")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 3] Training iVDFM model...")
            model = iVDFM(
                config=config,
                device=None,  # Auto-detect
            )
            model.fit(data=dataset)
            
            result = model.get_result()
            print(f"   Training completed: ELBO={result.training_elbo if hasattr(result, 'training_elbo') else 'N/A'}")
            if getattr(result, "factors", None) is not None:
                f = np.asarray(result.factors)
                f2 = f.reshape((-1, f.shape[-1])) if f.ndim == 3 else f
                print(f"   Factors shape: {f.shape}")
                print(f"   Factor std (per-dim): {np.std(f2, axis=0)}")
        except Exception as e:
            print("\n[FAILURE] Step 3: Training iVDFM model")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 4] Making predictions...")
            forecast = model.predict(horizon=6)
            print(f"   Forecast shape: {forecast.shape}")
            
            if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                print(f"   WARNING: Forecast contains NaN or Inf!")
        except Exception as e:
            print("\n[FAILURE] Step 4: Making predictions")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 5] Saving model...")
            model_path = project_root / "models" / "ivdfm_exchange.pt"
            model_path.parent.mkdir(exist_ok=True)
            model.save(model_path)
            print(f"   Model saved to: {model_path}")
        except Exception as e:
            print("\n[FAILURE] Step 5: Saving model")
            traceback.print_exc()
            raise

        elapsed = datetime.now() - start_time
        print(f"\n[SUCCESS] Tutorial completed in {elapsed}")
        
    except Exception as e:
        elapsed = datetime.now() - start_time
        print(f"\n[FAILURE] Tutorial failed after {elapsed}")
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()
