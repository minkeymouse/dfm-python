"""Tutorial: iVDFM for Macro Data

This tutorial demonstrates the complete workflow for training and prediction
using macro data with iVDFM (Identifiable Variational Dynamic Factor Model).
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

def _select_columns_for_speed(df: pd.DataFrame, *, target_col: str, max_cols: int) -> pd.DataFrame:
    """Select a small, deterministic subset of columns for fast tutorials."""
    cols = [c for c in df.columns if c != "date"]
    if target_col in cols:
        cols.remove(target_col)
        keep = [target_col] + cols[: max(0, max_cols - 1)]
    else:
        keep = cols[:max_cols]
    return df[[c for c in keep if c in df.columns]].copy()

@hydra.main(version_base=None, config_path="../config", config_name="ivdfm_macro")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("iVDFM Tutorial: Macro Data")
    print("=" * 80)
    
    start_time = datetime.now()
    model = None
    dataset = None
    config = None
    df = None
    
    try:
        print("\n[Step 1] Loading data...")
        try:
            df = pd.read_csv(project_root / "data" / "macro.csv")
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            print(f"   Data shape: {df.shape}")
        except Exception as e:
            print("\n[FAILURE] Step 1: Loading data")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 2] Creating Dataset...")
            config = iVDFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            
            # Macro caveat:
            # - choose 3 monthly (sparse) series as targets
            # - all other series are covariates
            # - time index is the default context (context_dim=1)
            target_series = ["KOEQUIPTE", "KOEMPTOTO", "KOHWRWEMP"]
            missing = [c for c in target_series if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required target series in macro.csv: {missing}")

            all_series = [c for c in df.columns if c != "date"]
            covariates = [c for c in all_series if c not in target_series]

            # Keep tutorial fast: use a subset of covariates but keep targets intact.
            covariates = covariates[:32]
            df_data = df[["date"] + target_series + covariates].copy()
            sequence_length = int(getattr(config, "sequence_length", 100) or 100)
            
            dataset = iVDFMDataset(
                data=df_data,
                time_idx="date",
                covariates=covariates,
                sequence_length=sequence_length,
                context=None,      # no auxiliary context columns
                context_dim=config.context_dim,     # from YAML
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
                sequence_length=sequence_length,
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
            model_path = project_root / "models" / "ivdfm_macro.pt"
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
