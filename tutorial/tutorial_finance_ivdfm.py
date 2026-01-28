"""Tutorial: iVDFM for Finance Data

This tutorial demonstrates the complete workflow for training and prediction
using finance data with iVDFM (Identifiable Variational Dynamic Factor Model).
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

@hydra.main(version_base=None, config_path="../config", config_name="ivdfm_finance")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("iVDFM Tutorial: Finance Data")
    print("=" * 80)
    
    start_time = datetime.now()
    model = None
    dataset = None
    config = None
    df = None
    
    try:
        print("\n[Step 1] Loading data...")
        try:
            df = pd.read_csv(project_root / "data" / "finance.csv")
            if "date" not in df.columns:
                from datetime import datetime as dt
                start_date = dt(1980, 1, 1)
                df["date"] = pd.date_range(start=start_date, periods=len(df), freq="D")
            # Drop unused columns (do not use for targets/covariates)
            df = df.drop(columns=[c for c in ["forward_returns", "risk_free_rate"] if c in df.columns])
            print(f"   Data shape: {df.shape}")
        except Exception as e:
            print("\n[FAILURE] Step 1: Loading data")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 2] Creating Dataset...")
            config = iVDFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            
            # Finance caveat:
            # - single target series: market_forward_excess_returns
            # - all other series are covariates
            # - time index is the default context
            target_col = "market_forward_excess_returns"
            all_series = [c for c in df.columns if c != "date"]
            if target_col not in all_series:
                raise ValueError(f"Expected target column '{target_col}' not found in finance.csv")

            targets = [target_col]
            covariates = [c for c in all_series if c != target_col]

            df_data = df[["date"] + targets + covariates].copy()
            sequence_length = int(getattr(config, "sequence_length", 200) or 200)
            
            dataset = iVDFMDataset(
                data=df_data,
                time_idx="date",
                covariates=covariates,
                sequence_length=sequence_length,
                context=None,  # auxiliary context columns (none for finance tutorial)
                context_dim=config.context_dim,  # from YAML
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

            # Simple prediction quality check on the last 6 time steps (in scaled space)
            try:
                y_true = dataset.target[-6:, :]  # scaled targets used during training
                if forecast.shape != y_true.shape:
                    print(f"   WARNING: forecast shape {forecast.shape} != y_true shape {y_true.shape}; skipping metric.")
                else:
                    mse = np.mean((forecast - y_true) ** 2, axis=0)
                    mae = np.mean(np.abs(forecast - y_true), axis=0)
                    print("   Prediction MSE per target (scaled):", mse)
                    print("   Prediction MAE per target (scaled):", mae)
            except Exception as eval_e:
                print(f"   WARNING: prediction evaluation failed: {eval_e}")
        except Exception as e:
            print("\n[FAILURE] Step 4: Making predictions")
            traceback.print_exc()
            raise

        try:
            print("\n[Step 5] Saving model...")
            model_path = project_root / "models" / "ivdfm_finance.pt"
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
