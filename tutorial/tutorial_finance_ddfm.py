"""Tutorial: DDFM for Finance Data

This tutorial demonstrates the complete workflow for training and prediction
using finance data with market_forward_excess_returns as the target variable.
"""

import sys
from pathlib import Path
import traceback
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from dfm_python import DDFM, DDFMDataset
from dfm_python.config import DDFMConfig
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf

def save_failure_report(step: str, error: Exception, context: dict, project_root: Path):
    """Save detailed failure report to file."""
    reports_dir = project_root / "failure_reports"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"finance_ddfm_failure_{timestamp}.json"
    
    report = {
        "experiment": "finance_ddfm",
        "step": step,
        "timestamp": timestamp,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n[FAILURE] Error report saved to: {report_file}")
    return report_file

@hydra.main(version_base=None, config_path="../config", config_name="ddfm_finance")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("DDFM Tutorial: Finance Data")
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
            context = {
                "data_shape": df.shape if df is not None else None,
                "data_columns": list(df.columns) if df is not None else None,
            }
            save_failure_report("Step 1: Loading data", e, context, project_root)
            raise

        try:
            print("\n[Step 2] Creating Dataset...")
            config = DDFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            
            # Finance DDFM: covariates are all columns except target_col
            target_col = getattr(config, "target_col", "market_forward_excess_returns")
            covariates_mode = getattr(config, "covariates", "all_except_target")

            # Keep all columns; DDFMDataset deduces targets = all - covariates
            df_filtered = df.copy()

            all_series = [c for c in df_filtered.columns if c not in ("date",)]
            if target_col not in all_series:
                raise ValueError(f"Expected target column '{target_col}' not found in finance.csv")

            if isinstance(covariates_mode, str) and covariates_mode.lower() == "all_except_target":
                covariates = [c for c in all_series if c != target_col]
            elif covariates_mode in (None, "none"):
                covariates = []
            else:
                # Allow explicit list in config if user provides one
                covariates = [c for c in (covariates_mode or []) if c in all_series and c != target_col]
            
            dataset = DDFMDataset(
                data=df_filtered,
                time_idx='date',
                covariates=covariates if covariates else None,
                scaler=StandardScaler()
            )
            print(f"   Dataset created: {dataset.data.shape}")
        except Exception as e:
            context = {
                "config_max_epoch": cfg.get('max_epoch', None),
                "data_shape": df.shape if df is not None else None,
            }
            save_failure_report("Step 2: Creating Dataset", e, context, project_root)
            raise

        try:
            print("\n[Step 3] Training DDFM model...")
            encoder_layers = getattr(config, 'encoder_layers', [32, 1])
            encoder_size = tuple(encoder_layers) if encoder_layers else (32, 1)
            decoder_type = getattr(config, "decoder_type", "linear")
            
            model = DDFM(
                dataset=dataset,
                config=config,
                encoder_size=encoder_size,
                decoder_type=decoder_type,
                activation=getattr(config, 'activation', 'relu'),
                learning_rate=getattr(config, 'learning_rate', 0.001),
                optimizer='Adam',
                n_mc_samples=getattr(config, 'n_mc_samples', 1),
                window_size=getattr(config, 'window_size', 100),
                max_iter=getattr(config, 'max_epoch', 200),  # Config uses max_epoch, DDFM uses max_iter
                tolerance=getattr(config, 'tolerance', 0.0005),
                disp=getattr(config, 'disp', 10),
                seed=getattr(config, 'seed', None),
                interpolation_method=getattr(config, 'interpolation_method', 'linear'),
                interpolation_limit=getattr(config, 'interpolation_limit', 10),
                interpolation_limit_direction=getattr(config, 'interpolation_limit_direction', 'both')
            )
            model.fit()
            model.build_state_space()
            
            result = model.get_result()
            print(f"   Converged: {result.converged if hasattr(result, 'converged') else 'N/A'}, Iterations: {getattr(model, '_num_iter', 'N/A')}")
        except Exception as e:
            context = {
                "config_max_epoch": cfg.get('max_epoch', None),
                "dataset_shape": dataset.data.shape if dataset is not None else None,
                "elapsed_time": str(datetime.now() - start_time),
            }
            save_failure_report("Step 3: Training DDFM model", e, context, project_root)
            raise

        try:
            print("\n[Step 4] Making predictions...")
            X_forecast, Z_forecast = model.predict(horizon=6, return_series=True, return_factors=True)
            print(f"   Forecast shape: {X_forecast.shape}")
            
            if np.any(np.isnan(X_forecast)) or np.any(np.isinf(X_forecast)):
                print(f"   WARNING: Forecast contains NaN or Inf!")

            # Simple prediction quality check on the last 6 time steps (scaled targets)
            try:
                # For finance, target is a single series: target_col
                y_scaled = dataset.y  # (T, 1) scaled targets
                y_true = y_scaled[-6:, :]
                y_pred = X_forecast  # DDFM.predict returns targets in scaled space when return_series=True
                if y_pred.shape != y_true.shape:
                    print(f"   WARNING: forecast shape {y_pred.shape} != y_true shape {y_true.shape}; skipping metric.")
                else:
                    mse = np.mean((y_pred - y_true) ** 2, axis=0)
                    mae = np.mean(np.abs(y_pred - y_true), axis=0)
                    print("   Prediction MSE (scaled):", mse)
                    print("   Prediction MAE (scaled):", mae)
            except Exception as eval_e:
                print(f"   WARNING: prediction evaluation failed: {eval_e}")
        except Exception as e:
            context = {
                "result_converged": result.converged if 'result' in locals() else None,
            }
            save_failure_report("Step 4: Making predictions", e, context, project_root)
            raise

        try:
            print("\n[Step 5] Saving model...")
            model_path = project_root / "models" / "ddfm_finance.pkl"
            model_path.parent.mkdir(exist_ok=True)
            model.save(model_path)
            print(f"   Model saved to: {model_path}")
        except Exception as e:
            context = {
                "model_path": str(model_path) if 'model_path' in locals() else None,
                "result_converged": result.converged if 'result' in locals() else None,
            }
            save_failure_report("Step 5: Saving model", e, context, project_root)
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
