"""Tutorial: DDFM for Macro Data

This tutorial demonstrates the complete workflow for training and prediction
using macro data with KOEQUIPTE as the target variable.
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
    report_file = reports_dir / f"macro_ddfm_failure_{timestamp}.json"
    
    report = {
        "experiment": "macro_ddfm",
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

@hydra.main(version_base=None, config_path="../config", config_name="ddfm_macro")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("DDFM Tutorial: Macro Data")
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
            context = {
                "data_shape": df.shape if df is not None else None,
                "data_columns": list(df.columns) if df is not None else None,
            }
            save_failure_report("Step 1: Loading data", e, context, project_root)
            raise

        try:
            print("\n[Step 2] Creating Dataset...")
            config = DDFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            
            # Filter data to only include series specified in config
            all_block_series = set()
            if hasattr(config, 'blocks') and config.blocks:
                for block_name, block_config in config.blocks.items():
                    if isinstance(block_config, dict) and 'series' in block_config:
                        all_block_series.update(block_config['series'])
            
            columns_to_keep = ['date'] + list(all_block_series) if all_block_series else list(df.columns)
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            df_filtered = df[columns_to_keep].copy()
            
            target_col = "KOEQUIPTE"
            covariates = [col for col in df_filtered.columns if col != target_col and col != 'date']
            
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
            
            model = DDFM(
                dataset=dataset,
                config=config,
                encoder_size=encoder_size,
                decoder_type="linear",
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
        except Exception as e:
            context = {
                "result_converged": result.converged if 'result' in locals() else None,
            }
            save_failure_report("Step 4: Making predictions", e, context, project_root)
            raise

        try:
            print("\n[Step 5] Saving model...")
            model_path = project_root / "models" / "ddfm_macro.pkl"
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
