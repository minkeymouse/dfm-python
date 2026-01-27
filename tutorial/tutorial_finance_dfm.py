"""Tutorial: DFM for Finance Data

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
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf

def save_failure_report(step: str, error: Exception, context: dict, project_root: Path):
    """Save detailed failure report to file."""
    reports_dir = project_root / "failure_reports"
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"finance_failure_{timestamp}.json"
    
    report = {
        "experiment": "finance",
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

def check_model_state(model, step: str):
    """Check model state for numerical issues and return diagnostics."""
    diagnostics = {}
    
    if model is None:
        return diagnostics
    
    try:
        if hasattr(model, 'training_state') and model.training_state is not None:
            state = model.training_state
            diagnostics['training_state'] = {}
            
            for param_name in ['A', 'C', 'Q', 'R', 'V_0']:
                if hasattr(state, param_name):
                    param = getattr(state, param_name)
                    if param is not None:
                        diagnostics['training_state'][param_name] = {
                            'shape': param.shape if hasattr(param, 'shape') else None,
                            'has_nan': bool(np.any(np.isnan(param))) if isinstance(param, np.ndarray) else None,
                            'has_inf': bool(np.any(np.isinf(param))) if isinstance(param, np.ndarray) else None,
                            'min': float(np.nanmin(param)) if isinstance(param, np.ndarray) else None,
                            'max': float(np.nanmax(param)) if isinstance(param, np.ndarray) else None,
                        }
        
        if hasattr(model, 'result') and model.result is not None:
            result = model.result
            diagnostics['result'] = {
                'converged': result.converged if hasattr(result, 'converged') else None,
                'num_iter': result.num_iter if hasattr(result, 'num_iter') else None,
                'loglik': float(result.loglik) if hasattr(result, 'loglik') and result.loglik is not None else None,
            }
        
        if hasattr(model, 'data_processed') and model.data_processed is not None:
            data = model.data_processed
            diagnostics['data_processed'] = {
                'shape': data.shape if hasattr(data, 'shape') else None,
                'has_nan': bool(np.any(np.isnan(data))) if isinstance(data, np.ndarray) else None,
                'has_inf': bool(np.any(np.isinf(data))) if isinstance(data, np.ndarray) else None,
            }
    except Exception as e:
        diagnostics['diagnostic_error'] = f"Failed to collect diagnostics: {str(e)}"
    
    return diagnostics

@hydra.main(version_base=None, config_path="../config", config_name="dfm_finance")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("DFM Tutorial: Finance Data")
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
            print(f"   Data shape: {df.shape}")
            
            # Truncate to ~1000 timesteps for faster testing (use most recent data)
            # This reduces complexity from O(T × m³) while still providing enough data
            if len(df) > 1000:
                df = df.iloc[-1000:].copy()
                df = df.reset_index(drop=True)
                print(f"   Truncated to {len(df)} timesteps (most recent) for faster testing")

            # Build a daily date index (finance.csv only has date_id=0..T-1)
            if "date" not in df.columns:
                start_date = datetime(1980, 1, 1)
                df["date"] = pd.date_range(start=start_date, periods=len(df), freq="D")
        except Exception as e:
            context = {
                "data_shape": df.shape if df is not None else None,
                "data_columns": list(df.columns) if df is not None else None,
            }
            save_failure_report("Step 1: Loading data", e, context, project_root)
            raise

        try:
            print("\n[Step 2] Creating Dataset...")
            config = DFMConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
            
            # Filter data to only include series specified in blocks (reduces observation dimension)
            # Collect all unique series from all blocks
            all_block_series = set()
            if hasattr(config, 'blocks') and config.blocks:
                for block_name, block_config in config.blocks.items():
                    if isinstance(block_config, dict) and 'series' in block_config:
                        all_block_series.update(block_config['series'])
            
            # Keep only series in blocks + time index + any required metadata columns
            columns_to_keep = ['date'] + list(all_block_series)
            # Filter to only columns that exist in the DataFrame
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            df_filtered = df[columns_to_keep].copy()
            print(f"   Filtered to {len(columns_to_keep) - 1} series (from {len(df.columns) - 1}) based on block configuration")
            
            dataset = DFMDataset(config=config, data=df_filtered, time_index="date")
            obs_dim = len(dataset.variables.columns)
            print(f"   Dataset created: obs_dim={obs_dim}")
        except Exception as e:
            context = {
                "config_max_iter": cfg.get('max_iter', None),
                "config_blocks": {k: len(v.get('series', [])) if isinstance(v, dict) else None 
                                 for k, v in cfg.get('blocks', {}).items()} if hasattr(cfg, 'blocks') else None,
                "data_shape": df.shape if df is not None else None,
                "filtered_series_count": len(columns_to_keep) - 1 if 'columns_to_keep' in locals() else None,
            }
            save_failure_report("Step 2: Creating Dataset", e, context, project_root)
            raise

        try:
            print("\n[Step 3] Training DFM model...")
            model = DFM(dataset=dataset, config=config, scaler=StandardScaler())
            model.fit()

            result = model.result
            print(f"   Converged: {result.converged}, Iterations: {result.num_iter}")
            
            # Check for numerical issues after training
            diagnostics = check_model_state(model, "after_training")
            if diagnostics.get('training_state'):
                for param_name, param_info in diagnostics['training_state'].items():
                    if param_info.get('has_nan') or param_info.get('has_inf'):
                        print(f"   WARNING: {param_name} contains NaN or Inf!")
        except Exception as e:
            context = {
                "config_max_iter": cfg.get('max_iter', None),
                "dataset_obs_dim": dataset.obs_dim if dataset is not None else None,
                "elapsed_time": str(datetime.now() - start_time),
            }
            diagnostics = check_model_state(model, "during_training")
            context.update(diagnostics)
            save_failure_report("Step 3: Training DFM model", e, context, project_root)
            raise

        try:
            print("\n[Step 4] Making predictions...")
            X_forecast, Z_forecast = model.predict(horizon=6)
            print(f"   Forecast shape: {X_forecast.shape}")
            
            # Check forecast for issues
            if np.any(np.isnan(X_forecast)) or np.any(np.isinf(X_forecast)):
                print(f"   WARNING: Forecast contains NaN or Inf!")
        except Exception as e:
            context = {
                "result_converged": result.converged if 'result' in locals() else None,
                "result_iterations": result.num_iter if 'result' in locals() else None,
            }
            diagnostics = check_model_state(model, "during_prediction")
            context.update(diagnostics)
            save_failure_report("Step 4: Making predictions", e, context, project_root)
            raise

        try:
            print("\n[Step 5] Saving model...")
            model_path = project_root / "models" / "dfm_finance.pkl"
            model_path.parent.mkdir(exist_ok=True)
            model.save(model_path)
            print(f"   Model saved to: {model_path}")
        except Exception as e:
            context = {
                "model_path": str(model_path) if 'model_path' in locals() else None,
                "result_converged": result.converged if 'result' in locals() else None,
            }
            diagnostics = check_model_state(model, "during_saving")
            context.update(diagnostics)
            save_failure_report("Step 5: Saving model", e, context, project_root)
            raise

        print("\n" + result.summary())
        
        elapsed = datetime.now() - start_time
        print(f"\n[SUCCESS] Tutorial completed in {elapsed}")
        
    except Exception as e:
        elapsed = datetime.now() - start_time
        print(f"\n[FAILURE] Tutorial failed after {elapsed}")
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()
