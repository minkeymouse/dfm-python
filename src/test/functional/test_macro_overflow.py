"""Test to investigate macro tutorial overflow issue.

This test reproduces the macro tutorial failure and pinpoints exactly where
the overflow occurs in the Kalman filter prediction step.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.functional.em import run_em_algorithm, EMConfig
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig, OmegaConf
import yaml


class TestMacroOverflow:
    """Test to investigate macro tutorial overflow issue."""
    
    @pytest.fixture
    def macro_data(self):
        """Load macro data from tutorial."""
        data_path = project_root / "data" / "macro.csv"
        if not data_path.exists():
            pytest.skip(f"Macro data not found at {data_path}")
        
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    @pytest.fixture
    def macro_config(self):
        """Load macro config from tutorial."""
        config_path = project_root / "config" / "dfm_macro.yaml"
        if not config_path.exists():
            pytest.skip(f"Macro config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        return DFMConfig.from_dict(cfg_dict)
    
    def test_macro_tutorial_reproduction(self, macro_data, macro_config):
        """Reproduce the macro tutorial failure and identify overflow location."""
        print("\n" + "="*80)
        print("Testing Macro Tutorial Overflow Issue")
        print("="*80)
        
        # Create dataset
        dataset = DFMDataset(config=macro_config, data=macro_data, time_index='date')
        
        # Create model
        model = DFM(dataset=dataset, config=macro_config, scaler=StandardScaler())
        
        # Run fit() which will initialize and run EM
        print("\n[Step 1] Running model.fit() (initialization + EM)...")
        try:
            final_state = model.fit()
            metadata = model.result.__dict__ if hasattr(model, 'result') else {}
            
            # Get processed data after fit
            X_processed = model.data_processed
            print(f"\nData shape: {X_processed.shape}")
            print(f"Data stats: mean={np.nanmean(X_processed):.4f}, std={np.nanstd(X_processed):.4f}")
            print(f"Data range: [{np.nanmin(X_processed):.4f}, {np.nanmax(X_processed):.4f}]")
            print(f"✓ EM converged: {metadata.get('converged', False)}")
            print(f"  Iterations: {metadata.get('num_iter', 0)}")
            print(f"  Final loglik: {metadata.get('loglik', 'N/A')}")
            
            # Check parameter magnitudes
            print("\n[Step 2] Checking parameter magnitudes...")
            A = final_state.A
            Q = final_state.Q
            V_0 = final_state.V_0
            C = final_state.C
            R = final_state.R
            
            print(f"  A: shape={A.shape}, max(abs)={np.max(np.abs(A)):.2e}")
            try:
                A_eig = np.linalg.eigvals(A)
                print(f"    max(eig)={np.max(np.abs(A_eig)):.2e}")
            except:
                print(f"    max(eig)=N/A (eigenvalue computation failed)")
            
            print(f"  Q: shape={Q.shape}, max(abs)={np.max(np.abs(Q)):.2e}")
            try:
                Q_eig = np.linalg.eigvalsh(Q)
                print(f"    max(eig)={np.max(np.abs(Q_eig)):.2e}, min(eig)={np.min(np.abs(Q_eig)):.2e}")
            except:
                print(f"    max(eig)=N/A (eigenvalue computation failed)")
            
            print(f"  V_0: shape={V_0.shape}, max(abs)={np.max(np.abs(V_0)):.2e}")
            try:
                V_0_eig = np.linalg.eigvalsh(V_0)
                print(f"    max(eig)={np.max(np.abs(V_0_eig)):.2e}, min(eig)={np.min(np.abs(V_0_eig)):.2e}")
            except:
                print(f"    max(eig)=N/A (eigenvalue computation failed)")
            print(f"  C: shape={C.shape}, max(abs)={np.max(np.abs(C)):.2e}")
            print(f"  R: shape={R.shape}, max(abs)={np.max(np.abs(R)):.2e}, max(eig)={np.max(np.abs(np.linalg.eigvalsh(R))):.2e}")
            
            # Check for problematic values
            if np.max(np.abs(V_0)) > 1e6:
                print(f"  ⚠ WARNING: V_0 has very large values: max={np.max(np.abs(V_0)):.2e}")
            if np.max(np.abs(Q)) > 1e6:
                print(f"  ⚠ WARNING: Q has very large values: max={np.max(np.abs(Q)):.2e}")
            if np.max(np.abs(A)) > 10:
                print(f"  ⚠ WARNING: A has large values: max={np.max(np.abs(A)):.2e}")
            
            # [Step 3] Try to run filter_and_smooth again (this is where overflow happens)
            print("\n[Step 3] Running filter_and_smooth with final parameters (reproducing overflow)...")
            
            # Create Kalman filter with final parameters
            kf = DFMKalmanFilter(
                transition_matrices=A,
                observation_matrices=C,
                transition_covariance=Q,
                observation_covariance=R,
                initial_state_mean=final_state.Z_0,
                initial_state_covariance=V_0
            )
            
            # Mask invalid values
            X_masked = np.ma.masked_invalid(X_processed)
            
            # Try to run filter - this is where overflow occurs
            print("  Attempting filter step...")
            try:
                # Check predicted covariance computation manually
                print("  Computing predicted covariance: A @ V_0 @ A.T + Q")
                
                # Check V_0 eigenvalues
                V_0_eigvals = np.linalg.eigvalsh(V_0)
                print(f"    V_0 eigenvalues: min={np.min(V_0_eigvals):.2e}, max={np.max(V_0_eigvals):.2e}")
                
                # Check A eigenvalues
                A_eigvals = np.linalg.eigvals(A)
                print(f"    A eigenvalues: min={np.min(np.abs(A_eigvals)):.2e}, max={np.max(np.abs(A_eigvals)):.2e}")
                
                # Compute A @ V_0 @ A.T
                print("    Computing A @ V_0 @ A.T...")
                AV0AT = A @ V_0 @ A.T
                print(f"    A @ V_0 @ A.T: max(abs)={np.max(np.abs(AV0AT)):.2e}")
                
                if np.any(~np.isfinite(AV0AT)):
                    print(f"    ⚠ ERROR: A @ V_0 @ A.T contains non-finite values!")
                    print(f"      Inf count: {np.sum(np.isinf(AV0AT))}")
                    print(f"      NaN count: {np.sum(np.isnan(AV0AT))}")
                    raise ValueError("A @ V_0 @ A.T contains non-finite values")
                
                # Add Q
                print("    Adding Q...")
                P_pred = AV0AT + Q
                print(f"    P_pred = A @ V_0 @ A.T + Q: max(abs)={np.max(np.abs(P_pred)):.2e}")
                
                if np.any(~np.isfinite(P_pred)):
                    print(f"    ⚠ ERROR: P_pred contains non-finite values!")
                    print(f"      Inf count: {np.sum(np.isinf(P_pred))}")
                    print(f"      NaN count: {np.sum(np.isnan(P_pred))}")
                    raise ValueError("P_pred contains non-finite values")
                
                # Check condition number
                P_pred_eigvals = np.linalg.eigvalsh(P_pred)
                cond_num = np.max(P_pred_eigvals) / (np.min(P_pred_eigvals) + 1e-10)
                print(f"    P_pred condition number: {cond_num:.2e}")
                
                if cond_num > 1e10:
                    print(f"    ⚠ WARNING: P_pred is ill-conditioned (cond={cond_num:.2e})")
                
                # Now try actual filter
                print("  Running actual filter...")
                filtered_means, filtered_covs = kf.filter(X_masked)
                print("  ✓ Filter completed successfully")
                
                # Try smoother
                print("  Running smoother...")
                smoothed_means, smoothed_covs, cross_covs, loglik = kf.filter_and_smooth(X_masked)
                print("  ✓ Filter and smooth completed successfully")
                
            except ValueError as e:
                if "array must not contain infs or NaNs" in str(e):
                    print(f"  ✗ OVERFLOW DETECTED: {e}")
                    print("\n  Overflow location identified:")
                    print("    - Error occurs in pykalman's _filter_correct function")
                    print("    - Specifically when computing predicted_observation_covariance")
                    print("    - Which requires: C @ P_pred @ C.T + R")
                    print("    - Where P_pred = A @ V_0 @ A.T + Q")
                    raise
                else:
                    raise
            except Exception as e:
                print(f"  ✗ ERROR: {type(e).__name__}: {e}")
                raise
            
        except Exception as e:
            print(f"\n✗ Test failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_parameter_magnitudes_after_em(self, macro_data, macro_config):
        """Test to check parameter magnitudes after EM convergence."""
        dataset = DFMDataset(config=macro_config, data=macro_data, time_index='date')
        model = DFM(dataset=dataset, config=macro_config, scaler=StandardScaler())
        X_processed = model.data_processed
        
        # Run EM
        final_state, metadata = run_em_algorithm(
            X_processed,
            model._initial_state,
            max_iter=10,
            threshold=1e-5,
            config=EMConfig()
        )
        
        # Check magnitudes
        A = final_state.A
        Q = final_state.Q
        V_0 = final_state.V_0
        
        # Compute predicted covariance for first timestep
        P_pred = A @ V_0 @ A.T + Q
        
        # Check for overflow conditions
        assert np.all(np.isfinite(P_pred)), "P_pred contains non-finite values"
        assert np.max(np.abs(P_pred)) < 1e10, f"P_pred too large: max={np.max(np.abs(P_pred)):.2e}"
        
        # Check condition number
        P_pred_eigvals = np.linalg.eigvalsh(P_pred)
        cond_num = np.max(P_pred_eigvals) / (np.min(P_pred_eigvals) + 1e-10)
        assert cond_num < 1e12, f"P_pred ill-conditioned: cond={cond_num:.2e}"
