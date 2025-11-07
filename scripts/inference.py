#!/usr/bin/env python3
"""
Inference script for Dynamic Factor Model (DFM).

This script loads a trained DFM model and performs inference,
including factor extraction and forecasting.

Usage:
    python scripts/inference.py --model outputs/ResDFM.pkl --data data/example_macro_data.csv --forecast-horizon 12
"""

import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dfm_python import load_config, load_data, dfm, DFMResult, calculate_rmse

def forecast_factors(result: DFMResult, horizon: int):
    """Forecast factors forward using the transition equation."""
    A = result.A
    Z_last = result.Z[-1, :]  # Last observed factor values
    
    # Forecast factors forward
    Z_forecast = np.zeros((horizon, Z_last.shape[0]))
    Z_forecast[0, :] = A @ Z_last
    
    for h in range(1, horizon):
        Z_forecast[h, :] = A @ Z_forecast[h-1, :]
    
    return Z_forecast

def forecast_series(result: DFMResult, horizon: int):
    """Forecast observed series using factor forecasts."""
    Z_forecast = forecast_factors(result, horizon)
    
    # Project factors to series space
    X_forecast = Z_forecast @ result.C.T
    
    # Unstandardize
    X_forecast_unstd = X_forecast * result.Wx + result.Mx
    
    return X_forecast_unstd, Z_forecast

def main():
    parser = argparse.ArgumentParser(description='DFM Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (pickle file)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to new data file for updating (optional)')
    parser.add_argument('--forecast-horizon', type=int, default=12,
                       help='Forecast horizon in periods (default: 12)')
    parser.add_argument('--output', type=str, default='outputs/forecasts.csv',
                       help='Output path for forecasts (default: outputs/forecasts.csv)')
    parser.add_argument('--factors-only', action='store_true',
                       help='Only output factor forecasts')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DFM INFERENCE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Forecast horizon: {args.forecast_horizon} periods")
    print("="*70 + "\n")
    
    # Load trained model
    print("Loading trained model...")
    try:
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        
        result = model_data['result']
        config = model_data['config']
        Time = model_data['Time']
        
        print(f"✓ Model loaded")
        print(f"  Training period: {Time[0]} to {Time[-1]}")
        print(f"  Number of factors: {result.Z.shape[1]}")
        print(f"  Number of series: {result.C.shape[0]}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # If new data provided, re-estimate
    if args.data:
        print(f"\nUpdating model with new data: {args.data}")
        try:
            X_new, Time_new, Z_new = load_data(args.data, config)
            result_new = dfm(X_new, config, threshold=1e-4)
            result = result_new
            Time = Time_new
            print(f"✓ Model updated with new data")
        except Exception as e:
            print(f"✗ Failed to update model: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Generate forecasts
    print(f"\nGenerating {args.forecast_horizon}-period forecasts...")
    try:
        X_forecast, Z_forecast = forecast_series(result, args.forecast_horizon)
        print(f"✓ Forecasts generated")
        
        # Calculate RMSE if we have actual values to compare against
        # This happens when new data is provided and we can compare forecasts to actuals
        forecast_rmse = None
        forecast_rmse_per_series = None
        
        if args.data:
            # Try to compare forecasts to actual values if available
            try:
                # Get the last forecast_horizon periods from the updated data
                if X_new.shape[0] >= args.forecast_horizon:
                    # Compare last forecast_horizon periods of actual data to forecasts
                    X_actual = X_new[-args.forecast_horizon:, :]
                    # Only calculate RMSE where we have actual values
                    forecast_rmse, forecast_rmse_per_series = calculate_rmse(X_actual, X_forecast, mask=None)
                    print(f"✓ Forecast RMSE calculated: {forecast_rmse:.6f}")
            except Exception as e:
                # If comparison fails, continue without RMSE
                print(f"  Note: Could not calculate forecast RMSE: {e}")
    except Exception as e:
        print(f"✗ Failed to generate forecasts: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create forecast time index
    last_date = Time[-1]
    if pd.infer_freq(Time) == 'M':  # Monthly
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=args.forecast_horizon,
            freq='M'
        )
    elif pd.infer_freq(Time) == 'Q':  # Quarterly
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=args.forecast_horizon,
            freq='Q'
        )
    else:
        # Default to monthly
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=args.forecast_horizon,
            freq='M'
        )
    
    # Save forecasts
    print(f"\nSaving forecasts to {args.output}...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.factors_only:
            # Save only factors
            forecast_df = pd.DataFrame(
                Z_forecast,
                index=forecast_dates,
                columns=[f'Factor_{i+1}' for i in range(Z_forecast.shape[1])]
            )
        else:
            # Save series forecasts
            series_names = [s.series_name for s in config.series]
            forecast_df = pd.DataFrame(
                X_forecast,
                index=forecast_dates,
                columns=series_names
            )
        
        forecast_df.to_csv(args.output)
        print(f"✓ Forecasts saved to {args.output}")
        
        # Print summary
        print("\nForecast Summary:")
        print("-"*70)
        if args.factors_only:
            print(f"Factor forecasts for {args.forecast_horizon} periods:")
            print(forecast_df.head())
        else:
            print(f"Series forecasts for {args.forecast_horizon} periods:")
            print(forecast_df.head())
            
            # Display RMSE if calculated
            if forecast_rmse is not None and not np.isnan(forecast_rmse):
                print("\nForecast Accuracy (RMSE):")
                print("-"*70)
                print(f"  Overall RMSE (averaged across all series): {forecast_rmse:.6f}")
                if forecast_rmse_per_series is not None and len(forecast_rmse_per_series) > 0:
                    print("\n  RMSE per Series:")
                    try:
                        series_names = [s.series_name for s in config.series]
                        for i, (name, rmse_val) in enumerate(zip(series_names, forecast_rmse_per_series)):
                            if not np.isnan(rmse_val):
                                print(f"    {name:40s}: {rmse_val:.6f}")
                    except Exception:
                        # Fallback if series names not available
                        for i, rmse_val in enumerate(forecast_rmse_per_series):
                            if not np.isnan(rmse_val):
                                print(f"    Series {i:3d}: {rmse_val:.6f}")
        
    except Exception as e:
        print(f"✗ Failed to save forecasts: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

