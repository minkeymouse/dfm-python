#!/usr/bin/env python3
"""Manual test script to verify update() method works correctly."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dfm_python.models.dfm import DFM
    from dfm_python.config import DFMConfig, SeriesConfig
    from dfm_python.lightning import DFMDataModule, DFMTrainer
    import pandas as pd
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_update_method():
    """Test that update() method works correctly."""
    print("\n=== Testing update() method ===\n")
    
    # Create simple config
    config = DFMConfig(
        series=[
            SeriesConfig(series_id='series1', frequency='m', transformation='lin', blocks=[1]),
            SeriesConfig(series_id='series2', frequency='m', transformation='lin', blocks=[1])
        ],
        blocks={'block1': {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
    )
    
    # Create simple data
    np.random.seed(42)
    T = 50
    N = 2
    data = np.random.randn(T, N)
    df = pd.DataFrame(data, columns=['series1', 'series2'])
    
    # Create model
    model = DFM()
    model.load_config(config)
    
    # Create data module
    data_module = DFMDataModule(config=config, data=df)
    data_module.setup()
    
    # Train model
    print("1. Training model...")
    trainer = DFMTrainer(max_epochs=10)
    trainer.fit(model, data_module)
    print("   ✓ Model trained successfully")
    
    # Test update method exists
    print("\n2. Checking update() method exists...")
    assert hasattr(model, 'update'), "update() method not found"
    assert callable(getattr(model, 'update', None)), "update() is not callable"
    print("   ✓ update() method exists and is callable")
    
    # Test update with standardized data
    print("\n3. Testing update() with new data...")
    result = model.result
    Mx = result.Mx
    Wx = result.Wx
    
    # Create new standardized data
    new_data = np.random.randn(5, N)
    new_data_std = (new_data - Mx) / Wx
    
    # Update model state
    model_updated = model.update(new_data_std)
    print("   ✓ update() executed without errors")
    
    # Check method chaining
    assert model_updated is model, "update() should return self for chaining"
    print("   ✓ update() returns self for method chaining")
    
    # Test predict after update
    print("\n4. Testing predict() after update()...")
    forecast = model.predict(horizon=3)
    assert forecast is not None, "predict() returned None"
    assert np.isfinite(forecast).all(), "predict() contains NaN/Inf"
    print(f"   ✓ predict() works after update(), forecast shape: {forecast.shape}")
    
    # Test update with different shapes
    print("\n5. Testing update() with different data shapes...")
    for shape in [(1, N), (10, N), (20, N)]:
        test_data = np.random.randn(*shape)
        test_data_std = (test_data - Mx) / Wx
        model.update(test_data_std)
        forecast = model.predict(horizon=1)
        assert np.isfinite(forecast).all(), f"Failed with shape {shape}"
    print("   ✓ update() works with different data shapes")
    
    # Test error handling
    print("\n6. Testing error handling...")
    try:
        model.update(np.random.randn(10))  # 1D array should fail
        print("   ✗ Should have raised ValueError for 1D array")
        return False
    except ValueError as e:
        if "2D array" in str(e):
            print("   ✓ Correctly raises ValueError for 1D array")
        else:
            print(f"   ✗ Wrong error message: {e}")
            return False
    
    # Test update before training
    print("\n7. Testing update() before training...")
    untrained_model = DFM()
    untrained_model.load_config(config)
    try:
        untrained_model.update(np.random.randn(5, N))
        print("   ✗ Should have raised ValueError for untrained model")
        return False
    except ValueError as e:
        if "not been trained" in str(e):
            print("   ✓ Correctly raises ValueError for untrained model")
        else:
            print(f"   ✗ Wrong error message: {e}")
            return False
    
    print("\n=== All tests passed! ===\n")
    return True

if __name__ == '__main__':
    try:
        success = test_update_method()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

