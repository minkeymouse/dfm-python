"""Tests for DFM result caching and performance optimizations.

This test suite verifies that:
1. get_result() caches results to avoid expensive recomputation
2. save() uses cached _result when available
3. update() doesn't trigger expensive recomputation when _result exists
4. Checkpoints always contain result to avoid recomputation on load
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import pickle
import time
from unittest.mock import patch, MagicMock

from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.config.constants import DEFAULT_DTYPE


class TestDFMResultCaching:
    """Test suite for DFM result caching optimizations."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return DFMConfig(
            max_iter=5,
            threshold=1e-3,
            blocks={'block1': {'num_factors': 2, 'series': ['series_0', 'series_1']}},
            frequency={'series_0': 'm', 'series_1': 'm'}
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing."""
        np.random.seed(42)
        n_samples = 50  # Small dataset for faster tests
        n_series = 2
        
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='ME')
        data = np.random.randn(n_samples, n_series)
        df = pd.DataFrame(data, index=dates, columns=[f'series_{i}' for i in range(n_series)])
        
        return df
    
    @pytest.fixture
    def trained_model(self, simple_config, sample_data):
        """Create and train a DFM model for testing."""
        dataset = DFMDataset(config=simple_config, data=sample_data)
        model = DFM(config=simple_config)
        
        # Train model (use small max_iter for faster tests)
        X = dataset.get_processed_data()
        model.fit(X=X, dataset=dataset)
        
        return model, dataset
    
    def test_get_result_caches_result(self, trained_model):
        """Test that get_result() caches result to avoid recomputation."""
        model, dataset = trained_model
        
        # Clear _result to simulate first call
        model._result = None
        
        # Mock _compute_smoothed_factors to track calls
        original_compute = model._compute_smoothed_factors
        call_count = {'count': 0}
        
        def tracked_compute(*args, **kwargs):
            call_count['count'] += 1
            return original_compute(*args, **kwargs)
        
        model._compute_smoothed_factors = tracked_compute
        
        # First call should compute
        result1 = model.get_result()
        assert call_count['count'] == 1, "First get_result() should call _compute_smoothed_factors"
        assert model._result is not None, "_result should be cached after first call"
        assert result1 is model._result, "get_result() should return cached _result"
        
        # Second call should use cache
        result2 = model.get_result()
        assert call_count['count'] == 1, "Second get_result() should NOT call _compute_smoothed_factors again"
        assert result2 is result1, "Second call should return same cached result"
        assert result2 is model._result, "Should return cached _result"
    
    def test_save_uses_cached_result(self, trained_model):
        """Test that save() uses cached _result when available."""
        model, dataset = trained_model
        
        # Ensure _result exists
        if model._result is None:
            model._result = model.get_result()
        
        # Mock get_result to track calls
        original_get_result = model.get_result
        call_count = {'count': 0}
        
        def tracked_get_result(*args, **kwargs):
            call_count['count'] += 1
            return original_get_result(*args, **kwargs)
        
        model.get_result = tracked_get_result
        
        # Save should use cached _result, not call get_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pkl"
            model.save(checkpoint_path)
            
            # save() should use _result directly, not call get_result()
            # (unless _result is None, which it isn't in this case)
            assert call_count['count'] == 0, "save() should use cached _result, not call get_result()"
            
            # Verify checkpoint contains result
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            assert 'result' in checkpoint, "Checkpoint should contain result"
            assert checkpoint['result'] is not None, "Checkpoint result should not be None"
    
    def test_save_computes_result_if_not_cached(self, trained_model):
        """Test that save() computes result if _result is None."""
        model, dataset = trained_model
        
        # Clear _result
        model._result = None
        
        # Mock get_result to track calls
        original_get_result = model.get_result
        call_count = {'count': 0}
        
        def tracked_get_result(*args, **kwargs):
            call_count['count'] += 1
            return original_get_result(*args, **kwargs)
        
        model.get_result = tracked_get_result
        
        # Save should call get_result() if _result is None
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pkl"
            model.save(checkpoint_path)
            
            # save() should call get_result() when _result is None
            assert call_count['count'] == 1, "save() should call get_result() when _result is None"
            
            # Verify checkpoint contains result
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            assert 'result' in checkpoint, "Checkpoint should contain result"
            assert checkpoint['result'] is not None, "Checkpoint result should not be None"
    
    def test_update_uses_cached_result(self, trained_model):
        """Test that update() uses cached _result and doesn't recompute."""
        model, dataset = trained_model
        
        # Ensure _result exists
        if model._result is None:
            model._result = model.get_result()
        
        # Mock _compute_smoothed_factors to track calls
        original_compute = model._compute_smoothed_factors
        call_count = {'count': 0}
        
        def tracked_compute(*args, **kwargs):
            call_count['count'] += 1
            return original_compute(*args, **kwargs)
        
        model._compute_smoothed_factors = tracked_compute
        
        # Create new data for update
        new_data = np.random.randn(5, 2).astype(DEFAULT_DTYPE)
        
        # update() should use cached _result, not recompute
        model.update(new_data)
        
        # _ensure_result() should return cached _result without calling _compute_smoothed_factors
        assert call_count['count'] == 0, "update() should use cached _result, not recompute"
        assert model._result is not None, "_result should still exist after update"
        
        # Verify result was extended
        original_length = len(model._result.Z) - 5  # Before update
        assert len(model._result.Z) == original_length + 5, "Result.Z should be extended with new data"
    
    def test_update_recomputes_if_result_none(self, trained_model):
        """Test that update() recomputes if _result is None (but this should be rare)."""
        model, dataset = trained_model
        
        # Clear _result to simulate problematic case
        model._result = None
        
        # Mock _compute_smoothed_factors to track calls
        original_compute = model._compute_smoothed_factors
        call_count = {'count': 0}
        
        def tracked_compute(*args, **kwargs):
            call_count['count'] += 1
            return original_compute(*args, **kwargs)
        
        model._compute_smoothed_factors = tracked_compute
        
        # Create new data for update
        new_data = np.random.randn(5, 2).astype(DEFAULT_DTYPE)
        
        # update() should call _ensure_result() which calls get_result() which calls _compute_smoothed_factors
        model.update(new_data)
        
        # This should trigger recomputation (but only once)
        assert call_count['count'] == 1, "update() should recompute if _result is None"
        assert model._result is not None, "_result should be set after update"
    
    def test_checkpoint_always_contains_result(self, trained_model):
        """Test that checkpoints always contain result after save()."""
        model, dataset = trained_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pkl"
            
            # Save model
            model.save(checkpoint_path)
            
            # Load checkpoint and verify
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            assert 'result' in checkpoint, "Checkpoint must contain 'result' key"
            assert checkpoint['result'] is not None, "Checkpoint result must not be None"
            assert hasattr(checkpoint['result'], 'Z'), "Result must have Z attribute"
            assert checkpoint['result'].Z is not None, "Result.Z must not be None"
    
    def test_load_restores_result(self, trained_model):
        """Test that load() restores _result from checkpoint."""
        model, dataset = trained_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pkl"
            
            # Save model
            model.save(checkpoint_path)
            
            # Clear _result to simulate fresh load
            original_result = model._result
            model._result = None
            
            # Load model
            loaded_model = DFM.load(checkpoint_path)
            
            # Verify _result was restored
            assert loaded_model._result is not None, "Loaded model should have _result"
            assert loaded_model._result.Z.shape == original_result.Z.shape, "Loaded result.Z should match original"
            np.testing.assert_array_equal(
                loaded_model._result.Z, 
                original_result.Z, 
                err_msg="Loaded result.Z should equal original"
            )
    
    def test_update_performance_with_cached_result(self, trained_model):
        """Test that update() is fast when _result is cached."""
        model, dataset = trained_model
        
        # Ensure _result exists
        if model._result is None:
            model._result = model.get_result()
        
        # Create new data for update
        new_data = np.random.randn(1, 2).astype(DEFAULT_DTYPE)  # Small data for fast test
        
        # Measure time for update (should be fast with cached result)
        start_time = time.time()
        model.update(new_data)
        elapsed = time.time() - start_time
        
        # update() with cached result should be fast (< 1 second for small data)
        assert elapsed < 1.0, f"update() with cached result should be fast, took {elapsed:.2f}s"
        
        # Verify result was updated
        assert model._result is not None, "_result should exist after update"
        assert len(model._result.Z) > 0, "Result.Z should have data"
    
    def test_multiple_updates_use_cached_result(self, trained_model):
        """Test that multiple update() calls all use cached result."""
        model, dataset = trained_model
        
        # Ensure _result exists
        if model._result is None:
            model._result = model.get_result()
        
        # Mock _compute_smoothed_factors to track calls
        original_compute = model._compute_smoothed_factors
        call_count = {'count': 0}
        
        def tracked_compute(*args, **kwargs):
            call_count['count'] += 1
            return original_compute(*args, **kwargs)
        
        model._compute_smoothed_factors = tracked_compute
        
        # Perform multiple updates
        for i in range(5):
            new_data = np.random.randn(1, 2).astype(DEFAULT_DTYPE)
            model.update(new_data)
        
        # None of the updates should trigger recomputation
        assert call_count['count'] == 0, "Multiple updates should not trigger recomputation"
        assert model._result is not None, "_result should exist after multiple updates"
        assert len(model._result.Z) >= 5, "Result.Z should be extended with each update"
