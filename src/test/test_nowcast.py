"""Tests for Nowcasting functionality.

This module provides comprehensive tests for nowcasting functionality in the dfm-python
package. Nowcasting estimates current-period values (e.g., current-quarter GDP) using
high-frequency indicators before official data is released.

**Test Organization**:
- `TestNowcast`: Tests nowcasting functionality (4 tests)
  - Basic nowcast calculation: Compute nowcast values for target series
  - Data view creation: Get data availability at specific dates
  - News decomposition: Decompose forecast updates into contributions from data releases
  - Error handling: Invalid dates, missing data, edge cases

**Dependencies**:
- Required: `sktime` (for preprocessing via DFMScaler)
- Requires: Trained DFM model (created via helper function `create_trained_dfm_model()`)
- Tests are skipped if sktime is not available

**Usage Patterns**:
- Create and train a DFM model (using helper function or manually)
- Access nowcast manager: `nowcast = model.nowcast`
- Calculate nowcast: `value = nowcast(target_series, view_date, target_period)`
- Get data view: `X_view, Time_view, Z_view = nowcast.get_data_view(view_date)`
- Decompose news: `news = nowcast.decompose(target_series, target_period, view_date_old, view_date_new)`

**Key Features Tested**:
- Nowcast calculation with confidence intervals
- Data availability views (jagged edge handling)
- News decomposition (attribution of forecast changes to data releases)
- Error handling for invalid inputs

**Related Test Files**:
- `test_dfm.py`: Tests DFM model creation and training (required for nowcasting)
- `test_transformations.py`: Tests preprocessing used by DFM models
"""

# Standard library imports
from datetime import datetime

import pytest

# Third-party imports
import numpy as np

# Local application imports
from dfm_python.models import DFM
from dfm_python.nowcast import NowcastResult
from dfm_python.nowcast.helpers import NewsDecompResult
from dfm_python.utils.time import TimeIndex, get_latest_time, to_python_datetime

# Local relative imports
from . import create_trained_dfm_model


# Skip all tests if sktime is not available (required for preprocessing)
pytest.importorskip("sktime", reason="sktime is required for nowcast tests")


# ============================================================================
# Nowcast Tests
# ============================================================================

class TestNowcast:
    """Test nowcasting functionality."""
    
    def test_nowcast_basic(self):
        """Test basic nowcast calculation.
        
        This test verifies that:
        1. A trained DFM model can access the nowcast property
        2. The nowcast can be called with a target series and view date
        3. The returned value is a float (when return_result=False)
        4. The value is reasonable (not NaN, not infinite)
        
        Expected behavior:
        - Nowcast property is accessible for trained model
        - Nowcast call returns float value when return_result=False
        - Nowcast call returns NowcastResult object when return_result=True
        - Returned values are finite and reasonable (not NaN, not infinite)
        - Result object has correct structure and values match direct call
        """
        # Create trained DFM model
        model = create_trained_dfm_model(num_series=5, num_factors=1, n_periods=50)
        
        # Access nowcast property (should succeed for trained model)
        nowcast = model.nowcast
        assert nowcast is not None, "Nowcast property should not be None"
        
        # Use None for view_date to default to latest available time
        # This is safer than hardcoding a date that might not align with the time index
        view_date = None  # Will default to get_latest_time(model.time)
        
        # Calculate nowcast for first series
        target_series = 'series_0'
        
        # Test 1: Get nowcast value (float)
        nowcast_value = nowcast(target_series, view_date=view_date, return_result=False)
        
        # Verify return type
        assert isinstance(nowcast_value, (float, np.floating)), \
            f"Nowcast value should be float, got {type(nowcast_value)}"
        
        # Verify value is reasonable
        assert not np.isnan(nowcast_value), "Nowcast value should not be NaN"
        assert not np.isinf(nowcast_value), "Nowcast value should not be infinite"
        assert np.isfinite(nowcast_value), "Nowcast value should be finite"
        
        # Test 2: Get full result object
        result = nowcast(target_series, view_date=view_date, return_result=True)
        
        # Verify return type
        assert isinstance(result, NowcastResult), \
            f"Nowcast result should be NowcastResult, got {type(result)}"
        
        # Verify result structure
        assert hasattr(result, 'nowcast_value'), "Result should have nowcast_value attribute"
        assert hasattr(result, 'target_series'), "Result should have target_series attribute"
        assert hasattr(result, 'view_date'), "Result should have view_date attribute"
        
        # Verify result values
        assert result.target_series == target_series, \
            f"Target series should match, got {result.target_series}"
        assert result.nowcast_value == nowcast_value, \
            "Result nowcast_value should match direct call value"
        assert not np.isnan(result.nowcast_value), "Result nowcast_value should not be NaN"
        assert not np.isinf(result.nowcast_value), "Result nowcast_value should not be infinite"
    
    def test_nowcast_get_data_view(self):
        """Test data view creation via get_data_view() method.
        
        This test verifies that:
        1. A trained DFM model can create data views at specific dates
        2. The returned tuple has correct types and shapes
        3. Data view caching works correctly
        4. Time index is valid and aligned with data
        
        Expected behavior:
        - Data view can be created at specific dates
        - Returned tuple has correct types (np.ndarray, TimeIndex, optional np.ndarray)
        - Data view shapes are correct (2D array with correct dimensions)
        - Time index aligns with data dimensions
        - Caching works correctly (same date returns cached results)
        - String date parsing works correctly
        - Different view dates produce valid results
        """
        # Create trained DFM model
        model = create_trained_dfm_model(num_series=5, num_factors=1, n_periods=50)
        
        # Access nowcast property
        nowcast = model.nowcast
        assert nowcast is not None, "Nowcast property should not be None"
        
        # Get a valid view date from the model's time index
        # Use latest time to ensure we have a valid date
        view_date = get_latest_time(model.time)
        
        # Test 1: Get data view (basic functionality)
        X_view, Time_view, Z_view = nowcast.get_data_view(view_date=view_date)
        
        # Verify return types
        assert isinstance(X_view, np.ndarray), \
            f"X_view should be np.ndarray, got {type(X_view)}"
        assert isinstance(Time_view, TimeIndex), \
            f"Time_view should be TimeIndex, got {type(Time_view)}"
        # Z_view can be None or np.ndarray
        if Z_view is not None:
            assert isinstance(Z_view, np.ndarray), \
                f"Z_view should be np.ndarray or None, got {type(Z_view)}"
        
        # Verify data shapes
        # X_view should have shape (T, N) where T is time periods, N is number of series
        assert X_view.ndim == 2, f"X_view should be 2D array, got shape {X_view.shape}"
        assert X_view.shape[1] == 5, \
            f"X_view should have 5 series (columns), got {X_view.shape[1]}"
        assert X_view.shape[0] > 0, \
            f"X_view should have at least 1 time period, got {X_view.shape[0]}"
        
        # Verify time index alignment
        assert len(Time_view) == X_view.shape[0], \
            f"Time_view length ({len(Time_view)}) should match X_view time dimension ({X_view.shape[0]})"
        
        # Verify data values are reasonable (not all NaN, not all infinite)
        assert not np.all(np.isnan(X_view)), "X_view should not be all NaN"
        assert not np.all(np.isinf(X_view)), "X_view should not be all infinite"
        
        # Test 2: Test caching behavior
        # Call get_data_view again with the same date - should use cache
        X_view2, Time_view2, Z_view2 = nowcast.get_data_view(view_date=view_date)
        
        # Verify same results (cached)
        try:
            np.testing.assert_array_equal(X_view, X_view2)
        except AssertionError as e:
            pytest.fail(f"Cached X_view should match original X_view. "
                       f"Original shape: {X_view.shape}, Cached shape: {X_view2.shape}. "
                       f"Error: {e}")
        # Time_view should be the same object or equal
        assert len(Time_view) == len(Time_view2), \
            f"Cached Time_view length ({len(Time_view2)}) should match original Time_view length ({len(Time_view)})"
        
        # Test 3: Test with string date (should parse correctly)
        # Convert view_date to ISO format string for reliable parsing
        # ISO format is more reliable than str() which may include microseconds
        view_date_str = view_date.isoformat()
        X_view3, Time_view3, Z_view3 = nowcast.get_data_view(view_date=view_date_str)
        
        # Verify results are consistent
        assert X_view3.shape == X_view.shape, \
            f"String date X_view shape ({X_view3.shape}) should match datetime date shape ({X_view.shape})"
        
        # Test 4: Test with different view date (earlier in data)
        # Get an earlier date from the time index to test different view dates
        if len(model.time) > 5:
            # Use an earlier time point (e.g., 10 periods before the end)
            early_idx = max(0, len(model.time) - 10)
            early_time = model.time[early_idx]
            
            # Convert to datetime - TimeIndex.__getitem__() typically returns datetime,
            # but to_python_datetime() handles edge cases (polars datetime, etc.)
            if isinstance(early_time, datetime):
                early_date = early_time
            else:
                early_date = to_python_datetime(early_time)
            
            X_view_early, Time_view_early, Z_view_early = nowcast.get_data_view(view_date=early_date)
            
            # Verify different view date gives valid results
            assert X_view_early.shape[1] == X_view.shape[1], \
                f"Different view dates should have same number of series, got {X_view_early.shape[1]} vs {X_view.shape[1]}"
            # Time dimension should be valid
            assert X_view_early.shape[0] > 0, \
                f"Early-date view should have at least 1 time period, got {X_view_early.shape[0]}"
            assert len(Time_view_early) == X_view_early.shape[0], \
                f"Time_view_early length ({len(Time_view_early)}) should align with X_view_early time dimension ({X_view_early.shape[0]})"
    
    def test_nowcast_get_data_view_invalid_date(self):
        """Test error handling for invalid view dates.
        
        This test verifies that:
        1. Invalid view dates raise appropriate errors
        2. Error messages are informative
        
        Expected behavior:
        - Invalid date strings raise appropriate errors or handle gracefully
        - Error messages are informative (mention date/time/parse)
        - Future dates return valid data views (may be empty or filtered)
        - Data view maintains correct series dimension even with edge case dates
        """
        # Create trained DFM model
        model = create_trained_dfm_model(num_series=5, num_factors=1, n_periods=50)
        
        # Access nowcast property
        nowcast = model.nowcast
        assert nowcast is not None, "Nowcast property should not be None"
        
        # Test 1: Invalid date string (should raise error or handle gracefully)
        # The exact behavior depends on parse_timestamp() implementation
        # It may raise ValueError or return a parsed date
        invalid_date_str = "invalid-date-string"
        
        # get_data_view() will call create_data_view() which uses parse_timestamp()
        # This may raise ValueError or handle it differently
        # We test that it doesn't crash silently
        try:
            X_view, Time_view, Z_view = nowcast.get_data_view(view_date=invalid_date_str)
            # If it doesn't raise an error, at least verify the result is valid
            assert X_view is not None, "Data view should not be None even with invalid date string"
        except (ValueError, TypeError) as e:
            # Expected behavior - invalid date should raise an error
            error_msg_lower = str(e).lower()
            assert "date" in error_msg_lower or "time" in error_msg_lower or "parse" in error_msg_lower, \
                f"Error message should mention date/time/parse for invalid date string '{invalid_date_str}', got: {e}"
        
        # Test 2: Date far in the future (should handle gracefully)
        # This should work but may return empty or filtered data view
        future_date = datetime(2100, 1, 1)
        X_view_future, Time_view_future, Z_view_future = nowcast.get_data_view(view_date=future_date)
        
        # Verify result is valid (may be empty or filtered)
        assert X_view_future is not None, "Future date should return valid data view"
        assert isinstance(X_view_future, np.ndarray), \
            f"Future date data view should be np.ndarray, got {type(X_view_future)}"
        assert X_view_future.shape[1] == 5, \
            f"Future date should maintain series dimension (5), got {X_view_future.shape[1]}"
    
    def test_nowcast_decompose(self):
        """Test news decomposition functionality.
        
        This test verifies that:
        1. A trained DFM model can decompose forecast updates into news contributions
        2. The decompose() method returns NewsDecompResult with correct structure
        3. Contributions are correctly attributed to data series
        4. The change equals y_new - y_old
        5. All result attributes are valid and properly structured
        
        Expected behavior:
        - News decomposition returns NewsDecompResult with correct structure
        - All required attributes exist (y_old, y_new, change, singlenews, etc.)
        - Change equals y_new - y_old (within numerical precision)
        - Contributions are correctly attributed to data series
        - Dictionary format (return_dict=True) matches NewsDecompResult format
        - Same view dates produce minimal change (near zero)
        """
        # Create trained DFM model
        model = create_trained_dfm_model(num_series=5, num_factors=1, n_periods=50)
        
        # Access nowcast property
        nowcast = model.nowcast
        assert nowcast is not None, "Nowcast property should not be None"
        
        # Get time index for creating view dates
        # We need two view dates: old (earlier) and new (later)
        # Use different time points to ensure there's a meaningful difference
        if len(model.time) < 10:
            # If not enough data, skip this test or use what we have
            pytest.skip("Not enough time periods for news decomposition test")
        
        # Use an earlier date for old view (e.g., 15 periods before the end)
        old_idx = max(0, len(model.time) - 15)
        old_time = model.time[old_idx]
        if isinstance(old_time, datetime):
            view_date_old = old_time
        else:
            view_date_old = to_python_datetime(old_time)
        
        # Use a later date for new view (e.g., 5 periods before the end)
        new_idx = max(old_idx + 1, len(model.time) - 5)
        new_time = model.time[new_idx]
        if isinstance(new_time, datetime):
            view_date_new = new_time
        else:
            view_date_new = to_python_datetime(new_time)
        
        # Verify view dates are in correct order
        assert view_date_new > view_date_old, \
            f"view_date_new ({view_date_new}) should be after view_date_old ({view_date_old})"
        
        # Get data view first to access Time_new (required for target_period selection)
        # Target period must exist in the view's time index, not just the full model time
        X_new, Time_new, _ = nowcast.get_data_view(view_date_new)
        
        # Select target_period from the view's time index (not full model time)
        # This ensures target_period exists in Time_new, which is required by decompose()
        target_time = get_latest_time(Time_new)
        if isinstance(target_time, datetime):
            target_period = target_time
        else:
            target_period = to_python_datetime(target_time)
        
        # Target series for nowcast
        target_series = 'series_0'
        
        # Test 1: Basic news decomposition (return NewsDecompResult)
        # Wrap in try-except for better error messages
        try:
            result = nowcast.decompose(
                target_series=target_series,
                target_period=target_period,
                view_date_old=view_date_old,
                view_date_new=view_date_new,
                return_dict=False
            )
        except ValueError as e:
            # Provide context about what failed for easier debugging
            pytest.fail(f"News decomposition failed: {e}. "
                       f"Target period: {target_period}, "
                       f"View dates: old={view_date_old}, new={view_date_new}, "
                       f"Time_new length: {len(Time_new)}")
        
        # Verify return type
        assert isinstance(result, NewsDecompResult), \
            f"Result should be NewsDecompResult, got {type(result)}"
        
        # Verify all required attributes exist
        assert hasattr(result, 'y_old'), "Result should have y_old attribute"
        assert hasattr(result, 'y_new'), "Result should have y_new attribute"
        assert hasattr(result, 'change'), "Result should have change attribute"
        assert hasattr(result, 'singlenews'), "Result should have singlenews attribute"
        assert hasattr(result, 'top_contributors'), "Result should have top_contributors attribute"
        assert hasattr(result, 'actual'), "Result should have actual attribute"
        assert hasattr(result, 'forecast'), "Result should have forecast attribute"
        assert hasattr(result, 'weight'), "Result should have weight attribute"
        assert hasattr(result, 't_miss'), "Result should have t_miss attribute"
        assert hasattr(result, 'v_miss'), "Result should have v_miss attribute"
        assert hasattr(result, 'innov'), "Result should have innov attribute"
        
        # Verify basic types and values
        assert isinstance(result.y_old, (float, np.floating)), \
            f"y_old should be float, got {type(result.y_old)}"
        assert isinstance(result.y_new, (float, np.floating)), \
            f"y_new should be float, got {type(result.y_new)}"
        assert isinstance(result.change, (float, np.floating)), \
            f"change should be float, got {type(result.change)}"
        
        # Verify values are finite
        assert not np.isnan(result.y_old), "y_old should not be NaN"
        assert not np.isnan(result.y_new), "y_new should not be NaN"
        assert not np.isnan(result.change), "change should not be NaN"
        assert not np.isinf(result.y_old), "y_old should not be infinite"
        assert not np.isinf(result.y_new), "y_new should not be infinite"
        assert not np.isinf(result.change), "change should not be infinite"
        
        # Verify change equals y_new - y_old (within numerical precision)
        assert abs(result.change - (result.y_new - result.y_old)) < 1e-10, \
            f"change should equal y_new - y_old, got change={result.change}, y_new-y_old={result.y_new - result.y_old}"
        
        # Verify singlenews is numpy array
        assert isinstance(result.singlenews, np.ndarray), \
            f"singlenews should be np.ndarray, got {type(result.singlenews)}"
        assert result.singlenews.ndim >= 1, \
            f"singlenews should be at least 1D, got shape {result.singlenews.shape}"
        
        # Verify top_contributors is a list
        assert isinstance(result.top_contributors, list), \
            f"top_contributors should be list, got {type(result.top_contributors)}"
        
        # Verify top_contributors structure (list of tuples)
        if len(result.top_contributors) > 0:
            assert isinstance(result.top_contributors[0], tuple), \
                f"top_contributors elements should be tuples, got {type(result.top_contributors[0])}"
            assert len(result.top_contributors[0]) == 2, \
                f"top_contributors tuples should have 2 elements, got {len(result.top_contributors[0])}"
        
        # Verify actual and forecast are numpy arrays
        assert isinstance(result.actual, np.ndarray), \
            f"actual should be np.ndarray, got {type(result.actual)}"
        assert isinstance(result.forecast, np.ndarray), \
            f"forecast should be np.ndarray, got {type(result.forecast)}"
        
        # Verify actual and forecast have compatible shapes
        assert result.actual.shape == result.forecast.shape, \
            f"actual and forecast should have same shape, got actual={result.actual.shape}, forecast={result.forecast.shape}"
        
        # Verify weight is numpy array
        assert isinstance(result.weight, np.ndarray), \
            f"weight should be np.ndarray, got {type(result.weight)}"
        
        # Verify t_miss and v_miss are numpy arrays
        assert isinstance(result.t_miss, np.ndarray), \
            f"t_miss should be np.ndarray, got {type(result.t_miss)}"
        assert isinstance(result.v_miss, np.ndarray), \
            f"v_miss should be np.ndarray, got {type(result.v_miss)}"
        
        # Verify t_miss and v_miss have compatible shapes (same length)
        assert result.t_miss.shape == result.v_miss.shape, \
            f"t_miss and v_miss should have same shape, got t_miss={result.t_miss.shape}, v_miss={result.v_miss.shape}"
        
        # Verify innov is numpy array
        assert isinstance(result.innov, np.ndarray), \
            f"innov should be np.ndarray, got {type(result.innov)}"
        
        # Test 2: Return dictionary format (backward compatibility)
        try:
            result_dict = nowcast.decompose(
                target_series=target_series,
                target_period=target_period,
                view_date_old=view_date_old,
                view_date_new=view_date_new,
                return_dict=True
            )
        except ValueError as e:
            pytest.fail(f"News decomposition (dict format) failed: {e}. "
                       f"Target period: {target_period}, "
                       f"View dates: old={view_date_old}, new={view_date_new}")
        
        # Verify return type is dictionary
        assert isinstance(result_dict, dict), \
            f"Result should be dict when return_dict=True, got {type(result_dict)}"
        
        # Verify dictionary has all required keys
        required_keys = ['y_old', 'y_new', 'change', 'singlenews', 'top_contributors',
                        'actual', 'forecast', 'weight', 't_miss', 'v_miss', 'innov']
        for key in required_keys:
            assert key in result_dict, f"Result dict should have key '{key}'"
        
        # Verify dictionary values match NewsDecompResult values
        assert abs(result_dict['y_old'] - result.y_old) < 1e-10, \
            "Dictionary y_old should match NewsDecompResult y_old"
        assert abs(result_dict['y_new'] - result.y_new) < 1e-10, \
            "Dictionary y_new should match NewsDecompResult y_new"
        assert abs(result_dict['change'] - result.change) < 1e-10, \
            "Dictionary change should match NewsDecompResult change"
        
        # Test 3: Same view dates (should show minimal or zero change)
        # This tests edge case where old and new views are the same
        # Note: For same view dates, we need to get the data view again to select target_period
        X_same, Time_same, _ = nowcast.get_data_view(view_date_new)
        target_time_same = get_latest_time(Time_same)
        if isinstance(target_time_same, datetime):
            target_period_same = target_time_same
        else:
            target_period_same = to_python_datetime(target_time_same)
        
        try:
            result_same = nowcast.decompose(
                target_series=target_series,
                target_period=target_period_same,
                view_date_old=view_date_new,  # Same as new
                view_date_new=view_date_new,
                return_dict=False
            )
        except ValueError as e:
            pytest.fail(f"News decomposition (same view dates) failed: {e}. "
                       f"Target period: {target_period_same}, "
                       f"View date: {view_date_new}")
        
        # Verify result is valid
        assert isinstance(result_same, NewsDecompResult), \
            "Result with same view dates should still be NewsDecompResult"
        
        # When view dates are the same, change should be very small (near zero)
        # Allow some numerical tolerance
        assert abs(result_same.change) < 1e-6, \
            f"Change should be near zero when view dates are same, got {result_same.change}"
        assert abs(result_same.y_new - result_same.y_old) < 1e-6, \
            "y_new and y_old should be nearly equal when view dates are same"
