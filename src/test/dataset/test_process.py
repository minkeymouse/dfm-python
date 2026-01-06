"""Tests for dataset.process module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python.dataset.time import TimeIndex
from dfm_python.utils.errors import DataValidationError


class TestTimeIndex:
    """Test suite for TimeIndex."""
    
    def test_time_index_initialization(self):
        """Test TimeIndex can be initialized."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='ME')  # 'ME' replaces deprecated 'M'
        time_index = TimeIndex(dates)
        # Verify TimeIndex has expected attributes and length
        assert len(time_index) == 10
        assert time_index[0] == dates[0]
    
    def test_time_index_parsing(self):
        """Test TimeIndex timestamp parsing."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='ME')
        time_index = TimeIndex(dates)
        # Test that individual elements can be accessed and are datetime objects
        first_date = time_index[0]
        assert isinstance(first_date, datetime)
        # Test slicing returns TimeIndex
        sliced = time_index[2:5]
        assert isinstance(sliced, TimeIndex)
        assert len(sliced) == 3


class TestParseTimestamp:
    """Test suite for parse_timestamp function."""
    
    def test_time_index_from_string(self):
        """Test TimeIndex can be created from string dates."""
        # Test ISO format
        dates_str = ['2020-01-01', '2020-02-01', '2020-03-01']
        time_index = TimeIndex(pd.Series(dates_str))
        assert len(time_index) == 3
        assert time_index[0].year == 2020
        assert time_index[0].month == 1
        assert time_index[0].day == 1
    
    def test_time_index_from_datetime(self):
        """Test TimeIndex can be created from datetime objects."""
        # Test that datetime objects work
        dates = [datetime(2020, 1, 1, 12, 30, 45), datetime(2020, 2, 1)]
        time_index = TimeIndex(pd.Series(dates))
        assert len(time_index) == 2
        assert time_index[0].hour == 12
    
    def test_time_index_invalid_series_dtype(self):
        """Test TimeIndex raises DataValidationError for non-datetime Series."""
        # Create Series with non-datetime dtype that cannot be converted
        # Use object dtype with non-datetime strings to ensure conversion fails
        import warnings
        invalid_series = pd.Series(['not', 'a', 'date', 'string'], dtype=object)
        # Suppress expected pandas warning about date parsing (test verifies error is raised)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Could not infer format.*", category=UserWarning)
            with pytest.raises(DataValidationError, match="Cannot convert Series with dtype"):
                TimeIndex(invalid_series)
    
    def test_time_index_invalid_type(self):
        """Test TimeIndex raises DataValidationError for unsupported input types."""
        # Try with unsupported type (dict)
        with pytest.raises(DataValidationError, match="Cannot create TimeIndex from"):
            TimeIndex({'a': 1, 'b': 2})
    
    def test_time_index_unsupported_index_type(self):
        """Test TimeIndex raises DataValidationError for unsupported index types."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='ME')
        time_index = TimeIndex(dates)
        # Try indexing with unsupported type (dict)
        with pytest.raises(DataValidationError, match="Unsupported index type"):
            _ = time_index[{'key': 'value'}]
    
    def test_time_index_comparison_invalid_type(self):
        """Test TimeIndex comparison raises TypeError for invalid types."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='ME')
        time_index = TimeIndex(dates)
        # Try comparison with invalid type (string) - TimeIndex doesn't implement comparison operators
        # so Python raises TypeError
        with pytest.raises(TypeError):
            _ = time_index >= "2020-01-01"
        with pytest.raises(TypeError):
            _ = time_index <= "2020-01-01"
        with pytest.raises(TypeError):
            _ = time_index > "2020-01-01"
        with pytest.raises(TypeError):
            _ = time_index < "2020-01-01"
    
    def test_time_index_invalid_string_format(self):
        """Test TimeIndex raises DataValidationError for invalid string formats."""
        # Try invalid string format
        import warnings
        invalid_dates = pd.Series(['invalid-date-format', 'also-invalid'], dtype=object)
        # Suppress expected pandas warning about date parsing (test verifies error is raised)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Could not infer format.*", category=UserWarning)
            with pytest.raises(DataValidationError, match="Cannot convert Series"):
                TimeIndex(invalid_dates)
    
    def test_time_index_invalid_list_type(self):
        """Test TimeIndex handles list input by converting to Series."""
        # TimeIndex should convert list to Series
        dates_list = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        time_index = TimeIndex(dates_list)
        assert len(time_index) == 2



