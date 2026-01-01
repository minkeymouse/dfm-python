"""Tests for dataset.process module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dfm_python.dataset.process import TimeIndex, parse_timestamp
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
    
    def test_parse_timestamp_string(self):
        """Test parsing timestamp from string."""
        from dfm_python.dataset.process import parse_timestamp
        # Test ISO format
        dt1 = parse_timestamp('2020-01-01')
        assert isinstance(dt1, datetime)
        assert dt1.year == 2020
        assert dt1.month == 1
        assert dt1.day == 1
        # Test ISO format with time
        dt2 = parse_timestamp('2020-01-01T12:30:45')
        assert isinstance(dt2, datetime)
        assert dt2.hour == 12
    
    def test_parse_timestamp_datetime(self):
        """Test parsing timestamp from datetime."""
        from dfm_python.dataset.process import parse_timestamp
        # Test that datetime objects are returned as-is
        dt = datetime(2020, 1, 1, 12, 30, 45)
        result = parse_timestamp(dt)
        assert result == dt
        assert isinstance(result, datetime)
    
    def test_time_index_invalid_series_dtype(self):
        """Test TimeIndex raises DataValidationError for non-datetime Series."""
        # Create Series with non-datetime dtype that cannot be converted
        # Use object dtype with non-datetime strings to ensure conversion fails
        invalid_series = pd.Series(['not', 'a', 'date', 'string'], dtype=object)
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
        """Test TimeIndex comparison raises DataValidationError for invalid types."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='ME')
        time_index = TimeIndex(dates)
        # Try comparison with invalid type (string)
        with pytest.raises(DataValidationError, match="Cannot compare TimeIndex with"):
            _ = time_index >= "2020-01-01"
        with pytest.raises(DataValidationError, match="Cannot compare TimeIndex with"):
            _ = time_index <= "2020-01-01"
        with pytest.raises(DataValidationError, match="Cannot compare TimeIndex with"):
            _ = time_index > "2020-01-01"
        with pytest.raises(DataValidationError, match="Cannot compare TimeIndex with"):
            _ = time_index < "2020-01-01"
    
    def test_parse_timestamp_invalid_string(self):
        """Test parse_timestamp raises DataValidationError for invalid string formats."""
        from dfm_python.dataset.process import parse_timestamp
        # Try invalid string format
        with pytest.raises(DataValidationError, match="Cannot parse datetime string"):
            parse_timestamp("invalid-date-format")
    
    def test_parse_timestamp_invalid_type(self):
        """Test parse_timestamp raises DataValidationError for unsupported types."""
        from dfm_python.dataset.process import parse_timestamp
        # Try unsupported type (list)
        with pytest.raises(DataValidationError, match="Cannot parse"):
            parse_timestamp([2020, 1, 1])



