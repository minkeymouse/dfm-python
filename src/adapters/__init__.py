"""Adapters for DB, Data View, and Model Result integration.

This module provides adapters for:
- Database integration (SQLite, PostgreSQL, etc.) - requires dfm-python[db]
- Data view management (time-point specific data views)
- Model result storage and loading (DFM estimation results persistence)

All adapters follow the Adapter Pattern, allowing flexible integration
with various backends while maintaining a consistent interface.

Default behavior (no environment variables needed):
- Model Result: File-based (pickle) stored in './model_results'
- Data View: File-based stored in './data/views'

Database usage (requires dfm-python[db] and environment variables):
- Install: pip install dfm-python[db]
- Set environment variables: DATABASE_TYPE, DATABASE_PATH (or DATABASE_HOST, etc.)
- Set: MODEL_RESULT_STORAGE_TYPE=database, DATA_VIEW_SOURCE=database
"""

__version__ = "0.3.0"

# Protocols and base classes
from .model_result import ModelResultSaver
from .database import DatabaseAdapter
from .data_view import DataViewManager

# Basic implementations (file-based, no dependencies)
from .model_result import PickleModelResultSaver
from .data_view import BasicDataViewManager

# Database implementations (require dfm-python[db])
try:
    from .database import SQLiteAdapter
    _has_db_support = True
except ImportError:
    _has_db_support = False
    SQLiteAdapter = None  # type: ignore

__all__ = [
    # Protocols
    "ModelResultSaver",
    "DatabaseAdapter",
    "DataViewManager",
    # Implementations
    "PickleModelResultSaver",
    "BasicDataViewManager",
]

if _has_db_support:
    __all__.append("SQLiteAdapter")

