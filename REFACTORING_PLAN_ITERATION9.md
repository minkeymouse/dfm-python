# Refactoring Plan - Iteration 9

**Date**: 2025-01-11  
**Focus**: Extract `transform_data()` and `_transform_series()` from `data_loader.py` to `data/transformer.py`  
**Scope**: Small, focused, reversible change - continues splitting `data_loader.py`

---

## Objective

Extract the `transform_data()` and `_transform_series()` functions from `data_loader.py` to a new `data/transformer.py` module. This continues the incremental splitting of the large `data_loader.py` file (575 lines) into a more organized `data/` package structure.

**Rationale**: These functions are:
- Related transformation functions (logical grouping)
- Self-contained (data transformation logic)
- Used together (`transform_data()` uses `_transform_series()`)
- Clear, single-purpose functions (data transformation)
- Matches MATLAB structure (`transformData.m` is a separate file)
- Reduces `data_loader.py` by ~135 lines

---

## Current Situation

**File**: `src/dfm_python/data_loader.py` (575 lines)

**Functions to Extract**:
- `_transform_series()` (lines 178-240, ~64 lines including docstring)
  - **Location**: `data_loader.py:178`
  - **Usage**: Only used by `transform_data()` (internal helper)
  - **Type**: Data transformation helper (single series transformation)
  - **Dependencies**: Only `numpy` (standard library)
  - **Self-contained**: No dependencies on other `data_loader.py` functions

- `transform_data()` (lines 243-312, ~71 lines including docstring)
  - **Location**: `data_loader.py:243`
  - **Usage**: Used by 2 modules:
    - `load_data()` in `data_loader.py` (line 532)
    - `__init__.py` (line 65, exported)
  - **Type**: Data transformation (multi-series transformation)
  - **Dependencies**: 
    - `numpy`, `pandas` (standard libraries)
    - `DFMConfig` from `.config`
    - `FREQUENCY_HIERARCHY` from `.utils.aggregation`
  - **Self-contained**: No dependencies on other `data_loader.py` functions (uses `_transform_series()` which will be moved together)

**Target Location**:
- `data/transformer.py` (new file, ~150 lines including imports and docstring)
  - Part of `data/` package structure
  - Will contain both transformation functions

**Assessment**: 
- Self-contained functions with clear purpose
- Low risk (functions are independent, only use standard libraries and config)
- Logical grouping (both are transformation functions)
- Justifies new module in `data/` package (necessary for organization)

---

## Tasks

### Task 1: Create `data/transformer.py`
- [ ] Create `src/dfm_python/data/transformer.py` with both functions
- [ ] Add necessary imports: `numpy`, `pandas`, `DFMConfig` from `..config`, `FREQUENCY_HIERARCHY` from `..utils.aggregation`
- [ ] Copy both functions with full docstrings
- [ ] Ensure `transform_data()` imports `_transform_series()` from same module

### Task 2: Update `data/__init__.py`
- [ ] Add `transform_data` to imports from `.transformer`
- [ ] Add `transform_data` to `__all__`
- [ ] Note: `_transform_series()` is private, don't export it

### Task 3: Update Imports in Dependent Modules
- [ ] Update `__init__.py`: `from .data_loader import transform_data` → `from .data.transformer import transform_data`
- [ ] Update `data_loader.py`: `transform_data()` usage in `load_data()` → import from `data.transformer`

### Task 4: Update `data_loader.py`
- [ ] Remove `_transform_series()` function definition (lines 178-240)
- [ ] Remove `transform_data()` function definition (lines 243-312)
- [ ] Add backward compatibility import: `from .data.transformer import transform_data`
- [ ] Update `load_data()` to import `transform_data` from `data.transformer` (or use backward compatibility import)

### Task 5: Verify Nothing Breaks
- [ ] Test imports: `from dfm_python.data.transformer import transform_data`
- [ ] Test backward compatibility: `from dfm_python.data_loader import transform_data` (if maintained)
- [ ] Verify function works identically
- [ ] Check that all dependent modules can import correctly

---

## Expected Outcome

- **Moved**: `transform_data()` and `_transform_series()` from `data_loader.py` to `data/transformer.py`
- **Reduced**: `data_loader.py` by ~135 lines (from 575 to ~440)
- **Created**: New `data/transformer.py` module (~150 lines)
- **Result**: Better organization, functions in proper location, continues `data/` package structure
- **Risk**: Low (functions are self-contained, dependencies are standard libraries and config)

---

## Code Changes Preview

**Create `src/dfm_python/data/transformer.py`**:
```python
"""Data transformation utilities for DFM estimation.

This module provides functions for transforming time series data according to
configuration specifications, including differences, percent changes, and log
transformations.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from ..config import DFMConfig
from ..utils.aggregation import FREQUENCY_HIERARCHY


def _transform_series(Z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """Transform a single time series according to formula.
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged]
    ...


def transform_data(Z: np.ndarray, Time: pd.DatetimeIndex, config: DFMConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Transform each data series according to configuration.
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged, uses _transform_series from same module]
    ...
```

**Update `data/__init__.py`**:
```python
"""Data loading and transformation utilities for DFM estimation.

This package provides comprehensive data handling for Dynamic Factor Models,
organized into focused modules for better maintainability.
"""

from .utils import rem_nans_spline, summarize
from .transformer import transform_data

__all__ = ['rem_nans_spline', 'summarize', 'transform_data']
```

**Update `data_loader.py`**:
- Remove: `def _transform_series(...)` function definition (lines 178-240)
- Remove: `def transform_data(...)` function definition (lines 243-312)
- Add: `from .data.transformer import transform_data` (for backward compatibility)
- Update: `load_data()` to use imported `transform_data` (or rely on backward compatibility)

**Update `__init__.py`**:
- Change: `from .data_loader import transform_data` → `from .data.transformer import transform_data`

---

## Rollback Plan

If anything breaks:
1. Restore both functions to `data_loader.py`
2. Remove `data/transformer.py`
3. Remove from `data/__init__.py`
4. Revert import changes in dependent modules
5. Verify imports work again

---

## Success Criteria

- [ ] Both functions added to `data/transformer.py`
- [ ] `transform_data` exported in `data/__init__.py`
- [ ] Both functions removed from `data_loader.py`
- [ ] All imports updated in dependent modules
- [ ] Backward compatibility maintained (if applicable)
- [ ] All imports work correctly
- [ ] Functions work identically
- [ ] No functional changes
- [ ] Code is cleaner and better organized
- [ ] `data_loader.py` reduced by ~135 lines

---

## Notes

- This is the **third step** in splitting `data_loader.py` (575 lines → eventually ~200 lines per module)
- Functions are **self-contained** (only use standard libraries, config, and aggregation utils)
- Continues **pattern** established in iterations 7-8
- New `data/transformer.py` module is **justified** as necessary for organization (matches MATLAB structure)
- This change is **reversible** and **low risk**
- Function names remain unchanged (no change needed)
- `_transform_series()` is private (not exported, only used internally by `transform_data()`)

---

## Why These Functions Next

1. **Logical Grouping**: Both are transformation functions, belong together
2. **Self-contained**: Only use standard libraries, config, and aggregation utils
3. **Clear purpose**: Single-purpose functions (data transformation)
4. **Low risk**: Functions are independent, easy to verify
5. **Continues pattern**: Third step in splitting `data_loader.py`
6. **Matches MATLAB**: `transformData.m` is a separate file
7. **Dependency**: `transform_data()` uses `_transform_series()`, so they should be moved together

---

## Dependencies

- `numpy` - Standard library ✅
- `pandas` - Standard library ✅
- `DFMConfig` from `..config` - Package dependency ✅
- `FREQUENCY_HIERARCHY` from `..utils.aggregation` - Package dependency ✅
- No dependencies on other `data_loader.py` functions ✅

---

## Future Iterations

After this iteration:
- **Iteration 10**: Extract config loading functions to `data/config_loader.py`
- **Iteration 11**: Extract data loading functions to `data/loader.py`
- **Final**: Remove or repurpose `data_loader.py` (or keep as thin wrapper)

This incremental approach allows for careful verification at each step.
