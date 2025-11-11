# Refactoring Plan - Iteration 8

**Date**: 2025-01-11  
**Focus**: Extract `summarize()` from `data_loader.py` to `data/utils.py`  
**Scope**: Small, focused, reversible change - continues splitting `data_loader.py`

---

## Objective

Extract the `summarize()` function from `data_loader.py` to `data/utils.py`. This continues the incremental splitting of the large `data_loader.py` file (671 lines) into a more organized `data/` package structure.

**Rationale**: This function is:
- Self-contained utility (data summarization/visualization)
- Used by only one module (`utils/__init__.py`)
- Clear, single-purpose function (data summary display)
- Matches MATLAB structure (`summarize.m` is a separate file)
- Logical next step (same module as `rem_nans_spline()` which was extracted in iteration 7)
- Reduces `data_loader.py` by ~94 lines

---

## Current Situation

**File**: `src/dfm_python/data_loader.py` (671 lines)

**Function to Extract**:
- `summarize()` (lines 578-671, ~94 lines including docstring)
  - **Location**: `data_loader.py:578`
  - **Usage**: Used by 1 module:
    - `utils/__init__.py` (line 4)
  - **Type**: Data utility (summarization/visualization)
  - **Dependencies**: 
    - `numpy`, `pandas` (standard libraries)
    - `DFMConfig` from `.config`
    - `safe_get_method` from `.core.helpers`
    - `_TRANSFORM_UNITS_MAP` from `.config`
    - `logger` (currently from `data_loader.py`)
  - **Self-contained**: No dependencies on other `data_loader.py` functions

**Target Location**:
- `data/utils.py` (existing file, 121 lines, will become ~215 lines)
  - Already contains `rem_nans_spline()` (extracted in iteration 7)
  - Good fit for data utilities (summarization alongside NaN handling)

**Assessment**: 
- Self-contained function with clear purpose
- Low risk (function is independent, only uses standard libraries and config)
- Logical next step (same module as previous extraction)
- Justifies continuation of `data/` package structure

---

## Tasks

### Task 1: Add Function to `data/utils.py`
- [ ] Add `summarize()` function to `data/utils.py`
- [ ] Add necessary imports: `logging`, `DFMConfig` from `..config`, `safe_get_method` from `..core.helpers`
- [ ] Add logger setup: `_logger = logging.getLogger(__name__)`
- [ ] Preserve function docstring and implementation
- [ ] Update module docstring to mention summarization

### Task 2: Update `data/__init__.py`
- [ ] Add `summarize` to imports from `.utils`
- [ ] Add `summarize` to `__all__`
- [ ] Function accessible via `from dfm_python.data import summarize`

### Task 3: Update Imports in Dependent Modules
- [ ] Update `utils/__init__.py`: `from ..data_loader import summarize` → `from ..data.utils import summarize`
- [ ] Verify no other modules import `summarize` directly

### Task 4: Update `data_loader.py`
- [ ] Remove `summarize()` function definition (lines 578-671)
- [ ] Add backward compatibility import: `from .data.utils import summarize`
- [ ] Update module docstring if needed (remove mention of summarization if it's the only reference)

### Task 5: Verify Nothing Breaks
- [ ] Test imports: `from dfm_python.data.utils import summarize`
- [ ] Test backward compatibility: `from dfm_python.data_loader import summarize` (if maintained)
- [ ] Verify function works identically
- [ ] Check that all dependent modules can import correctly

---

## Expected Outcome

- **Moved**: `summarize()` from `data_loader.py` to `data/utils.py`
- **Reduced**: `data_loader.py` by ~94 lines (from 671 to ~577)
- **Result**: Better organization, function in proper location, continues `data/` package structure
- **Risk**: Low (function is self-contained, dependencies are standard libraries and config)

---

## Code Changes Preview

**Add to `data/utils.py`**:
```python
"""Data utilities for DFM estimation.

This module provides utility functions for data preprocessing and handling,
including missing value treatment and data summarization.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import logging
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter

from ..config import DFMConfig, _TRANSFORM_UNITS_MAP
from ..core.helpers import safe_get_method

_logger = logging.getLogger(__name__)

# ... existing rem_nans_spline() ...

def summarize(X: np.ndarray, Time, config: DFMConfig, vintage: Optional[str] = None) -> None:
    """Display data summary table.
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged]
    ...
```

**Update `data/__init__.py`**:
```python
"""Data loading and transformation utilities for DFM estimation.

This package provides comprehensive data handling for Dynamic Factor Models,
organized into focused modules for better maintainability.
"""

from .utils import rem_nans_spline, summarize

__all__ = ['rem_nans_spline', 'summarize']
```

**Update `data_loader.py`**:
- Remove: `def summarize(...)` function definition (lines 578-671)
- Add: `from .data.utils import summarize` (for backward compatibility)

**Update `utils/__init__.py`**:
- Change: `from ..data_loader import summarize` → `from ..data.utils import summarize`

---

## Rollback Plan

If anything breaks:
1. Restore `summarize()` function to `data_loader.py`
2. Remove `summarize()` from `data/utils.py`
3. Remove from `data/__init__.py`
4. Revert import changes in `utils/__init__.py`
5. Verify imports work again

---

## Success Criteria

- [ ] `summarize()` added to `data/utils.py`
- [ ] `summarize()` exported in `data/__init__.py`
- [ ] `summarize()` removed from `data_loader.py`
- [ ] All imports updated in dependent modules
- [ ] Backward compatibility maintained (if applicable)
- [ ] All imports work correctly
- [ ] Function works identically
- [ ] No functional changes
- [ ] Code is cleaner and better organized
- [ ] `data_loader.py` reduced by ~94 lines

---

## Notes

- This is the **second step** in splitting `data_loader.py` (671 lines → eventually ~200 lines per module)
- Function is **self-contained** (only uses standard libraries, config, and helpers)
- Continues **pattern** established in iteration 7
- New `data/` package structure is **justified** as necessary for organization (matches MATLAB structure)
- This change is **reversible** and **low risk**
- Function name remains `summarize()` (no change needed)

---

## Why This Function Next

1. **Same Module**: Logical next step after `rem_nans_spline()` (both in `data/utils.py`)
2. **Self-contained**: Only uses standard libraries, config, and helpers
3. **Clear purpose**: Single-purpose function (data summarization)
4. **Low risk**: Function is independent, easy to verify
5. **Continues pattern**: Second step in splitting `data_loader.py`
6. **Matches MATLAB**: `summarize.m` is a separate file
7. **Low usage**: Used by only one module, easy to update

---

## Dependencies

- `numpy` - Standard library ✅
- `pandas` - Standard library ✅
- `logging` - Standard library ✅
- `DFMConfig` from `..config` - Package dependency ✅
- `safe_get_method` from `..core.helpers` - Package dependency ✅
- `_TRANSFORM_UNITS_MAP` from `..config` - Package dependency ✅
- No dependencies on other `data_loader.py` functions ✅

---

## Future Iterations

After this iteration:
- **Iteration 9**: Extract `transform_data()` and `_transform_series()` to `data/transformer.py`
- **Iteration 10**: Extract config loading functions to `data/config_loader.py`
- **Iteration 11**: Extract data loading functions to `data/loader.py`
- **Final**: Remove or repurpose `data_loader.py` (or keep as thin wrapper)

This incremental approach allows for careful verification at each step.
