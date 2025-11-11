# Refactoring Plan - Iteration 7

**Date**: 2025-01-11  
**Focus**: Extract `rem_nans_spline()` from `data_loader.py` to `data/utils.py`  
**Scope**: Small, focused, reversible change - first step in splitting `data_loader.py`

---

## Objective

Extract the `rem_nans_spline()` function from `data_loader.py` to a new `data/utils.py` module. This is the first incremental step in splitting the large `data_loader.py` file (783 lines) into a more organized `data/` package structure.

**Rationale**: This function is:
- Self-contained (only uses numpy/scipy, no dependencies on other `data_loader.py` functions)
- Used by multiple modules (`dfm.py`, `core/em/initialization.py`, `core/em/iteration.py`, `utils/__init__.py`)
- Clear, single-purpose function (NaN handling)
- Matches MATLAB structure (`remNaNs_spline.m` is a separate file)
- Reduces `data_loader.py` by ~110 lines

---

## Current Situation

**File**: `src/dfm_python/data_loader.py` (783 lines)

**Function to Extract**:
- `rem_nans_spline()` (lines 579-687, ~110 lines including docstring)
  - **Location**: `data_loader.py:579`
  - **Usage**: Used by 4 modules:
    - `dfm.py` (line 51)
    - `core/em/initialization.py` (line 72)
    - `core/em/iteration.py` (line 73)
    - `utils/__init__.py` (line 3)
  - **Type**: Data preprocessing utility (NaN handling)
  - **Dependencies**: Only `numpy`, `scipy.interpolate.CubicSpline`, `scipy.signal.lfilter`
  - **Self-contained**: No dependencies on other `data_loader.py` functions

**Target Location**:
- `data/utils.py` (new file, ~120 lines including imports and docstring)
  - Part of new `data/` package structure
  - Will eventually contain other data utilities (`summarize()` will be moved later)

**Assessment**: 
- Self-contained function with clear purpose
- Low risk (function is independent, only uses standard libraries)
- Establishes pattern for future extractions from `data_loader.py`
- Justifies new `data/` package (necessary for organization)

---

## Tasks

### Task 1: Create `data/` Package Structure
- [ ] Create `src/dfm_python/data/` directory
- [ ] Create `src/dfm_python/data/__init__.py` (re-export `rem_nans_spline` for backward compatibility)
- [ ] Create `src/dfm_python/data/utils.py` with function
- [ ] Add necessary imports: `numpy`, `scipy.interpolate.CubicSpline`, `scipy.signal.lfilter`
- [ ] Copy function with full docstring

### Task 2: Update Imports in Dependent Modules
- [ ] Update `dfm.py`: `from .data_loader import rem_nans_spline` → `from .data.utils import rem_nans_spline`
- [ ] Update `core/em/initialization.py`: `from ...data_loader import rem_nans_spline` → `from ...data.utils import rem_nans_spline`
- [ ] Update `core/em/iteration.py`: `from ...data_loader import rem_nans_spline` → `from ...data.utils import rem_nans_spline`
- [ ] Update `utils/__init__.py`: `from ..data_loader import rem_nans_spline` → `from ..data.utils import rem_nans_spline`

### Task 3: Update `data_loader.py`
- [ ] Remove `rem_nans_spline()` function definition (lines 579-687)
- [ ] Update module docstring if needed (remove mention of NaN handling if it's the only reference)
- [ ] Keep other functions unchanged

### Task 4: Maintain Backward Compatibility
- [ ] Add import in `data_loader.py`: `from .data.utils import rem_nans_spline`
- [ ] Re-export in `data_loader.py` for backward compatibility (if needed)
- [ ] OR: Update `data/__init__.py` to re-export, and update `data_loader.py` to import from `data`

### Task 5: Verify Nothing Breaks
- [ ] Test imports: `from dfm_python.data.utils import rem_nans_spline`
- [ ] Test backward compatibility: `from dfm_python.data_loader import rem_nans_spline` (if maintained)
- [ ] Verify function works identically
- [ ] Check that all dependent modules can import correctly

---

## Expected Outcome

- **Moved**: `rem_nans_spline()` from `data_loader.py` to `data/utils.py`
- **Reduced**: `data_loader.py` by ~110 lines (from 783 to ~673)
- **Created**: New `data/` package structure (foundation for future splits)
- **Result**: Better organization, function in proper location, establishes pattern for future extractions
- **Risk**: Low (function is self-contained, dependencies are standard libraries)

---

## Code Changes Preview

**Create `src/dfm_python/data/utils.py`**:
```python
"""Data utilities for DFM estimation.

This module provides utility functions for data preprocessing and handling,
including missing value treatment and data summarization.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter


def rem_nans_spline(X: np.ndarray, method: int = 2, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Treat NaNs in dataset for DFM estimation using standard interpolation methods.
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged]
    ...
```

**Create `src/dfm_python/data/__init__.py`**:
```python
"""Data loading and transformation utilities for DFM estimation.

This package provides comprehensive data handling for Dynamic Factor Models,
organized into focused modules for better maintainability.
"""

from .utils import rem_nans_spline

__all__ = ['rem_nans_spline']
```

**Update `data_loader.py`**:
- Remove: `def rem_nans_spline(...)` function definition (lines 579-687)
- Add: `from .data.utils import rem_nans_spline` (for backward compatibility)
- OR: Re-export via `data/__init__.py` and update imports

**Update dependent modules**:
- `dfm.py`: `from .data.utils import rem_nans_spline`
- `core/em/initialization.py`: `from ...data.utils import rem_nans_spline`
- `core/em/iteration.py`: `from ...data.utils import rem_nans_spline`
- `utils/__init__.py`: `from ..data.utils import rem_nans_spline`

---

## Rollback Plan

If anything breaks:
1. Restore `rem_nans_spline()` function to `data_loader.py`
2. Remove `data/utils.py` and `data/__init__.py`
3. Revert import changes in dependent modules
4. Verify imports work again

---

## Success Criteria

- [ ] `rem_nans_spline()` added to `data/utils.py`
- [ ] `rem_nans_spline()` exported in `data/__init__.py`
- [ ] `rem_nans_spline()` removed from `data_loader.py`
- [ ] All imports updated in dependent modules
- [ ] Backward compatibility maintained (if applicable)
- [ ] All imports work correctly
- [ ] Function works identically
- [ ] No functional changes
- [ ] Code is cleaner and better organized
- [ ] `data_loader.py` reduced by ~110 lines

---

## Notes

- This is the **first step** in splitting `data_loader.py` (783 lines → eventually ~200 lines per module)
- Function is **self-contained** (only uses numpy/scipy)
- Establishes **pattern** for future extractions (`summarize()`, `transform_data()`, etc.)
- New `data/` package is **justified** as necessary for organization (matches MATLAB structure)
- This change is **reversible** and **low risk**
- Function name remains `rem_nans_spline()` (no change needed)

---

## Why This Function First

1. **Self-contained**: Only uses standard libraries, no dependencies on other `data_loader.py` functions
2. **Clear purpose**: Single-purpose function (NaN handling)
3. **Low risk**: Function is independent, easy to verify
4. **Establishes pattern**: First step in splitting `data_loader.py`
5. **Matches MATLAB**: `remNaNs_spline.m` is a separate file
6. **High usage**: Used by 4 modules, so moving it improves organization

---

## Dependencies

- `numpy` - Standard library ✅
- `scipy.interpolate.CubicSpline` - Standard library ✅
- `scipy.signal.lfilter` - Standard library ✅
- No dependencies on other `data_loader.py` functions ✅

---

## Future Iterations

After this iteration:
- **Iteration 8**: Extract `summarize()` to `data/utils.py`
- **Iteration 9**: Extract `transform_data()` and `_transform_series()` to `data/transformer.py`
- **Iteration 10**: Extract config loading functions to `data/config_loader.py`
- **Iteration 11**: Extract data loading functions to `data/loader.py`
- **Final**: Remove or repurpose `data_loader.py` (or keep as thin wrapper)

This incremental approach allows for careful verification at each step.
