# Refactoring Plan - Iteration 3

**Date**: 2025-01-11  
**Focus**: Extract validation functions from `config.py`  
**Scope**: Small, focused, reversible change

---

## Objective

Extract validation functions (`validate_frequency()`, `validate_transformation()`) and related constants from `config.py` into a separate `config_validation.py` file. This is a small, focused step toward better organization of the 899-line `config.py` file.

---

## Current Situation

**File**: `src/dfm_python/config.py` (899 lines)

**Current Structure**:
- Validation functions: `validate_frequency()`, `validate_transformation()` (~12 lines)
- Validation constants: `_VALID_FREQUENCIES`, `_VALID_TRANSFORMATIONS` (~8 lines)
- Dataclasses: `BlockConfig`, `SeriesConfig`, `Params`, `DFMConfig` (~400 lines)
- Other: Constants, utilities, imports (~480 lines)

**Usage**:
- `validate_frequency()` used in `BlockConfig.__post_init__()` and `SeriesConfig.__post_init__()`
- `validate_transformation()` used in `SeriesConfig.__post_init__()`
- Both are only used within `config.py` (internal validation)

**Assessment**: 
- Small, self-contained functions
- Clear separation opportunity
- Low risk (only used internally)
- Reversible (can merge back if needed)

---

## Tasks

### Task 1: Create `config_validation.py`
- [x] Create new file `src/dfm_python/config_validation.py`
- [x] Move validation functions: `validate_frequency()`, `validate_transformation()`
- [x] Move validation constants: `_VALID_FREQUENCIES`, `_VALID_TRANSFORMATIONS`
- [x] Add necessary imports (warnings)
- [x] Add module docstring

### Task 2: Update `config.py`
- [x] Remove validation functions and constants
- [x] Add import: `from .config_validation import validate_frequency, validate_transformation`
- [x] Verify dataclasses still use validation functions correctly

### Task 3: Verify Nothing Breaks
- [x] Test imports: `from dfm_python.config import BlockConfig, SeriesConfig`
- [x] Test validation: Create `BlockConfig` and `SeriesConfig` with invalid values
- [x] Verify error messages still work correctly
- [x] Check that all internal usage still works

---

## Expected Outcome

- **Created**: `config_validation.py` (~25 lines)
- **Reduced**: `config.py` by ~20 lines (from 899 to ~879)
- **Result**: Better separation of concerns, clearer organization
- **Risk**: Very low (functions only used internally, clear separation)

---

## Rollback Plan

If anything breaks:
1. Restore validation functions to `config.py`
2. Remove `config_validation.py`
3. Verify imports work again

---

## Success Criteria

- [x] `config_validation.py` created with validation functions (77 lines)
- [x] `config.py` updated to import from `config_validation.py` (878 lines, down from 899)
- [x] All validation still works correctly
- [x] No functional changes
- [x] Code is cleaner and better organized

## Completion Status

✅ **COMPLETED** - All tasks finished successfully.

**Results**:
- Created `config_validation.py` (77 lines) with validation functions
- Reduced `config.py` by 21 lines (from 899 to 878)
- All imports and validation verified and working
- No functional changes
- Better separation of concerns

---

## Notes

- This is a **small, incremental step** toward better organization
- Future iterations can extract dataclasses to `config_models.py`
- This change is **reversible** and **low risk**
- Sets pattern for future refactoring of `config.py`

---

## Code Changes Preview

**New file**: `config_validation.py`
```python
"""Validation functions for DFM configuration."""

import warnings

# Valid frequency codes
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes
_VALID_TRANSFORMATIONS = {
    'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 
    'cch', 'cca', 'log'
}

def validate_frequency(frequency: str) -> str:
    """Validate frequency code."""
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {_VALID_FREQUENCIES}")
    return frequency

def validate_transformation(transformation: str) -> str:
    """Validate transformation code."""
    if transformation not in _VALID_TRANSFORMATIONS:
        warnings.warn(f"Unknown transformation code: {transformation}. Will use untransformed data.")
    return transformation
```

**Updated**: `config.py`
- Remove validation functions and constants
- Add: `from .config_validation import validate_frequency, validate_transformation`
