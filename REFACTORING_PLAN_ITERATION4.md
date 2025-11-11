# Refactoring Plan - Iteration 4

**Date**: 2025-01-11  
**Focus**: Move `_resolve_param()` from `dfm.py` to `core/helpers/utils.py`  
**Scope**: Small, focused, reversible change

---

## Objective

Move the `_resolve_param()` helper function from `dfm.py` to `core/helpers/utils.py`. This is a small, focused step toward consolidating helper functions and reducing `dfm.py` size.

---

## Current Situation

**File**: `src/dfm_python/dfm.py` (878 lines)

**Function to Move**:
- `_resolve_param()` (3 lines) - Simple parameter resolution utility
  - **Location**: `dfm.py:314`
  - **Usage**: Used 15 times in `_prepare_data_and_params()` function
  - **Type**: General utility pattern (override if provided, else default)

**Target Location**:
- `core/helpers/utils.py` (196 lines) - General utility helpers
  - Already contains: `append_or_initialize`, `create_empty_matrix`, `reshape_to_column_vector`, `safe_numerical_operation`, etc.
  - Good fit for general utility functions

**Assessment**: 
- Small, self-contained function
- General utility pattern (not DFM-specific)
- Used only within `dfm.py` currently, but could be reused
- Low risk (simple function, clear purpose)

---

## Tasks

### Task 1: Add Function to `core/helpers/utils.py`
- [x] Add `resolve_param()` function to `core/helpers/utils.py`
- [x] Add function docstring (can be simple, similar to current)
- [x] Export in `core/helpers/__init__.py` (add to `__all__`)

### Task 2: Update `dfm.py`
- [x] Remove `_resolve_param()` function definition
- [x] Add import: `from .core.helpers import resolve_param`
- [x] Update all 15 usages: `_resolve_param(...)` → `resolve_param(...)`
- [x] Verify function calls work correctly

### Task 3: Verify Nothing Breaks
- [x] Test imports: `from dfm_python.core.helpers import resolve_param`
- [x] Test DFM class: `from dfm_python import DFM`
- [x] Verify parameter resolution still works in `_prepare_data_and_params()`
- [x] Check that all internal usage still works

---

## Expected Outcome

- **Moved**: `_resolve_param()` from `dfm.py` to `core/helpers/utils.py`
- **Reduced**: `dfm.py` by 3 lines (from 878 to 875)
- **Result**: Better organization, function available for reuse
- **Risk**: Very low (simple function, clear purpose, used only internally)

---

## Code Changes Preview

**Add to `core/helpers/utils.py`**:
```python
def resolve_param(override: Any, default: Any) -> Any:
    """Resolve parameter: use override if provided, else use default.
    
    Parameters
    ----------
    override : Any
        Override value (used if not None)
    default : Any
        Default value (used if override is None)
        
    Returns
    -------
    Any
        Override value if not None, else default value
        
    Examples
    --------
    >>> resolve_param(10, 5)
    10
    >>> resolve_param(None, 5)
    5
    """
    return override if override is not None else default
```

**Update `dfm.py`**:
- Remove: `def _resolve_param(...)` function definition
- Add: `from .core.helpers import resolve_param`
- Update: All `_resolve_param(...)` calls to `resolve_param(...)`

**Update `core/helpers/__init__.py`**:
- Add `resolve_param` to imports and `__all__`

---

## Rollback Plan

If anything breaks:
1. Restore `_resolve_param()` function to `dfm.py`
2. Remove `resolve_param()` from `core/helpers/utils.py`
3. Remove from `core/helpers/__init__.py`
4. Revert import changes in `dfm.py`
5. Verify imports work again

---

## Success Criteria

- [x] `resolve_param()` added to `core/helpers/utils.py` (221 lines, up from 196)
- [x] `resolve_param()` exported in `core/helpers/__init__.py`
- [x] `_resolve_param()` removed from `dfm.py`
- [x] All 15 usages updated in `dfm.py`
- [x] All imports work correctly
- [x] Parameter resolution still works
- [x] No functional changes
- [x] Code is cleaner and better organized

## Completion Status

✅ **COMPLETED** - All tasks finished successfully.

**Results**:
- Moved `resolve_param()` from `dfm.py` to `core/helpers/utils.py`
- Reduced `dfm.py` by 5 lines (from 878 to 873)
- All 15 usages updated correctly
- All imports verified and working
- Function now available for reuse
- No functional changes

---

## Notes

- This is a **small, incremental step** toward consolidating helpers
- Future iterations can move `_safe_mean_std()` and `_standardize_data()` to `core/helpers/estimation.py`
- This change is **reversible** and **low risk**
- Function name changed from `_resolve_param()` to `resolve_param()` (removed `_` prefix since it's now a public helper)

---

## Why This Function First

1. **Smallest**: Only 3 lines
2. **Most general**: Not DFM-specific, pure utility
3. **Lowest risk**: Simple function, clear purpose
4. **High usage**: Used 15 times, so moving it has clear impact
5. **Good fit**: Matches pattern of other utilities in `utils.py`
