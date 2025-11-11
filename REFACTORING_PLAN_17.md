# Refactoring Plan - Iteration 17

**Date**: 2025-01-11  
**Focus**: Remove unused import from `news.py`  
**Priority**: Low  
**Effort**: Very Low  
**Risk**: Very Low

---

## Objective

Remove unused import `calculate_rmse` from `news.py` to improve code cleanliness, following the same pattern as Iteration 15's cleanup of `config.py`.

---

## Current State

### Problem
`news.py` imports `calculate_rmse` from `core.diagnostics` but it's never used in the file:
- Import: `from .core.diagnostics import calculate_rmse` (line 26)
- Usage: Not found in the file (grep shows no matches)

This is a small cleanup opportunity similar to Iteration 15.

### Current Import Section

```python
from .kalman import skf, fis, miss_data
from .config import DFMConfig
from .dfm import DFMResult
from .core.diagnostics import calculate_rmse  # <-- UNUSED
```

---

## Proposed Solution

### Step 1: Remove Unused Import
- **File**: `src/dfm_python/news.py`
- **Action**: Remove `calculate_rmse` from the import statement
- **Change**: `from .core.diagnostics import calculate_rmse` → remove this line

---

## Implementation Steps

### Step 1: Remove Unused Import
- **File**: `src/dfm_python/news.py`
- **Action**: Remove `calculate_rmse` from import statement
- **Verification**: Check syntax with `python3 -m py_compile`

### Step 2: Verify Functionality
- **Action**: Run syntax checks and basic import tests
- **Command**: `python3 -c "from dfm_python.news import news_dfm, update_nowcast"`
- **Note**: Full functional tests not required this iteration

---

## Benefits

1. **Cleaner Code**: Removes dead import
2. **Better Maintainability**: No confusion about what's actually used
3. **Consistency**: Follows same pattern as Iteration 15
4. **Small Improvement**: Low effort, low risk

---

## Risks and Mitigation

### Risk 1: Removing Used Import
- **Risk**: Very Low - verified with grep that it's not used
- **Mitigation**: Double-check with grep before removing

### Risk 2: Breaking Functionality
- **Risk**: Very Low - only removing unused import
- **Mitigation**: Syntax check and import test

---

## Testing Strategy

### Syntax Validation
- Run `python3 -m py_compile src/dfm_python/news.py`
- Verify imports work correctly

### Import Validation
- Test all imports from `news` package
- Verify functionality unchanged

---

## Rollback Plan

If issues are discovered:
1. Revert changes to `news.py` using git
2. All changes are in a single file, easy to revert
3. No external dependencies changed

---

## Success Criteria

- [x] Verified `calculate_rmse` is not used in `news.py`
- [ ] Removed unused import
- [ ] All syntax checks pass
- [ ] All imports work correctly
- [ ] Code is cleaner

---

## Notes

- This is a **small, focused cleanup** similar to Iteration 15
- Only removes unused import (no functional changes)
- Maintains the pattern of incremental cleanup improvements
- Low effort, low risk, small benefit
- If `calculate_rmse` is actually used, skip this iteration
