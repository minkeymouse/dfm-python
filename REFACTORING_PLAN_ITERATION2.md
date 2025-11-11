# Refactoring Plan - Iteration 2

**Date**: 2025-01-11  
**Focus**: Remove duplicate dead code - `core/numeric.py`  
**Scope**: Small, reversible, single-file cleanup

---

## Objective

Remove the duplicate `core/numeric.py` file (1052 lines) which is dead code. The new modular structure `core/numeric/` package already exists and is being used. This is a cleanup task with zero functional impact.

---

## Current Situation

- **Old file**: `src/dfm_python/core/numeric.py` (1052 lines) - dead code, shadowed by package
- **New structure**: `src/dfm_python/core/numeric/` package with:
  - `__init__.py` - re-exports all functions
  - `matrix.py` - matrix operations
  - `covariance.py` - covariance/variance computation
  - `regularization.py` - regularization and PSD enforcement
  - `clipping.py` - AR coefficient clipping
  - `utils.py` - general utilities
- **Status**: Python imports from `numeric/__init__.py` (package), not `numeric.py` (module)
- **Verification**: All imports already work correctly with the package structure

---

## Tasks

### Task 1: Verify Package Structure is Complete
- [x] Verify `core/numeric/__init__.py` exports all required functions
- [x] Verify imports work correctly (Python uses package, not module)
- [x] Double-check no direct file references to `numeric.py`

### Task 2: Move Dead Code to Trash
- [x] Move `src/dfm_python/core/numeric.py` to `trash/core_numeric.py` (preserve for safety)
- [x] Verify file moved successfully (1052 lines preserved)

### Task 3: Verify Nothing Breaks
- [x] Test basic imports: `from dfm_python.core.numeric import ...`
- [x] Test via `core/__init__.py`: `from dfm_python.core import ...`
- [x] Test via main API: `from dfm_python.dfm import DFM`
- [x] Test EM imports (which depend on numeric): `from dfm_python.core.em import ...`
- [x] Test helper imports: `from dfm_python.core.helpers.matrix import ...`
- [x] Verify no import errors in existing code

---

## Expected Outcome

- **Removed**: 1052 lines of duplicate dead code
- **Result**: Cleaner codebase, no functional changes
- **Risk**: Very low (file is not being used, package structure is complete)

---

## Rollback Plan

If anything breaks:
1. Restore `trash/core_numeric.py` to `src/dfm_python/core/numeric.py`
2. Verify imports work again

---

## Success Criteria

- [x] Package structure is complete and working
- [x] Old file moved to trash/ (1052 lines preserved)
- [x] All imports still work
- [x] No functional changes

## Completion Status

✅ **COMPLETED** - All tasks finished successfully.

**Results**:
- Removed 1052 lines of duplicate dead code
- All imports verified and working
- File safely preserved in `trash/core_numeric.py`
- Zero functional impact

---

## Notes

- This completes the removal of duplicate dead code files
- Total removed: 2252 lines (1200 + 1052)
- Both `core/em.py` and `core/numeric.py` have been successfully removed
- Next iteration: Consider other cleanup opportunities (large files, consolidation)
