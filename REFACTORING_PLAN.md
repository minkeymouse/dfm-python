# Refactoring Plan - Iteration 1

**Date**: 2025-01-11  
**Focus**: Remove duplicate dead code - `core/em.py`  
**Scope**: Small, reversible, single-file cleanup

---

## Objective

Remove the duplicate `core/em.py` file (1200 lines) which is dead code. The new modular structure `core/em/` package already exists and is being used. This is a cleanup task with zero functional impact.

---

## Current Situation

- **Old file**: `src/dfm_python/core/em.py` (1200 lines) - dead code, shadowed by package
- **New structure**: `src/dfm_python/core/em/` package with:
  - `__init__.py` - re-exports all functions
  - `convergence.py` - `em_converged()`
  - `initialization.py` - `init_conditions()`
  - `iteration.py` - `em_step()`
- **Status**: Python imports from `em/__init__.py` (package), not `em.py` (module)
- **Verification**: All imports already work correctly with the package structure

---

## Tasks

### Task 1: Verify Package Structure is Complete
- [x] Verify `core/em/__init__.py` exports all required functions
- [x] Verify imports work correctly (Python uses package, not module)
- [x] Double-check no direct file references to `em.py`

### Task 2: Update Documentation/Comments
- [x] Update comment in `core/__init__.py` that mentions `em.py`
- [x] Verify no other documentation references the old file

### Task 3: Move Dead Code to Trash
- [x] Move `src/dfm_python/core/em.py` to `trash/core_em.py` (preserve for safety)
- [x] Verify file moved successfully (1200 lines preserved)

### Task 4: Verify Nothing Breaks
- [x] Test basic imports: `from dfm_python.core.em import em_converged, init_conditions, em_step`
- [x] Test via `core/__init__.py`: `from dfm_python.core import em_converged, init_conditions, em_step`
- [x] Test via main API: `from dfm_python.dfm import DFM`
- [x] Verify no import errors in existing code

---

## Expected Outcome

- **Removed**: 1200 lines of duplicate dead code
- **Result**: Cleaner codebase, no functional changes
- **Risk**: Very low (file is not being used, package structure is complete)

---

## Rollback Plan

If anything breaks:
1. Restore `trash/core_em.py` to `src/dfm_python/core/em.py`
2. Revert any documentation changes
3. Verify imports work again

---

## Success Criteria

- [x] Package structure is complete and working
- [x] Old file moved to trash/ (1200 lines preserved)
- [x] All imports still work
- [x] No functional changes
- [x] Documentation updated

## Completion Status

✅ **COMPLETED** - All tasks finished successfully.

**Results**:
- Removed 1200 lines of duplicate dead code
- All imports verified and working
- File safely preserved in `trash/core_em.py`
- Documentation updated
- Zero functional impact

---

## Notes

- This is a **cleanup task**, not a refactoring
- The new structure is already in place and working
- Removing dead code improves maintainability
- Next iteration: Remove `core/numeric.py` (similar cleanup)
