# TODO: File Consolidation

## Current Status
- **File count: 22** (target: ≤20)
- **Iteration completed**: 
  - Merged `core/helpers/` package (3 files → 1)
  - Merged `core/numeric/` package (3 files → 1)
  - Merged `data/` package (3 files → 1)
  - Removed `data_loader.py` (backward compatibility wrapper, test updated)
  - Fixed circular import in `utils/__init__.py` (lazy loading)

## Remaining Consolidations (Priority Order)

### High Priority (Must do to reach ≤20)
1. **Consolidate test files** (5 files → 1-2)
   - Files: `test_api.py`, `test_dfm.py`, `test_factor.py`, `test_kalman.py`, `test_numeric.py`
   - Action: Merge into `test/__init__.py` or `test/test_all.py`
   - Impact: Saves 3-4 files (22 → 18-19)

### Notes
- All consolidations must maintain backward compatibility
- Update imports after each consolidation
- Verify imports work after each change
- Move deleted files to `trash/`
