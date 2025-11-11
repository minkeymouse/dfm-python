# TODO: File Consolidation

## Current Status
- **File count: 17** (target: ≤20) ✓ **BELOW LIMIT**
- **Iteration completed**: 
  - Merged `core/helpers/` package (3 files → 1)
  - Merged `core/numeric/` package (3 files → 1)
  - Merged `data/` package (3 files → 1)
  - Removed `data_loader.py` (backward compatibility wrapper, test updated)
  - Fixed circular import in `utils/__init__.py` (lazy loading)
  - **Merged test files** (5 files → 1): `test_api.py`, `test_dfm.py`, `test_factor.py`, `test_kalman.py`, `test_numeric.py` → `test/__init__.py`

## Remaining Work (Optional - file count already below limit)

### Code Quality Improvements
- Review and clean up any remaining code duplication
- Ensure consistent naming conventions across modules
- Verify all imports are properly organized
- Check for unused functions/imports

### Notes
- All consolidations must maintain backward compatibility
- File count is now 17 (well below 20-file limit)
- All test files successfully consolidated into `test/__init__.py`
- Old test files moved to `trash/`
