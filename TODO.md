# TODO: File Consolidation

## Current Status
- **File count: 15** (target: ≤20) ✓ **BELOW LIMIT**
- **Latest iteration**: Cleaned up outdated `data_loader` references in docstrings (2 files updated)
- **Previous iterations completed**: 
  - Merged `core/helpers/` package (3 files → 1)
  - Merged `core/numeric/` package (3 files → 1)
  - Merged `data/` package (3 files → 1)
  - Removed `data_loader.py` (backward compatibility wrapper, test updated)
  - Fixed circular import in `utils/__init__.py` (lazy loading)
  - **Merged test files** (5 files → 1): `test_api.py`, `test_dfm.py`, `test_factor.py`, `test_kalman.py`, `test_numeric.py` → `test/__init__.py`
  - **Merged `utils/aggregation.py` → `utils/__init__.py`** (17 → 16 files)
  - **Merged `core/diagnostics.py` → `core/__init__.py`** (16 → 15 files)

## ✅ COMPLETED: Clean Up Outdated Comments

### Goal
Remove outdated references to `data_loader` module in docstring examples. The `data_loader.py` file was removed in a previous iteration, and all functionality was moved to `data/` package. These comments were harmless but misleading.

### Scope
- **Files updated:** 2 files
  - `src/dfm_python/__init__.py` (line 38)
  - `src/dfm_python/dfm.py` (line 756)
- **Change type:** Documentation cleanup (comments/docstrings only)
- **Risk:** Very low (comments only, no code changes)
- **Reversibility:** Easy (just restore comments)

### Step-by-Step Plan

#### Step 1: Update `src/dfm_python/__init__.py`
- [x] Remove outdated comment: `# or for backward compatibility: from dfm_python.data_loader import load_data`
- [x] Keep the preferred import example: `from dfm_python.data import load_data  # Preferred import`
- [x] Verify docstring still makes sense after removal

#### Step 2: Update `src/dfm_python/dfm.py`
- [x] Remove outdated comment: `# or for backward compatibility: from dfm_python.data_loader import load_config, load_data`
- [x] Keep the preferred import example: `from dfm_python.data import load_config, load_data  # Preferred import`
- [x] Verify docstring still makes sense after removal

#### Step 3: Verify
- [x] Run syntax check: `python3 -m py_compile src/dfm_python/__init__.py src/dfm_python/dfm.py`
- [x] Verify imports still work: `python3 -c "from dfm_python import DFM, load_data; print('OK')"`
- [x] Check that docstrings are still clear and accurate

### ✅ Outcome Achieved
- Cleaner, more accurate documentation ✓
- No outdated references to removed modules ✓
- All functionality preserved (no code changes) ✓
- File count unchanged (15 files) ✓
- Syntax verified ✓
- Imports verified ✓

### Notes
- This was a documentation-only change
- No functional impact
- Improved code clarity and accuracy
- Safe and reversible

## Completed Iterations

### ✅ COMPLETED: Merge `core/diagnostics.py` → `core/__init__.py`
**Result:** File count reduced from 16 → 15 ✓

### ✅ COMPLETED: Merge `utils/aggregation.py` → `utils/__init__.py`
**Result:** File count reduced from 17 → 16 ✓

## Next Steps (Optional - file count already well below limit)

### Code Quality Improvements (Future)
- Review for any remaining code duplication
- Ensure consistent naming conventions across all modules
- Verify all imports are properly organized
- Check for unused functions/imports

### Notes
- Current file count (15) is well below 20-file limit
- All major consolidations completed
- Code structure is clean and well-organized
- All consolidations maintain backward compatibility through import path updates
