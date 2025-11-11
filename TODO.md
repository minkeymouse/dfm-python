# TODO: File Consolidation

## Current Status
- **File count: 16** (target: ≤20) ✓ **BELOW LIMIT**
- **Iteration completed**: 
  - Merged `core/helpers/` package (3 files → 1)
  - Merged `core/numeric/` package (3 files → 1)
  - Merged `data/` package (3 files → 1)
  - Removed `data_loader.py` (backward compatibility wrapper, test updated)
  - Fixed circular import in `utils/__init__.py` (lazy loading)
  - **Merged test files** (5 files → 1): `test_api.py`, `test_dfm.py`, `test_factor.py`, `test_kalman.py`, `test_numeric.py` → `test/__init__.py`
  - **Merged `utils/aggregation.py` → `utils/__init__.py`** (17 → 16 files)

## Next Iteration (Optional - file count already below limit)

### Potential Consolidation: Merge `core/diagnostics.py` → `core/__init__.py`
**Impact:** Would save 1 file (16 → 15)

**Rationale:**
- `core/__init__.py` is 69 lines and already re-exports diagnostics functions
- `diagnostics.py` is 429 lines with 4 functions
- All imports use `from .core.diagnostics` or `from .core import ...`
- No circular dependencies
- Simple merge: move content into `core/__init__.py` and update imports

**Files to Update:**
- `dfm.py`: `from .core.diagnostics import ...` → `from .core import ...`
- `__init__.py`: `from .core.diagnostics import ...` → `from .core import ...`
- `core/__init__.py`: Already imports from diagnostics, just need to merge content

**Risk:** Low - straightforward merge, `core/__init__.py` already acts as re-export hub

### Notes
- Current file count (16) is well below 20-file limit
- All consolidations maintain backward compatibility through import path updates
- Code structure is clean and well-organized
