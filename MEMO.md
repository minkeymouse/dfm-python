# Design Decisions & Consolidation Assessment

## Current State (2025-01-XX)

**File Count: 17** (Target: ≤20) ✓ **BELOW LIMIT**

### File Size Analysis

**Large Files (Already Consolidated):**
- `core/helpers/__init__.py`: 1473 lines (consolidated from 3 files)
- `core/em/__init__.py`: 1253 lines (consolidated from 2 files)
- `core/numeric/__init__.py`: 1098 lines (consolidated from 3 files)
- `test/__init__.py`: 1234 lines (consolidated from 5 files)

**Core Modules (Reasonable Size):**
- `dfm.py`: 906 lines (core estimation logic)
- `config.py`: 904 lines (configuration dataclasses + validation)
- `news.py`: 782 lines (news decomposition)
- `data/__init__.py`: 776 lines (consolidated from 3 files)
- `config_sources.py`: 558 lines (config source adapters)
- `kalman.py`: 466 lines (Kalman filter/smoother)

**Small Files (Consolidation Opportunities):**
- `utils/aggregation.py`: 334 lines (standalone module)
- `core/diagnostics.py`: 429 lines (standalone module)
- `api.py`: 420 lines (high-level API wrapper)
- `utils/__init__.py`: 43 lines (thin re-export wrapper)
- `core/__init__.py`: 69 lines (re-export wrapper)

## Consolidation Opportunities

### Priority 1: Merge `utils/aggregation.py` → `utils/__init__.py`
**Impact:** Saves 1 file (17 → 16)

**Rationale:**
- `utils/__init__.py` is only 43 lines and just re-exports from `aggregation.py`
- `aggregation.py` is self-contained (334 lines, 5 functions/constants)
- All imports use `from .utils.aggregation` or `from .utils import ...`
- No circular dependencies
- Simple merge: move content into `utils/__init__.py` and update imports

**Dependencies to Update:**
- `dfm.py`: `from .utils.aggregation import ...` → `from .utils import ...`
- `core/em/__init__.py`: `from ...utils.aggregation import ...` → `from ...utils import ...`
- `core/helpers/__init__.py`: `from ...utils.aggregation import ...` → `from ...utils import ...`
- `data/__init__.py`: `from ..utils.aggregation import ...` → `from ..utils import ...`
- `config.py`: `from .utils.aggregation import ...` → `from .utils import ...`
- `core/__init__.py`: `from ..utils.aggregation import ...` → `from ..utils import ...`

**Risk:** Low - straightforward merge with clear import path updates

### Priority 2: Merge `core/diagnostics.py` → `core/__init__.py`
**Impact:** Saves 1 file (16 → 15 after Priority 1)

**Rationale:**
- `core/__init__.py` is only 69 lines and already re-exports diagnostics functions
- `diagnostics.py` is 429 lines with 4 functions
- All imports use `from .core.diagnostics` or `from .core import ...`
- No circular dependencies
- Simple merge: move content into `core/__init__.py` and update imports

**Dependencies to Update:**
- `dfm.py`: `from .core.diagnostics import ...` → `from .core import ...`
- `__init__.py`: `from .core.diagnostics import ...` → `from .core import ...`
- `core/__init__.py`: Already imports from diagnostics, just need to merge content

**Risk:** Low - straightforward merge, `core/__init__.py` already acts as re-export hub

## Code Quality Observations

### Strengths
1. **Good Consolidation History:** Large packages already consolidated (helpers, em, numeric, data, tests)
2. **Clear Module Boundaries:** Each module has distinct responsibility
3. **Consistent Naming:** Functions follow snake_case, classes PascalCase
4. **No Dead Code:** All modules are actively used

### Areas for Improvement
1. **Import Patterns:** Some modules use deep imports (`from ...utils.aggregation`) that could be simplified after consolidation
2. **File Organization:** Two small standalone modules (`aggregation.py`, `diagnostics.py`) that are natural candidates for consolidation
3. **Wrapper Files:** `utils/__init__.py` and `core/__init__.py` are thin wrappers - merging would eliminate wrapper overhead

## Recommended Action Plan

### Iteration 1: Merge `utils/aggregation.py` → `utils/__init__.py`
1. Move all content from `utils/aggregation.py` to `utils/__init__.py`
2. Update all imports across codebase (6 files)
3. Move `utils/aggregation.py` to `trash/`
4. Verify imports work correctly
5. **Result:** 17 → 16 files

### Iteration 2: Merge `core/diagnostics.py` → `core/__init__.py`
1. Move all content from `core/diagnostics.py` to `core/__init__.py`
2. Update all imports across codebase (3 files)
3. Move `core/diagnostics.py` to `trash/`
4. Verify imports work correctly
5. **Result:** 16 → 15 files

## Notes
- Both consolidations are low-risk, high-value
- File count would drop to 15 (well below 20 limit)
- No functionality changes, only structural improvements
- All consolidations maintain backward compatibility through import updates
