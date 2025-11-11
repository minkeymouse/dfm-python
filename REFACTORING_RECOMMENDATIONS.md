# Refactoring Recommendations - Prioritized

**Date**: 2025-01-11  
**Status**: Post-cleanup assessment

---

## Quick Summary

After removing duplicate dead code (2252 lines), the codebase is cleaner. Remaining opportunities focus on **file organization** and **helper consolidation**.

---

## Top 3 Recommendations (Prioritized)

### 1. ~~Split `config.py` (899 lines)~~ → ✅ PARTIALLY COMPLETE (Iteration 3)

**Status**: Validation functions extracted (878 lines remaining)

**Why**: Mixes dataclasses (models) with validation logic  
**Impact**: High (readability, separation of concerns)  
**Effort**: Low (mostly moving code)  
**Risk**: Low (imports maintained via `__init__.py`)

**Action**:
- Create `config/models.py` (dataclasses: BlockConfig, SeriesConfig, Params, DFMConfig)
- Create `config/validation.py` (validate_frequency, validate_transformation)
- Create `config/__init__.py` (re-export everything)
- Update imports

**Files Affected**: `config.py` → split into 3 files

---

### 2. Extract Helpers from `dfm.py` (878 lines) → MEDIUM PRIORITY

**Why**: Helper functions could be reused, reduces file size  
**Impact**: Medium (organization, potential reuse)  
**Effort**: Low (move 3 functions)  
**Risk**: Low (self-contained functions)

**Action**:
- Move `_resolve_param()` → `core/helpers/utils.py`
- Move `_safe_mean_std()` → `core/helpers/estimation.py` or new `core/helpers/data.py`
- Move `_standardize_data()` → `core/helpers/estimation.py` or new `core/helpers/data.py`

**Files Affected**: `dfm.py` (reduce by ~90 lines), `core/helpers/`

---

### 3. Split `data_loader.py` (783 lines) → MEDIUM PRIORITY

**Why**: Mixes config loading, data loading, transformation, utilities  
**Impact**: Medium (organization, separation of concerns)  
**Effort**: Medium (update imports across codebase)  
**Risk**: Medium (used by many modules)

**Action**:
- Create `data/loader.py` (read_data, load_data)
- Create `data/transformer.py` (transform_data, _transform_series)
- Create `data/utils.py` (rem_nans_spline, summarize)
- Create `data/config_loader.py` (or move to config/)
- Create `data/__init__.py` (re-export)

**Files Affected**: `data_loader.py` → split into 4-5 files

---

## File Size Summary

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `config.py` | 878 | ⚠️ LARGE | **Monitor** (validation extracted ✅) |
| `config_validation.py` | 77 | ✅ OK | Validation functions (new) |
| `dfm.py` | 878 | ⚠️ LARGE | **Extract helpers** (MEDIUM) |
| `news.py` | 783 | ⚠️ LARGE | **Monitor** (LOW) |
| `data_loader.py` | 783 | ⚠️ LARGE | **Split** (MEDIUM) |

---

## Code Quality Findings

### ✅ Strengths
- No duplicate code
- Well-organized core packages (em/, numeric/, helpers/)
- Consistent naming (snake_case, PascalCase)
- Good documentation

### ⚠️ Opportunities
- Large files mix concerns
- Some helpers could be consolidated
- Config mixes models with logic

---

## Next Steps

1. **Next Iteration**: Split `config.py` (high impact, low risk)
2. **Future**: Extract helpers from `dfm.py`
3. **Future**: Consider splitting `data_loader.py`

---

## Comparison with MATLAB

**MATLAB**: Monolithic `dfm.m` (1109 lines)  
**Python**: More modular, but some files still large  
**Recommendation**: Continue splitting while maintaining clear interfaces

---

## Estimated Effort

- **Phase 1** (config.py split): 1 iteration
- **Phase 2** (helpers + data_loader): 2 iterations
- **Total**: 3 iterations for major improvements
