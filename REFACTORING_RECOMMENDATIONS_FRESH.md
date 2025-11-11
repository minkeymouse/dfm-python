# Refactoring Recommendations - Fresh Assessment

**Date**: 2025-01-11  
**Status**: Post-assessment (after iterations 1-5)

---

## Top Priority: Extract Remaining Helper from `dfm.py`

### Status
- ✅ `resolve_param()` extracted (iteration 4)
- ✅ `safe_mean_std()` extracted (iteration 5)
- ⚠️ 1 more helper remaining

### What to Move

**1. `_standardize_data()` (58 lines)**
- **Location**: `dfm.py:454`
- **Move to**: `core/helpers/estimation.py`
- **Usage**: Used by `_dfm_core()`
- **Type**: Data standardization wrapper
- **Dependency**: Uses `safe_mean_std()` (already moved ✅)

### Impact
- **Reduces `dfm.py`**: From 842 to ~784 lines
- **Improves organization**: Helper in proper location
- **Enables reuse**: Function available to other modules
- **Risk**: Low (function is self-contained, dependency already moved)

### What to Keep in `dfm.py`
- `_prepare_data_and_params()` - DFM-specific
- `_prepare_aggregation_structure()` - DFM-specific
- `_dfm_core()` - Core DFM logic
- `_run_em_algorithm()` - DFM-specific

---

## Second Priority: Split `data_loader.py`

### Why
- `data_loader.py` is 783 lines (large)
- Mixes 4 concerns: config loading, data loading, transformation, utilities

### What to Split

**Current Functions**:
- Config loading (~150 lines): `load_config_from_yaml()`, `load_config_from_spec()`, `load_config()`, `_load_config_from_dataframe()`
- Data loading (~200 lines): `read_data()`, `load_data()`
- Transformation (~150 lines): `transform_data()`, `_transform_series()`
- Utilities (~280 lines): `rem_nans_spline()`, `summarize()`

**Proposed Structure**:
```
data/
├── __init__.py           # Re-export public API
├── loader.py             # read_data(), load_data()
├── transformer.py        # transform_data(), _transform_series()
├── utils.py              # rem_nans_spline(), summarize()
└── config_loader.py      # Config loading (or move to config/)
```

### Impact
- **Improves organization**: Separates concerns
- **Reduces file size**: Largest file becomes ~280 lines
- **Risk**: Medium (used by many modules, need import updates)

---

## File Size Summary

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `config.py` | 878 | ⚠️ LARGE | **Monitor** (validation extracted ✅) |
| `dfm.py` | 842 | ⚠️ LARGE | **Extract 1 helper** (2 done ✅) |
| `news.py` | 783 | ⚠️ LARGE | **Monitor** (LOW) |
| `data_loader.py` | 783 | ⚠️ LARGE | **Split** (MEDIUM) |

---

## Code Quality Findings

### ✅ Strengths
- No duplicate code
- Well-organized core packages (em/, numeric/, helpers/)
- Consistent naming (snake_case, PascalCase)
- Good documentation
- Helper extraction pattern established (2 of 3 complete)

### ⚠️ Opportunities
- 1 helper function in `dfm.py` could be consolidated
- `data_loader.py` mixes multiple concerns
- Some files are large but acceptable

---

## Next Steps

1. **Next Iteration**: Extract remaining helper from `dfm.py` (medium impact, low risk)
2. **Future**: Consider splitting `data_loader.py` (medium impact, medium risk)
3. **Ongoing**: Monitor file sizes as code evolves

---

## Comparison with MATLAB

**MATLAB**: Monolithic `dfm.m` (1109 lines)  
**Python**: More modular, but some files still large  
**Recommendation**: Continue incremental splitting while maintaining clear interfaces

---

## Estimated Effort

- **Extract 1 helper**: 1 iteration (low risk, low effort)
- **Split data_loader**: 1-2 iterations (medium risk, medium effort)
- **Total**: 1-3 iterations for remaining improvements
