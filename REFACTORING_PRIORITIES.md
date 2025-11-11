# Refactoring Priorities - Specific Recommendations

**Date**: 2025-01-11  
**Status**: Post-assessment (after iterations 1-3)

---

## Top Priority: Extract Helpers from `dfm.py` → ✅ IN PROGRESS

**Status**: `resolve_param()` extracted (iteration 4), 2 more remaining

### Why
- `dfm.py` is 873 lines (down from 878)
- Contains 2 more helper functions that could be reused
- Moving them improves organization and reduces file size

### What to Move

**1. ✅ `resolve_param()` (3 lines)** - **COMPLETED** (iteration 4)
- **Location**: Was `dfm.py:314`, now `core/helpers/utils.py:150`
- **Move to**: `core/helpers/utils.py` ✅
- **Usage**: Used 15 times in `_prepare_data_and_params()`
- **Type**: General utility pattern

**2. `_safe_mean_std()` (28 lines)**
- **Location**: `dfm.py:459`
- **Move to**: `core/helpers/estimation.py` (or new `core/helpers/data.py`)
- **Usage**: Used by `_standardize_data()`
- **Type**: Data standardization utility

**3. `_standardize_data()` (58 lines)**
- **Location**: `dfm.py:490`
- **Move to**: `core/helpers/estimation.py` (or new `core/helpers/data.py`)
- **Usage**: Used by `_dfm_core()`
- **Type**: Data standardization wrapper

### Impact
- **Reduces `dfm.py`**: From 878 to ~789 lines
- **Improves organization**: Helpers in proper location
- **Enables reuse**: Functions available to other modules
- **Risk**: Low (self-contained functions)

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
| `dfm.py` | 873 | ⚠️ LARGE | **Extract helpers** (IN PROGRESS ✅) |
| `config.py` | 878 | ⚠️ LARGE | **Monitor** (validation extracted ✅) |
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
- Helper functions in `dfm.py` could be consolidated
- `data_loader.py` mixes multiple concerns
- Some files are large but acceptable

---

## Next Steps

1. **Next Iteration**: Extract helpers from `dfm.py` (medium impact, low risk)
2. **Future**: Consider splitting `data_loader.py` (medium impact, medium risk)
3. **Ongoing**: Monitor file sizes as code evolves

---

## Comparison with MATLAB

**MATLAB**: Monolithic `dfm.m` (1109 lines)  
**Python**: More modular, but some files still large  
**Recommendation**: Continue incremental splitting while maintaining clear interfaces

---

## Estimated Effort

- **Extract helpers**: 1 iteration (low risk, low effort)
- **Split data_loader**: 1-2 iterations (medium risk, medium effort)
- **Total**: 2-3 iterations for remaining improvements
