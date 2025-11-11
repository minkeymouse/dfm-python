# Refactoring Priorities - Current State

**Date**: 2025-01-11  
**Status**: Post-Iteration 6  
**Purpose**: Prioritized list of remaining refactoring opportunities

---

## Executive Summary

After 6 iterations, the codebase is **clean and well-organized**. Remaining opportunities focus on **file organization** rather than critical issues.

**Completed**:
- ✅ Removed 2252 lines of duplicate dead code
- ✅ Extracted validation from `config.py`
- ✅ Extracted 3 helpers from `dfm.py` (89 lines)
- ✅ `dfm.py` reduced from 873 to 784 lines (< 800 lines ✅)

**Remaining**: 2 MEDIUM priority tasks, 2 LOW priority tasks

---

## Prioritized Recommendations

### MEDIUM Priority

#### 1. Split `data_loader.py` into `data/` package
**File**: `src/dfm_python/data_loader.py` (783 lines)  
**Impact**: High (improves organization, maintainability)  
**Effort**: Medium (requires careful import updates)  
**Risk**: Low (functions are self-contained)

**Current Structure**:
```
data_loader.py (783 lines):
├── Config Loading (lines 39-176)
│   ├── load_config_from_yaml()
│   ├── _load_config_from_dataframe()
│   ├── load_config_from_spec()
│   └── load_config()
├── Data Transformation (lines 179-243)
│   ├── _transform_series()
│   └── transform_data()
├── Data Loading (lines 316-410)
│   ├── read_data()
│   └── sort_data()
├── Main Load Function (lines 455-577)
│   └── load_data()
├── NaN Handling (lines 579-690)
│   └── rem_nans_spline()
└── Utilities (lines 690-783)
    └── summarize()
```

**Proposed Structure**:
```
data/
├── __init__.py          # Public API (re-export main functions)
├── loader.py            # Data loading (~200 lines)
│   ├── read_data()
│   ├── sort_data()
│   └── load_data()
├── transformer.py       # Data transformation (~100 lines)
│   ├── _transform_series()
│   └── transform_data()
├── config_loader.py     # Config loading (~150 lines)
│   ├── load_config_from_yaml()
│   ├── _load_config_from_dataframe()
│   ├── load_config_from_spec()
│   └── load_config()
└── utils.py             # Utilities (~200 lines)
    ├── rem_nans_spline()
    └── summarize()
```

**Benefits**:
- Clear separation of concerns
- Easier to navigate and maintain
- Matches MATLAB's separation pattern (`load_data.m`, `remNaNs_spline.m`, `summarize.m`)
- Reduces file size (783 → ~200 lines per file)

**Implementation Plan**:
1. Create `data/` package structure
2. Move functions incrementally (one module at a time)
3. Update imports in dependent modules
4. Test after each move

---

#### 2. Consider separating `config.py` models from factory methods
**File**: `src/dfm_python/config.py` (878 lines)  
**Impact**: Medium (improves organization)  
**Effort**: Medium (requires import updates)  
**Risk**: Low (clear separation)

**Current Structure**:
```
config.py (878 lines):
├── Dataclasses (lines 58-297)
│   ├── BlockConfig
│   ├── SeriesConfig
│   ├── Params
│   └── DFMConfig
├── Factory Methods (lines 400-878)
│   ├── from_yaml()
│   ├── from_dict()
│   ├── from_spec()
│   └── Various helper methods
```

**Proposed Structure**:
```
Option A: Keep in config.py, move factory to config_sources.py
├── config.py (dataclasses only, ~300 lines)
└── config_sources.py (factory methods, ~600 lines)

Option B: Create config/ package
├── config/
│   ├── __init__.py      # Re-export public API
│   ├── models.py        # Dataclasses (~300 lines)
│   └── factory.py       # Factory methods (~600 lines)
```

**Benefits**:
- Clearer separation between models and factory methods
- Easier to navigate
- Reduces file size (878 → ~300-600 lines per file)

**Recommendation**: **Option A** (simpler, less disruptive)

---

### LOW Priority

#### 3. Monitor `news.py` for future splitting
**File**: `src/dfm_python/news.py` (783 lines)  
**Impact**: Low (current structure is acceptable)  
**Effort**: Low (only if file grows)  
**Risk**: None (monitoring only)

**Current Structure**:
```
news.py (783 lines):
├── Helper Functions (lines 35-139)
│   ├── _check_config_consistency()
│   └── para_const()
├── Main Functions (lines 140-483)
│   └── news_dfm()
└── High-level API (lines 484-783)
    └── update_nowcast()
```

**Assessment**: 
- Well-organized structure
- Functions are logically grouped
- No immediate need to split

**Action**: 
- Monitor if `news.py` grows beyond 800 lines
- Consider splitting only if it becomes hard to maintain

---

#### 4. Consider parameter grouping for `_dfm_core()`
**File**: `src/dfm_python/dfm.py`  
**Impact**: Low (improves readability)  
**Effort**: Low (refactoring only)  
**Risk**: Low (internal function)

**Current Situation**:
- `_dfm_core()` has 15+ parameters
- Could use dataclass for parameter group

**Proposed Structure**:
```python
@dataclass
class DFMCoreParams:
    """Parameters for _dfm_core() function."""
    X: np.ndarray
    config: DFMConfig
    threshold: Optional[float] = None
    max_iter: Optional[int] = None
    # ... other parameters

def _dfm_core(params: DFMCoreParams) -> DFMResult:
    """Core DFM estimation with grouped parameters."""
    ...
```

**Benefits**:
- Improves readability
- Easier to pass parameters
- Better type hints

**Recommendation**: **LOW Priority** - Only if readability becomes an issue

---

## Summary Table

| Priority | Task | File | Lines | Impact | Effort | Risk |
|----------|------|------|-------|--------|--------|------|
| **MEDIUM** | Split `data_loader.py` | `data_loader.py` | 783 | High | Medium | Low |
| **MEDIUM** | Separate `config.py` models | `config.py` | 878 | Medium | Medium | Low |
| **LOW** | Monitor `news.py` | `news.py` | 783 | Low | Low | None |
| **LOW** | Parameter grouping | `dfm.py` | 784 | Low | Low | Low |

---

## Implementation Strategy

### For MEDIUM Priority Tasks

**Approach**: Incremental, one module at a time
1. **Plan**: Create detailed refactoring plan
2. **Execute**: Move functions incrementally
3. **Test**: Verify after each move
4. **Consolidate**: Document changes

**Principles**:
- Small, reversible changes
- Test after each step
- Maintain backward compatibility
- Update imports carefully

### For LOW Priority Tasks

**Approach**: Monitor and evaluate
1. **Monitor**: Track file sizes and complexity
2. **Evaluate**: Assess if splitting improves maintainability
3. **Act**: Only if benefits outweigh costs

---

## Next Steps

### Immediate (Next Iteration)
1. Assess `data_loader.py` structure in detail
2. Create focused refactoring plan for splitting
3. Begin incremental splitting (one module at a time)

### Short-term
1. Continue incremental refactoring
2. Monitor file sizes
3. Document patterns

### Long-term
1. Maintain clean structure
2. Prevent accumulation of helpers in large modules
3. Keep separation of concerns clear

---

## Notes

- **No critical issues** remaining after iterations 1-6
- **Focus on organization** rather than critical fixes
- **Incremental approach** has been successful
- **Code quality is good** - remaining work is improvement, not necessity
