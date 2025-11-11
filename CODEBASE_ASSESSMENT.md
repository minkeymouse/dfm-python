# DFM-Python Codebase Assessment

## Current State

**File Count**: 27 files (excluding `__init__.py` and tutorials)  
**Target**: 20 files maximum  
**Reduction Needed**: 7 files (26% reduction)

## File Structure Analysis

### Main Package Files (8 files)
- `api.py` (420 lines) - API interface
- `config.py` (828 lines) - Configuration classes
- `config_sources.py` (558 lines) - Config source implementations
- `config_validation.py` (77 lines) - **MERGE TARGET**: Small validation functions
- `data_loader.py` (25 lines) - **MERGE/REMOVE TARGET**: Backward compatibility shim
- `dfm.py` (906 lines) - Main DFM class
- `kalman.py` (466 lines) - Kalman filter
- `news.py` (782 lines) - News decomposition

### Core/EM (3 files)
- `core/em/convergence.py` (72 lines) - **MERGE TARGET**: Small, used only in iteration
- `core/em/initialization.py` (615 lines) - EM initialization
- `core/em/iteration.py` (647 lines) - EM iteration

### Core/Helpers (3 files) - **GOOD STRUCTURE**
- `core/helpers/estimation.py` (433 lines) - Estimation + validation (already consolidated)
- `core/helpers/matrix.py` (448 lines) - Matrix + block operations (already consolidated)
- `core/helpers/utils.py` (562 lines) - General utilities (already consolidated)

### Core/Numeric (3 files) - **CONSOLIDATION OPPORTUNITY**
- `core/numeric/covariance.py` (272 lines) - **MERGE TARGET**: Covariance computation (matrix-related)
- `core/numeric/matrix.py` (335 lines) - Matrix operations
- `core/numeric/regularization.py` (478 lines) - Regularization, clipping, utilities

### Data (3 files) - **CONSOLIDATION OPPORTUNITY**
- `data/loader.py` (420 lines) - Data + config loading (already consolidated)
- `data/transformer.py` (148 lines) - Data transformation
- `data/utils.py` (222 lines) - **MERGE TARGET**: Data utilities (rem_nans_spline, summarize)

### Other (2 files)
- `core/diagnostics.py` (429 lines) - Diagnostics
- `utils/aggregation.py` (334 lines) - Aggregation utilities

### Tests (5 files)
- `test/test_api.py` (704 lines)
- `test/test_dfm.py` (853 lines)
- `test/test_factor.py` (118 lines) - **MERGE TARGET**: Small test file
- `test/test_kalman.py` (205 lines)
- `test/test_numeric.py` (478 lines)

## Prioritized Consolidation Plan

### High Priority (Must Do - 7 files to reach 20)

1. **Merge `core/em/convergence.py` → `core/em/iteration.py`**
   - **Size**: 72 lines
   - **Rationale**: Convergence is checked during iteration, logical grouping
   - **Impact**: -1 file
   - **Risk**: LOW

2. **Merge `config_validation.py` → `config.py`**
   - **Size**: 77 lines
   - **Rationale**: Validation is part of configuration logic
   - **Impact**: -1 file
   - **Risk**: LOW

3. **Merge `data/utils.py` → `data/loader.py`**
   - **Size**: 222 lines
   - **Rationale**: Data utilities (rem_nans_spline, summarize) are used with loading
   - **Impact**: -1 file
   - **Risk**: LOW

4. **Merge `core/numeric/covariance.py` → `core/numeric/matrix.py`**
   - **Size**: 272 lines
   - **Rationale**: Covariance computation is a matrix operation
   - **Impact**: -1 file
   - **Risk**: LOW-MEDIUM

5. **Remove or merge `data_loader.py`**
   - **Size**: 25 lines
   - **Rationale**: Backward compatibility shim, only used in 1 test file
   - **Options**: 
     - Update test to use `data.loader` directly
     - Or merge into `data/loader.py` as compatibility exports
   - **Impact**: -1 file
   - **Risk**: LOW (if updating test)

6. **Merge `test/test_factor.py` → `test/test_dfm.py`**
   - **Size**: 118 lines
   - **Rationale**: Factor tests are part of DFM functionality
   - **Impact**: -1 file
   - **Risk**: LOW

7. **Consider merging `core/numeric/matrix.py` → `core/numeric/regularization.py`**
   - **Size**: 335 lines → would create 813 line file
   - **Rationale**: All numeric operations in one place
   - **Impact**: -1 file
   - **Risk**: MEDIUM (large file, but manageable)

**Alternative for #7**: Keep `matrix.py` separate if file size becomes too large (>800 lines).

## Code Quality Observations

### Strengths
- ✅ Good consolidation already done in `core/helpers/`
- ✅ Clear section headers in merged files
- ✅ Consistent naming patterns
- ✅ Good documentation

### Issues
- ⚠️ Some small files that can be merged
- ⚠️ `data_loader.py` is just a compatibility shim (minimal value)
- ⚠️ Test files could be better organized

### Naming Consistency
- ✅ Functions use snake_case
- ✅ Private functions use `_` prefix
- ✅ Constants use UPPER_CASE
- ✅ Classes use PascalCase

## File Size Analysis

### Large Files (>500 lines)
- `dfm.py` (906) - Main class, acceptable
- `test_dfm.py` (853) - Main test, acceptable
- `config.py` (828) - Config classes, acceptable
- `news.py` (782) - News decomposition, acceptable
- `core/em/iteration.py` (647) - Will be 719 after merging convergence
- `core/em/initialization.py` (615) - Acceptable
- `core/helpers/utils.py` (562) - Acceptable
- `config_sources.py` (558) - Acceptable

### Medium Files (200-500 lines)
- `core/numeric/regularization.py` (478) - Will grow with merges
- `test_numeric.py` (478) - Acceptable
- `core/helpers/matrix.py` (448) - Acceptable
- `core/helpers/estimation.py` (433) - Acceptable
- `core/diagnostics.py` (429) - Acceptable
- `data/loader.py` (420) - Will be 642 after merging utils
- `api.py` (420) - Acceptable
- `kalman.py` (466) - Acceptable
- `test_api.py` (704) - Acceptable
- `core/numeric/matrix.py` (335) - Will be 607 after merging covariance
- `utils/aggregation.py` (334) - Acceptable
- `core/numeric/covariance.py` (272) - **TO BE MERGED**
- `data/utils.py` (222) - **TO BE MERGED**

### Small Files (<200 lines) - **MERGE TARGETS**
- `data/transformer.py` (148) - Acceptable
- `test_kalman.py` (205) - Acceptable
- `test_factor.py` (118) - **TO BE MERGED**
- `config_validation.py` (77) - **TO BE MERGED**
- `core/em/convergence.py` (72) - **TO BE MERGED**
- `data_loader.py` (25) - **TO BE REMOVED/MERGED**

## Consolidation Strategy

### Phase 1: Quick Wins (5 files)
1. `convergence.py` → `iteration.py`
2. `config_validation.py` → `config.py`
3. `data/utils.py` → `data/loader.py`
4. `test_factor.py` → `test_dfm.py`
5. Remove/update `data_loader.py`

**Result**: 27 → 22 files

### Phase 2: Numeric Consolidation (2 files)
6. `covariance.py` → `matrix.py`
7. Optionally: `matrix.py` → `regularization.py` (if file size acceptable)

**Result**: 22 → 20 files ✓

## Risk Assessment

**Low Risk**:
- Merging small files (<200 lines)
- Merging validation into config
- Merging convergence into iteration
- Merging test files

**Medium Risk**:
- Merging covariance into matrix (larger file)
- Merging all numeric into one file (very large file)

**Mitigation**:
- Keep file sizes <1000 lines where possible
- Use clear section headers
- Maintain logical grouping

## Recommendations

### Immediate Actions
1. **Start with Phase 1** (5 quick merges)
2. **Verify file count** after each merge
3. **Update imports** immediately
4. **Test syntax** after each merge

### Medium-term
1. **Complete Phase 2** (numeric consolidation)
2. **Review file sizes** - ensure no files >1000 lines
3. **Consider test organization** - may need further consolidation later

### Long-term
1. **Monitor file sizes** as code evolves
2. **Consider splitting** if any file exceeds 1000 lines
3. **Maintain consolidation** as new code is added

## Expected Outcome

After consolidation:
- **File count**: 20 files (meets target)
- **Largest files**: 
  - `dfm.py` (~906 lines)
  - `test_dfm.py` (~971 lines after merge)
  - `config.py` (~905 lines after merge)
  - `core/numeric/regularization.py` (~813 lines if fully merged)
- **Code organization**: Improved, related functionality grouped
- **Maintainability**: Better, fewer files to navigate

## Notes

- All merges preserve functionality
- Files moved to `trash/` (not deleted)
- Imports updated consistently
- Section headers used for clarity
- No functionality lost
