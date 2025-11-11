# Refactoring Plan - Iteration 1

## Objective
**Reduce file count from 44 → 38 (6 file reduction)** by consolidating small utility files into larger logical units.

## Scope
**ONE major improvement**: Consolidate small utility files into existing larger modules.

## Files to Consolidate

### 1. Merge `core/helpers/_common.py` → `core/helpers/utils.py`
- **Source**: `_common.py` (14 lines) - only contains `NUMERICAL_EXCEPTIONS` constant
- **Target**: `utils.py` (221 lines)
- **Impact**: -1 file
- **Dependencies to update**:
  - `core/helpers/estimation.py` (line 7)
  - `core/helpers/matrix.py` (line 8)
  - `core/helpers/utils.py` (line 6) - will become internal
- **Note**: `core/em/iteration.py` and `core/em/initialization.py` define their own `_NUMERICAL_EXCEPTIONS` - leave those as-is

### 2. Merge `core/numeric/utils.py` → `core/numeric/regularization.py`
- **Source**: `utils.py` (86 lines) - contains `_check_finite`, `_safe_divide`
- **Target**: `regularization.py` (282 lines)
- **Impact**: -1 file
- **Dependencies to update**:
  - `core/numeric/__init__.py` (lines 48-51) - update imports
  - All files importing from `core.numeric` that use `_check_finite` or `_safe_divide`
- **Note**: These are general utilities that fit well with regularization

### 3. Merge `core/numeric/clipping.py` → `core/numeric/regularization.py`
- **Source**: `clipping.py` (116 lines) - AR coefficient clipping functions
- **Target**: `regularization.py` (after merge #2)
- **Impact**: -1 file
- **Dependencies to update**:
  - `core/numeric/__init__.py` (lines 42-45) - update imports
  - All files importing clipping functions
- **Rationale**: Clipping is a form of regularization (ensuring stability)

### 4. Merge `data/config_loader.py` → `data/loader.py`
- **Source**: `config_loader.py` (143 lines) - config loading functions
- **Target**: `loader.py` (279 lines)
- **Impact**: -1 file
- **Dependencies to update**:
  - `data/__init__.py` (line 9) - update import
  - `data_loader.py` (line 14) - update import
  - `config_sources.py` (line 245) - update import
  - `api.py` (line 393) - update import
- **Note**: Both handle loading, so logical consolidation

### 5. Merge `core/helpers/config.py` → `core/helpers/utils.py`
- **Source**: `config.py` (67 lines) - `safe_get_method`, `safe_get_attr`
- **Target**: `utils.py` (after merge #1)
- **Impact**: -1 file
- **Dependencies to update**:
  - `core/helpers/__init__.py` (line 8) - update import
  - All files importing from `core.helpers` that use these functions
- **Note**: Small utility functions, fits well in utils

### 6. Merge `core/helpers/frequency.py` → `core/helpers/utils.py`
- **Source**: `frequency.py` (100 lines) - `get_tent_weights`, `infer_nQ`
- **Target**: `utils.py` (after merges #1 and #5)
- **Impact**: -1 file
- **Dependencies to update**:
  - `core/helpers/__init__.py` (line 31) - update import
  - All files importing frequency functions
- **Note**: Small utility functions, fits well in utils

## Execution Steps

### Step 1: Prepare workspace
1. Verify current file count: `find src -name "*.py" ! -name "__init__.py" | wc -l`
2. Create backup checkpoint (git status check)

### Step 2: Merge `_common.py` → `helpers/utils.py`
1. Read `_common.py` content
2. Add `NUMERICAL_EXCEPTIONS` constant to top of `utils.py`
3. Update `utils.py` to use internal constant (remove import)
4. Update `estimation.py` and `matrix.py` to import from `utils` instead of `_common`
5. Move `_common.py` to `trash/`
6. Verify no broken imports

### Step 3: Merge `numeric/utils.py` → `numeric/regularization.py`
1. Read `utils.py` content
2. Append functions to `regularization.py` (add section header)
3. Update `numeric/__init__.py` to import from `regularization` instead of `utils`
4. Check all files importing from `numeric` - update if needed
5. Move `utils.py` to `trash/`
6. Verify imports work

### Step 4: Merge `numeric/clipping.py` → `numeric/regularization.py`
1. Read `clipping.py` content
2. Append functions to `regularization.py` (add section header)
3. Update `numeric/__init__.py` to import from `regularization` instead of `clipping`
4. Check all files importing clipping functions - update if needed
5. Move `clipping.py` to `trash/`
6. Verify imports work

### Step 5: Merge `data/config_loader.py` → `data/loader.py`
1. Read `config_loader.py` content
2. Append functions to `loader.py` (add section header)
3. Update `data/__init__.py` to import from `loader` instead of `config_loader`
4. Update all files importing from `data.config_loader`:
   - `data_loader.py`
   - `config_sources.py`
   - `api.py`
5. Move `config_loader.py` to `trash/`
6. Verify imports work

### Step 6: Merge `helpers/config.py` → `helpers/utils.py`
1. Read `config.py` content
2. Append functions to `utils.py` (add section header)
3. Update `helpers/__init__.py` to import from `utils` instead of `config`
4. Check all files importing config helpers - update if needed
5. Move `config.py` to `trash/`
6. Verify imports work

### Step 7: Merge `helpers/frequency.py` → `helpers/utils.py`
1. Read `frequency.py` content
2. Append functions to `utils.py` (add section header)
3. Update `helpers/__init__.py` to import from `utils` instead of `frequency`
4. Check all files importing frequency helpers - update if needed
5. Move `frequency.py` to `trash/`
6. Verify imports work

### Step 8: Verification
1. Count files: `find src -name "*.py" ! -name "__init__.py" | wc -l` (should be 38)
2. Check for broken imports: `grep -r "from.*_common\|from.*numeric.*utils\|from.*numeric.*clipping\|from.*config_loader\|from.*helpers.*config\|from.*helpers.*frequency" src/`
3. Verify `__init__.py` files are updated correctly
4. Quick syntax check: `python -m py_compile` on modified files

## Risk Assessment

**Risk Level**: LOW
- Small, focused changes
- Clear dependencies
- Easy to verify
- Reversible (files moved to trash, not deleted)

**Potential Issues**:
- Circular imports (unlikely, but check)
- Missing imports after consolidation
- `__init__.py` not updated correctly

**Mitigation**:
- Test imports after each merge
- Keep deleted files in trash/ for easy recovery
- Update one file at a time, verify before proceeding

## Success Criteria

- [ ] File count reduced from 44 to 38
- [ ] All imports updated correctly
- [ ] No broken imports (verify with grep)
- [ ] All `__init__.py` files updated
- [ ] Deleted files moved to `trash/`
- [ ] Code compiles without syntax errors
- [ ] No functionality lost (functions preserved in target files)

## Next Iteration Preview

After this iteration:
- File count: 38 (target: 20, still need 18 more reductions)
- Next phase: Consolidate larger modules (array, block, validation helpers)
- Consider merging `data/utils.py` into `data/loader.py` or `data/transformer.py`

## Notes

- **DO NOT** run tests in this iteration (per user request)
- **DO NOT** create any new files
- **DO** move deleted files to `trash/` directory
- **DO** update all imports immediately after each merge
- **DO** verify file count after each merge step
