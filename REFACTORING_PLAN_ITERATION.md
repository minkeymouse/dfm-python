# Refactoring Plan - Focused Iteration

## Objective
**Reduce file count from 27 → 25 (2 file reduction)** by consolidating two small, closely-related files into their logical parent modules.

## Scope
**ONE major improvement**: Merge small utility modules into their parent modules where they are primarily used.

## Current State
- **File Count**: 27 files (target: 20)
- **Reduction Needed**: 7 more files
- **This Iteration**: 2 files (focused, low-risk)

## Files to Consolidate

### 1. Merge `core/em/convergence.py` → `core/em/iteration.py`
- **Source**: `convergence.py` (72 lines)
  - Contains: `em_converged()` function
  - Constants: `MIN_LOG_LIKELIHOOD_DELTA`, `DAMPING`, `MAX_LOADING_REPLACE`
- **Target**: `iteration.py` (647 lines)
- **Impact**: -1 file
- **Rationale**: 
  - Convergence is checked during EM iteration
  - `iteration.py` already imports constants from `convergence.py` (line 112)
  - Logical grouping: convergence checking is part of iteration logic
- **Dependencies to update**:
  - `core/em/__init__.py` (line 9, 35-42) - update imports
  - `core/__init__.py` (line 10, 43) - update imports
  - `dfm.py` (line 46) - update import
  - `test_dfm.py` (line 479) - update import
- **Risk**: LOW - Small file, clear relationship, already partially integrated

### 2. Merge `config_validation.py` → `config.py`
- **Source**: `config_validation.py` (77 lines)
  - Contains: `validate_frequency()`, `validate_transformation()`
  - Constants: `_VALID_FREQUENCIES`, `_VALID_TRANSFORMATIONS`
- **Target**: `config.py` (828 lines)
- **Impact**: -1 file
- **Rationale**:
  - Validation is part of configuration logic
  - Only used in `config.py` (lines 32, 78, 123, 124, 278, 413)
  - Natural fit: validation functions belong with config classes
- **Dependencies to update**:
  - `config.py` (line 32) - remove import, functions become internal
- **Risk**: LOW - Small file, only used in one place, clear relationship

## Execution Steps

### Step 1: Merge `convergence.py` → `iteration.py`
1. Read `convergence.py` content
2. Append convergence function and constants to end of `iteration.py`
   - Add section header: `# ============================================================================`
   - Add section: `# EM convergence checking`
3. Update `iteration.py` to remove import from `convergence` (line 112)
4. Update `core/em/__init__.py`:
   - Change `from .convergence import em_converged` to `from .iteration import em_converged`
   - Update constants import
5. Update `core/__init__.py`:
   - Update import path
6. Update `dfm.py`:
   - Update import path
7. Update `test_dfm.py`:
   - Update import path
8. Move `convergence.py` to `trash/`
9. Verify no broken imports

### Step 2: Merge `config_validation.py` → `config.py`
1. Read `config_validation.py` content
2. Append validation functions to end of `config.py`
   - Add section header: `# ============================================================================`
   - Add section: `# Configuration validation functions`
3. Update `config.py` to remove import (line 32)
   - Functions become internal to config module
4. Move `config_validation.py` to `trash/`
5. Verify no broken imports

### Step 3: Verification
1. Count files: `find src -name "*.py" ! -name "__init__.py" | wc -l` (should be 25)
2. Check for broken imports: `grep -r "from.*convergence\|from.*config_validation" src/`
3. Verify `__init__.py` files updated correctly
4. Quick syntax check: `python -m py_compile` on modified files

## Risk Assessment

**Risk Level**: LOW
- Both files are small (<80 lines)
- Clear parent-child relationships
- Limited dependencies
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

- [ ] File count reduced from 27 to 25
- [ ] All imports updated correctly
- [ ] No broken imports (verify with grep)
- [ ] All `__init__.py` files updated
- [ ] Deleted files moved to `trash/`
- [ ] Code compiles without syntax errors
- [ ] No functionality lost (functions preserved in target files)
- [ ] Clear section headers in merged files

## Expected File Sizes After Merge

- `core/em/iteration.py`: 647 + 72 = **719 lines** (acceptable)
- `config.py`: 828 + 77 = **905 lines** (acceptable, still <1000)

## Next Iteration Preview

After this iteration:
- File count: 25 (target: 20, still need 5 more reductions)
- Next candidates:
  - `data/utils.py` → `data/loader.py` (222 lines)
  - `test/test_factor.py` → `test/test_dfm.py` (118 lines)
  - `data_loader.py` → remove/update (25 lines, compatibility shim)
  - `core/numeric/covariance.py` → `core/numeric/matrix.py` (272 lines)

## Notes

- **DO NOT** run tests in this iteration (per user request)
- **DO NOT** create any new files
- **DO** move deleted files to `trash/` directory
- **DO** update all imports immediately after each merge
- **DO** verify file count after each merge step
- **DO** use clear section headers in merged files
