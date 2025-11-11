# TODO: File Consolidation

## Current Status
- **File count: 28** (target: ≤20)
- **Iteration completed**: Merged `core/helpers/` package (3 files → 1)
- **Removed**: `data_loader.py` (backward compatibility wrapper, test updated)

## Remaining Consolidations (Priority Order)

### High Priority (Must do to reach ≤20)
1. **Merge `core/numeric/` package** (3 files → 1)
   - Files: `matrix.py` (~336 lines), `covariance.py` (~273 lines), `regularization.py` (~479 lines)
   - Action: Merge into `core/numeric/__init__.py`
   - Impact: Saves 2 files (28 → 26)

2. **Merge `data/` package** (3 files → 1)
   - Files: `loader.py` (~420 lines), `transformer.py` (~148 lines), `utils.py` (~222 lines)
   - Action: Merge into `data/__init__.py`
   - Impact: Saves 2 files (26 → 24)

### Medium Priority (Need 4 more files to reach ≤20)
3. **Consolidate test files** (5 files → 1-2)
   - Files: `test_api.py`, `test_dfm.py`, `test_factor.py`, `test_kalman.py`, `test_numeric.py`
   - Action: Merge into `test/__init__.py` or `test/test_all.py`
   - Impact: Saves 3-4 files (24 → 20-21)

### Notes
- All consolidations must maintain backward compatibility
- Update imports after each consolidation
- Verify imports work after each change
- Move deleted files to `trash/`
