# Refactoring Plan - Iteration 5

**Date**: 2025-01-11  
**Focus**: Move `_safe_mean_std()` from `dfm.py` to `core/helpers/estimation.py`  
**Scope**: Small, focused, reversible change

---

## Objective

Move the `_safe_mean_std()` helper function from `dfm.py` to `core/helpers/estimation.py`. This continues the helper extraction pattern and reduces `dfm.py` size. This is a small, focused step toward consolidating helper functions.

---

## Current Situation

**File**: `src/dfm_python/dfm.py` (873 lines)

**Function to Move**:
- `_safe_mean_std()` (28 lines) - Data standardization utility
  - **Location**: `dfm.py:454`
  - **Usage**: Used once by `_standardize_data()` function
  - **Type**: Data standardization helper (computes mean/std with missing value handling)

**Target Location**:
- `core/helpers/estimation.py` (148 lines) - Estimation helpers
  - Already contains: `estimate_ar_coefficients_ols`, `compute_innovation_covariance`, `compute_sufficient_stats`
  - Good fit for data standardization functions used in estimation

**Assessment**: 
- Small, self-contained function
- Data standardization utility (fits in estimation helpers)
- Used only within `dfm.py` currently, but could be reused
- Low risk (simple function, clear purpose)

---

## Tasks

### Task 1: Add Function to `core/helpers/estimation.py`
- [x] Add `safe_mean_std()` function to `core/helpers/estimation.py`
- [x] Add function docstring (preserve existing documentation)
- [x] Export in `core/helpers/__init__.py` (add to imports and `__all__`)

### Task 2: Update `dfm.py`
- [x] Remove `_safe_mean_std()` function definition
- [x] Add import: `from .core.helpers import safe_mean_std`
- [x] Update usage in `_standardize_data()`: `_safe_mean_std(X)` → `safe_mean_std(X)`
- [x] Verify function call works correctly

### Task 3: Verify Nothing Breaks
- [x] Test imports: `from dfm_python.core.helpers import safe_mean_std`
- [x] Test DFM class: `from dfm_python import DFM`
- [x] Verify data standardization still works in `_standardize_data()`
- [x] Check that all internal usage still works

---

## Expected Outcome

- **Moved**: `safe_mean_std()` from `dfm.py` to `core/helpers/estimation.py`
- **Reduced**: `dfm.py` by ~28 lines (from 873 to ~845)
- **Result**: Better organization, function available for reuse
- **Risk**: Very low (simple function, clear purpose, used only once)

---

## Code Changes Preview

**Add to `core/helpers/estimation.py`**:
```python
def safe_mean_std(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation for each column, handling missing values.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (T x N)
        
    Returns
    -------
    means : np.ndarray
        Column means (N,)
    stds : np.ndarray
        Column standard deviations (N,)
        
    Notes
    -----
    - Handles missing values (NaN) by computing statistics only on finite values
    - Returns default values (mean=0, std=1) for columns with no valid data
    - Ensures std > 0 (minimum std = 1.0) to avoid division by zero
    """
    n_series = matrix.shape[1]
    means = np.zeros(n_series)
    stds = np.ones(n_series)
    for j in range(n_series):
        col = matrix[:, j]
        mask = np.isfinite(col)
        if np.any(mask):
            means[j] = float(np.nanmean(col[mask]))
            std_val = float(np.nanstd(col[mask]))
            stds[j] = std_val if std_val > 0 else 1.0
        else:
            means[j] = 0.0
            stds[j] = 1.0
    return means, stds
```

**Update `dfm.py`**:
- Remove: `def _safe_mean_std(...)` function definition
- Add: `from .core.helpers import safe_mean_std` (or import from estimation directly)
- Update: `Mx, Wx = _safe_mean_std(X)` → `Mx, Wx = safe_mean_std(X)`

**Update `core/helpers/__init__.py`**:
- Add `safe_mean_std` to imports from `.estimation`
- Add `safe_mean_std` to `__all__`

---

## Rollback Plan

If anything breaks:
1. Restore `_safe_mean_std()` function to `dfm.py`
2. Remove `safe_mean_std()` from `core/helpers/estimation.py`
3. Remove from `core/helpers/__init__.py`
4. Revert import changes in `dfm.py`
5. Verify imports work again

---

## Success Criteria

- [x] `safe_mean_std()` added to `core/helpers/estimation.py` (195 lines, up from 148)
- [x] `safe_mean_std()` exported in `core/helpers/__init__.py`
- [x] `_safe_mean_std()` removed from `dfm.py`
- [x] Usage updated in `_standardize_data()`
- [x] All imports work correctly
- [x] Data standardization still works
- [x] No functional changes
- [x] Code is cleaner and better organized

## Completion Status

✅ **COMPLETED** - All tasks finished successfully.

**Results**:
- Moved `safe_mean_std()` from `dfm.py` to `core/helpers/estimation.py`
- Reduced `dfm.py` by 31 lines (from 873 to 842)
- All imports verified and working
- Function now available for reuse
- No functional changes

---

## Notes

- This is a **small, incremental step** toward consolidating helpers
- Future iteration can move `_standardize_data()` to `core/helpers/estimation.py` (it uses `safe_mean_std()`)
- This change is **reversible** and **low risk**
- Function name changed from `_safe_mean_std()` to `safe_mean_std()` (removed `_` prefix since it's now a public helper)

---

## Why This Function

1. **Small**: Only 28 lines
2. **Clear purpose**: Data standardization utility
3. **Low risk**: Simple function, used only once
4. **Good fit**: Fits well in `estimation.py` with other estimation helpers
5. **Building block**: Used by `_standardize_data()`, which can be moved later

---

## Dependencies

- `_standardize_data()` in `dfm.py` uses this function
- After moving, `_standardize_data()` will import from `core.helpers.estimation`
- This sets up for future extraction of `_standardize_data()` itself
