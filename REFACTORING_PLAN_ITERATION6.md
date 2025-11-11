# Refactoring Plan - Iteration 6

**Date**: 2025-01-11  
**Focus**: Move `_standardize_data()` from `dfm.py` to `core/helpers/estimation.py`  
**Scope**: Small, focused, reversible change

---

## Objective

Move the `_standardize_data()` helper function from `dfm.py` to `core/helpers/estimation.py`. This completes the helper extraction from `dfm.py` and reduces the file size below 800 lines.

---

## Current Situation

**File**: `src/dfm_python/dfm.py` (842 lines)

**Function to Move**:
- `_standardize_data()` (58 lines) - Data standardization wrapper
  - **Location**: `dfm.py:454`
  - **Usage**: Used once by `_dfm_core()` function
  - **Type**: Data standardization helper (uses `safe_mean_std()` which is already moved)
  - **Dependencies**: Uses `safe_mean_std()` (already in `estimation.py` ✅), `_clean_matrix()` (from `core.numeric`)

**Target Location**:
- `core/helpers/estimation.py` (195 lines) - Estimation helpers
  - Already contains: `estimate_ar_coefficients_ols`, `compute_innovation_covariance`, `compute_sufficient_stats`, `safe_mean_std`
  - Good fit for data standardization functions used in estimation

**Assessment**: 
- Medium-sized, self-contained function
- Data standardization utility (fits in estimation helpers)
- Uses `safe_mean_std()` which is already in `estimation.py` ✅
- Low risk (function is self-contained, dependency already moved)

---

## Tasks

### Task 1: Add Function to `core/helpers/estimation.py`
- [ ] Add `standardize_data()` function to `core/helpers/estimation.py`
- [ ] Add necessary imports: `_clean_matrix` from `..numeric` (or import at module level)
- [ ] Add function docstring (preserve existing documentation)
- [ ] Export in `core/helpers/__init__.py` (add to imports and `__all__`)

### Task 2: Update `dfm.py`
- [ ] Remove `_standardize_data()` function definition
- [ ] Add import: `from .core.helpers import standardize_data`
- [ ] Update usage in `_dfm_core()`: `_standardize_data(...)` → `standardize_data(...)`
- [ ] Verify function call works correctly

### Task 3: Verify Nothing Breaks
- [ ] Test imports: `from dfm_python.core.helpers import standardize_data`
- [ ] Test DFM class: `from dfm_python import DFM`
- [ ] Verify data standardization still works in `_dfm_core()`
- [ ] Check that all internal usage still works

---

## Expected Outcome

- **Moved**: `standardize_data()` from `dfm.py` to `core/helpers/estimation.py`
- **Reduced**: `dfm.py` by ~58 lines (from 842 to ~784)
- **Result**: Better organization, function available for reuse, `dfm.py` < 800 lines ✅
- **Risk**: Low (function is self-contained, uses `safe_mean_std()` which is already moved)

---

## Code Changes Preview

**Add to `core/helpers/estimation.py`**:
```python
def standardize_data(
    X: np.ndarray,
    clip_data_values: bool,
    data_clip_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize data and handle missing values.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (T x N)
    clip_data_values : bool
        Whether to clip extreme values
    data_clip_threshold : float
        Threshold for clipping (in standard deviations)
        
    Returns
    -------
    x_standardized : np.ndarray
        Standardized data (T x N)
    Mx : np.ndarray
        Series means (N,)
    Wx : np.ndarray
        Series standard deviations (N,)
        
    Notes
    -----
    - Uses safe_mean_std() for robust mean/std computation
    - Handles NaN/Inf values via _clean_matrix()
    - Clips extreme values if enabled
    """
    from ..numeric import _clean_matrix  # Import here to avoid circular dependency
    
    Mx, Wx = safe_mean_std(X)
    
    # Handle zero/near-zero standard deviations
    min_std = 1e-6
    Wx = np.maximum(Wx, min_std)
    
    # Handle NaN standard deviations
    nan_std_mask = np.isnan(Wx) | np.isnan(Mx)
    if np.any(nan_std_mask):
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            f"Series with NaN mean/std detected: {np.sum(nan_std_mask)}. "
            f"Setting Wx=1.0, Mx=0.0 for these series."
        )
        Wx[nan_std_mask] = 1.0
        Mx[nan_std_mask] = 0.0
    
    # Standardize
    x_standardized = (X - Mx) / Wx
    
    # Clip extreme values if enabled
    if clip_data_values:
        n_clipped_before = np.sum(np.abs(x_standardized) > data_clip_threshold)
        x_standardized = np.clip(x_standardized, -data_clip_threshold, data_clip_threshold)
        if n_clipped_before > 0:
            import logging
            _logger = logging.getLogger(__name__)
            pct_clipped = 100.0 * n_clipped_before / x_standardized.size
            _logger.warning(
                f"Data value clipping applied: {n_clipped_before} values ({pct_clipped:.2f}%) "
                f"clipped beyond ±{data_clip_threshold} standard deviations."
            )
    
    # Replace any remaining NaN/Inf using consolidated utility
    default_inf_val = data_clip_threshold if clip_data_values else 100
    x_standardized = _clean_matrix(
        x_standardized,
        'general',
        default_nan=0.0,
        default_inf=default_inf_val
    )
    
    return x_standardized, Mx, Wx
```

**Update `dfm.py`**:
- Remove: `def _standardize_data(...)` function definition
- Add: `from .core.helpers import standardize_data` (or import from estimation directly)
- Update: `x_standardized, Mx, Wx = _standardize_data(...)` → `x_standardized, Mx, Wx = standardize_data(...)`

**Update `core/helpers/__init__.py`**:
- Add `standardize_data` to imports from `.estimation`
- Add `standardize_data` to `__all__`

---

## Rollback Plan

If anything breaks:
1. Restore `_standardize_data()` function to `dfm.py`
2. Remove `standardize_data()` from `core/helpers/estimation.py`
3. Remove from `core/helpers/__init__.py`
4. Revert import changes in `dfm.py`
5. Verify imports work again

---

## Success Criteria

- [ ] `standardize_data()` added to `core/helpers/estimation.py`
- [ ] `standardize_data()` exported in `core/helpers/__init__.py`
- [ ] `_standardize_data()` removed from `dfm.py`
- [ ] Usage updated in `_dfm_core()`
- [ ] All imports work correctly
- [ ] Data standardization still works
- [ ] No functional changes
- [ ] Code is cleaner and better organized
- [ ] `dfm.py` < 800 lines ✅

---

## Notes

- This **completes** helper extraction from `dfm.py` (all 3 helpers extracted)
- Function uses `safe_mean_std()` which is already in `estimation.py` ✅
- Function uses `_clean_matrix()` from `core.numeric` (need to import)
- This change is **reversible** and **low risk**
- Function name changes from `_standardize_data()` to `standardize_data()` (removed `_` prefix since it's now a public helper)

---

## Why This Function

1. **Completes extraction**: Last of 3 helpers from `dfm.py`
2. **Clear purpose**: Data standardization utility
3. **Low risk**: Function is self-contained, dependency already moved
4. **Good fit**: Fits well in `estimation.py` with `safe_mean_std()`
5. **Milestone**: Brings `dfm.py` below 800 lines

---

## Dependencies

- `safe_mean_std()` - Already in `estimation.py` ✅
- `_clean_matrix()` - From `core.numeric` (need to import in `estimation.py`)
- `logging` - For warnings (need to import logger in `estimation.py`)
