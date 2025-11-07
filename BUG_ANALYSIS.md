# Bug Analysis Report - dfm-python Package

## Date: 2025-11-07

### Summary
This document identifies potential bugs found during code inspection of the dfm-python package.

---

## Bug #1: A_temp Variable Scoping Issue (POTENTIAL BUG)

### Location
- **File:** `src/dfm_python/dfm.py`
- **Function:** `init_conditions()`
- **Lines:** 667-693

### Description
The variable `A_temp` is created inside a `try` block (line 676) but may persist across block iterations, similar to the `ff` bug that was just fixed.

### Code Context
```python
for i in range(n_b):  # Block loop
    # ...
    if len(idx_iM) > 0:
        # ...
        if Z_lag.shape[0] > 0 and Z_lag.shape[1] > 0:
            try:
                A_temp = inv(Z_lag.T @ Z_lag) @ Z_lag.T @ z  # Line 676
                A_i[:int(r_i), :int(r_i * p)] = A_temp.T
            except (np.linalg.LinAlgError, ValueError) as e:
                # Exception caught - A_temp NOT created
                A_i[:int(r_i), :int(r_i * p)] = 0.0
        
        # ...
        if len(z) > 0:
            e = z - Z_lag @ A_temp if 'A_temp' in locals() else z  # Line 693
```

### Problem
1. **Block 0**: If `try` succeeds, `A_temp` is created with dimensions for block 0
2. **Block 1**: If `try` fails (exception caught), `A_temp` is NOT created
3. **Line 693**: Checks `'A_temp' in locals()` - this returns `True` because `A_temp` from Block 0 still exists!
4. **Result**: Uses `A_temp` from Block 0 with `Z_lag` from Block 1 → **dimension mismatch or incorrect computation**

### Impact
- **Severity:** Medium
- **Likelihood:** Low (only occurs if one block's try succeeds and next block's try fails)
- **Affected:** Multi-block models where OLS regression fails for some blocks

### Suggested Fix
Reset `A_temp = None` at the start of each block iteration, similar to the `ff` fix:

```python
for i in range(n_b):
    r_i = int(r[i])
    ff = None
    A_temp = None  # <-- ADD THIS
    
    # ... rest of code ...
    
    if len(z) > 0:
        e = z - Z_lag @ A_temp if A_temp is not None else z  # <-- CHANGE CHECK
```

---

## Bug #2: bl_idxM and bl_idxQ Initialization Issue (POTENTIAL BUG)

### Location
- **File:** `src/dfm_python/dfm.py`
- **Function:** `em_step()`
- **Lines:** 1482-1517

### Description
`bl_idxM` and `bl_idxQ` are initialized as empty lists `[]` but then used with `np.hstack()` which expects numpy arrays.

### Code Context
```python
bl_idxM = []  # Line 1482 - Python list
bl_idxQ = []  # Line 1483 - Python list

for i in range(num_blocks):
    # ...
    if len(bl_idxM) == 0:  # Line 1504
        bl_idxM = bl_col_monthly  # numpy array
        bl_idxQ = bl_col_quarterly  # numpy array
    else:
        bl_idxM = np.hstack([bl_idxM, bl_col_monthly])  # Line 1508
        bl_idxQ = np.hstack([bl_idxQ, bl_col_quarterly])  # Line 1509
```

### Problem
- First iteration: `bl_idxM` is `[]` (list), then assigned numpy array → OK
- Second iteration: `bl_idxM` is numpy array, `np.hstack([bl_idxM, ...])` → OK
- **BUT**: If `num_blocks == 0` or loop doesn't execute, `bl_idxM` remains `[]` (list)
- Line 1516: `bl_idxM.astype(bool)` on a list → **AttributeError**

### Impact
- **Severity:** Low
- **Likelihood:** Very Low (only if `num_blocks == 0`, which shouldn't happen in practice)
- **Affected:** Edge case with empty block structure

### Suggested Fix
Initialize as empty numpy arrays instead of lists:

```python
bl_idxM = np.array([]).reshape(0, n_bl)  # Empty 2D array
bl_idxQ = np.array([]).reshape(0, n_bl)  # Empty 2D array
```

Or check before converting:

```python
if len(bl_idxM) > 0:
    bl_idxM = bl_idxM.astype(bool)
    bl_idxQ = bl_idxQ.astype(bool)
else:
    bl_idxM = np.array([]).reshape(0, n_bl).astype(bool)
    bl_idxQ = np.array([]).reshape(0, n_bl).astype(bool)
```

---

## Bug #3: F Variable May Be Uninitialized (POTENTIAL BUG)

### Location
- **File:** `src/dfm_python/dfm.py`
- **Function:** `init_conditions()`
- **Lines:** 547-559, 636

### Description
The variable `F` is created inside `if len(idx_iM) > 0:` block but may be referenced in the `else` branch.

### Code Context
```python
if len(idx_iM) > 0:
    # ...
    F = None  # Line 548
    max_lag = max(p + 1, pC)
    for kk in range(max_lag):
        lag_data = f[pC - kk:T - kk, :]
        if F is None:
            F = lag_data
        else:
            F = np.hstack([F, lag_data])
    # ...
    ff = F[:, :int(r_i * pC)]  # Line 566

# Later...
else:
    # ...
    if len(idx_iM) > 0 and 'f' in locals() and f is not None:
        F = np.hstack([f] + [np.roll(f, -kk, axis=0) for kk in range(1, max(p+1, pC))])  # Line 636
        ff = F[:, :int(r_i * pC)]
```

### Problem
The `else` branch at line 630 checks `if len(idx_iM) > 0` - but if we're in the `else` branch, it means the outer `if len(idx_iM) > 0:` (line 435) was False, so `len(idx_iM) == 0`. This condition can never be True!

### Impact
- **Severity:** Low (dead code)
- **Likelihood:** N/A (code never executes)
- **Affected:** None (unreachable code)

### Suggested Fix
Remove the unreachable condition or fix the logic:

```python
else:
    # If no monthly series, create dummy ff
    ff_padded_i = np.zeros((T, int(pC * r_i)))
    # Remove the unreachable if statement
```

---

## Bug #4: Dimension Mismatch in bl_idxQ_i (POTENTIAL BUG)

### Location
- **File:** `src/dfm_python/dfm.py`
- **Function:** `em_step()`
- **Lines:** 1700-1704

### Description
`bl_idxQ_i` is computed but may have incorrect dimensions when used to reshape empty arrays.

### Code Context
```python
if R_con.size > 0:
    bl_idxQ_i = np.where(bl_idxQ[i, :])[0]
    if len(bl_idxQ_i) > 0 and bl_idxQ_i.max() < R_con.shape[1]:
        # ...
    else:
        R_con_i = np.array([]).reshape(0, len(bl_idxQ_i))  # Line 1700
        q_con_i = np.array([])
else:
    bl_idxQ_i = np.where(bl_idxQ[i, :])[0]
    R_con_i = np.array([]).reshape(0, len(bl_idxQ_i))  # Line 1704
    q_con_i = np.array([])
```

### Problem
If `bl_idxQ_i` is empty (`len(bl_idxQ_i) == 0`), then `reshape(0, 0)` creates a 0x0 array, which may cause issues in subsequent operations expecting a specific shape.

### Impact
- **Severity:** Low
- **Likelihood:** Low (only if no quarterly series in block)
- **Affected:** Blocks with no quarterly series

### Suggested Fix
Add a check for empty `bl_idxQ_i`:

```python
if len(bl_idxQ_i) > 0:
    R_con_i = np.array([]).reshape(0, len(bl_idxQ_i))
else:
    R_con_i = np.array([]).reshape(0, 1)  # Default to 1 column
```

---

## Recommendations

### Priority Order
1. **Bug #1 (A_temp)**: Fix immediately - similar to the `ff` bug that was just fixed
2. **Bug #2 (bl_idxM/bl_idxQ)**: Fix for robustness - edge case but easy to fix
3. **Bug #3 (F variable)**: Clean up dead code
4. **Bug #4 (bl_idxQ_i)**: Low priority - may not cause issues in practice

### Testing Recommendations
1. Create test case for multi-block model where one block's OLS regression fails
2. Test edge case with `num_blocks == 0` (should be caught by validation, but test anyway)
3. Test blocks with no quarterly series to verify Bug #4 doesn't cause issues

---

## Notes
- All bugs found are potential issues that may not manifest in typical usage
- Bug #1 is the most critical as it's similar to the recently fixed `ff` scoping bug
- Most issues are related to variable scoping across loop iterations
- Consider adding more explicit variable resets at loop starts for safety

