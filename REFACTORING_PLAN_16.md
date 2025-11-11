# Refactoring Plan - Iteration 16

**Date**: 2025-01-11  
**Focus**: Reduce `_run_em_algorithm()` parameter count using `EMAlgorithmParams` dataclass  
**Priority**: Medium  
**Effort**: Low  
**Risk**: Low

---

## Objective

Reduce parameter count in `_run_em_algorithm()` by grouping all 23 parameters into an `EMAlgorithmParams` dataclass. This improves readability and maintainability, following the same pattern as `DFMParams` in Iteration 14.

---

## Current State

### Problem
`_run_em_algorithm()` has 23 individual parameters:
- Data: `y`, `y_est` (2 parameters)
- Model parameters: `A`, `C`, `Q`, `R`, `Z_0`, `V_0`, `r`, `p` (8 parameters)
- Structure parameters: `R_mat`, `q`, `nQ`, `i_idio`, `blocks`, `tent_weights_dict`, `clock`, `frequencies` (8 parameters)
- Config: `config` (1 parameter)
- Algorithm parameters: `threshold`, `max_iter`, `use_damped_updates`, `damping_factor` (4 parameters)

All parameters are required (no optional parameters).

### Current Function Signature

```python
def _run_em_algorithm(
    y: np.ndarray,
    y_est: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    r: np.ndarray,
    p: int,
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    nQ: int,
    i_idio: np.ndarray,
    blocks: np.ndarray,
    tent_weights_dict: Dict[str, np.ndarray],
    clock: str,
    frequencies: Optional[np.ndarray],
    config: DFMConfig,
    threshold: float,
    max_iter: int,
    use_damped_updates: bool,
    damping_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
```

---

## Proposed Solution

### Step 1: Create `EMAlgorithmParams` Dataclass

Create a dataclass that groups all parameters:

```python
@dataclass
class EMAlgorithmParams:
    """Parameters for EM algorithm execution.
    
    This dataclass groups all parameters required for running the EM algorithm,
    reducing function parameter count and improving readability.
    """
    # Data
    y: np.ndarray
    y_est: np.ndarray
    
    # Model parameters
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    r: np.ndarray
    p: int
    
    # Structure parameters
    R_mat: Optional[np.ndarray]
    q: Optional[np.ndarray]
    nQ: int
    i_idio: np.ndarray
    blocks: np.ndarray
    tent_weights_dict: Dict[str, np.ndarray]
    clock: str
    frequencies: Optional[np.ndarray]
    
    # Config and algorithm parameters
    config: DFMConfig
    threshold: float
    max_iter: int
    use_damped_updates: bool
    damping_factor: float
```

**Note**: Unlike `DFMParams`, all fields are required (no defaults) since `_run_em_algorithm()` requires all parameters.

### Step 2: Update `_run_em_algorithm()`

Change signature to accept `EMAlgorithmParams`:

```python
def _run_em_algorithm(
    params: EMAlgorithmParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Run EM algorithm until convergence.
    
    Parameters
    ----------
    params : EMAlgorithmParams
        All parameters required for EM algorithm execution
    
    Returns
    -------
    A, C, Q, R, Z_0, V_0 : np.ndarray
        Final parameter estimates
    loglik : float
        Final log-likelihood
    num_iter : int
        Number of iterations completed
    converged : bool
        Whether convergence was achieved
    """
    # Extract parameters from dataclass
    y = params.y
    y_est = params.y_est
    A = params.A
    C = params.C
    Q = params.Q
    R = params.R
    Z_0 = params.Z_0
    V_0 = params.V_0
    r = params.r
    p = params.p
    R_mat = params.R_mat
    q = params.q
    nQ = params.nQ
    i_idio = params.i_idio
    blocks = params.blocks
    tent_weights_dict = params.tent_weights_dict
    clock = params.clock
    frequencies = params.frequencies
    config = params.config
    threshold = params.threshold
    max_iter = params.max_iter
    use_damped_updates = params.use_damped_updates
    damping_factor = params.damping_factor
    
    # ... rest of function unchanged ...
```

### Step 3: Update Call Site in `_dfm_core()`

Update the call to `_run_em_algorithm()` to create `EMAlgorithmParams`:

```python
# Step 6: Run EM algorithm
em_params = EMAlgorithmParams(
    y=y,
    y_est=y_est,
    A=A,
    C=C,
    Q=Q,
    R=R,
    Z_0=Z_0,
    V_0=V_0,
    r=r,
    p=p,
    R_mat=R_mat,
    q=q,
    nQ=nQ,
    i_idio=i_idio,
    blocks=blocks,
    tent_weights_dict=tent_weights_dict,
    clock=clock,
    frequencies=frequencies,
    config=config,
    threshold=threshold,
    max_iter=max_iter,
    use_damped_updates=use_damped_updates,
    damping_factor=damping_factor,
)

A, C, Q, R, Z_0, V_0, loglik, num_iter, converged = _run_em_algorithm(em_params)
```

---

## Implementation Steps

### Step 1: Add `EMAlgorithmParams` Dataclass
- **File**: `src/dfm_python/dfm.py`
- **Location**: After `DFMParams` dataclass, before `DFM` class
- **Action**: Add `EMAlgorithmParams` dataclass definition
- **Verification**: Check syntax with `python3 -m py_compile`

### Step 2: Update `_run_em_algorithm()` Signature
- **File**: `src/dfm_python/dfm.py`
- **Action**: 
  - Change function signature to accept `params: EMAlgorithmParams`
  - Extract parameters from dataclass at start of function
  - Keep rest of function logic unchanged
- **Verification**: Check syntax, verify parameter extraction

### Step 3: Update Call Site in `_dfm_core()`
- **File**: `src/dfm_python/dfm.py`
- **Action**:
  - Create `EMAlgorithmParams` from individual parameters
  - Pass `EMAlgorithmParams` to `_run_em_algorithm()`
  - Keep return value handling unchanged
- **Verification**: Check syntax, verify function flow

### Step 4: Verify Functionality
- **Action**: Run syntax checks and basic import tests
- **Command**: `python3 -c "from dfm_python.dfm import EMAlgorithmParams, _run_em_algorithm"`
- **Note**: Full functional tests not required this iteration

---

## Benefits

1. **Improved Readability**: Function signature is cleaner and easier to understand
2. **Better Maintainability**: Adding new parameters only requires updating the dataclass
3. **Type Safety**: Dataclass provides better type hints and IDE support
4. **Grouped Parameters**: Related parameters are logically grouped
5. **Consistent Pattern**: Follows same pattern as `DFMParams` (Iteration 14)

---

## Risks and Mitigation

### Risk 1: Breaking Changes
- **Risk**: Low - All changes are internal. `_run_em_algorithm()` is a private function
- **Mitigation**: Function is only called from `_dfm_core()`, easy to update

### Risk 2: Parameter Extraction Logic Errors
- **Risk**: Low - Logic is straightforward (just extracting from dataclass)
- **Mitigation**: Carefully extract parameters, verify each parameter

### Risk 3: Missing Parameters
- **Risk**: Low - All parameters are explicitly listed in dataclass
- **Mitigation**: Compare parameter lists between old and new implementations

---

## Testing Strategy

### Syntax Validation
- Run `python3 -m py_compile src/dfm_python/dfm.py`
- Run `python3 -c "from dfm_python.dfm import EMAlgorithmParams, _run_em_algorithm"`

### Import Validation
- Verify all imports work correctly
- Check that `EMAlgorithmParams` is accessible

### Functional Testing (Future Iteration)
- Run existing tests to verify functionality unchanged
- Run tutorials to verify end-to-end workflow

---

## Rollback Plan

If issues are discovered:
1. Revert changes to `dfm.py` using git
2. All changes are in a single file, easy to revert
3. No external dependencies changed

---

## Success Criteria

- [x] `EMAlgorithmParams` dataclass created with all 23 parameters
- [x] `_run_em_algorithm()` accepts `EMAlgorithmParams` instead of individual parameters
- [x] Call site in `_dfm_core()` updated to create `EMAlgorithmParams`
- [x] All syntax checks pass
- [x] All imports work correctly
- [x] Code is cleaner and more maintainable

---

## Notes

- This refactoring is **internal only** - no public API changes
- `EMAlgorithmParams` is an internal dataclass (not exported in `__init__.py`)
- Follows same pattern as `DFMParams` (Iteration 14)
- This is a **small, focused change** that improves code quality without changing functionality
- All parameters are required (unlike `DFMParams` which had optional overrides)
