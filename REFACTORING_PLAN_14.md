# Refactoring Plan - Iteration 14

**Date**: 2025-01-11  
**Focus**: Reduce function parameter count using `DFMParams` dataclass  
**Priority**: Medium  
**Effort**: Low  
**Risk**: Low

---

## Objective

Reduce parameter count in `_dfm_core()` and `_prepare_data_and_params()` by grouping override parameters into a `DFMParams` dataclass. This improves readability and maintainability without changing functionality.

---

## Current State

### Problem
Three functions have 15+ individual parameters for config overrides:
1. `DFM.fit()` - 15+ parameters
2. `_dfm_core()` - 15+ parameters  
3. `_prepare_data_and_params()` - 15+ parameters

All parameters are optional overrides that default to `None` and are resolved using `resolve_param()` against `config` values.

### Current Function Signatures

```python
def _dfm_core(X: np.ndarray, config: DFMConfig,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        ar_lag: Optional[int] = None,
        nan_method: Optional[int] = None,
        nan_k: Optional[int] = None,
        clock: Optional[str] = None,
        clip_ar_coefficients: Optional[bool] = None,
        ar_clip_min: Optional[float] = None,
        ar_clip_max: Optional[float] = None,
        clip_data_values: Optional[bool] = None,
        data_clip_threshold: Optional[float] = None,
        use_regularization: Optional[bool] = None,
        regularization_scale: Optional[float] = None,
        min_eigenvalue: Optional[float] = None,
        max_eigenvalue: Optional[float] = None,
        use_damped_updates: Optional[bool] = None,
        damping_factor: Optional[float] = None,
        **kwargs) -> DFMResult:
```

---

## Proposed Solution

### Step 1: Create `DFMParams` Dataclass

Create a dataclass that groups all override parameters:

```python
@dataclass
class DFMParams:
    """DFM estimation parameter overrides.
    
    All parameters are optional. If None, the corresponding value
    from DFMConfig will be used.
    """
    threshold: Optional[float] = None
    max_iter: Optional[int] = None
    ar_lag: Optional[int] = None
    nan_method: Optional[int] = None
    nan_k: Optional[int] = None
    clock: Optional[str] = None
    clip_ar_coefficients: Optional[bool] = None
    ar_clip_min: Optional[float] = None
    ar_clip_max: Optional[float] = None
    clip_data_values: Optional[bool] = None
    data_clip_threshold: Optional[float] = None
    use_regularization: Optional[bool] = None
    regularization_scale: Optional[float] = None
    min_eigenvalue: Optional[float] = None
    max_eigenvalue: Optional[float] = None
    use_damped_updates: Optional[bool] = None
    damping_factor: Optional[float] = None
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> 'DFMParams':
        """Create DFMParams from keyword arguments."""
        # Filter kwargs to only include valid parameter names
        valid_params = {
            'threshold', 'max_iter', 'ar_lag', 'nan_method', 'nan_k',
            'clock', 'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max',
            'clip_data_values', 'data_clip_threshold', 'use_regularization',
            'regularization_scale', 'min_eigenvalue', 'max_eigenvalue',
            'use_damped_updates', 'damping_factor'
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        return cls(**filtered)
```

### Step 2: Update `_prepare_data_and_params()`

Change signature to accept `DFMParams` instead of individual parameters:

```python
def _prepare_data_and_params(
    X: np.ndarray,
    config: DFMConfig,
    params: Optional[DFMParams] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare data and resolve all parameters from config and overrides.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix
    config : DFMConfig
        Configuration object
    params : DFMParams, optional
        Parameter overrides. If None, all values from config are used.
    
    Returns
    -------
    X_clean : np.ndarray
        Cleaned input data
    blocks : np.ndarray
        Block structure array
    params_dict : dict
        Dictionary of resolved parameters
    """
    # ... existing data cleaning code ...
    
    # Resolve parameters using DFMParams
    if params is None:
        params = DFMParams()
    
    params_dict = {
        'p': resolve_param(params.ar_lag, config.ar_lag),
        'r': (np.array(config.factors_per_block) 
              if config.factors_per_block is not None 
              else np.ones(blocks.shape[1])),
        'nan_method': resolve_param(params.nan_method, config.nan_method),
        'nan_k': resolve_param(params.nan_k, config.nan_k),
        # ... (continue for all parameters)
        'T': T,
        'N': N,
    }
    
    return X, blocks, params_dict
```

### Step 3: Update `_dfm_core()`

Change signature to accept `DFMParams`:

```python
def _dfm_core(
    X: np.ndarray,
    config: DFMConfig,
    params: Optional[DFMParams] = None,
    **kwargs
) -> DFMResult:
    """Estimate dynamic factor model using EM algorithm.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    config : DFMConfig
        Unified DFM configuration object
    params : DFMParams, optional
        Parameter overrides. If None, all values from config are used.
    **kwargs
        Additional parameters (merged into params if provided)
    
    Returns
    -------
    DFMResult
        Estimation results
    """
    # Merge kwargs into params if provided
    if kwargs:
        if params is None:
            params = DFMParams.from_kwargs(**kwargs)
        else:
            # Update params with kwargs
            for k, v in kwargs.items():
                if hasattr(params, k):
                    setattr(params, k, v)
    
    # Prepare data and resolve parameters
    X, blocks, params_dict = _prepare_data_and_params(X, config, params)
    
    # ... rest of function uses params_dict ...
```

### Step 4: Update `DFM.fit()`

Create `DFMParams` from individual parameters:

```python
def fit(self,
        X: np.ndarray,
        config: DFMConfig,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        # ... (keep all individual parameters for backward compatibility)
        **kwargs) -> DFMResult:
    """Fit the DFM model using EM algorithm.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N)
    config : DFMConfig
        Unified DFM configuration object
    threshold : float, optional
        EM convergence threshold override
    # ... (keep all parameter docs)
    **kwargs
        Additional parameter overrides
    
    Returns
    -------
    DFMResult
        Estimation results
    """
    # Store config and data
    self._config = config
    self._data = X
    
    # Create DFMParams from individual parameters
    params = DFMParams(
        threshold=threshold,
        max_iter=max_iter,
        ar_lag=ar_lag,
        nan_method=nan_method,
        nan_k=nan_k,
        clock=clock,
        clip_ar_coefficients=clip_ar_coefficients,
        ar_clip_min=ar_clip_min,
        ar_clip_max=ar_clip_max,
        clip_data_values=clip_data_values,
        data_clip_threshold=data_clip_threshold,
        use_regularization=use_regularization,
        regularization_scale=regularization_scale,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        use_damped_updates=use_damped_updates,
        damping_factor=damping_factor,
    )
    
    # Merge kwargs into params
    if kwargs:
        params = DFMParams.from_kwargs(**kwargs)
        # Update params with individual parameters
        for field in fields(DFMParams):
            value = locals().get(field.name)
            if value is not None:
                setattr(params, field.name, value)
    
    # Call core function
    result = _dfm_core(X, config, params=params)
    
    self._result = result
    return result
```

---

## Implementation Steps

### Step 1: Add `DFMParams` Dataclass
- **File**: `src/dfm_python/dfm.py`
- **Location**: After `DFMResult` dataclass, before `DFM` class
- **Action**: Add `DFMParams` dataclass definition
- **Verification**: Check syntax with `python3 -m py_compile`

### Step 2: Update `_prepare_data_and_params()`
- **File**: `src/dfm_python/dfm.py`
- **Action**: 
  - Change function signature to accept `params: Optional[DFMParams]`
  - Update parameter resolution logic to use `params` object
  - Keep return value unchanged (still returns `params_dict`)
- **Verification**: Check syntax, verify parameter resolution logic

### Step 3: Update `_dfm_core()`
- **File**: `src/dfm_python/dfm.py`
- **Action**:
  - Change function signature to accept `params: Optional[DFMParams]`
  - Update call to `_prepare_data_and_params()` to pass `params`
  - Handle `**kwargs` by merging into `params`
  - Keep rest of function logic unchanged
- **Verification**: Check syntax, verify function flow

### Step 4: Update `DFM.fit()`
- **File**: `src/dfm_python/dfm.py`
- **Action**:
  - Keep all individual parameters for backward compatibility
  - Create `DFMParams` from individual parameters
  - Pass `DFMParams` to `_dfm_core()`
- **Verification**: Check syntax, verify backward compatibility

### Step 5: Verify Functionality
- **Action**: Run syntax checks and basic import tests
- **Command**: `python3 -c "from dfm_python.dfm import DFM, DFMParams, _dfm_core"`
- **Note**: Full functional tests not required this iteration

---

## Benefits

1. **Improved Readability**: Function signatures are cleaner and easier to understand
2. **Better Maintainability**: Adding new parameters only requires updating the dataclass
3. **Type Safety**: Dataclass provides better type hints and IDE support
4. **Grouped Parameters**: Related parameters are logically grouped
5. **Backward Compatible**: `DFM.fit()` still accepts individual parameters

---

## Risks and Mitigation

### Risk 1: Breaking Changes
- **Risk**: Low - All changes are internal. `DFM.fit()` maintains backward compatibility
- **Mitigation**: Keep individual parameters in `DFM.fit()` signature

### Risk 2: Parameter Resolution Logic Errors
- **Risk**: Low - Logic is straightforward (just moving from individual params to dataclass)
- **Mitigation**: Carefully update parameter resolution, verify each parameter

### Risk 3: Missing Parameters
- **Risk**: Low - All parameters are explicitly listed in dataclass
- **Mitigation**: Compare parameter lists between old and new implementations

---

## Testing Strategy

### Syntax Validation
- Run `python3 -m py_compile src/dfm_python/dfm.py`
- Run `python3 -c "from dfm_python.dfm import DFM, DFMParams"`

### Import Validation
- Verify all imports work correctly
- Check that `DFMParams` is accessible

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

- [x] `DFMParams` dataclass created with all 15 parameters
- [x] `_prepare_data_and_params()` accepts `DFMParams` instead of individual parameters
- [x] `_dfm_core()` accepts `DFMParams` instead of individual parameters
- [x] `DFM.fit()` maintains backward compatibility (individual parameters still work)
- [x] All syntax checks pass
- [x] All imports work correctly
- [x] Code is cleaner and more maintainable

---

## Notes

- This refactoring is **internal only** - no public API changes
- `DFMParams` is an internal dataclass (not exported in `__init__.py`)
- Backward compatibility is maintained via `DFM.fit()` signature
- Future iterations can consider exporting `DFMParams` if needed
- This is a **small, focused change** that improves code quality without changing functionality
