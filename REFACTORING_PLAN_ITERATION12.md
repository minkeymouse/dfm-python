# Refactoring Plan - Iteration 12

**Date**: 2025-01-11  
**Iteration**: 12  
**Focus**: Extract Hydra registration code from `config.py` to `config_sources.py`

---

## Objective

Extract the Hydra ConfigStore registration code from `config.py` to `config_sources.py` to improve organization and reduce `config.py` size. This is a small, self-contained improvement that maintains functionality while improving code organization.

**Goal**: Move Hydra registration code (~40 lines) from `config.py` to `config_sources.py`, reducing `config.py` from 878 to ~838 lines.

---

## Current State

### `config.py` (878 lines)
- **Structure**:
  - Dataclasses (lines 58-494, ~437 lines) - BlockConfig, SeriesConfig, Params, DFMConfig
  - Factory Methods (lines 527-807, ~280 lines) - Class methods on DFMConfig
  - Hydra Registration (lines 810-849, ~40 lines) - ConfigStore registration
  - Re-exports (lines 852-878, ~27 lines) - Backward compatibility imports

### Hydra Registration Code (lines 810-849)
- **Content**: 
  - Conditional registration (only if Hydra available)
  - SeriesConfigSchema dataclass definition
  - DFMConfigSchema dataclass definition
  - ConfigStore registration calls
  - Error handling with warnings

- **Dependencies**: 
  - `HYDRA_AVAILABLE`, `ConfigStore` (from top of config.py)
  - `DFMConfig`, `SeriesConfig` (from config.py)
  - `warnings` module
  - `dataclasses` module

- **Usage**: 
  - Executed at module import time
  - Registers schemas with Hydra ConfigStore
  - Optional (only if Hydra is available)

---

## Proposed Changes

### Step 1: Move Hydra Registration to `config_sources.py`
- **Target**: `src/dfm_python/config_sources.py`
- **Action**: Add Hydra registration code at the end of the file
- **Dependencies**: Import `DFMConfig`, `SeriesConfig` from `..config`
- **Imports needed**: `warnings`, `dataclasses`, `List`, `Optional`, `Dict`, `Any`

### Step 2: Update `config.py`
- **Action**: Remove Hydra registration code (lines 810-849)
- **Action**: Remove unused imports if any (check `warnings`, `dataclasses`)
- **Result**: Reduce file size by ~40 lines

### Step 3: Verify Imports
- **Check**: Ensure `config_sources.py` can import from `config.py` without circular dependency
- **Note**: `config_sources.py` already imports from `config.py`, so this should be safe

---

## Detailed Implementation Steps

### Step 1: Add Hydra Registration to `config_sources.py`

**File**: `src/dfm_python/config_sources.py`

**Add at end of file** (after existing code):
```python
# ============================================================================
# Hydra ConfigStore Registration (optional - only if Hydra is available)
# ============================================================================

try:
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    ConfigStore = None

if HYDRA_AVAILABLE and ConfigStore is not None:
    try:
        cs = ConfigStore.instance()
        if cs is not None:
            from dataclasses import dataclass as schema_dataclass
            from typing import List, Optional
            
            @schema_dataclass
            class SeriesConfigSchema:
                """Schema for SeriesConfig validation in Hydra."""
                series_id: str
                series_name: str
                frequency: str
                units: str
                transformation: str
                category: str
                blocks: List[int]
            
            @schema_dataclass
            class DFMConfigSchema:
                """Schema for unified DFMConfig validation in Hydra."""
                series: List[SeriesConfigSchema]
                block_names: List[str]
                factors_per_block: Optional[List[int]] = None
                ar_lag: int = 1
                threshold: float = 1e-5
                max_iter: int = 5000
                nan_method: int = 2
                nan_k: int = 3
                clock: str = 'm'
            
            # Register schemas
            cs.store(group="dfm", name="base_dfm_config", node=DFMConfigSchema)
            cs.store(group="model", name="base_model_config", node=DFMConfigSchema)
            cs.store(name="dfm_config_schema", node=DFMConfigSchema)
            cs.store(name="model_config_schema", node=DFMConfigSchema)
            
    except Exception as e:
        import warnings
        warnings.warn(f"Could not register Hydra structured config schemas: {e}. "
                     f"Configs will still work via from_dict() but without schema validation.")
```

**Key points**:
- Import `DFMConfig`, `SeriesConfig` are not needed (schemas are independent)
- Add necessary imports: `warnings`, `dataclasses`, `List`, `Optional`
- Preserve exact logic and error handling
- Keep conditional execution (only if Hydra available)

### Step 2: Remove Hydra Registration from `config.py`

**File**: `src/dfm_python/config.py`

**Remove**:
- Lines 810-849 (Hydra registration code)
- Check if `warnings` import is still needed (used elsewhere in file)
- Check if `dataclasses` import is still needed (used for dataclass decorators)

**Result**: File reduced from 878 to ~838 lines

### Step 3: Verify No Circular Dependencies

**Check**:
- `config_sources.py` imports from `config.py` (already does)
- `config.py` does not import from `config_sources.py` (except re-exports at end)
- Hydra registration in `config_sources.py` does not create circular dependency

**Note**: This should be safe because:
- `config_sources.py` already imports from `config.py`
- Hydra registration code does not use `DFMConfig` or `SeriesConfig` classes directly
- Schemas are independent dataclasses

---

## Verification Steps

After implementation, verify:

1. **Syntax check**: `python3 -m py_compile src/dfm_python/config.py src/dfm_python/config_sources.py`
2. **Import test**: `python3 -c "from dfm_python.config import DFMConfig; print('OK')"`
3. **Hydra test** (if Hydra available): `python3 -c "from dfm_python.config_sources import *; print('OK')"`
4. **No circular dependency**: Verify imports work correctly

---

## Expected Impact

### File Size Changes
- **`config.py`**: 878 lines → ~838 lines (40 lines, 5% reduction)
- **`config_sources.py`**: 504 lines → ~544 lines (+40 lines)

### Functional Impact
- **Zero** - Hydra registration works identically
- **Location**: Moved from `config.py` to `config_sources.py`
- **Execution**: Still happens at module import time

### Organization Impact
- **Medium** - Hydra-related code now in `config_sources.py` (more logical location)
- **Separation**: Config models stay in `config.py`, Hydra integration in `config_sources.py`

---

## Risk Assessment

**Risk Level**: **LOW**

**Reasons**:
- Self-contained code block
- No functional changes
- Hydra registration is optional (graceful degradation)
- Low risk of breaking changes

**Mitigation**:
- Preserve exact logic and error handling
- Test imports after change
- Verify Hydra registration still works (if Hydra available)

---

## Rollback Plan

If issues arise:

1. **Revert `config_sources.py`**: Remove Hydra registration code
2. **Revert `config.py`**: Restore Hydra registration code (lines 810-849)
3. **Verify**: All imports work correctly

All changes are reversible and isolated.

---

## Success Criteria

- [ ] Hydra registration code moved to `config_sources.py`
- [ ] Hydra registration code removed from `config.py`
- [ ] `config.py` reduced by ~40 lines
- [ ] All imports work correctly
- [ ] No functional changes
- [ ] Syntax check passes
- [ ] Import tests pass

---

## Notes

- This is a **small, focused improvement** that improves organization
- Hydra registration is optional, so failures are non-critical
- Code is self-contained and easy to move
- Follows the pattern of keeping related code together (Hydra integration with other source adapters)
- Reduces `config.py` size slightly, making it more manageable
- Does not address the larger factory methods separation (that would be a larger task)
