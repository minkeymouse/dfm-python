# Refactoring Plan - Iteration 10

**Date**: 2025-01-11  
**Focus**: Extract config loading functions from `data_loader.py` to `data/config_loader.py`  
**Scope**: Small, focused, reversible change - continues splitting `data_loader.py`

---

## Objective

Extract the config loading functions from `data_loader.py` to a new `data/config_loader.py` module. This continues the incremental splitting of the large `data_loader.py` file (439 lines) into a more organized `data/` package structure.

**Rationale**: These functions are:
- Related config loading functions (logical grouping)
- Self-contained (config loading logic)
- Clear, single-purpose functions (configuration loading)
- Matches MATLAB structure (`load_spec.m` is a separate file)
- Reduces `data_loader.py` by ~138 lines

---

## Current Situation

**File**: `src/dfm_python/data_loader.py` (439 lines)

**Functions to Extract**:
- `load_config_from_yaml()` (lines 39-67, ~30 lines including docstring)
  - **Location**: `data_loader.py:39`
  - **Usage**: Used by `load_config()` (internal)
  - **Type**: Config loading (YAML file loading)
  - **Dependencies**: `YamlSource` from `.config`
  - **Self-contained**: No dependencies on other `data_loader.py` functions

- `_load_config_from_dataframe()` (lines 75-122, ~50 lines including docstring)
  - **Location**: `data_loader.py:75`
  - **Usage**: Used by 2 modules:
    - `load_config_from_spec()` in `data_loader.py` (line 143)
    - `config_sources.py` (line 245)
    - `api.py` (line 393)
  - **Type**: Config loading helper (DataFrame to DFMConfig conversion)
  - **Dependencies**: `DFMConfig`, `SeriesConfig`, `BlockConfig` from `.config`, `pandas`
  - **Self-contained**: No dependencies on other `data_loader.py` functions

- `load_config_from_spec()` (lines 125-143, ~20 lines including docstring)
  - **Location**: `data_loader.py:125`
  - **Usage**: Used by `load_config()` (internal)
  - **Type**: Config loading (spec CSV file loading)
  - **Dependencies**: `_load_config_from_dataframe()`, `pandas`, `Path`
  - **Self-contained**: Uses `_load_config_from_dataframe()` which will be moved together

- `load_config()` (lines 146-176, ~40 lines including docstring)
  - **Location**: `data_loader.py:146`
  - **Usage**: Used by 3 modules:
    - `__init__.py` (exported, used in examples)
    - `dfm.py` (referenced in docstring)
    - Examples/tutorials
  - **Type**: Config loading (convenience function)
  - **Dependencies**: `DFMConfig`, `load_config_from_yaml()`, `load_config_from_spec()`
  - **Self-contained**: Uses other config loading functions which will be moved together

**Target Location**:
- `data/config_loader.py` (new file, ~150 lines including imports and docstring)
  - Part of `data/` package structure
  - Will contain all 4 config loading functions

**Assessment**: 
- Self-contained functions with clear purpose
- Low risk (functions are independent, only use standard libraries and config)
- Logical grouping (all are config loading functions)
- Justifies new module in `data/` package (necessary for organization)

---

## Tasks

### Task 1: Create `data/config_loader.py`
- [ ] Create `src/dfm_python/data/config_loader.py` with all 4 functions
- [ ] Add necessary imports: `pandas`, `DFMConfig`, `SeriesConfig`, `BlockConfig`, `YamlSource` from `..config`, `Path`
- [ ] Copy all functions with full docstrings
- [ ] Ensure functions can import from each other (same module)

### Task 2: Update `data/__init__.py`
- [ ] Add `load_config` to imports from `.config_loader`
- [ ] Add `load_config` to `__all__`
- [ ] Note: `_load_config_from_dataframe()` is private, don't export it

### Task 3: Update Imports in Dependent Modules
- [ ] Update `__init__.py`: If it imports `load_config`, update to import from `data.config_loader`
- [ ] Update `config_sources.py`: `from .data_loader import _load_config_from_dataframe` → `from .data.config_loader import _load_config_from_dataframe`
- [ ] Update `api.py`: `from .data_loader import _load_config_from_dataframe` → `from .data.config_loader import _load_config_from_dataframe`

### Task 4: Update `data_loader.py`
- [ ] Remove all 4 config loading function definitions (lines 39-176)
- [ ] Add backward compatibility imports: `from .data.config_loader import load_config, _load_config_from_dataframe`
- [ ] Update `load_config_from_spec()` usage if it's still referenced (it will be in config_loader.py)

### Task 5: Verify Nothing Breaks
- [ ] Test imports: `from dfm_python.data.config_loader import load_config`
- [ ] Test backward compatibility: `from dfm_python.data_loader import load_config` (if maintained)
- [ ] Verify functions work identically
- [ ] Check that all dependent modules can import correctly

---

## Expected Outcome

- **Moved**: All 4 config loading functions from `data_loader.py` to `data/config_loader.py`
- **Reduced**: `data_loader.py` by ~138 lines (from 439 to ~301)
- **Created**: New `data/config_loader.py` module (~150 lines)
- **Result**: Better organization, functions in proper location, continues `data/` package structure
- **Risk**: Low (functions are self-contained, dependencies are standard libraries and config)

---

## Code Changes Preview

**Create `src/dfm_python/data/config_loader.py`**:
```python
"""Configuration loading utilities for DFM estimation.

This module provides functions for loading DFM configuration from various
sources, including YAML files and spec CSV files.
"""

from pathlib import Path
from typing import Union
import pandas as pd

from ..config import DFMConfig, SeriesConfig, BlockConfig, YamlSource


def load_config_from_yaml(configfile: Union[str, Path]) -> DFMConfig:
    """Load model configuration from YAML file.
    
    [Full docstring from data_loader.py]
    """
    return YamlSource(configfile).load()


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load configuration from DataFrame (internal helper).
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged]
    ...


def load_config_from_spec(specfile: Union[str, Path]) -> DFMConfig:
    """Load configuration from spec CSV file.
    
    [Full docstring from data_loader.py]
    """
    specfile = Path(specfile)
    if not specfile.exists():
        raise FileNotFoundError(f"Spec file not found: {specfile}")
    
    df = pd.read_csv(specfile)
    return _load_config_from_dataframe(df)


def load_config(configfile: Union[str, Path, DFMConfig]) -> DFMConfig:
    """Load configuration from file or return DFMConfig directly.
    
    [Full docstring from data_loader.py]
    """
    # [Function body unchanged]
    ...
```

**Update `data/__init__.py`**:
```python
"""Data loading and transformation utilities for DFM estimation.

This package provides comprehensive data handling for Dynamic Factor Models,
organized into focused modules for better maintainability.
"""

from .utils import rem_nans_spline, summarize
from .transformer import transform_data
from .config_loader import load_config

__all__ = ['rem_nans_spline', 'summarize', 'transform_data', 'load_config']
```

**Update `data_loader.py`**:
- Remove: All 4 config loading function definitions (lines 39-176)
- Add: `from .data.config_loader import load_config, _load_config_from_dataframe` (for backward compatibility)

**Update dependent modules**:
- `config_sources.py`: `from .data.config_loader import _load_config_from_dataframe`
- `api.py`: `from .data.config_loader import _load_config_from_dataframe`
- `__init__.py`: Update if it imports `load_config` directly

---

## Rollback Plan

If anything breaks:
1. Restore all 4 functions to `data_loader.py`
2. Remove `data/config_loader.py`
3. Remove from `data/__init__.py`
4. Revert import changes in dependent modules
5. Verify imports work again

---

## Success Criteria

- [ ] All 4 functions added to `data/config_loader.py`
- [ ] `load_config` exported in `data/__init__.py`
- [ ] All functions removed from `data_loader.py`
- [ ] All imports updated in dependent modules
- [ ] Backward compatibility maintained (if applicable)
- [ ] All imports work correctly
- [ ] Functions work identically
- [ ] No functional changes
- [ ] Code is cleaner and better organized
- [ ] `data_loader.py` reduced by ~138 lines

---

## Notes

- This is the **fourth step** in splitting `data_loader.py` (783 lines → eventually ~200 lines per module)
- Functions are **self-contained** (only use standard libraries, config, and pandas)
- Continues **pattern** established in iterations 7-9
- New `data/config_loader.py` module is **justified** as necessary for organization (matches MATLAB structure)
- This change is **reversible** and **low risk**
- Function names remain unchanged (no change needed)
- `_load_config_from_dataframe()` is private (not exported, only used internally)

---

## Why These Functions Next

1. **Logical Grouping**: All 4 functions are config loading functions, belong together
2. **Self-contained**: Only use standard libraries, config, and pandas
3. **Clear purpose**: Single-purpose functions (configuration loading)
4. **Low risk**: Functions are independent, easy to verify
5. **Continues pattern**: Fourth step in splitting `data_loader.py`
6. **Matches MATLAB**: `load_spec.m` is a separate file
7. **Dependencies**: Functions use each other, so they should be moved together

---

## Dependencies

- `pandas` - Standard library ✅
- `Path` from `pathlib` - Standard library ✅
- `DFMConfig`, `SeriesConfig`, `BlockConfig` from `..config` - Package dependency ✅
- `YamlSource` from `..config` - Package dependency ✅
- No dependencies on other `data_loader.py` functions ✅

---

## Future Iterations

After this iteration:
- **Iteration 11**: Extract data loading functions to `data/loader.py`
- **Final**: Remove or repurpose `data_loader.py` (or keep as thin wrapper)

This incremental approach allows for careful verification at each step.
