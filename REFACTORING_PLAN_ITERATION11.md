# Refactoring Plan - Iteration 11

**Date**: 2025-01-11  
**Iteration**: 11  
**Focus**: Complete `data_loader.py` split by extracting remaining data loading functions to `data/loader.py`

---

## Objective

Complete the incremental splitting of `data_loader.py` by extracting the final 3 data loading functions to `data/loader.py`. This completes the `data/` package structure and matches the MATLAB separation pattern.

**Goal**: Extract `read_data()`, `sort_data()`, and `load_data()` from `data_loader.py` to `data/loader.py`, reducing `data_loader.py` to a thin wrapper or empty file.

---

## Current State

### `data_loader.py` (300 lines)
- **Functions remaining**:
  - `read_data()` (lines 40-133, ~95 lines) - Read time series data from file
  - `sort_data()` (lines 136-176, ~40 lines) - Sort data columns to match config order
  - `load_data()` (lines 179-300, ~125 lines) - Load and transform time series data
- **Total to extract**: ~261 lines
- **Backward compatibility imports**: Already present for `rem_nans_spline`, `summarize`, `transform_data`, `load_config`

### `data/` Package Structure
```
data/
├── __init__.py          # Exports: rem_nans_spline, summarize, transform_data, load_config
├── utils.py             # rem_nans_spline, summarize (222 lines) ✅
├── transformer.py       # transform_data, _transform_series (148 lines) ✅
└── config_loader.py      # load_config, _load_config_from_dataframe, etc. (143 lines) ✅
```

### Dependencies
- `read_data()`: Uses `Path`, `pandas`, `numpy` (standard libraries)
- `sort_data()`: Uses `DFMConfig`, `numpy`, `logging`
- `load_data()`: Uses `read_data()`, `sort_data()`, `transform_data()` (from `data.transformer`), `FREQUENCY_HIERARCHY` (from `utils.aggregation`)

### Current Usage
- `api.py`: `from .data_loader import load_data as _load_data`
- `__init__.py`: Docstring example mentions `from dfm_python.data_loader import load_data`
- `dfm.py`: Docstring example mentions `from dfm_python.data_loader import load_data`
- `load_data()` is the main public function; `read_data()` and `sort_data()` are used internally by `load_data()`

---

## Proposed Changes

### Step 1: Create `data/loader.py`
- **New file**: `src/dfm_python/data/loader.py`
- **Content**: All 3 functions (`read_data()`, `sort_data()`, `load_data()`)
- **Imports needed**:
  - `logging`, `Path`, `List`, `Optional`, `Tuple`, `Union` (typing)
  - `numpy`, `pandas`
  - `DFMConfig` (from `..config`)
  - `transform_data` (from `.transformer`)
  - `FREQUENCY_HIERARCHY` (from `...utils.aggregation`)

### Step 2: Update `data/__init__.py`
- **Add**: `from .loader import load_data`
- **Update**: `__all__` to include `load_data`
- **Note**: `read_data()` and `sort_data()` are internal helpers, not exported

### Step 3: Update `data_loader.py`
- **Remove**: All 3 function definitions (lines 40-300, ~261 lines)
- **Add**: Backward compatibility import: `from .data.loader import load_data, read_data, sort_data`
- **Update**: Module docstring to reflect new structure
- **Result**: `data_loader.py` becomes a thin wrapper (~40 lines)

### Step 4: Update Dependent Modules
- **`api.py`**: Change `from .data_loader import load_data as _load_data` → `from .data.loader import load_data as _load_data`
- **`__init__.py`**: Update docstring example (if present)
- **`dfm.py`**: Update docstring example (if present)

---

## Detailed Implementation Steps

### Step 1: Create `data/loader.py`

**File**: `src/dfm_python/data/loader.py`

**Structure**:
```python
"""Data loading functions for DFM estimation.

This module provides functions for reading, sorting, and loading time series data
for Dynamic Factor Model estimation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import DFMConfig
from .transformer import transform_data
from ...utils.aggregation import FREQUENCY_HIERARCHY

logger = logging.getLogger(__name__)


def read_data(datafile: Union[str, Path]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Read time series data from file.
    
    [Full docstring from data_loader.py]
    """
    # [Full implementation from data_loader.py, lines 40-133]
    ...


def sort_data(Z: np.ndarray, Mnem: List[str], config: DFMConfig) -> Tuple[np.ndarray, List[str]]:
    """Sort data columns to match configuration order.
    
    [Full docstring from data_loader.py]
    """
    # [Full implementation from data_loader.py, lines 136-176]
    ...


def load_data(datafile: Union[str, Path], config: DFMConfig,
              sample_start: Optional[Union[pd.Timestamp, str]] = None,
              sample_end: Optional[Union[pd.Timestamp, str]] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load and transform time series data for DFM estimation.
    
    [Full docstring from data_loader.py]
    """
    # [Full implementation from data_loader.py, lines 179-300]
    ...
```

**Key points**:
- Copy all 3 functions exactly as they are (no changes to logic)
- Update import paths:
  - `from .config import DFMConfig` → `from ..config import DFMConfig`
  - `from .data.transformer import transform_data` → `from .transformer import transform_data`
  - `from .utils.aggregation import FREQUENCY_HIERARCHY` → `from ...utils.aggregation import FREQUENCY_HIERARCHY`
- Preserve all docstrings and comments
- Preserve function signatures exactly

### Step 2: Update `data/__init__.py`

**File**: `src/dfm_python/data/__init__.py`

**Change**:
```python
# Before:
from .utils import rem_nans_spline, summarize
from .transformer import transform_data
from .config_loader import load_config

__all__ = ['rem_nans_spline', 'summarize', 'transform_data', 'load_config']

# After:
from .utils import rem_nans_spline, summarize
from .transformer import transform_data
from .config_loader import load_config
from .loader import load_data

__all__ = ['rem_nans_spline', 'summarize', 'transform_data', 'load_config', 'load_data']
```

**Note**: Only `load_data()` is exported. `read_data()` and `sort_data()` are internal helpers.

### Step 3: Update `data_loader.py`

**File**: `src/dfm_python/data_loader.py`

**Changes**:
1. Remove all 3 function definitions (lines 40-300)
2. Add backward compatibility imports
3. Update module docstring

**Result**:
```python
"""Data loading and transformation utilities for DFM estimation.

This module provides comprehensive data handling for Dynamic Factor Models.
For backward compatibility, functions are re-exported from the data package.

Note: This module is maintained for backward compatibility. New code should
import directly from dfm_python.data package.
"""

# Backward compatibility imports
from .data.loader import load_data, read_data, sort_data
from .data.utils import rem_nans_spline, summarize
from .data.transformer import transform_data
from .data.config_loader import load_config, _load_config_from_dataframe

# Re-export for backward compatibility
__all__ = [
    'load_data',
    'read_data',
    'sort_data',
    'rem_nans_spline',
    'summarize',
    'transform_data',
    'load_config',
    '_load_config_from_dataframe',
]
```

**Expected size**: ~30-40 lines (thin wrapper)

### Step 4: Update Dependent Modules

#### `api.py`
**File**: `src/dfm_python/api.py`

**Change**:
```python
# Before:
from .data_loader import load_data as _load_data

# After:
from .data.loader import load_data as _load_data
```

#### `__init__.py` (if docstring example exists)
**File**: `src/dfm_python/__init__.py`

**Change** (if present):
```python
# Before (in docstring):
    >>> from dfm_python.data_loader import load_data

# After (in docstring):
    >>> from dfm_python.data import load_data
    >>> # or for backward compatibility:
    >>> from dfm_python.data_loader import load_data
```

#### `dfm.py` (if docstring example exists)
**File**: `src/dfm_python/dfm.py`

**Change** (if present):
```python
# Before (in docstring):
    >>> from dfm_python.data_loader import load_config, load_data

# After (in docstring):
    >>> from dfm_python.data import load_config, load_data
    >>> # or for backward compatibility:
    >>> from dfm_python.data_loader import load_config, load_data
```

---

## Verification Steps

After implementation, verify:

1. **Syntax check**: `python3 -m py_compile src/dfm_python/data/loader.py`
2. **Import test**: `python3 -c "from dfm_python.data.loader import load_data, read_data, sort_data; print('OK')"`
3. **Backward compatibility**: `python3 -c "from dfm_python.data_loader import load_data; print('OK')"`
4. **Package export**: `python3 -c "from dfm_python.data import load_data; print('OK')"`
5. **API usage**: `python3 -c "from dfm_python.api import load_data; print('OK')"`

---

## Expected Impact

### File Size Changes
- **`data_loader.py`**: 300 lines → ~40 lines (87% reduction)
- **`data/loader.py`**: 0 lines → ~261 lines (new file)
- **`data/__init__.py`**: 11 lines → 12 lines (+1 line)

### Functional Impact
- **Zero** - Functions work identically
- **Backward compatibility**: Maintained (imports still work from `data_loader.py`)

### Organization Impact
- **High** - Completes `data/` package structure
- **Matches MATLAB structure**: `load_data.m` → `data/loader.py`
- **Clear separation**: All data-related functions now in `data/` package

---

## Risk Assessment

**Risk Level**: **LOW**

**Reasons**:
- Functions are self-contained
- No complex dependencies
- Backward compatibility maintained
- Similar pattern to previous iterations (7-10)
- Low risk of breaking changes

**Mitigation**:
- Maintain backward compatibility imports
- Test imports after each step
- Preserve function signatures exactly
- Update docstrings carefully

---

## Rollback Plan

If issues arise:

1. **Revert `data/loader.py`**: Delete the new file
2. **Revert `data/__init__.py`**: Remove `load_data` from exports
3. **Revert `data_loader.py`**: Restore function definitions
4. **Revert dependent modules**: Restore original imports

All changes are reversible and isolated.

---

## Success Criteria

- [ ] `data/loader.py` created with all 3 functions
- [ ] `data/__init__.py` exports `load_data`
- [ ] `data_loader.py` reduced to thin wrapper (~40 lines)
- [ ] All imports work correctly
- [ ] Backward compatibility maintained
- [ ] No functional changes
- [ ] Syntax check passes
- [ ] Import tests pass

---

## Notes

- This completes the `data_loader.py` split started in iteration 7
- Follows the same pattern as iterations 7-10
- Maintains backward compatibility for gradual migration
- Final result: `data/` package with 4 modules (utils, transformer, config_loader, loader)
- `data_loader.py` becomes a thin backward-compatibility wrapper
