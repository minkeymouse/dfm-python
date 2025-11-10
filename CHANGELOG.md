# Changelog

## [0.2.3] - 2025-01-XX

### Changed

- **Matplotlib Version Requirement**: Relaxed matplotlib version requirement from `>=3.10.7` to `>=3.5.0`
  - Improves compatibility with other packages (e.g., `ydata-profiling`) that require `matplotlib<=3.10`
  - Reduces dependency conflicts in Kaggle and other environments

## [0.2.2] - 2025-01-XX

### Changed

- **Python Version Support**: Lowered minimum Python version requirement from 3.12 to 3.10
  - Updated `requires-python` in `pyproject.toml` to `>=3.10`
  - Added Python 3.10 and 3.11 to package classifiers
  - Updated README to reflect Python 3.10+ requirement
  - Code is fully compatible with Python 3.10, 3.11, and 3.12

### Fixed

- **Kaggle Compatibility**: Package can now be installed in Kaggle notebooks (Python 3.10 environment)

## [0.2.1] - 2025-01-XX

### Fixed

- **Mixed Frequency Edge Case**: Fixed dimension mismatch in `init_conditions` when slower frequency idio components are added to `C` but not to `A`. Now properly creates `BQ`, `SQ`, and `initViQ` with correct dimensions.
- **UnboundLocalError**: Fixed `F` variable initialization in `init_conditions` to prevent `UnboundLocalError` when block initialization fails.
- **Kalman Filter**: Fixed `Y_t` initialization in `skf` to prevent `UnboundLocalError` when no observations are present.
- **MergedConfigSource**: Fixed handling of partial configuration dictionaries (e.g., only `max_iter` and `threshold`) to correctly merge with base configuration.
- **Tutorials**: Fixed pandas FutureWarning by updating `'M'` frequency to `'ME'` in both `basic_tutorial.py` and `hydra_tutorial.py`.
- **Test Suite**: Recreated `test_config.py` from scratch to fix indentation errors and align with new API.

### Documentation

- **README**: Significantly simplified and streamlined (reduced from 653 to ~250 lines) while maintaining all essential information.
- Improved clarity and focus on most common use cases.

## [0.2.0] - 2025-01-XX

### Major Refactoring

This release represents a comprehensive refactoring focused on code quality, maintainability, and robustness.

### Fixed

- **Mixed Frequency Edge Case**: Fixed dimension mismatch in `init_conditions` when slower frequency idio components are added to `C` but not to `A`. Now properly creates `BQ`, `SQ`, and `initViQ` with correct dimensions.
- **UnboundLocalError**: Fixed `F` variable initialization in `init_conditions` to prevent `UnboundLocalError` when block initialization fails.
- **Kalman Filter**: Fixed `Y_t` initialization in `skf` to prevent `UnboundLocalError` when no observations are present.
- **MergedConfigSource**: Fixed handling of partial configuration dictionaries (e.g., only `max_iter` and `threshold`) to correctly merge with base configuration.
- **Tutorials**: Fixed pandas FutureWarning by updating `'M'` frequency to `'ME'` in both `basic_tutorial.py` and `hydra_tutorial.py`.
- **Test Suite**: Recreated `test_config.py` from scratch to fix indentation errors and align with new API.

#### Architecture Improvements

- **Class-Centric Design**: Complete migration from functional API to class-based design
  - Removed deprecated `dfm()` function wrapper
  - Core algorithm now in `DFM` class with `fit()` method
  - Module-level convenience API for easy usage
  - All legacy backward compatibility code removed

- **Modular Structure**: Reorganized code into focused `core` modules
  - `core/em.py`: EM algorithm core (init_conditions, em_step, em_converged)
  - `core/numeric.py`: Numerical utilities (matrix operations, regularization, clipping)
  - `core/diagnostics.py`: Diagnostic functions and output formatting
  - `core/results.py`: Result metrics (RMSE calculation)
  - `core/grouping.py`: Frequency grouping utilities
  - Removed 17 proxy functions, using direct imports for cleaner code

#### Code Quality Improvements

- **Exception Handling**: Replaced all generic `Exception` catches with specific numerical exception types
  - Defined `_NUMERICAL_EXCEPTIONS` tuple for common numerical errors
  - More specific error handling: `LinAlgError`, `ValueError`, `ZeroDivisionError`, `OverflowError`, `FloatingPointError`
  - Better debugging and error tracking throughout the package

- **Logging**: Improved logging infrastructure
  - Converted `print()` statements to proper logging in diagnostics module
  - Added early return for performance when logging is disabled
  - Consistent logging levels (info, warning, debug) across modules
  - User-facing functions (e.g., `print_series_diagnosis`) still use `print()` for direct output

- **Import Optimization**: Cleaner import structure
  - Moved pandas import to module top with availability check
  - Removed redundant imports and circular dependencies
  - Consistent import patterns across all modules

#### Code Simplification

- **Removed Redundancy**: Eliminated ~120 lines of redundant code
  - Removed all proxy functions (17 total)
  - Direct imports from core modules
  - Cleaner, more maintainable codebase

- **Helper Functions**: Extracted common patterns into reusable helpers
  - `_check_finite()`: Centralized NaN/Inf validation
  - `_ensure_square_matrix()`: Matrix shape validation
  - `_get_config_param()`: Centralized config parameter access
  - `_resolve_param()`: Parameter override resolution

#### Type Safety

- **TypedDict**: Added `NaNHandlingOptions` TypedDict for type-safe NaN handling
- **Improved Type Hints**: Better type annotations throughout
- **TYPE_CHECKING**: Used for circular import resolution

#### Numerical Stability

- **Enhanced Validation**: Better validation of initial conditions and intermediate results
- **Improved Error Messages**: More descriptive error messages with context
- **Robust Fallbacks**: Better fallback mechanisms for numerical edge cases

### Changed

- **BREAKING**: Removed `dfm()` function - use `DFM().fit()` instead
- **BREAKING**: Removed deprecated `load_config_from_*` methods
- Module structure reorganized for better maintainability
- All diagnostic output now uses logging instead of print (except user-facing functions)
- Exception handling now catches specific exception types instead of generic `Exception`

### Added

- `_NUMERICAL_EXCEPTIONS` tuple for consistent exception handling
- `NaNHandlingOptions` TypedDict for type-safe NaN handling
- Helper functions for common patterns (`_check_finite`, `_ensure_square_matrix`, etc.)
- Early return optimization in diagnostics when logging is disabled
- Better error messages with context

### Removed

- Deprecated `dfm()` function wrapper
- All proxy functions (17 total) - now using direct imports
- Legacy backward compatibility code
- Redundant imports and circular dependencies
- ~120 lines of redundant code

### Fixed

- Improved exception handling specificity
- Better error messages for debugging
- Fixed import circular dependencies using TYPE_CHECKING
- Enhanced numerical stability validation

### Documentation

- Updated README to reflect new class-based API
- Updated all docstrings to use new API
- Improved code comments and documentation

## [0.1.7] - 2025-01-XX

### Changed
- **BREAKING**: Removed CSV config loading - use YAML files or create `DFMConfig` objects directly
- Made package more generic by removing application-specific dependencies
- Improved code documentation and comments throughout codebase
- Refactored variable naming for clarity and consistency
- Updated README to be more concise and focused

### Added
- Direct `DFMConfig` object creation support in `load_config()`
- Application-specific adapter guidance for custom formats
- Database-backed data loading adapter interface documentation
- Enhanced function docstrings with detailed parameter descriptions

### Removed
- CSV configuration file support (deprecated, use YAML or direct object creation)
- `load_config_from_csv()` function (deprecated, kept for backward compatibility with warning)
- Spec file dependencies

### Fixed
- Improved error messages for frequency constraint violations
- Enhanced type hints and documentation

## [0.1.6] - Previous release

### Changed
- Frequency generalization improvements
- Code cleanup and refactoring

## [0.1.5] - Previous release

### Changed
- Additional improvements and bug fixes

## [0.1.4] - 2024-11-07

### Fixed
- Fixed critical bug: `ff` variable scoping issue in `init_conditions()` causing dimension mismatch errors in multi-block models with different factor counts
- Fixed `A_temp` variable scoping issue that could cause incorrect computations when OLS regression fails for some blocks
- Fixed `bl_idxM` and `bl_idxQ` initialization issue that could cause AttributeError when no blocks are processed
- Removed unreachable code in `init_conditions()` else branch
- Improved empty array handling in `em_step()` for quarterly series constraints

### Changed
- Enhanced variable scoping safety by resetting block-specific variables at start of each iteration
- Improved error handling for edge cases with empty arrays

## [0.1.3] - 2024-11-07

### Added
- Comprehensive README with detailed inputs/outputs documentation
- Clock-based mixed-frequency framework documentation
- Enhanced module-level docstrings throughout the codebase
- Detailed API reference with examples
- Improved error messages with context and solutions

### Changed
- Optimized code performance (removed redundant computations)
- Improved memory usage (replaced unnecessary copies with views)
- Enhanced code documentation and comments
- Updated package description and metadata

### Fixed
- Removed duplicate `frequencies_array` computation
- Fixed `optNaN` dictionary mutation issue
- Improved code organization and clarity

### Documentation
- Complete input/output specifications
- 5 comprehensive usage examples
- Troubleshooting guide with solutions
- Architecture overview
- Clock-based framework explanation

## [0.1.2] - Previous release

### Fixed
- Fixed `ModuleNotFoundError: No module named 'utils'` by moving utils into dfm_python package
- Improved import paths and package structure

## [0.1.1] - Initial release

- Initial PyPI release
- Core DFM estimation functionality
- Mixed-frequency data support
- News decomposition
