# Changelog

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
