# Changelog

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
