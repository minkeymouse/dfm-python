# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting high-dimensional time series. Built for economists, researchers, and data scientists working with mixed-frequency macroeconomic data.

## What is a Dynamic Factor Model?

A Dynamic Factor Model is a powerful statistical framework that:

- **Extracts common factors** from many observed time series (e.g., GDP, consumption, investment, employment)
- **Handles mixed frequencies** seamlessly (monthly, quarterly, annual data together)
- **Deals with missing data** robustly (late-arriving indicators, publication delays)
- **Provides nowcasts** (current-quarter GDP estimates before official release)
- **Forecasts** future values of both factors and observed series

**Key Innovation**: All latent factors evolve at a common "clock" frequency (e.g., monthly), while observed series can be at different frequencies. This is achieved through deterministic tent kernel aggregation, allowing quarterly GDP to be modeled using monthly latent factors.

## Features

### Core Capabilities
- ✅ **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual series in one model
- ✅ **Clock-based framework**: All factors evolve at a common clock frequency (typically monthly)
- ✅ **Block structure**: Flexible factor organization (global common factor + sector-specific factors)
- ✅ **Idiosyncratic components**: Per-series state augmentation for better fit (AR(1) for clock-frequency, tent-length chains for slower frequencies)
- ✅ **Robust missing data**: Spline interpolation preprocessing + Kalman filter handling during estimation
- ✅ **News decomposition**: Attribute forecast changes to specific data releases
- ✅ **Nowcasting & Forecasting**: Generate predictions for any horizon

### Technical Features
- ✅ **Multiple configuration methods**: YAML files, CSV specs, Python dictionaries, or Hydra
- ✅ **Advanced numerical stability**: 
  - Adaptive ridge regularization for ill-conditioned matrices
  - Q matrix floor (0.01 for factors) to prevent scale issues
  - C matrix normalization (||C[:,j]|| = 1) for clock-frequency factors
  - Spectral radius capping (< 0.99) for stationarity
  - Variance floors for all covariance matrices
- ✅ **Production-ready**: Comprehensive error handling, extensive testing, well-documented
- ✅ **Flexible API**: Module-level convenience functions or object-oriented interface

## Installation

```bash
pip install dfm-python
```

**Requirements**: 
- Python >= 3.10
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

Optional dependencies:
- `hydra-core` (for Hydra configuration management)
- `matplotlib` (for plotting)

## Quick Start

### Example 1: Simple Monthly Data

```python
import dfm_python as dfm
import numpy as np
import pandas as pd

# Generate or load your data (T × N: time periods × series)
# Data should be standardized or will be standardized automatically
X = np.random.randn(100, 10)  # 100 months, 10 series

# Create configuration
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

series = [
    SeriesConfig(series_id=f'series_{i}', frequency='m', transformation='lin', blocks=[1])
    for i in range(10)
]
blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
config = DFMConfig(series=series, blocks=blocks)

# Fit the model
model = dfm.DFM()
result = model.fit(X, config, max_iter=100, threshold=1e-4)

# Access results
factors = result.Z          # (101 × 1) Smoothed common factor
loadings = result.C         # (10 × 1) Factor loadings
smoothed_data = result.X_sm # (100 × 10) Smoothed series
forecasts = result.forecast(horizon=12)  # 12-step ahead forecasts
```

### Example 2: Mixed-Frequency Data (Monthly + Quarterly)

```python
import dfm_python as dfm
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

# Create config with mixed frequencies
series = [
    # Monthly series
    SeriesConfig(series_id='employment', frequency='m', transformation='lin', blocks=[1]),
    SeriesConfig(series_id='retail_sales', frequency='m', transformation='lin', blocks=[1]),
    # Quarterly series
    SeriesConfig(series_id='gdp', frequency='q', transformation='lin', blocks=[1]),
    SeriesConfig(series_id='consumption', frequency='q', transformation='lin', blocks=[1]),
]

blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
config = DFMConfig(
    series=series, 
    blocks=blocks,
    augment_idio=True,        # Enable idiosyncratic components (default: True)
    augment_idio_slow=True    # Enable tent-length chains for quarterly series
)

# Load your data (CSV with Date column + series columns)
dfm.from_dict(config.to_dict())  # Or use config directly
dfm.load_data('data/mixed_frequency_data.csv', sample_start='2000-01-01')

# Train
dfm.train(max_iter=5000, threshold=1e-5)

# Forecast
X_forecast, Z_forecast = dfm.predict(horizon=12)
```

### Example 3: Using Spec CSV (Recommended for CSV Workflows)

```python
import dfm_python as dfm
from dfm_python.config import Params

# 1. Create spec CSV (see Data Format section)
# 2. Load configuration
params = Params(
    max_iter=5000,
    threshold=1e-5,
    clock='m',
    augment_idio=True,  # Enable idio components
    idio_rho0=0.1       # Initial AR coefficient for idio
)
dfm.from_spec('data/spec.csv', params=params)

# 3. Load data
dfm.load_data('data/data.csv', sample_start='2020-01-01', sample_end='2023-12-31')

# 4. Train
dfm.train()

# 5. Forecast and visualize
X_forecast, Z_forecast = dfm.predict(horizon=12)
dfm.plot(kind='factor', factor_index=0, forecast_horizon=12, save_path='factor_forecast.png')

# 6. Access results
result = dfm.get_result()
print(f"Converged: {result.converged}, Iterations: {result.num_iter}")
print(f"Log-likelihood: {result.loglik:.2f}")
```

## Configuration Guide

### Configuration Methods

The package supports multiple ways to configure your model:

#### 1. Spec CSV + Params (Recommended for CSV workflows)

**Best for**: Users with existing CSV files and Excel-based workflows

```python
from dfm_python.config import Params

params = Params(
    max_iter=5000,
    threshold=1e-5,
    clock='m',
    augment_idio=True,
    augment_idio_slow=True,
    idio_rho0=0.1,
    idio_min_var=1e-8
)
dfm.from_spec('data/spec.csv', params=params)
```

#### 2. YAML Configuration

**Best for**: Users who prefer declarative configuration files

```python
import dfm_python as dfm

# Load from YAML
dfm.from_yaml('config/default.yaml')
dfm.load_data('data/sample_data.csv')
dfm.train()
```

#### 3. Hydra Decorator (Advanced)

**Best for**: Users already using Hydra for configuration management

```python
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import dfm_python as dfm

@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dfm.load_config(hydra=cfg)
    dfm.load_data(str(get_original_cwd() / "data" / "sample_data.csv"))
    dfm.train(max_iter=cfg.max_iter)
    X_forecast, Z_forecast = dfm.predict(horizon=cfg.get('forecast_horizon', 12))

if __name__ == "__main__":
    main()
```

**CLI Overrides**: `python script.py max_iter=10 threshold=1e-4 blocks.Block_Global.factors=2`

#### 4. Direct Python Objects

**Best for**: Programmatic configuration and integration with other Python code

```python
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

config = DFMConfig(
    series=[
        SeriesConfig(series_id='gdp', frequency='q', transformation='lin', blocks=[1]),
        SeriesConfig(series_id='employment', frequency='m', transformation='lin', blocks=[1]),
    ],
    blocks={'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')},
    augment_idio=True,
    augment_idio_slow=True
)
```

### Key Configuration Parameters

#### Model Structure
- **`clock`**: Base frequency for all latent factors (default: `'m'` for monthly)
  - Options: `'d'` (daily), `'w'` (weekly), `'m'` (monthly), `'q'` (quarterly), `'sa'` (semi-annual), `'a'` (annual)
  - All factors evolve at this frequency, regardless of observation frequencies

- **`blocks`**: Factor block structure
  - `Block_Global`: Common factor affecting all series
  - Additional blocks: Sector-specific factors (e.g., `Block_Consumption`, `Block_Investment`)

- **`factors`**: Number of factors per block (typically 1-3)

#### Estimation Parameters
- **`max_iter`**: Maximum EM iterations (default: 5000)
- **`threshold`**: EM convergence threshold (default: 1e-5)
  - Algorithm stops when log-likelihood change < threshold
- **`ar_lag`**: Autoregressive lag for factors (default: 1)
  - Typically 1 (AR(1) process), can be increased for more complex dynamics

#### Idiosyncratic Components (New in v0.2.3+)
- **`augment_idio`**: Enable state augmentation with idiosyncratic components (default: `True`)
  - **Clock-frequency series**: Adds 1×1 AR(1) idio state per series
  - **Slower-frequency series**: Adds tent-length chains (if `augment_idio_slow=True`)
  - Improves model fit by capturing series-specific dynamics beyond common factors

- **`augment_idio_slow`**: Enable tent-length chains for slower-frequency series (default: `True`)
  - Quarterly series get 5-state chains (matching tent kernel length)
  - Semi-annual series get 7-state chains, annual get 9-state chains

- **`idio_rho0`**: Initial AR coefficient for idiosyncratic components (default: 0.1)
  - Controls persistence of idio components (0.1 = low persistence, 0.9 = high persistence)

- **`idio_min_var`**: Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
  - Prevents numerical issues from near-zero variances

#### Numerical Stability
- **`regularization_scale`**: Scale factor for adaptive ridge regularization (default: 1e-5)
  - Automatically applied when matrix condition number > 1e8
- **`damping_factor`**: Damping factor when likelihood decreases (default: 0.8)
- **`min_eigenvalue`**: Minimum eigenvalue for positive definite matrices (default: 1e-8)
- **`max_eigenvalue`**: Maximum eigenvalue cap (default: 1e6)
- **`clip_ar_coefficients`**: Clip AR coefficients to [-0.99, 0.99] for stationarity (default: `True`)

**Automatic Stability Features** (always enabled):
- **Q matrix floor**: Factor innovation variances are bounded below by 0.01 to prevent scale issues
- **C matrix normalization**: Clock-frequency factor loadings are normalized (||C[:,j]|| = 1) to stabilize scale
  - Automatically skipped for mixed-frequency data to preserve tent weight constraints
- **Spectral radius cap**: A matrix eigenvalues are capped at 0.99 to ensure stationarity
- **R matrix floor**: Observation error variances are bounded below by `idio_min_var` (default: 1e-8)

See `Params` dataclass documentation for the complete list of parameters.

## Data Format

### CSV Format

Your data CSV should have:
- **First column**: `Date` (YYYY-MM-DD format)
- **Subsequent columns**: One per time series, column names must match `series_id` in configuration
- **Missing values**: Empty cells or `NaN`
- **Mixed frequencies**: Quarterly series only have values at quarter-end months (e.g., March, June, September, December)

**Example**:
```csv
Date,gdp_real,consumption,investment,employment,retail_sales
2000-01-01,,98.5,95.0,100.2,102.1
2000-02-01,,98.7,95.2,100.5,102.3
2000-03-01,100.5,99.0,95.5,100.8,102.5
2000-04-01,,99.2,95.7,101.0,102.7
2000-05-01,,99.4,95.9,101.2,102.9
2000-06-01,101.2,99.8,96.2,101.5,103.1
```

**Notes**:
- Quarterly series (e.g., `gdp_real`) only have values at quarter-end months
- Monthly series (e.g., `employment`, `retail_sales`) have values every month
- Missing values are handled automatically via spline interpolation and Kalman filtering

### Supported Frequencies

- **`'m'`**: Monthly
- **`'q'`**: Quarterly
- **`'sa'`**: Semi-annual
- **`'a'`**: Annual

**Important**: Series with frequencies faster than the clock (e.g., daily/weekly when clock is monthly) are **not supported** and will raise a clear error.

### Data Transformations

Each series can have a transformation applied:
- **`'lin'`**: No transformation (levels)
- **`'log'`**: Natural logarithm
- **`'diff'`**: First difference
- **`'ldiff'`**: Log difference (log of first difference)
- **`'pch'`**: Percentage change
- **`'pca'`**: Principal component (for pre-processed data)

## Understanding the Results

### DFMResult Object

After training, you get a `DFMResult` object containing:

```python
result = model.fit(X, config)

# Factors and Loadings
result.Z          # (T+1 × m) Smoothed factor estimates (one row per time period)
result.C          # (N × m) Factor loadings (how each series loads on each factor)

# Model Parameters
result.A          # (m × m) Factor transition matrix (AR dynamics)
result.Q          # (m × m) Innovation covariance (factor shocks)
result.R          # (N × N) Observation covariance (idiosyncratic errors)

# Smoothed Data
result.X_sm       # (T × N) Smoothed observed series (unstandardized)
result.x_sm       # (T × N) Smoothed observed series (standardized)

# Initial Conditions
result.Z_0        # (m,) Initial factor state
result.V_0        # (m × m) Initial factor covariance

# Convergence Information
result.converged  # bool: Whether EM algorithm converged
result.num_iter   # int: Number of EM iterations completed
result.loglik     # float: Final log-likelihood value

# Forecasts (if predict() was called)
result.forecast_X # Forecasted series
result.forecast_Z # Forecasted factors
```

### Interpreting Factors

- **`Z[:, 0]`**: First factor (typically the "common factor" capturing overall economic activity)
- **`Z[:, 1]`**: Second factor (if multiple factors, captures additional variation)
- **`C[i, j]`**: Loading of series `i` on factor `j` (how much series `i` responds to factor `j`)

### Example: Extracting Common Factor

```python
result = model.fit(X, config)

# Extract common factor (first factor)
common_factor = result.Z[:, 0]

# Project factor onto a specific series
series_idx = 0  # First series
series_factor_component = result.Z @ result.C[series_idx, :].T

# Idiosyncratic component (residual)
idio_component = result.X_sm[:, series_idx] - series_factor_component
```

## Advanced Features

### News Decomposition

Decompose forecast updates into contributions from individual data releases:

```python
from dfm_python import news_dfm

# Before new data release
result_old = model.fit(X_old, config)

# After new data release
X_new = ...  # Updated data
result_new = model.fit(X_new, config)

# Decompose forecast update
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old, result_old, t_fcst=100, v_news=0
)

# Forecast update
forecast_update = y_new - y_old
# singlenews contains contribution from each series
```

### Custom Block Structure

Model sector-specific factors:

```python
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

series = [
    # Consumption block
    SeriesConfig(series_id='retail_sales', frequency='m', transformation='lin', blocks=[1, 2]),
    SeriesConfig(series_id='consumer_confidence', frequency='m', transformation='lin', blocks=[1, 2]),
    # Investment block
    SeriesConfig(series_id='investment', frequency='q', transformation='lin', blocks=[1, 3]),
    SeriesConfig(series_id='capital_orders', frequency='m', transformation='lin', blocks=[1, 3]),
]

blocks = {
    'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m'),
    'Block_Consumption': BlockConfig(factors=1, ar_lag=1, clock='m'),
    'Block_Investment': BlockConfig(factors=1, ar_lag=1, clock='m'),
}

config = DFMConfig(series=series, blocks=blocks)
```

## Troubleshooting

### Convergence Issues

**Problem**: EM algorithm doesn't converge within `max_iter` iterations.

**Solutions**:
- Increase `max_iter` (default: 5000)
- Relax `threshold` (default: 1e-5, try 1e-4)
- Check data quality: excessive missing data, extreme outliers
- Verify block structure: ensure sufficient series per block
- Try different `regularization_scale` (default: 1e-5)

### Dimension Mismatch Errors

**Problem**: `ValueError` about dimensions or series IDs.

**Solutions**:
- Ensure `series_id` in config exactly matches CSV column names (case-sensitive)
- Verify all series have valid frequency settings (`'m'`, `'q'`, `'sa'`, `'a'`)
- Check that block structure is consistent (all series must load on `Block_Global`)
- Ensure no series have frequency faster than clock (e.g., daily/weekly with monthly clock)

### Numerical Instability

**Problem**: NaN/Inf values, singular matrix errors, or warnings about regularization.

**Solutions**:
- Enable data clipping: `clip_data_values=True` (default: True)
- Adjust `min_eigenvalue` and `max_eigenvalue` thresholds
- Increase `regularization_scale` (default: 1e-5, try 1e-4)
- Check for extreme outliers in data (use `data_clip_threshold`, default: 100.0)
- Verify sufficient data variation in each block
- Consider disabling idio augmentation: `augment_idio=False` (if causing issues)

**Note**: The package includes automatic stability features:
- Q matrix floor (0.01 for factors) prevents scale issues
- C matrix normalization stabilizes loading scales
- Spectral radius capping ensures stationarity
- These are always enabled and help prevent most numerical issues

### Factor Evolution Issues

**Problem**: Factors are constant, near-zero, or don't evolve.

**Solutions**:
- Check that innovation variances (Q diagonal) are not zero
- Verify sufficient data variation in each block
- Ensure at least 2-3 series per block for reliable factor estimation
- Increase `regularization_scale` for sparse data
- Check that AR coefficients (A matrix) are not all zero

### Missing Data Warnings

**Problem**: Warnings about excessive missing data (>50%).

**Solutions**:
- Review data quality: remove series with >70% missing data
- Use appropriate `nan_method` (default: 2 = spline interpolation)
- Consider imputing critical series before estimation
- Verify frequency settings: quarterly series should only have values at quarter-ends

## API Reference

### Module-Level Functions (Convenience API)

```python
import dfm_python as dfm

# Configuration
dfm.from_spec(spec_path, params=None)      # Load from spec CSV + Params
dfm.from_yaml(yaml_path)                    # Load from YAML file
dfm.from_dict(config_dict)                  # Load from dictionary
dfm.load_config(hydra=cfg)                  # Load from Hydra DictConfig

# Data
dfm.load_data(data_path, sample_start=None, sample_end=None)

# Estimation
dfm.train(max_iter=None, threshold=None, **kwargs)

# Forecasting
dfm.predict(horizon=12)                     # Returns (X_forecast, Z_forecast)

# Visualization
dfm.plot(kind='factor', factor_index=0, forecast_horizon=None, save_path=None)

# Results
dfm.get_result()                            # Returns DFMResult object
dfm.reset()                                  # Reset singleton state
```

### Object-Oriented API

```python
from dfm_python import DFM, DFMConfig, SeriesConfig, BlockConfig

# Create model
model = DFM()

# Create config
config = DFMConfig(...)

# Fit
result = model.fit(X, config, max_iter=5000, threshold=1e-5)

# Access results
factors = result.Z
loadings = result.C
```

### DFMResult Object

Complete reference for `DFMResult`:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `Z` | (T+1 × m) | Smoothed factor estimates |
| `C` | (N × m) | Factor loadings matrix |
| `A` | (m × m) | Factor transition matrix |
| `Q` | (m × m) | Innovation covariance |
| `R` | (N × N) | Observation covariance |
| `X_sm` | (T × N) | Smoothed data (unstandardized) |
| `x_sm` | (T × N) | Smoothed data (standardized) |
| `Z_0` | (m,) | Initial factor state |
| `V_0` | (m × m) | Initial factor covariance |
| `converged` | bool | Convergence status |
| `num_iter` | int | Number of EM iterations |
| `loglik` | float | Final log-likelihood |

## Architecture

### Core Modules

- **`config.py`**: Configuration dataclasses (`DFMConfig`, `SeriesConfig`, `BlockConfig`, `Params`)
- **`config_sources.py`**: Configuration adapters (YAML/Dict/CSV/Hydra)
- **`dfm.py`**: Core estimation (`DFM` class, `_dfm_core` function, EM algorithm orchestration)
- **`core/em.py`**: EM algorithm implementation (`init_conditions`, `em_step`, `em_converged`)
- **`core/numeric.py`**: Numerical utilities (covariance computation, regularization, stability)
- **`kalman.py`**: Kalman filter and smoother (`run_kf`, `skf`, `fis`)
- **`data.py`**: Data loading and preprocessing (`load_data`, `transform_data`, `rem_nans_spline`)
- **`utils.py`**: Mixed-frequency utilities (tent weights, aggregation structure, idio chain lengths)
- **`api.py`**: High-level convenience API and module-level functions
- **`news.py`**: News decomposition

### Key Design Principles

1. **Clock-based synchronization**: All factors evolve at a common frequency
2. **Tent kernel aggregation**: Deterministic weights for slower-frequency series
3. **Idiosyncratic components**: Per-series state augmentation for better fit
4. **Advanced numerical stability**: 
   - Adaptive ridge regularization (condition-number-based)
   - Q matrix floor (0.01 for factors) to prevent scale issues
   - C matrix normalization (||C[:,j]|| = 1) for clock-frequency factors
   - Spectral radius capping (< 0.99) for stationarity
   - Variance floors for all covariance matrices
5. **Flexible configuration**: Multiple input methods (YAML, CSV, Dict, Hydra)
6. **Backward compatibility**: Old code continues to work, new features are opt-in

## Testing

Run the full test suite:

```bash
pytest src/test/ -v
```

Run specific test categories:

```bash
# Core DFM estimation
pytest src/test/test_dfm.py -v

# Kalman filter/smoother
pytest src/test/test_kalman.py -v

# Numerical stability
pytest src/test/test_numeric.py -v

# Idiosyncratic components (new)
pytest src/test/test_dfm.py -k "idio" -v
```

**Test Coverage**: 70+ tests covering:
- Core DFM estimation and EM algorithm
- Kalman filter and smoother
- Mixed-frequency data handling
- Idiosyncratic component augmentation
- Numerical stability and edge cases
- Configuration loading and validation
- API functionality

## Tutorials

### Basic Tutorial (Spec CSV Approach)

```bash
python tutorial/basic_tutorial.py --spec data/sample_spec.csv --data data/sample_data.csv
```

Comprehensive tutorial covering:
- Loading configuration from spec CSV
- Data preparation and transformation
- Model training and convergence
- Interpreting results
- Forecasting and visualization

### Hydra Tutorial (Advanced Configuration)

```bash
python tutorial/hydra_tutorial.py max_iter=10 threshold=1e-4
```

Demonstrates:
- Hydra-based configuration management
- CLI parameter overrides
- Complex multi-block configurations

## Project Status

**Version**: 0.2.7  
**Status**: Stable and production-ready  
**Python**: 3.10+  
**PyPI**: https://pypi.org/project/dfm-python/

### Recent Improvements (v0.2.7)

- ✅ **Fixed C normalization issue**: C normalization now only applies when C norm is in reasonable range (0.1 ~ 10)
  - Prevents Q from becoming too small when C norm is very small
  - Prevents factor values from becoming extreme
- ✅ **Improved stability**: Conditional C normalization prevents pathological factor paths
- ✅ **Better diagnostics**: Added variance guard binding checks in tutorial

### Previous Improvements (v0.2.6)

- ✅ **Q matrix cap**: Factor innovation variances bounded above by 1.0 to prevent explosion
- ✅ **R matrix cap**: Observation error variances bounded above by 10.0 for standardized data
- ✅ **Improved C normalization**: C norm clipping prevents Q explosion during normalization
- ✅ **Enhanced initial values**: PCA loadings clipped to prevent extreme initial values
- ✅ **Bug fix**: Fixed `np.linalg.lu` error by using `np.linalg.slogdet` for determinant computation
- ✅ **Better stability**: Q and R caps ensure model convergence even with challenging data

### Previous Improvements (v0.2.5)

- ✅ **Q matrix floor**: Factor innovation variances bounded below by 0.01 to prevent scale issues
- ✅ **C matrix normalization**: Clock-frequency factor loadings normalized (||C[:,j]|| = 1) for scale stability
  - Automatically preserves tent weight constraints for mixed-frequency data
- ✅ **Enhanced numerical stability**: Multiple layers of guards prevent numerical issues
- ✅ **Comprehensive testing**: 74+ tests covering all features and edge cases
- ✅ **Improved documentation**: User-friendly README, comprehensive examples, detailed tutorials

### Previous Improvements (v0.2.3)

- ✅ **Idiosyncratic component augmentation**: Per-series state augmentation for improved fit
- ✅ **Adaptive ridge regularization**: Condition-number-based regularization for numerical stability
- ✅ **Enhanced numerical guards**: Spectral radius capping, variance floors, sign-aligned PCs

### Project Goals

The package follows these design principles:

1. **Stability**: No complex numbers, sensible EM convergence, robust numerical operations
2. **Generality**: Works with any valid CSV schema and configuration
3. **Completeness**: Full pipeline from initialization to forecasting
4. **Flexibility**: Multiple configuration methods, extensible architecture
5. **Maintainability**: Clean code, comprehensive tests, extensive documentation

## Citation

If you use `dfm-python` in your research, please cite:

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting and Forecasting},
  author = {DFM Python Contributors},
  year = {2025},
  url = {https://pypi.org/project/dfm-python/},
  version = {0.2.7}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest src/test/ -q`
- Code follows existing patterns and style
- New features include tests
- Documentation is updated

## Support

For issues, questions, or contributions:
- Check the [Troubleshooting](#troubleshooting) section
- Review the tutorials in `tutorial/`
- Examine test files in `src/test/` for usage examples

---

**Quick Links**:
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
