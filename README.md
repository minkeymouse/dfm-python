# dfm-python

A comprehensive Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting economic and financial time series. This package provides a robust, production-ready solution for handling mixed-frequency data, missing observations, and performing news decomposition analysis.

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Understanding Dynamic Factor Models](#understanding-dynamic-factor-models)
- [Configuration Guide](#configuration-guide)
- [Data Format](#data-format)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Dynamic Factor Models (DFM) are powerful tools for analyzing high-dimensional time series data by extracting common factors that drive multiple observed series. This package implements:

- **EM Algorithm**: Expectation-Maximization algorithm for parameter estimation
- **Kalman Filtering**: State-space filtering and smoothing for factor extraction
- **Mixed-Frequency Support**: Seamlessly handle daily, monthly, and quarterly data together
- **Missing Data Handling**: Robust interpolation and estimation with incomplete data
- **News Decomposition**: Analyze how new data releases affect forecasts

### What is a Dynamic Factor Model?

A DFM models observed time series as:

```
y_t = C × Z_t + e_t    (Observation equation)
Z_t = A × Z_{t-1} + v_t  (State equation)
```

Where:
- `y_t`: Observed time series at time t
- `Z_t`: Unobserved common factors
- `C`: Factor loadings (how series relate to factors)
- `A`: Transition matrix (how factors evolve)
- `e_t`, `v_t`: Error terms

The model extracts common patterns across multiple series while accounting for their individual dynamics.

## Key Features

### 🎯 Core Capabilities

- **Mixed-Frequency Data**: Handle daily, monthly, and quarterly series simultaneously
- **Robust Missing Data**: Automatic interpolation and handling of missing observations (tested with up to 50% missing data)
- **Block Structure**: Organize series into logical blocks (e.g., Global, Consumption, Investment factors)
- **Flexible Configuration**: Support for both YAML and CSV configuration formats
- **News Decomposition**: Decompose forecast updates into contributions from individual data releases
- **Production Ready**: Comprehensive error handling, logging, and validation

### 📊 Supported Frequencies

- **Daily** (`d`): High-frequency data (e.g., stock prices, exchange rates)
- **Monthly** (`m`): Standard economic indicators (e.g., unemployment, industrial production)
- **Quarterly** (`q`): Low-frequency aggregates (e.g., GDP, national accounts)

### 🔧 Data Transformations

The package supports various transformations to ensure stationarity:
- `lin`: Levels (no transformation)
- `chg`: First difference
- `pch`: Percent change
- `pca`: Percent change at annual rate
- `log`: Natural logarithm
- And more (see [Configuration Guide](#configuration-guide))

## Installation

### Basic Installation

```bash
pip install dfm-python
```

### Optional Dependencies

**For Hydra configuration support** (experiment management):
```bash
pip install dfm-python[hydra]
```

**For development tools** (testing, linting):
```bash
pip install dfm-python[dev]
```

**For database integration** (application-specific):
```bash
pip install dfm-python[database]
```

**Install everything**:
```bash
pip install dfm-python[all]
```

### Requirements

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## Quick Start

### Minimal Example

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd

# 1. Load configuration
config = load_config('config.yaml')

# 2. Load and prepare data
X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))

# 3. Estimate the model
result = dfm(X, config, threshold=1e-4)

# 4. Access results
factors = result.Z          # Extracted factors (T × r)
loadings = result.C         # Factor loadings (N × r)
smoothed = result.X_sm      # Smoothed data (T × N)
```

### Complete Working Example

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd
import numpy as np

# Load configuration (YAML or CSV)
config = load_config('config/example_config.yaml')

# Load data
X, Time, Z = load_data(
    'data/sample_data.csv', 
    config, 
    sample_start=pd.Timestamp('2000-01-01')
)

print(f"Data loaded: {X.shape[0]} time periods, {X.shape[1]} series")
print(f"Time range: {Time[0]} to {Time[-1]}")

# Estimate DFM model
print("\nEstimating DFM model...")
result = dfm(X, config, threshold=1e-4, max_iter=1000)

# Explore results
print(f"\n✓ Model estimated successfully!")
print(f"  Factors extracted: {result.Z.shape[1]}")
print(f"  Factor loadings: {result.C.shape}")
print(f"  Transition matrix: {result.A.shape}")

# Access common factor (first factor)
common_factor = result.Z[:, 0]
print(f"\n  Common factor (first 5 values): {common_factor[:5]}")

# Reconstruct a series from factors
series_0_reconstructed = result.Z @ result.C[0, :].T
print(f"\n  Reconstructed series 0 (first 5 values): {series_0_reconstructed[:5]}")
```

## Understanding Dynamic Factor Models

### Why Use DFM?

Dynamic Factor Models are ideal when you have:
- **Many time series** that share common patterns
- **Mixed-frequency data** (e.g., quarterly GDP with monthly indicators)
- **Missing data** that needs to be handled robustly
- **Nowcasting needs** (forecasting the present using partial information)

### How It Works

1. **Factor Extraction**: The model identifies common factors that drive multiple series
2. **State-Space Framework**: Uses Kalman filtering to estimate unobserved factors
3. **EM Algorithm**: Iteratively estimates parameters until convergence
4. **Smoothing**: Produces smoothed estimates of both factors and data

### Model Structure

The package supports **block-structured factors**:

- **Global Block**: Common factors affecting all series
- **Specific Blocks**: Factors affecting subsets of series (e.g., Consumption, Investment)

This allows you to model both common economic trends and sector-specific dynamics.

## Configuration Guide

The DFM model is configured through a `DFMConfig` object, which can be loaded from YAML or CSV files.

### YAML Configuration (Recommended)

YAML format provides the most flexibility and clarity:

```yaml
# config/example_config.yaml
model:
  series:
    gdp_real:
      series_name: "Real GDP (Quarterly)"
      frequency: "q"              # Quarterly
      transformation: "pca"       # Percent change at annual rate
      category: "GDP"
      units: "Index (2000=100)"
      blocks: [Global, Consumption, Investment]  # Which factors affect this series
    
    consumption:
      series_name: "Consumption (Monthly)"
      frequency: "m"              # Monthly
      transformation: "pch"       # Percent change
      category: "Consumption"
      units: "Index (2000=100)"
      blocks: [Global, Consumption]
    
    stock_index:
      series_name: "Stock Market Index (Daily)"
      frequency: "d"              # Daily
      transformation: "pch"       # Percent change
      category: "Financial"
      units: "Index"
      blocks: [Global, Investment]

  blocks:
    Global:
      factors: 2                  # 2 common factors
    Consumption:
      factors: 1                  # 1 consumption-specific factor
    Investment:
      factors: 1                  # 1 investment-specific factor

dfm:
  ar_lag: 1                       # AR lag for factors
  threshold: 1e-5                 # Convergence threshold
  max_iter: 5000                  # Maximum EM iterations
  nan_method: 2                   # Missing data handling method
  nan_k: 3                        # Spline interpolation parameter
```

### CSV Configuration (Simpler)

CSV format is more compact and easier to edit in spreadsheets:

```csv
series_id,series_name,frequency,transformation,category,units,Block_Global,Block_Consumption,Block_Investment
gdp_real,Real GDP (Quarterly),q,pca,GDP,Index (2000=100),1,1,1
gdp_nominal,Nominal GDP (Quarterly),q,pca,GDP,Index (2000=100),1,1,0
consumption,Consumption (Monthly),m,pch,Consumption,Index (2000=100),1,1,0
investment,Investment (Monthly),m,pch,Investment,Index (2000=100),1,0,1
stock_index,Stock Market Index (Daily),d,pch,Financial,Index,1,0,1
exchange_rate,Exchange Rate (Daily),d,pch,Financial,Rate,1,0,0
```

**Important**: The first block (Global) must have value `1` for all series, as it represents the common factor that all series load on.

### Frequency Codes

- `d`: Daily
- `w`: Weekly
- `m`: Monthly
- `q`: Quarterly
- `sa`: Semi-annual
- `a`: Annual

### Transformation Codes

- `lin`: Levels (no transformation)
- `chg`: First difference (change)
- `ch1`: Year-over-year change
- `pch`: Percent change
- `pc1`: Year-over-year percent change
- `pca`: Percent change at annual rate
- `cch`: Continuously compounded rate of change
- `cca`: Continuously compounded annual rate
- `log`: Natural logarithm

## Data Format

### CSV Data File Structure

Your data CSV file should follow this structure:

```csv
Date,gdp_real,gdp_nominal,consumption,investment,stock_index,exchange_rate
2000-01-01,,,98.5,95.0,995.87,0.997
2000-02-01,,,98.7,95.2,996.75,1.010
2000-03-01,100.5,105.6,99.0,95.5,979.65,0.992
2000-04-01,,,99.3,95.8,991.26,0.991
...
```

### Requirements

1. **Date Column**: First column must be named `Date` (case-sensitive)
2. **Column Names**: Must exactly match `series_id` values in your configuration
3. **Date Format**: YYYY-MM-DD format (e.g., `2000-01-01`)
4. **Missing Values**: Use empty cells or `NaN` for missing observations

### Mixed-Frequency Data

The package automatically handles mixed frequencies:

- **Quarterly series**: Only include values at quarter-end months (March, June, September, December). Other months should be empty/NaN.
- **Monthly series**: Include values for all months
- **Daily series**: Include values for all days (or aggregate to monthly for the model)

**Example**: If you have quarterly GDP data, your CSV should look like:
```csv
Date,gdp_real
2000-01-01,        # Empty - not a quarter-end
2000-02-01,        # Empty - not a quarter-end
2000-03-01,100.5   # Q1 value
2000-04-01,        # Empty - not a quarter-end
...
```

### Sample Data

The package includes sample data (`data/sample_data.csv`) with:
- **12 time series** spanning 2000-2010
- **Mixed frequencies**: 2 quarterly, 8 monthly, 2 daily
- **Missing values**: Realistic missing data patterns for robustness testing

## Usage Examples

### Example 1: Basic DFM Estimation

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd

# Load configuration and data
config = load_config('config/example_config.yaml')
X, Time, Z = load_data('data/sample_data.csv', config)

# Estimate model
result = dfm(X, config, threshold=1e-4, max_iter=1000)

# Extract common factor
common_factor = result.Z[:, 0]
print(f"Common factor extracted: {len(common_factor)} time periods")
```

### Example 2: Working with Results

```python
# Access all result components
factors = result.Z              # (T × r) Factor estimates
loadings = result.C             # (N × r) Factor loadings
transition = result.A           # (r × r) Transition matrix
covariance = result.Q           # (r × r) Factor covariance
smoothed_data = result.X_sm     # (T × N) Smoothed data

# Compute factor contribution to a specific series
series_idx = 0
series_factor_contribution = result.Z @ result.C[series_idx, :].T

# Get factor loadings for interpretation
print("Factor loadings for first series:")
print(result.C[0, :])
```

### Example 3: News Decomposition

```python
from dfm_python import update_nowcast, news_dfm
import pandas as pd

# Estimate model on old data
result_old = dfm(X_old, config)

# Update with new data
result_new = dfm(X_new, config)

# Decompose news (how new data affects forecast)
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old, X_new, result_old, 
    t_fcst=100,  # Forecast time index
    v_news=0     # Target variable index
)

print(f"Forecast update: {y_new - y_old}")
print(f"News contributions: {singlenews}")
```

### Example 4: Handling Missing Data

```python
# The package automatically handles missing data
# Missing values are interpolated using spline interpolation

# Check missing data patterns
import numpy as np
missing_pct = np.isnan(X).sum(axis=0) / X.shape[0] * 100
for i, series_id in enumerate(config.SeriesID):
    print(f"{series_id}: {missing_pct[i]:.1f}% missing")

# Estimate model (missing data handled automatically)
result = dfm(X, config)

# Smoothed data has no missing values
assert np.isnan(result.X_sm).sum() == 0, "All missing values should be filled"
```

## API Reference

### Main Functions

#### `load_config(file: str) -> DFMConfig`

Load configuration from YAML or CSV file.

**Parameters:**
- `file`: Path to configuration file (`.yaml` or `.csv`)

**Returns:**
- `DFMConfig`: Configuration object

**Example:**
```python
config = load_config('config.yaml')
```

#### `load_data(file: str, config: DFMConfig, sample_start: Optional[pd.Timestamp] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]`

Load and transform data from CSV file.

**Parameters:**
- `file`: Path to CSV data file
- `config`: DFM configuration object
- `sample_start`: Optional start date (filters data before this date)

**Returns:**
- `X`: Transformed data matrix (T × N), ready for estimation
- `Time`: Time index (pandas DatetimeIndex)
- `Z`: Original untransformed data (T × N)

**Example:**
```python
X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))
```

#### `dfm(X: np.ndarray, config: DFMConfig, threshold: Optional[float] = None, max_iter: Optional[int] = None) -> DFMResult`

Estimate Dynamic Factor Model using EM algorithm.

**Parameters:**
- `X`: Data matrix (T × N) with possible missing values
- `config`: DFM configuration object
- `threshold`: Convergence threshold (default: 1e-5). Smaller = more precise but slower
- `max_iter`: Maximum EM iterations (default: 5000)

**Returns:**
- `DFMResult`: Result object containing all estimation outputs

**Example:**
```python
result = dfm(X, config, threshold=1e-4, max_iter=1000)
```

### Configuration Classes

#### `DFMConfig`

Main configuration dataclass defining the model structure.

**Attributes:**
- `series`: List of `SeriesConfig` objects
- `block_names`: List of block names (e.g., ["Global", "Consumption"])
- `factors_per_block`: Number of factors per block
- `ar_lag`: AR lag for factors (typically 1)
- `threshold`: EM convergence threshold
- `max_iter`: Maximum EM iterations
- `nan_method`: Missing data handling method (1-5)
- `nan_k`: Spline interpolation parameter

#### `SeriesConfig`

Individual series configuration.

**Attributes:**
- `series_id`: Unique identifier (must match CSV column name)
- `series_name`: Human-readable name
- `frequency`: Frequency code (`d`, `m`, `q`, etc.)
- `transformation`: Transformation code (`pch`, `pca`, etc.)
- `category`: Series category (for organization)
- `units`: Units of measurement
- `blocks`: List of blocks this series loads on

### Result Object

#### `DFMResult`

Result dataclass containing all estimation outputs.

**Key Attributes:**
- `Z`: Smoothed factor estimates (T × r)
- `C`: Factor loadings (N × r)
- `A`: Transition matrix (r × r)
- `Q`: Factor covariance (r × r)
- `R`: Observation covariance (N × N)
- `X_sm`: Smoothed data (T × N)
- `Mx`, `Wx`: Standardization parameters (means and std devs)

**Example:**
```python
result = dfm(X, config)
factors = result.Z           # Extract factors
loadings = result.C          # Extract loadings
smoothed = result.X_sm       # Extract smoothed data
```

## Advanced Features

### News Decomposition

Decompose forecast updates into contributions from individual data releases:

```python
from dfm_python import news_dfm

# Calculate news decomposition
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old,      # Old vintage data
    X_new,      # New vintage data
    result,     # DFM estimation results
    t_fcst=100, # Forecast time index
    v_news=0    # Target variable index (or list for multiple targets)
)

# singlenews: Individual news contributions from each new data point
# y_new - y_old: Total forecast update
```

### Multiple Target Variables

Support for decomposing news for multiple target variables simultaneously:

```python
# Multiple targets
targets = [0, 1, 2]  # Indices of target variables
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old, X_new, result, t_fcst=100, v_news=targets
)

# y_old, y_new: Now arrays of shape (n_targets,)
# singlenews: Array of shape (N, n_targets)
```

### Custom Missing Data Handling

Control how missing data is handled:

```python
from dfm_python import DFMConfig, SeriesConfig

config = DFMConfig(
    series=[...],
    nan_method=2,  # Method: 1=median fill, 2=remove rows + spline, 3=remove all-NaN rows, etc.
    nan_k=3        # Spline interpolation parameter
)
```

### Block Structure Design

Design your factor blocks to match your economic intuition:

```yaml
blocks:
  Global:
    factors: 2        # Common factors affecting all series
  Consumption:
    factors: 1        # Consumption-specific factor
  Investment:
    factors: 1        # Investment-specific factor
```

Series can load on multiple blocks, allowing for rich factor structures.

## Troubleshooting

### Common Issues

#### Import Error: `ModuleNotFoundError: No module named 'utils'`

**Solution**: This was fixed in version 0.1.2. Update your package:
```bash
pip install --upgrade dfm-python
```

#### Quarterly Series Show 100% Missing After Transformation

**Cause**: Quarterly data must have values only at quarter-end months (March, June, September, December).

**Solution**: Ensure your CSV has quarterly values only at quarter-end months, with NaN/empty for other months.

#### Convergence Warnings

**Cause**: EM algorithm may not converge if:
- Data has too much missing data (>50% per series)
- Initial conditions are poor
- Threshold is too strict

**Solution**:
- Check data quality
- Increase `max_iter`
- Relax `threshold` (try 1e-3 instead of 1e-5)
- Review missing data patterns

#### Dimension Mismatch Errors

**Cause**: Configuration doesn't match data structure.

**Solution**:
- Verify `series_id` in config matches CSV column names exactly
- Check that block structure is consistent
- Ensure all series have valid frequency codes

### Getting Help

- Check the [sample data and configuration files](#sample-data) for reference
- Review the [API Reference](#api-reference) for function details
- Ensure you're using the latest version: `pip install --upgrade dfm-python`

## Testing

The package includes comprehensive test suite and sample data:

```bash
# Run all tests
python -m pytest src/test/

# Run sample data test
python src/test/test_synthetic.py

# Test with your own data
python -c "
from dfm_python import load_config, load_data, dfm
config = load_config('config/example_config.yaml')
X, Time, Z = load_data('data/sample_data.csv', config)
result = dfm(X, config, max_iter=50)  # Quick test
print('✓ Test passed!')
"
```

## Architecture

The package is designed to be **generic and application-agnostic**:

- **Core Module** (`dfm_python`): Pure Python implementation, no external dependencies beyond scientific stack
- **Modular Design**: Separate modules for estimation, filtering, news decomposition
- **Extensible**: Easy to add application-specific adapters (database, APIs, etc.)

## Requirements

- **Python**: >= 3.12
- **numpy**: >= 1.24.0
- **pandas**: >= 2.0.0
- **scipy**: >= 1.10.0

### Optional Dependencies

- **hydra-core**: >= 1.3.0 (for Hydra configuration)
- **omegaconf**: >= 2.3.0 (for YAML configuration)
- **pytest**: >= 7.0.0 (for testing)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! The package follows these principles:

- **Core module remains generic**: No application-specific code in core
- **Comprehensive testing**: All features should have tests
- **Documentation**: Code should be well-documented
- **Backward compatibility**: Changes should maintain API compatibility when possible

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting and Forecasting},
  author = {DFM Python Contributors},
  year = {2024},
  url = {https://pypi.org/project/dfm-python/},
  version = {0.1.2}
}
```

## Acknowledgments

This package implements Dynamic Factor Models following established econometric methodology, with a focus on practical nowcasting and forecasting applications.

---

**Package Status**: Stable (v0.1.2)  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.12+
