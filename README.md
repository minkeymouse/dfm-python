# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, production-ready Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting economic and financial time series. This package implements the FRBNY (Federal Reserve Bank of New York) approach to mixed-frequency DFMs with clock-based synchronization and tent kernel aggregation.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Overview: What We Provide](#package-overview-what-we-provide)
- [Detailed Inputs and Outputs](#detailed-inputs-and-outputs)
- [Configuration Guide](#configuration-guide)
- [Data Format Requirements](#data-format-requirements)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Clock-Based Mixed-Frequency Framework](#clock-based-mixed-frequency-framework)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Overview

Dynamic Factor Models (DFM) are powerful econometric tools for analyzing high-dimensional time series data by extracting common factors that drive multiple observed series. This package provides a complete implementation suitable for:

- **Nowcasting**: Forecasting the present using partial information
- **Forecasting**: Multi-step ahead predictions
- **Factor Analysis**: Extracting common trends from multiple series
- **News Decomposition**: Understanding how new data releases affect forecasts

### What is a Dynamic Factor Model?

A DFM models observed time series as:

```
y_t = C × Z_t + e_t    (Observation equation)
Z_t = A × Z_{t-1} + v_t  (State equation)
```

Where:
- `y_t`: Observed time series at time t (n × 1 vector)
- `Z_t`: Unobserved common factors (m × 1 vector)
- `C`: Factor loadings matrix (n × m) - how series relate to factors
- `A`: Transition matrix (m × m) - how factors evolve over time
- `e_t`, `v_t`: Error terms (observation and state innovations)

The model extracts common patterns across multiple series while accounting for their individual dynamics and handling missing data robustly.

## Key Features

### 🎯 Core Capabilities

- **Mixed-Frequency Data**: Seamlessly handle daily, weekly, monthly, quarterly, semi-annual, and annual series simultaneously
- **Clock-Based Synchronization**: All latent factors evolve at a common "clock" frequency (typically monthly)
- **Tent Kernel Aggregation**: Deterministic tent kernels map lower-frequency observations to higher-frequency latent states
- **Robust Missing Data**: Automatic interpolation and handling of missing observations (tested with up to 50% missing data)
- **Block Structure**: Organize series into logical blocks (e.g., Global, Consumption, Investment factors)
- **Flexible Configuration**: Support for both YAML and CSV configuration formats
- **News Decomposition**: Decompose forecast updates into contributions from individual data releases
- **Production Ready**: Comprehensive error handling, logging, validation, and numerical stability

### 📊 Supported Frequencies

- **Daily** (`d`): High-frequency data (e.g., stock prices, exchange rates)
- **Weekly** (`w`): Weekly indicators (e.g., unemployment claims)
- **Monthly** (`m`): Standard economic indicators (e.g., unemployment, industrial production)
- **Quarterly** (`q`): Low-frequency aggregates (e.g., GDP, national accounts)
- **Semi-annual** (`sa`): Semi-annual aggregates
- **Annual** (`a`): Annual aggregates

### 🔧 Data Transformations

The package supports various transformations to ensure stationarity:

- `lin`: Levels (no transformation)
- `chg`: First difference (change)
- `ch1`: Year-over-year change
- `pch`: Percent change
- `pc1`: Year-over-year percent change
- `pca`: Percent change at annual rate
- `cch`: Continuously compounded rate of change
- `cca`: Continuously compounded annual rate
- `log`: Natural logarithm

## Installation

### Basic Installation

```bash
pip install dfm-python
```

### Development Installation

```bash
git clone https://github.com/your-repo/dfm-python.git
cd dfm-python
pip install -e .
```

### Requirements

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

### Optional Dependencies

- **hydra-core** >= 1.3.0 (for Hydra configuration support)
- **omegaconf** >= 2.3.0 (for YAML configuration)
- **pytest** >= 7.0.0 (for testing)

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
factors = result.Z          # Extracted factors (T × m)
loadings = result.C         # Factor loadings (N × m)
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

## Package Overview: What We Provide

### Core Modules

1. **`dfm_python.config`**: Configuration management
   - `DFMConfig`: Main configuration class
   - `SeriesConfig`: Individual series configuration
   - Configuration loading from YAML/CSV
   - Validation and type checking

2. **`dfm_python.data_loader`**: Data handling
   - `load_config()`: Load configuration from files
   - `load_data()`: Load and transform data from CSV
   - `transform_data()`: Apply transformations to time series
   - Automatic date parsing and alignment

3. **`dfm_python.dfm`**: Core estimation
   - `dfm()`: Main estimation function
   - `init_conditions()`: Initial parameter estimation
   - `em_step()`: EM algorithm iteration
   - `DFMResult`: Result dataclass

4. **`dfm_python.kalman`**: Filtering and smoothing
   - `run_kf()`: Complete Kalman filter and smoother
   - `skf()`: Standard Kalman filter (forward pass)
   - `fis()`: Fixed-interval smoother (backward pass)
   - `miss_data()`: Missing data handling

5. **`dfm_python.news`**: News decomposition
   - `news_dfm()`: News decomposition analysis
   - `update_nowcast()`: Update forecasts with new data
   - `para_const()`: Parameter-constrained filtering

6. **`dfm_python.utils`**: Utilities
   - `rem_nans_spline()`: Missing value treatment
   - `summarize()`: Data summary statistics
   - `get_aggregation_structure()`: Tent kernel aggregation
   - `generate_tent_weights()`: Tent weight generation

### Key Classes and Functions

**Main Entry Points:**
- `load_config(file)`: Load configuration
- `load_data(file, config)`: Load and transform data
- `dfm(X, config)`: Estimate DFM model

**Result Object:**
- `DFMResult`: Contains all estimation outputs (factors, loadings, parameters, smoothed data)

**Configuration:**
- `DFMConfig`: Model configuration (series, blocks, parameters)
- `SeriesConfig`: Individual series specification

## Detailed Inputs and Outputs

### Input: Configuration File

The configuration defines the model structure and estimation parameters.

#### YAML Format (Recommended)

```yaml
# config/example_config.yaml
model:
  series:
    gdp_real:
      series_name: "Real GDP (Quarterly)"
      frequency: "q"              # Frequency code
      transformation: "pca"       # Transformation code
      category: "GDP"
      units: "Index (2000=100)"
      blocks: [Global, Consumption, Investment]  # Block loadings
    
    consumption:
      series_name: "Consumption (Monthly)"
      frequency: "m"
      transformation: "pch"
      category: "Consumption"
      units: "Index (2000=100)"
      blocks: [Global, Consumption]

  blocks:
    Global:
      factors: 2                  # Number of factors in this block
    Consumption:
      factors: 1
    Investment:
      factors: 1

dfm:
  ar_lag: 1                       # AR lag for factors
  threshold: 1e-5                 # Convergence threshold
  max_iter: 5000                  # Maximum EM iterations
  nan_method: 2                   # Missing data method
  nan_k: 3                        # Spline parameter
  clock: "m"                      # Global clock frequency
```

#### CSV Format (Alternative)

```csv
series_id,series_name,frequency,transformation,category,units,Block_Global,Block_Consumption,Block_Investment
gdp_real,Real GDP (Quarterly),q,pca,GDP,Index (2000=100),1,1,1
consumption,Consumption (Monthly),m,pch,Consumption,Index (2000=100),1,1,0
investment,Investment (Monthly),m,pch,Investment,Index (2000=100),1,0,1
```

**Configuration Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `series` | List[SeriesConfig] | Series definitions | Required |
| `block_names` | List[str] | Block names (e.g., ["Global", "Consumption"]) | Required |
| `factors_per_block` | List[int] | Number of factors per block | [1, 1, ...] |
| `ar_lag` | int | AR lag for factors | 1 |
| `threshold` | float | EM convergence threshold | 1e-5 |
| `max_iter` | int | Maximum EM iterations | 5000 |
| `nan_method` | int | Missing data method (1-5) | 2 |
| `nan_k` | int | Spline interpolation parameter | 3 |
| `clock` | str | Global clock frequency | "m" |

### Input: Data File

The data file is a CSV with the following structure:

```csv
Date,gdp_real,consumption,investment,stock_index,exchange_rate
2000-01-01,,98.5,95.0,995.87,0.997
2000-02-01,,98.7,95.2,996.75,1.010
2000-03-01,100.5,99.0,95.5,979.65,0.992
2000-04-01,,99.3,95.8,991.26,0.991
...
```

**Data Requirements:**

1. **Date Column**: First column must be named `Date` (case-sensitive)
2. **Column Names**: Must exactly match `series_id` values in configuration
3. **Date Format**: YYYY-MM-DD format (e.g., `2000-01-01`)
4. **Missing Values**: Use empty cells or `NaN` for missing observations
5. **Mixed Frequencies**: 
   - Quarterly: Only include values at quarter-end months (March, June, September, December)
   - Monthly: Include values for all months
   - Daily: Include values for all days (or aggregate to monthly)

### Output: DFMResult Object

The `dfm()` function returns a `DFMResult` dataclass containing all estimation outputs:

```python
@dataclass
class DFMResult:
    # Factor estimates and loadings
    Z: np.ndarray          # Smoothed factors (T × m)
    C: np.ndarray          # Factor loadings (N × m)
    
    # Model parameters
    A: np.ndarray          # Transition matrix (m × m)
    Q: np.ndarray          # Factor covariance (m × m)
    R: np.ndarray          # Observation covariance (N × N)
    
    # Smoothed data
    x_sm: np.ndarray       # Standardized smoothed data (T × N)
    X_sm: np.ndarray       # Unstandardized smoothed data (T × N)
    
    # Initial conditions
    Z_0: np.ndarray        # Initial state (m,)
    V_0: np.ndarray        # Initial covariance (m × m)
    
    # Standardization parameters
    Mx: np.ndarray         # Series means (N,)
    Wx: np.ndarray         # Series standard deviations (N,)
    
    # Model structure
    r: np.ndarray          # Factors per block (n_blocks,)
    p: int                 # AR lag
```

**Output Dimensions:**

| Output | Shape | Description |
|--------|-------|-------------|
| `Z` | (T, m) | Smoothed factor estimates, where T = time periods, m = total factors |
| `C` | (N, m) | Factor loadings, where N = number of series |
| `A` | (m, m) | Transition matrix for factor dynamics |
| `Q` | (m, m) | Covariance of factor innovations |
| `R` | (N, N) | Covariance of observation residuals (diagonal) |
| `X_sm` | (T, N) | Smoothed data (original scale) |
| `x_sm` | (T, N) | Smoothed data (standardized) |

**How to Use Outputs:**

```python
result = dfm(X, config)

# Extract common factor (first factor)
common_factor = result.Z[:, 0]

# Get factor loadings for a specific series
series_idx = 0
loadings = result.C[series_idx, :]  # Shape: (m,)

# Reconstruct a series from factors
series_reconstructed = result.Z @ result.C[series_idx, :].T  # Shape: (T,)

# Get smoothed version of original data
smoothed_data = result.X_sm  # Shape: (T, N)

# Access model parameters
transition_matrix = result.A
factor_covariance = result.Q
observation_covariance = result.R
```

## Configuration Guide

### Series Configuration

Each series requires:

- **`series_id`**: Unique identifier (must match CSV column name)
- **`series_name`**: Human-readable name
- **`frequency`**: One of `d`, `w`, `m`, `q`, `sa`, `a`
- **`transformation`**: Transformation code (see [Data Transformations](#-data-transformations))
- **`blocks`**: List of blocks this series loads on (must include first block = 1)
- **`category`**: Optional category for organization
- **`units`**: Optional units description

### Block Structure

Blocks organize factors into logical groups:

```yaml
blocks:
  Global:
    factors: 2        # Common factors affecting all series
  Consumption:
    factors: 1        # Consumption-specific factor
  Investment:
    factors: 1        # Investment-specific factor
```

**Rules:**
- All series must load on the first block (Global)
- Series can load on multiple blocks
- Each block can have multiple factors

### Clock Parameter

The `clock` parameter defines the base frequency for all latent factors:

```yaml
dfm:
  clock: "m"  # Monthly clock (default)
```

**How Clock Works:**
- All latent factors (global and block-level) evolve at the clock frequency
- Series with frequencies **slower** than clock (e.g., quarterly with monthly clock) use tent kernels
- Series with frequencies **faster** than clock (e.g., daily with monthly clock) use missing data approach
- This synchronization simplifies the Kalman filter and ensures consistent factor evolution

## Data Format Requirements

### CSV Structure

```csv
Date,series_id_1,series_id_2,series_id_3,...
2000-01-01,value_1,value_2,value_3,...
2000-02-01,value_1,value_2,value_3,...
...
```

### Mixed-Frequency Handling

**Quarterly Series:**
- Include values only at quarter-end months (March, June, September, December)
- Leave other months empty/NaN

**Example:**
```csv
Date,gdp_real
2000-01-01,        # Empty - not quarter-end
2000-02-01,        # Empty - not quarter-end
2000-03-01,100.5   # Q1 value
2000-04-01,        # Empty - not quarter-end
2000-05-01,        # Empty - not quarter-end
2000-06-01,101.2   # Q2 value
```

**Monthly Series:**
- Include values for all months

**Daily Series:**
- Include values for all days (or aggregate to monthly for the model)

### Missing Data

Missing values are handled automatically:
- Use empty cells or `NaN` in CSV
- Package interpolates using spline interpolation
- Tested robustly with up to 50% missing data

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
factors = result.Z              # (T × m) Factor estimates
loadings = result.C             # (N × m) Factor loadings
transition = result.A           # (m × m) Transition matrix
covariance = result.Q           # (m × m) Factor covariance
smoothed_data = result.X_sm    # (T × N) Smoothed data

# Compute factor contribution to a specific series
series_idx = 0
series_factor_contribution = result.Z @ result.C[series_idx, :].T

# Get factor loadings for interpretation
print("Factor loadings for first series:")
print(result.C[0, :])
```

### Example 3: News Decomposition

```python
from dfm_python import news_dfm

# Estimate model on old data
result_old = dfm(X_old, config)

# Estimate model on new data
result_new = dfm(X_new, config)

# Decompose news (how new data affects forecast)
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old,      # Old vintage data
    X_new,      # New vintage data
    result_old, # DFM estimation results
    t_fcst=100, # Forecast time index
    v_news=0    # Target variable index (or list for multiple targets)
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

### Example 5: Custom Configuration

```python
from dfm_python.config import DFMConfig, SeriesConfig

# Create configuration programmatically
series_list = [
    SeriesConfig(
        series_id="gdp_real",
        series_name="Real GDP",
        frequency="q",
        transformation="pca",
        category="GDP",
        units="Index",
        blocks=[1, 1, 0]  # Loads on Global and Consumption blocks
    ),
    SeriesConfig(
        series_id="consumption",
        series_name="Consumption",
        frequency="m",
        transformation="pch",
        category="Consumption",
        units="Index",
        blocks=[1, 1, 0]
    ),
]

config = DFMConfig(
    series=series_list,
    block_names=["Global", "Consumption", "Investment"],
    factors_per_block=[2, 1, 1],
    clock="m"
)

# Use configuration
result = dfm(X, config)
```

## API Reference

### Main Functions

#### `load_config(file: Union[str, Path]) -> DFMConfig`

Load configuration from YAML or CSV file.

**Parameters:**
- `file`: Path to configuration file (`.yaml` or `.csv`)

**Returns:**
- `DFMConfig`: Configuration object

**Example:**
```python
config = load_config('config.yaml')
```

#### `load_data(file: Union[str, Path], config: DFMConfig, sample_start: Optional[pd.Timestamp] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]`

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
- `factors_per_block`: Number of factors per block (optional, defaults to 1 per block)
- `ar_lag`: AR lag for factors (typically 1)
- `threshold`: EM convergence threshold (default: 1e-5)
- `max_iter`: Maximum EM iterations (default: 5000)
- `nan_method`: Missing data handling method (1-5, default: 2)
- `nan_k`: Spline interpolation parameter (default: 3)
- `clock`: Global clock frequency (default: "m")

#### `SeriesConfig`

Individual series configuration.

**Attributes:**
- `series_id`: Unique identifier (must match CSV column name)
- `series_name`: Human-readable name
- `frequency`: Frequency code (`d`, `w`, `m`, `q`, `sa`, `a`)
- `transformation`: Transformation code (`pch`, `pca`, etc.)
- `category`: Series category (for organization)
- `units`: Units of measurement
- `blocks`: List of blocks this series loads on (must include first block = 1)

### Result Object

#### `DFMResult`

Result dataclass containing all estimation outputs.

**Key Attributes:**
- `Z`: Smoothed factor estimates (T × m)
- `C`: Factor loadings (N × m)
- `A`: Transition matrix (m × m)
- `Q`: Factor covariance (m × m)
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

**Missing Data Methods:**
- `1`: Replace all missing values with median, then apply moving average
- `2`: Remove rows with >80% NaN, then fill remaining NaNs with spline interpolation (recommended)
- `3`: Remove only rows that are completely NaN (all series missing)
- `4`: Remove all-NaN rows, then fill remaining NaNs with spline interpolation
- `5`: Fill missing values with spline interpolation (no row removal)

## Clock-Based Mixed-Frequency Framework

### Overview

This package implements a **clock-based** mixed-frequency framework following the FRBNY approach. The key innovation is synchronizing all latent factors to a common "clock" frequency, typically monthly.

### How It Works

1. **Global Clock**: All latent factors (global and block-level) evolve at the clock frequency
2. **Tent Kernels**: Lower-frequency observations (e.g., quarterly) are mapped to higher-frequency latent states (e.g., monthly) using deterministic tent kernels
3. **Missing Data Approach**: Higher-frequency observations (e.g., daily) are handled via missing data when clock is slower

### Tent Kernels

Tent kernels are deterministic weight patterns that aggregate lower-frequency observations to higher-frequency latent states. For example, quarterly GDP is connected to 5 monthly latent states with weights `[1, 2, 3, 2, 1]` (peaking at the middle month).

**Supported Frequency Pairs:**
- Quarterly → Monthly: `[1, 2, 3, 2, 1]` (5 periods)
- Semi-annual → Monthly: `[1, 2, 3, 4, 3, 2, 1]` (7 periods)
- Annual → Monthly: `[1, 2, 3, 4, 5, 4, 3, 2, 1]` (9 periods)
- And more (see `TENT_WEIGHTS_LOOKUP` in `utils.aggregation`)

**Maximum Tent Size**: 12 periods (larger gaps use missing data approach)

### Configuration

Set the clock frequency in your configuration:

```yaml
dfm:
  clock: "m"  # Monthly clock (default)
```

**Clock Options:**
- `"d"`: Daily
- `"w"`: Weekly
- `"m"`: Monthly (recommended)
- `"q"`: Quarterly

### Benefits

- **Simplified Kalman Filter**: Single unified filter at clock frequency
- **Consistent Factor Evolution**: All factors evolve synchronously
- **Robust Aggregation**: Deterministic tent kernels ensure proper aggregation
- **Flexible**: Works with any frequency combination

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

#### Insufficient Data Errors

**Cause**: Not enough finite observations for covariance calculation.

**Solution**:
- Check data quality and missing data patterns
- Adjust `nan_method` (try method 2 or 4)
- Ensure sufficient time periods (T >= 10 recommended)

### Getting Help

- Check the [sample data and configuration files](#data-format-requirements) for reference
- Review the [API Reference](#api-reference) for function details
- Ensure you're using the latest version: `pip install --upgrade dfm-python`

## Testing

The package includes comprehensive test suite:

```bash
# Run all tests
python -m pytest src/test/

# Run specific test file
python -m pytest src/test/test_dfm.py

# Run with verbose output
python -m pytest src/test/ -v

# Test with sample data
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

### Module Structure

```
dfm_python/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration classes (DFMConfig, SeriesConfig)
├── data_loader.py       # Data loading and transformation
├── dfm.py               # Core estimation (dfm, init_conditions, em_step)
├── kalman.py            # Kalman filter and smoother
├── news.py              # News decomposition
└── utils/
    ├── __init__.py      # Utility exports
    ├── data_utils.py    # Data preprocessing utilities
    └── aggregation.py   # Tent kernel aggregation
```

## Requirements

- **Python**: >= 3.12
- **numpy**: >= 1.24.0
- **pandas**: >= 2.0.0
- **scipy**: >= 1.10.0

### Optional Dependencies

- **hydra-core**: >= 1.3.0 (for Hydra configuration)
- **omegaconf**: >= 2.3.0 (for YAML configuration)
- **pytest**: >= 7.0.0 (for testing)

## Contributing

Contributions are welcome! The package follows these principles:

- **Core module remains generic**: No application-specific code in core
- **Comprehensive testing**: All features should have tests
- **Documentation**: Code should be well-documented
- **Backward compatibility**: Changes should maintain API compatibility when possible

## License

MIT License - see LICENSE file for details.

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

This package implements Dynamic Factor Models following established econometric methodology, with a focus on practical nowcasting and forecasting applications. The clock-based mixed-frequency framework follows the FRBNY (Federal Reserve Bank of New York) approach.

---

**Package Status**: Stable (v0.1.2)  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.12+
