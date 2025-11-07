# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting high-dimensional time series. This package implements the FRBNY (Federal Reserve Bank of New York) approach with clock-based synchronization and tent kernel aggregation for mixed-frequency data.

## Table of Contents

- [Overview](#overview)
- [Dynamic Factor Model Theory](#dynamic-factor-model-theory)
- [Core Features](#core-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [License](#license)

## Overview

Dynamic Factor Models extract common latent factors from multiple observed time series, enabling:
- **Nowcasting**: Real-time forecasting of current economic conditions using partial information
- **Forecasting**: Multi-step ahead predictions of key economic indicators
- **Factor Analysis**: Identification of common trends driving multiple series
- **News Decomposition**: Attribution of forecast changes to individual data releases

The model handles high-dimensional datasets (dozens to hundreds of series) with mixed frequencies (daily, weekly, monthly, quarterly) and robustly manages missing data.

## Dynamic Factor Model Theory

### Mathematical Framework

A Dynamic Factor Model represents observed time series as:

```
y_t = C × Z_t + e_t          (Observation equation)
Z_t = A × Z_{t-1} + v_t      (State equation)
```

Where:
- **y_t**: n × 1 vector of observed series at time t
- **Z_t**: m × 1 vector of unobserved common factors
- **C**: n × m factor loading matrix (how series relate to factors)
- **A**: m × m transition matrix (how factors evolve over time)
- **e_t**: n × 1 observation errors ~ N(0, R)
- **v_t**: m × 1 state innovations ~ N(0, Q)

### State-Space Representation

The DFM is a state-space model where:
- **State**: Latent factors Z_t evolve according to a VAR(p) process
- **Observation**: Observed series y_t are linear combinations of factors plus noise
- **Estimation**: Parameters (C, A, Q, R) estimated via Expectation-Maximization (EM) algorithm

### Expectation-Maximization Algorithm

The EM algorithm iteratively:
1. **E-step**: Run Kalman filter/smoother to estimate factors Z_t given current parameters
2. **M-step**: Update parameters (C, A, Q, R) to maximize likelihood given estimated factors
3. **Convergence**: Iterate until log-likelihood change falls below threshold

This yields maximum likelihood estimates of all parameters and smoothed factor estimates.

### Block Structure

Factors are organized into blocks representing different economic domains:
- **Global factors**: Common to all series (e.g., business cycle)
- **Sector factors**: Specific to groups of series (e.g., consumption, investment)
- **Idiosyncratic components**: Series-specific dynamics

Each series loads on one or more blocks, allowing flexible factor structures.

### Mixed-Frequency Framework

The package implements a **clock-based** approach where:
- All latent factors evolve at a common "clock" frequency (typically monthly)
- Lower-frequency observations (e.g., quarterly) map to higher-frequency states via deterministic tent kernels
- Higher-frequency observations (e.g., daily) handled via missing data approach

**Tent Kernels**: For quarterly-to-monthly mapping, quarterly observations connect to 5 monthly latent states with weights [1, 2, 3, 2, 1], peaking at the middle month. This ensures quarterly aggregates equal weighted sums of monthly states.

## Core Features

### Mixed-Frequency Data Handling

Seamlessly combine series with different reporting frequencies:
- **Daily** (d): High-frequency financial data
- **Weekly** (w): Weekly indicators
- **Monthly** (m): Standard economic indicators
- **Quarterly** (q): National accounts, GDP
- **Semi-annual** (sa): Semi-annual aggregates
- **Annual** (a): Annual aggregates

The model automatically handles frequency conversion using tent kernels for lower frequencies and missing data interpolation for higher frequencies.

### Robust Missing Data

Missing observations are handled via:
- Spline interpolation for gaps in time series
- Kalman filter naturally accounts for missing data in likelihood
- Tested with up to 50% missing data per series

### Block Structure

Organize factors into logical groups:
- Global factors affecting all series
- Sector-specific factors (consumption, investment, external, etc.)
- Flexible loading structure: series can load on multiple blocks

### Data Transformations

Support for standard econometric transformations:
- `lin`: Levels (no transformation)
- `chg`: First difference
- `pch`: Percent change
- `pca`: Percent change at annual rate
- `log`: Natural logarithm
- And more (ch1, pc1, cch, cca)

Transformations ensure stationarity and proper scaling of series.

### News Decomposition

Decompose forecast updates into contributions from individual data releases:
- Identify which new data points drive forecast changes
- Quantify impact of each data release
- Essential for understanding nowcast revisions

## Installation

```bash
pip install dfm-python
```

### Requirements

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## Quick Start

```python
from dfm_python import load_config, load_data, dfm

# Load configuration and data
config = load_config('config/example_config.yaml')
X, Time, mnemonics = load_data('data/example_data.csv', config)

# Estimate model
result = dfm(X, config, threshold=1e-5, max_iter=5000)

# Access results
factors = result.Z          # Smoothed factors (T × m)
loadings = result.C         # Factor loadings (N × m)
smoothed = result.X_sm      # Smoothed data (T × N)
rmse = result.rmse          # Model fit (RMSE)
```

## Configuration

### YAML Format

```yaml
# DFM estimation parameters
dfm:
  clock: "m"              # Base frequency for latent factors
  threshold: 1e-5         # EM convergence threshold
  max_iter: 5000          # Maximum EM iterations
  ar_lag: 1               # AR lag for factors

# Model structure
model:
  blocks:
    Global:
      factors: 3          # Number of factors in block
    Consumption:
      factors: 2
    Investment:
      factors: 2

# Series definitions
series:
  gdp_real:
    series_name: "Real GDP (Quarterly)"
    frequency: "q"
    transformation: "pca"
    units: "Index"
    blocks: [Global]
  
  consumption:
    series_name: "Consumption (Monthly)"
    frequency: "m"
    transformation: "pch"
    units: "Index"
    blocks: [Global, Consumption]
```

### Key Parameters

- **clock**: Base frequency for all latent factors (typically "m" for monthly)
- **threshold**: Convergence criterion for EM algorithm (smaller = more precise)
- **max_iter**: Maximum EM iterations (typically 100-5000)
- **ar_lag**: Autoregressive lag for factors (typically 1 for AR(1))

### Numerical Stability Parameters

For reproducibility and transparency, all numerical stability techniques are configurable:

```yaml
dfm:
  # AR Coefficient Clipping
  clip_ar_coefficients: true  # Enable AR coefficient clipping for stationarity
  ar_clip_min: -0.99          # Minimum AR coefficient (must be > -1)
  ar_clip_max: 0.99           # Maximum AR coefficient (must be < 1)
  warn_on_ar_clip: true       # Warn when clipping occurs
  
  # Data Value Clipping
  clip_data_values: true      # Enable clipping of extreme data values
  data_clip_threshold: 100.0  # Clip values beyond this many standard deviations
  warn_on_data_clip: true     # Warn when clipping occurs
  
  # Regularization
  use_regularization: true     # Enable regularization for numerical stability
  regularization_scale: 1e-6   # Scale factor for ridge regularization
  min_eigenvalue: 1e-8        # Minimum eigenvalue for positive definite matrices
  max_eigenvalue: 1e6         # Maximum eigenvalue cap
  warn_on_regularization: true # Warn when regularization is applied
  
  # Damped Updates
  use_damped_updates: true    # Enable damped updates when likelihood decreases
  damping_factor: 0.8        # Damping factor (0.8 = 80% new, 20% old)
  warn_on_damped_update: true # Warn when damped updates are used
```

**Important**: These parameters are fully transparent and documented. When enabled, the package logs warnings explaining when and why each technique is applied, allowing researchers to understand any potential bias introduced. All parameters can be disabled or adjusted for research purposes.

## Data Format

### CSV Structure

Data file should be CSV with series as columns and dates as rows:

```csv
Date,gdp_real,consumption,investment
2000-01-01,,98.5,95.0
2000-02-01,,98.7,95.2
2000-03-01,100.5,99.0,95.5
2000-04-01,,99.3,95.8
```

**Requirements**:
- First column: `Date` (YYYY-MM-DD format)
- Column names: Must match `series_id` in configuration
- Missing values: Use empty cells or NaN
- Mixed frequencies: Quarterly series only have values at quarter-end months (Mar, Jun, Sep, Dec)

## Usage Examples

### Basic Estimation

```python
from dfm_python import load_config, load_data, dfm

# Load configuration and data
config = load_config('config/example_config.yaml')
X, Time, mnemonics = load_data('data/example_data.csv', config)

# Estimate model
result = dfm(X, config, threshold=1e-5, max_iter=5000)

# Extract common factor (first factor)
common_factor = result.Z[:, 0]

# Reconstruct series from factors
series_reconstructed = result.Z @ result.C[0, :].T
```

### Working with Results

```python
# Access all components
factors = result.Z              # (T × m) Factor estimates
loadings = result.C             # (N × m) Factor loadings
transition = result.A           # (m × m) Transition matrix
covariance = result.Q           # (m × m) Factor covariance
smoothed_data = result.X_sm    # (T × N) Smoothed data

# Model fit statistics
overall_rmse = result.rmse
rmse_per_series = result.rmse_per_series

# Convergence info
converged = result.converged
iterations = result.num_iter
loglikelihood = result.loglik
```

### News Decomposition

```python
from dfm_python import news_dfm

# Estimate on old and new data
result_old = dfm(X_old, config)
result_new = dfm(X_new, config)

# Decompose news
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old,      # Old vintage data
    X_new,      # New vintage data
    result_old, # DFM results
    t_fcst=100, # Forecast time index
    v_news=0    # Target variable index
)

# Forecast update
forecast_update = y_new - y_old

# News contributions from each series
for i, contribution in enumerate(singlenews):
    if not np.isnan(contribution):
        print(f"Series {i}: {contribution:.4f}")
```

## API Reference

### Main Functions

#### `load_config(file: str) -> DFMConfig`

Load configuration from YAML file.

**Parameters**:
- `file`: Path to YAML configuration file

**Returns**: `DFMConfig` object

#### `load_data(file: str, config: DFMConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]`

Load and transform data from CSV file.

**Parameters**:
- `file`: Path to CSV data file
- `config`: DFM configuration object

**Returns**: 
- `X`: Transformed data matrix (T × N)
- `Time`: Time index (DatetimeIndex)
- `mnemonics`: List of series IDs

#### `dfm(X: np.ndarray, config: DFMConfig, threshold: float = None, max_iter: int = None) -> DFMResult`

Estimate Dynamic Factor Model using EM algorithm.

**Parameters**:
- `X`: Data matrix (T × N) with possible missing values
- `config`: DFM configuration object
- `threshold`: Convergence threshold (default: from config)
- `max_iter`: Maximum EM iterations (default: from config)

**Returns**: `DFMResult` object containing all estimation outputs

### Result Object

#### `DFMResult`

Dataclass containing all estimation outputs:

**Factor Estimates**:
- `Z`: Smoothed factors (T × m)
- `C`: Factor loadings (N × m)

**Model Parameters**:
- `A`: Transition matrix (m × m)
- `Q`: Factor covariance (m × m)
- `R`: Observation covariance (N × N)

**Smoothed Data**:
- `X_sm`: Unstandardized smoothed data (T × N)
- `x_sm`: Standardized smoothed data (T × N)

**Initial Conditions**:
- `Z_0`: Initial state (m,)
- `V_0`: Initial covariance (m × m)

**Standardization**:
- `Mx`: Series means (N,)
- `Wx`: Series standard deviations (N,)

**Model Structure**:
- `r`: Factors per block (n_blocks,)
- `p`: AR lag

**Convergence Info**:
- `converged`: Whether EM converged
- `num_iter`: Number of iterations
- `loglik`: Final log-likelihood

**Model Fit**:
- `rmse`: Overall RMSE
- `rmse_per_series`: RMSE per series (N,)

## Advanced Features

### Clock-Based Mixed-Frequency Framework

All latent factors evolve at a common clock frequency (typically monthly). Lower-frequency observations map to higher-frequency states via deterministic tent kernels:

- **Quarterly → Monthly**: 5-period tent with weights [1, 2, 3, 2, 1]
- **Semi-annual → Monthly**: 7-period tent
- **Annual → Monthly**: 9-period tent
- **Maximum tent size**: 12 periods (larger gaps use missing data approach)

This ensures quarterly aggregates equal weighted sums of monthly latent states.

### Kalman Filtering and Smoothing

The package uses standard Kalman filter for forward pass and fixed-interval smoother for backward pass:
- **Filter**: Computes filtered state estimates and covariances
- **Smoother**: Computes smoothed estimates using all available data
- **Missing data**: Handled naturally by Kalman filter

### Block Structure and Factor Loading

Each series loads on one or more blocks:
- All series must load on first block (global factor)
- Series can load on multiple blocks simultaneously
- Each block can have multiple factors
- Loadings estimated via PCA (monthly) or constrained least squares (quarterly)

### Idiosyncratic Components

Each series has an idiosyncratic component modeled as AR(1) process:
- Monthly series: Single AR(1) component
- Quarterly series: 5 AR(1) components (one per tent period) sharing same AR coefficient

## Troubleshooting

### Convergence Issues

If EM algorithm doesn't converge:
- Check data quality and missing data patterns
- Increase `max_iter` (try 10000)
- Relax `threshold` (try 1e-3 instead of 1e-5)
- Review initial conditions

### Dimension Mismatch Errors

Ensure:
- `series_id` in config matches CSV column names exactly
- Block structure is consistent
- All series have valid frequency codes

### Numerical Instability

The package includes comprehensive numerical stability features that are fully configurable:

**Automatic Stability Techniques**:
- **AR Coefficient Clipping**: Ensures stationarity by clipping AR coefficients to [-0.99, 0.99]
- **Data Value Clipping**: Clips extreme outliers beyond configurable threshold (default: 100 std devs)
- **Regularization**: Adds ridge regularization to prevent ill-conditioned matrices
- **Positive Definite Enforcement**: Ensures covariance matrices remain positive semi-definite
- **Damped Updates**: Prevents likelihood decreases by using weighted parameter updates

**Transparency**:
- All techniques are configurable via YAML config
- Warnings are logged when techniques are applied
- Statistics are tracked for monitoring
- Techniques can be disabled for research purposes

**Best Practices**:
- Monitor warnings to understand when stability techniques are applied
- If techniques are frequently needed, investigate data quality
- Adjust thresholds based on your data characteristics
- Consider disabling techniques if they introduce unacceptable bias

If issues persist, check data for outliers, extreme values, or near-unit root behavior.

## Architecture

The package is modular and application-agnostic:

**Core Modules**:
- `config.py`: Configuration management
- `data_loader.py`: Data loading and transformation
- `dfm.py`: Core estimation (EM algorithm)
- `kalman.py`: Kalman filter and smoother
- `news.py`: News decomposition
- `utils/`: Utility functions (aggregation, data preprocessing)

**Design Principles**:
- Generic core (no application-specific code)
- Robust error handling
- Comprehensive numerical stability
- Clean, maintainable code structure

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting and Forecasting},
  author = {DFM Python Contributors},
  year = {2025},
  url = {https://pypi.org/project/dfm-python/},
  version = {0.1.5}
}
```

## Acknowledgments

This package implements Dynamic Factor Models following established econometric methodology, with a focus on practical nowcasting and forecasting applications. The clock-based mixed-frequency framework follows the FRBNY (Federal Reserve Bank of New York) approach.

---

**Package Status**: Stable (v0.1.5+)  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.12+
