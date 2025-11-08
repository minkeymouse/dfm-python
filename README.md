# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting high-dimensional time series. Implements clock-based synchronization and tent kernel aggregation for mixed-frequency data.

## Features

- **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual (no daily/weekly)
- **Clock-based framework**: All factors evolve at a common clock frequency
- **Block structure**: Flexible factor organization (global, sector-specific)
- **Robust missing data**: Spline interpolation and Kalman filter handling
- **News decomposition**: Attribute forecast changes to data releases
- **Generic design**: YAML configs or direct object creation, extensible via adapters

## Installation

```bash
pip install dfm-python
```

**Requirements**: Python >= 3.12, numpy >= 1.24.0, pandas >= 2.0.0, scipy >= 1.10.0

## Quick Start

### Option 1: YAML Configuration

```python
from dfm_python import load_config, load_data, dfm

# Load configuration from YAML
config = load_config('config.yaml')
X, Time, _ = load_data('data.csv', config)

# Estimate model
result = dfm(X, config)

# Access results
factors = result.Z          # (T × m) Smoothed factors
loadings = result.C         # (N × m) Factor loadings
smoothed = result.X_sm      # (T × N) Smoothed data
```

### Option 2: Direct Configuration

```python
from dfm_python import DFMConfig, SeriesConfig, load_data, dfm

# Create configuration programmatically
config = DFMConfig(
    series=[
        SeriesConfig(
            series_id='gdp_real',
            series_name='Real GDP',
            frequency='q',
            transformation='pca',
            category='National Accounts',
            units='Index',
            blocks=[1]  # Loads on first block (global)
        ),
        SeriesConfig(
            series_id='consumption',
            series_name='Consumption',
            frequency='m',
            transformation='pch',
            category='Consumption',
            units='Index',
            blocks=[1]
        )
    ],
    block_names=['Global'],
    clock='m',
    threshold=1e-5,
    max_iter=5000
)

# Load data and estimate
X, Time, _ = load_data('data.csv', config)
result = dfm(X, config)
```

## Configuration

### YAML Format

```yaml
dfm:
  clock: "m"              # Clock frequency (base for all factors)
  threshold: 1e-5        # EM convergence threshold
  max_iter: 5000         # Maximum EM iterations
  ar_lag: 1             # AR lag for factors

series:
  gdp_real:
    series_name: "Real GDP"
    frequency: "q"
    transformation: "pca"
    category: "National Accounts"
    units: "Index"
    blocks: [Global]
  
  consumption:
    series_name: "Consumption"
    frequency: "m"
    transformation: "pch"
    category: "Consumption"
    units: "Index"
    blocks: [Global]
```

### Key Parameters

- **clock**: Base frequency for latent factors (typically "m" for monthly)
- **threshold**: EM convergence criterion (default: 1e-5)
- **max_iter**: Maximum EM iterations (default: 5000)
- **ar_lag**: Autoregressive lag for factors (default: 1)

### Numerical Stability

All numerical stability techniques are configurable and transparent:

```yaml
dfm:
  clip_ar_coefficients: true    # Ensure stationarity
  clip_data_values: true         # Clip extreme outliers
  use_regularization: true       # Prevent ill-conditioned matrices
  use_damped_updates: true      # Prevent likelihood decreases
```

Warnings are logged when techniques are applied. All parameters can be disabled for research purposes.

## Data Format

### File-Based (CSV)

Data file with series as columns and dates as rows:

```csv
Date,gdp_real,consumption,investment
2000-01-01,,98.5,95.0
2000-02-01,,98.7,95.2
2000-03-01,100.5,99.0,95.5
```

**Requirements**:
- First column: `Date` (YYYY-MM-DD format)
- Column names: Must match `series_id` in configuration
- Missing values: Empty cells or NaN
- Mixed frequencies: Quarterly series only at quarter-end months

**Frequency Support**:
- **Supported**: Monthly (m), Quarterly (q), Semi-annual (sa), Annual (a)
- **Not supported**: Daily (d), Weekly (w) - raises ValueError
- All series must have frequency ≤ clock frequency

### Database-Backed

For production applications, implement adapters that return:
- `X`: Transformed data matrix (T × N)
- `Time`: Time index (pandas DatetimeIndex)
- `Z`: Original untransformed data (T × N)

## Usage Examples

### Basic Estimation

```python
from dfm_python import load_config, load_data, dfm

config = load_config('config.yaml')
X, Time, _ = load_data('data.csv', config)
result = dfm(X, config)

# Extract common factor
common_factor = result.Z[:, 0]

# Model fit
rmse = result.rmse
```

### News Decomposition

```python
from dfm_python import news_dfm

# Estimate on old and new data
result_old = dfm(X_old, config)
result_new = dfm(X_new, config)

# Decompose news
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old, result_old, t_fcst=100, v_news=0
)

# Forecast update
forecast_update = y_new - y_old
```

## API Reference

### `load_config(configfile) -> DFMConfig`

Load configuration from YAML file or return existing DFMConfig object.

**Parameters**:
- `configfile`: Path to YAML file (.yaml, .yml) or `DFMConfig` object

**Returns**: `DFMConfig` object

### `load_data(datafile, config, sample_start=None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]`

Load and transform time series data.

**Parameters**:
- `datafile`: Path to data file (CSV supported)
- `config`: `DFMConfig` object
- `sample_start`: Optional start date (YYYY-MM-DD) to filter data

**Returns**: `(X, Time, Z)` where:
- `X`: Transformed data matrix (T × N)
- `Time`: Time index (DatetimeIndex)
- `Z`: Original untransformed data (T × N)

### `dfm(X, config, threshold=None, max_iter=None) -> DFMResult`

Estimate Dynamic Factor Model using EM algorithm.

**Parameters**:
- `X`: Data matrix (T × N) with possible missing values
- `config`: `DFMConfig` object
- `threshold`: Convergence threshold (default: from config)
- `max_iter`: Maximum iterations (default: from config)

**Returns**: `DFMResult` object

### `DFMResult` Object

Contains all estimation outputs:

- **Factors**: `Z` (T × m), `C` (N × m)
- **Parameters**: `A` (m × m), `Q` (m × m), `R` (N × N)
- **Smoothed Data**: `X_sm` (T × N), `x_sm` (T × N)
- **Initial Conditions**: `Z_0` (m,), `V_0` (m × m)
- **Standardization**: `Mx` (N,), `Wx` (N,)
- **Convergence**: `converged`, `num_iter`, `loglik`
- **Model Fit**: `rmse`, `rmse_per_series`

## Architecture

**Core Modules**:
- `config.py`: Configuration management
- `data_loader.py`: Data loading and transformation
- `dfm.py`: Core estimation (EM algorithm)
- `kalman.py`: Kalman filter and smoother
- `news.py`: News decomposition

**Design Principles**:
- Generic core (no application-specific code)
- Flexible configuration (YAML or direct object creation)
- Extensible via application-specific adapters
- Robust error handling and numerical stability

## Troubleshooting

**Convergence Issues**:
- Increase `max_iter` (try 10000)
- Relax `threshold` (try 1e-3)
- Check data quality and missing data patterns

**Dimension Mismatch**:
- Ensure `series_id` in config matches data column names
- Verify block structure consistency
- Check frequency codes are valid

**Numerical Instability**:
- Monitor warnings for stability techniques
- Adjust thresholds based on data characteristics
- Investigate data quality if techniques frequently needed

## License

MIT License

## Citation

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting and Forecasting},
  author = {DFM Python Contributors},
  year = {2025},
  url = {https://pypi.org/project/dfm-python/}
}
```

---

**Package Status**: Stable  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.12+
