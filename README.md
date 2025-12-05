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

**Key Innovation**: All latent factors evolve at a common "clock" frequency (configurable: `'d'`, `'w'`, `'m'`, `'q'`, `'sa'`, `'a'`), while observed series can be at different frequencies. This is achieved through deterministic tent kernel aggregation, allowing quarterly GDP to be modeled using monthly latent factors, or any slower-frequency series to be modeled using higher-frequency latent factors.

## Features

### Core Capabilities
- ✅ **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual series in one model
- ✅ **Clock-based framework**: All factors evolve at a common clock frequency (configurable: daily, weekly, monthly, quarterly, semi-annual, annual)
- ✅ **Block structure**: Flexible factor organization (global common factor + sector-specific factors)
- ✅ **Idiosyncratic components**: Per-series state augmentation for better fit (AR(1) for clock-frequency, tent-length chains for slower frequencies)
- ✅ **Preprocessed data**: Package expects preprocessed data from users - users handle all preprocessing (imputation, scaling, feature engineering) using sktime or other tools
- ✅ **News decomposition**: Attribute forecast changes to specific data releases
- ✅ **Nowcasting & Forecasting**: Generate predictions for any horizon
- ✅ **Deep Dynamic Factor Models (DDFM)**: Nonlinear encoder with PyTorch for capturing complex factor structures (optional, requires `dfm-python[deep]`)

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
- `torch>=2.0.0` (for Deep Dynamic Factor Models - install with `pip install dfm-python[deep]`)

## Quick Start

### Example 1: Simple Monthly Data

```python
import dfm_python as dfm
import numpy as np
from datetime import datetime
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
forecasts = result.forecast(horizon=None)  # Default: 1 year of periods based on clock frequency
```

### Example 2: Deep Dynamic Factor Model (DDFM)

```python
import dfm_python as dfm
from dfm_python import create_model, DFMConfig, SeriesConfig, BlockConfig
import numpy as np

# Generate or load your data
X = np.random.randn(100, 10)  # 100 months, 10 series

# Create configuration
series = [
    SeriesConfig(series_id=f'series_{i}', frequency='m', transformation='lin', blocks=[1])
    for i in range(10)
]
config = DFMConfig(
    series=series,
    blocks=[BlockConfig(block_name='Global', factors=1)],
    clock='m',
    factors_per_block=[1],
)

# Create and fit DDFM model
ddfm = create_model('ddfm', encoder_layers=[64, 32], num_factors=1, epochs=100)
result = ddfm.fit(X, config)

# Access results (same structure as DFM)
factors = result.Z          # (100 × 1) Extracted factors
loadings = result.C         # (10 × 1) Factor loadings
X_forecast, Z_forecast = ddfm.predict(horizon=12)
```

**Note**: DDFM requires PyTorch. Install with `pip install dfm-python[deep]`.

### Example 3: Mixed-Frequency Data (Monthly + Quarterly)

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

# Forecast (horizon=None defaults to 1 year of periods based on clock frequency)
X_forecast, Z_forecast = dfm.predict(horizon=None)  # Or specify: horizon=12
```

### Example 3: Using YAML Configuration (Recommended)

```python
import dfm_python as dfm

# 1. Load configuration from YAML file
dfm.from_yaml('config/default.yaml')

# 2. Load data
dfm.load_data('data/data.csv', sample_start='2020-01-01', sample_end='2023-12-31')

# 3. Train
dfm.train()

# 4. Forecast and visualize (horizon=None defaults to 1 year of periods)
X_forecast, Z_forecast = dfm.predict(horizon=None)  # Or specify: horizon=12
dfm.plot(kind='factor', factor_index=0, forecast_horizon=None, save_path='factor_forecast.png')

# 5. Access results
result = dfm.get_result()
print(f"Converged: {result.converged}, Iterations: {result.num_iter}")
print(f"Log-likelihood: {result.loglik:.2f}")
```

### Example 4: Fast Reuse with `load_pickle`

Once a model has been trained you can reload it instantly for forecasting/nowcasting without
re-running EM:

```python
import dfm_python as dfm
import pickle
from pathlib import Path

# After training
outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
save_path = outputs_dir / "model_result.pkl"
payload = {
    'result': dfm.get_result(),
    'config': dfm.get_config(),
}
with open(save_path, 'wb') as f:
    pickle.dump(payload, f)  # Or rely on the auto-save printed after train()

# Later (e.g., in production job)
model = dfm.DFM()
model.load_pickle(
    save_path,
    data=dfm.get_data(),        # Or model.load_data(...)
    time_index=dfm.get_time(),  # Optional if stored in pickle
)

# Fast inference (milliseconds instead of minutes)
X_forecast, Z_forecast = model.predict(horizon=12)
gdp_nowcast = model.nowcast(
    target_series='gdp',
    view_date='2024-03-15'
)
```

`DFM.load_pickle()` accepts either the auto-save payload (`{'result': ..., 'config': ...}`) or a
raw `DFMResult`. Provide `load_data_path` or `data`/`time_index` when you need nowcasting/news
decomposition abilities (which require the data view).

## Configuration Guide

### Configuration Methods

The package supports multiple ways to configure your model:

#### 1. YAML Configuration

**Best for**: Users who prefer declarative configuration files

```python
import dfm_python as dfm

# Load from YAML
dfm.from_yaml('config/default.yaml')
dfm.load_data('data/sample_data.csv')
dfm.train()
```

#### 2. Hydra Decorator (Advanced)

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
    # horizon=None defaults to 1 year of periods based on clock frequency
    forecast_horizon = cfg.get('forecast_horizon', None)
    X_forecast, Z_forecast = dfm.predict(horizon=forecast_horizon)

if __name__ == "__main__":
    main()
```

**CLI Overrides**: `python script.py max_iter=10 threshold=1e-4 blocks.Block_Global.factors=2`

#### 3. Direct Python Objects

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
  - **Generic support**: The package automatically adapts forecast horizons, period calculations, and date handling based on the clock frequency
  - **Default behavior**: When `horizon=None` in `predict()`, defaults to 1 year of periods based on clock frequency:
    - Monthly (`'m'`): 12 periods
    - Quarterly (`'q'`): 4 periods
    - Semi-annual (`'sa'`): 2 periods
    - Annual (`'a'`): 1 period
    - Daily (`'d'`): 365 periods (approximate)
    - Weekly (`'w'`): 52 periods (approximate)

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
- **Data must be preprocessed**: Users must handle missing values, scaling, and feature engineering before passing data to the package. Use sktime or other tools for preprocessing.

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

### Nowcasting API

The package provides a comprehensive nowcasting API for pseudo real-time evaluation and downstream model integration:

#### Basic Nowcasting

```python
import dfm_python as dfm
from dfm_python import Nowcast

# Load config and data, train model
dfm.from_yaml('config/default.yaml')
dfm.load_data('data/data.csv')
dfm.train()

# Get Nowcast instance
model = dfm.DFM()
nowcast = model.nowcast  # or: nowcast = Nowcast(model)

# Calculate nowcast (simple callable interface)
value = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
print(f"Nowcast value: {value:.4f}")

# Or get full result with metadata
result = nowcast('gdp', view_date='2024-01-15', target_period='2024Q1', return_result=True)
print(f"Nowcast: {result.nowcast_value:.4f}")
print(f"Data availability: {result.data_availability}")
print(f"Factors at view: {result.factors_at_view}")
```

#### News Decomposition

Decompose forecast updates into contributions from individual data releases:

```python
# News decomposition between two data views
news = nowcast.decompose(
    target_series='gdp',
    target_period='2024Q1',
    view_date_old='2024-01-15',
    view_date_new='2024-02-15'
)

# Access results (NewsDecompResult)
print(f"Old forecast: {news.y_old:.4f}")
print(f"New forecast: {news.y_new:.4f}")
print(f"Change: {news.change:.4f}")
print(f"Top contributors: {news.top_contributors}")

# Or get dictionary format (backward compatibility)
news_dict = nowcast.decompose(
    target_series='gdp',
    target_period='2024Q1',
    view_date_old='2024-01-15',
    view_date_new='2024-02-15',
    return_dict=True
)
```

#### Pseudo Real-Time Backtesting

Perform backtesting to evaluate model performance in pseudo real-time:

```python
from datetime import datetime

# Perform backtest
backtest_result = nowcast.backtest(
    target_series='gdp',
    target_date='2024Q4',
    backward_steps=20,  # 20 backward steps
    higher_freq=True,   # Use frequency one step faster than clock
    include_actual=True # Compare with actual values
)

# Access metrics
print(f"Overall RMSE: {backtest_result.overall_rmse:.4f}")
print(f"Overall MAE: {backtest_result.overall_mae:.4f}")
print(f"Failed steps: {backtest_result.failed_steps}")

# Access point-wise metrics
print(f"RMSE per step: {backtest_result.rmse_per_step}")
print(f"MAE per step: {backtest_result.mae_per_step}")

# Access individual nowcast results
for i, nowcast_result in enumerate(backtest_result.nowcast_results):
    print(f"Step {i}: nowcast={nowcast_result.nowcast_value:.4f}, "
          f"view_date={nowcast_result.view_date}")

# Access news decomposition between steps
for i, news_result in enumerate(backtest_result.news_results):
    if news_result is not None:
        print(f"Step {i}: forecast change={news_result.change:.4f}")

# Visualize results
backtest_result.plot(save_path='backtest_results.png', show=False)
```

#### Pseudo Real-Time Evaluation (Alternative Method)

Generate datasets for model evaluation across multiple periods:

```python
from datetime import datetime

# Define evaluation periods
from dfm_python.engine.time import datetime_range
periods = datetime_range(start=datetime(2020, 1, 1), end=datetime(2023, 12, 31), freq='QE')

# Generate evaluation dataset
dataset = model.generate_dataset(
    target_series='gdp',
    periods=periods,
    backward=4,  # 4 backward data views per period
    forward=0    # No forward forecast for evaluation
)

# Access features and targets
X_features = dataset['X']  # (n_samples, n_features)
y_baseline = dataset['y_baseline']  # DFM baseline predictions
y_actual = dataset['y_actual']  # Actual values
y_target = dataset['y_target']  # Prediction error (for downstream models)
```

### Model Result Storage

Save and load model results for later use:

#### File-Based Storage (Default)

```python
from adapters import PickleModelResultSaver

# Initialize saver (defaults to './model_results')
saver = PickleModelResultSaver(base_dir='./model_results')

# Save model result
result_id = saver.save_model_result(
    result=dfm.get_result(),
    config=dfm.get_config(),
    metadata={'tag': 'baseline', 'experiment': 'exp1'}
)

# Load model result
result, config, metadata = saver.load_model_result(result_id)

# List all saved results
result_ids = saver.list_model_results()
```

#### SQLite Storage (Optional - Future Feature)

SQLite storage is planned for future releases. Currently, file-based storage is fully functional and recommended for production use.

```python
# Note: SQLiteAdapter is a placeholder for future development
# File-based storage (PickleModelResultSaver) is the recommended approach

# When SQLite support is available (future release):
# from adapters import SQLiteAdapter
# adapter = SQLiteAdapter(database_path='./dfm_results.db')
# result_id = adapter.save_model_result(result, config, metadata)
# result, config, metadata = adapter.load_model_result(result_id)
```

### Data View Management

Manage time-point specific data views for pseudo real-time evaluation:

#### File-Based Data Views (Default)

```python
from adapters import BasicDataViewManager

# Initialize with data
X, Time, Z = dfm.get_data(), dfm.get_time(), dfm.get_original_data()
manager = BasicDataViewManager(data_source=(X, Time, Z))

# Get data view at specific date
X_view, Time_view, Z_view = manager.get_data_view(
    view_date='2024-01-15',
    config=dfm.get_config()
)

# Data view filters data based on release_date in SeriesConfig
# Series with release_date > view_date are masked (set to NaN)
```

#### SQLite Data Views (Optional - Future Feature)

SQLite data view storage is planned for future releases. Currently, `BasicDataViewManager` provides in-memory data views with caching, which is sufficient for most use cases.

```python
# Note: SQLite data view storage is a placeholder for future development
# BasicDataViewManager with in-memory caching is the recommended approach

# When SQLite support is available (future release):
# from adapters import SQLiteAdapter
# adapter = SQLiteAdapter(database_path='./dfm_data.db')
# view_id = adapter.save_data_view(view_date, X_view, Time_view, config)
# X_view, Time_view, config = adapter.load_data_view(view_id)
```

### Advanced News Decomposition

For advanced users, low-level functions are available in the `nowcasting` module:

```python
from dfm_python.nowcasting import para_const

# Before new data release
result_old = model.fit(X_old, config)

# After new data release
X_new = ...  # Updated data

# Use para_const for news calculation
Res_old = para_const(X_old, result_old, lag=1)
# ... (see nowcast.py for full implementation)

# Recommended: Use Nowcast class for most use cases (see Nowcasting API section above)
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
import pandas as pd

# Configuration
dfm.from_yaml(yaml_path)                    # Load from YAML file
dfm.from_dict(config_dict)                  # Load from dictionary
dfm.load_config(hydra=cfg)                  # Load from Hydra DictConfig

# Data
dfm.load_data(data_path, sample_start=None, sample_end=None)

# Estimation
dfm.train(max_iter=None, threshold=None, **kwargs)

# Forecasting (horizon=None defaults to 1 year of periods based on clock frequency)
dfm.predict(horizon=None)                   # Returns (X_forecast, Z_forecast)
# Or specify explicit horizon: dfm.predict(horizon=12)

# Visualization
dfm.plot(kind='factor', factor_index=0, forecast_horizon=None, save_path=None)

# Nowcasting
model = dfm.DFM()
nowcast = model.nowcast                     # Get Nowcast instance
value = nowcast('gdp', view_date='2024-01-15')  # Simple nowcast
result = nowcast.backtest('gdp', '2024Q4', backward_steps=20)  # Backtesting

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
- **`core/helpers.py`**: Helper functions (datetime conversion, time index extraction, config access)
- **`core/timestamp.py`**: Timestamp utilities (datetime operations, frequency mapping, period calculations)
- **`kalman.py`**: Kalman filter and smoother (`run_kf`, `skf`, `fis`)
- **`data.py`**: Data loading and preprocessing (`load_data`, `transform_data`, `rem_nans_spline`)
- **`utils.py`**: Mixed-frequency utilities (tent weights, aggregation structure, idio chain lengths, frequency helpers)
- **`api.py`**: High-level convenience API and module-level functions
- **`nowcast.py`**: Nowcasting, news decomposition, and backtesting functionality
  - `Nowcast` class: Unified interface for nowcasting operations
  - `NowcastResult`, `NewsDecompResult`, `BacktestResult`: Structured result dataclasses
  - `backtest()`: Pseudo real-time evaluation framework

### Key Design Principles

1. **Clock-based synchronization**: All factors evolve at a common frequency (generic support for any frequency)
2. **Tent kernel aggregation**: Deterministic weights for slower-frequency series
3. **Idiosyncratic components**: Per-series state augmentation for better fit
4. **Generic frequency handling**: 
   - No hardcoded assumptions about specific frequencies (e.g., monthly)
   - All period calculations, forecast horizons, and date operations adapt to clock frequency
   - Helper functions provide generic frequency operations (`get_periods_per_year()`, `get_annual_factor()`, `clock_to_datetime_freq()`)
5. **Advanced numerical stability**: 
   - Adaptive ridge regularization (condition-number-based)
   - Q matrix floor (0.01 for factors) to prevent scale issues
   - C matrix normalization (||C[:,j]|| = 1) for clock-frequency factors
   - Spectral radius capping (< 0.99) for stationarity
   - Variance floors for all covariance matrices
6. **Flexible configuration**: Multiple input methods (YAML, CSV, Dict, Hydra)
7. **Backward compatibility**: Old code continues to work, new features are opt-in

## Deep Dynamic Factor Models (DDFM)

The package includes support for **Deep Dynamic Factor Models (DDFM)**, which use nonlinear encoders to extract factors from observed data. DDFM is particularly useful when the relationship between observed variables and latent factors is nonlinear.

**Note**: DDFM has a **separate high-level API class** (`dfm_python.DDFM`) from the linear DFM class (`dfm_python.DFM`). Both classes share the same interface (load_config, load_data, train, predict, plot) but use different underlying models.

### When to Use DDFM vs Linear DFM

- **Use Linear DFM** when:
  - The relationship between factors and observations is approximately linear
  - You need interpretable factor loadings
  - Computational speed is important
  - You have limited data

- **Use DDFM** when:
  - The relationship between factors and observations is nonlinear
  - You suspect complex interactions between variables
  - You have sufficient data for training neural networks
  - You want to capture non-Gaussian factor structures

### DDFM Architecture

The DDFM implementation uses:
- **Nonlinear Encoder**: Multi-layer perceptron (MLP) to extract factors from observations
- **Linear Decoder**: Linear transformation from factors back to observations (for interpretability)
- **Linear Dynamics**: VAR(1) or VAR(2) dynamics for factors (estimated via OLS)
- **Idiosyncratic Modeling**: AR(1) dynamics for residual components (optional)
- **Kalman Smoothing**: Final smoothing step using Kalman filter with full state-space (factors + idio)

**Key Features:**
- **VAR(2) Support**: Can model factor dynamics with up to 2 lags using companion form
- **Idiosyncratic Components**: Models residual components with AR(1) dynamics for better fit
- **Decoder Parameter Extraction**: Directly extracts observation matrix from trained decoder (no OLS re-estimation)
- **Full State-Space**: Complete state-space model with factors and idio in state vector

### DDFM Examples

#### Example 1: Basic DDFM Usage

```python
from dfm_python import create_model, DFMConfig, SeriesConfig, BlockConfig
import numpy as np

# Create data
X = np.random.randn(200, 15)  # 200 time periods, 15 series

# Create config
series = [
    SeriesConfig(series_id=f'series_{i}', frequency='m', transformation='lin', blocks=[1])
    for i in range(15)
]
config = DFMConfig(
    series=series,
    blocks={'Block_Global': BlockConfig(factors=2, ar_lag=1, clock='m')},
    clock='m',
)

# Create DDFM model
ddfm = create_model(
    'ddfm',
    encoder_layers=[64, 32, 16],  # Encoder architecture
    num_factors=2,                 # Number of factors
    activation='tanh',             # Activation function
    epochs=100,                    # Training epochs
    batch_size=32,                 # Batch size
    learning_rate=0.001,           # Learning rate
)

# Fit model
result = ddfm.fit(X, config)

# Use results (same API as linear DFM)
factors = result.Z
loadings = result.C
X_forecast, Z_forecast = ddfm.predict(horizon=12)
```

### Testing DDFM

The package includes synthetic DGP tests for validating DDFM:

```python
from dfm_python.engine.synthetic_dgp import SyntheticDGP
from dfm_python import create_model

# Create synthetic data with known factors
dgp = SyntheticDGP(seed=42, n=10, r=1, poly_degree=2)  # Nonlinear factors
X = dgp.simulate(200)

# Fit DDFM
ddfm = create_model('ddfm', encoder_layers=[64, 32], num_factors=1)
result = ddfm.fit(X, config)

# Evaluate factor recovery
r2 = dgp.evaluate(result.Z)  # Trace R² metric
print(f"Factor recovery R²: {r2:.3f}")
```

## Package Structure

The package is organized into clear layers for better maintainability and extensibility:

- **`engine/`**: Pure mathematical/algorithmic code shared by all factor models
  - EM algorithm, Kalman filter, numerical utilities
  - Time series utilities, diagnostics, synthetic DGP
  - Model-agnostic low-level engine code

- **`models/`**: Factor model implementations
  - `base.py`: BaseFactorModel interface
  - `dfm.py`: Linear Dynamic Factor Model (EM-based)
  - `ddfm.py`: Deep Dynamic Factor Model (PyTorch-based, fully integrated)
  
**Note**: The original DDFM implementation (TensorFlow/Keras) has been archived to `archive/DDFM_original/` for reference. The current implementation is fully integrated in `src/dfm_python/models/ddfm.py`.

- **`nowcasting/`**: Nowcasting and news decomposition domain logic
  - `nowcast.py`: Nowcast class and core logic
  - News decomposition and backtest functionality

- **`api.py` / `__init__.py`**: Public high-level API
  - `DFM` class, `create_model()` factory
  - Module-level convenience functions

- **`core/`**: Backward compatibility layer (re-exports from `engine/`)
  - Maintains old import paths for existing code
  - New code should use `engine.*` directly

## Testing

Run the full test suite:

```bash
pytest src/test/ -v
```

Run specific test categories:

```bash
# Core DFM estimation
pytest src/test/test_api.py -v

# Kalman filter/smoother
pytest src/test/test_kalman.py -v

# DDFM tests (requires PyTorch)
pytest src/test/test_ddfm.py -v

# Regression tests (ensure backward compatibility)
pytest src/test/test_dfm_regression.py -v

# Integration tests
pytest src/test/test_integration.py -v
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

### Basic Tutorial (Hydra-based)

The tutorial demonstrates the complete DFM workflow using Hydra for configuration management:

```bash
# Using default YAML config
python tutorial/basic_tutorial.py \
  --config-path config \
  --config-name default \
  data_path=data/sample_data.csv

# With CLI overrides
python tutorial/basic_tutorial.py \
  --config-path config \
  --config-name default \
  data_path=data/sample_data.csv \
  max_iter=10 \
  threshold=1e-4 \
  blocks.Block_Global.factors=2
```

Comprehensive tutorial covering:
- **Hydra-based configuration**: All configs managed through YAML files
- **CLI parameter overrides**: Override any parameter via command line
- **Data preparation and transformation**
- **Model training and convergence**
- **Understanding DFM results**: Detailed explanation of all result components
- **Forecasting and visualization**
- **Model result storage**: Save/load model results
- **Data view management**: Pseudo real-time evaluation

## Storage and Adapters

The package provides flexible storage adapters for model results and data views:

### Model Result Storage

- **File-based (default)**: `PickleModelResultSaver` - Saves to pickle files in `./model_results/`
  - Fully functional and recommended for production use
  - No additional dependencies required
- **SQLite (optional - future)**: `SQLiteAdapter` - Planned for future releases
  - Currently a placeholder for future development
  - Will require `dfm-python[db]` installation when available

### Data View Management

- **File-based (default)**: `BasicDataViewManager` - In-memory data views with caching
  - Fully functional and recommended for production use
  - Efficient caching for repeated access
  - No additional dependencies required
- **SQLite (optional - future)**: `SQLiteAdapter` - Planned for future releases
  - Currently a placeholder for future development
  - Will provide persistent data view storage when available

### Environment Variables

For database integration, set these environment variables:

```bash
# Database configuration
DATABASE_TYPE=sqlite  # or postgresql
DATABASE_PATH=./dfm_results.db  # For SQLite
# For PostgreSQL:
# DATABASE_HOST=localhost
# DATABASE_PORT=5432
# DATABASE_NAME=dfm_db
# DATABASE_USER=user
# DATABASE_PASSWORD=password

# Storage type selection
MODEL_RESULT_STORAGE_TYPE=file  # or database
DATA_VIEW_SOURCE=file  # or database
```

**Note**: File-based storage works out of the box with no configuration needed. Database storage requires `dfm-python[db]` installation and environment variable configuration.

## Project Status

**Version**: 0.3.1  
**Status**: Stable and production-ready  
**Python**: 3.10+  
**PyPI**: https://pypi.org/project/dfm-python/

### Recent Improvements (v0.3.1)

- ✅ **Pandas-backed Data Views + Kalman Cache**  
  `create_data_view()` now applies release-date masks via pandas, avoiding per-view NumPy copies. `Nowcast.backtest()` reuses cached Kalman filter/smoother states across successive data views, giving large speed-ups for 12-step pseudo real-time tests.
- ✅ **Backtest Trajectory Plot**  
  `BacktestResult.plot_trajectory()` draws 이전 vs. 업데이트 nowcast curves plus monthly actuals along a relative week axis (`-12주 … 현재`). The tutorial simply calls this helper—no more hand-built plotting code.
- ✅ **Higher-Frequency Backsteps**  
  `Nowcast.backtest(..., backward_steps=12, higher_freq=True)` now emits weekly snapshots even when the model clock is monthly, so you can inspect 12주 worth of “pseudo releases” immediately.
- ✅ **Reload & Reuse**  
  `DFM.load_pickle()` restores config, time index, and cached pandas data views so you can call `predict()`/`nowcast()` right after loading without re-reading CSVs.
- ✅ **API + Docs Refresh**  
  README/tutorial sections now highlight the faster backtests, new trajectory plots, and guidelines for interpreting flattened Q-matrix values or “strange” news decomposition outputs.

### Previous Improvements (v0.2.9)

- ✅ **Generic Clock Frequency Support**: Full support for any clock frequency (daily, weekly, monthly, quarterly, semi-annual, annual)
  - Forecast horizons default to 1 year of periods based on clock frequency (generic, not hardcoded)
  - All period calculations and date handling are clock-frequency-agnostic
  - Helper functions for generic frequency operations (`get_periods_per_year()`, `get_annual_factor()`, `clock_to_datetime_freq()`)
- ✅ **Nowcasting API**: Comprehensive nowcasting API with `nowcast()`, `generate_dataset()`, and `get_state()` methods
- ✅ **Model Result Storage**: File-based and SQLite adapters for saving/loading model results
- ✅ **Data View Management**: Time-point specific data views for pseudo real-time evaluation
- ✅ **Helper Function Consolidation**: Reduced code duplication with consolidated helper functions
  - `to_python_datetime()`: Generic datetime conversion (handles pandas Timestamp, strings, etc.)
  - `extract_last_date()`: Generic time index extraction
  - `clock_to_datetime_freq()`: Shared clock-to-datetime frequency mapping
  - `get_periods_per_year()`: Generic periods-per-year calculation
  - `get_annual_factor()`: Generic annualization factor calculation
  - `get_next_period_end()`: Generic period end calculation
- ✅ **Generic Naming**: All docstrings and naming made generic (no feature-specific references)
- ✅ **Code Refactoring**: Removed all hardcoded month assumptions and magic numbers

### Previous Improvements (v0.2.7)

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
  version = {0.3.1}
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
