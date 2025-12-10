# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python implementation of **Dynamic Factor Models (DFM)** and **Deep Dynamic Factor Models (DDFM)** for forecasting high-dimensional time series. Built with PyTorch Lightning.

## Features

### Core Capabilities
- **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual series in one model
- **Clock-based framework**: All factors evolve at a common clock frequency
- **Block structure**: Flexible factor organization (global + sector-specific factors)
- **Idiosyncratic components**: Per-series state augmentation for better fit
- **Preprocessed data**: Users handle preprocessing (imputation, scaling) using sktime or other tools
- **Forecasting**: Generate predictions for any horizon
- **Nowcasting**: Estimate current period values using incomplete data
- **Deep DFM (DDFM)**: Nonlinear encoder with PyTorch for capturing complex factor structures

### Technical Features
- **PyTorch Lightning**: Standard training interface with DataModule and Trainer
- **Multiple configuration methods**: YAML files, Python dictionaries, or Hydra
- **Advanced numerical stability**: Adaptive regularization, spectral radius capping, variance floors
- **Error handling**: Comprehensive error handling, extensive testing, well-documented

## Installation

```bash
pip install dfm-python
```

**Requirements**: 
- Python >= 3.10
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- pytorch-lightning >= 2.0.0
- torch >= 2.0.0
- sktime >= 0.40.1

## Quick Start

### Example 1: Linear DFM

```python
from dfm_python import DFM, DFMDataModule, DFMTrainer
import pandas as pd
import numpy as np

# Load and preprocess data
df = pd.read_csv('data/macro.csv')
df_processed = df[[col for col in df.columns if col != 'date']]

# Create DataModule
dm = DFMDataModule(
    config_path='config/model/dfm.yaml',
    data=df_processed
)
dm.setup()

# Create model and load config
model = DFM()
model.load_config('config/model/dfm.yaml')

# Create trainer and fit
trainer = DFMTrainer(max_iter=10, threshold=1e-4)
trainer.fit(model, dm)

# Predict
Xf, Zf = model.predict(horizon=6)
```

### Example 2: Deep DFM (DDFM)

```python
from dfm_python import DDFM, DFMDataModule, DDFMTrainer
import pandas as pd

# Load and preprocess data
df = pd.read_csv('data/finance.csv')
df_processed = df[[col for col in df.columns if col != 'date']]

# Create DataModule
dm = DFMDataModule(
    config_path='config/model/ddfm.yaml',
    data=df_processed
)
dm.setup()

# Create DDFM model
ddfm_model = DDFM(
    encoder_layers=[64, 32],
    num_factors=2,
    learning_rate=0.005,
    epochs=100
)
ddfm_model.load_config('config/model/ddfm.yaml')

# Create trainer and fit
trainer = DDFMTrainer(max_epochs=10)
trainer.fit(ddfm_model, dm)

# Predict
Xf, Zf = ddfm_model.predict(horizon=6)
```

### Example 3: Nowcasting with update().predict()

```python
from dfm_python import DFM, DFMDataModule, DFMTrainer
import pandas as pd
import numpy as np

# Load data and train model (same as above)
df = pd.read_csv('data/macro.csv')
df_processed = df[[col for col in df.columns if col != 'date']]

dm = DFMDataModule(config_path='config/model/dfm.yaml', data=df_processed)
dm.setup()

model = DFM()
model.load_config('config/model/dfm.yaml')

trainer = DFMTrainer(max_iter=10, threshold=1e-4)
trainer.fit(model, dm)

# Get standardization parameters from trained model
result = model.result
Mx = result.Mx  # Mean for standardization
Wx = result.Wx  # Standard deviation for standardization

# Prepare new data (in practice, this would be real-time incomplete data)
# Standardize using the same parameters from training
X_new_raw = df_processed.iloc[-5:].values  # Last 5 periods as example
X_new_std = (X_new_raw - Mx) / Wx

# Update model state with new standardized data, then predict
# Pattern: model.update(X_std).predict(horizon=1)
X_nowcast, Z_nowcast = model.update(X_new_std).predict(horizon=1)

# Extract nowcast for target series
target_idx = df_processed.columns.get_loc('KOEQUIPTE')
nowcast_value = X_nowcast[0, target_idx]

print(f"Nowcast value: {nowcast_value:.6f}")
```

## Configuration

### YAML Configuration (Recommended)

Create a YAML configuration file:

```yaml
# config/model/dfm.yaml
clock: m  # Monthly clock frequency
blocks:
  Block_Global:
    factors: 2
    ar_lag: 1
    clock: m

series:
  - series_id: KOEQUIPTE
    frequency: m
    transformation: lin
    blocks: [1]
  - series_id: KOGDP___D
    frequency: q
    transformation: lin
    blocks: [1]

augment_idio: true
augment_idio_slow: true
```

### Python Configuration

```python
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig

config = DFMConfig(
    clock='m',
    series=[
        SeriesConfig(series_id='series1', frequency='m', transformation='lin', blocks=[1]),
        SeriesConfig(series_id='series2', frequency='q', transformation='lin', blocks=[1]),
    ],
    blocks={'Block_Global': BlockConfig(factors=2, ar_lag=1, clock='m')},
    augment_idio=True,
    augment_idio_slow=True
)
```

## Data Format

### CSV Format

Your data CSV should have:
- **First column**: `date` (YYYY-MM-DD format)
- **Subsequent columns**: One per time series, column names must match `series_id` in configuration
- **Missing values**: Empty cells or `NaN`

**Example**:
```csv
date,KOEQUIPTE,KOGDP___D
2000-01-01,100.2,
2000-02-01,100.5,
2000-03-01,100.8,100.5
2000-04-01,101.0,
```

**Important**: 
- Data must be preprocessed before passing to the package
- Use sktime or other tools for imputation, scaling, and transformations
- Quarterly series should only have values at quarter-end months

## Tutorials

The package includes tutorial scripts demonstrating complete workflows:

```bash
# Linear DFM tutorial
python tutorial/tutorial_macro_dfm.py
python tutorial/tutorial_finance_dfm.py

# Deep DFM tutorial
python tutorial/tutorial_macro_ddfm.py
python tutorial/tutorial_finance_ddfm.py
```

## API Reference

### Core Classes

- **`DFM`**: Linear Dynamic Factor Model (EM algorithm)
- **`DDFM`**: Deep Dynamic Factor Model (PyTorch encoder)
- **`DFMDataModule`**: PyTorch Lightning DataModule for data handling
- **`DFMTrainer`**: Trainer for DFM (EM algorithm)
- **`DDFMTrainer`**: Trainer for DDFM (gradient descent)

### Key Methods

```python
# Configuration
model.load_config(source)  # Load from YAML, dict, or Hydra config

# Training
trainer.fit(model, datamodule)  # Standard Lightning pattern

# Prediction
Xf, Zf = model.predict(horizon=6)  # Forecast future values

# Nowcasting (update state with new data, then predict)
result = model.result
X_new_std = (X_new_raw - result.Mx) / result.Wx  # Standardize new data
X_nowcast, Z_nowcast = model.update(X_new_std).predict(horizon=1)  # Update and predict
```

### Result Objects

```python
# DFMResult / DDFMResult
result.Z          # (T+1 × m) Smoothed factor estimates
result.C          # (N × m) Factor loadings matrix
result.A          # (m × m) Factor transition matrix
result.Q          # (m × m) Innovation covariance
result.R          # (N × N) Observation covariance
result.converged  # bool: Convergence status
result.num_iter   # int: Number of iterations

# NowcastResult
result.nowcast_value        # Estimated value
result.confidence_interval  # Confidence interval
result.factors_at_view     # Factor state at view date
```

## Mixed-Frequency Data Handling

### Clock Frequency and Target Series

The package supports mixed-frequency data where the **clock frequency** (the frequency at which factors evolve) can differ from the **target series frequency**. The clock must always be the **minimum frequency** among all series (i.e., the fastest frequency).

### Supported Frequency Combinations

| Frequencies | Clock | Target | Status | Description |
|------------|-------|--------|--------|-------------|
| `[d]` | `d` | `d` | ✅ | All daily series |
| `[d, w]` | `d` | `d` | ✅ | Daily clock, weekly series (7-day uniform average) |
| `[w, m]` | `w` | `m` | ✅ | Weekly clock, monthly series (4-week uniform average) |
| `[m]` | `m` | `m` | ✅ | All monthly series |
| `[m, q]` | `m` | `q` | ✅ | Monthly clock, quarterly series (5-month tent kernel) |

### Weekly Clock with Monthly Series

When `clock='w'` and target series is `frequency='m'`:

**Data Structure:**
- All data is organized at weekly frequency (T × N, where T is weekly time points)
- Weekly series: Full weekly observations (week-over-week changes)
- Monthly series: 3-4 NaN values per month, 1 observation per month (week-over-week changes)

**Model Interpretation:**
- **Weekly data**: Week-over-week changes (전월비 증감)
- **Monthly data**: 4-week uniform average of weekly changes (주간 전월비 증감의 4주 평균)
- **DFM factors**: Weekly frequency, week-over-week changes
- **Monthly observation**: Connected to 4 weekly state variables via uniform weights `[1, 1, 1, 1]`

**Tent Kernel:**
```python
('m', 'w'): np.array([1, 1, 1, 1])  # 4-week uniform average
# Normalized: [0.25, 0.25, 0.25, 0.25]
```

**Prediction:**
- `predict(horizon=4)` returns 4 weekly forecasts (clock frequency)
- Monthly series predictions are also at weekly frequency
- **To get monthly forecast**: Average the 4 weekly predictions for that month
- The monthly prediction equals the average of 4 weekly factors (due to uniform average constraint)

**Mathematical Relationship:**
```
Monthly observation = c × (f_t + f_{t-1} + f_{t-2} + f_{t-3})
                     = c × 4 × mean(f_t, f_{t-1}, f_{t-2}, f_{t-3})
                     = mean(f_t, f_{t-1}, f_{t-2}, f_{t-3})  (when c = 0.25)
```

Where:
- `f_t` is the weekly factor at time t
- `c` is the loading coefficient (constrained to be uniform via R_mat)
- The constraint `R_mat @ Cc = q` ensures `Cc[0] = Cc[1] = Cc[2] = Cc[3]` (uniform average)

### Monthly Clock with Quarterly Series

When `clock='m'` and target series is `frequency='q'`:

**Tent Kernel:**
```python
('q', 'm'): np.array([1, 2, 3, 2, 1])  # 5-month tent kernel
# Normalized: [0.111, 0.222, 0.333, 0.222, 0.111]
```

**Model Interpretation:**
- Quarterly observation = weighted average of 5 monthly factors (tent shape, peaking at middle month)
- This is the traditional DFM case (quarterly → monthly)

### Prediction Behavior

**Important Notes:**
1. **Predictions are always at clock frequency**: `predict(horizon=4)` with `clock='w'` returns 4 weekly forecasts
2. **No NaN in predictions**: All series (including slower-frequency ones) get predictions at clock frequency
3. **Aggregation for target frequency**: To get target-frequency forecasts, aggregate clock-frequency predictions:
   - Monthly target with weekly clock: Average 4 weekly predictions per month
   - Quarterly target with monthly clock: Apply tent kernel `[1, 2, 3, 2, 1]` to 5 monthly predictions

**Example:**
```python
# clock='w', target='m', predict(horizon=4)
X_forecast = model.predict(horizon=4)  # Returns (4 × N) weekly forecasts

# Get monthly forecast for target series
monthly_idx = 2  # Index of monthly target series
weekly_forecasts = X_forecast[:, monthly_idx]  # 4 weekly values
monthly_forecast = np.mean(weekly_forecasts)  # Average to get monthly value
```

## Architecture

### Core Modules

- **`models/`**: Model implementations
  - `base.py`: BaseFactorModel (common interface)
  - `dfm.py`: Linear DFM (EM algorithm)
  - `ddfm.py`: Deep DFM (PyTorch encoder)
  
- **`ssm/`**: State-space model components
  - `kalman.py`: Kalman filter and smoother
  - `em.py`: EM algorithm implementation
  
- **`config/`**: Configuration management
  - `schema.py`: Configuration dataclasses
  - `results.py`: Result dataclasses
  
- **`lightning/`**: PyTorch Lightning integration
  - `data_module.py`: DFMDataModule
  - `scaling.py`: Data scaling utilities

## Testing

Run the test suite:

```bash
pytest src/test/ -v
```

Run specific tests:

```bash
pytest src/test/test_nowcast_implementation.py -v
pytest src/test/test_models.py -v
pytest src/test/test_trainer.py -v
```

## Troubleshooting

### Convergence Issues

- Increase `max_iter` in `DFMTrainer`
- Relax `threshold` (default: 1e-4)
- Check data quality and preprocessing

### Numerical Instability

- The package includes automatic stability features:
  - Adaptive regularization for ill-conditioned matrices
  - Spectral radius capping (< 0.99) for stationarity
  - Variance floors for all covariance matrices

### Missing Data

- Preprocess data before passing to the package
- Use sktime for imputation and scaling
- Verify frequency settings match your data

## Project Status

**Version**: 0.5.4  
**Status**: Stable  
**Python**: 3.10+  

### What's New in 0.5.4

- **Version consistency**: Unified version numbers across all files
- **Documentation updates**: Updated README and package documentation
- **Stability improvements**: Enhanced numerical stability and error handling

### What's New in 0.5.1

- **New `update()` method**: Replaces legacy `nowcast()` with flexible `update().predict()` pattern
- **Improved API**: Users now control all preprocessing (masking, imputation, standardization)
- **Code cleanup**: Removed legacy code and overengineering (~200+ lines removed)
- **Bug fixes**: Fixed DDFM `update()` method for single-factor models
- **Updated tutorials**: All 4 tutorials rewritten with new pattern and VAR(1) configuration
- **VAR(1) only**: Simplified to VAR(1) factor dynamics throughout  

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest src/test/ -q`
- Code follows existing patterns and style
- New features include tests
- Documentation is updated
