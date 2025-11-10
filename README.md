# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting high-dimensional time series. Implements clock-based synchronization and tent kernel aggregation for mixed-frequency data.

## Features

- **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual
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

### Module-Level API (Recommended)

```python
import dfm_python as dfm
from dfm_python.config import Params

# 1. Load config from spec CSV + Params
params = Params(max_iter=5000, threshold=1e-5)
dfm.from_spec('data/sample_spec.csv', params=params)

# 2. Load data
dfm.load_data('data/sample_data.csv', sample_start='2021-01-01', sample_end='2022-12-31')

# 3. Train
dfm.train(max_iter=5000)

# 4. Forecast
X_forecast, Z_forecast = dfm.predict(horizon=12)

# 5. Plot
dfm.plot(kind='factor', factor_index=0, forecast_horizon=12, save_path='outputs/factor.png')

# 6. Access results
result = dfm.get_result()
factors = result.Z          # (T × m) Smoothed factors
loadings = result.C         # (N × m) Factor loadings
smoothed = result.X_sm     # (T × N) Smoothed data
```

### Alternative: YAML Configuration

```python
import dfm_python as dfm

# Load from YAML
dfm.from_yaml('config/default.yaml')
dfm.load_data('data/sample_data.csv')
dfm.train()
X_forecast, Z_forecast = dfm.predict(horizon=12)
```

### Alternative: Direct Class Usage

```python
from dfm_python import DFM, DFMConfig, SeriesConfig, BlockConfig

# Create config
config = DFMConfig(
    series=[SeriesConfig(frequency='m', transformation='lin', blocks=[1], series_id='s1')],
    blocks={'Block_Global': BlockConfig(factors=1, clock='m')}
)

# Fit model
model = DFM()
result = model.fit(X, config, max_iter=100)
factors = result.Z
```

## Configuration

### Spec CSV + Params (Recommended for CSV workflows)

```python
from dfm_python.config import Params

params = Params(
    max_iter=5000,
    threshold=1e-5,
    regularization_scale=1e-5,
    damping_factor=0.8
)
dfm.from_spec('data/sample_spec.csv', params=params)
```

### Hydra Decorator (Recommended for Hydra users)

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
```

**CLI Overrides**: `python script.py max_iter=10 threshold=1e-4 blocks.Block_Global.factors=2`

### Dictionary

```python
config_dict = {
    'clock': 'm',
    'max_iter': 5000,
    'blocks': {'Block_Global': {'factors': 1, 'clock': 'm'}},
    'series': [{'series_id': 's1', 'frequency': 'm', 'transformation': 'lin', 'blocks': [0]}]
}
dfm.from_dict(config_dict)
```

## Key Parameters

- **clock**: Base frequency for latent factors (typically "m" for monthly)
- **threshold**: EM convergence criterion (default: 1e-5)
- **max_iter**: Maximum EM iterations (default: 5000)
- **ar_lag**: Autoregressive lag for factors (default: 1)
- **regularization_scale**: Scale factor for ridge regularization (default: 1e-5)
- **damping_factor**: Damping factor for parameter updates (default: 0.8)

All numerical stability parameters are configurable and transparent. See `Params` dataclass for full list.

## Data Format

CSV file with series as columns and dates as rows:

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

**Frequency Support**: Monthly (m), Quarterly (q), Semi-annual (sa), Annual (a)

## News Decomposition

```python
from dfm_python import news_dfm

# Decompose forecast updates
y_old, y_new, singlenews, actual, forecast, weight, t_miss, v_miss, innov = news_dfm(
    X_old, result_old, t_fcst=100, v_news=0
)

# Forecast update
forecast_update = y_new - y_old
```

## API Reference

### Module-Level Functions

- `dfm.from_spec(spec_path, params=None)` - Load from spec CSV + Params
- `dfm.from_yaml(yaml_path)` - Load from YAML file
- `dfm.from_dict(config_dict)` - Load from dictionary
- `dfm.load_data(data_path, sample_start=None, sample_end=None)` - Load data
- `dfm.train(max_iter=None, threshold=None, **kwargs)` - Train model
- `dfm.predict(horizon=12)` - Forecast series and factors
- `dfm.plot(kind='factor', factor_index=0, forecast_horizon=None, save_path=None)` - Plot results
- `dfm.get_result()` - Get `DFMResult` object

### DFMResult Object

Contains all estimation outputs:
- **Factors**: `Z` (T × m), `C` (N × m)
- **Parameters**: `A` (m × m), `Q` (m × m), `R` (N × N)
- **Smoothed Data**: `X_sm` (T × N), `x_sm` (T × N)
- **Convergence**: `converged`, `num_iter`, `loglik`
- **Model Fit**: `rmse`, `rmse_per_series`

## Architecture

**Core Modules**:
- `config.py`: Configuration management (DFMConfig, SeriesConfig, BlockConfig, Params)
- `dfm.py`: Core estimation (DFM class with fit() method)
- `api.py`: High-level convenience API
- `kalman.py`: Kalman filter and smoother
- `news.py`: News decomposition

**Core Submodules**:
- `core/em.py`: EM algorithm core
- `core/numeric.py`: Numerical utilities
- `core/diagnostics.py`: Diagnostic functions
- `core/results.py`: Result metrics
- `core/grouping.py`: Frequency grouping utilities

## Testing

```bash
pytest src/test/ -v
```

Tests are organized into 5 files:
- `test_dfm.py` - Core DFM estimation
- `test_kalman.py` - Kalman filter/smoother
- `test_news.py` - News decomposition
- `test_config.py` - Configuration and API
- `test_api.py` - Edge cases and tutorials

## Tutorials

1. **`tutorial/basic_tutorial.py`** - Spec CSV + Params approach
   ```bash
   python tutorial/basic_tutorial.py --spec data/sample_spec.csv --data data/sample_data.csv
   ```

2. **`tutorial/hydra_tutorial.py`** - Hydra decorator approach
   ```bash
   python tutorial/hydra_tutorial.py max_iter=10 threshold=1e-4
   ```

## Troubleshooting

**Convergence Issues**: Increase `max_iter`, relax `threshold`, check data quality

**Dimension Mismatch**: Ensure `series_id` matches data column names, verify block structure

**Numerical Instability**: Monitor warnings, adjust thresholds, investigate data quality

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

**Package Status**: Stable (v0.2.1)  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.12+  
**Latest Changes**: See [CHANGELOG.md](CHANGELOG.md)
