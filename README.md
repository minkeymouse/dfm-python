# dfm-python: Dynamic Factor Models for Python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of **Dynamic Factor Models (DFM)** for nowcasting and forecasting high-dimensional time series. Implements clock-based synchronization and tent kernel aggregation for mixed-frequency data.

## Features

- **Mixed-frequency data**: Monthly, quarterly, semi-annual, annual
- **Clock-based framework**: All factors evolve at a common clock frequency
- **Block structure**: Flexible factor organization (global, sector-specific)
- **Robust missing data**: Spline interpolation and Kalman filter handling
- **News decomposition**: Attribute forecast changes to data releases
- **Generic design**: YAML configs or direct object creation, extensible via adapters
- **Numerical stability**: Comprehensive error handling and robust matrix operations
- **Well-documented**: Extensive docstrings and tutorials

## Project Goals (prioritized) & Iteration Policy

Top 10 functional goals (priority order):
1. Class-oriented DFM supporting DictConfig (Hydra) and Spec + class configs.
2. Plausible, stable results: no complex numbers; sensible EM convergence; enforce Q diag ≥ 1e-8.
3. Generic Block DFM working with any valid CSV schema (given correct spec).
4. Full pipeline: initialization, Kalman filter/smoother, EM, nowcasting, forecasting.
5. Complete APIs: visualization, results access, nowcasting, forecasting (MATLAB parity where helpful).
6. Mixed frequency: generalized clock with tent weights for slower series over manageable gaps.
7. Missing data handling: robust preprocessing plus KF handling; aligned with Nowcast behavior.
8. Frequency guardrails: any series faster than the clock raises a clear error.
9. Generic naming/logic/patterns; no overengineering; ready for varied environments.
10. Structure: ≤ 20 Python files under `src/` (src/dfm_python + src/test) without sacrificing functionality or tests.

Iteration policy:
- Every change must pass:
  - `pytest src/test -q`
  - `python tutorial/basic_tutorial.py --spec data/sample_spec.csv --data data/sample_data.csv --max-iter 1`
- Tests and tutorial are protected assets: do not delete or move them.
- File count rule: total Python files under `src/` (src/dfm_python + src/test) must be ≤ 20.

## Installation

```bash
pip install dfm-python
```

**Requirements**: Python >= 3.10, numpy >= 1.24.0, pandas >= 2.0.0, scipy >= 1.10.0

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
- `config.py` and `config_sources.py`: Configuration dataclasses and adapters (YAML/Dict/CSV/Hydra)
- `dfm.py`: Core estimation (DFM class with fit() method; EM/KF helpers)
- `kalman.py`: Kalman filter and smoother
- `news.py`: News decomposition
- `api.py`: High-level convenience API and module-level functions

Note: Some functionality historically referenced under `dfm_python.core.*` has been consolidated into the modules above. No separate `core/` package is required.

**Key Improvements** (v0.2.3+):
- Consolidated covariance/variance computation functions for robustness
- Named constants replacing magic numbers for better maintainability
- Enhanced docstrings with comprehensive examples
- Improved error handling and numerical stability
- Unified logging message formats

## Testing

```bash
pytest src/test/ -v
```

Tests are organized into modules under `src/test/`:
- `test_dfm.py` - Core DFM estimation
- `test_kalman.py` - Kalman filter/smoother
- `test_numeric.py` - Numerical utilities and stability
- `test_factor.py` - Factor properties and thresholds
- `test_api.py` - Module-level API and tutorials
- `__init__.py` - Consolidated helpers and additional tests

**Test Coverage**: 38+ tests covering core functionality, edge cases, and numerical stability.

## Tutorials

1. **`tutorial/basic_tutorial.py`** - Spec CSV + Params approach
   ```bash
   python tutorial/basic_tutorial.py --spec data/sample_spec.csv --data data/sample_data.csv
   ```
   Comprehensive tutorial with detailed explanations of DFM components and practical usage examples.

2. **`tutorial/hydra_tutorial.py`** - Hydra decorator approach
   ```bash
   python tutorial/hydra_tutorial.py max_iter=10 threshold=1e-4
   ```
   Demonstrates Hydra-based configuration with CLI overrides.

## Troubleshooting

**Convergence Issues**: 
- Increase `max_iter` (default: 5000)
- Relax `threshold` (default: 1e-5)
- Check data quality and missing data patterns
- Adjust `regularization_scale` if numerical issues occur

**Dimension Mismatch**: 
- Ensure `series_id` matches data column names
- Verify block structure in configuration
- Check that all series have valid frequency settings

**Numerical Instability**: 
- Monitor warnings in logs
- Adjust `min_eigenvalue` and `max_eigenvalue` thresholds
- Investigate data quality (extreme values, excessive missing data)
- Use `clip_data_values=True` for extreme outliers

**Factor Evolution Issues**:
- Check that innovation variances (Q diagonal) are not zero
- Verify sufficient data variation in each block
- Consider increasing `regularization_scale` for sparse data

## Code Quality

The package has been extensively refactored for:
- **Code deduplication**: Common patterns consolidated into reusable functions
- **Maintainability**: Named constants replace magic numbers
- **Documentation**: Comprehensive docstrings with examples
- **Robustness**: Enhanced error handling and numerical stability
- **Consistency**: Unified coding patterns and logging formats

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

**Package Status**: Stable (v0.2.3+)  
**PyPI**: https://pypi.org/project/dfm-python/  
**Python**: 3.10+  
**Latest Changes**: 
- Extensive code refactoring for maintainability and robustness
- Consolidated covariance/variance computation functions
- Named constants replacing magic numbers
- Enhanced documentation and tutorials
- See [CHANGELOG.md](CHANGELOG.md) for details
