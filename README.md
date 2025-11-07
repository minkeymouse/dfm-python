# dfm-python

A generic Python implementation of Dynamic Factor Models (DFM) for nowcasting and forecasting.

## Features

- **Generic DFM Estimation**: EM algorithm for estimating dynamic factor models
- **Mixed-Frequency Data**: Handle data with different frequencies (daily, monthly, quarterly)
- **Missing Data Handling**: Robust handling of missing observations
- **News Decomposition**: Compare forecasts between vintages and attribute changes
- **Flexible Configuration**: Support for YAML and CSV configuration files
- **Optional Hydra Integration**: Use Hydra for experiment management (optional dependency)

## Installation

```bash
pip install dfm-python
```

### Optional Dependencies

For Hydra configuration support:
```bash
pip install dfm-python[hydra]
```

For database integration (application-specific):
```bash
pip install dfm-python[database]
```

Install all optional dependencies:
```bash
pip install dfm-python[all]
```

## Quick Start

### Basic Usage

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd

# Load configuration from YAML or CSV
config = load_config('config.yaml')

# Load data from CSV file
X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))

# Estimate DFM
Res = dfm(X, config, threshold=1e-4)

# Access results
factors = Res.Z  # Extracted factors
loadings = Res.C  # Factor loadings
```

### Complete Example with Sample Data

The package includes sample data and configuration files for testing. Here's a complete working example:

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd
import numpy as np

# Load configuration (YAML or CSV format)
config = load_config('config/example_config.yaml')
# Alternative: config = load_config('spec/example_spec.csv')

# Load sample data
X, Time, Z = load_data(
    'data/sample_data.csv', 
    config, 
    sample_start=pd.Timestamp('2000-01-01')
)

print(f"Data shape: {X.shape}")
print(f"Time range: {Time[0]} to {Time[-1]}")
print(f"Number of series: {len(config.series)}")

# Estimate DFM model
print("\nEstimating DFM model...")
Res = dfm(X, config, threshold=1e-4, max_iter=1000)

# Access results
print(f"\nExtracted {Res.Z.shape[1]} factors")
print(f"Factor loadings shape: {Res.C.shape}")
print(f"Transition matrix shape: {Res.A.shape}")

# Display factor estimates
factors = Res.Z  # Factor estimates (T x r)
loadings = Res.C  # Factor loadings (N x r)
transition = Res.A  # Transition matrix
covariance = Res.Q  # Factor covariance

print(f"\nFirst few factor values:")
print(factors[:5, :])
```

**Expected Output:**
```
Loading data...
Data shape: (128, 10)
Time range: 2000-05-01 00:00:00 to 2010-12-01 00:00:00
Number of series: 10

Estimating DFM model...
Extracted 4 factors
Factor loadings shape: (10, 4)
Transition matrix shape: (4, 4)

First few factor values:
[[ 0.123 -0.045  0.012  0.008]
 [ 0.145 -0.032  0.015  0.011]
 ...]
```

### Configuration

The DFM module uses a configuration object (`DFMConfig`) that defines:

- **Series**: Each time series with frequency, transformation, and block membership
- **Blocks**: Factor blocks with number of factors per block
- **Estimation Parameters**: EM algorithm settings (threshold, max_iter, etc.)

#### YAML Configuration Example

The package includes a sample configuration file (`config/example_config.yaml`) that demonstrates the full configuration structure:

```yaml
# config/example_config.yaml
model:
  series:
    gdp_real:
      series_name: "Real GDP (Quarterly)"
      frequency: "q"
      transformation: "pca"
      category: "GDP"
      units: "Index (2000=100)"
      blocks: [Global, Consumption, Investment]
    
    consumption:
      series_name: "Consumption (Monthly)"
      frequency: "m"
      transformation: "pch"
      category: "Consumption"
      units: "Index (2000=100)"
      blocks: [Global, Consumption]
  
  blocks:
    Global:
      factors: 2
    Consumption:
      factors: 1
    Investment:
      factors: 1

dfm:
  ar_lag: 1
  threshold: 1e-5
  max_iter: 5000
```

#### CSV Configuration Example

The package also includes a CSV configuration file (`spec/example_spec.csv`) for simpler setups:

```csv
series_id,series_name,frequency,transformation,category,units,Block_Global,Block_Consumption,Block_Investment
gdp_real,Real GDP (Quarterly),q,pca,GDP,Index (2000=100),1,1,1
consumption,Consumption (Monthly),m,pch,Consumption,Index (2000=100),1,1,0
investment,Investment (Monthly),m,pch,Investment,Index (2000=100),1,0,1
```

**Note:** The first block (Global) must always have value `1` for all series, as it represents the common factor that all series load on.

### Data Format

CSV data files should have:
- First column: Date (YYYY-MM-DD format)
- Subsequent columns: One per series (matching `series_id` from config)
- Column names: Must match `series_id` from configuration exactly

The package includes sample data (`data/sample_data.csv`) with 10 time series spanning 2000-2010:

```csv
Date,gdp_real,gdp_nominal,consumption,investment,exports,imports,unemployment,industrial_production,retail_sales,consumer_confidence
2000-01-01,100.0,105.0,98.5,95.0,102.0,100.5,5.2,98.0,97.5,100.0
2000-02-01,100.2,105.3,98.7,95.2,102.2,100.7,5.1,98.3,97.8,100.2
...
```

**Important Notes:**
- Date column must be named `Date` (case-sensitive)
- Series column names must exactly match `series_id` values in the configuration
- Missing values should be represented as empty cells or `NaN`
- The data supports mixed frequencies (monthly and quarterly series are handled automatically)

## Core Components

### DFMConfig

Configuration dataclass that defines the model structure:

```python
from dfm_python import DFMConfig, SeriesConfig

config = DFMConfig(
    series=[
        SeriesConfig(
            series_id="gdp",
            frequency="q",
            transformation="pca",
            blocks=[1, 0]  # Binary: loads on block 0 (Global)
        )
    ],
    block_names=["Global", "Investment"],
    factors_per_block=[1, 1],
    ar_lag=1,
    threshold=1e-5,
    max_iter=5000
)
```

### DFM Estimation

```python
from dfm_python import dfm

# Estimate DFM model
result = dfm(X, config, threshold=1e-4, max_iter=1000)

# Access results
factors = result.Z          # Factor estimates (T x r)
loadings = result.C         # Factor loadings (N x r)
transition = result.A         # Transition matrix
covariance = result.Q         # Factor covariance
```

### News Decomposition

Compare forecasts between two vintages:

```python
from dfm_python import update_nowcast

# Compare old vs new vintage
news_df, forecast_df = update_nowcast(
    X_old, X_new, Time, config, result,
    series="gdp",
    vintage_old="2024-01-01",
    vintage_new="2024-02-01"
)
```

## API Reference

### Main Functions

- `load_config(file)`: Load configuration from YAML or CSV
- `load_data(file, config)`: Load and transform data from CSV
- `dfm(X, config)`: Estimate DFM model using EM algorithm
- `update_nowcast(...)`: Perform news decomposition between vintages

### Configuration

- `DFMConfig`: Main configuration class
- `SeriesConfig`: Individual series configuration

### Results

- `DFMResult`: Result object with factors, loadings, and model parameters

## Architecture

The DFM module is designed to be **generic and application-agnostic**:

- **Core Module** (`dfm_python`): Generic DFM estimation logic
- **Adapters** (application-specific): Database integration, API clients, etc.

This separation allows the core module to be used in any context while keeping application-specific code separate.

## Testing with Sample Data

The package includes sample data and configuration files for testing and demonstration:

- **Data**: `data/sample_data.csv` - 10 time series from 2000-2010
- **Config (YAML)**: `config/example_config.yaml` - Full configuration example
- **Config (CSV)**: `spec/example_spec.csv` - Simplified CSV configuration

To test the package with sample data:

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd

# Load sample configuration
config = load_config('config/example_config.yaml')

# Load sample data
X, Time, Z = load_data('data/sample_data.csv', config)

# Estimate model
result = dfm(X, config, threshold=1e-4)

# Check results
print(f"Factors extracted: {result.Z.shape[1]}")
print(f"Factor loadings shape: {result.C.shape}")
print(f"Number of series: {result.C.shape[0]}")
print(f"Number of time periods: {result.Z.shape[0]}")
```

You can also run the test suite:

```bash
# From project root
python -m pytest src/test/
# Or run specific test
python src/test/test_synthetic.py
```

## Requirements

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## License

MIT License

## Contributing

Contributions are welcome! Please ensure that:
- Core module remains generic (no application-specific code)
- All optional dependencies are properly handled
- Tests pass for the core functionality

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting},
  author = {DFM Python Contributors},
  year = {2024},
  url = {https://github.com/yourusername/dfm-python}
}
```
