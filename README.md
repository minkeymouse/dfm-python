# dfm-python: Dynamic Factor Models for Python

A comprehensive Python package for Dynamic Factor Models (DFM), Deep Dynamic Factor Models (DDFM), and Kernelized Dynamic Factor Models (KDFM).

## Overview

This package provides implementations of three factor model variants:

- **DFM (Dynamic Factor Model)**: Linear factor model with EM algorithm estimation
- **DDFM (Deep Dynamic Factor Model)**: Nonlinear encoder with PyTorch and MCMC training
- **KDFM (Kernelized Dynamic Factor Model)**: Two-stage VARMA architecture with Krylov FFT for O(T log T) efficiency

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- scikit-learn, sktime (for preprocessing)
- statsmodels (for VAR/VECM benchmarks)
- pytorch-lightning (for KDFM/DDFM training)

### Install from Source

```bash
# Clone repository (if not already available)
cd dfm-python

# Install in development mode
pip install -e .

# Or using uv
uv pip install -e .
```

## Quick Start

### KDFM Example

```python
from dfm_python import KDFM, KDFMDataModule, KDFMTrainer
from dfm_python.config import KDFMConfig, SeriesConfig
import pandas as pd

# Load data
df = pd.read_csv('data/your_data.csv')
df_processed = df[[col for col in df.columns if col != 'date']]

# Create configuration
series_configs = [SeriesConfig(series_id=col, frequency='d') for col in df_processed.columns]
config = KDFMConfig(
    series=series_configs,
    ar_order=1,
    ma_order=0,
    structural_method='cholesky',
    learning_rate=0.001,
    max_epochs=50,
    batch_size=32
)

# Create DataModule
data_module = KDFMDataModule(config=config, data=df_processed)
data_module.setup()

# Create model
model = KDFM(
    config=config,
    ar_order=1,
    ma_order=0,
    learning_rate=0.001,
    max_epochs=50
)

# Initialize from data
sample_batch = next(iter(data_module.train_dataloader()))
sample_data = sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
if sample_data.ndim == 3:
    sample_data = sample_data[0]
model.initialize_from_data(sample_data)

# Train
trainer = KDFMTrainer(max_epochs=50, enable_progress_bar=True)
trainer.fit(model, data_module)

# Predict
X_forecast, Z_forecast = model.predict(horizon=6)
print(f"Forecast shape: {X_forecast.shape}")
```

### DFM Example

```python
from dfm_python import DFM, DFMDataModule, DFMTrainer
import pandas as pd

# Load data
df = pd.read_csv('data/your_data.csv')

# Create DataModule
data_module = DFMDataModule(config_path='config/dfm_config.yaml', data=df)
data_module.setup()

# Create and fit model
model = DFM()
model.load_config('config/dfm_config.yaml')
model.fit(data_module.get_processed_data())

# Predict
X_forecast = model.predict(horizon=6)
```

## Package Structure

```
dfm-python/
├── src/
│   └── dfm_python/
│       ├── models/          # Model implementations (DFM, DDFM, KDFM)
│       ├── config/           # Configuration schemas
│       ├── datamodule/       # Data loading and preprocessing modules
│       ├── trainer/          # Training procedures
│       ├── ssm/              # State-space model components
│       ├── dataset/          # Data loading and preprocessing
│       └── utils/            # Utility functions
├── tutorial/               # Tutorial scripts and examples
├── tests/                   # Test suite
└── README.md               # This file
```

## Key Features

### KDFM (Kernelized Dynamic Factor Model)

- **Two-stage VARMA architecture**: AR stage for VAR coefficients, optional MA stage for moving average dynamics
- **Structural identification**: Learnable structural shock identification (Cholesky, full, low-rank)
- **Efficient computation**: Krylov FFT for O(T log T) forward pass
- **Direct IRF computation**: Avoids numerical error accumulation
- **Gradient descent training**: Uses PyTorch for training with custom logging

### DFM (Dynamic Factor Model)

- **EM algorithm**: Expectation-Maximization for parameter estimation
- **Block structure**: Supports mixed-frequency data with block-organized factors
- **Kalman filtering**: Uses pykalman for efficient state estimation
- **NumPy-based**: All computations in NumPy for numerical stability

### DDFM (Deep Dynamic Factor Model)

- **Nonlinear encoder**: Neural network encoder for nonlinear factor extraction
- **MCMC training**: Denoising training procedure
- **PyTorch-based**: Full PyTorch implementation with custom logging system

## IRF Analysis: KDFM's PRIMARY CONTRIBUTION

**IRF (Impulse Response Function) analysis is KDFM's PRIMARY CONTRIBUTION and fundamental innovation**. Unlike traditional models that compute IRFs as post-processing steps (VAR) or treat them as implicit filters (SpaceTime), KDFM estimates IRFs **directly as the primary object of estimation**, built into the model architecture via companion matrix parameterization. This direct IRF estimation enables explicit structural shock analysis with orthogonal, interpretable shocks that traditional models cannot provide.

### Key Concepts

- **Direct IRF Estimation (PRIMARY CONTRIBUTION)**: KDFM estimates IRF directly as the primary object via companion matrix powers $K_h = C (A^{\text{AR}})^h B$, making IRF estimation the primary goal rather than a derived quantity. This architectural innovation distinguishes KDFM from all existing models.
- **Reduced-form IRF**: Response to reduced-form residuals (correlated shocks), computed directly as matrix powers
- **Structural IRF**: Response to orthogonal structural shocks (causally interpretable), computed as $K_h^{\text{struct}} = K_h S$ where $S$ is the learnable structural identification matrix
- **Explicit Shock Analysis**: Structural shocks $\varepsilon_t$ are directly interpretable (orthogonal, unit variance), enabling causal interpretation where each shock has clear economic/scientific meaning
- **Numerical Stability**: Direct matrix power computation avoids error accumulation that occurs in traditional VAR methods when converting coefficients to moving average representation recursively

### Quick Example

```python
from dfm_python.models.functional.irf import compute_irf
import torch

# After training KDFM model, extract companion matrices
ar_transition = model.companion_ar.get_companion_matrix()
B = model.companion_ar.B
C = model.companion_ar.C
S = model.structural_id.get_structural_matrix()

# For pure VAR (ma_order=0), MA stage is identity
K = ar_transition.shape[0] // config.ar_order
ma_transition = torch.eye(K, device=ar_transition.device)
B_prime = torch.eye(K, device=B.device)
C_prime = torch.eye(K, device=C.device)

# Compute IRF
irf_reduced, irf_structural = compute_irf(
    A_ar=ar_transition,
    A_ma=ma_transition,
    B=B, C=C,
    B_prime=B_prime, C_prime=C_prime,
    S=S,
    horizon=20,
    structural=True
)

# irf_structural shape: (20, K, K)
# irf_structural[h, i, j] = response of variable i to structural shock j at horizon h
```

### IRF Tutorial

See `tutorial/tutorial_irf_analysis.py` for a comprehensive IRF analysis tutorial covering:
- Computing reduced-form and structural IRFs
- Visualizing IRF responses
- Interpreting IRF results
- Comparing IRFs across models

## Documentation

- **Tutorials**: See `tutorial/` directory for detailed examples
- **IRF Tutorial**: `tutorial/tutorial_irf_analysis.py` - Comprehensive IRF analysis guide
- **API Documentation**: See docstrings in source code
- **Examples**: 
  - `tutorial/tutorial_macro_kdfm.py` - KDFM for macro data
  - `tutorial/tutorial_finance_kdfm.py` - KDFM for finance data
  - `tutorial/tutorial_irf_analysis.py` - IRF analysis (PRIMARY STRENGTH)

## Testing

Run tests to verify installation:

```bash
# Run all tests
pytest dfm-python/src/test/test_*.py -v

# Run specific model tests
pytest dfm-python/src/test/test_models.py::TestKDFM -v
pytest dfm-python/src/test/test_models.py::TestDFM -v
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{kdfm2026,
  title={Kernelized Dynamic Factor Models for Scientific Time Series},
  author={...},
  journal={ICLR},
  year={2026}
}
```

## Troubleshooting

### Common Errors and Solutions

#### Method Signature Error: `_can_compute_irf() takes 7 positional arguments but 8 were given`

**Error**: This error occurs when calling `model.get_result()` or during forecast generation.

**Cause**: Method signature mismatch in `_can_compute_irf()` method (fixed in v0.5.5+).

**Solution**: 
- Ensure you're using the latest version of dfm-python (v0.5.5+)
- The method signature has been fixed in Iteration 11
- If error persists, check that `dfm-python/src/dfm_python/models/kdfm.py` has the correct method signature
- The method should return `Tuple[bool, str]` and accept 6 parameters (excluding self): `ma_transition, ar_input, ar_output, ma_input, ma_output, structural_matrix`

**Fixed in**: Iteration 11 (2025-12-29)

#### Forecast Generation Errors

**Error**: Forecasts contain NaN/Inf values or method signature errors.

**Causes**:
1. Method signature mismatch (fixed in Iteration 11)
2. Numerical instability in companion matrix (eigenvalues >= 1.0)
3. Missing or incorrect inverse transformation

**Solutions**:
1. Ensure method signature fix is applied (see above)
2. Check companion matrix stability: `max_eigenvalue < 1.0`
3. Verify inverse transformation is applied correctly
4. Check forecast generation logs for detailed error messages
5. Use `experiment/scripts/kdfm_forecasts.py` which includes comprehensive error handling

#### Eigenvalue Inconsistencies

**Error**: Eigenvalues differ between source files and comprehensive analysis.

**Solution**: 
- Source of truth: `results/kdfm_analysis/*_kdfm_results.json` files
- Regenerate `comprehensive_analysis.json` using: `python experiment/utils/comprehensive_result_analysis.py`
- Validation automatically detects inconsistencies

### API Reference

#### IRF Computation

The IRF computation API is consistent across all methods:

- `compute_irf()`: Main function for IRF computation (in `dfm_python.models.functional.irf`)
- `_can_compute_irf()`: Validation method (in `KDFM` class)
- `_compute_irfs_from_params()`: Parameter-based IRF computation (in `KDFM` class)

**Method Signature**:
```python
def _can_compute_irf(
    self,
    ma_transition: Optional[np.ndarray],
    ar_input: Optional[np.ndarray],
    ar_output: Optional[np.ndarray],
    ma_input: Optional[np.ndarray],
    ma_output: Optional[np.ndarray],
    structural_matrix: Optional[np.ndarray]
) -> Tuple[bool, str]:
    """Returns (can_compute, error_msg) tuple."""
```

**Note**: `ar_transition` is NOT a parameter (only used in actual IRF computation, not validation).

## License

[Specify license]

## Contributing

[Contributing guidelines]

## Support

[Support information]
