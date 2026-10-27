# dfm-python

[![PyPI version](https://img.shields.io/pypi/v/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/dfm-python.svg)](https://pypi.org/project/dfm-python/)
[![CI](https://img.shields.io/github/actions/workflow/status/minkeymouse/dfm-python/ci.yml?branch=main&label=CI)](https://github.com/minkeymouse/dfm-python/actions/workflows/ci.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/github/license/minkeymouse/dfm-python.svg)](https://github.com/minkeymouse/dfm-python/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/dfm-python.svg)](https://pypi.org/project/dfm-python/)

Dynamic factor models in Python, linear and deep, on one scikit-learn-style
estimator API: hyperparameters in the constructor, `fit(data)` on a
`PanelData`, DataFrame forecasts, fitted state in `result_`, pickle-free
save directories.

| Model   | What it is                                                                                     | Backend |
|---------|------------------------------------------------------------------------------------------------|---------|
| `DFM`   | linear, mixed-frequency dynamic factor model, estimated by EM with Kalman filtering and smoothing | numpy   |
| `DDFM`  | deep dynamic factor model: a nonlinear encoder trained with MCMC denoising and a linear state space on its factors | PyTorch (`deep` extra) |
| `iVDFM` | identifiable variational dynamic factor model, trained on the ELBO with a context-conditioned innovation prior | PyTorch (`deep` extra) |
| `AFM`   | Attention Factor Model of Epstein, Wang, Choi and Pelger (2025): a cross-sectional conditional factor model whose residuals are traded portfolios, with PCA and Avellaneda-Lee benchmarks | PyTorch (`deep` extra) |

## Installation

```bash
pip install dfm-python              # DFM; or: uv add dfm-python
pip install "dfm-python[deep]"      # also DDFM, iVDFM and AFM
```

Python 3.10 to 3.13. The dependencies are numpy, pandas, scipy, scikit-learn
and threadpoolctl. The torch models need the `deep` extra (torch 2.10 or
newer and einops); without it they can be built and configured, and `fit`
and `load` raise `ImportError` naming the extra. For a CPU-only or a specific
CUDA torch build, install `torch` from the
[PyTorch index](https://pytorch.org/get-started/locally/) first.

| Extra     | Adds              | Needed for                                                  |
|-----------|-------------------|-------------------------------------------------------------|
| `deep`    | torch, einops     | DDFM, iVDFM, AFM (`dfm_python.tuning` too)                  |
| `yaml`    | omegaconf         | YAML configs (`load_yaml`, `XConfig.from_yaml`), `tutorial/` |
| `tune`    | optuna            | iVDFM hyperparameter search (`dfm_python.tuning`)           |
| `metrics` | sktime            | `calculate_rmse`, `calculate_mae`                           |
| `polars`  | polars, pyarrow   | polars input and output (`set_output`)                      |
| `all`     | all of the above  |                                                             |
| `dev`     | pytest, ruff, build, twine, omegaconf, pyyaml, optuna, statsmodels | tests and packaging |

```bash
pip install "dfm-python[deep,yaml,tune]"
```

Code that needs a missing extra raises `ImportError` naming it.

## Quick start

A one-factor DFM on three monthly indicators and quarterly GDP growth, on a
monthly clock. It runs in about ten seconds.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

rng = np.random.default_rng(0)
f = np.zeros(120)
for t in range(1, 120):
    f[t] = 0.8 * f[t - 1] + rng.standard_normal()
idx = pd.date_range("2015-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame({"ip": f, "sales": 0.7 * f, "jobs": -0.5 * f}, index=idx)
df += 0.5 * rng.standard_normal(df.shape)
df["gdp"] = np.convolve(f, [1, 2, 3, 2, 1])[:120] / 9   # tent-weighted monthly factor
df.loc[df.index.month % 3 != 0, "gdp"] = np.nan          # quarterly: NaN between releases

data = PanelData(y=df, freq={"gdp": "q"})                # unlisted series run at the clock
model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]}).fit(data)
model.loglik_, model.n_iter_, model.stop_reason_         # n_iter_: E-steps run (about 230), 'converged'
fc = model.predict(6)               # DataFrame 6 x 4, every series at every month, data units
zf = model.predict_factors(6)       # DataFrame 6 x 1, column factor_0
model.save("runs/quickstart", overwrite=True)            # dfm_python.load() reads it back
```

The factors and loadings, a split of the data into its parts, prediction
intervals, and polars output (the `polars` extra):

```python
model.factors_                      # DataFrame 120 x 1 (factor_0), the smoothed factor on the training dates
model.loadings_, model.explained_variance_       # 4 x 1 loadings (scaled space); share of variance per series
parts = model.decompose(by="factor")             # columns (factor_0 | mean | idiosyncratic, series): sum = data
bands = model.predict_interval(6, coverage=0.9)  # columns (series, lower | upper), Gaussian, data units
model.set_output(output="polars")   # predict, predict_interval, decompose, ... now return polars frames
model.predict(6).columns            # ['date', 'ip', 'sales', 'jobs', 'gdp']
```

A walk-forward backtest and a time budget:

```python
from dfm_python.evaluation import backtest

spec = DFM(n_factors=1, tent_weights={"q:m": [1, 2, 3, 2, 1]}, max_time=60)  # stop EM after 60 s
result = backtest(spec, data, start=95, h=3, step=3, coverage=0.9)  # fit to row 95, then update every 3 rows
result.forecasts                    # origin, time, horizon, series, forecast, actual, lower, upper
result.metrics(by="horizon")        # RMSE, MAE, bias, n, coverage for h = 1, 2, 3
```

## Other models

These need the `deep` extra. DDFM: targets `s0..s2`, encoder covariates
`oil` and `fx` in `exog`.

```python
import numpy as np
import pandas as pd
from dfm_python import DDFM, PanelData

idx = pd.date_range("2010-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((120, 5)), index=idx,
                  columns=["s0", "s1", "s2", "oil", "fx"])
data = PanelData(y=df[["s0", "s1", "s2"]], exog=df[["oil", "fx"]])
model = DDFM(n_factors=2, max_iter=20, scaler="standard", random_state=3).fit(data)
fc = model.predict(6)               # DataFrame 6 x 3, data units
```

iVDFM: the `exog` column is the context of the innovation prior, so a
forecast takes its future values.

```python
import numpy as np
import pandas as pd
from dfm_python import iVDFM, PanelData

idx = pd.date_range("2010-01-31", periods=150, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((150, 4)), index=idx,
                  columns=["s0", "s1", "s2", "policy_rate"])
data = PanelData(y=df[["s0", "s1", "s2"]].iloc[:144], exog=df[["policy_rate"]].iloc[:144])
model = iVDFM(n_factors=2, window=48, max_epochs=5, scaler="standard", random_state=3).fit(data)
fc = model.predict(6, futr_exog=df[["policy_rate"]].iloc[144:150])   # DataFrame 6 x 3
```

AFM: asset returns `(T, N)` and characteristics `(T, N, M)`; it allocates
instead of forecasting.

```python
import numpy as np
import pandas as pd
from dfm_python import AFM, PanelData

rng = np.random.default_rng(0)
T, N, M = 120, 20, 8
data = PanelData.from_arrays(0.01 * rng.standard_normal((T, N)),
                             characteristics=rng.standard_normal((T, N, M)),
                             index=pd.bdate_range("2020-01-01", periods=T, name="date"))
model = AFM(n_factors=3, hist_len=20, max_epochs=5, random_state=0).fit(data)
weights = model.allocate()          # DataFrame (T - hist_len) x N
model.result_.sharpe, model.book_returns_
```

## Documentation

| Page | Contents |
|------|----------|
| [User guide](https://github.com/minkeymouse/dfm-python/blob/main/docs/guide.md) | `PanelData`, the estimator contract, `update` and `predict(data=)`, run control, interpretation, prediction intervals, backtesting, polars, saving and loading, YAML configs, reproducibility, logging, errors, upgrading from 0.8 |
| [Models](https://github.com/minkeymouse/dfm-python/blob/main/docs/models.md) | DFM, DDFM, iVDFM and AFM: full examples, hyperparameters, notes, method references |
| [Migrating from 0.7](https://github.com/minkeymouse/dfm-python/blob/main/docs/migration-0.7.md) | the 0.7 to 0.8 API table |
| [Development](https://github.com/minkeymouse/dfm-python/blob/main/docs/development.md) | setup, tests, lint, repository layout, golden tests, releases |
| [CHANGELOG](https://github.com/minkeymouse/dfm-python/blob/main/CHANGELOG.md) | release notes |
| [SECURITY.md](https://github.com/minkeymouse/dfm-python/blob/main/SECURITY.md) | trust model of run directories and YAML files; how to report a vulnerability |

The scripts in
[`tutorial/`](https://github.com/minkeymouse/dfm-python/tree/main/tutorial)
fit DFM, DDFM and iVDFM to the three datasets in `data/` from the YAML files
in `config/` (repository only, not in the installed package).

## Citing

Software citation metadata is in
[CITATION.cff](https://github.com/minkeymouse/dfm-python/blob/main/CITATION.cff);
BibTeX entries and the method reference of each model are in
[docs/citing.md](https://github.com/minkeymouse/dfm-python/blob/main/docs/citing.md).
Papers by the author that use dfm-python:

- Chang, M. and Kim, J.-Y. (2026). Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics. [arXiv:2603.22886](https://arxiv.org/abs/2603.22886). Introduces iVDFM.
- Chang, M. (2026). Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage. 7th ACM International Conference on AI in Finance (ICAIF '26). [arXiv:2610.25802](https://arxiv.org/abs/2610.25802). Uses AFM as a baseline; code in [minkeymouse/nfe](https://github.com/minkeymouse/nfe).

## License

MIT. See [LICENSE](https://github.com/minkeymouse/dfm-python/blob/main/LICENSE).

## Star history

<a href="https://star-history.com/#minkeymouse/dfm-python&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=minkeymouse/dfm-python&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=minkeymouse/dfm-python&type=Date" />
    <img alt="Star history chart for minkeymouse/dfm-python" src="https://api.star-history.com/svg?repos=minkeymouse/dfm-python&type=Date" />
  </picture>
</a>
