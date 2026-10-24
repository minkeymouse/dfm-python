# User guide

The parts of the API shared by every model. Each model's hyperparameters and
a full example are in [models.md](https://github.com/minkeymouse/dfm-python/blob/main/docs/models.md).

## PanelData

`PanelData` is the one data container. It is frozen and validated, uses only
numpy and pandas, and never imputes, scales or reorders values.

| Slot          | Shape                                                     | Meaning                                          |
|---------------|-----------------------------------------------------------|--------------------------------------------------|
| `y`           | DataFrame, T x N                                          | the target series (macro panel or asset returns) |
| `exog`        | DataFrame, T x K, or None                                 | per-time inputs that are not forecast            |
| `series_exog` | DataFrame, T x (M*N), columns `(feature, series)`, or None | per-time, per-series features                   |
| `freq`        | dict or None                                              | each series' sampling frequency                  |

The index is a unique, increasing `DatetimeIndex`, `PeriodIndex` or
`RangeIndex` (arrays get a `RangeIndex`), shared by every slot. `freq` takes
the per-series form `{"gdp": "q"}` or the grouped form `{"q": ["gdp"]}` and
stores the per-series form; unlisted series use the model's clock.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData, UnsupportedInputError

dates = pd.date_range("2020-01-31", periods=24, freq="ME", name="date")
wide = pd.DataFrame({"date": dates, "gdp": np.nan,
                     "ip": np.linspace(0.0, 1.0, 24), "oil": np.ones(24)})
wide.loc[2::3, "gdp"] = 1.0                       # quarterly: observed every third month

p = PanelData.from_wide(wide, time_col="date", exog_cols=["oil"])   # targets: gdp, ip
q1 = PanelData.from_wide(wide, time_col="date", target_cols=["gdp", "ip"],
                         freq={"gdp": "q"})                         # per-series form
q2 = PanelData(y=wide.set_index("date")[["gdp", "ip"]], freq={"q": ["gdp"]})  # grouped form
assert q1.freq == q2.freq == {"gdp": "q"}

long = p.to_long()                                # unique_id, ds, y (+ exog columns)
assert PanelData.from_long(long, exog_cols=["oil"]).equals(p)

arr = PanelData.from_arrays(np.zeros((24, 2)), series_exog=np.zeros((24, 2, 3)),
                            index=dates, names=["x", "z"])           # y (T, N), series_exog (T, N, M)
assert arr.series_exog_array().shape == (24, 2, 3)

try:
    DFM().fit(p)                                  # DFM takes no exog
except UnsupportedInputError as err:
    print(err)                                    # names the tag capability:exog
```

Constructors:

| Constructor | Input |
|-------------|-------|
| `PanelData(y, exog=None, series_exog=None, freq=None)` | DataFrames |
| `from_wide(df, *, time_col, target_cols, exog_cols, freq)` | one wide frame; targets default to every column that is not time or exog |
| `from_long(df, *, id_col="unique_id", time_col="ds", target_col="y", series_exog_cols=(), exog_cols=(), freq)` | one row per `(series, time)` |
| `from_arrays(y, *, exog, series_exog, index, names, exog_names, series_exog_names, freq)` | numpy arrays |

`fit`, `predict` and `update` also accept a bare DataFrame (`PanelData(y=df)`),
a polars DataFrame (its Date or Datetime column becomes the index, see
[polars input and output](#polars-input-and-output)) or a 2-D array.
`data.loc(start, stop)` slices by time.

Which slots each model accepts; anything else raises `UnsupportedInputError`
naming the tag (`Model.get_tags()`):

| Model | `y`           | `exog`                   | `series_exog`                       | mixed `freq`              | missing values |
|-------|---------------|--------------------------|-------------------------------------|---------------------------|----------------|
| DFM   | series        | no                       | no                                  | yes, one slower frequency | yes            |
| DDFM  | targets       | encoder covariates       | no                                  | no                        | interpolated   |
| iVDFM | targets       | context u_t of the prior | no                                  | no                        | interpolated   |
| AFM   | asset returns | no                       | characteristics (attention factors) | no                        | no             |

## The estimator contract

- `Model(**hyperparameters)` stores keyword-only hyperparameters and does
  nothing else.
- `fit(data)` takes a `PanelData` (or a DataFrame) and returns the model.
- `predict(h)` returns a DataFrame of the `h` steps after the last
  conditioning date, one column per series in your order, in data units.
  `predict_factors(h)` returns the factor forecast, one column per name in
  `factor_names_`. AFM allocates instead: `allocate()` and `backtest(data)`;
  its `predict` raises `TypeError`.
- Fitted state lives in attributes ending in `_`: `result_`, `n_iter_`,
  `converged_`, `scaler_`, `factors_`, ... Before `fit` they raise
  `NotFittedError`.
- Where the tags below allow it, `decompose(by=)` and the interpretation
  attributes (`factors_`, `loadings_`, `explained_variance_`) describe the
  training data and `predict_interval(h, coverage)` gives prediction
  intervals. On every model, `set_output(output="polars")` makes the
  prediction methods return polars frames.
- `update(data)` conditions later forecasts on new data with the fitted
  parameters; `update(data, refit=True)` also re-estimates where the model
  supports it.
- `save(path)` writes a pickle-free run directory; `Model.load(path)` and
  `dfm_python.load(path)` read it back without data.
- `get_params`, `set_params` and `sklearn.base.clone` work as in
  scikit-learn. `get_config()` / `Model.from_config(cfg)` convert to and from
  the typed config mirror (`DFMConfig`, `DDFMConfig`, `iVDFMConfig`,
  `AFMConfig`).

```python
import numpy as np
import pandas as pd
from sklearn.base import clone

import dfm_python
from dfm_python import DFM, PanelData

idx = pd.date_range("2015-01-31", periods=96, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((96, 3)),
                  index=idx, columns=["a", "b", "c"])
train, new = PanelData(y=df.iloc[:84]), PanelData(y=df.iloc[84:])

model = DFM(n_factors=1, max_iter=50, scaler="standard")  # hyperparameters only
model = model.fit(train)                                  # fit returns the model
fc = model.predict(6)               # DataFrame 6 x 3, index 2022-01-31 ... 2022-06-30
zf = model.predict_factors(6)       # DataFrame 6 x 1 (factor_0)
model.result_.loglik, model.n_iter_, model.converged_

model.update(new)                   # condition on the 12 new rows; returns model
assert model.predict(6).index[0] == pd.Timestamp("2023-01-31")

model.save("runs/contract", overwrite=True)
same = dfm_python.load("runs/contract")                  # or DFM.load(...); no data needed
pd.testing.assert_frame_equal(same.predict(6), model.predict(6))

params = model.get_params()                              # {"n_factors": 1, ...}
two = clone(model).set_params(n_factors=2)               # unfitted copy with 2 factors
cfg = model.get_config()                                 # DFMConfig with the same fields
assert DFM.from_config(cfg).get_params() == params
```

The method capabilities, from `Model.get_tags()`. A method whose tag is
False raises `UnsupportedInputError` naming the tag:

| Tag | Methods and attributes | DFM | DDFM | iVDFM | AFM |
|-----|------------------------|-----|------|-------|-----|
| `capability:update` | `update(data)` | yes | yes | yes | yes |
| `capability:factor_forecast` | `predict_factors(h)` | yes | yes | yes | no |
| `capability:interpret` | `factor_names_`, `factors_`, `loadings_`, `loadings_kind_`, `explained_variance_`, `decompose()` | yes | yes | yes | no |
| `capability:pred_int` | `predict_interval(h, coverage, data)` | yes | no | no | no |
| `capability:device` | the `device` hyperparameter | no | yes | yes | yes |

Public names exported from `dfm_python`: the models `DFM`, `DDFM`, `iVDFM`,
`AFM` and their base `BaseFactorModel`; the configs `DFMConfig`, `DDFMConfig`,
`iVDFMConfig`, `AFMConfig`; `PanelData`; `load`; the results `BaseResult`,
`DFMResult`, `DDFMResult`; `calculate_rmse`; the errors `NotFittedError` and
`UnsupportedInputError`; `set_verbosity` and `get_verbosity`.

## update and predict(data=)

After `update(data)`, `predict` conditions on `data`; fitted parameters change
only with `refit=True`. The mechanism is each model's own and is published in
the tags `update_kind` and `predict_data`. `predict(h, data=)` never changes
the model: it runs on a copy.

| Model | `update_kind` | `update(data)` | `update(data, refit=True)` | `predict_data` | `data=` / `update` take |
|-------|---------------|----------------|----------------------------|----------------|-------------------------|
| DFM   | `filter` | the Kalman filter continues over the new rows (prior `(A x_T|T, A P_T|T A' + Q)` from the filter over the conditioning rows) and appends their filtered states; `result_` unchanged | one EM iteration on all rows, not a full refit; refreshes `result_`, `n_iter_`, `converged_`, `loglik_`, `factors_`, `states_`, `loadings_`, `explained_variance_` | `new_rows` | only the rows after the last conditioning date; overlap raises `DataValidationError` |
| DDFM  | `encoder_forward` | scales the new rows with the fitted scalers (never refitted), interpolates them with the conditioning rows, encodes them with the trained weights and appends their factors | `UnsupportedInputError` | `new_rows` | only the rows after the last conditioning date; overlap raises `DataValidationError` |
| iVDFM | `encoder_forward` | encodes the rows as one sequence with the fixed weights and appends the result to the factor path the forecast starts from (also appends the window-end factors) | `UnsupportedInputError` | `window` | any window: every row is encoded and the state at its last row starts the forecast |
| AFM   | `reattach` | attaches the data for later `allocate()` | continues training from the current weights | (none) | `update` and `allocate(data=)` take any panel with the training assets and characteristics |

For DFM, DDFM and iVDFM, `predict(h, data=d)` equals `update(d)` followed by
`predict(h)`, and DFM's `predict_interval(h, data=d)` equals `update(d)`
followed by `predict_interval(h)`. For DFM both run the Kalman filter over the
conditioning rows and the new rows in one pass and forecast from the last
filtered state. `update(data)` leaves the interpretation attributes and
`decompose` on the training rows; only DFM's `update(data, refit=True)`
recomputes them, on all rows. To re-estimate a model from scratch, call `fit`
on the whole panel.

## Interpretation

DFM, DDFM and iVDFM (tag `capability:interpret`) describe the training data
through their factors. `fit` and `load` set:

| Attribute | Contents |
|-----------|----------|
| `factor_names_` | the factor names: the columns of `factors_`, `loadings_` and `predict_factors` |
| `factors_` | the factor path on `index_`, one column per factor; each model's definition is in [models.md](https://github.com/minkeymouse/dfm-python/blob/main/docs/models.md) |
| `loadings_` | series x factors, in the scaled (model) space |
| `loadings_kind_` | `"exact"` or `"mean_jacobian"` (below) |
| `explained_variance_` | per series, `1 - var(x - common) / var(x)` over the observed training entries of the scaled data `x`, where `common` is the model's common component; NaN for a series with fewer than two observed entries or no variance, negative when `common` fits worse than a constant |

`loadings_kind_` says what `loadings_` holds. `"exact"`: the series are
linear in the factors (DFM, DDFM with `decoder_type="linear"`) and
`loadings_` are the model's coefficients. `"mean_jacobian"`: the decoder is
nonlinear (DDFM with `decoder_type="mlp"`, every iVDFM) and `loadings_` is its
Jacobian with respect to the factors averaged over the rows of `factors_`, a
local linear summary rather than coefficients.

`decompose(by=...)` splits the training data, in data units, on the rows of
`factors_`. The parts add up to the data.

| `by` | Columns | Models |
|------|---------|--------|
| `"component"` (default) | `(component, series)`: `"common"`, the factors through the loadings or the decoder, inverse-scaled (so it includes the series mean), and `"idiosyncratic"`, the data minus `"common"` (NaN where the data is missing) | DFM, DDFM, iVDFM |
| `"factor"` | `(part, series)`: one part per name in `factor_names_`, `"mean"` (the scaler's offset; for DDFM plus the decoder bias in data units) and `"idiosyncratic"` | DFM, DDFM with `decoder_type="linear"`; scaler None, `"standard"`, `"robust"`, `"minmax"` or `"maxabs"` |

`by="factor"` with another model, decoder or scaler raises
`UnsupportedInputError`; another `by` raises `ConfigurationError`.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

rng = np.random.default_rng(4)
f = np.zeros(96)
for t in range(1, 96):
    f[t] = 0.7 * f[t - 1] + rng.standard_normal()
idx = pd.date_range("2015-01-31", periods=96, freq="ME", name="date")
df = pd.DataFrame(np.outer(f, [1.0, 0.8, -0.6]) + 0.5 * rng.standard_normal((96, 3)),
                  index=idx, columns=["a", "b", "c"])
model = DFM(n_factors=1, max_iter=50, scaler="standard").fit(PanelData(y=df))

model.factor_names_                 # array(['factor_0'], dtype=object)
model.factors_                      # DataFrame 96 x 1 (factor_0) on index_
model.loadings_                     # DataFrame 3 x 1: a, b, c on factor_0, scaled space
model.loadings_kind_                # 'exact'
model.explained_variance_           # Series a, b, c

parts = model.decompose(by="factor")          # columns (part, series): factor_0, mean, idiosyncratic
total = parts["factor_0"] + parts["mean"] + parts["idiosyncratic"]
np.testing.assert_allclose(total.to_numpy(), df.to_numpy())
common = model.decompose()["common"]          # DataFrame 96 x 3, data units
```

## Prediction intervals

DFM (tag `capability:pred_int`) gives `predict_interval(h, coverage=0.9,
data=None)`: central Gaussian intervals of the `h` steps after the last
conditioning date, in data units. The result is indexed like `predict(h)`,
with MultiIndex columns `(series, bound)`, `bound` `"lower"` or `"upper"`,
and `attrs["coverage"]`. The bounds are the numbers `predict(h)` returns
minus and plus `z * sd`, with `z` the standard normal quantile of
`(1 + coverage) / 2`. The standard deviations come from the Kalman forecast
moments: the filtered state covariance `P` at the last conditioning row,
propagated by `P <- A P A' + Q` for each step, gives the scaled series
variance `diag(C P C' + R)`, which the scaler's slope converts to data units.
This needs an affine scaler (None, `"standard"`, `"robust"`, `"minmax"` or
`"maxabs"`); any other raises `UnsupportedInputError`. The fitted parameters
are taken as known, so the intervals leave out estimation uncertainty.
`data=` conditions on new rows on a copy, as in `predict(h, data=)`. The
other models raise `UnsupportedInputError`.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2015-01-31", periods=96, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((96, 3)), index=idx, columns=["a", "b", "c"])
model = DFM(n_factors=1, max_iter=50, scaler="standard").fit(PanelData(y=df.iloc[:84]))

bands = model.predict_interval(6, coverage=0.9)   # DataFrame 6 x 6: (a, lower), (a, upper), ...
bands["a"]                                        # columns lower, upper
mid = (bands.xs("lower", axis=1, level="bound") + bands.xs("upper", axis=1, level="bound")) / 2
np.testing.assert_allclose(mid.to_numpy(), model.predict(6).to_numpy())   # centred on predict
later = model.predict_interval(6, data=PanelData(y=df.iloc[84:]))        # after 12 new rows; model unchanged
```

## polars input and output

polars frames need the `polars` extra (polars and pyarrow).

Input: `fit`, `predict(data=)`, `update`, `allocate(data=)` and `backtest`
take a bare polars DataFrame. Its one Date or Datetime column becomes the
time index, as in `PanelData.from_wide(df, time_col=<that column>)`, and the
other columns are the series. Without such a column the rows get a
`RangeIndex`; with several the call raises `DataValidationError`. For exog,
`freq` or a choice of time column, build the panel with
`PanelData.from_wide` or `from_long`, which take polars frames too. iVDFM's
`futr_exog` also takes a polars frame, indexed by its Date or Datetime column.

Output: `set_output(output="polars")` returns the model and makes `predict`,
`predict_factors`, `predict_interval`, `decompose` and `allocate` return a
`polars.DataFrame`. The index becomes the first column, named after the index
(`"time"` when unnamed; a `PeriodIndex` becomes its start timestamps), and
MultiIndex columns are joined with `"__"` (`"a__lower"`, `"common__a"`); a
column named like the index raises `ValueError`. `set_output(output="pandas")`
switches back. Fitted attributes (`factors_`, `loadings_`, `result_`, ...)
stay pandas. The choice is not a hyperparameter: `get_params`, `clone` and
`save` leave it out, so a clone or a loaded model returns pandas.

`PanelData.to_polars(layout=...)` converts a panel. `"long"` (the default) is
`to_long()` as polars (`unique_id`, `ds`, `y`, then the exog columns), which
`from_long` reads back; `"wide"` is the time column, the `y` columns, then the
exog columns, which `from_wide(df, time_col=..., exog_cols=...)` reads back.
`series_exog` has no wide layout (`UnsupportedInputError`).

```python
import numpy as np
import pandas as pd
import polars as pl
from dfm_python import DFM, PanelData

idx = pd.date_range("2015-01-31", periods=60, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(5).standard_normal((60, 3)), index=idx, columns=["a", "b", "c"])
frame = pl.from_pandas(df.reset_index())           # columns date, a, b, c
model = DFM(n_factors=1, max_iter=20, scaler="standard").fit(frame[:48])   # "date" is the index
assert model.index_[-1] == pd.Timestamp("2018-12-31")
model.update(frame[48:])                           # the 12 new rows, also polars

model.set_output(output="polars")
fc = model.predict(6)                              # polars: date, a, b, c
bands = model.predict_interval(6)                  # polars: date, a__lower, a__upper, ...
assert bands.columns[:3] == ["date", "a__lower", "a__upper"]
assert isinstance(model.factors_, pd.DataFrame)    # fitted attributes stay pandas

panel = PanelData(y=df[["a", "b"]], exog=df[["c"]])
wide = panel.to_polars(layout="wide")              # date, a, b, c
long = panel.to_polars(layout="long")              # unique_id, ds, y, c
assert PanelData.from_wide(wide, time_col="date", exog_cols=["c"]).equals(panel)
assert PanelData.from_long(long, exog_cols=["c"]).equals(panel)
```

## Saving and loading

`save(path, overwrite=False)` writes a run directory: to a temporary sibling,
reloaded to verify, then renamed into place. `Model.load(path)` and
`dfm_python.load(path)` (which dispatches on the saved `model_type`) read it
without data. `save(path, overwrite=True)` replaces whatever is at `path`.

| File             | Contents                                                              |
|------------------|-----------------------------------------------------------------------|
| `config.json`    | format version, model type, library version, `get_params()`, tags, runtime (device, versions, threads) |
| `data_spec.json` | series, exog and series-exog names, `freq`, the training index         |
| `state.npz`      | non-parameter arrays: DFM's state space, scaled data and smoothed state; DDFM's factors and state space; iVDFM's factors, training targets and explained variance; AFM's residuals; the scaler's arrays |
| `state.json`     | the rest of the fitted state (training diagnostics, engine settings)   |
| `result.npz`, `result.json` | `result_`, so a loaded model reports what the fitted one did |
| `weights.pt`     | torch models only: the network's `state_dict`                         |
| `scaler.json`    | the scaler, for `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `QuantileTransformer` (and None) |
| `scaler.pkl`     | any other scaler                                                      |

Nothing is pickled by default: arrays are read with `allow_pickle=False` and
weights with `torch.load(..., weights_only=True)`; torch 2.10 is the floor
because it fixes CVE-2026-24747 in that loader. A scaler
outside the list above is written to `scaler.pkl`, and loading it needs
`allow_pickle=True`, as does a `config.json` naming an estimator class other
than the public classes of `sklearn.preprocessing`. **`allow_pickle=True` can
run arbitrary code: enable it only for directories you trust** (see
[SECURITY.md](https://github.com/minkeymouse/dfm-python/blob/main/SECURITY.md)).
`load(path, device=)` chooses where torch weights are loaded (`map_location`),
so a run saved on a GPU loads on a CPU-only machine. A single-file `.pkl` or
`.pt` checkpoint from 0.7 raises `ConfigurationError`: load it with the
release that wrote it. `load` recomputes the interpretation attributes from
the saved arrays, so runs saved by 0.8.x have them too, except that an iVDFM
run saved by 0.8.x has no `explained_variance_` and its `decompose` raises
`UnsupportedInputError` (it holds no training targets). `set_output` is not
saved.

```python
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

import dfm_python
from dfm_python import DFM, PanelData
from dfm_python.utils.errors import ConfigurationError

idx = pd.date_range("2015-01-31", periods=60, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(1).standard_normal((60, 3)), index=idx, columns=["a", "b", "c"])
model = DFM(max_iter=20, scaler=PowerTransformer()).fit(PanelData(y=df))
model.save("runs/power", overwrite=True)
assert "scaler.pkl" in os.listdir("runs/power")
try:
    dfm_python.load("runs/power")           # refuses to unpickle by default
except ConfigurationError:
    model = dfm_python.load("runs/power", allow_pickle=True)
```

## YAML configurations

A YAML file has a `model:` section with the constructor parameters and an
optional `data:` section with the `PanelData.from_wide` arguments
(`time_col`, `target_cols`, `exog_cols`, `freq`). Needs the `yaml` extra
(PyYAML also works for `load_yaml`).

| Function | Returns |
|----------|---------|
| `load_yaml(path)` | both sections as dicts |
| `XConfig.from_yaml(path)` | the config built from the `model:` section |
| `Model.from_config(cfg)` | exactly the constructor call `Model(**cfg.to_dict())` |
| `cfg.to_yaml(path)` | writes the config back |

A flat file without `model:`, or an unknown key (including a pre-redesign
name such as `num_factors`, `threshold` or `seed`), raises
`ConfigurationError` naming the canonical key. With omegaconf, `load_yaml`
resolves interpolations, including `${oc.env:NAME}` environment lookups, and
`save()` writes the resolved parameters to `config.json`. **Load YAML files
only from sources you trust.**

```yaml
# config/dfm_macro.yaml (abridged)
model:
  blocks:
    Block_Global:
      n_factors: 3
      series: [KORFBEV, KORWHLS, ..., KOIMPCONA]
  clock: w                       # frequency of the latent factors
  tent_weights:
    m:w: [1, 2, 1]               # monthly series on the weekly clock
  max_iter: 10
  tol: 1.0e-5
  scaler: standard
data:                            # PanelData.from_wide arguments
  time_col: date
  target_cols: [KORFBEV, KORWHLS, ..., KOIMPCONA]
  freq:                          # grouped form; unlisted series use the clock
    w: [KORFBEV, KORWHLS, ...]
    m: [KOEMPTOTO, KOHWRWEMP, ...]
```

```python
import pandas as pd
from dfm_python import DFM, DDFM, PanelData
from dfm_python.config import DDFMConfig, DFMConfig, load_yaml

model = DFM.from_config(DFMConfig.from_yaml("config/dfm_macro.yaml"))   # weekly clock, monthly series
sections = load_yaml("config/dfm_macro.yaml")                          # {"model": {...}, "data": {...}}
data = PanelData.from_wide(pd.read_csv("data/macro.csv"), **sections["data"])
assert data.freq["KOEQUIPTE"] == "m"

ddfm = DDFM.from_config(DDFMConfig.from_yaml("config/ddfm_exchange.yaml"))
ddfm.get_config().to_yaml("ddfm.yaml")                                 # round-trips
assert DDFMConfig.from_yaml("ddfm.yaml") == ddfm.get_config()
```

The repository ships example files in
[`config/`](https://github.com/minkeymouse/dfm-python/tree/main/config),
datasets in [`data/`](https://github.com/minkeymouse/dfm-python/tree/main/data)
and scripts in [`tutorial/`](https://github.com/minkeymouse/dfm-python/tree/main/tutorial).
None of them is in the installed package.

| File | Model | Data |
|------|-------|------|
| `dfm_macro.yaml`     | DFM, weekly clock, 7 weekly and 7 monthly series     | `data/macro.csv` |
| `dfm_finance.yaml`   | DFM, 27 daily series                                  | `data/finance.csv` |
| `dfm_exchange.yaml`  | DFM, 8 daily exchange rates                           | `data/exchange_rate.csv` |
| `ddfm_macro.yaml`    | DDFM, `KOEQUIPTE` with 13 covariates                  | `data/macro.csv` |
| `ddfm_finance.yaml`  | DDFM, `market_forward_excess_returns` with covariates | `data/finance.csv` |
| `ddfm_exchange.yaml` | DDFM, 8 exchange rates                                | `data/exchange_rate.csv` |
| `ivdfm_macro.yaml`   | iVDFM, 3 monthly targets                              | `data/macro.csv` |
| `ivdfm_finance.yaml` | iVDFM, `market_forward_excess_returns`                | `data/finance.csv` |
| `ivdfm_exchange.yaml`| iVDFM, 8 exchange rates                               | `data/exchange_rate.csv` |
| `afm_finance.yaml`   | AFM hyperparameters (`model:` only)                   | your returns and characteristics |

Each `tutorial/tutorial_{macro,finance,exchange}_{dfm,ddfm,ivdfm}.py` loads
its YAML, fits, forecasts, saves and reloads. `--rows`, `--max-iter` or
`--max-epochs`, `--run-dir`, and `--device` for the torch models shorten or
redirect a run:

```bash
python tutorial/tutorial_macro_dfm.py --rows 400 --max-iter 2   # quick check
```

## Reproducibility

The same data and hyperparameters give the same numbers. Seeds:

| Model | Parameter      | Default | What is seeded |
|-------|----------------|---------|----------------|
| DFM   | none           | n/a     | nothing; EM is deterministic |
| DDFM  | `random_state` | 3       | a local `np.random.RandomState` in fit; `torch.manual_seed` when the networks are built |
| iVDFM | `random_state` | 3       | `torch.manual_seed`, then `np.random.seed`, before the networks are built |
| AFM   | `random_state` | None    | `torch.manual_seed` before the network is built |

`random_state` is recorded in `config.json`, and saving a run with
`random_state=None` warns. Results are bitwise reproducible on CPU with one
BLAS thread, one torch thread and deterministic algorithms, on the same
machine and library versions; elsewhere expect agreement to rounding. GPU
runs are not bitwise reproducible.

```python
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from dfm_python import DDFM, PanelData

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

idx = pd.date_range("2015-01-31", periods=60, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(2).standard_normal((60, 3)), index=idx, columns=["a", "b", "c"])
with threadpool_limits(limits=1):
    fcs = [DDFM(n_factors=2, max_iter=5, n_mc_samples=2, random_state=3, device="cpu")
           .fit(PanelData(y=df)).predict(6) for _ in range(2)]
pd.testing.assert_frame_equal(fcs[0], fcs[1], check_exact=True)
```

## Logging

The package is silent by default: its logger has only a `NullHandler`.

| Control | Effect |
|---------|--------|
| `dfm_python.set_verbosity("debug" / "info" / "warning" / "error")` | shows the package's records on standard output from that level up; `get_verbosity()` returns the level |
| `DFM_PYTHON_VERBOSITY=info` (environment) | applied at import |
| `verbose=0 / 1 / 2` (every model) | 0 is silent; 1 shows training progress (EM iterations, epochs) during `fit`; 2 adds debug records |

```python
import numpy as np
import pandas as pd
import dfm_python
from dfm_python import DFM, PanelData

df = pd.DataFrame(np.random.default_rng(3).standard_normal((48, 3)), columns=["a", "b", "c"])
DFM(max_iter=3, scaler="standard", verbose=1).fit(PanelData(y=df))  # prints EM progress
dfm_python.set_verbosity("warning")                                   # warnings from now on
```

## Errors

The exceptions are in `dfm_python.utils.errors` (`NotFittedError` and
`UnsupportedInputError` are also exported from `dfm_python`). Each derives
from the matching built-in, so `except ValueError` still works. Messages name
the argument and the fix: the closest canonical key, the excluding tag, or
the value to pass.

| Exception | Bases | Raised when |
|---|---|---|
| `DFMError` | `Exception` | root of the library's errors |
| `ConfigurationError`, `ConfigValidationError` | `DFMError`, `ValueError` | invalid hyperparameters; unknown config keys; flat YAML; single-file checkpoints; an invalid `by`, `coverage`, `output` or `layout` |
| `DataValidationError` | `DFMError`, `ValueError` | malformed `PanelData`, misaligned index, overlapping `new_rows` data, a polars frame with several Date/Datetime columns |
| `UnsupportedInputError` | `DataValidationError` | a slot, frequency layout, `refit` or method that the tags exclude; `decompose(by="factor")` or `predict_interval` with a decoder or scaler that does not allow it |
| `NotFittedError` | `DFMError`, `sklearn.exceptions.NotFittedError` | `predict`, `allocate`, `update`, `save` or `result_` before `fit` |
| `ModelNotTrainedError` | alias of `NotFittedError` | raised inside the engines |
| `NumericalError`, `NumericalStabilityError` | `DFMError`, `ArithmeticError` | numerical failures inside the engines |

## Upgrading from 0.8

Changes after 0.8.3 that can break code written for 0.8 or change its
numbers:

- DFM `factors_` holds the smoothed current factors, one column per name in
  `factor_names_` (`factor_0`, ... for one block, `<block name>_<j>` for
  several). The frame it held before (`state_0` ... `state_{m-1}`: the
  factors, their lags and the idiosyncratic states) is `states_`.
- DFM `predict_factors(h)` returns the current factors (columns
  `factor_names_`), not the state vector. The full state forecast is no
  longer returned by the estimator; `model.engine_.predict(horizon=h)`
  returns `(series, states)` arrays, the series in the internal order.
- iVDFM `factors_` is the factor path on `index_` (`result_.Z`, the training
  series encoded as one sequence). The `(window, time)` frame it held before
  is `window_factors_`.
- DFM `em_acceleration="squarem"` is the new default, so DFM fits give other
  numbers: at the same `tol`, SQUAREM usually stops in fewer E-steps at a
  log-likelihood at least as high as plain EM's. `n_iter_` counts E-steps
  (EM maps) and `max_iter` bounds them. `em_acceleration=None` runs plain EM.
- DFM tent kernels whose first weight is not 1: the loadings on the lags are
  now proportional to the whole kernel, so any positive multiple of a kernel
  gives the same model. Before, `[2, 4, 6, 4, 2]` gave loadings proportional
  to `[1, 4, 6, 4, 2]`. Kernels that start with 1, such as `[1, 2, 3, 2, 1]`
  and `[1, 2, 1]`, are unchanged.

Numbers: the DFM Kalman filter and smoother use the model's structure from
56 states up, and the M-step reads only the covariance blocks it needs; with
`em_acceleration=None` and a kernel that starts with 1, DFM results match
0.8.3 up to rounding.

Saved runs: runs saved by 0.8.x load. A DFM run loads with
`em_acceleration=None` (it was fitted by plain EM). An iVDFM run has no
`explained_variance_` and cannot `decompose`; one saved before 0.8.3 holds
the average over its training windows in `factors_`. Refit to get the new
attributes.
