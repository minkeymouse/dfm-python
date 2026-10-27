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
- `fit(data, callback=None)` takes a `PanelData` (or a DataFrame) and
  returns the model. `max_time` and `callback` bound training (see
  [Run control](#run-control)); `stop_reason_` says why it stopped.
- `predict(h)` returns a DataFrame of the `h` steps after the last
  conditioning date, one column per series in your order, in data units.
  `predict_factors(h)` returns the factor forecast, one column per name in
  `factor_names_`. AFM allocates instead: `allocate()` and `backtest(data)`;
  its `predict` raises `TypeError`.
- Fitted state lives in attributes ending in `_`: `result_`, `n_iter_`,
  `converged_`, `stop_reason_`, `scaler_`, `factors_`, ... Before `fit` they
  raise `NotFittedError`.
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
- DDFM, iVDFM and AFM (tag `backend="torch"`) need the `deep` extra. Without
  torch they can be built, cloned and configured, and `fit` and `load` raise
  `ImportError` naming `pip install "dfm-python[deep]"`.

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
| `capability:pred_int` | `predict_interval(h, coverage, data)` | yes | yes | yes | no |
| `capability:device` | the `device` hyperparameter | no | yes | yes | yes |

Public names exported from `dfm_python`: the models `DFM`, `DDFM`, `iVDFM`,
`AFM` and their base `BaseFactorModel`; the configs `DFMConfig`, `DDFMConfig`,
`iVDFMConfig`, `AFMConfig`; `PanelData`; `load`; the results `BaseResult`,
`DFMResult`, `DDFMResult`; `calculate_rmse`; the errors `NotFittedError` and
`UnsupportedInputError`; `set_verbosity` and `get_verbosity`. The
walk-forward `backtest` and its `BacktestResult` are in
`dfm_python.evaluation`.

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

## Run control

Every model trains in iterations: DFM's E-steps (those of the SQUAREM
cycles included), DDFM's MCMC iterations, iVDFM's and AFM's epochs. Besides
the model's own criterion (`tol`, and `patience` for iVDFM and AFM) and the
iteration budget (`max_iter`, `max_epochs` for iVDFM and AFM), two controls
end training:

- `max_time`, a hyperparameter of every model (seconds, None by default):
  training stops after the first iteration that ends past the budget,
  keeping what a `max_iter` stop there would keep. `converged_` is False and
  `fit` warns with `sklearn.exceptions.ConvergenceWarning`. For DFM the
  budget includes the initialization.
- `fit(data, callback=f)`: `f(info)` runs after every iteration, and a true
  return value stops training there, without a warning. An exception raised
  by `f` propagates and leaves the model unfitted.

`stop_reason_` (also in saved runs) says which ended it.
When the model's own stop and a callback or time stop fall on the same
iteration, the model's own is reported.

| `stop_reason_` | Meaning |
|---|---|
| `"converged"` | the objective changed by less than `tol`; AFM with `factor_model="pca", trading="ou"`, which has nothing to train |
| `"early_stopping"` | iVDFM and AFM: `patience` epochs without improvement. DFM: a numerical guard (a log-likelihood decrease, a failed E-step or M-step, non-finite values) stopped EM, which keeps the accepted parameters with the best log-likelihood |
| `"max_iter"` | the iteration budget ran out |
| `"max_time"` | the time budget ran out |
| `"callback"` | the callback returned a true value |

The `info` dict:

| Key | Contents |
|---|---|
| `model` | the class name |
| `iteration` | the iterations run so far, from 1 |
| `max_iter` | the iteration budget |
| `elapsed` | seconds since training started |
| `objective`, `objective_name` | DFM: `"loglik"`, the log-likelihood of that E-step's parameters (NaN at a SQUAREM extrapolation whose E-step or M-step failed). DDFM: `"loss"`, the mean squared error that `tol` watches. iVDFM: `"elbo"`, the epoch's mean negative ELBO. AFM: `"loss"`, the epoch's `-(Sharpe + lambda_var * EV)` |
| `state` | DFM only: the engine's `DFMModelState` of the parameters EM would end on there (series in the internal order `[clock \| slower]`) |

```python
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from dfm_python import DFM, PanelData

idx = pd.date_range("2015-01-31", periods=96, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((96, 3)), index=idx, columns=["a", "b", "c"])

trace = []
def log_and_stop(info):
    trace.append((info["iteration"], info["objective"]))   # (E-step, log-likelihood)
    return info["iteration"] >= 5                          # stop after 5 E-steps

model = DFM(n_factors=1).fit(PanelData(y=df), callback=log_and_stop)
assert model.stop_reason_ == "callback" and model.n_iter_ == 5 and not model.converged_

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rushed = DFM(n_factors=1, max_time=1e-9).fit(PanelData(y=df))   # stops after the first E-step
assert rushed.stop_reason_ == "max_time"
assert any(issubclass(w.category, ConvergenceWarning) for w in caught)
```

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
linear in the factors (DFM, DDFM with `decoder_type="linear"`, iVDFM with
`decoder_type="linear"` and `n_regimes=1`) and `loadings_` are the model's
coefficients. `"mean_jacobian"`: the decoder is nonlinear (DDFM with
`decoder_type="mlp"`, the other iVDFMs) and `loadings_` is its Jacobian with
respect to the factors averaged over the rows of `factors_`, a local linear
summary rather than coefficients.

`decompose(by=...)` splits the training data, in data units, on the rows of
`factors_`. The parts add up to the data.

| `by` | Columns | Models |
|------|---------|--------|
| `"component"` (default) | `(component, series)`: `"common"`, the factors through the loadings or the decoder, inverse-scaled (so it includes the series mean), and `"idiosyncratic"`, the data minus `"common"` (NaN where the data is missing) | DFM, DDFM, iVDFM |
| `"factor"` | `(part, series)`: one part per name in `factor_names_`, `"mean"` (the scaler's offset; for DDFM and iVDFM plus the decoder bias in data units) and `"idiosyncratic"` | `loadings_kind_` `"exact"`; scaler None, `"standard"`, `"robust"`, `"minmax"` or `"maxabs"` |

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

DFM, DDFM and iVDFM (tag `capability:pred_int`) give
`predict_interval(h, coverage=0.9, data=None)`: intervals of the `h` steps
after the last conditioning date, in data units, each holding `coverage` of
the forecast distribution. The result is indexed like `predict(h)`, with
MultiIndex columns `(series, bound)`, `bound` `"lower"` or `"upper"`, and
`attrs["coverage"]`. `data=` conditions on new rows on a copy, as in
`predict(h, data=)`. Every model takes its fitted parameters as known, so
the intervals leave out estimation uncertainty. AFM raises
`UnsupportedInputError`.

| Model | Distribution | Centre |
|-------|--------------|--------|
| DFM | Gaussian, from the Kalman forecast moments: the filtered state covariance `P` at the last conditioning row, propagated by `P <- A P A' + Q`, gives the scaled series variance `diag(C P C' + R)` | `predict(h)` |
| DDFM, linear decoder | Gaussian: the factor covariance starts from the covariance of the Monte Carlo factor draws at the last conditioning row and propagates through the AR(1) transition; the decoder weight maps it to the series, plus per series the mean square of the training residual around the decoder | `predict(h)` |
| DDFM, MLP decoder | the quantiles of 2000 simulated paths: factors drawn from the same Gaussian state space, decoded, plus the residual noise; seeded with `random_state` | not `predict(h)` |
| iVDFM | the quantiles of 1000 sampled forecast paths: innovations from the prior given the future context (`futr_exog=`, as in `predict`), decoded, plus the decoder noise `decoder_var`; drawn from the torch RNG seeded with `random_state_`, which is restored afterwards | not `predict(h)` |

The Gaussian bounds are `predict(h)` minus and plus `z * sd`, with `z` the
standard normal quantile of `(1 + coverage) / 2`, and need an affine scaler
(None, `"standard"`, `"robust"`, `"minmax"` or `"maxabs"`): DFM raises
`UnsupportedInputError` for any other, and DDFM inverse-transforms the
scaled bounds, which assumes the scaler maps each series separately and
monotonically (`"quantile"` does). The simulated bounds are the empirical
`(1 - coverage) / 2` and `(1 + coverage) / 2` quantiles of the paths after
the inverse transform, so they take any scaler; their centre is the
distribution's, which differs from `predict`'s forecast where the decoder is
nonlinear or, for iVDFM, the prior is not centred at zero. With
`random_state=None` the simulated bounds change from call to call.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2015-01-31", periods=96, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((96, 3)), index=idx, columns=["a", "b", "c"])
model = DFM(n_factors=1, max_iter=50).fit(PanelData(y=df.iloc[:84]))

bands = model.predict_interval(6, coverage=0.9)   # DataFrame 6 x 6: (a, lower), (a, upper), ...
bands["a"]                                        # columns lower, upper
mid = (bands.xs("lower", axis=1, level="bound") + bands.xs("upper", axis=1, level="bound")) / 2
np.testing.assert_allclose(mid.to_numpy(), model.predict(6).to_numpy())   # centred on predict
later = model.predict_interval(6, data=PanelData(y=df.iloc[84:]))        # after 12 new rows; model unchanged
```

## Backtesting

`dfm_python.evaluation.backtest(estimator, data, *, start, h, step=1,
refit=False, coverage=None)` evaluates a forecasting model walk-forward. The
origins are the rows `start`, `start + step`, ... of `data` that at least
one row follows (`start` is a row position, negative from the end, or a
time stamp). At each origin the model sees the rows up to and including it,
and nothing later:

- at the first origin, and wherever `refit` says so, a fresh clone of
  `estimator` is fitted on them (`refit=True`: every origin; `refit=k`:
  every k-th origin);
- elsewhere the current model's `update` conditions on them with its fitted
  parameters (the rows since the previous origin for DFM and DDFM, every
  row up to the origin for iVDFM).

Then `predict(h)`, and with `coverage` `predict_interval(h, coverage)`,
forecast the `h` rows after the origin. An iVDFM fitted with exog gets the
exog rows after the origin as `futr_exog` when `data` holds them, which
treats exog as known in advance. `estimator` itself is never fitted.

The `BacktestResult` holds `forecasts`, a long frame with one row per
origin, horizon and series (`origin`, `time`, `horizon`, `series`,
`forecast`, `actual`, and `lower`, `upper` with `coverage`; the actual is NaN
where the data is missing or ends), `origins`, `fit_seconds` (per origin)
and `coverage`. `metrics(by=)` gives RMSE, MAE, bias (mean of forecast minus
actual), `n` and, with intervals, the empirical coverage, over the rows with
an actual: by `"horizon"` (the default), `"series"`, both as a tuple, or
`None` for one row.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData
from dfm_python.evaluation import backtest

rng = np.random.default_rng(6)
f = np.zeros(120)
for t in range(1, 120):
    f[t] = 0.7 * f[t - 1] + rng.standard_normal()
idx = pd.date_range("2015-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.outer(f, [1.0, 0.8, -0.6]) + 0.5 * rng.standard_normal((120, 3)),
                  index=idx, columns=["a", "b", "c"])

result = backtest(DFM(n_factors=1), PanelData(y=df), start="2022-12-31", h=3,
                  step=3, refit=4, coverage=0.8)       # refit at every 4th origin, update between
result.origins                       # 2022-12-31, 2023-03-31, ..., 2024-09-30
result.forecasts.head(3)             # origin, time, horizon, series, forecast, actual, lower, upper
result.metrics()                     # RMSE, MAE, bias, n, coverage per horizon
result.metrics(by=("horizon", "series"))
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
`random_state=None` warns. The simulated prediction intervals (DDFM with the
MLP decoder, iVDFM) are seeded with it too. Results are bitwise reproducible on CPU with one
BLAS thread, one torch thread and deterministic algorithms, on the same
machine and library versions; elsewhere expect agreement to rounding. GPU
runs are not bitwise reproducible. The example needs the `deep` extra.

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
| `DataValidationError` | `DFMError`, `ValueError` | malformed `PanelData`, misaligned index, overlapping `new_rows` data, a polars frame with several Date/Datetime columns; infinite values in `fit`, `predict(data=)` or `update` data (iVDFM fills them like missing values); a target series with no observed value, a single one, or the same value at every observation up to rounding (`fit`, naming each series) |
| `UnsupportedInputError` | `DataValidationError` | a slot, frequency layout, `refit` or method that the tags exclude; `decompose(by="factor")` or `predict_interval` with a decoder or scaler that does not allow it |
| `NotFittedError` | `DFMError`, `sklearn.exceptions.NotFittedError` | `predict`, `allocate`, `update`, `save` or `result_` before `fit` |
| `ModelNotTrainedError` | alias of `NotFittedError` | raised inside the engines |
| `NumericalError`, `NumericalStabilityError` | `DFMError`, `ArithmeticError` | numerical failures inside the engines |

A missing optional dependency raises `ImportError` naming the extra to
install, for example `pip install "dfm-python[deep]"` for the torch models.

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
- DFM's default `tol` is 1e-6 (was 1e-5): EM runs longer and ends at a
  higher log-likelihood. DFM, DDFM and iVDFM default to `scaler="standard"`
  (was None). Pass `tol=1e-5, scaler=None` for the 0.8 defaults.
- DFM `fit(data, checkpoint_callback=...)` is removed: use
  `fit(data, callback=...)` (see [Run control](#run-control)).
- DDFM, iVDFM and AFM need the `deep` extra, `pip install
  "dfm-python[deep]"`; a plain `pip install dfm-python` no longer installs
  torch. `dfm_python.config.Tensor`, `Loss` and `Optimizer` are removed.
- `fit` raises `DataValidationError` for a target series with no observed
  value, a single one, or a constant one, and `fit`, `predict(data=)` and
  `update` for infinite values.
- iVDFM forecasts change: with `factor_order=1` (the default) step h was one
  transition too far ahead; with `factor_order` > 1 the companion matrix
  gave the factors no autoregressive dynamics and the forecast started from
  zero lags. DDFM forecasts change: they now run the trained decoder,
  including its bias (and the MLP for `decoder_type="mlp"`), instead of the
  first columns of its output weight.

Numbers: the DFM Kalman filter and smoother use the model's structure from
56 states up, and the M-step reads only the covariance blocks it needs; with
`em_acceleration=None` and a kernel that starts with 1, DFM results match
0.8.3 up to rounding.

Saved runs: runs saved by 0.8.x load. A DFM run loads with
`em_acceleration=None` (it was fitted by plain EM). An iVDFM run with
`factor_order` > 1 saved before this release loads with a warning: it was
trained with the defective companion matrix, so refit it. An iVDFM run has no
`explained_variance_` and cannot `decompose`; one saved before 0.8.3 holds
the average over its training windows in `factors_`. Refit to get the new
attributes.
