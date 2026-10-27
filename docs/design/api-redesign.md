# API redesign specification

This document specifies how the four models in `dfm_python` (DFM, DDFM, iVDFM, AFM) become one consistent library. It changes packaging, naming and the order of calls. It does not change numerical code: EM with its Kalman filter and smoother, MCMC denoising, the ELBO loop and the Sharpe objective are called unchanged. Given the same data and the same hyperparameters, every engine produces the same numbers as before on CPU, and golden regression tests enforce this.

Backward compatibility with the old API is not a goal. The next minor release ships the new API for all four models and removes the old constructors outright. Projects that need the old API, including the scripts behind published papers, pin an older release on their side. No version numbers are assigned here.

## 1. Summary

The target follows sklearn fitted-attribute conventions, neuralforecast's `fit(data)`, and Hugging Face config and save conventions:

- `Model(**hyperparameters)` stores hyperparameters and does nothing else.
- `fit(data)` returns `self`. `data` is one shared container, `PanelData`.
- `predict(h)` returns a time-indexed `pandas.DataFrame`, and factors come from `predict_factors(h)`. No method returns a tuple. AFM, which allocates rather than forecasts, uses `allocate()`.
- Fitted state lives in attributes ending in `_`. The fitted result has one home, `result_`.
- `save(dir)` writes a pickle-free directory. `Model.load(dir)` and `dfm_python.load(dir)` need no dataset.

The current classes are not rewritten. Each moves verbatim into a private engine module, and the public class is a thin estimator that builds the engine's inputs from `PanelData` and passes every hyperparameter explicitly. The new path therefore runs today's code, and golden fixtures captured from the current code before any engine moves prove that the same data and the same hyperparameters give the same numbers.

Config precedence bugs are fixed when each model is migrated: constructor hyperparameters are the single source of truth, and YAML builds the same hyperparameters. Today iVDFM's keyword defaults overwrite config values and DDFM ignores its config except interpolation; after migration, the value you set, in Python or in YAML, is the value the engine receives.

### Decisions on contested points

| Question | Decision and reason |
|---|---|
| iVDFM keyword defaults (`factor_order=1`, `encoder_hidden_dim=200`, `decoder_var`, `tolerance`, `seed=3`, ...) overwrite config values; DDFM ignores its config except interpolation | Bugs, fixed when each model is migrated. The constructor is the only place a hyperparameter is set, YAML and `from_config` build the same constructor call, and the estimator passes every value to the engine explicitly. Nothing is merged, so what you set is what runs. |
| iVDFM forecasts are in scaled model space | The fitted target scaler lives on today's iVDFM dataset, so the estimator inverse-transforms the engine's forecast with `scaler_` and returns data units, like DFM and DDFM. With `scaler=None` the transform is the identity. The engine's model-space output is unchanged and is what the golden test compares. |
| DFM column order `[clock | slower]` | `fit` applies a stable partition (relative order kept within each group) and reports in the user's order. Ordered input gives the identity permutation, so results are bitwise unchanged. Any other order raises today, so no result depends on it. The internal order is exactly what the error message asks users to build by hand, so no new permutation sensitivity appears. A golden test on permuted input pins this. |
| Meaning of `update()` | One promise: after `update(data)`, `predict` conditions on `data`, and fitted parameters change only with `refit=True`. The mechanism stays per model and is published as tag `update_kind` and one docstring table. Unifying the mechanism would change numbers. |
| Default horizon | `h` is required. Today's per-model defaults (one year of clock periods, 6, 1) are dropped. |
| AFM's `predict` | AFM gets `allocate()` and `backtest()`. `AFM.predict` raises `TypeError` naming `allocate`. `predict(h)` then means the same thing everywhere it exists. |
| iVDFM `stride` | `stride=None` means `window // 2`, the dataset default and the only stride that ever reached training (the config's `stride=1` is never read by `fit`). |
| Pre-fitted scaler | `fit` clones and fits `scaler`, and warns if a fitted instance's statistics are discarded. |
| DFM load without a dataset | Each engine gains a private `_from_saved` that bypasses `__init__` (DFM's `__init__` derives blocks from dataset columns). Blocks, frequencies, column orders, `data_processed` and the last smoothed state are saved. |
| Parity scope | Bitwise on CPU with one BLAS thread, `torch.set_num_threads(1)` and deterministic algorithms, on the machine that captured the fixtures. Elsewhere the golden comparison falls back to a stated tolerance (section 11). GPU runs are compared to a tolerance. |
| Seed defaults (DDFM 3, iVDFM 3, AFM None, DFM none) | Kept per model, documented in one table, and always recorded in `config.json`. |
| iVDFM tutorials (broken today) | Rewritten on the new API, together with every other tutorial. |

## 2. Goals and non-goals

Goals:

1. A user who knows sklearn or neuralforecast can guess the API.
2. One data container for all four models, AFM included.
3. One way to set each hyperparameter: a keyword-only constructor argument. The `kwargs > config > defaults` merge, AFM's `pick()` and the `getattr(config, ...)` fallbacks leave the public path.
4. One result location, one save format, and one meaning each for `device`, `random_state` and `verbose`.
5. Same data and same hyperparameters give the same numbers from each engine (bitwise on CPU), enforced by golden fixtures captured before any engine code moves.
6. An open door to a later sktime-like ecosystem through tags, a registry and a public `check_estimator`, with no sktime dependency or compatibility claim.

Non-goals:

- No backward compatibility with the old API: no old constructors, parameter aliases, flat YAML loading or single-file checkpoint reading.
- No new models, losses, inference algorithms, probabilistic forecasts, backtesting framework or trainer.
- No change to estimation code, initialization order or RNG consumption order.
- No sklearn `check_estimator` compliance; only `BaseEstimator` machinery is reused.
- No `safetensors`, no Hub upload, and no bundling of training data in saved runs.
- No cleanup of untyped access inside the frozen engines (most of the 76 `hasattr` calls).

## 3. Public API at a glance

### DFM (numpy, mixed frequency, EM + Kalman)

```python
import numpy as np, pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2010-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((120, 4)),
                  index=idx, columns=["gdp", "s0", "s1", "s2"])
df.loc[df.index.month % 3 != 0, "gdp"] = np.nan        # quarterly series, any column position

data = PanelData(y=df, freq={"gdp": "q"})                # unlisted series use the clock frequency
model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
            max_iter=50, tol=1e-5, scaler="standard")    # blocks=None: one global block
model.fit(data)                                          # returns model
fc = model.predict(6)                                    # DataFrame 6 x 4, columns in your order
zf = model.predict_factors(6)                            # DataFrame 6 x state_dim
model.result_.loglik, model.n_iter_, model.converged_
model.save("runs/dfm_macro")
model = DFM.load("runs/dfm_macro")                       # no dataset needed
```

### DDFM (torch autoencoder + MCMC denoising)

```python
from dfm_python import DDFM, PanelData

data = PanelData.from_wide(raw_df, time_col="date", exog_cols=["oil", "fx"])  # other columns are targets
model = DDFM(n_factors=4, encoder_layers=(16,), decoder_type="linear",
             learning_rate=0.005, n_mc_samples=10, max_iter=200, tol=5e-4,
             scaler="standard", random_state=3, device="cpu", verbose=1)
model.fit(data)                                          # builds the state space too
fc = model.predict(6)                                    # targets only
fc_new = model.predict(6, data=data_with_new_rows)       # conditions on new rows; model unchanged
model.update(data_with_new_rows)                         # explicit, mutating; returns model
model.save("runs/ddfm_macro")
```

### iVDFM (torch, identifiable VAE on innovations)

```python
from dfm_python import iVDFM, PanelData
from dfm_python.tuning import tune_hyperparameters       # needs the `tune` extra

data = PanelData.from_wide(raw_df, time_col="date", exog_cols=["policy_rate"])  # exog = context u_t
model = iVDFM(n_factors=5, factor_order=3, window=200, stride=None,
              innovation_distribution="laplace", learning_rate=1e-3,
              batch_size=256, max_epochs=10, scaler="standard",
              random_state=3, device="auto")
model.fit(data)
fc = model.predict(6, futr_exog=future_policy_df)        # data units
best = tune_hyperparameters(model, data, n_trials=30, horizon=96, metric="sMSE")
model.set_params(**best).fit(data)
model.save("runs/ivdfm_macro")
```

### AFM (torch, attention factors for statistical arbitrage)

```python
from dfm_python import AFM, PanelData

data = PanelData.from_arrays(y=returns, series_exog=characteristics,        # (T, N), (T, N, M)
                             index=dates, names=tickers, series_exog_names=char_names)
model = AFM(n_factors=3, hist_len=20, max_epochs=20, random_state=0, device="auto")
model.fit(data)
w = model.allocate()                                     # DataFrame (T - hist_len) x N of weights
res = model.backtest(test_data)                          # AFMResult: residuals, weights, book returns, Sharpe
model.result_.sharpe, model.residuals_, model.book_returns_
AFM(n_factors=3, factor_model="pca", trading="ou").fit(PanelData(y=returns_df))
model.save("runs/afm")
```

## 4. Data container

`dfm_python.PanelData` is a frozen, validated container that uses only numpy and pandas. It holds no device or scaler, and it never imputes, scales or reorders values.

| Slot | Type and shape | Meaning | Used by |
|---|---|---|---|
| `y` | DataFrame, T x N | observed target series (macro panel or asset returns) | all |
| `exog` | DataFrame, T x K, or None | per-time inputs that are not forecast | DDFM (encoder covariates), iVDFM (context u_t) |
| `series_exog` | DataFrame, T x (M*N), column MultiIndex `(feature, series)`, or None | per-time, per-series features | AFM (characteristics) |
| `freq` | `dict[str, str]` or None | per-series sampling frequency | DFM |

Constructors: `PanelData(y, exog=None, series_exog=None, freq=None)`, `from_wide(df, *, time_col, target_cols, exog_cols, freq)`, `from_long(df, *, id_col="unique_id", time_col="ds", target_col="y", series_exog_cols=(), exog_cols=())`, and `from_arrays(y, *, exog, series_exog, index, names, exog_names, series_exog_names, freq)`. Accessors: `index`, `series_names`, `exog_names`, `series_exog_names`, `n_timesteps`, `is_mixed_frequency`, `series_exog_array()` (T x N x M), `to_long()` and `loc(start, stop)`. Polars input works with the `polars` extra. `fit`, `predict` and `update` also accept a bare DataFrame (meaning `PanelData(y=df)`) or a 2-D array.

Validation:

- The index is unique and increasing, and is a `DatetimeIndex`, `PeriodIndex` or `RangeIndex`. Arrays get a `RangeIndex`.
- All slots share the index.
- `freq` accepts the per-series form `{"gdp": "q"}` and the grouped form `{"q": ["gdp"]}`, normalized in one place. Today the `BaseModelConfig` docstring recommends the grouped form, but the constructor rejects it.

AFM needs no special signature. In long format a firm characteristic is just another column on each `(unique_id, ds)` row, which makes it a per-series exogenous variable. `AFMDataset`'s `window`, `stride` and `scaler` are stored but never read by `fit` or `get_tensors`, so dropping them loses nothing.

A slot the model's tags exclude raises `UnsupportedInputError` and is never silently ignored. Examples: `exog` given to DFM, `series_exog` given to DDFM, a mixed-frequency panel given to iVDFM, or attention-mode AFM without `series_exog`.

The four `*Dataset` classes become private adapters that `fit` builds from `PanelData` and the hyperparameters. Each calls today's constructor with the same arguments in the same order:

- DDFM builds `[exog | y]` in that column order. The existing code scales the covariates with a fresh `type(scaler)()`; the dataset keeps that fitted instance (the engine's `covariate_scaler`) so that `update` scales new covariates with the training statistics.
- iVDFM runs today's interpolation, then forward and backward fill, then zero fill, then windowing.
- DFM applies the stable column partition and builds the `DFMConfig` that its dataset requires.

Scaling has one public owner, the `scaler` hyperparameter, and the fitted copy is exposed as `scaler_`. Inside the engines the fitted scaler stays where it is today, and `scaler_` points at it.

## 5. Config

### The constructor is the config

Every public model subclasses `sklearn.base.BaseEstimator`. scikit-learn is already a core dependency. `__init__` takes keyword-only hyperparameters and only stores them, with no validation, seeding, device work or data access. That gives `get_params`, `set_params`, `clone`, `__repr__` and Hydra `instantiate` for free. Validation runs at the start of `fit`.

The `*Config` dataclasses stay as the typed, serializable mirror of the constructor, in the role of Hugging Face's `PretrainedConfig`. Each is rebuilt on a new base, `dfm_python.config.EstimatorConfig`, when its model is migrated. A config has exactly the constructor's fields and defaults. `get_config()` returns one whose fields equal `get_params()`. `from_config(cfg)` is equivalent to `Model(**cfg.to_dict())`. `from_yaml` (with the `yaml` extra), `to_yaml`, `to_dict` and `from_dict` round-trip.

Unknown keys raise `ConfigurationError`, naming the key and the closest canonical name. There is no alias table: old names such as `num_factors`, `threshold` or `seed` are unknown keys.

### Precedence

Constructor hyperparameters are the single source of truth. `from_config`, `from_dict` and `from_yaml` build exactly the constructor call `Model(**params)`, and nothing else contributes a value. There is no merge of keywords over config over defaults, no `pick()`, and no `getattr(config, ...)` fallback on the public path. Today's precedence bugs are fixed when each model is migrated:

- iVDFM: keyword defaults overwrite config values.
- DDFM: the config is ignored except interpolation.
- AFM: keyword, then config, then default, through `pick()`.
- DFM: keyword arguments mutate the caller's config.

A consequence worth stating in the CHANGELOG: shipped iVDFM and DDFM YAMLs now run the values they state. For example, `ivdfm_macro`'s `factor_order: 3` used to run as 1, and `ivdfm_finance`'s `encoder_hidden_dim: 50` used to run as 200.

### YAML layout

A YAML file has a `model:` section (constructor parameters) and a `data:` section (`time_col`, `target_cols`, `exog_cols`, `freq`). `dfm_python.config.load_yaml(path)` returns both sections as plain dicts, `XConfig.from_yaml(path)` builds the config from the `model:` section, and `PanelData.from_wide(df, **sections["data"])` uses the other. A flat file without a `model:` section raises `ConfigurationError`. The shipped YAMLs are rewritten in this layout as each model is migrated.

### Canonical hyperparameters

Today's names are in parentheses for reference only; they are not accepted. A default is today's constructor default, the value that ran when the argument was omitted, wherever it differs from the config default. For example, DDFM `learning_rate` is 0.005 and iVDFM `factor_order` is 1.

| Model | Parameters |
|---|---|
| DFM | `n_factors` (used when `blocks=None`; None means 1), `blocks` (block keys `n_factors`, default 1, and `series`, required), `clock`, `tent_weights`, `max_iter`, `tol` (`threshold`), `ar_clip` (keys `min`, `max`), `data_clip`, `regularization` (keys `scale`, `min_eigenvalue`, `max_eigenvalue`), `damping_factor`, `scaler`, `verbose` |
| DDFM | `n_factors`, `encoder_layers` (hidden widths; today's `encoder_size` is `(*encoder_layers, n_factors)`), `decoder_type`, `activation`, `learning_rate`, `optimizer`, `n_mc_samples`, `window_size`, `max_iter`, `tol` (`tolerance`), `interpolation_method`/`_limit`/`_limit_direction`, `scaler`, `random_state` (`seed`), `device`, `verbose` (`disp`; the engine always gets `disp=10`) |
| iVDFM | `n_factors` (`num_factors`), `window`, `stride`, `time_context`, encoder/decoder/prior `*_hidden_dim` and `*_n_hidden_layers`, `decoder_type`, `activation`, `slope`, `factor_order`, `innovation_distribution`, `decoder_var`, `n_regimes` (`num_regimes`), `regime_temperature`, `beta_kl`, `use_layer_norm`, `mixing`, `learning_rate`, `optimizer`, `optimizer_weight_decay`, `optimizer_momentum`, `batch_size`, `max_epochs`, `tol` (`tolerance`), `patience`, six `scheduler_*`, `scaler`, `random_state` (`seed`), `device`, `verbose` |
| AFM | `n_factors` (`num_factors`), `factor_model`, `trading`, `embed_dim`, `hist_len`, `n_kernels`, `ridge`, `squash_lambda`, `pca_window`, `reestim`, `lambda_var`, `turnover_cost`, `short_cost`, `risk_free`, `learning_rate`, `max_epochs`, `patience`, `random_state` (`seed`), `device`, `verbose` |

Naming rules:

- `n_*` for counts.
- `max_iter` for outer algorithm iterations (EM, MCMC) and `max_epochs` for gradient passes.
- `tol` everywhere.
- `random_state` for seeds.

Removed parameters:

- Inferred in `fit`: `data_dim`, `context_dim`, `n_char`.
- Never read: `DFMConfig.idio_rho0` and `idio_min_var` (neither `to_em_config` nor the initialization reads them); `DDFMConfig.use_batch_norm`, `use_idiosyncratic`, `min_obs_idio` and `lags_input`; `DDFMConfig.num_factors`, which can contradict `encoder_size[-1]`; `iVDFMConfig.covariates`; DDFM's `max_epoch_pre_train` and `min_target_interporate_ratio`, which the engine stores but never reads (pre-training runs `n_mc_samples` epochs).
- Replaced: DDFM's `encoder_size` by `n_factors` plus `encoder_layers`; DDFM's `dataset` and `config` by `fit(data)` and the constructor.
- Moved: `target_scaler` and `frequency` become `scaler` and `PanelData.freq`; `iVDFMConfig.context` becomes `PanelData.exog`.
- Deleted outright: `DFM.load_config`, which calls a missing `_load_config_common` and cannot run.

`blocks=None` gives one global block with `n_factors` factors (1 when `n_factors` is None). That is what `DFMConfig.from_dict` already does for a config without blocks. Setting both `n_factors` and `blocks` raises `ConfigurationError`. `tent_weights=None` is fine for single-frequency data; for mixed-frequency data today's resolution path raises `DataValidationError`, so `tent_weights` is required for the slower frequency and there is no default kernel (section 15, question 3). `fit` raises `ConfigurationError` before the engine runs; the message gives the symmetric kernel that `generate_tent_weights(2k - 1)` produces, with `k` the clock periods per slower period (`[1, 2, 3, 2, 1]` for `q` on an `m` clock), and the exact argument to pass. A key is `"f:clock"` or `"f"`, `f` slower than `clock`, with at most 12 positive weights. A series faster than the clock, or a second slower frequency, raises `UnsupportedInputError`.

## 6. fit / predict / result_ contract

```python
class BaseFactorModel(sklearn.base.BaseEstimator):
    def fit(self, data, **fit_kwargs) -> Self
    def predict(self, h: int, data=None, **predict_kwargs) -> pd.DataFrame   # task == "forecast"
    def predict_factors(self, h: int, data=None) -> pd.DataFrame            # capability:factor_forecast
    def update(self, data, *, refit: bool = False) -> Self                  # capability:update
    def save(self, path, *, overwrite: bool = False) -> None
    @classmethod
    def load(cls, path, *, device=None, allow_pickle: bool = False) -> Self
    def get_config(self) -> EstimatorConfig
    @classmethod
    def from_config(cls, cfg) -> Self
    @classmethod
    def get_tags(cls) -> dict
```

### fit

`fit(data)` always returns `self`. It runs these steps:

1. Validate the parameters and data.
2. Build the private adapter.
3. Construct the engine with every parameter explicit. For iVDFM and AFM the engine's `__init__` seeds and builds the network, as today.
4. Train.
5. For DDFM, run `build_state_space()`, a deterministic least-squares step.
6. Set `result_` eagerly from today's `get_result()`.

Refitting starts from scratch. AFM's warm start is reached only through `update(data, refit=True)`. `fit` never mutates `data` or a hyperparameter. Derived values (`time_context_`, `n_char_`, internal column order) become underscore attributes. The only `fit_kwargs` is DFM's keyword-only `checkpoint_callback`.

### predict and allocate

`predict` never mutates fitted state, and no flag changes its return type:

- DDFM's `predict(data, update=True)` mutates the model today. The estimator runs its `update` on a deep copy of the engine, so `predict(h, data=new)` equals `update(new).predict(h)` and the model is untouched.
- DDFM's `return_series` and `return_factors` flags become two methods, `predict` and `predict_factors`.
- `h` is a required integer. Anything else raises `TypeError`. Today `iVDFM.predict(6)` binds 6 to `data`.

What `data` means keeps today's per-model semantics, stated by tag `predict_data`:

| Model | `predict_data` | What `data` is |
|---|---|---|
| DFM | `new_rows` | rows strictly after the last conditioning timestamp (`index_[-1]` until `update`); overlap raises `DataValidationError`; same series and `freq` (`None` keeps the fitted one); a `RangeIndex` must continue the model's index |
| DDFM | `new_rows` | rows strictly after the last conditioning timestamp, as for DFM; overlap raises `DataValidationError`; a `RangeIndex` must continue the model's index (until the fix before the 0.8.0 release this was `full_panel`, the training rows plus the new ones; see section 14) |
| iVDFM | `window` | every row passed is encoded, and the last state starts the forecast |

iVDFM's extra keywords keep today's defaults: `futr_exog` (was `context_data`), `deterministic=True` and `n_samples=1` (was `num_samples`).

`allocate(data=None)` exists only on models tagged `task="allocation"` (AFM) and returns weights. `backtest(data)` returns an `AFMResult` holding residuals, weights, book returns, Sharpe ratio and explained variance computed on `data`.

### Forecast DataFrame

- **Index.** The `h` timestamps after the last conditioning timestamp. The step comes from `index_.freq`, then from `pd.infer_freq`, then from the clock (`m` maps to `ME`, `q` to `QE`, `w` to `W`). A `RangeIndex` continues by position. The training index name is kept, or `"time"` if it had none. For `allocate`, the index is `data.index[hist_len:]`, and row *i* holds the weights applied to returns at that timestamp.
- **Columns.** `feature_names_in_` in the user's original order, even though DFM reorders internally.
- **Values.** float64.
- **Units.** Data units for every forecaster. The forecast is inverse-transformed with `scaler_` when a scaler is set; with `scaler=None` the data is never scaled.
- **attrs.** `{"model", "kind": "forecast" | "factors" | "weights", "origin": last conditioning timestamp}`.
- **Mixed frequency.** DFM reports slower series at every clock date, as today (section 15, question 5).

### result_ and fitted attributes

`result_` holds today's `DFMResult`, `DDFMResult`, `iVDFMResult` or `AFMResult`. The estimators have no `training_state`, `_result`, `get_result()` or `result`. The `*ModelState` dataclasses become private and are used only by save and load.

Common fitted attributes: `result_`, `n_features_in_`, `feature_names_in_`, `exog_names_in_`, `index_`, `n_iter_`, `converged_`, `n_factors_`, `scaler_` and `random_state_`.

Tag-dependent fitted attributes:

- `factors_`: a DataFrame on `index_`. iVDFM uses a `(window, time)` MultiIndex, and DFM state columns are named `state_0..m`.
- `device_` and `network_`: torch models only. `network_` is the engine module.
- `series_exog_names_in_`, `residuals_` and `book_returns_`: AFM only.
- `engine_` (the private `DFMEngine`), `internal_feature_names_` (the series in the engine's `[clock | slower]` order, which is the order of the series-indexed matrices `C`, `R` and `x_sm` in `result_`), `freq_` (each series' frequency, user order) and `loglik_`: DFM only.

### update

`update(data, *, refit=False) -> self` keeps today's mechanism for each model:

| Model | `update_kind` | `refit=False` | `refit=True` |
|---|---|---|---|
| DFM | `filter` | the Kalman filter continues over the new rows with fixed parameters (prior `(A x_T|T, A P_T|T A' + Q)` from the filter over the conditioning rows) and appends their filtered states; `result_` unchanged | today's `retrain=True`: one EM iteration on all rows with threshold `tol`; `result_` refreshed |
| DDFM | `encoder_forward` | the new rows are scaled with the scalers fitted in `fit` (never refitted), interpolated together with the conditioning rows as `fit` interpolates its data, encoded under `no_grad`; their factors are appended (until the fix before the 0.8.0 release this was `encoder_forward_rescale`: the whole panel re-encoded and the target scaler refitted; see section 14) | raises `UnsupportedInputError` |
| iVDFM | `encoder_forward` | encoder forward under `no_grad`; appends factors | raises `UnsupportedInputError` |
| AFM | `reattach` | attaches data for later `allocate()` | today's `retrain=True`: training continues from the current weights |

DFM's `rescale` and iVDFM's `append` and `store_full_history` flags are not part of the estimator. It calls the engine with today's default values for them. DFM's `predict(h, data=new)` runs the same continued filter on a copy and forecasts from the last filtered state, so it equals `update(new).predict(h)`; one new row and many new rows take the same path. (Before the numerical-stability rewrite, listed under Unreleased in the CHANGELOG, `predict(h, data=)` started from `V_0` and `update` from the last filtered covariance without a transition step, so the two differed.)

### Tags

- Task and backend: `task` (`forecast` or `allocation`) and `backend` (`numpy` or `torch`).
- Data capabilities: `capability:exog`, `capability:series_exog`, `capability:mixed_frequency`, `capability:missing_values`.
- Method capabilities: `capability:update`, `capability:factor_forecast`, `capability:device`.
- Behaviour: `update_kind`, `predict_data`, `non_deterministic`.

`dfm_python.registry.all_estimators()` lists the migrated models.

## 7. Save/load layout

`save(path)` writes to a sibling temporary directory, reloads it to verify (keeping DFM's write-and-verify safety), and then renames it into place. `overwrite=False` refuses an existing directory.

| File | Contents |
|---|---|
| `config.json` | `format_version` (a file-format counter, independent of releases); `model_type`; `library_version` (read at save time); `params` (`get_params()`, with estimators written as `{"class", "params"}`, and `random_state` always present); `tags`; `runtime` (device, torch and numpy versions, determinism flag, thread count) |
| `data_spec.json` | target names in user order and internal order, exog and series-exog names, normalized `freq`, index kind, freq, name, first, last and length |
| `state.npz` | every array that is not a parameter or buffer, saved with `allow_pickle=False` and namespaced keys. DFM: `ssm/A`, `ssm/C`, `ssm/Q`, `ssm/R`, `ssm/Z_0`, `ssm/V_0`, `mixed_freq/*` (constraint matrix and vector when set, idio indicator), `structure/blocks`, `structure/r`, `data/data_processed`, `data/last_smoothed_state`, `data/last_smoothed_cov` (P_T\|T from the filter call `update` starts from; as a safeguard, all NaN, with a `RuntimeWarning`, if that filter is not finite with the fitted parameters, in which case `update` raises on the fitted and the loaded model alike; `load` does not read it), and `data/smoothed_state`, the engine's smoothed state including the rows `update` appended (an addition to this list, as DDFM's `data/frame`: `update` extends it and forecasts from its last row, so without it a reloaded model that was updated could not forecast or update like the saved one); the engine config, the block order (JSON sorts keys, and the block order is the factor order), the training diagnostics and the conditioning index are in `state.json`, and `data_spec.json` adds `internal_feature_names_` and `freq_`. DDFM: `factors` (MC x T x r), `eps`, `last_neurons`, `ssm/F`, `ssm/H`, `ssm/Q`, `ssm/R`, `ssm/mu_0`, `ssm/Sigma_0`, and `data/frame`, the scaled conditioning frame `[exog | y]` with its NaN, the training rows followed by the rows `update` appended (an addition to this list: `update` interpolates the new rows together with that frame, so without it a reloaded model could not update like the fitted one; its column names, `target_series` and `covariates` are in `state.json`), and the fitted covariate scaler (`covariate_scaler/<attr>` arrays, its JSON part under `covariate_scaler` in `state.json`; a scaler class outside the five known ones is pickled into `covariate_scaler/pickle` and read only with `allow_pickle=True`, as its `scaler.pkl` already requires). The DDFM arrays keep their memory order (the transition matrix is column-major after fit, and a row-major copy rounds the forecast differently). iVDFM: `factors`, `innovations`, regime weights. AFM: `residuals`, `book_returns`. All models: `scaler/<attr>` arrays |
| `state.json` | non-array `*ModelState` fields: `r`, `p`, block names, `mixed_freq`, tent weights, frequencies, `num_iter`, `converged`, `loglik` or `sharpe` |
| `result.npz`, `result.json` | arrays and scalars of `result_`, so a loaded model reports exactly what the fitted one did. `model_state_dict` and `fit_params` are rebuilt from the weights and state files, not duplicated |
| `weights.pt` | torch models only: `network_.state_dict()`, loaded with `torch.load(path, map_location=device, weights_only=True)` |
| `scaler.json` | class name and scalar parameters for the Standard, Robust, MinMax, MaxAbs and Quantile scalers |
| `scaler.pkl` | only for other scalers; `load` requires `allow_pickle=True` |

Numpy and torch state sit side by side. Non-parameter arrays go to `state.npz` for all four models, and torch tensors go only to `weights.pt`. DFM has no `weights.pt`, and `load` checks the `backend` tag against the files present.

`Model.load(path, device=None)` works in four steps:

1. Check `model_type` in `config.json`.
2. Rebuild the engine through `_from_saved(spec, state)`, which bypasses `__init__`. This removes DFM's required `dataset=` and DDFM's roughly 50-line minimal-dataset rebuild.
3. Load the weights with `map_location`, so a GPU-saved run loads on a CPU-only machine.
4. Restore `scaler_` and `result_`.

`dfm_python.load(path)` dispatches on `model_type`. The default path never unpickles. Only directories in this layout are read. A single-file `.pkl` or `.pt` checkpoint from an earlier release raises `ConfigurationError`, whose message says to load it with the release that wrote it.

## 8. Device, seed, logging, verbosity

**Device.** The torch models take `device: str | torch.device = "auto"`, meaning CUDA if available and otherwise CPU. That is today's default for all three, so default runs do not move. It is resolved in `fit` to `device_`. `load(device=...)` overrides it. Today DDFM and AFM cannot be overridden at all. `PanelData` and the adapters have no device. DFM has no `device` parameter, because an argument that does nothing would surprise users.

**Seed.**

| Model | Parameter | Default | What is seeded (unchanged) |
|---|---|---|---|
| DFM | none | n/a | nothing; EM is deterministic |
| DDFM | `random_state` | 3 | local `np.random.RandomState` in fit; `torch.manual_seed` in the encoder and decoder builders and `layer/mlp.py` |
| iVDFM | `random_state` | 3 | `torch.manual_seed` then `np.random.seed` in the engine `__init__`, just before building components |
| AFM | `random_state` | None | `torch.manual_seed` in the engine `__init__` |

The global seeding call sites stay where they are and call one helper, `utils.seed.apply_seed(seed, *, torch_global, numpy_global)`, which does exactly what each site does today. Moving any of them would change RNG consumption order. DDFM's local `np.random.RandomState` is not global seeding and stays as it is. `fit` constructs the engine (which seeds, then builds) and then trains, which is the order of today's dataset-in-constructor path, and the golden fixtures are captured in that order. A grep test forbids global seeding outside the helper. `random_state` is always saved in `config.json`.

**Logging and verbosity.**

- The package logger gets only a `NullHandler`, and nothing is attached at import.
- `dfm_python.set_verbosity("debug" | "info" | "warning" | "error")` and `get_verbosity()` are the opt-in. `DFM_PYTHON_VERBOSITY` is honoured.
- Every model takes `verbose: int = 0`: 0 is silent, 1 shows progress and 2 is debug. The progress handler is attached only during `fit`.
- DDFM's `disp` interval maps to `verbose=1`. iVDFM's hard-coded `verbose=True` becomes the parameter.
- The five logger modules collapse into `logger.py` and `progress.py`. Done: `logger.py` keeps `get_logger`, `setup_logging`, `configure_logging`, `set_verbosity`, `get_verbosity` and the environment verbosity; `progress.py` holds `fit_verbosity`, `quiet_engine_loggers` and the engines' progress reporting (`log_em_iteration`, `log_convergence`, `iVDFMTrainLogger`) with unchanged messages and levels. `log_blocks_diagnostics` moved to `numeric/builder.py`. The unused training and inference logger classes and helpers were removed.

None of this changes numbers.

## 9. Error handling conventions

| Exception | Bases | Raised when |
|---|---|---|
| `DFMError` | `Exception` | root of the library's errors (name kept) |
| `ConfigurationError`, `ConfigValidationError` | `DFMError`, `ValueError` | invalid hyperparameters; unknown config keys; flat YAML; single-file checkpoints |
| `DataValidationError` | `DFMError`, `ValueError` | malformed `PanelData`, misaligned index, overlapping `new_rows` data |
| `UnsupportedInputError` | `DataValidationError` | a slot, frequency layout or `refit` that the tags exclude |
| `NotFittedError` | `DFMError`, `sklearn.exceptions.NotFittedError` | `predict`, `allocate`, `update`, `save` or `result_` before `fit` |
| `ModelNotTrainedError` | alias of `NotFittedError` | raised inside the engines; the alias means their raise sites need no edit |
| `NumericalError`, `NumericalStabilityError` | `DFMError`, `ArithmeticError` | unchanged engine sites |

Warnings:

- `UserWarning` when a fitted scaler's statistics are discarded, or when a run is saved with `random_state=None`.

Messages name the argument and the fix, such as the closest canonical key or the excluding tag. No input is silently ignored. Today's silent cases, DDFM `predict(data=..., update=False)` and iVDFM `fit(data=...)`, now either work or raise.

## 10. Current to target mapping

| Model | Aspect | Current | Target |
|---|---|---|---|
| DFM | construct | `DFM(dataset, config=None, scaler=None, **kwargs)`; kwargs mutate the caller's config; `DFMDataset(config, data, time_index)` | `DFM(*, n_factors, blocks=None, clock, tent_weights, max_iter, tol, ..., scaler=None, verbose=0)`; `PanelData(y, freq=...)` |
| DFM | fit | `fit(X=None, checkpoint_callback=None) -> DFMModelState`; wrong column order raises | `fit(data, *, checkpoint_callback=None) -> self`; stable reorder |
| DFM | predict | `predict(horizon=None, *, data=None) -> (X, Z)` | `predict(h, data=None)` and `predict_factors(h, data=None)`, both DataFrames |
| DFM | update | `update(data, rescale=False, retrain=False) -> None` | `update(data, *, refit=False) -> self` |
| DFM | results | `training_state`, `_result`, `get_result()`, `result` | `result_`, `n_iter_`, `converged_`, `factors_` |
| DFM | save/load | pickle file; `load(path, config=None, dataset=<required>)`; `load_config` broken | directory; `DFM.load(path)`; `from_config` |
| DDFM | construct | `DDFM(dataset, config, encoder_size, ..., seed=3)` (18 keywords), config ignored; `DDFMDataset(data, time_idx, covariates, scaler)` | `DDFM(*, n_factors, encoder_layers, ..., random_state=3, device="auto", verbose=0)`; covariates in `exog` |
| DDFM | fit | `fit() -> None`, then a required `build_state_space()` | `fit(data) -> self` |
| DDFM | predict | `predict(horizon=None, *, data, update=True, return_series, return_factors)`, ndarray or tuple, mutates | `predict(h, data=None) -> DataFrame` (pure); `predict_factors(h)` |
| DDFM | update / save | `update(DDFMDataset) -> None`; `torch.save` dict, load rebuilds a minimal dataset; no device override | `update(data) -> self`; directory with `weights.pt`; `device=` |
| iVDFM | construct | `iVDFM(dataset=None, data_dim, num_factors, context_dim, window, config, ..., seed=3, **kwargs)` (25 keywords); defaults clobber config; seeds in `__init__`; `set_dataset()` | `iVDFM(*, n_factors, window, stride=None, ..., random_state=3, device="auto", verbose=0)`; pure `__init__`; context in `exog` |
| iVDFM | fit | `fit(*args, **kwargs) -> self` on the `set_dataset` data; `fit(data=)` ignored | `fit(data) -> self` |
| iVDFM | predict | `predict(data=None, context_data=None, context=None, horizon=1, deterministic=True, num_samples=1) -> ndarray`, model space | `predict(h, data=None, *, futr_exog=None, deterministic=True, n_samples=1) -> DataFrame`, data units |
| iVDFM | update / tune | `update(data_or_dataset, *, append, store_full_history, verbose)`; `hyperparameter_optimization(...)` | `update(data) -> self`; `tuning.tune_hyperparameters(estimator, data, ...)` (same optuna code) |
| iVDFM | save/load | `torch.save` checkpoint or weights-only file with a partial config; `load(path, *args, **kwargs)` | directory with the full config; `iVDFM.load(path, device=)` |
| AFM | construct | `AFM(dataset=None, config=None, *, n_char, n_factors, ..., seed)`; `pick()`; `set_dataset()`; `AFMDataset(returns, characteristics, window, stride, scaler)` | `AFM(*, n_factors, factor_model, trading, hist_len, ..., random_state=None, device="auto", verbose=0)`; `PanelData(y, series_exog)` |
| AFM | fit / predict | `fit(X=None) -> AFMModelState`; `predict(horizon=None, *, data=None) -> dict` with batch dimension 1 | `fit(data) -> self`; `allocate(data=None) -> DataFrame`; `backtest(data) -> AFMResult` |
| AFM | update / save | `update(data, retrain=False)`; `torch.save` dict, `load` without `map_location` | `update(data, *, refit=False) -> self`; directory; `AFM.load(path, device=)` |

## 11. Contract and golden tests

The contract tests live in `src/test/contract/`, parametrized over `registry.all_estimators()` with one tiny CPU fixture per tag combination. `dfm_python.testing.check_estimator(est)` exposes the same checks for future models. Every model must pass the following:

1. **Pure construction.** The global torch and numpy RNG states are unchanged, no CUDA memory is allocated, and `vars(est)` equals `get_params()`.
2. **Parameter machinery.** `get_params`/`set_params` and `clone` round-trip. Config fields equal the constructor parameters. `from_config(get_config())` gives an equal estimator, YAML and JSON round-trips are lossless, and unknown keys raise `ConfigurationError`.
3. **Unfitted access.** `predict`, `allocate`, `update`, `save` and `result_` raise `NotFittedError`, and `hasattr(model, "result_")` is false.
4. **fit returns self and is repeatable.** Two fits on CPU give bitwise-identical `result_` arrays.
5. **fit adds only underscore attributes**, and mutates neither the input data nor any hyperparameter.
6. **Output shape.** Forecasters return `h` rows starting one step after the conditioning data, columns equal to `feature_names_in_` in user order, and finite float64 values. Allocators return index `data.index[hist_len:]` with asset columns, and their `predict` raises `TypeError`.
7. **Purity of predict and allocate.** The fitted-state hash is unchanged after repeated calls, including with `data=`.
8. **Save/load round-trip.** The file set matches section 7 for the backend. `allow_pickle=False` and `weights_only=True` loads succeed, and no pickle appears for default scalers. Reloaded forecasts and `result_` are bitwise equal, `dfm_python.load` dispatches correctly, and a CUDA-saved model loads on CPU (skipped without CUDA).
9. **Capability enforcement.** Each slot, layout or `refit` that the tags exclude raises `UnsupportedInputError`.
10. **Device** (tag-gated). `device="cpu"` puts every parameter on CPU.
11. **Seeds.** The same `random_state` reproduces results bitwise on CPU, `random_state` is recorded in `config.json`, and the grep test finds no global seeding outside `apply_seed`.
12. **Silence.** Importing the package adds no handlers. With `verbose=0`, `fit` prints nothing and logs nothing at INFO.
13. **update.** It returns `self`, `predict` afterwards conditions on the new rows, and parameters are unchanged when `refit=False`.
14. **Single source of truth.** The constructor, `from_dict`, `from_config` and `from_yaml` give equal `get_params()`, and every hyperparameter reaches the engine unchanged. For example, an iVDFM YAML with `factor_order: 3` gives an engine with `factor_order == 3`, and a DDFM YAML learning rate reaches the optimizer.

**Golden suite** (`src/test/parity/`). Golden fixtures are captured from the current code before any engine code moves. Each fixture stores the data it was fitted on (or the recipe that regenerates it), the hyperparameter values the constructed object actually held, and the engine outputs. The cases cover each tutorial configuration on truncated shipped data, a synthetic panel, DFM mixed frequency, DFM on permuted columns, iVDFM with several regimes, and AFM in attention and PCA modes. Until a model is migrated, its golden test runs today's API. When it is migrated, the test switches to the estimator with the same hyperparameters, through both the constructor and a YAML, and adds save-then-load. The purpose is one claim: same data plus same hyperparameters gives the same numbers from the engine.

Comparison runs on CPU with pinned threads (`threadpoolctl` limit 1, `torch.set_num_threads(1)`, deterministic algorithms). Each fixture records a platform fingerprint (machine, processor, numpy and torch versions, BLAS build). On a matching platform the comparison is `assert_array_equal`. On any other platform it falls back to `assert_allclose` with a tolerance stated per output (float64 EM outputs tighter than float32 network outputs) and says so in the test output. Before an engine moves, the engineer also checks bitwise equality in one process on one machine, old path against new path.

## 12. Release plan

The next minor release ships the new API for all four models. The work lands model by model on the main branch, so for a while migrated and unmigrated models sit side by side, but nothing is released until all four are migrated.

- **Removed, not deprecated:** the old constructors (`dataset=`, `config=`, positional datasets); the public `*Dataset` classes (their code lives on inside the adapters); `set_dataset`, `build_state_space`, `hyperparameter_optimization`, `get_result`, `result` and `training_state`; DDFM's `return_*` and `predict(update=...)` flags; DFM's `rescale` and iVDFM's `append` and `store_full_history` flags; `ConfigSource`, `YamlSource` and `make_config_source`; flat YAML files; and single-file checkpoints.
- **Rewritten:** the README, every tutorial and every shipped YAML, on the new API and the new YAML layout.
- **CHANGELOG:** lists the breaking changes, the old-to-new call table from section 10, and the shipped YAML values that now take effect (section 5). Projects that need the old behaviour pin an earlier release.

## 13. Implementation plan

Each item is one reviewable change, ends with the suite and `ruff check` green, and touches at most one model. Golden capture comes before any production change. Shared infrastructure comes next, beside the old code, and AFM is the first model, because it is the smallest and has the simplest state. Until their own item, DFM, DDFM and iVDFM keep their current public classes and tests unchanged.

### This phase: foundation and AFM

1. **Golden capture.** Add `src/test/parity/` with `_golden.py` (thread pinning, fingerprint, `assert_golden`), a capture script `make_golden.py`, compressed `.npz` fixtures with a JSON manifest for all four models, and one test module per model that runs today's API and compares with the fixtures. AFM cases use the constructor-dataset order. No production change.
2. **Errors and logging.** Rebase the exceptions as in section 9, add `NotFittedError` and `UnsupportedInputError`, and make `ModelNotTrainedError` an alias of `NotFittedError`. Give the package logger only a `NullHandler`, stop attaching handlers in `get_logger`, add `set_verbosity`, `get_verbosity`, `DFM_PYTHON_VERBOSITY` and a `fit`-scoped progress context, and add the silence test. The logger modules are collapsed later.
3. **Seed helper.** Add `utils/seed.py` with `apply_seed`, route every global seeding site through it in place, add the grep test, and confirm the golden suite is unchanged.
4. **PanelData.** Add the container in `dataset/panel.py` with validation, the `from_*` constructors, `to_long`, `loc`, `freq` normalization, polars input and the internal coercion that `fit` uses, with unit tests.
5. **Estimator base, config mirror, tags, registry and contract harness.** Add the new `BaseFactorModel` in `dfm_python/base.py`, the config-mirror base with strict keys and the YAML layout of section 5, tag defaults and `get_tags`, `dfm_python.registry`, and `dfm_python.testing.check_estimator` with the checks of section 11, exercised on a dummy estimator in `src/test/contract/`.
6. **Save/load primitives.** Add `dfm_python/io/` with the atomic directory writer, the JSON and npz codecs, the known-scaler codec and the gated pickle fallback, the base class's `save` and `load` template, and `dfm_python.load`.
7. **AFM engine and estimator.** Move today's class verbatim to `models/afm/_engine.py`, adding only a `device` keyword and `_from_saved`. Write the `AFM` estimator (`fit`, `allocate`, `backtest`, `update`, a raising `predict`, tags) and the `AFMConfig` mirror. Switch the AFM golden test to the estimator.
8. **AFM save/load, contract registration and docs.** Implement AFM's state hooks for the section 7 layout, register AFM, make the contract suite pass for it, rewrite the AFM tests, the README's AFM section and `config/afm_finance.yaml`, and update the top-level exports.

### Later phases

9. **iVDFM.** Engine move with `_from_saved`, estimator with explicit parameters, `stride=None`, forecasts inverse-transformed with `scaler_`, `update`, save and load, config mirror, contract registration and the golden switch.
10. **iVDFM tuning and examples.** Extract `tuning.tune_hyperparameters` around today's optuna code. Rewrite the iVDFM YAMLs and tutorials.
11. **DDFM.** Engine move, the `n_factors`/`encoder_layers` split, the `[exog | y]` adapter, `fit` including the state space, deep-copy `predict(data=)`, `predict_factors`, `update`, `_from_saved` replacing the minimal-dataset rebuild, the config mirror without the dead fields, contract registration and the golden switch.
12. **DFM.** Engine move, the adapter that builds `DFMConfig` and `DFMDataset` with the stable partition and output in user order, `_from_saved` with `data_processed` and the last smoothed state, `new_rows` validation and `update`. Remove `load_config`. Add the permuted-input golden case, contract registration and the golden switch.
    Done. The engine is `models/dfm/_engine.py` (`DFMEngine`) and the estimator `models/dfm/estimator.py`. The old engine config dataclass is the private `_DFMEngineConfig`, and `DFMConfig` is the constructor mirror. The golden fixtures were recaptured before the move with more outputs and an update path; every previously pinned output stayed bitwise identical, and all five DFM cases pass bitwise through the engine, the constructor, the YAML and save-then-load. `update()`'s broken validator import was fixed in the move.
13. **DFM and DDFM examples.** Rewrite the YAMLs and tutorials on the new API.
    Done. Every shipped YAML uses the `model:`/`data:` layout, and the nine tutorials run on the new API without Hydra (`--rows`, `--max-iter`/`--max-epochs`, `--run-dir`, `--device`).
14. **Cleanup.** Remove the public `*Dataset` exports, `ConfigSource`, `YamlSource` and `make_config_source`; rename the old engine base class; collapse the logger modules; make the tutorials a CI smoke job on truncated data.
    Done. The old engine base class was deleted rather than renamed: once no engine inherited it, nothing used it (`models/base.py` keeps `BaseEncoder` and `BaseDecoder` for the DDFM and iVDFM networks), and each engine sets its attributes in `_init_engine_state`. `DictSource`, `detect_config_type` (still used privately by the DFM engine config) and `BaseModelConfig` are no longer exported either. The unused `examples` extra (hydra-core) was removed. The CI `tutorials` job runs the nine scripts with 120 rows (400 for the weekly-clock macro DFM) and one iteration or epoch.
15. **Documentation.** Rewrite the README and write the CHANGELOG entry described in section 12.
    Done. The README documents the shared contract, `PanelData`, the four models, `update` and `predict(data=)`, saving, YAML, reproducibility, logging, errors and migration; `src/test/test_readme.py` runs its Python examples. The CHANGELOG's "Breaking: estimator API for all four models" entry holds the section 10 table for every model, the removals, the renames and the shipped YAML values that now take effect.

## 14. Risks

- **Parity hazards.** The known ones are:
  - DDFM's `[exog | y]` order and its `type(scaler)()` covariate scaler
  - iVDFM's in-dataset imputation
  - the seed-then-build order
  - eager `get_result()`, which must not consume RNG ahead of a stochastic `predict`
  - the deep copy behind DDFM's pure `predict(data=)`

  Each has a golden case. Because the engines are today's classes called with explicit arguments, the hazards sit in small adapters.
- **CPU-only bitwise parity.** It holds only with pinned threads, and only on the platform that captured the fixtures; elsewhere the golden suite compares to a tolerance. A GPU run can be matched only to a tolerance.
- **Shipped YAMLs change behaviour.** Fixing precedence means the iVDFM and DDFM YAML values that used to be overridden now take effect, so running a shipped YAML can give different numbers than the same file under an earlier release. The golden suite pins engine numbers for given hyperparameters, not the old YAML-to-number mapping. The CHANGELOG lists the affected fields.
- **Rename churn.** `tol`, `random_state`, `n_factors` and `encoder_layers` touch every migrated script. Old keyword names raise `TypeError`, and old YAML keys raise `ConfigurationError` naming the canonical key.
- **Two-column unpacking.** Code that still does `X, Z = model.predict(h)` on a model with exactly two targets silently gets the column labels. The CHANGELOG calls this out. Any other column count raises.
- **Eager `result_`** may lengthen `fit` on large DFM panels. If so, it can become lazy without an API change.
- **Known DDFM engine bugs, left in place.** The migration preserves the engine's numbers, so these stay until a separate fix (each changes golden outputs and needs a recapture):
  - the `DDFMResult` fields `use_idiosyncratic` and `fit_params` are never set (the `training_*` fields, `training_loss` and `encoder_layers` are set since 0.8.2).

- **DDFM update bugs, fixed before the 0.8.0 release.** `update(data)` and `predict(h, data=)` now have one semantics (tags `update_kind="encoder_forward"`, `predict_data="new_rows"`): `data` holds only the rows after the last conditioning timestamp (overlap raises `DataValidationError`, as for DFM); the scalers fitted in `fit` (`scaler_` for the targets, the kept `type(scaler)()` instance for the covariates) scale them and are never refitted; missing values are interpolated together with the conditioning rows with the fit's interpolation settings (`interpolate_dataframe`, whose forward/backward fill closes the gaps it leaves); only the new rows' factors are appended. `predict(h, data=new)` equals `update(new).predict(h)` and leaves the model unchanged. The DDFM fit outputs and `predict(h)` are bitwise unchanged; the update outputs of the golden fixtures were recaptured (`ddfm_exchange`, `ddfm_finance`: the update now takes the 12 held-out rows; `ddfm_macro` has no rows after its fitted ones and lost its update outputs). Fixed:
  - `_update` overwrote the interpolated targets with the un-interpolated `data_original`, so missing target values reached the encoder as NaN, the factors of those rows were NaN and the forecast started from a zero state;
  - the update panel's covariates were scaled with a fresh `type(scaler)()` fitted on that panel, not with the training statistics;
  - the target scaler was refitted on the training rows plus the full update panel, which repeated the training rows;
  - a second `update` inverse-transformed the training targets with the scaler the first one refitted;
  - refitting the target scaler inverse-transformed the training targets only when the scaler had `mean_` and `scale_` (`StandardScaler`); with `"minmax"`, `"robust"`, `"maxabs"`, `"quantile"` or another transformer it stacked the still-scaled training targets with the raw update targets and refitted on that mixed-unit matrix (forecasts such as 11.2 and -14.0 for series between 46 and 64 with `"minmax"`);
  - the encoded factors of the whole update panel were appended after the training factors, so rows that overlapped the training data were appended twice.

- **iVDFM AR initialization with more factors than PCA factors, fixed before the 0.8.0 release.** The engine initializes the AR coefficients by OLS on the PCA factors of the last training window and zero-padded them to `n_factors` when there were fewer (more factors than targets, or than window rows). With `factor_order > 1` and one regime the padded factors got all-zero coefficient rows, which `iVDFMCompanionSSM.norm` L1-normalized by zero, so the loss was NaN in the first batch and `fit` raised `ConfigurationError` for `n_factors` above the number of targets with `n_regimes == 1`. With one regime, the factors without a PCA factor now keep the SSM's non-data random initialization, `norm` leaves a zero-norm row unnormalized, and the `ConfigurationError` is gone; `factor_order == 1` and `n_regimes > 1` (which trained finitely) keep their zero-padded OLS values and their numbers, bitwise. The shipped `ivdfm_macro` and `ivdfm_finance` YAMLs are back to `n_factors: 5`. Every iVDFM golden fixture is bitwise unchanged.

- **iVDFM tuning issues, fixed before the 0.8.0 release.** `tuning.tune_hyperparameters` (signature unchanged) now seeds optuna's default sampler, `TPESampler(seed=random_state)`, from the trials' `random_state` (`fixed["random_state"]`, else `estimator.random_state`), so the same seed reproduces the trials and the best parameters; `random_state=None` keeps the unseeded sampler. A study resumed from storage (`load_if_exists=True`) with `n` trials is sampled with seed `random_state + n`, so it draws new trials instead of replaying the first run's. The `window` and `stride` upper bounds (step 10) are aligned down to the step before `suggest_int`, the bound optuna already fell back to, so the search space is unchanged and optuna no longer warns that the range is not divisible by the step. No golden fixture covers tuning. Fixed:
  - two searches with the same `random_state` drew different trials (unseeded TPE sampler);
  - with the seeded sampler, resuming a stored study replayed the first run's trials (the seed is now offset by the number of stored trials);
  - optuna warned "the range is not divisible by `step`" whenever `max_window - min_window`, or `1000` minus the lowest stride, was not a multiple of 10.

- **DFM engine issues, fixed after the migration.** The migration kept the DFM engine frozen for identical numbers. The numerical-stability rewrite that followed (CHANGELOG, Unreleased) fixed its known issues and changed the DFM numbers on purpose, with the DFM golden fixtures recaptured and checked against an independent reference Kalman filter; the DDFM, iVDFM and AFM fixtures are bitwise unchanged. Fixed:
  - mixed-frequency EM reported the log-likelihood as exactly `-T/2` (`-200` for 400 rows, `-60` for 120); `result_.loglik` is now the exact missing-data log-likelihood of the returned parameters;
  - EM diverged, or left parameters under which the Kalman filter over the training data was not finite, on some mixed-frequency panels (truncated slices of `data/macro.csv`); the EM log-likelihood is now monotone, and EM returns the best parameters with `converged=False` if it ever falls;
  - one or two EM iterations on short weekly/monthly slices gave explosive forecasts;
  - `predict(h, data=)` filtered the new rows from `V_0` while `update` started from the last filtered covariance; both now continue the training filter and agree.

  Still true by design: `update(data, refit=True)` (the old `retrain=True`) runs one EM iteration on all rows, not a full refit. The transition matrix `A` is not capped (the old 0.99 eigenvalue cap biased the likelihood), so on non-stationary inputs the maximum-likelihood `A` can be explosive and long-horizon forecasts grow geometrically; `DFM.fit` warns (`UserWarning`) when the spectral radius of the fitted `A` is above 1, and the README tells users to difference or transform series to stationarity first. The `max_iter` default stays 5000: the golden DFM panels need up to 566 EM iterations to converge at `tol=1e-5`.

## 15. Open questions for the owner

1. **Naming.** Accept the sklearn names (`n_factors`, `n_regimes`, `random_state`, `tol`), or keep `num_factors`, `seed` and `tolerance`? Is `allocate` the right verb for AFM?
2. **Seed defaults.** Keep 3, 3 and None permanently, or unify to None and accept that default runs give different numbers?
3. **DFM tent-weight default.** Decided: no default kernel. Today's path raises `DataValidationError` for mixed-frequency data without `tent_weights` (with the symmetric kernel the same data fits), so `tent_weights` is required for the slower frequency. `fit` raises `ConfigurationError` whose message gives the symmetric `generate_tent_weights` kernel and the exact `tent_weights={...}` argument (section 5).
4. **DFM default scaler.** Decided: keep `None`, the value that ran when the argument was omitted; `scaler="standard"` standardizes. The README's claim of internal standardization is corrected.
5. **Mixed-frequency output.** Decided: keep reporting slower series at every clock date, as the engine computes them (the tent-weighted combination of the state forecast); documented in `DFM.predict` and the README.
6. **`DDFMResult.C` orientation.** Decided: `result_.C` is the loading matrix H, N x r, the DFM orientation (it was H transposed, factors x N). The change is one line in the engine's `get_result`, and no code read the old orientation.
7. **sklearn coupling.** Rely on sklearn's `BaseEstimator` (and its private `_get_param_names`), or vendor about 60 lines?
