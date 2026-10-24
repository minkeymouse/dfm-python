# Models

One section per model: what it does, a full example, the main
hyperparameters with their defaults, notes and the method reference. The
complete parameter list of each model is in its docstring (`help(DFM)`). The
shared API is in [guide.md](https://github.com/minkeymouse/dfm-python/blob/main/docs/guide.md).

## DFM

The linear dynamic factor model, estimated by EM with Kalman filtering and
smoothing. The series are `PanelData.y`, in any column order, and their
frequencies `PanelData.freq`. The latent factors evolve at the `clock`
frequency. Every series is sampled at the clock or at one slower frequency:
a series faster than the clock, or a second slower frequency, raises
`UnsupportedInputError`. A slower series is NaN at the clock dates between
its observations and loads on the factors through a tent kernel, given in
`tent_weights` under the key `"f:clock"`: its loadings on the factor lags are
proportional to the kernel's weights, so any positive multiple of a kernel
gives the same model. There is no default kernel: mixed-frequency data
without one raises `ConfigurationError` naming the symmetric kernel to pass.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2010-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((120, 4)),
                  index=idx, columns=["gdp", "s0", "s1", "s2"])
df.loc[df.index.month % 3 != 0, "gdp"] = np.nan        # quarterly series, any column position

data = PanelData(y=df.iloc[:-6], freq={"gdp": "q"})     # or freq={"q": ["gdp"]}
new = PanelData(y=df.iloc[-6:-3], freq={"gdp": "q"})   # rows after the last fitted date
more = PanelData(y=df.iloc[-3:], freq={"gdp": "q"})    # rows after those
model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
            max_iter=50, tol=1e-5, scaler="standard")   # blocks=None: one global block
model.fit(data)                                         # initialization, then EM (SQUAREM)
model.result_.loglik, model.n_iter_, model.converged_   # n_iter_: E-steps run
model.factors_                                          # DataFrame 114 x 1 (factor_0); states_ is 114 x 13
model.loadings_, model.explained_variance_              # 4 x 1 (scaled space); Series per series
fc = model.predict(6)                                   # DataFrame 6 x 4, data units, your column order
zf = model.predict_factors(6)                           # DataFrame 6 x 1 (factor_0)
bands = model.predict_interval(6, coverage=0.9)         # columns (series, lower | upper), data units
parts = model.decompose(by="factor")                    # columns (factor_0 | mean | idiosyncratic, series)
fc_new = model.predict(6, data=new)                     # filters the new rows on a copy
model.update(new)                                       # Kalman filter and smoother, fixed parameters
model.update(more, refit=True)                          # one EM iteration on all rows; refreshes result_, factors_, ...
model.save("runs/dfm", overwrite=True)
model = DFM.load("runs/dfm")                            # no data needed
```

| Hyperparameter   | Default | Meaning |
|------------------|---------|---------|
| `n_factors`      | None    | size of the single global block used when `blocks` is None (None means 1) |
| `blocks`         | None    | `{"name": {"n_factors": 2, "series": [...]}, ...}`; replaces `n_factors` (set one of the two); the block order is the factor order |
| `clock`          | `"m"`   | frequency of the factors: `"d"`, `"w"`, `"m"`, `"q"`, `"sa"` or `"a"` |
| `tent_weights`   | None    | tent kernel of the slower frequency, for example `{"q:m": [1, 2, 3, 2, 1]}` or `{"m:w": [1, 2, 1]}`; required with mixed frequencies |
| `max_iter`, `tol` | 5000, 1e-5 | the most E-steps EM runs, and the convergence threshold on the relative change of the log-likelihood between accepted parameters |
| `em_acceleration` | `"squarem"` | `"squarem"` (EM accelerated by SQUAREM) or None (plain EM) |
| `scaler`         | None    | `"standard"`, `"robust"`, `"minmax"`, `"maxabs"`, `"quantile"` or an unfitted transformer |
| `ar_clip`, `data_clip`, `damping_factor` | None | opt-in stabilizers, off by default; `ar_clip` and `damping_factor` give up the monotone log-likelihood of EM |
| `regularization` | None    | only sets the smallest loading norm that the final factor normalization rescales |

Notes:

- Scaling: `scaler=None` fits the data as given; the engine then rejects
  series whose standard deviation is above 100 or below 1e-4. Pass
  `scaler="standard"` (or another scaler) unless the data is already
  standardized.
- Column order: the engine orders the series `[clock | slower]` internally,
  a stable partition of your order (`internal_feature_names_`). Forecasts,
  intervals, `decompose`, `loadings_`, `explained_variance_` and `freq_`
  follow your order; the series-indexed matrices of `result_` (`C`, `R`,
  `x_sm`) are in the internal order.
- Mixed-frequency forecasts report every series at every clock date: a
  quarterly series on a monthly clock gets a value for each forecast month
  (the tent-weighted combination of the factor forecasts).
- Estimation: the E-step is an exact Kalman filter and Durbin-Koopman
  smoother that uses every observed entry (missing values are handled per
  element), the M-step is the closed-form Banbura-Modugno update, and
  `result_.loglik` is the exact Gaussian log-likelihood of the observed data
  at the returned parameters. The log-likelihood of the accepted parameters
  (every iterate of plain EM) does not decrease; EM stops when its relative
  change between accepted parameters is below `tol` (`converged_=True`) or
  after `max_iter` E-steps (`converged_=False`). If, after the first E-step,
  the log-likelihood falls or a parameter or the filter becomes non-finite,
  EM stops early and returns the best accepted parameters, with
  `converged_=False` (in the first E-step it raises).
- Acceleration: `em_acceleration="squarem"` runs the SqS3 scheme of Varadhan
  and Roland (2008) in cycles of three E-steps: from an accepted point, two
  EM maps, a quadratic extrapolation along them (the step length is clipped
  between 1 and a bound that starts at 1 and adapts, so the first cycle is
  plain EM), and one EM map from the extrapolated point. That map's output is the next accepted point when the
  extrapolated point's log-likelihood is not below the cycle's start;
  otherwise EM keeps the second EM map's output. `n_iter_` counts every
  E-step and `max_iter` bounds them; `ar_clip` and `damping_factor` apply to
  every EM map. At the same `tol`, SQUAREM usually stops in fewer E-steps at
  a log-likelihood at least as high as plain EM's (the README quick start:
  about 110 E-steps against 222). `em_acceleration=None` runs plain EM.
- Stationarity: the transition matrix `A` is not capped. On non-stationary
  inputs (trending levels, unit roots) the maximum-likelihood `A` can be
  explosive, and long-horizon forecasts then grow geometrically. Difference
  or otherwise transform the series to stationarity before fitting. `fit`
  warns (`UserWarning`) when the spectral radius of `result_.A` is above 1.
- `update(data, refit=True)` runs one EM iteration (one EM map, with either
  `em_acceleration`) on all rows, starting from the current parameters. To
  re-estimate from scratch, call `fit` on the whole panel.
- Factors: `factors_` is `E[f_t | Y]`, the smoothed current value of each
  factor (the factor states without their lags), on `index_`. Its columns
  `factor_names_` are `factor_<j>` for one block and `<block name>_<j>`, in
  block order, for several. `states_` is the whole smoothed state `result_.Z`
  (`state_0` ... `state_{m-1}`: the factors, their lags and the
  idiosyncratic states). `predict_factors` forecasts the columns of
  `factors_`.
- Loadings (`loadings_kind_="exact"`, scaled space): for a clock-frequency
  series, its entry of `C` on the factor; for a slower series, its loading on
  the tent-aggregated factor `sum_l w_l f_{t-l}`. The tent restriction makes
  the entry of `C` on lag `l` that loading times `w_l`, so `loadings_` holds
  the lag-0 entry divided by the first tent weight `w_0`.
  `explained_variance_` compares the observed scaled entries (for a slower
  series, its release dates) with the factor states (current values and
  lags) through `C`.
- `decompose()`: `"common"` is the factor states through `C`,
  inverse-scaled, so it includes the series mean; `"idiosyncratic"` (the
  idiosyncratic states and the observation noise) is the rest.
  `decompose(by="factor")` gives each factor's states (current value and
  lags) through `C`, times the scaler's slope, so a slower series' part is
  summed over the tent lags; `"mean"` is the scaler's offset. It needs an
  affine scaler, as `predict_interval` does.
- `predict_interval(h, coverage, data)`: Gaussian intervals around
  `predict(h)` from the Kalman forecast moments, for every series at every
  forecast date (method in the
  [guide](https://github.com/minkeymouse/dfm-python/blob/main/docs/guide.md)).
- Cost: an EM iteration (E-step) is a Kalman filter and smoother pass, about
  `O(T m^3)` with `m` the state dimension (factor lags plus one idiosyncratic
  state per series, so `m` grows with `N`). From 56 states up the filter and
  smoother use the DFM structure of `A` and `C` (AR(1) diagonals, lag
  shifts, loadings in a few columns): products with `A` and `C` are scalings,
  gathers and small products, so a row takes three dense products in the
  filter and five in the smoother instead of about twenty; the results equal
  those of the dense recursions up to rounding. Below 56 states the dense
  recursions are faster (fewer numpy calls per row) and run instead. The
  passes run the BLAS on one thread (the matrices are small; multithreaded
  BLAS was 28x slower on a 4-core machine). A 1000 x 100 panel with two
  factors takes about 1 s per E-step on one core, half the time of the
  dense recursions; on long panels with few states the per-row Python
  overhead dominates (about 0.2 ms per row at 37 states).
- No news decomposition of data releases.

Compared with statsmodels' `DynamicFactorMQ` (statsmodels 0.15), which
implements the same EM:

|                     | statsmodels `DynamicFactorMQ`                                                 | dfm-python `DFM` |
|---------------------|-------------------------------------------------------------------------------|------------------|
| Mixed frequencies   | monthly with quarterly                                                        | any clock (`d`, `w`, `m`, `q`, `sa`, `a`) with one slower frequency |
| Aggregation weights | fixed `[1, 2, 3, 2, 1]`                                                       | the tent kernel you pass in `tent_weights` |
| Data layout         | quarterly series in `endog_quarterly`, or after the monthly ones with `k_endog_monthly` | any column order; frequencies in `PanelData.freq` |
| News decomposition  | `news()`                                                                      | not available |

Method: Banbura, M. and Modugno, M. (2014), "Maximum likelihood estimation
of factor models on datasets with arbitrary pattern of missing data",
*Journal of Applied Econometrics* 29(1), for the EM algorithm. The
quarterly-on-monthly kernel `[1, 2, 3, 2, 1]` is the aggregation of Mariano,
R. S. and Murasawa, Y. (2003), "A new coincident index of business cycles
based on monthly and quarterly series", *Journal of Applied Econometrics*
18(4). SQUAREM: Varadhan, R. and Roland, C. (2008), "Simple and globally
convergent methods for accelerating the convergence of any EM algorithm",
*Scandinavian Journal of Statistics* 35(2).

Tutorials: `tutorial/tutorial_{macro,finance,exchange}_dfm.py`.

## DDFM

The deep dynamic factor model encodes each time step with a nonlinear
encoder into `n_factors` factors and reconstructs the targets with a linear
(or MLP) decoder. The autoencoder is pre-trained, then trained with MCMC
denoising (`n_mc_samples` corrupted copies of the targets per iteration, at
most `max_iter` iterations). `fit` then builds a linear state space on the
factors, an AR(1) transition and the decoder's last linear layer, which the
forecasts use. The `exog` columns are the encoder's covariates: encoded with
the targets but not reconstructed or forecast. Missing values are
interpolated.

```python
import numpy as np
import pandas as pd
from dfm_python import DDFM, PanelData

rng = np.random.default_rng(0)
dates = pd.date_range("2010-01-31", periods=132, freq="ME", name="date")
raw = pd.DataFrame(rng.standard_normal((132, 5)), columns=["s0", "s1", "s2", "oil", "fx"])
raw.insert(0, "date", dates)

data = PanelData.from_wide(raw.iloc[:120], time_col="date", exog_cols=["oil", "fx"])
model = DDFM(n_factors=2, encoder_layers=(16,), decoder_type="linear",  # encoder 5 -> 16 -> 2
             learning_rate=0.005, n_mc_samples=10, max_iter=50, tol=5e-4,
             scaler="standard", random_state=3, device="auto")
model.fit(data)                             # trains, then builds the state space
model.result_.A, model.result_.C            # AR(1) transition (2 x 2), loadings (3 x 2: series x factors)
model.factors_                              # DataFrame 120 x 2 (factor_0, factor_1)
model.loadings_, model.loadings_kind_       # 3 x 2 (s0, s1, s2 x factors), "exact": the decoder's weight
model.explained_variance_                   # Series s0, s1, s2
parts = model.decompose(by="factor")        # factor_0, factor_1, mean, idiosyncratic: sum = data
fc = model.predict(6)                       # DataFrame 6 x 3 (s0, s1, s2), data units
zf = model.predict_factors(6)               # DataFrame 6 x 2

new = PanelData.from_wide(raw.iloc[120:], time_col="date", exog_cols=["oil", "fx"])  # the 12 new rows
fc_new = model.predict(6, data=new)         # conditions on the new rows on a copy
model.update(new)                           # encodes the new rows; the scalers are not refitted
pd.testing.assert_frame_equal(model.predict(6), fc_new)

model.save("runs/ddfm", overwrite=True)     # config.json, weights.pt, state.npz, ...
model = DDFM.load("runs/ddfm", device="cpu")
```

| Hyperparameter    | Default     | Meaning |
|-------------------|-------------|---------|
| `n_factors`       | 4           | number of latent factors (the encoder's output width) |
| `encoder_layers`  | `(16,)`     | widths of the encoder's hidden layers |
| `decoder_type`    | `"linear"`  | `"linear"` or `"mlp"` |
| `activation`      | `"relu"`    | `"relu"`, `"tanh"` or `"sigmoid"` |
| `learning_rate`, `optimizer` | 0.005, `"Adam"` | `"Adam"`, `"AdamW"` or `"SGD"` |
| `n_mc_samples`    | 10          | corrupted copies of the targets per MCMC iteration |
| `window_size`     | 100         | time steps per training batch |
| `max_iter`, `tol` | 200, 5e-4   | MCMC iterations and the stopping threshold on the relative change of the reconstruction MSE |
| `interpolation_method`, `interpolation_limit`, `interpolation_limit_direction` | `"linear"`, 10, `"both"` | how missing values are filled |
| `scaler`          | None        | scaler of the targets; the covariates get a fresh instance of its class |
| `random_state`    | 3           | seed; None does not seed |
| `device`          | `"auto"`    | CUDA if available, else CPU |

Notes:

- `data=` and `update` take only the rows after the last conditioning date
  (overlap raises `DataValidationError`), with the same series and exog
  columns. The scalers fitted in `fit` scale the new rows and are never
  refitted. Missing values in the new rows are interpolated together with
  the conditioning rows, and gaps the interpolation leaves carry the last
  value forward. Only the new rows' factors are appended.
- `update(data, refit=True)` raises `UnsupportedInputError`.
- `result_.C` is N x r (series by factors), as in DFM.
- `factors_` is `result_.Z`, the MC-averaged encoder factors on `index_`,
  columns `factor_names_` (`factor_0` ... `factor_{r-1}`).
- `loadings_` (targets x factors, scaled space; the exog columns are not
  decoded): with `decoder_type="linear"`, the decoder's weight, equal to
  `result_.C` (`loadings_kind_="exact"`); with `decoder_type="mlp"`, the
  decoder's Jacobian with respect to the factors averaged over the rows of
  `factors_` (`"mean_jacobian"`; autograd derivatives, so relu has slope 0
  at its kink), which differs from `result_.C`. `explained_variance_`
  compares the observed scaled targets with the decoder's output on
  `factors_`.
- `decompose()`: `"common"` is the decoder's output on `factors_`,
  inverse-scaled. `decompose(by="factor")` needs the linear decoder and an
  affine scaler (data = offset + s * scaled): factor `j`'s part of target `i`
  is `s_i H_ij f_jt` (`H` is `loadings_`) and `"mean"` is
  `offset_i + s_i b_i` (`b` the decoder bias). The MLP decoder or another
  scaler raises `UnsupportedInputError`.
- No `predict_interval` (tag `capability:pred_int` is False).

Tutorials: `tutorial/tutorial_{macro,finance,exchange}_ddfm.py`.

## iVDFM

The identifiable variational dynamic factor model learns nonlinear factors
with AR(`factor_order`) dynamics from sliding windows of the panel and is
trained on the ELBO. The prior of the innovations is conditioned on a context
u_t, which makes the factors identifiable (iVAE-style conditioning on the
innovations): u_t is the `exog` columns plus `time_context` generated time
features. Missing target and exog values are interpolated. Forecasts are in
data units (inverse-transformed with `scaler_`).

```python
import numpy as np
import pandas as pd
from dfm_python import iVDFM, PanelData

rng = np.random.default_rng(0)
dates = pd.date_range("2010-01-31", periods=156, freq="ME", name="date")
raw = pd.DataFrame(rng.standard_normal((156, 4)), columns=["s0", "s1", "s2", "policy_rate"])
raw.insert(0, "date", dates)

data = PanelData.from_wide(raw.iloc[:144], time_col="date", exog_cols=["policy_rate"])
model = iVDFM(n_factors=2, factor_order=2, window=48, stride=None,  # stride=None: window // 2
              time_context=1, scaler="standard", max_epochs=20,
              random_state=3, device="auto")
model.fit(data)
model.result_.training_elbo, model.n_iter_, model.converged_
model.factors_                              # DataFrame 144 x 2 on index_: the factor path result_.Z
model.window_factors_                       # the factors of every training window, (window, time) MultiIndex
model.loadings_, model.loadings_kind_       # 3 x 2 mean decoder Jacobian, "mean_jacobian"
model.explained_variance_                   # Series s0, s1, s2
common = model.decompose()["common"]        # DataFrame 144 x 3: the decoded path, data units

future = raw.set_index("date")[["policy_rate"]]                # future values of the exog
fc = model.predict(6, futr_exog=future.iloc[144:150])          # DataFrame 6 x 3, data units
zf = model.predict_factors(6, futr_exog=future.iloc[144:150])  # DataFrame 6 x 2

new = PanelData.from_wide(raw.iloc[96:150], time_col="date", exog_cols=["policy_rate"])
fc_new = model.predict(6, new, futr_exog=future.iloc[150:156])  # encodes new on a copy
model.update(new)                           # encoder forward pass; factors_ is unchanged
pd.testing.assert_frame_equal(model.predict(6, futr_exog=future.iloc[150:156]), fc_new)

model.save("runs/ivdfm", overwrite=True)
model = iVDFM.load("runs/ivdfm", device="cpu")
```

| Hyperparameter            | Default       | Meaning |
|---------------------------|---------------|---------|
| `n_factors`               | 3             | number of latent factors; may exceed the number of targets |
| `factor_order`            | 1             | AR order of the factor dynamics |
| `window`, `stride`        | None, None    | training window length (None: the whole series) and step between windows (None: `window // 2`) |
| `time_context`            | 1             | generated time features in u_t |
| `innovation_distribution` | `"laplace"`   | `"laplace"`, `"gaussian"`, `"student_t"`, `"gamma"`, `"beta"` or `"exponential"` |
| `decoder_type`            | `"mlp"`       | `"mlp"`, `"residual"` or `"linear"` |
| `n_regimes`               | 1             | K > 1 adds regime-indexed priors, decoders and AR coefficients |
| `beta_kl`                 | 1.0           | weight of the KL term in the ELBO |
| `learning_rate`, `batch_size`, `max_epochs` | 1e-3, 32, 100 | training |
| `tol`, `patience`         | 5e-4, 20      | stopping on the ELBO; early stopping restores the best epoch |
| `scaler`                  | None          | scaler of the targets (not of exog) |
| `random_state`, `device`  | 3, `"auto"`   | seed and device |

The network widths and depths (`encoder_hidden_dim`, `decoder_hidden_dim`,
`prior_hidden_dim` and their `*_n_hidden_layers`), `activation`, `slope`,
`decoder_var`, `regime_temperature`, `use_layer_norm`, `mixing`, the
optimizer and the learning-rate scheduler (`scheduler_type` and its
parameters) are documented in `help(iVDFM)`.

Notes:

- The AR coefficients are initialized by OLS on the PCA factors of the last
  training window; with `factor_order > 1` the factors beyond those keep the
  default random initialization.
- Without `futr_exog`, a model fitted with exog warns and fills the future
  context with time features. A `futr_exog` DataFrame is matched by column
  name and its index must be the forecast dates (a default `RangeIndex` or an
  array is taken by position); a polars frame's Date or Datetime column is
  its index.
- `update(data, refit=True)` raises `UnsupportedInputError`.
- `factors_` is the factor path `result_.Z`: the training series encoded as
  one sequence with the trained networks, one row per date of `index_`;
  `predict` starts from its last row. `window_factors_` holds the factors of
  every training window on a `(window, time)` MultiIndex in chronological
  order, so a date covered by several windows appears once per window.
  `update` changes neither.
- `loadings_kind_` is `"mean_jacobian"` for every `decoder_type`: `loadings_`
  (targets x factors, scaled space) is the Jacobian of the decoded path
  `result_.x_sm` with respect to the factors, averaged over the rows of
  `factors_`. The path is decoded as for `result_.x_sm`: through the
  `mixing` layer when there is one, and with `n_regimes` K > 1 the K
  decoders mixed with uniform weights 1/K, not with the regime probabilities
  of the context. `explained_variance_` compares the observed (finite)
  scaled targets with `result_.x_sm`.
- `decompose()`: `"common"` is `result_.x_sm` inverse-scaled.
  `decompose(by="factor")` raises `UnsupportedInputError`, and there is no
  `predict_interval`.

Hyperparameter search uses optuna (the `tune` extra). It fits clones of the
estimator on a holdout split, never changes the estimator, and returns the
best values under the estimator's names:

```python
from dfm_python.tuning import tune_hyperparameters

best = tune_hyperparameters(model, data, n_trials=3, max_epochs_per_trial=2,  # use more trials
                            horizon=12, min_window=24, max_window=44, metric="sMSE")
model.set_params(**best).fit(data)
```

The estimator's `random_state` (or `fixed["random_state"]`) also seeds
optuna's TPE sampler, so the same seed, data and arguments give the same
trials and, on CPU, the same `best`; `random_state=None` leaves the sampler
unseeded. Resuming a stored study (`storage`, `load_if_exists=True`) that
already has `n` trials seeds the sampler with `random_state + n`, so the
resumed run draws new trials. `window` and `stride` step by 10, and a range
that is not a multiple of 10 is cut to its last step (`min_window=10,
max_window=25` searches 10 and 20).

Method: Chang, M. and Kim, J.-Y. (2026), "Conditionally Identifiable Latent
Representation for Multivariate Time Series with Structural Dynamics",
[arXiv:2603.22886](https://arxiv.org/abs/2603.22886). The paper introduces
iVDFM; this package is its reference implementation.

Tutorials: `tutorial/tutorial_{macro,finance,exchange}_ivdfm.py`.

## AFM

AFM takes asset returns `(T, N)` and, for learned attention factors, firm
characteristics `(T, N, M)`. It is trained on an after-cost Sharpe ratio plus
an explained-variance term (`lambda_var`). AFM allocates rather than
forecasts: `allocate()` returns portfolio weights, `backtest(data)` evaluates
them on other data, and `predict` raises `TypeError`.

```python
import numpy as np
import pandas as pd
from dfm_python import AFM, PanelData

rng = np.random.default_rng(0)
T, N, M = 120, 20, 8                        # dates, assets, characteristics
returns = 0.01 * rng.standard_normal((T, N))
characteristics = rng.standard_normal((T, N, M))
dates = pd.bdate_range("2020-01-01", periods=T, name="date")
tickers = [f"a{i}" for i in range(N)]

data = PanelData.from_arrays(y=returns, series_exog=characteristics, index=dates,
                             names=tickers, series_exog_names=[f"c{j}" for j in range(M)])
model = AFM(n_factors=3, hist_len=20, max_epochs=20, random_state=0, device="auto")
model.fit(data)

weights = model.allocate()                  # DataFrame (T - hist_len) x N, index dates[hist_len:]
model.result_.sharpe, model.result_.explained_variance
model.residuals_                            # DataFrame T x N of residual returns
model.book_returns_                         # Series of after-cost returns on dates[hist_len:]

res = model.backtest(data.loc(dates[60], None))   # AFMResult on other data, no training
res.sharpe, res.weights.shape               # (1, n_eval, N)

model.save("runs/afm", overwrite=True)
model = AFM.load("runs/afm", device="cpu")
weights = model.allocate(data=data)         # a loaded model keeps no data: pass it

# Benchmarks: PCA factors need no characteristics
pca = AFM(n_factors=3, hist_len=20, factor_model="pca", trading="ou", max_epochs=5)
pca.fit(PanelData(y=pd.DataFrame(returns, index=dates, columns=tickers)))
```

| Hyperparameter  | Default        | Meaning |
|-----------------|----------------|---------|
| `n_factors`     | None           | number of factors K; required (None raises in `fit`) |
| `factor_model`  | `"attention"`  | `"attention"` (needs `series_exog`) or `"pca"` (rolling-window PCA on the returns) |
| `trading`       | `"longconv"`   | learned LongConv filter, or `"ou"` for the Avellaneda-Lee rule |
| `embed_dim`     | 32             | characteristic embedding dimension (attention) |
| `hist_len`      | 20             | residual-history length; the first `hist_len` dates get no weights |
| `n_kernels`, `squash_lambda` | 32, 1e-3 | LongConv kernels and its Squash regularizer |
| `ridge`         | 0.01           | ridge penalty of the closed-form loadings |
| `pca_window`, `reestim` | 252, 21 | PCA benchmark window and re-estimation stride |
| `lambda_var`    | 0.1            | weight of the explained-variance regularizer |
| `turnover_cost`, `short_cost`, `risk_free` | 5e-4, 1e-4, 0.0 | costs per unit turnover (5 bps) and short position (1 bp), risk-free rate |
| `learning_rate`, `max_epochs`, `patience` | 1e-3, 50, None | full-batch Adam training |
| `random_state`, `device` | None, `"auto"` | seed (None does not seed) and device |

Notes:

- The three model classes of the paper are `factor_model="attention",
  trading="longconv"` (Attention Factors), `factor_model="pca",
  trading="longconv"` (PCA + LongConv) and `factor_model="pca", trading="ou"`
  (PCA + Avellaneda-Lee).
- In long format (`unique_id`, `ds`, `y` plus one column per characteristic)
  build the same panel with
  `PanelData.from_long(df, series_exog_cols=["c0", ..., "c7"])`.
- `update(data)` attaches the data for later `allocate()`;
  `update(data, refit=True)` continues training from the current weights.
- `set_output(output="polars")` applies to `allocate`. AFM has no
  `decompose` or `predict_interval` (tags `capability:interpret` and
  `capability:pred_int` are False).

Method: Epstein, Wang, Choi and Pelger (2025), "Attention Factors for
Statistical Arbitrage", arXiv:2510.11616. This is an implementation from the
paper (no reference code exists). The LongConv filter follows Fu et al.
(2023), and the OU benchmark the Avellaneda and Lee (2010) recipe. AFM is a
baseline in Chang, M. (2026), "Amortized Neural Filtering for Calibrated
Posteriors in Statistical Arbitrage",
[arXiv:2610.25802](https://arxiv.org/abs/2610.25802) (ICAIF '26), whose code
is in [minkeymouse/nfe](https://github.com/minkeymouse/nfe).
