# DFM against statsmodels' DynamicFactorMQ

`test_statsmodels_dfm.py` checks dfm-python's `DFM` against statsmodels'
`DynamicFactorMQ` on real data. Rerun it with

```bash
PYTHONPATH=src python -m pytest src/test/reference -q
```

It needs statsmodels (dev group; skipped without it) and takes about 40 s
(10 tests). Measured with statsmodels 0.15.0, numpy 2.3.5, scipy 1.15.3,
pandas 2.3.3, dfm-python 0.8.3 (commit ad3b0eb); the `tol=1e-6` rows after
the default changed.

## Configuration

Both libraries run the EM of Banbura and Modugno (2014) for this model:

| | statsmodels `DynamicFactorMQ` | dfm-python `DFM` |
|---|---|---|
| one factor, VAR(1) | `factors=1, factor_orders=1` | `n_factors=1` (VAR(1) is fixed) |
| AR(1) idiosyncratic state per series | `idiosyncratic_ar1=True` (default) | always |
| quarterly series | last columns, `k_endog_monthly`; tent `[1, 2, 3, 2, 1]` (fixed) | `freq={"name": "q"}`, `tent_weights={"q:m": [1, 2, 3, 2, 1]}` |
| observation noise R | `obs_cov_diag=True`: fixed at 1e-4 (default `False`: R = 0) | estimated, floored at 1e-4 |
| initial state | `em_initialization=True` (default): E[x_0 \| Y], Cov(x_0 \| Y) | the same, always |
| scaling | `standardize=False` on pre-standardized data | `scaler=None` on the same data |
| start values | PCA, OLS loadings, SARIMAX AR(1)s | BM14 `InitCond` (PCA on splined data) |
| iterations | plain EM, stops when `2 \|dl\| / (\|l_1\| + \|l_2\|) < 1e-6` or after `maxiter=500` | SQUAREM, stops when `\|dl / l\| < tol` (default 1e-6; the test's own fit uses 1e-5) |

Data: `data/macro.csv` resampled to month ends (weekly series: monthly mean;
monthly series: their value), 2000-01 to 2025-10 (310 months; KOROCOINF and
KOEQUIPTE are missing in the last month), standardized with each series'
mean and population standard deviation. Slice `monthly`: the 14 series (4338
observed entries). Slice `mixed`: KOEQUIPTE and KOWRCCNSE replaced by
quarterly series `(x_t + 2 x_{t-1} + 3 x_{t-2} + 2 x_{t-3} + x_{t-4}) / 3` in
the third month of each quarter (3925 observed entries).

## Results

### 1. The likelihood is the same function

statsmodels' log-likelihood at dfm-python's fitted parameters and initial
state (state vector permuted, R set to dfm-python's), against `DFM.loglik_`:

| slice | dfm-python `loglik_` | statsmodels at the same parameters | difference |
|---|---|---|---|
| monthly | -5143.3640 | -5143.3640 | 7e-12 |
| mixed | -4621.6367 | -4621.6367 | 4e-12 |

Both add `-log(2 pi) / 2` per observed entry (-3986.36 and -3606.83).

### 2. The EM map is the same

10 plain EM iterations from the same parameters (dfm-python's after one
iteration), dfm-python's result against statsmodels' (factor rescaled to a
unit loading column as dfm-python reports it):

| slice | parameters | loglik (relative) | common component | 12-step forecasts | forecast sd (relative) |
|---|---|---|---|---|---|
| monthly | 7.7e-9 | 1.9e-8 | 3.8e-8 | 2.4e-9 | 6.6e-9 |
| mixed | 9.8e-9 | 3.1e-8 | 5.4e-8 | 9.7e-9 | 9.0e-9 |

The forecast standard deviations are dfm-python's `predict_interval` against
statsmodels' `get_forecast(12).var_pred_mean`. These residuals come from R:
dfm-python re-estimates it every iteration (at most 1e-8 above its floor
here), statsmodels keeps it fixed. With dfm-python's R update disabled
(patched, not part of the test) every column drops below 7e-10 (monthly) and
2e-13 (mixed). One EM map from statsmodels' start values (dfm-python's
`em_step` on the permuted state, not part of the test) agrees to 3e-11
(monthly) and 1e-15 (mixed) in A, C, Q, Z_0 and V_0, and to 1e-8 in R.

### 3. Each library's own fit

dfm-python with SQUAREM at `tol=1e-5` (66 and 143 E-steps; the 0.8
default) against statsmodels' EM after 200 iterations (the test) and 500
(its default; neither converges by its tolerance). The last two rows are
dfm-python's default `tol=1e-6` (106 and 710 E-steps, 10 s and 90 s), not
part of the test because of the mixed fit's run time. Common component: lowest
per-series correlation, rows 5 to 310. Forecasts: largest difference over
h = 1..12 and the series, in standardized units and in forecast standard
deviations (`predict_interval`, 0.77 to 1.13).

| slice | statsmodels iterations | (a) common corr | factor \|corr\| | (b) forecast diff | (b) in sd | (c) loglik dfm - sm |
|---|---|---|---|---|---|---|
| monthly | 200 | 0.999626 | 0.999626 | 0.108 | 0.101 | +7.93 |
| monthly | 500 | 0.999983 | 0.999983 | 0.108 | 0.101 | -0.62 |
| mixed | 200 | 0.999497 | 0.999797 | 0.052 | 0.055 | +5.16 |
| mixed | 500 | 0.999982 | 0.999993 | 0.032 | 0.034 | -1.43 |
| monthly, `tol=1e-6` | 500 | 0.999963 | 0.999963 | 0.059 | 0.061 | +29.19 |
| mixed, `tol=1e-6` | 500 | 0.999894 | 0.999938 | 0.059 | 0.061 | +32.19 |

Test thresholds: correlation > 0.999, forecast difference <= 0.2 sd, loglik
gap < 0.003 per observed entry (13 and 12 here).

Run to tight convergence (statsmodels 3000 iterations with `tolerance=0`,
since its default tolerance stops the monthly run after 2497 at -5119.47;
dfm-python `tol=1e-9`: 169 and 1523 E-steps) the logliks still differ
(statsmodels against dfm-python: monthly -5117.27 against -5110.37, mixed
-4593.78 against -4586.82), while the factors
correlate to 0.9999995 and 0.9999997, the monthly series' common components
to 1.0000 (4 decimals) and the forecasts differ by at most 0.050 and 0.047.

## Why the own fits differ

The EM is the same (results 1 and 2), so the differences come from where
each run starts and stops on a flat likelihood:

- Heywood ridge. KORMCMP correlates 0.90 with KORELEC and 0.60 to 0.76 with
  the other weekly retail series; the maximum likelihood puts its
  idiosyncratic variance at 0 (the factor becomes KORMCMP; R = 1e-4 keeps the
  likelihood bounded). Plain EM crawls along that ridge: statsmodels' loglik
  still rises by 0.004 per iteration after 3000 iterations (idiosyncratic
  variance 1.8e-4), dfm-python's SQUAREM gets to 1e-5 in 169 E-steps. The
  common components hardly change along the ridge; the loglik does (tens of
  units).
- Initial state. EM sets the prior of the first row's state to its smoothed
  moments, so the prior variance shrinks every iteration (the factor's is
  0.0017 at dfm-python's monthly fit) and the first row follows the prior
  more than the data. In the monthly slice the first row's factor is 0.27
  (dfm-python) against -1.02 (statsmodels after 200 iterations, mapped to
  dfm-python's scale; -0.09 after 500), for a factor of standard deviation
  2.1: the lowest correlation over all rows is 0.99896, 0.99957 from the
  second row on.
- Pre-sample factor lags. A quarterly observation in 2000-03 loads on
  `f_{-2}, f_{-1}`, which only the EM-estimated initial state pins down. The
  smoothed lags at the first row are 14.7, 11.2, 3.6, 1.1 (statsmodels) and
  7.1, 5.5, 1.7, 0.6 (dfm-python), for a factor of standard deviation 2.1, so
  the quarterly common components differ in the first four rows (by up to 1.0
  standardized units; correlation over all rows 0.97) and agree from the
  fifth (statsmodels 500 iterations).
- Start values. With KORMCMP and KORELEC dropped (12 series, same rows) the
  two fits end in different maxima: statsmodels -4641.89 (3000 iterations,
  `tolerance=0`; its default stops after 164 at -4643.23), dfm-python
  -4655.00 (`tol=1e-8`, 986 E-steps; -4654.89 after 6000), with
  KOEQUIPTE's loading 0.13 against 0.31 factor standard deviations. Swapping
  the starts swaps the results: dfm-python's EM from statsmodels' start values
  reaches -4641.85, statsmodels' EM from dfm-python's start -4655.04.
- Stopping. `tol=1e-5` on the relative loglik change can stop early. On rows
  2005-01 to 2025-10 (monthly) dfm-python stops after 22 E-steps at -4230.98,
  where statsmodels has -4208.02 after 150 iterations; with `tol=1e-6` it
  goes on to -4181.80 (386 E-steps) and with `1e-7` to the KORMCMP ridge,
  -4178.40 (404). The mixed slice of those rows agrees as in the table
  (against 150 statsmodels iterations: common corr 0.99997, forecasts 0.023
  sd, loglik +0.98).

## Constant terms of the log-likelihood

Both report the Gaussian log-likelihood of the standardized observed entries,
`-log(2 pi) / 2` per entry included. What changes it at equal parameters:

| | monthly | mixed |
|---|---|---|
| statsmodels' parameters (200 iterations), R = 0 instead of 1e-4 | +27.69 | -1.34 |
| statsmodels' default `obs_cov_diag=False` (R = 0) fit, 200 iterations | -5208.32 | -4675.51 |
| the same with `obs_cov_diag=True` | -5151.29 | -4626.80 |
| `standardize=True` (std with ddof=1) minus the ddof=0 data | +7.005 | +7.009 |
| sum over series of n_i log(s_ddof1 / s_ddof0) | 7.011 | 7.015 |

- R: statsmodels' default (R = 0) is a different model. The effect is not a
  constant: it depends on the idiosyncratic variances (large near the
  KORMCMP ridge). With R = 0 the EM path from the same start is also slower.
  Compare with `obs_cov_diag=True`.
- Standardization: statsmodels' `standardize=True` divides by the ddof=1
  standard deviation, dfm-python's `scaler="standard"` by the ddof=0 one
  (`DFM(scaler="standard")` on the raw slice gives the same `loglik_` as
  `scaler=None` on the standardized data). Each reports the likelihood of its
  own standardized data, so the two differ by about
  `sum_i n_i log(s_ddof1,i / s_ddof0,i)`, about 0.5 per series. In data units
  (subtract `sum_i n_i log s_i`) they agree: -1630.132 and -1630.125
  (monthly), -545.056 and -545.050 (mixed).

## Findings for dfm-python

No defect: the likelihood (constants included), the E-step and the M-step
match statsmodels. Differences in parametrization and defaults that the
comparison table in `docs/models.md` does not list:

- R is estimated with a floor of 1e-4 (`initialize_observation_noise`, the
  M-step's `_observation_noise_floor`), where Banbura-Modugno's code and
  statsmodels' `obs_cov_diag=True` fix it at 1e-4 and statsmodels' default
  has R = 0. Here it leaves the floor by at most 2.4e-8 (the two quarterly
  series of the mixed fit).
- `scaler="standard"` uses the ddof=0 standard deviation, statsmodels ddof=1
  (log-likelihoods about 0.5 per series apart).
- Default stopping: `tol` on `|dl / l|` against statsmodels' `1e-6` on
  `2 |dl| / (|l_1| + |l_2|)`, with SQUAREM against plain EM. SQUAREM reaches a
  higher loglik per E-step, but the former default `tol=1e-5` could stop far
  from the maximum (the 2005 slice above), so the default is now `1e-6`: on
  the two slices it ends 29 and 32 loglik units above statsmodels' 500
  iterations.
- Start values: BM14 `InitCond` can start in a different basin than
  statsmodels' start values (the 12-series panel above).

## Run time

Per EM iteration on 310 rows (machine shared with other jobs): dfm-python
0.11 s (16 states) and 0.13 s (27 states), statsmodels 0.016 s and 0.04 to
0.06 s. dfm-python's filter loops over rows in Python; statsmodels' is
compiled. The own fits take 7.4 s and 19 s (dfm-python), 3.3 s and 8 to 12 s
(statsmodels, 200 iterations).
