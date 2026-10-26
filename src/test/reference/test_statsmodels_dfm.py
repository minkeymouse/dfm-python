"""DFM against statsmodels' ``DynamicFactorMQ`` on real data (``data/macro.csv``).

Both libraries implement the EM algorithm of Banbura and Modugno (2014) for
the same state space model when configured as here: one factor with VAR(1)
dynamics, an AR(1) idiosyncratic state for every series, quarterly series
loaded through the ``[1, 2, 3, 2, 1]`` tent on the factor and on their own
monthly AR(1) idiosyncratic state, and the initial state re-estimated by EM
(statsmodels ``em_initialization=True``, its default; dfm-python always).
Both get the same standardized data (``standardize=False``, ``scaler=None``).
The one difference left is the observation noise R: statsmodels
(``obs_cov_diag=True``) fixes it at 1e-4, dfm-python estimates it with 1e-4
as a floor; at these fits it is at the floor or within 2.5e-8 of it.

Data: ``data/macro.csv`` resampled to month ends (a weekly series: its
monthly mean; a monthly series: its value), 2000-01 to 2025-10 (310 months,
ragged last rows), each series standardized with its mean and population
standard deviation (what ``scaler="standard"`` does).

- ``monthly``: the 14 series.
- ``mixed``: the same panel with KOEQUIPTE and KOWRCCNSE turned into quarterly
  series, ``(x_t + 2 x_{t-1} + 3 x_{t-2} + 2 x_{t-3} + x_{t-4}) / 3`` (the
  quarterly growth of the 3-month average level, for monthly growth rates
  ``x``) in the third month of each quarter, NaN in the other months.

Checks, with the values measured on this data in the docstrings (the tables
and the diagnosis of the differences are in ``README.md`` next to this file):

1. The likelihoods are the same function: statsmodels' log-likelihood at
   dfm-python's fitted parameters is dfm-python's ``loglik_``.
2. The EM maps are the same: plain EM iterations from the same parameters
   give the same parameters, log-likelihood, common components, forecasts
   and forecast standard deviations.
3. Each library's own fit (its own start values, acceleration and stopping)
   gives the same common components and forecasts up to optimization error,
   and log-likelihoods a few units apart.
"""

import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

dynamic_factor_mq = pytest.importorskip("statsmodels.tsa.statespace.dynamic_factor_mq")

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from dfm_python import DFM, PanelData

DATA = Path(__file__).resolve().parents[3] / "data" / "macro.csv"
ROWS = ("2000-01", "2025-10")
QUARTERLY = ("KOEQUIPTE", "KOWRCCNSE")
TENT = (1.0, 2.0, 3.0, 2.0, 1.0)  # the only kernel statsmodels supports
SLICES = ("monthly", "mixed")
HORIZON = 12
SM_MAXITER = 200  # statsmodels' EM iterations in its own fit; its default 500 doubles the run time
EM_ITERATIONS = 10  # plain EM iterations compared in check 2
# The first rows of the common components depend on the EM-estimated initial
# state, which the data hardly pin down: the factor in the first row, and for
# a quarterly series the pre-sample factor lags in its first len(TENT) - 1
# rows (README). The correlations of the own fits skip those rows.
BURN_IN = len(TENT) - 1


def _standardized_panel(kind: str) -> tuple[pd.DataFrame, int]:
    """The standardized monthly panel and its number of quarterly series (its last columns)."""
    panel = pd.read_csv(DATA, parse_dates=["date"], index_col="date").resample("ME").mean()
    n_quarterly = 0
    if kind == "mixed":
        for name in QUARTERLY:
            x = panel[name]
            q = sum(w * x.shift(lag) for lag, w in enumerate(TENT)) / 3.0
            panel[name] = q.where(q.index.month % 3 == 0)
        panel = panel[[c for c in panel.columns if c not in QUARTERLY] + list(QUARTERLY)]
        n_quarterly = len(QUARTERLY)
    y = panel.loc[ROWS[0]:ROWS[1]]
    return (y - y.mean()) / y.std(ddof=0), n_quarterly


def _panel_data(z: pd.DataFrame, n_quarterly: int) -> PanelData:
    freq = {name: "q" for name in z.columns[z.shape[1] - n_quarterly:]}
    return PanelData(y=z, freq=freq or None)


def _dfm(n_quarterly: int, **params) -> DFM:
    tent = {"q:m": list(TENT)} if n_quarterly else None
    return DFM(n_factors=1, clock="m", tent_weights=tent, scaler=None, **params)


def _statsmodels_model(z: pd.DataFrame, n_quarterly: int, obs_var: np.ndarray | None = None):
    """The ``DynamicFactorMQ`` model; ``obs_var`` replaces its fixed observation noise variances (1e-4)."""
    endog = z.copy()
    endog.index = endog.index.to_period("M")
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog, k_endog_monthly=z.shape[1] - n_quarterly, factors=1, factor_orders=1,
        idiosyncratic_ar1=True, standardize=False, obs_cov_diag=True,
    )
    if obs_var is not None:
        mod["obs_cov"] = np.diag(obs_var)  # not a parameter with idiosyncratic_ar1: EM keeps it
    return mod


def _statsmodels_fit(mod, **kwargs):
    with warnings.catch_warnings():  # EM stops at maxiter on this data
        warnings.simplefilter("ignore", ConvergenceWarning)
        return mod.fit_em(**kwargs)


def _statsmodels_parameters(model: DFM, n_quarterly: int) -> tuple[np.ndarray, ...]:
    """``DynamicFactorMQ`` parameters, initial state mean and covariance, and R of a fitted one-factor DFM.

    dfm-python's state is ``[f_t, ..., f_{t-L+1} | AR(1) state of each monthly
    series | 5-state tent chain of each quarterly series]`` with ``L = 2``
    (VAR(1) plus one lag) without quarterly series and ``L = 5`` with them;
    statsmodels' is ``[f_t (and f_{t-1..t-4} with quarterly series) | the
    monthly AR(1) states | the quarterly chains interleaved lag by lag]``. The
    lag ``f_{t-1}`` of the monthly-only model is not loaded and does not enter
    the VAR, so dropping it changes nothing. A quarterly series' loading is
    its entry on ``f_t`` (tent weight 1).
    """
    res = model.result_
    n_series = res.C.shape[0]
    n_monthly = n_series - n_quarterly
    k = len(TENT)
    lags = res.A.shape[0] - n_monthly - k * n_quarterly
    idio = np.r_[lags + np.arange(n_monthly), lags + n_monthly + k * np.arange(n_quarterly)]
    params = np.r_[res.C[:, 0], res.A[0, 0], np.sqrt(res.Q[0, 0]), res.A[idio, idio], res.Q[idio, idio]]
    chains = [lags + n_monthly + k * j + lag for lag in range(k) for j in range(n_quarterly)]
    order = np.r_[np.arange(k if n_quarterly else 1), lags + np.arange(n_monthly), chains].astype(int)
    return params, res.Z_0[order], res.V_0[np.ix_(order, order)], np.diag(res.R)


def _unit_loading_scale(params, n_series: int) -> np.ndarray:
    """statsmodels parameters with the factor rescaled to a unit-norm loading column, as dfm-python reports it."""
    out = np.array(params, dtype=np.float64)
    d = np.linalg.norm(out[:n_series])
    out[:n_series] /= d  # loadings
    out[n_series + 1] *= d  # Cholesky factor of the factor innovation variance
    return out


def _statsmodels_common(res, n_quarterly: int) -> np.ndarray:
    """Common component (T x N): the factor states through the design matrix."""
    n_factor_states = len(TENT) if n_quarterly else 1
    design = res.filter_results.design[:, :n_factor_states, 0]
    return (design @ res.smoothed_state[:n_factor_states]).T


def _forecast_sd(model: DFM) -> np.ndarray:
    """Standard deviations (HORIZON x N) of dfm-python's forecasts: half the width of its one-sd ``predict_interval``."""
    band = model.predict_interval(HORIZON, coverage=math.erf(1 / math.sqrt(2)))
    return (band.xs("upper", axis=1, level="bound") - band.xs("lower", axis=1, level="bound")).to_numpy() / 2


@pytest.fixture(scope="module", params=SLICES)
def own_fits(request):
    """Each library's own fit: dfm-python's defaults (SQUAREM, tol=1e-5), statsmodels' EM for SM_MAXITER iterations."""
    z, n_quarterly = _standardized_panel(request.param)
    model = _dfm(n_quarterly).fit(_panel_data(z, n_quarterly))
    assert list(model.internal_feature_names_) == list(z.columns)  # result_ rows are in the data's order
    res = _statsmodels_fit(_statsmodels_model(z, n_quarterly), maxiter=SM_MAXITER)
    return SimpleNamespace(z=z, n_quarterly=n_quarterly, model=model, res=res)


@pytest.fixture(scope="module", params=SLICES)
def em_path(request):
    """EM_ITERATIONS plain EM iterations from the same parameters in both libraries.

    The start is dfm-python's fit after one EM iteration from its own
    initialization (and its R, which statsmodels then keeps fixed). The end
    is dfm-python's fit after 1 + EM_ITERATIONS iterations: dfm-python
    rescales the factor once at the end of a fit (likelihood-invariant), and
    the plain EM map commutes with that rescaling.
    """
    z, n_quarterly = _standardized_panel(request.param)
    data = _panel_data(z, n_quarterly)
    start = _dfm(n_quarterly, max_iter=1, em_acceleration=None, tol=0.0).fit(data)
    end = _dfm(n_quarterly, max_iter=EM_ITERATIONS + 1, em_acceleration=None, tol=0.0).fit(data)
    params, mean, cov, obs_var = _statsmodels_parameters(start, n_quarterly)
    mod = _statsmodels_model(z, n_quarterly, obs_var)
    mod.ssm.initialize_known(mean, cov)
    res = _statsmodels_fit(mod, start_params=params, maxiter=EM_ITERATIONS, tolerance=0.0)
    return SimpleNamespace(z=z, n_quarterly=n_quarterly, model=end, res=res)


def test_statsmodels_loglik_at_dfm_parameters(own_fits):
    """statsmodels' log-likelihood at dfm-python's fitted parameters is dfm-python's loglik_.

    Same function, constant terms included: both add -log(2 pi) / 2 per
    observed entry. R is dfm-python's, which stays within 0.1% of the 1e-4
    that statsmodels fixes: at the floor in the monthly fit, 1.000242e-4 and
    1.000128e-4 for the two quarterly series in the mixed one (with 1e-4 the
    loglik moves by 1.8e-8). Measured differences: 7e-12 (monthly, loglik
    -5143.364) and 4e-12 (mixed, -4621.637).
    """
    model = own_fits.model
    params, mean, cov, obs_var = _statsmodels_parameters(model, own_fits.n_quarterly)
    np.testing.assert_allclose(obs_var, 1e-4, rtol=1e-3, atol=0)
    mod = _statsmodels_model(own_fits.z, own_fits.n_quarterly, obs_var)
    mod.ssm.initialize_known(mean, cov)
    assert mod.loglike(params) == pytest.approx(model.loglik_, rel=1e-12, abs=0)


def test_em_iterations_match(em_path):
    """Plain EM from the same parameters: the same parameters, loglik, common component and forecasts.

    Measured after 10 iterations (monthly / mixed): parameters (statsmodels'
    vector, the factor rescaled as dfm-python reports it) within 7.7e-9 /
    9.8e-9, loglik within 1.9e-8 / 3.1e-8 relative, common component within
    3.8e-8 / 5.4e-8, 12-step forecasts within 2.4e-9 / 9.7e-9 (standardized
    units) and their standard deviations (``predict_interval`` against
    ``get_forecast``) within 6.6e-9 / 9.0e-9 relative. These residuals come
    from R alone: dfm-python re-estimates it each iteration (up to 1e-8 above
    its floor here) where statsmodels keeps the start value; with dfm-python's
    R update disabled they drop below 7e-10 (README).
    """
    model, res, n_quarterly = em_path.model, em_path.res, em_path.n_quarterly
    assert model.n_iter_ == EM_ITERATIONS + 1
    assert res.mle_retvals.iter == EM_ITERATIONS
    dfm_params = _statsmodels_parameters(model, n_quarterly)[0]
    sm_params = _unit_loading_scale(res.params, em_path.z.shape[1])
    np.testing.assert_allclose(dfm_params, sm_params, rtol=0, atol=1e-7)
    assert model.loglik_ == pytest.approx(res.llf, rel=1e-7, abs=0)
    common = model.decompose()["common"].to_numpy()
    np.testing.assert_allclose(common, _statsmodels_common(res, n_quarterly), rtol=0, atol=1e-6)
    forecast = model.predict(HORIZON).to_numpy()
    np.testing.assert_allclose(forecast, res.forecast(HORIZON).to_numpy(), rtol=0, atol=1e-7)
    sm_sd = np.sqrt(np.diagonal(res.get_forecast(HORIZON).var_pred_mean, axis1=1, axis2=2))
    np.testing.assert_allclose(_forecast_sd(model), sm_sd, rtol=1e-7, atol=0)


def test_own_fits_common_component(own_fits):
    """Per series, the common components correlate above 0.999 (the factors too, up to sign and scale).

    Measured (rows from the fifth on): lowest correlation 0.999626 (monthly,
    KOROCOINF) and 0.999497 (mixed, KOEQUIPTE); |corr| of the factors
    0.999626 / 0.999797. Against statsmodels' default 500 iterations: 0.999983
    / 0.999982. Over all rows the lowest correlations are 0.998963 (monthly:
    the first row's factor, -1.02 against 0.27 for a factor of standard
    deviation 2.1, depends on the EM-estimated initial state) and 0.9745
    (mixed, the quarterly series: the pre-sample factor lags) (README).
    """
    dfm_common = own_fits.model.decompose()["common"].to_numpy()[BURN_IN:]
    sm_common = _statsmodels_common(own_fits.res, own_fits.n_quarterly)[BURN_IN:]
    corr = [np.corrcoef(dfm_common[:, i], sm_common[:, i])[0, 1] for i in range(dfm_common.shape[1])]
    assert min(corr) > 0.999, dict(zip(own_fits.z.columns, np.round(corr, 6)))
    factor = own_fits.model.factors_.iloc[BURN_IN:, 0].to_numpy()
    assert abs(np.corrcoef(factor, own_fits.res.smoothed_state[0, BURN_IN:])[0, 1]) > 0.999


def test_own_fits_forecasts(own_fits):
    """The 1- to 12-step forecasts differ by at most 0.2 forecast standard deviations.

    The standard deviation is dfm-python's (``predict_interval`` at one
    standard deviation, 0.77 to 1.13 here; check 2 matches it with
    statsmodels' at the same parameters). Measured largest ratio: 0.101
    (monthly, KOEQUIPTE at h = 2) and 0.055 (mixed, KOWRCCNSE at h = 2);
    largest absolute differences 0.108 / 0.052 standardized units. Against
    statsmodels' default 500 iterations: 0.101 / 0.034.
    """
    forecast = own_fits.model.predict(HORIZON)
    sm_forecast = own_fits.res.forecast(HORIZON).to_numpy()
    ratio = np.abs(forecast.to_numpy() - sm_forecast) / _forecast_sd(own_fits.model)
    assert ratio.max() <= 0.2, pd.DataFrame(ratio, columns=forecast.columns).max().round(3).to_dict()


def test_own_fits_loglik(own_fits):
    """The log-likelihoods at the two fits' end points differ by less than 0.003 per observed entry.

    Measured dfm-python minus statsmodels: +7.93 (monthly, 4338 entries,
    0.0018 per entry) and +5.16 (mixed, 3925 entries, 0.0013): dfm-python's
    SQUAREM at tol 1e-5 (66 and 143 E-steps) ends higher than 200 plain EM
    iterations; against statsmodels' default 500 iterations the gaps are
    -0.62 / -1.43. Check 1 shows the functions are the same, so the gap is
    optimization progress, which on this flat likelihood depends on start
    values and stopping (README).
    """
    n_obs = int(own_fits.z.notna().to_numpy().sum())
    gap = own_fits.model.loglik_ - own_fits.res.llf
    assert abs(gap) < 0.003 * n_obs, gap
