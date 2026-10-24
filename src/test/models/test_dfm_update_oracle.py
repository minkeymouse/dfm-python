"""Oracle tests for ``DFM.update`` and ``DFM.predict(data=)``: continue the training filter.

On the ``dfm_synthetic`` panel the model is fitted on all rows but the last
``n`` and then given those ``n`` rows. With the fitted parameters held fixed,
the state the forecasts start from must be the filtered state
``x_{T+n | T+n}`` of the independent reference filter
(``src/test/ssm/_reference_kf.py``) run over the training and new rows in one
pass, so ``predict_factors(h)[k-1] = A^k x_{T+n|T+n}`` (rel 1e-8). Updating
with ``n`` rows at once and one row at a time must agree, and
``predict_factors(h, data=new)`` must use the same continuation.

``update`` and ``predict(data=)`` continue the filter over the conditioning
rows: the prior of the first new row is ``(A x_T|T, A P_T|T A' + Q)``.
"""

from __future__ import annotations

import copy
import logging
import warnings

import numpy as np
import pytest

from ..ssm import _reference_kf as ref

RTOL = 1e-8
H = 4


def _fit(n_new: int):
    """``(model, train_panel, new_panel, X_all)``: DFM fitted on the first ``T - n_new`` synthetic rows.

    ``X_all`` is every row (training and new) scaled with the fitted ``scaler_``,
    in the engine's column order.
    """
    from dfm_python import DFM, PanelData

    from ..parity._cases import dfm_synthetic_inputs

    df, _, config, _ = dfm_synthetic_inputs()
    cols = list(config["frequency"])
    y = df.set_index("date")
    freq = {c: "m" for c in cols}
    train, new = PanelData(y=y.iloc[:-n_new], freq=freq), PanelData(y=y.iloc[-n_new:], freq=freq)
    model = DFM(blocks={"Block_Global": {"n_factors": 2, "series": cols}}, clock="m",
                max_iter=20, tol=1e-5, scaler="standard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train)
    internal = list(model.internal_feature_names_)
    X_all = model.scaler_.transform(y[internal].to_numpy(np.float64))
    np.testing.assert_allclose(X_all[:-n_new], model.engine_.data_processed, rtol=1e-12, atol=1e-12)
    return model, train, new, X_all


def _reference_forecast(model, X_all: np.ndarray) -> np.ndarray:
    """``A^k x_{T+n|T+n}``, k = 1..H, from the reference filter over all rows at the fitted parameters."""
    res = model.result_
    A = np.asarray(res.A, dtype=np.float64)
    x = ref.kalman_filter(res.A, res.C, res.Q, res.R, res.Z_0, res.V_0, X_all)["fm"][-1]
    out = []
    for _ in range(H):
        x = A @ x
        out.append(x)
    return np.array(out)


def _states(model, data=None) -> np.ndarray:
    """The full state forecast behind predict_factors (which reports its current-factor columns)."""
    return np.asarray(_quiet(model._run_forecast, "predict_factors", H, data)[1])


def _quiet(fn, *args, **kwargs):
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("n_new", [1, 5])
def test_update_continues_the_training_filter(n_new):
    """After update(new rows) the forecast starts from the reference x_{T+n|T+n} (rel 1e-8)."""
    model, _, new, X_all = _fit(n_new)
    expected = _reference_forecast(model, X_all)
    _quiet(model.update, new)
    got = _states(model)
    assert ref.rel_err(got, expected) < RTOL


def test_update_one_row_at_a_time_matches_all_at_once():
    """update(5 rows) and five update(1 row) calls end in the same state (rel 1e-8)."""
    model, _, new, _ = _fit(5)
    together = copy.deepcopy(model)
    _quiet(together.update, new)
    one_by_one = copy.deepcopy(model)
    for k in range(5):
        from dfm_python import PanelData

        row = PanelData(y=new.y.iloc[k:k + 1], freq=new.freq)
        _quiet(one_by_one.update, row)
    a = _states(together)
    b = _states(one_by_one)
    assert ref.rel_err(a, b) < RTOL


@pytest.mark.parametrize("n_new", [1, 5])
def test_predict_with_data_continues_the_training_filter(n_new):
    """predict_factors(h, data=new) forecasts from the reference x_{T+n|T+n} (rel 1e-8), model unchanged."""
    model, _, new, X_all = _fit(n_new)
    expected = _reference_forecast(model, X_all)
    before = _states(model)
    got = _states(model, new)
    assert ref.rel_err(got, expected) < RTOL
    np.testing.assert_array_equal(_states(model), before)
    np.testing.assert_array_equal(_quiet(model.predict_factors, H, data=new).to_numpy(),
                                  got[:, model._factor_states()])
