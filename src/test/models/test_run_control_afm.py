"""Run control of AFM: ``max_time``, ``fit(callback=...)`` and ``stop_reason_``.

A stop by the callback or by ``max_time`` after iteration ``k`` keeps what a
``max_iter`` (``max_epochs``) stop after ``k`` iterations keeps, so each such
fit is compared with the fit whose iteration budget is ``k``. ``max_time`` is
tested with the run-control clock replaced by one the callback advances.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from dfm_python import AFM
from dfm_python.testing import make_panel
from dfm_python.utils import run_control
from dfm_python.utils.errors import ConfigurationError

# tol=0, no patience and no learning-rate schedule (whose step depends on max_epochs): only the iteration
# budget, max_time or the callback stops training.
CASES = {
    "AFM": (AFM, {"n_factors": 2, "hist_len": 5, "max_epochs": 4, "factor_model": "pca", "trading": "longconv",
                  "n_kernels": 4, "pca_window": 20, "reestim": 5, "random_state": 0, "device": "cpu"}),
}
BUDGET = {AFM: "max_epochs"}
OBJECTIVE = {AFM: "loss"}
INFO_KEYS = {"model", "iteration", "max_iter", "elapsed", "objective", "objective_name"}


@pytest.fixture(params=list(CASES))
def case(request):
    cls, params = CASES[request.param]
    return cls, dict(params), make_panel(cls)


def _fit(cls, params, data, **fit_kwargs):
    """Fit, failing on any ConvergenceWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return cls(**params).fit(data, **fit_kwargs)


def _budget(cls, params, k):
    return {**params, BUDGET[cls]: k}


def _assert_same_fit(model, reference):
    """The fitted outputs of ``model`` equal those of ``reference``, bitwise."""
    assert model.n_iter_ == reference.n_iter_
    pd.testing.assert_frame_equal(model.residuals_, reference.residuals_, check_exact=True)
    pd.testing.assert_series_equal(model.book_returns_, reference.book_returns_, check_exact=True)


def _drop_stop_reason(path):
    state = path / "state.json"
    content = json.loads(state.read_text())
    del content["stop_reason"]
    state.write_text(json.dumps(content))


def test_info_dict_after_every_iteration(case):
    cls, params, data = case
    infos = []
    model = _fit(cls, params, data, callback=infos.append)
    budget = params[BUDGET[cls]]
    assert model.stop_reason_ == "max_iter" and model.converged_ is False
    assert model.n_iter_ == budget == len(infos)
    for k, info in enumerate(infos, start=1):
        state = info.pop("state", None)
        assert state is None
        assert set(info) == INFO_KEYS
        assert info["model"] == cls.__name__ and info["objective_name"] == OBJECTIVE[cls]
        assert info["iteration"] == k and info["max_iter"] == budget
        assert type(info["iteration"]) is int and type(info["max_iter"]) is int
        assert type(info["elapsed"]) is float and type(info["objective"]) is float
        assert np.isfinite(info["objective"])
    elapsed = [info["elapsed"] for info in infos]
    assert elapsed[0] >= 0 and elapsed == sorted(elapsed)
    # The objective is the training loss the engine reports for its last iteration.
    assert infos[-1]["objective"] == model.network_.loss_now


def test_callback_returning_true_stops_like_max_iter(case):
    cls, params, data = case
    seen = []

    def callback(info):
        seen.append(info["iteration"])
        return info["iteration"] == 3

    model = _fit(cls, params, data, callback=callback)
    assert model.stop_reason_ == "callback" and model.converged_ is False
    assert model.n_iter_ == 3 and seen == [1, 2, 3]
    reference = _fit(cls, _budget(cls, params, 3), data)
    assert reference.stop_reason_ == "max_iter"
    _assert_same_fit(model, reference)


def test_callback_on_the_last_iteration_reports_max_iter(case):
    cls, params, data = case
    model = _fit(cls, params, data, callback=lambda info: True if info["iteration"] == info["max_iter"] else None)
    assert model.stop_reason_ == "max_iter" and model.n_iter_ == params[BUDGET[cls]]


def test_raising_callback_leaves_the_model_unfitted(case):
    cls, params, data = case
    model = _fit(cls, _budget(cls, params, 2), data)
    assert model.stop_reason_ == "max_iter"

    class Stop(Exception):
        pass

    def callback(info):
        if info["iteration"] == 2:
            raise Stop

    with pytest.raises(Stop):
        model.fit(data, callback=callback)
    assert not model._is_fitted() and not hasattr(model, "stop_reason_") and not hasattr(model, "result_")


def test_max_time_stops_at_the_first_boundary_past_the_budget(case, monkeypatch):
    cls, params, data = case
    now = [0.0]
    monkeypatch.setattr(run_control, "perf_counter", lambda: now[0])
    elapsed = []

    def tick(info):  # every iteration takes one second
        elapsed.append(info["elapsed"])
        now[0] += 1.0

    with pytest.warns(ConvergenceWarning, match=rf"{cls.__name__}: training stopped after 3 .*max_time=2\.5") as record:
        model = cls(**params, max_time=2.5).fit(data, callback=tick)
    # The warning points at the caller of fit.
    assert [w.filename for w in record if w.category is ConvergenceWarning] == [__file__]
    assert model.stop_reason_ == "max_time" and model.converged_ is False
    assert model.n_iter_ == 3 and elapsed == [0.0, 1.0, 2.0]
    _assert_same_fit(model, _fit(cls, _budget(cls, params, 3), data))


def test_max_time_without_a_callback(case):
    cls, params, data = case
    with pytest.warns(ConvergenceWarning, match="max_time"):
        model = cls(**params, max_time=1e-9).fit(data)
    assert model.stop_reason_ == "max_time" and model.n_iter_ == 1 and model.converged_ is False
    # A budget that is not reached changes nothing.
    _assert_same_fit(_fit(cls, {**params, "max_time": 600}, data), _fit(cls, params, data))


def test_save_load_round_trips_stop_reason(case, tmp_path):
    cls, params, data = case
    model = _fit(cls, params, data, callback=lambda info: info["iteration"] == 2)
    model.save(tmp_path / "run")
    loaded = cls.load(tmp_path / "run")
    assert loaded.stop_reason_ == "callback" and loaded.n_iter_ == 2 and loaded.max_time is None
    # A run saved by 0.8.x has no stop_reason: it follows from converged.
    _drop_stop_reason(tmp_path / "run")
    assert cls.load(tmp_path / "run").stop_reason_ == "max_iter"


@pytest.mark.parametrize("cls", [AFM])
def test_callback_must_be_callable(cls):
    with pytest.raises(TypeError, match="callback must be None or a callable"):
        cls().fit(make_panel(cls), callback="print")
    with pytest.raises(TypeError):
        cls().fit(make_panel(cls), checkpoint_callback=print)


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), "10", True])
@pytest.mark.parametrize("name", ["AFM"])
def test_invalid_max_time_is_rejected(name, bad):
    cls, params = CASES[name]
    with pytest.raises(ConfigurationError, match="max_time must be None or a finite number of seconds > 0"):
        cls(**params, max_time=bad).fit(make_panel(cls))


@pytest.mark.parametrize("cls", [AFM])
def test_max_time_is_a_hyperparameter(cls):
    assert cls().max_time is None and cls().get_params()["max_time"] is None
    model = cls(max_time=30.0)
    assert model.get_config().max_time == 30.0 and cls.from_config(model.get_config()).max_time == 30.0
    assert any("max_time" in params for params in cls.get_test_params())


def test_afm_early_stopping_and_nothing_to_train():
    cls, params = CASES["AFM"]
    data = make_panel(cls)
    stopped = _fit(cls, {**params, "learning_rate": 1e-12, "patience": 1}, data)
    assert stopped.converged_ and stopped.stop_reason_ == "early_stopping" and stopped.n_iter_ == 2
    ou = _fit(cls, {**params, "trading": "ou", "max_time": 1e-9}, data)
    assert ou.stop_reason_ == "converged" and ou.n_iter_ == 1 and not ou.converged_


def test_afm_progress_names_the_stop(capsys):
    cls, params = CASES["AFM"]
    _fit(cls, {**params, "verbose": 1}, make_panel(cls), callback=lambda info: info["iteration"] == 2)
    assert "AFM: finished after 2 epoch(s), stopped by the callback;" in capsys.readouterr().out


def test_afm_refit_refreshes_stop_reason_within_max_time():
    cls, params = CASES["AFM"]
    data = make_panel(cls)
    model = _fit(cls, params, data, callback=lambda info: info["iteration"] == 2)
    assert model.stop_reason_ == "callback"
    model.update(data, refit=True)
    assert model.stop_reason_ == "max_iter" and model.n_iter_ == params["max_epochs"]
    model.set_params(max_time=1e-9)
    with pytest.warns(ConvergenceWarning, match="max_time") as record:
        model.update(data, refit=True)
    assert [w.filename for w in record if w.category is ConvergenceWarning] == [__file__]
    assert model.stop_reason_ == "max_time" and model.n_iter_ == 1
    # The refit validates a max_time set after fit.
    model.set_params(max_time=-1.0)
    with pytest.raises(ConfigurationError, match="max_time must be None or a finite number of seconds > 0"):
        model.update(data, refit=True)
    assert model.stop_reason_ == "max_time"


def test_afm_runs_saved_without_stop_reason_load_patience_stops_as_early_stopping(tmp_path):
    """AFM's converged_ means a patience stop, so a 0.8.x run with converged=True loads as "early_stopping"."""
    cls, params = CASES["AFM"]
    model = _fit(cls, {**params, "learning_rate": 1e-12, "patience": 1}, make_panel(cls))
    model.save(tmp_path / "run")
    _drop_stop_reason(tmp_path / "run")
    loaded = cls.load(tmp_path / "run")
    assert loaded.converged_ and loaded.stop_reason_ == "early_stopping"


OWN_STOPS = [
    ("AFM", {"learning_rate": 1e-12, "patience": 1}, "early_stopping"),
]


@pytest.mark.parametrize(("name", "extra", "reason"), OWN_STOPS, ids=[f"{n}-{r}" for n, _, r in OWN_STOPS])
def test_the_models_own_stop_wins_at_the_same_iteration(name, extra, reason, monkeypatch):
    """A callback stop and a max_time stop at the iteration where the model's own rule stops give way to it."""
    cls, params = CASES[name]
    params, data = {**params, **extra}, make_panel(cls)
    infos = []
    reference = _fit(cls, params, data, callback=infos.append)
    assert reference.stop_reason_ == reason
    last = infos[-1]["iteration"]
    now = [0.0]
    monkeypatch.setattr(run_control, "perf_counter", lambda: now[0])

    def stop_at_last(info):
        if info["iteration"] == last:
            now[0] = 10.0  # past max_time
            return True
        return None

    model = _fit(cls, {**params, "max_time": 5.0}, data, callback=stop_at_last)  # no ConvergenceWarning
    assert model.stop_reason_ == reason and model.converged_ == reference.converged_
    _assert_same_fit(model, reference)
