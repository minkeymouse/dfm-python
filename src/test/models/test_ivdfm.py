"""Tests for the iVDFM estimator (dfm_python.iVDFM) on PanelData.

Ported from the old engine-API tests (``set_dataset``/``fit()``/``get_result()``,
``predict(horizon=...)`` as an ndarray, ``context_data``, single-file
``.pt`` checkpoints) to the estimator: ``fit(data)``, ``result_``,
``predict(h)`` as a DataFrame in data units, ``futr_exog``, exog columns as the
context u_t and directory save/load. The class ``TestiVDFM`` adds the
estimator checks of the AFM tests (pure keyword-only constructor, documented
defaults, tags, validation, no mutation, failed refit, verbosity, sklearn
machinery). The shipped YAMLs and the tutorials that use them are checked at
the end. Pure engine math is in test_ivdfm_engine.py; the item-level estimator
details (adapter, factor order, update, save layout) in
test_ivdfm_estimator.py.
"""

import ast
import contextlib
import importlib.util
import inspect
import io
import json
import re
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

import dfm_python
from dfm_python import PanelData, iVDFM
from dfm_python.config import iVDFMConfig, load_yaml
from dfm_python.config.schema.results import iVDFMResult
from dfm_python.models.ivdfm._engine import iVDFMEngine
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

_REPO = Path(__file__).resolve().parents[3]
_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None

# Small and fast: the old tests' 100 x 5 panel, 3 factors, windows of 20.
SMALL = {
    "n_factors": 3, "window": 20, "stride": 1, "encoder_hidden_dim": 16, "decoder_hidden_dim": 16,
    "prior_hidden_dim": 16, "batch_size": 8, "max_epochs": 2, "random_state": 0, "device": "cpu",
}


def _values(T=100, N=5, seed=42):
    return np.random.default_rng(seed).standard_normal((T, N))


def _panel(T=100, N=5, seed=42, exog=0, start="2000-01-03") -> PanelData:
    values = _values(T, N + exog, seed)
    index = pd.date_range(start, periods=T, freq="B", name="date")
    y = pd.DataFrame(values[:, :N], index=index, columns=[f"series_{i}" for i in range(N)])
    x = None
    if exog:
        x = pd.DataFrame(values[:, N:], index=index, columns=[f"context{k + 1}" for k in range(exog)])
    return PanelData(y=y, exog=x)


def _model(**params) -> iVDFM:
    return iVDFM(**{**SMALL, **params})


def _weights(model):
    return {k: v.detach().clone() for k, v in model.network_.state_dict().items()}


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def fitted() -> iVDFM:
    return _model().fit(_panel())


# -- ported from the old engine-API tests --------------------------------------


class TestiVDFMWorkflow:
    def test_fit_basic(self):
        data = _panel()
        model = _model()
        assert model.fit(data) is model
        assert isinstance(model.result_, iVDFMResult)
        assert model.result_.factors is not None and model.result_.innovations is not None
        assert model.n_iter_ == 2
        n_windows = 100 - 20 + 1  # stride 1
        assert model.window_factors_.shape == (n_windows * 20, 3) and model.factors_.shape == (100, 3)
        assert np.isfinite(model.window_factors_.to_numpy()).all() and np.isfinite(model.factors_.to_numpy()).all()

    def test_fit_with_dataframe(self):
        df = pd.DataFrame(_values(), columns=[f"series_{i}" for i in range(5)])
        model = _model().fit(df)  # a bare DataFrame is PanelData(y=df)
        assert model.n_features_in_ == 5 and model.network_.data_dim == 5
        assert list(model.feature_names_in_) == [f"series_{i}" for i in range(5)]
        assert isinstance(model.index_, pd.RangeIndex)

    def test_fit_with_exog_columns(self):
        """Context columns are exog: u_t is the time feature plus the exog columns."""
        values = np.random.default_rng(0).standard_normal((100, 4))
        df = pd.DataFrame(values, columns=["target1", "target2", "context1", "context2"])
        data = PanelData.from_wide(df, exog_cols=["context1", "context2"])
        model = _model().fit(data)
        assert model.network_.data_dim == 2  # targets only
        assert model.context_dim_ == 3 and model.network_.context_dim == 3  # time (1) + exog (2)
        assert list(model.exog_names_in_) == ["context1", "context2"]

    def test_fit_with_exog_arrays(self):
        """Array input: exog from columns of an array (the old context indices)."""
        values = _values()
        data = PanelData.from_arrays(values[:, 2:], exog=values[:, :2])
        model = _model().fit(data)
        assert model.network_.data_dim == 3
        assert model.context_dim_ == 3

    def test_predict_not_fitted(self):
        with pytest.raises(NotFittedError):
            _model().predict(5)

    def test_predict_basic(self, fitted):
        forecast = fitted.predict(5)
        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (5, 5)
        assert list(forecast.columns) == list(fitted.feature_names_in_)
        assert forecast.index[0] == _panel().index[-1] + pd.offsets.BDay(1)
        assert (forecast.dtypes == np.float64).all() and np.isfinite(forecast.to_numpy()).all()

    def test_predict_with_futr_exog(self):
        """Future context values (the old context_data) go in futr_exog, after the time features."""
        model = _model(time_context=2).fit(_panel(exog=1))
        assert model.context_dim_ == 3
        futr = np.random.default_rng(1).standard_normal((5, 1))
        forecast = model.predict(5, futr_exog=futr)
        assert forecast.shape == (5, 5) and np.isfinite(forecast.to_numpy()).all()
        named = pd.DataFrame(futr, columns=["context1"])
        assert model.predict(5, futr_exog=named).equals(forecast)

    def test_update_not_fitted(self):
        with pytest.raises(NotFittedError):
            _model().update(_panel(T=20))

    def test_update_basic(self):
        model = _model().fit(_panel())
        assert model.network_.factors.shape == (81, 20, 3)  # (windows, window, r) after training
        weights, result = _weights(model), model.result_
        new = _panel(T=30, seed=123, start="2000-05-22")
        assert model.update(new) is model
        # One factor per new window end (30 - 20 + 1 windows) appended to the window-length training path.
        assert model.network_.factors.shape == (20 + 11, 3)
        assert _same_weights(weights, _weights(model)) and model.result_ is result
        assert model.predict(5).index[0] == new.index[-1] + pd.offsets.BDay(1)

    def test_result_not_fitted(self):
        model = _model()
        assert not hasattr(model, "result_")
        with pytest.raises(NotFittedError):
            getattr(model, "result_")  # noqa: B009 (the attribute access is what raises)

    def test_result(self, fitted):
        result = fitted.result_
        assert result.factors is not None
        assert result.innovations is not None
        assert result.reconstructions is not None
        assert np.isfinite(result.training_elbo)
        assert result.num_epochs == fitted.n_iter_

    def test_save_and_load(self, fitted, tmp_path):
        fitted.save(tmp_path / "run")
        assert (tmp_path / "run" / "weights.pt").exists() and (tmp_path / "run" / "config.json").exists()
        loaded = iVDFM.load(tmp_path / "run")
        assert loaded.n_factors_ == fitted.n_factors_ == loaded.network_.latent_dim
        assert loaded.n_features_in_ == fitted.n_features_in_ == loaded.network_.data_dim
        assert loaded.context_dim_ == fitted.context_dim_ == loaded.network_.context_dim
        assert loaded.predict(5).equals(fitted.predict(5))
        assert loaded.get_params() == fitted.get_params()
        assert dfm_python.load(tmp_path / "run").predict(5).equals(fitted.predict(5))
        on_cpu = iVDFM.load(tmp_path / "run", device="cpu")
        assert all(p.device.type == "cpu" for p in on_cpu.network_.parameters())

    def test_single_file_checkpoint_is_rejected(self, fitted, tmp_path):
        path = tmp_path / "old_ivdfm_model.pt"
        torch.save({"model_state_dict": fitted.network_.state_dict()}, path)
        with pytest.raises(ConfigurationError):
            iVDFM.load(path)

    @pytest.mark.parametrize("factor_order", [1, 2, 3])
    def test_different_factor_orders(self, factor_order):
        model = _model(factor_order=factor_order).fit(_panel())
        assert model.network_.factor_order == factor_order
        assert model.result_.config["factor_order"] == factor_order
        assert np.isfinite(model.predict(3).to_numpy()).all()

    @pytest.mark.parametrize("distribution", ["laplace", "gaussian"])
    def test_different_innovation_distributions(self, distribution):
        model = _model(innovation_distribution=distribution).fit(_panel())
        assert model.network_.innovation_distribution == distribution
        assert np.isfinite(model.predict(3).to_numpy()).all()

    def test_data_mismatch_raises(self, fitted):
        """The old data_dim mismatch: new data must have the training series (and exog), by name."""
        fewer = PanelData(y=_panel().y.iloc[:, :4])
        renamed = PanelData(y=_panel().y.rename(columns={"series_0": "other"}))
        with_exog = _panel(exog=1)
        for data in (fewer, renamed, with_exog):
            with pytest.raises(DataValidationError):
                fitted.update(data)
            with pytest.raises(DataValidationError):
                fitted.predict(5, data)

    def test_full_workflow(self):
        """fit -> predict -> update -> predict -> result_."""
        model = _model()
        model.fit(_panel())
        assert model.predict(5).shape == (5, 5)
        new = _panel(T=30, seed=123, start="2000-05-22")
        model.update(new)
        forecast = model.predict(5)
        assert forecast.shape == (5, 5) and forecast.attrs["origin"] == new.index[-1]
        assert model.result_ is not None and model.result_.factors is not None


# -- estimator checks modelled on the AFM tests ---------------------------------


class TestiVDFM:
    def test_constructor_only_stores(self):
        torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()
        model = iVDFM(n_factors=3, window=20, random_state=0, device="cpu")
        assert torch.equal(torch_state, torch.random.get_rng_state())
        assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
        assert vars(model) == model.get_params()
        assert not hasattr(model, "result_") and not hasattr(model, "network_")

    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            iVDFM(3)
        params = inspect.signature(iVDFM.__init__).parameters
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in params.items() if name != "self")

    def test_defaults_are_the_documented_values(self):
        """Every ``name : type, default value`` line of the class docstring matches get_params()."""
        params = iVDFM().get_params()
        documented = {}
        # cleandoc: Python >= 3.13 strips the common indentation at compile time, earlier versions keep it.
        for line in inspect.cleandoc(iVDFM.__doc__).splitlines():
            match = re.match(r"^(\w+) : .*\bdefault (.+)$", line)
            if match:
                documented[match.group(1)] = ast.literal_eval(match.group(2))
        assert len(documented) >= 25
        for name, value in documented.items():
            assert params[name] == value, name
        # The spec's per-model seed default, and the device / verbosity conventions.
        assert params["random_state"] == 3 and params["device"] == "auto" and params["verbose"] == 0
        assert params["stride"] is None and params["window"] is None and params["scaler"] == "standard"

    def test_defaults_are_the_engine_defaults(self):
        """The defaults are what ran when an argument was omitted (the engine's own defaults)."""
        np_state = np.random.get_state()
        with torch.random.fork_rng():
            engine = iVDFMEngine()  # no dataset or dims: nothing is built
        np.random.set_state(np_state)
        params = iVDFM().get_params()
        names = {
            "n_factors": "latent_dim", "factor_order": "factor_order", "time_context": "time_context",
            "encoder_hidden_dim": "encoder_hidden_dim", "encoder_n_hidden_layers": "encoder_n_hidden_layers",
            "decoder_type": "decoder_type", "decoder_hidden_dim": "decoder_hidden_dim",
            "decoder_n_hidden_layers": "decoder_n_hidden_layers", "prior_hidden_dim": "prior_hidden_dim",
            "prior_n_hidden_layers": "prior_n_hidden_layers", "activation": "activation", "slope": "slope",
            "innovation_distribution": "innovation_distribution", "decoder_var": "decoder_var",
            "n_regimes": "num_regimes", "regime_temperature": "regime_temperature", "beta_kl": "beta_kl",
            "use_layer_norm": "use_layer_norm", "mixing": "mixing", "learning_rate": "learning_rate",
            "optimizer": "optimizer_type", "optimizer_weight_decay": "optimizer_weight_decay",
            "optimizer_momentum": "optimizer_momentum", "batch_size": "batch_size", "max_epochs": "max_epochs",
            "tol": "tolerance", "patience": "patience", "scheduler_type": "scheduler_type",
            "scheduler_step_size": "scheduler_step_size", "scheduler_gamma": "scheduler_gamma",
            "scheduler_patience": "scheduler_patience", "scheduler_factor": "scheduler_factor",
            "scheduler_min_lr": "scheduler_min_lr", "random_state": "seed",
        }
        for name, attr in names.items():
            assert params[name] == getattr(engine, attr), name

    def test_tags(self):
        tags = iVDFM.get_tags()
        assert tags["task"] == "forecast" and tags["backend"] == "torch"
        assert tags["capability:exog"] and tags["capability:update"] and tags["capability:device"]
        assert tags["capability:factor_forecast"] and tags["capability:missing_values"]
        assert tags["capability:interpret"] and not tags["capability:pred_int"]
        assert not tags["capability:series_exog"] and not tags["capability:mixed_frequency"]
        assert tags["update_kind"] == "encoder_forward" and tags["predict_data"] == "window"
        assert tags["non_deterministic"] is False  # the default random_state=3
        assert iVDFM().get_tags()["non_deterministic"] is False
        assert iVDFM(random_state=None).get_tags()["non_deterministic"] is True

    @pytest.mark.parametrize("params, name", [
        ({"n_factors": None}, "n_factors"),
        ({"n_factors": 0}, "n_factors"),
        ({"n_factors": 2.0}, "n_factors"),
        ({"window": 0}, "window"),
        ({"stride": -1}, "stride"),
        ({"time_context": 0}, "time_context"),
        ({"factor_order": 0}, "factor_order"),
        ({"n_regimes": 0}, "n_regimes"),
        ({"encoder_hidden_dim": 0}, "encoder_hidden_dim"),
        ({"decoder_n_hidden_layers": 0}, "decoder_n_hidden_layers"),
        ({"innovation_distribution": "cauchy"}, "innovation_distribution"),
        ({"decoder_type": "cnn"}, "decoder_type"),
        ({"decoder_var": float("nan")}, "decoder_var"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"optimizer": "RMSprop"}, "optimizer"),
        ({"tol": -1.0}, "tol"),
        ({"patience": 0}, "patience"),
        ({"scheduler_type": "linear"}, "scheduler_type"),
        ({"scheduler_gamma": 2.0}, "scheduler_gamma"),
        ({"mixing": 1}, "mixing"),
        ({"scaler": "zscore"}, "scaler"),
        ({"random_state": 1.5}, "random_state"),
        ({"verbose": -1}, "verbose"),
        ({"device": "nonsense"}, "device"),
    ])
    def test_invalid_parameters(self, params, name):
        model = _model(**params)
        with pytest.raises(ConfigurationError, match=name):
            model.fit(_panel(T=40))
        assert not hasattr(model, "result_")

    def test_failed_refit_clears_the_old_fit(self):
        model = _model().fit(_panel(T=40))
        assert hasattr(model, "result_")
        model.set_params(innovation_distribution="bogus")
        with pytest.raises(ConfigurationError, match="innovation_distribution"):
            model.fit(_panel(T=40))
        assert not hasattr(model, "result_") and not hasattr(model, "network_")
        with pytest.raises(NotFittedError):
            model.predict(3)

    def test_does_not_mutate_input(self):
        data = _panel(exog=2)
        y, exog = data.y.copy(), data.exog.copy()
        y_missing = data.y.copy()
        y_missing.iloc[[3, 7], 1] = np.nan
        with_missing = PanelData(y=y_missing, exog=data.exog)
        model = _model()
        params = model.get_params()
        model.fit(with_missing)
        model.predict(3, futr_exog=np.zeros((3, 2)))
        model.update(data)
        assert data.y.equals(y) and data.exog.equals(exog)
        assert with_missing.y.isna().sum().sum() == 2
        assert model.get_params() == params

    def test_unsupported_inputs(self):
        panel = _panel(T=40)
        mixed = PanelData(y=panel.y, freq={"series_0": "q"})
        with pytest.raises(UnsupportedInputError):
            _model().fit(mixed)
        model = _model().fit(panel)
        with pytest.raises(UnsupportedInputError, match="refit"):
            model.update(panel, refit=True)

    def test_device(self):
        model = _model().fit(_panel(T=40))
        assert model.device_ == torch.device("cpu")
        assert all(p.device.type == "cpu" for p in model.network_.parameters())
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            auto = _model(device="auto").fit(_panel(T=40))
        assert auto.device_ == torch.device("cpu")
        assert _model(device=torch.device("cpu")).fit(_panel(T=40)).device_ == torch.device("cpu")

    def test_verbose_progress(self, capsys):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _model(verbose=0).fit(_panel(T=40))
        assert out.getvalue() == "" and capsys.readouterr().out == ""
        with contextlib.redirect_stdout(out):
            _model(verbose=1).fit(_panel(T=40))
        assert "Epoch" in out.getvalue() + capsys.readouterr().out

    def test_verbose_leaves_the_numbers_unchanged(self):
        quiet = _model(verbose=0).fit(_panel(T=40))
        with contextlib.redirect_stdout(io.StringIO()):
            loud = _model(verbose=2).fit(_panel(T=40))
        assert _same_weights(_weights(quiet), _weights(loud))
        assert quiet.predict(4).equals(loud.predict(4))

    def test_torch_device_param_saves_and_writes_yaml(self, tmp_path):
        model = _model(device=torch.device("cpu")).fit(_panel(T=40))
        model.save(tmp_path / "run")
        assert json.loads((tmp_path / "run" / "config.json").read_text())["params"]["device"] == "cpu"
        loaded = iVDFM.load(tmp_path / "run")
        assert loaded.get_params()["device"] == "cpu" and loaded.device_ == torch.device("cpu")
        assert model.get_config().to_plain_dict()["device"] == "cpu"
        if _YAML:
            model.get_config().to_yaml(tmp_path / "ivdfm.yaml")
            assert iVDFMConfig.from_yaml(tmp_path / "ivdfm.yaml").device == "cpu"

    def test_sklearn_machinery(self):
        model = _model(encoder_hidden_dim=(16, 8))
        assert clone(model).get_params() == model.get_params()
        assert model.set_params(window=10) is model and model.window == 10
        with pytest.raises(ConfigurationError, match="n_factors"):
            model.set_params(num_factors=3)
        assert "iVDFM(" in repr(model)
        cfg = model.get_config()
        assert isinstance(cfg, iVDFMConfig) and cfg.to_dict() == model.get_params()
        assert iVDFM.from_config(cfg).get_params() == model.get_params()


# -- shipped YAMLs and tutorials -------------------------------------------------

_SHIPPED = {
    "macro": {
        "model": {"n_factors": 5, "window": 200, "factor_order": 3, "innovation_distribution": "laplace",
                  "decoder_var": 0.01, "encoder_hidden_dim": 200, "encoder_n_hidden_layers": 3,
                  "decoder_hidden_dim": 200, "decoder_n_hidden_layers": 3, "prior_hidden_dim": 100,
                  "prior_n_hidden_layers": 2, "activation": "lrelu", "slope": 0.1, "learning_rate": 0.001,
                  "optimizer": "Adam", "batch_size": 256, "max_epochs": 10, "tol": 1e-5, "scaler": "standard",
                  "random_state": 3},
        "data": {"time_col": "date", "target_cols": ["KOEQUIPTE", "KOEMPTOTO", "KOHWRWEMP"], "exog_cols": []},
        "csv": "macro.csv",
    },
    "finance": {
        "model": {"n_factors": 5, "window": 200, "factor_order": 2, "innovation_distribution": "laplace",
                  "decoder_var": 0.01, "encoder_hidden_dim": 50, "encoder_n_hidden_layers": 2,
                  "decoder_hidden_dim": 50, "decoder_n_hidden_layers": 2, "prior_hidden_dim": 100,
                  "prior_n_hidden_layers": 2, "activation": "lrelu", "slope": 0.1, "learning_rate": 0.001,
                  "optimizer": "Adam", "batch_size": 128, "max_epochs": 10, "tol": 1e-5, "scaler": "standard",
                  "random_state": 3},
        "data": {"time_col": "date", "target_cols": ["market_forward_excess_returns"], "exog_cols": []},
        "csv": "finance.csv",
    },
    "exchange": {
        "model": {"n_factors": 2, "window": 200, "factor_order": 1, "innovation_distribution": "laplace",
                  "decoder_var": 0.01, "encoder_hidden_dim": 200, "encoder_n_hidden_layers": 3,
                  "decoder_hidden_dim": 200, "decoder_n_hidden_layers": 3, "prior_hidden_dim": 100,
                  "prior_n_hidden_layers": 2, "activation": "lrelu", "slope": 0.1, "learning_rate": 0.001,
                  "optimizer": "Adam", "batch_size": 256, "max_epochs": 10, "tol": 1e-5, "scaler": None,
                  "random_state": 3},
        "data": {"time_col": "date",
                 "target_cols": ["AUD_USD", "GBP_USD", "CAD_USD", "CHF_USD", "CNY_USD", "JPY_USD", "SGD_USD",
                                 "NZD_USD"],
                 "exog_cols": []},
        "csv": "exchange_rate.csv",
    },
}


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_shipped_yaml_builds_the_estimator(name):
    path = _REPO / "config" / f"ivdfm_{name}.yaml"
    expected = {**iVDFM().get_params(), **_SHIPPED[name]["model"]}
    cfg = iVDFMConfig.from_yaml(path)
    assert cfg.to_dict() == expected
    assert iVDFM.from_config(cfg).get_params() == expected
    sections = load_yaml(path)
    assert sections["data"] == _SHIPPED[name]["data"]
    # Every column the data: section names is in the shipped CSV (finance.csv gets its 'date' in the tutorial).
    header = set(pd.read_csv(_REPO / "data" / _SHIPPED[name]["csv"], nrows=0).columns) | {"date"}
    data = sections["data"]
    assert set(data["target_cols"]) | set(data["exog_cols"]) <= header


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_tutorial_runs_on_truncated_data(name, tmp_path):
    """The tutorial runs end to end: YAML, PanelData.from_wide, fit, predict, predict_factors, save and load."""
    path = _REPO / "tutorial" / f"tutorial_{name}_ivdfm.py"
    spec = importlib.util.spec_from_file_location(f"_tutorial_{name}_ivdfm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.main(["--rows", "260", "--max-epochs", "1", "--device", "cpu", "--run-dir", str(tmp_path / "run")])
    text = out.getvalue()
    assert "6-step forecast (data units)" in text and "reloaded forecast is identical" in text
    loaded = iVDFM.load(tmp_path / "run")
    assert loaded.n_iter_ == 1 and loaded.get_params()["window"] == 200
    assert np.isfinite(loaded.predict(6).to_numpy()).all()
