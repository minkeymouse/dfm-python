"""``dfm_python.testing``: the dummies pass ``check_estimator`` and each broken dummy fails its check."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from dfm_python.base import BaseFactorModel
from dfm_python.config import EstimatorConfig
from dfm_python.dataset.panel import PanelData
from dfm_python.testing import check_estimator, make_panel

from ._dummies import (
    _DummyAllocator,
    _DummyForecaster,
    _DummyForecasterConfig,
    _DummyTorchForecaster,
)

_ALL_CHECKS = [
    "tags", "pure_construction", "parameters", "config_mirror", "config_serialization", "unknown_keys",
    "unfitted", "fit", "fit_attributes", "output_shape", "purity", "capabilities", "silence",
]


# ---------------------------------------------------------------------------
# Passing estimators
# ---------------------------------------------------------------------------


def test_dummy_forecaster_passes():
    assert check_estimator(_DummyForecaster) == [*_ALL_CHECKS, "update", "save_load"]


def test_dummy_torch_forecaster_passes_with_device_check():
    ran = check_estimator(_DummyTorchForecaster)
    assert "device" in ran and "update" not in ran and ran[-1] == "save_load"


def test_dummy_allocator_passes():
    assert check_estimator(_DummyAllocator) == [*_ALL_CHECKS, "update", "save_load"]


def test_instance_is_checked_with_its_own_params():
    assert "fit" in check_estimator(_DummyForecaster(shrink=0.5, window=4))


def test_explicit_dataframe_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((30, 2)), index=pd.RangeIndex(30, name="t"), columns=["a", "b"])
    assert "output_shape" in check_estimator(_DummyForecaster, data=df)


def test_rejects_non_estimators():
    with pytest.raises(TypeError, match="BaseFactorModel"):
        check_estimator(object)


def test_save_load_check_runs_last():
    assert check_estimator(_DummyForecaster)[-1] == "save_load"


# ---------------------------------------------------------------------------
# make_panel
# ---------------------------------------------------------------------------


def test_make_panel_default_slots():
    panel = make_panel({})
    assert isinstance(panel, PanelData)
    assert panel.n_periods == 48 and panel.series_names == ("s0", "s1", "s2")
    assert panel.exog_names == () and panel.characteristic_names == () and panel.freq is None
    assert not panel.y.isna().to_numpy().any()
    assert isinstance(panel.index, pd.DatetimeIndex) and panel.index.name == "date"


def test_make_panel_follows_capabilities():
    tags = {
        "capability:exog": True,
        "capability:series_exog": True,
        "capability:mixed_frequency": True,
        "capability:missing_values": True,
    }
    panel = make_panel(tags)
    assert panel.exog_names == ("x0", "x1")
    assert panel.characteristic_names == ("c0", "c1")
    assert panel._is_mixed_frequency() and panel.freq["s0"] == "q"
    y = panel.y
    assert y["s0"].isna().sum() == 32  # observed at quarter ends only
    assert y["s2"].isna().sum() == 2


def test_make_panel_accepts_a_class_and_is_seeded_locally():
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    a = make_panel(_DummyAllocator, seed=3)
    b = make_panel(_DummyAllocator.get_tags(), seed=3)
    assert a._equals(b) and not a._equals(make_panel(_DummyAllocator, seed=4))
    assert a.characteristic_names == ("c0", "c1")
    assert np.abs(a.y.to_numpy()).max() < 0.1  # return-sized
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])


# ---------------------------------------------------------------------------
# Broken estimators: each one fails exactly the check it breaks
# ---------------------------------------------------------------------------


def _fails(estimator, check: str):
    with pytest.raises(AssertionError, match=rf"check '{check}' failed"):
        check_estimator(estimator)


class _BadTags(_DummyForecaster):
    _tags = {"capability:update": False}  # update_kind is still set


class _DrawsInInit(_DummyForecaster):
    def __init__(self, *, shrink=0.0, window=None, lags=(1,), verbose=0):
        torch.rand(1)
        super().__init__(shrink=shrink, window=window, lags=lags, verbose=verbose)


class _NumpySeedsInInit(_DummyForecaster):
    def __init__(self, *, shrink=0.0, window=None, lags=(1,), verbose=0):
        np.random.standard_normal()
        super().__init__(shrink=shrink, window=window, lags=lags, verbose=verbose)


class _ExtraAttrInInit(_DummyForecaster):
    def __init__(self, *, shrink=0.0, window=None, lags=(1,), verbose=0):
        super().__init__(shrink=shrink, window=window, lags=lags, verbose=verbose)
        self.cache = {}


class _ConvertsInInit(_DummyForecaster):
    @classmethod
    def get_test_params(cls):
        return [{"lags": (1, 2)}]

    def __init__(self, *, shrink=0.0, window=None, lags=(1,), verbose=0):
        super().__init__(shrink=shrink, window=window, lags=tuple(list(lags)), verbose=verbose)


class _PositionalInit(_DummyForecaster):
    def __init__(self, shrink=0.0, *, window=None, lags=(1,), verbose=0):
        super().__init__(shrink=shrink, window=window, lags=lags, verbose=verbose)


@dataclass
class _MissingFieldConfig(EstimatorConfig):
    shrink: float = 0.0
    lags: tuple = (1,)
    verbose: int = 0


class _MissingConfigField(_DummyForecaster):
    _config_class = _MissingFieldConfig


@dataclass
class _WrongDefaultConfig(_DummyForecasterConfig):
    shrink: float = 0.5


class _WrongConfigDefault(_DummyForecaster):
    _config_class = _WrongDefaultConfig


class _NoConfig(_DummyForecaster):
    _config_class = None


@dataclass
class _LossyConfig(_DummyForecasterConfig):
    def to_plain_dict(self):
        return {**super().to_plain_dict(), "shrink": round(self.shrink, 1)}


class _LossySerialization(_DummyForecaster):
    _config_class = _LossyConfig

    @classmethod
    def get_test_params(cls):
        return [{"shrink": 0.25}]


@dataclass
class _LenientConfig(_DummyForecasterConfig):
    @classmethod
    def from_dict(cls, d):
        names = cls.field_names()
        return super().from_dict({k: v for k, v in d.items() if k in names})


class _LenientKeys(_DummyForecaster):
    _config_class = _LenientConfig


class _ResultBeforeFit(_DummyForecaster):
    result_ = None


class _PredictBeforeFitReturns(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        if not self._is_fitted():
            return None
        return super().predict(h, data=data)


class _FitReturnsNone(_DummyForecaster):
    def fit(self, data, **fit_kwargs):
        super().fit(data, **fit_kwargs)


class _NotRepeatable(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        self.result_.scale = float(np.random.standard_normal())


class _DeclaredNonDeterministic(_NotRepeatable):
    _tags = {"non_deterministic": True}


class _NonDeterministicUnlessShrunk(_NotRepeatable):
    """Declares non_deterministic per instance (like AFM with random_state=None)."""

    @classmethod
    def get_test_params(cls):
        return [{}]

    def _instance_tags(self):
        return {"non_deterministic": self.shrink == 0.0}


class _PublicAttrInFit(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        self.cache = 1


class _MutatesParams(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        self.lags = (9,)


class _MutatesData(_DummyForecaster):
    @classmethod
    def get_test_params(cls):
        return [{"window": 5}]

    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        data._y.iloc[0, 0] += 1.0  # outside the window, so the fit itself is unaffected


class _NoFittedDataAttributes(_DummyForecaster):
    def _set_data_attributes(self, data):
        self.index_ = data.index


class _WrongLength(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        frame = super().predict(h, data=data)
        return pd.concat([frame, frame.iloc[-1:]])


class _WrongColumns(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        return super().predict(h, data=data).iloc[:, ::-1]


class _MissingAttrs(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        frame = super().predict(h, data=data)
        frame.attrs = {}
        return frame


class _AcceptsFloatHorizon(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        return super().predict(int(h), data=data)


class _ImpurePredict(_DummyForecaster):
    def predict(self, h, data=None, **predict_kwargs):
        frame = super().predict(h, data=data)
        self.result_.n_obs += 1
        return frame


class _IgnoresExog(_DummyForecaster):
    def _validate_data(self, data):
        return PanelData._coerce(data)


class _PrintsInFit(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        print("epoch 1")


class _LogsInFit(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        logging.getLogger("dfm_python.test_dummy").info("epoch 1")


class _LogsDebugInFit(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        logging.getLogger("dfm_python.test_dummy").debug("epoch 1")


class _UpdateReturnsNone(_DummyForecaster):
    def update(self, data, *, refit=False):
        out = super().update(data, refit=refit)
        return out if refit else None


class _UpdateKeepsOrigin(_DummyForecaster):
    def update(self, data, *, refit=False):
        self._check_is_fitted("update")
        self._validate_data(data)
        return self


class _RefitReturnsNone(_DummyForecaster):
    def update(self, data, *, refit=False):
        out = super().update(data, refit=refit)
        return None if refit else out


class _WrongDevice(_DummyTorchForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        self.device_ = torch.device("meta")


class _UpdateChangesNetwork(_DummyTorchForecaster):
    _tags = {"capability:update": True, "update_kind": "encoder_forward"}

    def update(self, data, *, refit=False):
        self._check_is_fitted("update")
        panel = self._validate_data(data)
        with torch.no_grad():
            self.network_.bias.add_(1.0)
        self.last_ = panel.y.to_numpy()[-1].copy()
        self.index_ = panel.index
        return self


class _NoSaveHooks(_DummyForecaster):
    _save_state = BaseFactorModel._save_state


class _LossyLoad(_DummyForecaster):
    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        super()._load_state(est, path, device, allow_pickle)
        est.result_.scale = float(np.float32(est.result_.scale))


class _PredictDiffersAfterLoad(_DummyForecaster):
    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        super()._load_state(est, path, device, allow_pickle)
        est.mean_ = est.mean_ + 1e-12


class _ExtraFile(_DummyForecaster):
    def _save_state(self, path):
        super()._save_state(path)
        (path / "notes.txt").write_text("not in the layout")


class _PicklesKnownScaler(_DummyForecaster):
    def _fit(self, data, **fit_kwargs):
        super()._fit(data)
        self.scaler_ = StandardScaler().fit(data.y.to_numpy())

    def _save_state(self, path):
        scaler = self.scaler_
        del self.scaler_
        try:
            super()._save_state(path)
        finally:
            self.scaler_ = scaler
        with open(path / "scaler.pkl", "wb") as fh:
            pickle.dump(scaler, fh)


class _TorchSkipsWeights(_DummyTorchForecaster):
    def _write_weights(self, path, module=None):
        pass


class _LoadIgnoresDevice(_DummyTorchForecaster):
    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        super()._load_state(est, path, device, allow_pickle)
        est.device_ = torch.device("meta")


class _SaveMutatesState(_DummyForecaster):
    def _save_state(self, path):
        super()._save_state(path)
        self.result_.n_obs += 1


class _AllocatorWrongIndex(_DummyAllocator):
    def allocate(self, data=None):
        return super().allocate(data).iloc[1:]


class _AllocatorIgnoresSeriesExogTag(_DummyAllocator):
    _tags = {"capability:series_exog": False}

    def _validate_data(self, data):
        panel = PanelData._coerce(data)
        super()._validate_data(PanelData(panel.y, exog=panel.exog, freq=panel.freq))  # every check but series_exog
        return panel


@pytest.mark.parametrize(
    "estimator, check",
    [
        (_BadTags, "tags"),
        (_DrawsInInit, "pure_construction"),
        (_NumpySeedsInInit, "pure_construction"),
        (_ExtraAttrInInit, "pure_construction"),
        (_ConvertsInInit, "pure_construction"),
        (_PositionalInit, "parameters"),
        (_MissingConfigField, "config_mirror"),
        (_WrongConfigDefault, "config_mirror"),
        (_NoConfig, "config_mirror"),
        (_LossySerialization, "config_serialization"),
        (_LenientKeys, "unknown_keys"),
        (_ResultBeforeFit, "unfitted"),
        (_PredictBeforeFitReturns, "unfitted"),
        (_FitReturnsNone, "fit"),
        (_NotRepeatable, "fit"),
        (_PublicAttrInFit, "fit_attributes"),
        (_MutatesParams, "fit_attributes"),
        (_MutatesData, "fit_attributes"),
        (_NoFittedDataAttributes, "fit_attributes"),
        (_WrongLength, "output_shape"),
        (_WrongColumns, "output_shape"),
        (_MissingAttrs, "output_shape"),
        (_AcceptsFloatHorizon, "output_shape"),
        (_ImpurePredict, "purity"),
        (_IgnoresExog, "capabilities"),
        (_RefitReturnsNone, "capabilities"),
        (_PrintsInFit, "silence"),
        (_LogsInFit, "silence"),
        (_UpdateReturnsNone, "update"),
        (_UpdateKeepsOrigin, "update"),
        (_WrongDevice, "device"),
        (_UpdateChangesNetwork, "update"),
        (_AllocatorWrongIndex, "output_shape"),
        (_AllocatorIgnoresSeriesExogTag, "capabilities"),
        (_NoSaveHooks, "save_load"),
        (_LossyLoad, "save_load"),
        (_PredictDiffersAfterLoad, "save_load"),
        (_ExtraFile, "save_load"),
        (_PicklesKnownScaler, "save_load"),
        (_TorchSkipsWeights, "save_load"),
        (_LoadIgnoresDevice, "save_load"),
        (_SaveMutatesState, "save_load"),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else v,
)
def test_broken_estimator_fails_its_check(estimator, check):
    _fails(estimator, check)


def test_capability_failure_names_the_slot():
    with pytest.raises(AssertionError, match="fit with characteristics while capability:series_exog is False"):
        check_estimator(_AllocatorIgnoresSeriesExogTag)


def test_declared_non_deterministic_skips_repeatability():
    assert "fit" in check_estimator(_DeclaredNonDeterministic)


def test_checks_use_the_instance_tags():
    assert _NonDeterministicUnlessShrunk.get_tags()["non_deterministic"] is False
    assert "fit" in check_estimator(_NonDeterministicUnlessShrunk)  # shrink=0.0: non-deterministic
    with pytest.raises(AssertionError, match="check 'fit' failed"):
        check_estimator(_NonDeterministicUnlessShrunk(shrink=0.5))  # declared deterministic, but is not


def test_debug_logging_is_not_noise():
    assert "silence" in check_estimator(_LogsDebugInFit)


def test_failure_message_names_params_and_cause():
    with pytest.raises(AssertionError) as info:
        check_estimator(_MutatesParams)
    message = str(info.value)
    assert "_MutatesParams()" in message and "lags" in message


def test_check_restores_package_logger():
    package_logger = logging.getLogger("dfm_python")
    handlers, level = list(package_logger.handlers), package_logger.level
    check_estimator(_DummyForecaster)
    assert package_logger.handlers == handlers and package_logger.level == level


@pytest.mark.parametrize(
    "estimator, message",
    [
        (_NoSaveHooks, "does not implement _save_state"),
        (_LossyLoad, "result_.scale is not bitwise identical"),
        (_PredictDiffersAfterLoad, "predict after _PredictDiffersAfterLoad.load: values are not bitwise identical"),
        (_ExtraFile, "files outside the layout: ['notes.txt']"),
        (_PicklesKnownScaler, "pickled a StandardScaler"),
        (_TorchSkipsWeights, "no weights.pt"),
        (_LoadIgnoresDevice, "device_=device(type='meta')"),
        (_SaveMutatesState, "save changed the fitted state"),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_save_load_failure_messages(estimator, message):
    with pytest.raises(AssertionError) as info:
        check_estimator(estimator)
    assert "check 'save_load' failed" in str(info.value)
    assert message in str(info.value), str(info.value)
