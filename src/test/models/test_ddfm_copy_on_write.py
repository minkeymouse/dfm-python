"""DDFM under pandas copy-on-write (the default from pandas 3).

Under copy-on-write ``DataFrame.values`` is a read-only view, so DDFM's old
writes through it (``data_denoised.values[...] = ...`` in
``sampling.denoise_targets`` and ``data_imputed.values[missing] = ...`` in
``DDFMEngine._impute_missing_targets``) raised ``ValueError: assignment
destination is read-only`` and ``DDFM.fit`` failed on a fresh install. On
pandas 2 these tests switch copy-on-write on with ``mode.copy_on_write``; on
pandas 3 it is always on.
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pandas as pd
import pytest
import torch

from dfm_python import DDFM, PanelData
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.models.ddfm.sampling import denoise_targets

_PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3

TINY = {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 2,
        "random_state": 0, "device": "cpu"}


def _copy_on_write():
    """Copy-on-write on: the pandas 3 default, an option on pandas 2."""
    if _PANDAS_3:
        return contextlib.nullcontext()
    return pd.option_context("mode.copy_on_write", True)


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _frame(T: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.standard_normal((T, 4)), columns=["x0", "y0", "x1", "y1"])
    frame.iloc[[3, 4, 17], 1] = np.nan
    frame.iloc[[9, 30], 3] = np.nan
    return frame


def _panel(*, exog: bool, T: int = 48, seed: int = 0) -> PanelData:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    y = pd.DataFrame(rng.standard_normal((T, 3)), index=idx, columns=["s0", "s1", "s2"])
    y.iloc[[5, 6], 0] = np.nan
    y.iloc[20, 2] = np.nan
    x = pd.DataFrame(rng.standard_normal((T, 1)), index=idx, columns=["x0"]) if exog else None
    return PanelData(y=y, exog=x)


@pytest.mark.parametrize("covariates", [None, ["x0", "x1"]], ids=["all_targets", "covariates"])
def test_denoise_targets_writes_the_denoised_targets_in_place(covariates):
    frame = _frame()
    dataset = DDFMDataset(data=frame, time_idx="index", covariates=covariates)
    rng = np.random.default_rng(1)
    eps = 0.1 * rng.standard_normal((len(frame), len(dataset.target_series)))
    data_imputed = dataset.data.interpolate(limit_direction="both")
    data_denoised = dataset.data.copy()
    before = data_denoised.to_numpy(copy=True)
    lags_input = 0
    with _copy_on_write():
        _, Phi, _, _ = denoise_targets(
            eps=eps, data_imputed=data_imputed, data_denoised=data_denoised, dataset=dataset,
            lags_input=lags_input, interpolation_method="linear", interpolation_limit=10,
            interpolation_limit_direction="both",
        )
    after = data_denoised.to_numpy()
    ti = dataset.target_indices
    expected = data_imputed.to_numpy()[lags_input + 1:, ti] - eps[:-1, :] @ Phi
    # The frame passed in is updated in place, bitwise, on the target columns only.
    assert after[lags_input + 1:, ti].tobytes() == expected.tobytes()
    np.testing.assert_array_equal(after[: lags_input + 1], before[: lags_input + 1])
    others = np.setdiff1d(np.arange(before.shape[1]), ti)
    np.testing.assert_array_equal(after[:, others], before[:, others])


def _imputing_engine(frame: pd.DataFrame, covariates):
    dataset = DDFMDataset(data=frame, time_idx="index", covariates=covariates)
    data_imputed = dataset.data.interpolate(limit_direction="both").copy()
    return types.SimpleNamespace(_dataset=dataset, data_imputed=data_imputed,
                                 target_indices=dataset.target_indices)


def test_impute_missing_targets_fills_the_missing_targets_in_place():
    engine = _imputing_engine(_frame(), covariates=None)
    frame_before = engine.data_imputed
    before = engine.data_imputed.to_numpy(copy=True)
    missing = engine._dataset.missing_y
    y_pred = np.random.default_rng(2).standard_normal(before.shape)
    with _copy_on_write():
        DDFMEngine._impute_missing_targets(engine, y_pred)
    after = engine.data_imputed.to_numpy()
    assert engine.data_imputed is frame_before
    np.testing.assert_array_equal(after[missing], y_pred[missing])
    np.testing.assert_array_equal(after[~missing], before[~missing])


def test_impute_missing_targets_with_covariates():
    # Up to 0.8.1 the covariate branch wrote into a fancy-index copy, so the
    # missing targets were never imputed when covariates were given.
    engine = _imputing_engine(_frame(), covariates=["x0", "x1"])
    missing = engine._dataset.missing_y
    assert missing.any()
    before = engine.data_imputed.to_numpy(copy=True)
    y_pred = np.random.default_rng(3).standard_normal((len(before), len(engine._dataset.target_series)))
    with _copy_on_write():
        DDFMEngine._impute_missing_targets(engine, y_pred)
    after = engine.data_imputed.to_numpy()
    targets = engine._dataset.target_indices
    np.testing.assert_array_equal(after[:, targets][missing], y_pred[missing])
    np.testing.assert_array_equal(after[:, targets][~missing], before[:, targets][~missing])
    covariates = [i for i in range(after.shape[1]) if i not in targets]
    np.testing.assert_array_equal(after[:, covariates], before[:, covariates])


@pytest.mark.filterwarnings("error::UserWarning")  # e.g. torch.from_numpy on a read-only frame array
@pytest.mark.parametrize("exog", [False, True], ids=["all_targets", "exog"])
def test_ddfm_fit_and_predict_under_copy_on_write(exog):
    panel = _panel(exog=exog)
    longer = _panel(exog=exog, T=54)
    y = longer.y.iloc[48:].copy()
    y.iloc[2, 1] = np.nan  # update scales and interpolates the new rows in frames
    new = PanelData(y=y, exog=None if longer.exog is None else longer.exog.iloc[48:])
    with _copy_on_write():
        model = DDFM(**TINY, scaler="standard").fit(panel)
        outputs = [model.predict(3), model.predict_factors(3), model.predict(3, data=new)]
        model.update(new)
        outputs += [model.predict(3), model.factors_, model.loadings_, model.explained_variance_]
        outputs += [model.decompose()["common"], model.decompose(by="factor").drop(columns="idiosyncratic")]
    for out in outputs:
        assert np.isfinite(out.to_numpy()).all()


@pytest.mark.skipif(_PANDAS_3, reason="pandas 3 has no copy-on-write switch to compare against")
@pytest.mark.parametrize("exog", [False, True], ids=["all_targets", "exog"])
def test_copy_on_write_does_not_change_the_fit(exog):
    panel = _panel(exog=exog)
    with torch.random.fork_rng():
        plain = DDFM(**TINY, scaler="standard").fit(panel)
    with torch.random.fork_rng(), pd.option_context("mode.copy_on_write", True):
        cow = DDFM(**TINY, scaler="standard").fit(panel)
    assert plain.predict(3).to_numpy().tobytes() == cow.predict(3).to_numpy().tobytes()
    assert plain.factors_.to_numpy().tobytes() == cow.factors_.to_numpy().tobytes()


def test_to_tensor_layout_does_not_depend_on_copy_on_write():
    # A copied frame's ``.values`` is a Fortran-ordered view, read-only under copy-on-write.
    # to_tensor must give the same (C-contiguous) tensor layout in both modes, otherwise the
    # CPU kernels see differently strided inputs and the fits differ bitwise.
    from dfm_python.config.types import to_tensor

    frame = pd.DataFrame(np.random.default_rng(0).standard_normal((48, 4))).copy()
    strides = []
    for cow in (False, True):
        with pd.option_context("mode.copy_on_write", cow):
            view = frame.values
            strides.append(to_tensor(view, dtype=torch.float32).stride())
    assert strides[0] == strides[1] == (4, 1)
