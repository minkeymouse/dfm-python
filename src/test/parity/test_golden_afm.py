"""Golden tests for AFM.

``test_golden_afm_engine`` runs the engine (the pre-redesign class, now
``AFMEngine``) exactly as the fixtures were captured. ``test_golden_afm_estimator``
runs the public ``AFM`` estimator on the same data with the recorded
hyperparameters (``seed`` -> ``random_state``, ``device="cpu"``), built from the
constructor and from a YAML file, and must reproduce the same numbers.
``test_golden_afm_save_load`` fits the estimator, saves it, loads it back
(``AFM.load`` and ``dfm_python.load``) and requires the loaded model to report
the same numbers: ``allocate()`` bitwise equal to the fitted model's, and every
golden output reproduced from the loaded model alone.
"""

import importlib.util

import numpy as np
import pytest

import dfm_python
from dfm_python import AFM, PanelData
from dfm_python.config import AFMConfig

from ._cases import cases_for, check_case
from ._cases_afm import AFM_ATTRS, afm_inputs
from ._cases_common import _attrs, _reset_global_rngs, normalized
from ._golden import assert_golden, load_golden

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the recorded engine hyperparameters."""
    params = {k: v for k, v in hyper.items() if k not in ("seed", "n_char", "device")}
    params["random_state"] = hyper["seed"]
    params["device"] = "cpu"
    return params


def _outputs(model: AFM, weights) -> dict:
    """The golden outputs of a fitted (or loaded) estimator and its ``allocate`` weights."""
    outputs = {
        "residuals": model.result_.residual_portfolios,
        "book_returns": model.result_.book_returns,
        "sharpe": np.float32(model.result_.sharpe),
        "explained_variance": np.float32(model.result_.explained_variance),
        "num_iter": np.int64(model.n_iter_),
        "converged": np.bool_(model.converged_),
        "loss_now": np.float64(model.network_.training_state.loss_now),
        # float32 -> float64 -> float32 is exact, so this stays a bitwise comparison.
        "predict_weights": weights.to_numpy().astype(np.float32)[None],
    }
    for key, value in model.network_.state_dict().items():
        outputs[f"sd/{key}"] = value.detach().cpu().numpy()
    return outputs


def _build(build: str, params: dict, tmp_path) -> AFM:
    if build == "constructor":
        return AFM(**params)
    if not _YAML:
        pytest.skip("YAML needs omegaconf or PyYAML")
    path = tmp_path / "afm.yaml"
    AFMConfig(**params).to_yaml(path)
    return AFM.from_config(AFMConfig.from_yaml(path))


@pytest.mark.parametrize("build", ["constructor", "yaml"])
@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm_estimator(case, build, pinned, tmp_path):
    manifest, _ = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    returns, characteristics, recipe = afm_inputs(hyper["factor_model"])
    assert normalized(recipe) == manifest["data"]

    model = _build(build, params, tmp_path)
    assert model.get_params() == {**AFM().get_params(), **params}
    data = PanelData.from_arrays(returns, characteristics=characteristics)

    _reset_global_rngs()
    assert model.fit(data) is model
    weights = model.allocate()

    # Every hyperparameter reached the engine unchanged.
    assert normalized(_attrs(model.network_, AFM_ATTRS)) == hyper
    assert model.n_char_ == hyper["n_char"]
    assert all(dtype == np.float64 for dtype in weights.dtypes)

    assert_golden(_outputs(model, weights), case.name)

    # The derived attributes are the same numbers in float64.
    np.testing.assert_array_equal(model.residuals_.to_numpy(), model.result_.residual_portfolios[0].astype(np.float64))
    np.testing.assert_array_equal(model.book_returns_.to_numpy(), model.result_.book_returns[0].astype(np.float64))


@pytest.mark.parametrize("loader", ["AFM.load", "dfm_python.load"])
@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm_save_load(case, loader, pinned, tmp_path):
    manifest, _ = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    returns, characteristics, _ = afm_inputs(hyper["factor_model"])
    data = PanelData.from_arrays(returns, characteristics=characteristics)

    _reset_global_rngs()
    model = AFM(**_estimator_params(hyper)).fit(data)
    weights = model.allocate()
    model.save(tmp_path / "run")
    del model

    loaded = (AFM.load if loader == "AFM.load" else dfm_python.load)(tmp_path / "run", device="cpu")
    assert type(loaded) is AFM
    reloaded = loaded.allocate(data=data)
    assert reloaded.index.equals(weights.index) and list(reloaded.columns) == list(weights.columns)
    assert reloaded.to_numpy().tobytes() == weights.to_numpy().tobytes()

    # The loaded model alone reproduces the golden numbers, and its engine holds the recorded hyperparameters.
    assert_golden(_outputs(loaded, reloaded), case.name)
    assert normalized(_attrs(loaded.network_, AFM_ATTRS)) == hyper
