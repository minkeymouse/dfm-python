"""Golden cases of AFM: the data recipes, the engine runs and the outputs each fixture pins."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._cases_common import (
    Case,
    CaseRun,
    force_cpu,
    _attrs,
    _digest,
    _np,
    _reset_global_rngs,
)


AFM_ATTRS = ["factor_model", "trading", "n_factors", "n_char", "embed_dim", "hist_len",
              "n_kernels", "ridge", "squash_lambda", "pca_window", "reestim", "lambda_var",
              "turnover_cost", "short_cost", "risk_free", "learning_rate", "max_epochs",
              "patience", "seed", "device"]


def afm_inputs(factor_model: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """The AFM case's returns (T, N), characteristics (T, N, M) or None, and data recipe."""
    # Same panel as src/test/models/test_afm.py::_make_data.
    T, N, M, seed = 60, 6, 8, 0
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, N)) * 0.01
    characteristics = rng.standard_normal((T, N, M))
    use_chars = factor_model == "attention"
    data = {
        "generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N, "M": M,
        "returns": "rng.standard_normal((T, N)) * 0.01",
        "characteristics": "rng.standard_normal((T, N, M)), drawn after returns",
        "characteristics_passed": use_chars,
        "dataset": "AFMDataset(returns=returns, characteristics=characteristics if passed else omitted)",
        "input_sha256": _digest(returns, characteristics) if use_chars else _digest(returns),
    }
    return returns, (characteristics if use_chars else None), data


def _afm_case(factor_model: str, trading: str, **overrides: Any) -> CaseRun:
    # The engine path: the pre-redesign AFM class, now the private AFMEngine,
    # called exactly as the old public constructor was (dataset in __init__).
    # test_golden_afm.py also runs the public AFM estimator against the same fixtures.
    # ``overrides`` replace the common keywords (the early-stopping case).
    from dfm_python.dataset.afm_dataset import AFMDataset
    from dfm_python.models.afm._engine import AFMEngine

    returns, characteristics, data = afm_inputs(factor_model)

    with force_cpu():
        _reset_global_rngs()
        ds = AFMDataset(returns=returns, characteristics=characteristics)
        kwargs = {"n_factors": 3, "hist_len": 10, "max_epochs": 5, "seed": 0, **overrides}
        model = AFMEngine(dataset=ds, factor_model=factor_model, trading=trading, **kwargs)
        hyper = _attrs(model, AFM_ATTRS)
        state = model.fit()
        weights = model.predict()["weights"]

    outputs = {
        "residuals": _np(state.residuals),
        "book_returns": _np(state.book_returns),
        # float() of a float32 tensor: stored as float32 (exact) so the fallback
        # tolerance matches the precision the value was computed in.
        "sharpe": np.float32(state.sharpe),
        "explained_variance": np.float32(state.explained_variance),
        "num_iter": np.int64(state.num_iter),
        "converged": np.bool_(state.converged),
        # The last epoch's objective, a Python float (float64 of the float32 loss).
        "loss_now": np.float64(state.loss_now),
        "predict_weights": _np(weights),
    }
    for k, v in model.state_dict().items():
        outputs[f"sd/{k}"] = _np(v)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


#: The AFM cases, in registry order.
CASES: list[Case] = [
    Case("afm_attention_longconv", "afm", lambda: _afm_case("attention", "longconv")),
    Case("afm_pca_longconv", "afm", lambda: _afm_case("pca", "longconv")),
    Case("afm_pca_ou", "afm", lambda: _afm_case("pca", "ou")),
    # patience stops training early (at epoch 17 of 40); the learning rate is
    # high enough that the loss stalls.
    Case("afm_pca_longconv_patience", "afm",
         lambda: _afm_case("pca", "longconv", max_epochs=40, learning_rate=0.1, patience=3)),
]
