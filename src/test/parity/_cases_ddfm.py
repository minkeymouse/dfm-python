"""Golden cases of DDFM: the data recipes, the engine runs and the outputs each fixture pins."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._cases_common import (
    Case,
    CaseRun,
    force_cpu,
    REPO_ROOT,
    _attrs,
    _class_name,
    _digest,
    _file_digest,
    _macro_recipe,
    _macro_tail,
    _np,
    _reset_global_rngs,
)


_DDFM_ATTRS = ["encoder_size", "decoder_size", "decoder_type", "activation", "learning_rate",
               "optimizer_type", "n_mc_samples", "window_size", "max_epoch_pre_train", "max_iter",
               "tolerance", "disp", "min_target_interporate_ratio", "interpolation_method",
               "interpolation_limit", "interpolation_limit_direction", "num_series", "input_dim",
               "output_dim", "initializer_seed", "lags_input", "device"]
_DDFM_ROWS = 400
_DDFM_MAX_ITER = 3
_DDFM_SEED = 3
_DDFM_HORIZON = 6


def _ddfm_hyper(model: Any) -> dict[str, Any]:
    hyper = _attrs(model, _DDFM_ATTRS)
    hyper["scaler"] = _class_name(model.scaler)
    return hyper


def ddfm_engine_outputs(model: Any) -> dict[str, np.ndarray]:
    """The fitted engine's numbers: ``predict(horizon=6)`` (no data, so no update), then :func:`_ddfm_fit_outputs`.

    ``model`` is an engine after ``fit`` and ``build_state_space``: the one a
    case builds, or the ``network_`` of a fitted DDFM estimator.
    """
    X, Z = model.predict(horizon=_DDFM_HORIZON)
    return _ddfm_fit_outputs(model, X, Z)


def _ddfm_fit_outputs(model: Any, X: Any, Z: Any) -> dict[str, np.ndarray]:
    """The fitted engine's numbers, read after ``fit``, ``build_state_space`` and ``predict``."""
    st = model.training_state
    result = model.get_result()
    outputs = {
        "factors": _np(model.factors),
        "eps": _np(model.eps),
        "F": _np(st.F), "H": _np(st.H), "Q": _np(st.Q), "R": _np(st.R),
        "mu_0": _np(st.mu_0), "Sigma_0": _np(st.Sigma_0),
        "num_iter": np.int64(model._num_iter),
        "predict_X": _np(X), "predict_Z": _np(Z),
        # Set from the second MCMC iteration on (max_iter=3 reaches it).
        "loss_now": np.float64(model.loss_now),
        "converged": np.bool_(getattr(model, "_converged", False)),
        "last_neurons": _np(model.last_neurons),
        # result C is not pinned: it is H (pinned above). It was H transposed
        # when the fixtures were captured; the estimator uses the DFM orientation.
        "result_Z": _np(result.Z),
        "result_x_sm": _np(result.x_sm),
        # The training dataset's target scaler (copied: an update must not change it).
        "scaler_mean": np.array(model.scaler.mean_, copy=True),
        "scaler_scale": np.array(model.scaler.scale_, copy=True),
    }
    for k, v in model.state_dict().items():
        outputs[f"sd/{k}"] = _np(v)
    return outputs


#: Outputs of the update path (``_ddfm_update_outputs``), which mutates the engine.
DDFM_UPDATE_KEYS = ("update_predict_X", "update_predict_Z", "update_factors", "update_scaler_mean",
                    "update_scaler_scale")


def _ddfm_update_outputs(model: Any, new_rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """``_update(new_rows)`` then ``predict(horizon=6)``: mutates the model.

    ``new_rows`` are the rows after the fitted ones, in data units, with the
    engine's columns ``[covariates | targets]``. The update scales them with
    the scalers fitted at fit time (``update_scaler_*`` pin that the target
    scaler is unchanged), interpolates them against the training rows,
    encodes them and appends their factors; the forecast then conditions on
    the last appended factor row. (Before the 0.8.0 release fix, the update
    took the whole panel, training rows included, and refitted the target
    scaler on it; the update outputs were recaptured then.)
    """
    model._update(new_rows)
    X_u, Z_u = model.predict(horizon=_DDFM_HORIZON)
    return {
        "update_predict_X": _np(X_u),
        "update_predict_Z": _np(Z_u),
        "update_factors": _np(model.factors),
        "update_scaler_mean": np.array(model.scaler.mean_, copy=True),
        "update_scaler_scale": np.array(model.scaler.scale_, copy=True),
    }


# The values of the flat config/ddfm_macro.yaml (commit 5d19c19) that the
# ddfm_macro fixture was captured with (at 0add6b5, through the old
# DDFMConfig.from_dict): the keys the case reads. The shipped file is now in
# the model:/data: layout with the canonical names, and DDFMConfig is the
# estimator's mirror, so the case keeps its captured values here.
# The data recipe still names the file the values came from.
_DDFM_MACRO_FLAT_CONFIG: dict[str, Any] = {
    "encoder_layers": [64, 32, 2], "activation": "relu", "learning_rate": 0.001, "n_mc_samples": 1,
    "window_size": 100, "tolerance": 0.0005, "disp": 10, "interpolation_method": "linear",
    "interpolation_limit": 10, "interpolation_limit_direction": "both",
}


def _ddfm_macro() -> CaseRun:
    from sklearn.preprocessing import StandardScaler

    from dfm_python.models.ddfm._engine import DDFMEngine as DDFM
    from dfm_python.dataset.ddfm_dataset import DDFMDataset

    # The old tutorial_macro_ddfm.py on the last rows of data/macro.csv, with a few MCMC
    # iterations and an explicit seed (the YAML's seed is null, which would not
    # be reproducible).
    df = _macro_tail(_DDFM_ROWS)
    target = "KOEQUIPTE"
    covariates = [c for c in df.columns if c not in ("date", target)]
    data = _macro_recipe(rows=f"last {_DDFM_ROWS} rows", target=target, covariates=covariates,
                         config="config/ddfm_macro.yaml", input_sha256=_digest(df))

    with force_cpu():
        _reset_global_rngs()
        config = dict(_DDFM_MACRO_FLAT_CONFIG)
        dataset = DDFMDataset(data=df, time_idx="date", covariates=covariates, scaler=StandardScaler())
        # The capture also passed the config object, which the engine read only
        # for the interpolation values (the engine has no config parameter
        # now); they are passed explicitly here.
        model = DDFM(
            dataset=dataset,
            encoder_size=tuple(config["encoder_layers"]), decoder_type="linear",
            activation=config["activation"], learning_rate=config["learning_rate"], optimizer="Adam",
            n_mc_samples=config["n_mc_samples"], window_size=config["window_size"],
            max_iter=_DDFM_MAX_ITER, tolerance=config["tolerance"], disp=config["disp"], seed=_DDFM_SEED,
            interpolation_method=config["interpolation_method"],
            interpolation_limit=config["interpolation_limit"],
            interpolation_limit_direction=config["interpolation_limit_direction"],
        )
        hyper = _ddfm_hyper(model)
        model.fit()
        model.build_state_space()
        outputs = ddfm_engine_outputs(model)
        # No update path: data/macro.csv has no rows after the 400 fitted ones,
        # and an update takes only rows after the conditioning data. (The
        # update outputs captured before the 0.8.0 release fix re-encoded the
        # same 400 rows, which update now rejects.)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_DDFM_TRUNC_ROWS = 300
_DDFM_HOLDOUT = 12
EXCHANGE_CSV = REPO_ROOT / "data" / "exchange_rate.csv"
FINANCE_CSV = REPO_ROOT / "data" / "finance.csv"


def _ddfm_rows_recipe() -> str:
    return (f"last {_DDFM_TRUNC_ROWS} rows; fit on the first {_DDFM_TRUNC_ROWS - _DDFM_HOLDOUT}, "
            f"update with the last {_DDFM_HOLDOUT}")


def _ddfm_fit_and_update(panel: pd.DataFrame, time_idx: str, covariates: list[str] | None,
                         **model_kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit DDFM on ``panel`` minus the last rows, then update with those rows."""
    from sklearn.preprocessing import StandardScaler

    from dfm_python.models.ddfm._engine import DDFMEngine as DDFM
    from dfm_python.dataset.ddfm_dataset import DDFMDataset

    fit_df = panel.iloc[:-_DDFM_HOLDOUT]
    with force_cpu():
        _reset_global_rngs()
        dataset = DDFMDataset(data=fit_df, time_idx=time_idx, covariates=covariates,
                              scaler=StandardScaler())
        model = DDFM(dataset=dataset, **model_kwargs)
        hyper = _ddfm_hyper(model)
        model.fit()
        model.build_state_space()
        outputs = ddfm_engine_outputs(model)
        new_rows = panel.iloc[-_DDFM_HOLDOUT:].reset_index(drop=True)[list(model.data.columns)]
        outputs.update(_ddfm_update_outputs(model, new_rows))
    return outputs, hyper


def ddfm_exchange_panel() -> pd.DataFrame:
    """The last rows of data/exchange_rate.csv, read as the old tutorial_exchange_ddfm.py read it."""
    df = pd.read_csv(EXCHANGE_CSV, index_col=0, parse_dates=True)
    return df.iloc[-_DDFM_TRUNC_ROWS:]


def _ddfm_exchange() -> CaseRun:
    # The old tutorial_exchange_ddfm.py with the values of the flat
    # config/ddfm_exchange.yaml it read at capture written inline, a few MCMC
    # iterations and seed 3. Every column is a target.
    panel = ddfm_exchange_panel()
    data = {
        "source": "data/exchange_rate.csv", "source_sha256": _file_digest(EXCHANGE_CSV),
        "read": "pd.read_csv(path, index_col=0, parse_dates=True)",
        "rows": _ddfm_rows_recipe(),
        "dataset": "DDFMDataset(data=frame, time_idx='index', scaler=StandardScaler())",
        "targets": list(panel.columns), "covariates": [],
        "fit_input_sha256": _digest(panel.iloc[:-_DDFM_HOLDOUT]),
        "update_input_sha256": _digest(panel.iloc[-_DDFM_HOLDOUT:]),
    }
    outputs, hyper = _ddfm_fit_and_update(
        panel, "index", None,
        encoder_size=(16, 4), decoder_type="linear", activation="relu", learning_rate=0.005,
        optimizer="Adam", n_mc_samples=10, window_size=200, max_iter=_DDFM_MAX_ITER,
        tolerance=0.0005, disp=10, seed=_DDFM_SEED, interpolation_method="linear",
        interpolation_limit=10, interpolation_limit_direction="both",
    )
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_FINANCE_TARGET = "market_forward_excess_returns"


def ddfm_finance_panel() -> tuple[pd.DataFrame, list[str]]:
    """The last rows of data/finance.csv as tutorial_finance_ddfm.py prepares it, and its covariates.

    A daily ``date`` column from 1980-01-01 is added over the whole file,
    ``forward_returns`` and ``risk_free_rate`` are dropped, and every other
    non-date column (``date_id`` included) is a covariate of the target.
    """
    df = pd.read_csv(FINANCE_CSV)
    df["date"] = pd.date_range(start="1980-01-01", periods=len(df), freq="D")
    df = df.drop(columns=["forward_returns", "risk_free_rate"])
    df = df.iloc[-_DDFM_TRUNC_ROWS:]
    covariates = [c for c in df.columns if c not in ("date", _FINANCE_TARGET)]
    return df, covariates


def _ddfm_finance() -> CaseRun:
    # The old tutorial_finance_ddfm.py with the values of the flat
    # config/ddfm_finance.yaml at capture written inline, a few MCMC iterations
    # and seed 3 (that YAML's seed was null). The MLP decoder covers
    # decoder_size and the MLP last_neurons.
    panel, covariates = ddfm_finance_panel()
    data = {
        "source": "data/finance.csv", "source_sha256": _file_digest(FINANCE_CSV),
        "read": ("pd.read_csv(path); date = pd.date_range('1980-01-01', periods=len(df), freq='D'); "
                 "forward_returns and risk_free_rate dropped"),
        "rows": _ddfm_rows_recipe(),
        "dataset": "DDFMDataset(data=frame, time_idx='date', covariates=covariates, scaler=StandardScaler())",
        "targets": [_FINANCE_TARGET], "covariates": covariates,
        "fit_input_sha256": _digest(panel.iloc[:-_DDFM_HOLDOUT]),
        "update_input_sha256": _digest(panel.iloc[-_DDFM_HOLDOUT:]),
    }
    outputs, hyper = _ddfm_fit_and_update(
        panel, "date", covariates,
        encoder_size=(64, 32, 5), decoder_type="mlp", activation="relu", learning_rate=0.001,
        optimizer="Adam", n_mc_samples=10, window_size=200, max_iter=_DDFM_MAX_ITER,
        tolerance=0.0005, disp=10, seed=_DDFM_SEED, interpolation_method="linear",
        interpolation_limit=10, interpolation_limit_direction="both",
    )
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


#: The DDFM cases, in registry order.
CASES: list[Case] = [
    Case("ddfm_macro", "ddfm", _ddfm_macro),
    Case("ddfm_exchange", "ddfm", _ddfm_exchange),
    Case("ddfm_finance", "ddfm", _ddfm_finance),
]
