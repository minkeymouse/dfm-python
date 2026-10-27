"""Golden cases of iVDFM: the data recipes, the engine runs and the outputs each fixture pins."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._cases_common import (
    Case,
    CaseRun,
    force_cpu,
    _attrs,
    _class_name,
    _digest,
    _macro_recipe,
    _macro_tail,
    _np,
    _reset_global_rngs,
)


# The engine path: the pre-redesign iVDFM class, now the private iVDFMEngine,
# called exactly as the old public constructor was (dataset in __init__).
# test_golden_ivdfm.py also runs the public iVDFM estimator against the same fixtures.
_IVDFM_ATTRS =["data_dim", "latent_dim", "context_dim", "time_context", "window",
                "encoder_hidden_dim", "encoder_n_hidden_layers", "decoder_type",
                "decoder_hidden_dim", "decoder_n_hidden_layers", "prior_hidden_dim",
                "prior_n_hidden_layers", "activation", "slope", "factor_order",
                "innovation_distribution", "decoder_var", "num_regimes", "regime_temperature",
                "beta_kl", "use_layer_norm", "mixing", "learning_rate", "optimizer_type",
                "optimizer_weight_decay", "optimizer_momentum", "batch_size", "max_epochs",
                "tolerance", "patience", "scheduler_type", "scheduler_step_size",
                "scheduler_gamma", "scheduler_patience", "scheduler_factor", "scheduler_min_lr",
                "seed", "device"]


def _ivdfm_hyper(model: Any, dataset: Any) -> dict[str, Any]:
    hyper = _attrs(model, _IVDFM_ATTRS)
    hyper["dataset"] = {"window": dataset.window, "stride": dataset.stride,
                        "context_dim": dataset.context_dim,
                        "scaler": _class_name(dataset.scaler)}
    return hyper


def _ivdfm_outputs(model: Any, horizon: int = 6) -> dict[str, np.ndarray]:
    pred = model.predict(horizon=horizon)
    outputs = {
        "factors": _np(model.factors),
        "innovations": _np(model.innovations),
        "num_iter": np.int64(model._num_iter),
        "predict": _np(pred),
        # The last epoch's mean reconstruction loss and ELBO (numpy float64 means
        # of Python floats), and whether training stopped early.
        "loss_now": np.float64(model.loss_now),
        "elbo": np.float64(model._elbo),
        "converged": np.bool_(model._converged),
    }
    regime_weights = getattr(model, "regime_weights", None)
    if regime_weights is not None:
        outputs["regime_weights"] = _np(regime_weights)
    for k, v in model.state_dict().items():
        outputs[f"sd/{k}"] = _np(v)
    return outputs


def ivdfm_time_features(horizon: int, time_context: int, T_train: int) -> np.ndarray:
    """The engine's future time features: ``horizon`` steps after ``T_train``, ``time_context`` columns.

    The formula of the engine's ``predict`` when ``context_data`` is None,
    written out here so that a case can build its own ``context_data``.
    ``T_train`` is the length of the engine's factor path (the conditioning
    series), ``engine._factor_path().shape[0]``.
    """
    time_indices = np.arange(T_train, T_train + horizon, dtype=np.float32)
    time_indices = time_indices / (T_train - 1) if T_train > 1 else time_indices / T_train
    if time_context == 1:
        return time_indices.reshape(-1, 1)
    features = [time_indices.reshape(-1, 1)]
    for i in range(1, time_context):
        freq = 2 * np.pi * (i + 1) / T_train if T_train > 1 else 2 * np.pi * (i + 1)
        periodic = np.sin(freq * np.arange(T_train, T_train + horizon, dtype=np.float32))
        features.append(periodic.reshape(-1, 1))
    return np.hstack(features)


_IVDFM_ROWS = 400
_IVDFM_MAX_EPOCHS = 3

# The flat config/ivdfm_macro.yaml the ivdfm_macro fixture was captured from
# (commit 6c6e985), as OmegaConf loaded it. The shipped file has since been
# rewritten in the model:/data: layout with the canonical names, so the case
# keeps the values it was captured with here. The data recipe still names the
# file the values came from.
_IVDFM_MACRO_FLAT_CONFIG: dict[str, Any] = {
    "num_factors": 5, "sequence_length": 200, "context": None, "context_dim": 1,
    "encoder_hidden_dim": 200, "encoder_n_layers": 3, "decoder_hidden_dim": 200, "decoder_n_layers": 3,
    "prior_hidden_dim": 100, "prior_n_layers": 2, "activation": "lrelu", "slope": 0.1, "factor_order": 3,
    "innovation_distribution": "laplace", "decoder_var": 0.01, "learning_rate": 0.001, "optimizer": "Adam",
    "batch_size": 256, "max_epochs": 10, "tolerance": 1e-05, "seed": None, "frequency": None, "clock": "w",
    "scaler": "standard",
}


def _ivdfm_macro() -> CaseRun:
    from dfm_python.models.ivdfm._engine import iVDFMEngine as iVDFM
    from dfm_python.config.schema.model import _iVDFMEngineConfig as iVDFMConfig
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset

    # The pre-redesign tutorial_macro_ivdfm.py on the last rows of data/macro.csv.
    # That tutorial did not run (it passed context_dim to iVDFMDataset), so the
    # dataset gets its data, covariates, window (sequence_length) and scaler,
    # and the model is built in the constructor-dataset order.
    yaml_cfg = dict(_IVDFM_MACRO_FLAT_CONFIG)
    df = _macro_tail(_IVDFM_ROWS)
    targets = ["KOEQUIPTE", "KOEMPTOTO", "KOHWRWEMP"]
    covariates = [c for c in df.columns if c != "date" and c not in targets][:32]
    df_data = df[["date"] + targets + covariates].copy()
    window = int(yaml_cfg.get("sequence_length") or 100)
    data = _macro_recipe(rows=f"last {_IVDFM_ROWS} rows", targets=targets, covariates=covariates,
                         window=window, config="config/ivdfm_macro.yaml",
                         input_sha256=_digest(df_data))

    with force_cpu():
        _reset_global_rngs()
        config = iVDFMConfig.from_dict(yaml_cfg)
        dataset = iVDFMDataset(data=df_data, time_idx="date", covariates=covariates,
                               window=window, scaler=config.scaler)
        model = iVDFM(dataset=dataset, config=config, max_epochs=_IVDFM_MAX_EPOCHS)
        hyper = _ivdfm_hyper(model, dataset)
        model.fit()
        outputs = _ivdfm_outputs(model)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


def _ivdfm_regimes() -> CaseRun:
    from dfm_python.models.ivdfm._engine import iVDFMEngine as iVDFM
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset

    T, N, seed = 60, 4, 0
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((T, N))
    data = {"generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N,
            "recipe": "y = rng.standard_normal((T, N)); iVDFMDataset(data=y, window=24, stride=1, time_context=1)",
            "input_sha256": _digest(y)}

    with force_cpu():
        _reset_global_rngs()
        dataset = iVDFMDataset(data=y, window=24, stride=1, time_context=1)
        model = iVDFM(dataset=dataset, num_factors=3, num_regimes=2, factor_order=2,
                      max_epochs=2, batch_size=16)
        hyper = _ivdfm_hyper(model, dataset)
        model.fit()
        outputs = _ivdfm_outputs(model)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


IVDFM_EXOG_HORIZON = 6


def ivdfm_exog_context_inputs() -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """The ``ivdfm_exog_context`` panel, its future exog (horizon x 2) and data recipe.

    The panel has a month-end ``date`` column, targets ``y0..y2`` with two NaN
    and exog (context) columns ``x0``, ``x1``.
    """
    T, N, K, seed = 72, 3, 2, 7
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((T, N))
    x = rng.standard_normal((T, K))
    x_future = rng.standard_normal((IVDFM_EXOG_HORIZON, K))
    y[5, 0] = np.nan
    y[40, 2] = np.nan
    df = pd.DataFrame(y, columns=[f"y{i}" for i in range(N)])
    for k in range(K):
        df[f"x{k}"] = x[:, k]
    df.insert(0, "date", pd.date_range("2015-01-31", periods=T, freq="ME"))
    data = {
        "generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N, "K": K,
        "recipe": ("y = rng.standard_normal((T, N)); x = rng.standard_normal((T, K)); "
                   "x_future = rng.standard_normal((6, K)); y[5, 0] = y[40, 2] = NaN; "
                   "columns date (pd.date_range('2015-01-31', periods=T, freq='ME')), y0..y2, x0, x1"),
        "dataset": ("iVDFMDataset(data=df, time_idx='date', context=['x0', 'x1'], window=24, "
                    "time_context=2, scaler='standard')"),
        "context_data": ("predict_futr: [time features of predict with time_context=2 and "
                         "T_train=window | x_future]"),
        "input_sha256": _digest(df),
        "x_future_sha256": _digest(x_future),
    }
    return df, x_future, data


def _ivdfm_exog_context() -> CaseRun:
    from dfm_python.models.ivdfm._engine import iVDFMEngine as iVDFM
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset

    df, x_future, data = ivdfm_exog_context_inputs()
    window, time_context = 24, 2

    with force_cpu():
        _reset_global_rngs()
        dataset = iVDFMDataset(data=df, time_idx="date", context=["x0", "x1"], window=window,
                               time_context=time_context, scaler="standard")
        model = iVDFM(dataset=dataset, num_factors=2, num_regimes=2, factor_order=2,
                      time_context=time_context, batch_size=4, max_epochs=3)
        hyper = _ivdfm_hyper(model, dataset)
        model.fit()
        outputs = _ivdfm_outputs(model, horizon=IVDFM_EXOG_HORIZON)
        # The engine normalizes the future time step by the length of its factor path (the series length).
        T_train = model._factor_path().shape[0]
        context_data = np.hstack([ivdfm_time_features(IVDFM_EXOG_HORIZON, time_context, T_train), x_future])
        outputs["predict_futr"] = _np(model.predict(horizon=IVDFM_EXOG_HORIZON, context_data=context_data))
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


#: The iVDFM cases, in registry order.
CASES: list[Case] = [
    Case("ivdfm_macro", "ivdfm", _ivdfm_macro),
    Case("ivdfm_regimes", "ivdfm", _ivdfm_regimes),
    Case("ivdfm_exog_context", "ivdfm", _ivdfm_exog_context),
]
