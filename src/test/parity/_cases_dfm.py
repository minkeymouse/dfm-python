"""Golden cases of DFM: the data recipes, the engine runs and the outputs each fixture pins."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from ._cases_common import (
    Case,
    CaseRun,
    MACRO_CSV,
    _class_name,
    _digest,
    _macro_recipe,
    _macro_tail,
    _reset_global_rngs,
)


def _dfm_hyper(model: Any) -> dict[str, Any]:
    """The engine's config, EM config and structure as the DFM fixtures record them.

    The fixtures predate ``em_acceleration``; every case runs plain EM
    (``em_acceleration=None``), which is left out of the record, so any other
    value differs from the fixture.
    """
    cfg = model._config
    config_fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                     if not f.name.startswith("_")}
    em_config = dataclasses.asdict(cfg.to_em_config())
    for fields in (config_fields, em_config):
        if fields["em_acceleration"] is None:
            del fields["em_acceleration"]
    return {
        "config": config_fields,
        "em_config": em_config,
        "num_factors": model.num_factors,
        "r": model.r,
        "p": model.p,
        "blocks": model.blocks,
        "scaler": _class_name(model.scaler),
    }


#: Forecast horizon of every DFM predict call in the golden cases.
DFM_HORIZON = 6
#: Outputs of the update path (``_dfm_update_outputs``), pinned by the cases that pass new rows.
DFM_UPDATE_KEYS = (
    "predict_data_X", "predict_data_Z",
    "update_Z", "update_data_processed", "update_predict_X", "update_predict_Z",
    "refit_A", "refit_C", "refit_Q", "refit_R", "refit_Z_0", "refit_V_0", "refit_loglik",
    "refit_num_iter", "refit_converged", "refit_result_Z", "refit_predict_X", "refit_predict_Z",
)


def dfm_fit_outputs(model: Any, X: Any, Z: Any) -> dict[str, np.ndarray]:
    """The fitted engine's numbers, given its ``predict(horizon=6)`` forecast ``(X, Z)``.

    ``last_filtered_cov`` is the last filtered state covariance on the training
    data, the covariance ``update()`` starts from (it equals the smoothed
    covariance at ``T``).
    """
    result = model.get_result()
    filtered_cov = model._create_kalman_filter().filter(np.ma.masked_invalid(model.data_processed))[1]
    outputs = {
        "A": model.A, "C": model.C, "Q": model.Q, "R": model.R, "Z_0": model.Z_0, "V_0": model.V_0,
        "loglik": np.float64(result.loglik),
        "num_iter": np.int64(result.num_iter),
        "converged": np.bool_(result.converged),
        "predict_X": X, "predict_Z": Z,
        "result_Z": result.Z, "result_x_sm": result.x_sm,
        "data_processed": model.data_processed,
        "scaler_mean": model.scaler.mean_, "scaler_scale": model.scaler.scale_,
        "last_filtered_cov": filtered_cov[-1],
    }
    return {k: np.asarray(v) for k, v in outputs.items()}


def _dfm_update_outputs(model: Any, new: np.ndarray) -> dict[str, np.ndarray]:
    """The update path on new rows ``new`` (float64, dataset column order).

    ``predict(horizon=6, data=new)`` on the fitted model (it does not mutate the
    model), then ``update(new)`` and ``update(new, retrain=True)`` each on a deep
    copy of it, each followed by ``predict(horizon=6)``.
    """
    import copy

    X_d, Z_d = model.predict(horizon=DFM_HORIZON, data=new)
    outputs: dict[str, Any] = {"predict_data_X": X_d, "predict_data_Z": Z_d}

    upd = copy.deepcopy(model)
    upd.update(new)
    X_u, Z_u = upd.predict(horizon=DFM_HORIZON)
    outputs.update({
        "update_Z": upd.get_result().Z, "update_data_processed": upd.data_processed,
        "update_predict_X": X_u, "update_predict_Z": Z_u,
    })

    refit = copy.deepcopy(model)
    refit.update(new, retrain=True)
    X_r, Z_r = refit.predict(horizon=DFM_HORIZON)
    outputs.update({
        "refit_A": refit.A, "refit_C": refit.C, "refit_Q": refit.Q, "refit_R": refit.R,
        "refit_Z_0": refit.Z_0, "refit_V_0": refit.V_0,
        "refit_loglik": np.float64(refit.get_result().loglik),
        "refit_num_iter": np.int64(refit.get_result().num_iter),
        "refit_converged": np.bool_(refit.get_result().converged),
        "refit_result_Z": refit.get_result().Z,
        "refit_predict_X": X_r, "refit_predict_Z": Z_r,
    })
    return {k: np.asarray(v) for k, v in outputs.items()}


def _dfm_fit(config_dict: dict[str, Any], df: pd.DataFrame,
             new_rows: pd.DataFrame | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit DFM on ``df``; with ``new_rows`` (same columns as ``df``) also run the update path."""
    from sklearn.preprocessing import StandardScaler

    from dfm_python.config.schema.model import _DFMEngineConfig
    from dfm_python.dataset.dfm_dataset import DFMDataset
    from dfm_python.models.dfm._engine import DFMEngine

    # The engine path: the pre-redesign DFM class, now the private DFMEngine,
    # called as tutorial/tutorial_macro_dfm.py called the old public class.
    _reset_global_rngs()
    # The fixtures were captured with plain EM, before em_acceleration existed.
    config = _DFMEngineConfig.from_dict({**config_dict, "em_acceleration": None})
    dataset = DFMDataset(config=config, data=df, time_index="date")
    model = DFMEngine(dataset=dataset, config=config, scaler=StandardScaler())
    hyper = _dfm_hyper(model)
    model.fit()
    X, Z = model.predict(horizon=DFM_HORIZON)
    outputs = dfm_fit_outputs(model, X, Z)
    outputs["columns"] = np.array(list(dataset.variables.columns), dtype=str)
    if new_rows is not None:
        new = new_rows[list(dataset.variables.columns)].to_numpy(np.float64)
        outputs.update(_dfm_update_outputs(model, new))
    return outputs, hyper


_DFM_SYNTHETIC_NEW_ROWS = 6


def dfm_synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """The ``dfm_synthetic`` panel, its update rows (both with a ``date`` column), config dict and data recipe."""
    T, N, r, seed = 80, 5, 2, 0
    rng = np.random.default_rng(seed)
    f = np.zeros((T, r))
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal(r)
    loadings = rng.standard_normal((N, r))
    y = f @ loadings.T + 0.5 * rng.standard_normal((T, N))
    cols = [f"s{i}" for i in range(N)]
    df = pd.DataFrame(y, columns=cols)
    df.insert(0, "date", pd.date_range("2010-01-31", periods=T, freq="ME"))
    # New rows for the update path: the same process continued, from a second generator.
    T_new, new_seed = _DFM_SYNTHETIC_NEW_ROWS, 1
    rng_new = np.random.default_rng(new_seed)
    f_new = np.zeros((T_new, r))
    prev = f[-1]
    for t in range(T_new):
        prev = 0.7 * prev + rng_new.standard_normal(r)
        f_new[t] = prev
    y_new = f_new @ loadings.T + 0.5 * rng_new.standard_normal((T_new, N))
    new_df = pd.DataFrame(y_new, columns=cols)
    new_df.insert(0, "date", pd.date_range(df["date"].iloc[-1], periods=T_new + 1, freq="ME")[1:])
    config = {
        "blocks": {"Block_Global": {"num_factors": r, "series": cols}},
        "frequency": {c: "m" for c in cols},
        "clock": "m",
        "max_iter": 20,
        "threshold": 1e-5,
    }
    data = {
        "generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N,
        "recipe": ("f[0]=0; f[t] = 0.7 f[t-1] + rng.standard_normal(2) for t=1..T-1; "
                   "loadings = rng.standard_normal((N, 2)); "
                   "y = f @ loadings.T + 0.5 * rng.standard_normal((T, N))"),
        "columns": cols, "index": "pd.date_range('2010-01-31', periods=T, freq='ME') in column 'date'",
        "config_dict": config,
        "input_sha256": _digest(df),
        "update_rows": {
            "generator": "numpy.random.default_rng", "seed": new_seed, "T_new": T_new,
            "recipe": ("g = f[T-1]; for t=0..T_new-1: g = 0.7 g + rng_new.standard_normal(2), f_new[t] = g; "
                       "y_new = f_new @ loadings.T + 0.5 * rng_new.standard_normal((T_new, N)) "
                       "(same loadings)"),
            "index": "the next T_new month ends after the last fitted date, in column 'date'",
            "input_sha256": _digest(new_df),
        },
    }
    return df, new_df, config, data


def _dfm_synthetic() -> CaseRun:
    df, new_df, config, data = dfm_synthetic_inputs()
    outputs, hyper = _dfm_fit(config, df, new_rows=new_df)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_DFM_WM_ROWS = 400

# The values of the flat config/dfm_macro.yaml (as OmegaConf loaded it, key
# order kept) that the dfm_macro_* fixtures were captured with, through the
# old DFMConfig.from_dict (now _DFMEngineConfig.from_dict). The shipped file is
# now in the model:/data: layout with the canonical names, and DFMConfig is the
# estimator's mirror, so the cases keep their captured values here. The
# engine config never reads clip_ar_coefficients, ar_clip_min/max,
# use_regularization, regularization_scale, augment_idio*, idio_rho0,
# idio_min_var or scaler (ar_clip and regularization ran as None); they are
# kept so the dict the engine config is built from is the captured one.
# The data recipes still name the file the values came from.
_DFM_MACRO_FLAT_CONFIG: dict[str, Any] = {
    "threshold": 1e-05, "max_iter": 10, "clip_ar_coefficients": True, "ar_clip_min": -0.99,
    "ar_clip_max": 0.99, "use_regularization": True, "regularization_scale": 1e-05, "augment_idio": True,
    "augment_idio_slow": True, "idio_rho0": 0.1, "idio_min_var": 1e-08, "clock": "w", "scaler": "standard",
    "tent_weights": {"m:w": [1, 2, 1]},
    "blocks": {"Block_Global": {"num_factors": 3, "series": [
        "KORFBEV", "KORWHLS", "KORMCMP", "KORELEC", "KORCNST", "KORMACH", "KOROCOINF", "KOEMPTOTO",
        "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOEQUIPTE", "KOWRCCNSE", "KOIMPCONA"]}},
    "frequency": {
        "m": ["KOEMPTOTO", "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOEQUIPTE", "KOWRCCNSE", "KOIMPCONA"],
        "w": ["KORFBEV", "KORWHLS", "KORMCMP", "KORELEC", "KORCNST", "KORMACH", "KOROCOINF"],
    },
}


def _dfm_macro_flat_config() -> dict[str, Any]:
    """A deep copy of :data:`_DFM_MACRO_FLAT_CONFIG` (the cases mutate their copy)."""
    import copy

    return copy.deepcopy(_DFM_MACRO_FLAT_CONFIG)


def _dfm_macro_weekly_monthly() -> CaseRun:
    # The old tutorial_macro_dfm.py on the last rows of data/macro.csv: weekly
    # clock with monthly series, the flat config/dfm_macro.yaml values unchanged.
    config = _dfm_macro_flat_config()
    df = _macro_tail(_DFM_WM_ROWS)
    data = _macro_recipe(rows=f"last {_DFM_WM_ROWS} rows", config="config/dfm_macro.yaml",
                         input_sha256=_digest(df))
    outputs, hyper = _dfm_fit(config, df)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_MQ_MONTHLY = ["KOROCOINF", "KOEMPTOTO", "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOWRCCNSE"]
_MQ_QUARTERLY = ["KOEQUIPTE", "KOIMPCONA"]
_MQ_MONTHS = 120
_MQ_PERMUTED = ["KOEQUIPTE", "KOROCOINF", "KOWRCDURE", "KOIMPCONA",
                "KOEMPTOTO", "KOCONPRCF", "KOWRCCNSE", "KOHWRWEMP"]


def _mq_panel() -> pd.DataFrame:
    raw = pd.read_csv(MACRO_CSV)
    raw["date"] = pd.to_datetime(raw["date"])
    cols = _MQ_MONTHLY + _MQ_QUARTERLY
    df = raw.loc[raw["date"].dt.is_month_end, ["date"] + cols]
    df = df.dropna(how="all", subset=cols).iloc[-_MQ_MONTHS:].reset_index(drop=True)
    not_quarter_end = df["date"].dt.month % 3 != 0
    df.loc[not_quarter_end, _MQ_QUARTERLY] = np.nan
    return df


def _mq_config() -> dict[str, Any]:
    config = _dfm_macro_flat_config()
    config.update({
        "clock": "m",
        "tent_weights": {"q:m": [1, 2, 3, 2, 1]},
        "blocks": {"Block_Global": {"num_factors": 2, "series": _MQ_MONTHLY + _MQ_QUARTERLY}},
        "frequency": {"m": list(_MQ_MONTHLY), "q": list(_MQ_QUARTERLY)},
    })
    return config


def _mq_recipe(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    return _macro_recipe(
        rows=(f"rows whose date is a month end, all-NaN rows dropped, last {_MQ_MONTHS}; "
              "quarterly series set to NaN where month % 3 != 0"),
        monthly=_MQ_MONTHLY, quarterly=_MQ_QUARTERLY, column_order=columns,
        config="config/dfm_macro.yaml with clock='m', tent_weights={'q:m': [1, 2, 3, 2, 1]}, "
               "one global block of 2 factors over monthly + quarterly series, grouped frequency",
        input_sha256=_digest(df))


def _dfm_macro_monthly_quarterly() -> CaseRun:
    df = _mq_panel()
    outputs, hyper = _dfm_fit(_mq_config(), df)
    return CaseRun(outputs=outputs, hyperparameters=hyper,
                   data=_mq_recipe(df, _MQ_MONTHLY + _MQ_QUARTERLY))


def _dfm_macro_monthly_quarterly_permuted() -> CaseRun:
    # Same panel and config, input columns permuted (quarterly interleaved with
    # monthly). Records the error raised today, or the outputs if none is raised.
    df = _mq_panel()[["date"] + _MQ_PERMUTED]
    data = _mq_recipe(df, _MQ_PERMUTED)
    try:
        outputs, hyper = _dfm_fit(_mq_config(), df)
    except Exception as exc:  # recorded, not swallowed: the test asserts the same error
        return CaseRun(outputs={}, hyperparameters={}, data=data,
                       raised={"type": type(exc).__name__, "message": str(exc)})
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_MQ_UPDATE_ROWS = 6


def _dfm_macro_monthly_quarterly_update() -> CaseRun:
    # The monthly/quarterly panel minus its last rows, fitted with the same
    # config; those last rows are then the new rows of the update path.
    panel = _mq_panel()
    df, new_df = panel.iloc[:-_MQ_UPDATE_ROWS], panel.iloc[-_MQ_UPDATE_ROWS:]
    data = _mq_recipe(df, _MQ_MONTHLY + _MQ_QUARTERLY)
    data["rows"] += f"; fit on all but the last {_MQ_UPDATE_ROWS} rows, which are the update rows"
    data["update_input_sha256"] = _digest(new_df)
    outputs, hyper = _dfm_fit(_mq_config(), df, new_rows=new_df)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


#: The DFM cases, in registry order.
CASES: list[Case] = [
    Case("dfm_synthetic", "dfm", _dfm_synthetic),
    # Weekly clock with monthly series (tent [1, 2, 1]). Before the DFM
    # numerical-stability rewrite EM diverged on this panel and the case was
    # pinned to the capture platform; it is now compared everywhere.
    Case("dfm_macro_weekly_monthly", "dfm", _dfm_macro_weekly_monthly),
    Case("dfm_macro_monthly_quarterly", "dfm", _dfm_macro_monthly_quarterly),
    Case("dfm_macro_monthly_quarterly_permuted", "dfm", _dfm_macro_monthly_quarterly_permuted),
    # The update and refit paths on a mixed-frequency panel. Before the DFM
    # numerical-stability rewrite the refit diverged (smoothed states ~1e41) and
    # the case was pinned to the capture platform; it is now compared everywhere.
    Case("dfm_macro_monthly_quarterly_update", "dfm", _dfm_macro_monthly_quarterly_update),
]
