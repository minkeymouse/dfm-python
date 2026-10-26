"""Golden cases: each one fits a model through today's public API.

A case builds its data from a recipe, constructs the model, trains it and
returns a :class:`CaseRun` with

* ``outputs``: the engine numbers that the fixture pins,
* ``hyperparameters``: the values the constructed object actually holds, read
  off its attributes right after construction (not the values passed in), and
* ``data``: the recipe plus a digest of the exact input that was fitted.

``make_golden.py`` writes these to ``fixtures/<case>.npz`` and ``<case>.json``;
the ``test_golden_*`` modules re-run the same case and compare. When a model is
migrated to the new estimator API its test switches to the estimator and passes
the recorded hyperparameters to the new constructor.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MACRO_CSV = REPO_ROOT / "data" / "macro.csv"

# Global RNGs are reset to this value before every case, so a case never depends
# on what ran before it. The models then seed themselves as they do today.
GLOBAL_RNG_PRESEED = 20251019


# --------------------------------------------------------------------- types
@dataclasses.dataclass
class CaseRun:
    outputs: dict[str, np.ndarray]
    hyperparameters: dict[str, Any]
    data: dict[str, Any]
    raised: dict[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class Case:
    """A golden case.

    ``cross_platform=False`` marks a case too ill-conditioned to compare off
    the capture platform at any useful tolerance: it is compared bitwise on
    the capture platform and skipped elsewhere (``_golden.skip_unless_comparable``).
    No registered case uses it today. ``dfm_macro_weekly_monthly`` and
    ``dfm_macro_monthly_quarterly_update`` did until the DFM numerical-stability
    rewrite; their recaptured fits agree within 1e-6 relative (2.5e-10 at most,
    norm-relative, on parameters and forecasts) between the default AVX-512
    kernels and forced AVX2 (Haswell, Sandybridge) or SSE (Nehalem) kernels.
    """

    name: str
    model: str
    run: Callable[[], CaseRun]
    cross_platform: bool = True

    def __repr__(self) -> str:  # readable pytest ids
        return self.name


# ------------------------------------------------------------------- helpers
def jsonable(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON types (lists, dicts, str, float, int, bool, None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if _is_torch(obj, "device"):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"class": type(obj).__name__,
                "fields": jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})}
    return {"class": type(obj).__name__}


def normalized(obj: Any) -> Any:
    """``jsonable`` followed by a JSON round trip, for comparing with a manifest."""
    return json.loads(json.dumps(jsonable(obj), sort_keys=True))


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, pd.DataFrame):
            h.update(json.dumps([str(c) for c in a.columns]).encode())
            h.update(json.dumps([str(i) for i in a.index]).encode())
            for col in a.columns:
                if np.issubdtype(a[col].dtype, np.number):
                    h.update(np.ascontiguousarray(a[col].to_numpy(np.float64)).tobytes())
                else:
                    h.update(json.dumps([str(v) for v in a[col]]).encode())
        else:
            arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attrs(obj: Any, names: list[str]) -> dict[str, Any]:
    return {n: getattr(obj, n) for n in names}


def _class_name(obj: Any) -> str | None:
    return None if obj is None else type(obj).__name__


def _is_torch(obj: Any, cls: str) -> bool:
    """``isinstance(obj, torch.<cls>)`` without importing torch (the DFM cases run without it)."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, getattr(torch, cls))


def _np(t: Any) -> np.ndarray:
    if _is_torch(t, "Tensor"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _reset_global_rngs() -> None:
    random.seed(GLOBAL_RNG_PRESEED)
    np.random.seed(GLOBAL_RNG_PRESEED)
    try:
        import torch
    except ImportError:  # the DFM cases also run without the deep extra
        return
    torch.manual_seed(GLOBAL_RNG_PRESEED)


@contextlib.contextmanager
def force_cpu() -> Iterator[None]:
    """Make today's ``cuda if available else cpu`` device choice resolve to CPU."""
    import torch

    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        yield


def _macro_tail(n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(MACRO_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.iloc[-n_rows:].reset_index(drop=True)


def _macro_recipe(**extra: Any) -> dict[str, Any]:
    return {"source": "data/macro.csv", "source_sha256": _file_digest(MACRO_CSV), **extra}


# ----------------------------------------------------------------------- AFM
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


# ----------------------------------------------------------------------- DFM
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


# ---------------------------------------------------------------------- DDFM
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


# --------------------------------------------------------------------- iVDFM
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


# ------------------------------------------------------------------ registry
CASES: dict[str, Case] = {c.name: c for c in [
    Case("afm_attention_longconv", "afm", lambda: _afm_case("attention", "longconv")),
    Case("afm_pca_longconv", "afm", lambda: _afm_case("pca", "longconv")),
    Case("afm_pca_ou", "afm", lambda: _afm_case("pca", "ou")),
    # patience stops training early (at epoch 17 of 40); the learning rate is
    # high enough that the loss stalls.
    Case("afm_pca_longconv_patience", "afm",
         lambda: _afm_case("pca", "longconv", max_epochs=40, learning_rate=0.1, patience=3)),
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
    Case("ddfm_macro", "ddfm", _ddfm_macro),
    Case("ddfm_exchange", "ddfm", _ddfm_exchange),
    Case("ddfm_finance", "ddfm", _ddfm_finance),
    Case("ivdfm_macro", "ivdfm", _ivdfm_macro),
    Case("ivdfm_regimes", "ivdfm", _ivdfm_regimes),
    Case("ivdfm_exog_context", "ivdfm", _ivdfm_exog_context),
]}


def cases_for(model: str) -> list[Case]:
    return [c for c in CASES.values() if c.model == model]


def check_case(name: str, run: CaseRun) -> None:
    """Compare a fresh run of case ``name`` with its fixture."""
    from ._golden import assert_golden, load_golden

    manifest, _ = load_golden(name)
    assert normalized(run.data) == manifest["data"], f"golden '{name}': input data differs from the fixture"
    assert normalized(run.raised) == manifest["raised"], (
        f"golden '{name}': raised {run.raised}, fixture recorded {manifest['raised']}")
    assert normalized(run.hyperparameters) == manifest["hyperparameters"], (
        f"golden '{name}': constructed hyperparameters differ from the fixture")
    assert_golden(run.outputs, name)
