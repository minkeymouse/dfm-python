"""``DFM``: the linear dynamic factor model estimator (sklearn conventions, PanelData input).

A thin layer over the private engine
:class:`~dfm_python.models.dfm._engine.DFMEngine`. ``fit`` builds the
engine's inputs from :class:`~dfm_python.PanelData`: the per-series
frequencies ``freq_`` (unlisted series use the clock), the internal column
order (the stable partition ``[clock | slower]`` of the user's series that
:class:`~dfm_python.dataset.dfm_dataset.DFMDataset` requires) and the engine
config with every hyperparameter explicit. Same data and hyperparameters give
the engine's numbers. The series-indexed matrices of ``result_`` (``C``,
``R``, ``x_sm``) are in the internal order ``internal_feature_names_``;
forecasts are returned in data units and the user's column order.

``update(data)`` continues the Kalman filter over new rows with fixed
parameters, ``update(data, refit=True)`` runs one EM iteration on all rows,
and ``predict(h, data=new_rows)`` filters on a copy of the engine, leaving the
model unchanged. ``save(path)`` writes the pickle-free directory
(``state.npz``/``state.json``, ``result.npz``/``result.json``) from which
``DFM.load`` rebuilds the engine without data (``DFMEngine._from_saved``).
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ...base import BaseFactorModel
from ...config.constants import FREQUENCY_HIERARCHY, MAX_TENT_SIZE, PERIODS_PER_YEAR, VALID_FREQUENCIES
from ...config.estimator import _unknown_keys_message
from ...config.schema.model import DFMConfig, _DFMEngineConfig
from ...config.schema.params import DFMModelState
from ...config.schema.results import DFMResult
from ...dataset.dfm_dataset import DFMDataset
from ...dataset.panel import PanelData
from ...functional.em import EM_ACCELERATIONS
from ...functional.state_layout import DFMStateLayout
from ...io.layout import index_from_spec, index_to_spec
from ...logger import get_logger
from ...logger.progress import fit_verbosity, quiet_engine_loggers
from ...utils.errors import ConfigurationError, DataValidationError, NumericalError, UnsupportedInputError
from ...utils.interpret import affine_scaler_params, explained_variance
from ._engine import DFMEngine
from .tent import generate_tent_weights

__all__ = ["DFM"]

_logger = get_logger(__name__)

#: Scaler names accepted by ``scaler``, and the class each one builds.
_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
}
_BLOCK_KEYS = ("n_factors", "series")
_AR_CLIP_KEYS = ("min", "max")
_REGULARIZATION_KEYS = ("scale", "min_eigenvalue", "max_eigenvalue")
# Loggers the engine (and the code it calls) writes to. With verbose=0 they are
# raised to ERROR during fit and predict: the engine logs its progress at INFO
# and numerical diagnostics (stabilization, Kalman fallbacks) as warnings.
_ENGINE_LOGGERS = (
    "dfm_python.models.dfm",
    "dfm_python.functional",
    "dfm_python.ssm",
    "dfm_python.numeric",
    "dfm_python.dataset.dfm_dataset",
    "dfm_python.logger.progress",  # log_em_iteration and log_convergence
    "dfm_python.utils",
    "dfm_python.config",
)
# Name of the adapter's time column; it is never a series name (a suffix is
# added if a series has this name). DFMDataset needs a time column, and the
# engine never reads its values.
_TIME_COLUMN = "__dfm_python_time__"
# The name of the default block (blocks=None), which spans every series.
_GLOBAL_BLOCK = "Block_Global"
# result_ fields not written to result.npz/json: always None for DFM (the scaler is scaler_).
_RESULT_REBUILT = ("target_scaler",)
# The engine's state-space matrices (training_state fields), saved as ssm/<name>.
_SSM_FIELDS = ("A", "C", "Q", "R", "Z_0", "V_0")
# Optional mixed-frequency arrays (training_state fields), saved as mixed_freq/<name> when set.
_MIXED_FREQ_ARRAYS = ("constraint_matrix", "constraint_vector", "idio_indicator")


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def _hierarchy(code: str) -> int:
    return FREQUENCY_HIERARCHY[code]


def _symmetric_tent_kernel(slower: str, clock: str) -> list:
    """The symmetric tent kernel for a ``slower`` series on a ``clock`` clock.

    ``generate_tent_weights(2k - 1)`` with ``k = PERIODS_PER_YEAR[clock] //
    PERIODS_PER_YEAR[slower]``, the clock periods per slower period: ``q`` on
    an ``m`` clock gives ``[1, 2, 3, 2, 1]``. Used only in error messages.
    """
    k = PERIODS_PER_YEAR[clock] // PERIODS_PER_YEAR[slower]
    return generate_tent_weights(2 * k - 1).tolist()


def _factor_state_groups(layout: DFMStateLayout) -> list:
    """The state indices of each factor (blocks in order): its current value first, then its ``L - 1`` lags.

    A block's companion form stacks its ``r_b`` factors lag by lag, so factor
    ``j`` of block ``b`` at lag ``l`` is state ``offset_b + l * r_b + j``.
    """
    return [
        int(layout.factor_offsets[b]) + j + layout.r[b] * np.arange(layout.L)
        for b in range(len(layout.r))
        for j in range(layout.r[b])
    ]


class DFM(BaseFactorModel):
    """Linear dynamic factor model estimated by EM with Kalman filtering and smoothing.

    Latent factors follow a VAR(1) at the clock frequency; each series loads
    on the factors of the blocks it belongs to and has an idiosyncratic
    component. Series sampled at one frequency slower than the clock (for
    example quarterly series on a monthly clock) load on the factors through
    a tent kernel over the clock periods of their sampling period, and are
    NaN at the clock dates between observations. Missing values are allowed.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int, optional
        Number of factors of the single global block used when ``blocks`` is
        None; None means 1. Must stay None when ``blocks`` is given.
    blocks : dict, optional
        Block structure ``{name: {"n_factors": int, "series": [names]}}``.
        ``series`` is required and non-empty; ``n_factors`` defaults to 1.
        None means one global block over every series.
    clock : {"d", "w", "m", "q", "sa", "a"}, default "m"
        Frequency of the latent factors. Every series is sampled at the clock
        or at one slower frequency, given by ``PanelData.freq``.
    tent_weights : dict, optional
        Tent kernel of the slower frequency ``f``, under the key ``"f:clock"``
        (or ``"f"``), for example ``{"q:m": [1, 2, 3, 2, 1]}``: positive
        weights, at most 12. Required when the data has a slower frequency;
        there is no default kernel, and the error names the symmetric kernel
        of ``generate_tent_weights`` to pass.
    max_iter : int, default 5000
        Maximum number of EM iterations (E-steps, each one EM map).
    tol : float, default 1e-5
        EM convergence threshold on the relative change of the
        log-likelihood between consecutive accepted parameters.
    ar_clip : dict, optional
        ``{"min": float, "max": float}``: clip the AR coefficients of the
        transition matrix to this range after each M step (a missing key uses
        the engine's default bound). None does not clip.
    data_clip : float, optional
        Clip the data at this many standard deviations around the mean before
        EM. None does not clip.
    regularization : dict, optional
        Keys among ``"scale"``, ``"min_eigenvalue"`` and
        ``"max_eigenvalue"``; missing keys use the engine's values. None uses
        the engine's defaults. The EM M-step is the exact closed-form update
        with no ridge term and no eigenvalue cap, so ``"scale"`` and
        ``"max_eigenvalue"`` are accepted and validated but not used;
        ``"min_eigenvalue"`` is the smallest loading-column norm the final
        factor normalization rescales.
    damping_factor : float, optional
        In (0, 1]: each EM update is ``damping_factor * new + (1 -
        damping_factor) * old``. None does not damp.
    em_acceleration : {"squarem", None}, default "squarem"
        ``"squarem"`` accelerates EM with the SqS3 scheme of Varadhan and
        Roland (2008): from the current parameters two EM maps, a quadratic
        extrapolation along them, and one stabilizing EM map from the
        extrapolated point, whose output is kept when the extrapolated point's
        log-likelihood is not below that of the cycle's start (otherwise the
        second EM map is). It climbs the same likelihood in fewer E-steps, so
        at a given ``tol`` it usually stops at a higher log-likelihood than
        plain EM; every E-step counts towards ``max_iter``, and ``ar_clip`` and
        ``damping_factor`` apply to every EM map. None runs plain EM.
    scaler : str or sklearn transformer, optional
        Scaler of the series: "standard", "robust", "minmax", "maxabs",
        "quantile", or an unfitted transformer, which is cloned. None (the
        default) fits on the data as given, which must then already be on a
        comparable scale (the engine rejects series with a standard deviation
        above 100 or below 1e-4). Forecasts are always in data units.
    verbose : int, default 0
        0 is silent; 1 logs the engine's progress; 2 also logs debug records.

    Attributes
    ----------
    result_ : DFMResult
        A copy of the engine's result: ``A``, ``C``, ``Q``, ``R``, ``Z_0``,
        ``V_0``, the smoothed state ``Z`` (T x m) and ``x_sm = Z @ C.T``,
        ``loglik``, ``num_iter``, ``converged``, ``r`` and ``p``. The
        series-indexed matrices (``C``, ``R``, ``x_sm``) are in the internal
        order ``internal_feature_names_``, not the user's.
    factor_names_ : numpy.ndarray
        The factor names (object array of str): ``factor_0`` ..
        ``factor_{r-1}`` for one block, ``<block name>_<j>`` in block order
        for several.
    factors_ : pandas.DataFrame
        The smoothed factors ``E[f_t | Y]`` (the current values of the
        factors in the state, without their lags), float64 on ``index_``,
        columns ``factor_names_``.
    states_ : pandas.DataFrame
        ``result_.Z`` as float64 on ``index_``, columns ``state_0`` ..
        ``state_{m-1}`` (the factors, their lags and the idiosyncratic states).
    loadings_ : pandas.DataFrame
        float64, index ``feature_names_in_``, columns ``factor_names_``, in
        the scaled space. A clock-frequency series: its entry of ``C`` on the
        factor. A slower-frequency series: its loading on the tent-aggregated
        factor ``sum_l w_l f_{t-l}``: the tent restriction makes the entry of
        ``C`` on lag ``l`` the loading times ``w_l``, so the loading is the
        entry on lag 0 divided by the first tent weight ``w_0``.
    loadings_kind_ : str
        ``"exact"``: the observation equation is linear in the factors.
    explained_variance_ : pandas.Series
        Per series (index ``feature_names_in_``), ``1 - var(x - common) /
        var(x)`` over the observed training entries of the scaled data ``x``,
        where ``common`` is the image of the factor states (current values and
        lags) under ``C``.
    engine_ : DFMEngine
        The fitted engine.
    scaler_ : sklearn transformer or None
        The scaler fitted on the training data, in the internal order.
    n_features_in_, feature_names_in_, exog_names_in_, index_
        Number and names (user order) of the series, the exog names (always
        empty) and the training index.
    internal_feature_names_ : numpy.ndarray
        The series in the engine's order ``[clock | slower]``.
    freq_ : dict
        Frequency of every series, in the user's order.
    n_iter_ : int
        EM iterations (E-steps) run.
    converged_ : bool
        True when EM stopped on ``tol`` before ``max_iter``.
    loglik_ : float
        The engine's log-likelihood of the last EM iteration.
    n_factors_ : int
        Total number of factors (the sum over the blocks).
    conditioning_index_ : pandas.Index
        Time index of the data the forecasts start from; ``predict(h)``
        returns the ``h`` stamps after its last entry.

    Notes
    -----
    Forecasts report every series at every clock date, slower series
    included: a quarterly series on a monthly clock gets a value for each
    forecast month (the tent-weighted combination of the factor forecasts),
    as the engine computes it.

    The transition matrix ``A`` is not capped, so on non-stationary inputs
    the maximum-likelihood ``A`` can be explosive and long-horizon forecasts
    grow geometrically; ``fit`` warns when its spectral radius is above 1.
    Difference or transform such series to stationarity before fitting.

    Examples
    --------
    >>> from dfm_python import DFM, PanelData
    >>> data = PanelData(y=df, freq={"gdp": "q"})  # doctest: +SKIP
    >>> model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
    ...             max_iter=50, scaler="standard").fit(data)  # doctest: +SKIP
    >>> model.predict(6)  # doctest: +SKIP
    >>> model.predict_factors(6)  # doctest: +SKIP
    >>> model.predict_interval(6, coverage=0.9)  # doctest: +SKIP
    >>> model.decompose(by="factor")  # doctest: +SKIP
    >>> model.predict(6, data=new_rows)  # filters the rows after index_[-1]; model unchanged  # doctest: +SKIP
    >>> model.update(new_rows).save("dfm_run")  # doctest: +SKIP
    >>> DFM.load("dfm_run").predict(6)  # doctest: +SKIP
    """

    _tags = {
        "task": "forecast",
        "backend": "numpy",
        "capability:mixed_frequency": True,
        "capability:missing_values": True,
        "capability:update": True,
        "capability:factor_forecast": True,
        "capability:interpret": True,
        "capability:pred_int": True,
        "update_kind": "filter",
        "predict_data": "new_rows",
        "non_deterministic": False,
    }
    _config_class = DFMConfig

    def __init__(
        self,
        *,
        n_factors: Optional[int] = None,
        blocks: Optional[Mapping] = None,
        clock: str = "m",
        tent_weights: Optional[Mapping] = None,
        max_iter: int = 5000,
        tol: float = 1e-5,
        ar_clip: Optional[Mapping] = None,
        data_clip: Optional[float] = None,
        regularization: Optional[Mapping] = None,
        damping_factor: Optional[float] = None,
        em_acceleration: Optional[str] = "squarem",
        scaler: Any = None,
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.blocks = blocks
        self.clock = clock
        self.tent_weights = tent_weights
        self.max_iter = max_iter
        self.tol = tol
        self.ar_clip = ar_clip
        self.data_clip = data_clip
        self.regularization = regularization
        self.damping_factor = damping_factor
        self.em_acceleration = em_acceleration
        self.scaler = scaler
        self.verbose = verbose

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small parameter sets for the contract fixture (the quarterly ``s0`` first, then two monthly series).

        One global factor with plain EM, and two blocks (a global one and one
        over ``s1`` and ``s2``) with SQUAREM, both standardized. With
        ``max_iter=6`` SQUAREM runs a cycle of three plain EM maps (its step
        bound starts at 1), then a cycle with an extrapolated point.
        """
        base = {"clock": "m", "tent_weights": {"q:m": [1, 2, 3, 2, 1]}, "max_iter": 6, "scaler": "standard"}
        blocks = {
            "global": {"n_factors": 1, "series": ["s0", "s1", "s2"]},
            "pair": {"n_factors": 1, "series": ["s1", "s2"]},
        }
        return [{**base, "em_acceleration": None}, {**base, "blocks": blocks, "em_acceleration": "squarem"}]

    def _fail(self, message: str) -> None:
        raise ConfigurationError(f"{type(self).__name__}: {message}")

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` naming the first invalid hyperparameter."""
        fail = self._fail
        if self.n_factors is not None and (not _is_int(self.n_factors) or self.n_factors < 1):
            fail(f"n_factors must be None or a positive integer, got {self.n_factors!r}.")
        if self.blocks is not None:
            if self.n_factors is not None:
                fail(
                    "set n_factors or blocks, not both: n_factors is the size of the single global block used "
                    "when blocks is None; with blocks, give each block its own n_factors."
                )
            self._check_blocks()
        if not isinstance(self.clock, str) or self.clock not in VALID_FREQUENCIES:
            fail(f"clock must be one of {sorted(VALID_FREQUENCIES)}, got {self.clock!r}.")
        self._check_tent_weights()
        if not _is_int(self.max_iter) or self.max_iter < 1:
            fail(f"max_iter must be a positive integer, got {self.max_iter!r}.")
        if not _is_real(self.tol) or self.tol < 0:
            fail(f"tol must be a finite number >= 0, got {self.tol!r}.")
        self._check_ar_clip()
        if self.data_clip is not None and (not _is_real(self.data_clip) or self.data_clip <= 0):
            fail(f"data_clip must be None or a finite number > 0 (standard deviations), got {self.data_clip!r}.")
        self._check_regularization()
        damping = self.damping_factor
        if damping is not None and (not _is_real(damping) or not 0 < damping <= 1):
            fail(f"damping_factor must be None or a number in (0, 1], got {damping!r}.")
        if self.em_acceleration not in EM_ACCELERATIONS:
            fail(f"em_acceleration must be one of {list(EM_ACCELERATIONS)}, got {self.em_acceleration!r}.")
        self._check_scaler()
        if not _is_int(self.verbose) or self.verbose < 0:
            fail(f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}.")

    def _check_blocks(self) -> None:
        fail = self._fail
        blocks = self.blocks
        if not isinstance(blocks, Mapping) or len(blocks) == 0:
            fail(f"blocks must be None or a non-empty dict {{name: {{'n_factors': int, 'series': [...]}}}}, got {blocks!r}.")
        for name, spec in blocks.items():
            if not isinstance(name, str) or not name:
                fail(f"blocks: block names must be non-empty strings, got {name!r}.")
            if not isinstance(spec, Mapping):
                fail(f"blocks[{name!r}] must be a dict with keys 'n_factors' and 'series', got {spec!r}.")
            unknown = [key for key in spec if key not in _BLOCK_KEYS]
            if unknown:
                fail(f"blocks[{name!r}]: " + _unknown_keys_message(unknown, _BLOCK_KEYS, "block key"))
            n = spec.get("n_factors", 1)
            if not _is_int(n) or n < 1:
                fail(f"blocks[{name!r}]['n_factors'] must be a positive integer, got {n!r}.")
            if "series" not in spec:
                fail(f"blocks[{name!r}] has no 'series': list the series that load on this block.")
            series = spec["series"]
            if not isinstance(series, (list, tuple)) or len(series) == 0:
                fail(f"blocks[{name!r}]['series'] must be a non-empty list of series names, got {series!r}.")
            if len(set(series)) != len(series):
                fail(f"blocks[{name!r}]['series'] lists a series more than once: {list(series)!r}.")

    def _check_tent_weights(self) -> None:
        fail = self._fail
        weights = self.tent_weights
        if weights is None:
            return
        if not isinstance(weights, Mapping):
            fail(f"tent_weights must be None or a dict such as {{'q:m': [1, 2, 3, 2, 1]}}, got {weights!r}.")
        seen: dict = {}
        for key, values in weights.items():
            slower, sep, clock = key.partition(":") if isinstance(key, str) else (None, "", None)
            if slower not in VALID_FREQUENCIES or (sep and clock not in VALID_FREQUENCIES):
                fail(
                    f"tent_weights key {key!r} must be 'f:clock' or 'f' with frequency codes from "
                    f"{sorted(VALID_FREQUENCIES)}, for example 'q:{self.clock}'."
                )
            if sep and clock != self.clock:
                fail(f"tent_weights key {key!r} names clock {clock!r}, but clock is {self.clock!r}.")
            if _hierarchy(slower) <= _hierarchy(self.clock):
                fail(
                    f"tent_weights key {key!r}: {slower!r} is not slower than the clock {self.clock!r}; tent weights "
                    "apply only to a slower frequency."
                )
            if slower in seen:
                fail(f"tent_weights gives frequency {slower!r} twice (keys {seen[slower]!r} and {key!r}); keep one.")
            seen[slower] = key
            if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, np.ndarray)):
                fail(f"tent_weights[{key!r}] must be a list of positive numbers, got {values!r}.")
            values = list(np.asarray(values).ravel()) if isinstance(values, np.ndarray) else list(values)
            if not values or not all(_is_real(v) and v > 0 for v in values):
                fail(f"tent_weights[{key!r}] must be a non-empty list of finite numbers > 0, got {values!r}.")
            if len(values) > MAX_TENT_SIZE:
                fail(f"tent_weights[{key!r}] has {len(values)} weights; at most {MAX_TENT_SIZE} are supported.")

    def _check_ar_clip(self) -> None:
        clip = self.ar_clip
        if clip is None:
            return
        if not isinstance(clip, Mapping) or len(clip) == 0:
            self._fail(f"ar_clip must be None or a dict {{'min': float, 'max': float}}, got {clip!r}.")
        unknown = [key for key in clip if key not in _AR_CLIP_KEYS]
        if unknown:
            self._fail("ar_clip: " + _unknown_keys_message(unknown, _AR_CLIP_KEYS, "key"))
        for key, value in clip.items():
            if not _is_real(value):
                self._fail(f"ar_clip[{key!r}] must be a finite number, got {value!r}.")
        if "min" in clip and "max" in clip and not clip["min"] < clip["max"]:
            self._fail(f"ar_clip['min'] must be below ar_clip['max'], got {dict(clip)!r}.")

    def _check_regularization(self) -> None:
        reg = self.regularization
        if reg is None:
            return
        if not isinstance(reg, Mapping):
            self._fail(
                f"regularization must be None or a dict with keys among {list(_REGULARIZATION_KEYS)}, got {reg!r}."
            )
        unknown = [key for key in reg if key not in _REGULARIZATION_KEYS]
        if unknown:
            self._fail("regularization: " + _unknown_keys_message(unknown, _REGULARIZATION_KEYS, "key"))
        for key, value in reg.items():
            if not _is_real(value) or value < 0 or (key != "scale" and value == 0):
                bound = ">= 0" if key == "scale" else "> 0"
                self._fail(f"regularization[{key!r}] must be a finite number {bound}, got {value!r}.")

    def _check_scaler(self) -> None:
        scaler = self.scaler
        if scaler is None:
            return
        if isinstance(scaler, str):
            if scaler.lower() not in _SCALERS:
                self._fail(f"scaler must be None, one of {sorted(_SCALERS)} or a transformer instance, got {scaler!r}.")
            return
        if not (hasattr(scaler, "fit_transform") and hasattr(scaler, "inverse_transform")):
            self._fail(f"scaler must have fit_transform and inverse_transform, got {type(scaler).__name__}.")

    def _series_frequencies(self, panel: PanelData) -> dict:
        """``freq_``: each series' frequency (``PanelData.freq``, else the clock), in the user's order.

        Raises
        ------
        UnsupportedInputError
            For a series faster than the clock, or more than one slower frequency.
        ConfigurationError
            For a slower frequency without tent weights; the message gives the
            symmetric kernel and the exact argument.
        """
        clock = self.clock
        given = panel.freq or {}
        freq = {name: given.get(name, clock) for name in panel.series_names}
        faster = [name for name, code in freq.items() if _hierarchy(code) < _hierarchy(clock)]
        if faster:
            raise UnsupportedInputError(
                f"DFM: series {faster} are sampled faster than the clock {clock!r} "
                f"({ {name: freq[name] for name in faster} }). Every series must be sampled at the clock or at "
                "one slower frequency; use a faster clock or aggregate those series."
            )
        slower = sorted({code for code in freq.values() if code != clock}, key=_hierarchy)
        if len(slower) > 1:
            raise UnsupportedInputError(
                f"DFM supports one frequency slower than the clock, but the data has {slower} on clock {clock!r}. "
                "Use a slower clock or aggregate the series to one frequency."
            )
        weights = self.tent_weights or {}
        for code in slower:
            if f"{code}:{clock}" not in weights and code not in weights:
                kernel = _symmetric_tent_kernel(code, clock)
                if len(kernel) <= MAX_TENT_SIZE:
                    hint = (
                        f"The symmetric kernel generate_tent_weights({len(kernel)}) over the "
                        f"{PERIODS_PER_YEAR[clock] // PERIODS_PER_YEAR[code]} clock periods of a {code!r} period is "
                        f"{kernel}: pass tent_weights={{'{code}:{clock}': {kernel}}}."
                    )
                else:
                    hint = (
                        f"The symmetric kernel over the {PERIODS_PER_YEAR[clock] // PERIODS_PER_YEAR[code]} clock "
                        f"periods of a {code!r} period would have {len(kernel)} weights, more than the "
                        f"{MAX_TENT_SIZE} supported: pass a kernel of at most {MAX_TENT_SIZE} weights as "
                        f"tent_weights={{'{code}:{clock}': [...]}}, or use a slower clock."
                    )
                raise ConfigurationError(
                    f"DFM: series {[n for n, c in freq.items() if c == code]} are sampled at {code!r}, slower than "
                    f"the clock {clock!r}, and tent_weights has no key '{code}:{clock}' (or '{code}'). Tent weights "
                    f"are required for a slower frequency; there is no default kernel. {hint}"
                )
        return freq

    def _internal_order(self, freq: dict) -> list:
        """The stable partition ``[clock | slower]`` of the series (DFMDataset's column rule)."""
        clock_level = _hierarchy(self.clock)
        at_clock = [name for name, code in freq.items() if _hierarchy(code) <= clock_level]
        slower = [name for name, code in freq.items() if _hierarchy(code) > clock_level]
        return at_clock + slower

    def _check_series(self, panel: PanelData) -> None:
        names = list(panel.series_names)
        bad = [name for name in names if not isinstance(name, str)]
        if bad:
            raise DataValidationError(
                f"DFM: series names must be strings, got {bad!r}. Rename the columns, for example "
                "df.columns = df.columns.map(str)."
            )
        if self.blocks is not None:
            known = set(names)
            for block, spec in self.blocks.items():
                unknown = [name for name in spec["series"] if name not in known]
                if unknown:
                    raise DataValidationError(
                        f"DFM: blocks[{block!r}] names series {unknown} that are not in the data; the series are "
                        f"{names}."
                    )

    def _engine_config_dict(self, freq: dict, internal: list) -> dict:
        """The engine's config as the dict ``_DFMEngineConfig.from_dict`` reads, every value explicit."""
        if self.blocks is None:
            n = 1 if self.n_factors is None else int(self.n_factors)
            # An empty series list is the engine config's global block: every series of the dataset.
            blocks = {_GLOBAL_BLOCK: {"num_factors": n, "series": []}}
        else:
            blocks = {
                name: {"num_factors": int(spec.get("n_factors", 1)), "series": list(spec["series"])}
                for name, spec in self.blocks.items()
            }
        tent = None
        if self.tent_weights is not None:
            tent = {
                key: (list(np.asarray(values).ravel()) if isinstance(values, np.ndarray) else list(values))
                for key, values in self.tent_weights.items()
            }
        return {
            "blocks": blocks,
            "frequency": {name: freq[name] for name in internal},
            "clock": self.clock,
            "max_iter": self.max_iter,
            "threshold": self.tol,
            "ar_clip": None if self.ar_clip is None else dict(self.ar_clip),
            "data_clip": self.data_clip,
            "regularization": None if self.regularization is None else dict(self.regularization),
            "damping_factor": self.damping_factor,
            "em_acceleration": self.em_acceleration,
            "tent_weights": tent,
        }

    def _adapter(self, panel: PanelData, freq: dict, internal: list) -> tuple[_DFMEngineConfig, DFMDataset]:
        """The engine's config and dataset, built from ``panel`` in the internal order.

        The frame is ``panel.y`` in the internal order on a RangeIndex, with an
        integer time column inserted first; ``DFMDataset`` sets it as the index
        and keeps the column order (its own ``[clock | slower]`` reorder is the
        identity on this order).
        """
        config = _DFMEngineConfig.from_dict(self._engine_config_dict(freq, internal))
        frame = panel.y[internal].reset_index(drop=True)
        time_col = _TIME_COLUMN
        while time_col in set(frame.columns):
            time_col += "_"
        frame.insert(0, time_col, np.arange(len(frame)))
        dataset = DFMDataset(config=config, data=frame, time_index=time_col)
        columns = list(dataset.variables.columns)
        if columns != internal:  # pragma: no cover - guards the column rule shared with DFMDataset
            raise AssertionError(f"DFMDataset reordered the series to {columns}; expected {internal}.")
        return config, dataset

    def _fresh_scaler(self) -> Any:
        """The engine's scaler: a new instance for a name, or a clone of the instance (warns if fitted)."""
        scaler = self.scaler
        if scaler is None:
            return None
        if isinstance(scaler, str):
            return _SCALERS[scaler.lower()]()
        try:
            check_is_fitted(scaler)
        except (SklearnNotFittedError, TypeError):
            pass
        else:
            warnings.warn(
                f"{type(self).__name__}: the scaler {type(scaler).__name__} passed in is already fitted; fit uses an "
                "unfitted clone and refits it on the training data, so its statistics are discarded. The fitted "
                "copy is scaler_.",
                UserWarning,
                stacklevel=5,  # _fresh_scaler < _fit < BaseFactorModel.fit < DFM.fit < caller
            )
        return clone(scaler)

    def fit(self, data: Any, *, checkpoint_callback: Any = None) -> DFM:
        """Estimate the model on ``data`` by EM and return ``self``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            The series in ``y`` (any column order) and their sampling
            frequencies in ``freq``; missing values are allowed, and a slower
            series is NaN between its observations. ``exog`` and
            ``series_exog`` are rejected.
        checkpoint_callback : callable, optional
            Passed to the engine's EM loop unchanged; it receives the engine's
            state, whose series-indexed matrices are in the internal order
            ``[clock | slower]``.

        Raises
        ------
        ConfigurationError
            For an invalid hyperparameter, or a slower frequency without tent
            weights.
        UnsupportedInputError
            For ``exog``, ``series_exog``, a series faster than the clock or
            two slower frequencies.
        DataValidationError
            For a block naming a series that is not in the data, or a
            non-string series name.

        Warns
        -----
        UserWarning
            When ``scaler`` is an already fitted transformer (it is cloned and
            refitted).
        UserWarning
            When the fitted transition matrix ``result_.A`` has spectral
            radius above 1: the model is non-stationary and forecasts grow
            geometrically with the horizon. ``A`` is not capped, so difference
            or transform non-stationary series before fitting.

        A failed fit leaves the model unfitted (earlier fitted attributes are
        removed).
        """
        return super().fit(data, checkpoint_callback=checkpoint_callback)

    def _fit(self, data: PanelData, *, checkpoint_callback: Any = None) -> None:
        self._validate_params()
        self._check_series(data)
        freq = self._series_frequencies(data)
        internal = self._internal_order(freq)
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
            config, dataset = self._adapter(data, freq, internal)
            scaler = self._fresh_scaler()
            if self.verbose >= 2:
                _logger.debug("DFM: engine config %s, internal order %s", config, internal)
            engine = DFMEngine(dataset=dataset, config=config, scaler=scaler)
            engine.fit(checkpoint_callback=checkpoint_callback)
            result = engine.get_result()
        # The engine keeps no reference to the training data; it keeps its
        # scaled training matrix (data_processed), which the filter reads.
        engine._dataset = None

        self.engine_ = engine
        # A copy: the engine keeps its own result, which an update extends.
        self.result_ = copy.deepcopy(result)
        self.scaler_ = engine.scaler
        self.n_iter_ = int(result.num_iter)
        self.converged_ = bool(result.converged)
        self.loglik_ = float(result.loglik)
        self.n_factors_ = int(engine.num_factors)
        self.internal_feature_names_ = np.asarray(internal, dtype=object)
        self.freq_ = freq
        self.conditioning_index_ = data.index
        self._set_interpretation(data.index)
        self._warn_if_explosive(self.result_.A)

    def _warn_if_explosive(self, A: np.ndarray) -> None:
        """Warn when the fitted transition matrix has spectral radius above 1 (forecasts grow geometrically).

        The M-step does not cap the eigenvalues of ``A`` (a cap biases the
        likelihood), so on non-stationary inputs the MLE can be explosive.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.size == 0 or not np.isfinite(A).all():
            return
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        if rho > 1.0:
            warnings.warn(
                f"{type(self).__name__}: the fitted transition matrix A has spectral radius {rho:.4f} > 1, so the "
                "fitted model is non-stationary and forecasts grow geometrically with the horizon. A is not "
                "capped; difference or transform the series to stationarity before fitting.",
                UserWarning,
                stacklevel=5,  # _warn_if_explosive < _fit < BaseFactorModel.fit < DFM.fit < caller
            )

    @staticmethod
    def _state_columns(m: int) -> list:
        return [f"state_{i}" for i in range(m)]

    @classmethod
    def _state_frame(cls, Z: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """``Z`` as a float64 frame on ``index``, columns ``state_0`` .. ``state_{m-1}``."""
        values = np.asarray(Z, dtype=np.float64)
        return pd.DataFrame(values, index=index, columns=cls._state_columns(values.shape[1]))

    def _training_rows(self) -> np.ndarray:
        """A C-ordered copy of the scaled data rows ``result_`` describes (internal order, NaN kept).

        The engine's data also holds the rows ``update`` filtered later.
        """
        return np.array(self.engine_.data_processed[: self.result_.Z.shape[0]], dtype=np.float64, order="C")

    def _state_and_loadings(self) -> tuple[DFMStateLayout, np.ndarray, np.ndarray]:
        """The state layout and C-ordered float64 copies of ``result_.Z`` and ``result_.C``."""
        layout = DFMStateLayout.from_state(self.engine_.training_state)
        Z = np.array(self.result_.Z, dtype=np.float64, order="C")
        C = np.array(self.result_.C, dtype=np.float64, order="C")
        return layout, Z, C

    def _factor_states(self) -> list:
        """The state index of each factor's current value, in ``factor_names_`` order."""
        layout = DFMStateLayout.from_state(self.engine_.training_state)
        return [int(states[0]) for states in _factor_state_groups(layout)]

    def _set_interpretation(self, index: pd.Index) -> None:
        """Set ``factor_names_``, ``factors_``, ``states_``, ``loadings_``, ``loadings_kind_`` and ``explained_variance_``.

        From ``result_`` (its rows on ``index``) and the training rows of the
        engine's data; ``load`` calls it too, so the loaded attributes are
        computed from the same arrays.
        """
        layout, Z, C = self._state_and_loadings()
        r = layout.r
        if len(r) == 1:
            names = [f"factor_{j}" for j in range(r[0])]
        else:
            names = [f"{block}_{j}" for block, n in zip(self.blocks, r) for j in range(n)]
        current = self._factor_states()
        loadings = C[:, current]
        if layout.n_slower:
            tent = next(iter(self.engine_.training_state.tent_weights_dict.values()))
            loadings[layout.n_clock:] /= float(tent[0])
        X = self._training_rows()
        F = layout.n_factor_states
        explained = explained_variance(X, Z[:, :F] @ C[:, :F].T)
        internal, user = list(self.internal_feature_names_), list(self.feature_names_in_)
        self.factor_names_ = np.asarray(names, dtype=object)
        self.states_ = self._state_frame(Z, index)
        self.factors_ = pd.DataFrame(Z[:, current], index=index, columns=names)
        self.loadings_ = pd.DataFrame(loadings, index=internal, columns=names).loc[user]
        self.loadings_kind_ = "exact"
        self.explained_variance_ = pd.Series(explained, index=internal, name="explained_variance").loc[user]

    def _user_positions(self) -> np.ndarray:
        """Positions of the user's series (``feature_names_in_``) in the internal order."""
        return pd.Index(self.internal_feature_names_).get_indexer(self.feature_names_in_)

    def _affine_scale(self, method: str) -> np.ndarray:
        """Per series (internal order), ``s`` with ``inverse_transform(x) = s * x + inverse_transform(0)``.

        Raises
        ------
        UnsupportedInputError
            When ``scaler_`` is not None, StandardScaler, RobustScaler,
            MinMaxScaler or MaxAbsScaler.
        """
        params = affine_scaler_params(self.scaler_, len(self.internal_feature_names_))
        if params is None:
            raise UnsupportedInputError(
                f"DFM.{method} needs a scaler that is affine per series (None, StandardScaler, RobustScaler, "
                f"MinMaxScaler or MaxAbsScaler) to convert to data units; scaler_ is a "
                f"{type(self.scaler_).__name__}."
            )
        return np.asarray(params[1], dtype=np.float64)

    def _new_rows(self, data: Any, method: str) -> tuple[np.ndarray, pd.Index]:
        """Validate new rows for a fitted model; return them in the internal order and the extended index.

        ``data`` must hold the training series (``feature_names_in_``, same
        order), the same per-series frequencies (``freq=None`` inherits
        ``freq_``), an index of the training index's type, and at least one
        row, all strictly after the last conditioning time stamp.

        Returns
        -------
        X_new : numpy.ndarray
            float64, ``(T_new, N)``, the series in ``internal_feature_names_``
            order, in data units (the engine applies ``scaler_``).
        extended : pandas.Index
            ``conditioning_index_`` followed by ``data.index``.
        """
        panel = self._validate_data(data)  # inputs the tags exclude raise UnsupportedInputError first
        name = f"DFM.{method}"
        expected = list(self.feature_names_in_)
        if list(panel.series_names) != expected:
            raise DataValidationError(
                f"{name}: data has series {list(panel.series_names)}, but the model was fitted on {expected}. "
                "Pass the same series in the same order."
            )
        if panel.freq is not None:
            freq = {series: panel.freq.get(series, self.clock) for series in expected}
            if freq != self.freq_:
                raise DataValidationError(
                    f"{name}: data has frequencies {freq}, but the model was fitted with {self.freq_}. Pass the "
                    "same freq, or freq=None to keep the fitted frequencies."
                )
        if panel.n_timesteps < 1:
            raise DataValidationError(f"{name}: data has no time steps.")
        extended = self._extend_conditioning_index(panel.index, name)
        X_new = panel.y[list(self.internal_feature_names_)].to_numpy(dtype=np.float64)
        return X_new, extended

    def update(self, data: Any, *, refit: bool = False) -> DFM:
        """Condition later forecasts on new rows ``data`` and return ``self``.

        This is the engine's update, unchanged (tag ``update_kind`` is
        ``"filter"``): ``data`` holds only the rows after the last conditioning
        time stamp (tag ``predict_data="new_rows"``), scaled with ``scaler_``
        (never refitted).

        - ``refit=False``: the Kalman filter with the fitted parameters
          continues over the new rows: the prior of the first new row is
          ``(A x_T|T, A P_T|T A' + Q)`` from the filter over all conditioning
          rows, exactly as if the filter had run over the conditioning and the
          new rows in one pass (one row and many rows take the same path). The
          filtered states of the new rows are appended to the engine's state,
          so ``predict(h)`` then starts from the filtered state of the last new
          row. ``result_``, the parameters and the attributes ``fit`` derives
          from them (``factors_``, ``states_``, ``loadings_``,
          ``explained_variance_``) are unchanged.
        - ``refit=True``: one EM iteration (today's ``retrain=True``, with
          threshold ``tol``) on all rows, starting from the fitted parameters,
          through the same EM driver as ``fit``. ``result_``, ``n_iter_``,
          ``converged_``, ``loglik_``, ``factors_``, ``states_``,
          ``loadings_`` and ``explained_variance_`` then describe the
          refitted model on all rows; ``loglik_`` is the exact log-likelihood
          of the returned parameters on all rows. If the iteration would lower
          it, the fitted parameters are kept (``converged_`` is False).

        In both cases ``conditioning_index_`` is extended by ``data.index``.

        ``predict(h, data=new)`` forecasts from the same filtered state as
        ``update(new).predict(h)``, without changing the model.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            The new rows of the training series (same names and order, same
            frequencies; ``freq=None`` keeps the fitted ones), on an index of
            the training index's type, strictly after ``conditioning_index_``.
            A ``RangeIndex`` must continue the model's index. Missing values
            are allowed.
        refit : bool, default False
            Run one EM iteration on all rows instead of filtering.

        Raises
        ------
        DataValidationError
            If the series or frequencies differ from the training data, the
            index type differs, the rows overlap the conditioning data or
            ``data`` is empty.
        UnsupportedInputError
            For ``exog`` or ``series_exog``.
        """
        self._check_is_fitted("update")
        X_new, extended = self._new_rows(data, "update")
        engine = self.engine_
        with fit_verbosity(self.verbose), quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
            engine.update(X_new, retrain=refit)
            if refit:
                result = engine.get_result()
        self.conditioning_index_ = extended
        if refit:
            self.result_ = copy.deepcopy(result)
            self.n_iter_ = int(result.num_iter)
            self.converged_ = bool(result.converged)
            self.loglik_ = float(result.loglik)
            self._set_interpretation(extended)
        return self

    def _run_forecast(self, method: str, h: Any, data: Any) -> tuple[np.ndarray, np.ndarray, int, pd.Index]:
        self._check_is_fitted(method)
        h = self._validate_horizon(h)
        if data is None:
            with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
                X, Z = self.engine_.predict(horizon=h)
            return X, Z, h, self.conditioning_index_
        X_new, extended = self._new_rows(data, method)
        # Continue the training filter over the new rows, on a copy of the
        # engine (the model is unchanged).
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
            X, Z = copy.deepcopy(self.engine_).predict(horizon=h, data=X_new)
        return X, Z, h, extended

    def _frame(self, values: np.ndarray, h: int, columns: list, kind: str, origin_index: pd.Index) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(values, dtype=np.float64),
            index=self._forecast_index(origin_index, h),
            columns=columns,
        )
        frame.attrs = {"model": type(self).__name__, "kind": kind, "origin": origin_index[-1]}
        return frame

    def _forecast_frame(self, method: str, h: Any, data: Any) -> pd.DataFrame:
        """The pandas frame :meth:`predict` returns (series in the user's order)."""
        X, _, h, origin = self._run_forecast(method, h, data)
        frame = self._frame(X, h, list(self.internal_feature_names_), "forecast", origin)
        out = frame[list(self.feature_names_in_)]
        out.attrs = dict(frame.attrs)
        return out

    def predict(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the series ``h`` clock periods ahead, in data units.

        The state follows the fitted transition from the last smoothed state;
        the series are its image under the loadings, inverse-transformed with
        ``scaler_`` (the identity with ``scaler=None``). Every series has a
        value at every forecast date, slower series included.

        Parameters
        ----------
        h : int
            Forecast horizon (a positive integer; anything else raises
            ``TypeError``).
        data : PanelData, DataFrame or 2-D array, optional
            New rows to condition on (tag ``predict_data="new_rows"``: only
            the rows strictly after ``conditioning_index_``, as for
            :meth:`update`). The Kalman filter continues over them from the
            filter over the conditioning rows, on a copy of the engine, and
            the forecast starts from the last filtered state after
            ``data.index[-1]``. The model is unchanged; the forecast equals
            ``update(data)`` followed by ``predict(h)``.

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows after the last conditioning time stamp, one
            column per series in the user's order; ``attrs`` holds ``model``,
            ``kind="forecast"`` and ``origin``. A polars frame after
            ``set_output(output="polars")``.

        Raises
        ------
        DataValidationError
            For ``data`` that does not hold the training series, overlaps the
            conditioning data or is empty (see :meth:`update`).
        """
        return self._emit(self._forecast_frame("predict", h, data))

    def predict_factors(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the factors ``h`` clock periods ahead (same arguments as :meth:`predict`).

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows, columns ``factor_names_`` (the current value of
            each factor, as in ``factors_``); ``attrs["kind"]`` is
            ``"factors"``. A polars frame after ``set_output(output="polars")``.
        """
        _, Z, h, origin = self._run_forecast("predict_factors", h, data)
        Z = np.asarray(Z)[:, self._factor_states()]
        return self._emit(self._frame(Z, h, list(self.factor_names_), "factors", origin))

    def predict_interval(self, h: int, coverage: float = 0.9, data: Any = None) -> pd.DataFrame:
        """Central Gaussian prediction intervals of the series ``h`` clock periods ahead, in data units.

        From the filtered state covariance ``P`` at the last conditioning row
        (the last row of ``data`` when given), the forecast covariance
        follows ``P <- A P A' + Q`` for each step ahead and the series
        variance is ``diag(C P C' + R)``. The bounds are the forecast of
        :meth:`predict` (the same numbers) minus and plus ``z * sd``, with
        ``z`` the standard normal quantile of ``(1 + coverage) / 2`` and
        ``sd`` the standard deviation in data units: the scaled one times
        ``|s|``, where ``scaler_.inverse_transform(x) = s * x + offset``.

        Parameters
        ----------
        h : int
            Forecast horizon, as in :meth:`predict`.
        coverage : float, default 0.9
            Nominal coverage of each interval, in (0, 1).
        data : PanelData, DataFrame or 2-D array, optional
            New rows to condition on, as in :meth:`predict`; the model is
            unchanged.

        Returns
        -------
        pandas.DataFrame
            float64, indexed like :meth:`predict`, MultiIndex columns
            ``(series, bound)``: the series in the user's order, ``bound``
            ``"lower"`` and ``"upper"``; ``attrs`` holds those of
            :meth:`predict` and ``coverage``. A polars frame after
            ``set_output(output="polars")``.

        Raises
        ------
        ConfigurationError
            For ``coverage`` that is not a number in (0, 1).
        UnsupportedInputError
            When ``scaler_`` is not None, StandardScaler, RobustScaler,
            MinMaxScaler or MaxAbsScaler.
        DataValidationError
            For ``data`` that does not hold the training series, overlaps the
            conditioning data or is empty (see :meth:`update`).
        """
        self._check_is_fitted("predict_interval")
        if not (_is_real(coverage) and 0 < coverage < 1):
            raise ConfigurationError(f"DFM.predict_interval: coverage must be a number in (0, 1), got {coverage!r}.")
        scale = self._affine_scale("predict_interval")
        point = self._forecast_frame("predict_interval", h, data)
        engine = self.engine_
        X = engine.data_processed
        if data is not None:
            X_new, _ = self._new_rows(data, "predict_interval")
            X = np.vstack([X, X_new if engine.scaler is None else engine.scaler.transform(X_new)])
        state = engine.training_state
        A, C, Q, R = state.A, state.C, state.Q, state.R
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
            P = engine._create_kalman_filter().filter(X)[1][-1]
        variance = np.empty((len(point), C.shape[0]))
        for step in range(len(point)):
            P = A @ P @ A.T + Q
            variance[step] = np.einsum("ij,ij->i", C @ P, C) + np.diag(R)
        # z = -Phi^-1((1 - coverage) / 2): (1 + coverage) / 2 rounds to 1 for the floats just below 1.
        half = -NormalDist().inv_cdf((1 - coverage) / 2) * np.sqrt(variance) * np.abs(scale)
        centre = point.to_numpy()
        half = half[:, self._user_positions()]
        values = np.stack([centre - half, centre + half], axis=2).reshape(len(point), -1)
        columns = pd.MultiIndex.from_product([list(point.columns), ["lower", "upper"]], names=["series", "bound"])
        frame = pd.DataFrame(values, index=point.index, columns=columns)
        frame.attrs = {**point.attrs, "coverage": float(coverage)}
        return self._emit(frame)

    def decompose(self, *, by: str = "component") -> pd.DataFrame:
        """Split the training data into its parts, in data units, on the rows of ``factors_``.

        The common component is the image of the factor states (current
        values and lags, ``states_``) under ``C``, inverse-transformed with
        ``scaler_``, so it includes the series mean; the idiosyncratic part is
        the data minus the common component (the idiosyncratic states and the
        observation noise), NaN where the data is missing.

        Parameters
        ----------
        by : {"component", "factor"}, default "component"
            ``"component"``: MultiIndex columns ``(component, series)``,
            ``component`` ``"common"`` and ``"idiosyncratic"``.
            ``"factor"``: MultiIndex columns ``(part, series)``, one ``part``
            per name of ``factor_names_`` (the factor's states through ``C``,
            in data units: for a slower-frequency series, summed over the tent
            lags), ``"mean"`` (the offset ``scaler_.inverse_transform(0)``)
            and ``"idiosyncratic"``. In both, the parts add up to the data.

        Returns
        -------
        pandas.DataFrame
            float64, indexed like ``factors_`` (``index_`` after ``fit``, all
            rows after ``update(data, refit=True)``), the series in the user's
            order. A polars frame after ``set_output(output="polars")``.

        Raises
        ------
        ConfigurationError
            For another ``by``.
        UnsupportedInputError
            For ``by="factor"`` when ``scaler_`` is not None, StandardScaler,
            RobustScaler, MinMaxScaler or MaxAbsScaler (the factor parts add
            up in data units only for an affine scaler).
        """
        self._check_is_fitted("decompose")
        if by not in ("component", "factor"):
            raise ConfigurationError(f"DFM.decompose: by must be 'component' or 'factor', got {by!r}.")
        layout, Z, C = self._state_and_loadings()
        X = self._training_rows()
        scaler = self.scaler_
        data = X if scaler is None else scaler.inverse_transform(X)
        if by == "component":
            F = layout.n_factor_states
            common = Z[:, :F] @ C[:, :F].T
            if scaler is not None:
                common = scaler.inverse_transform(common)
            parts = {"common": common, "idiosyncratic": data - common}
        else:
            scale = self._affine_scale("decompose(by='factor')")
            offset = np.zeros(X.shape[1]) if scaler is None else scaler.inverse_transform(np.zeros((1, X.shape[1])))[0]
            groups = _factor_state_groups(layout)
            parts = {name: (Z[:, g] @ C[:, g].T) * scale for name, g in zip(self.factor_names_, groups)}
            explained = offset + sum(parts.values())
            parts["mean"] = np.broadcast_to(offset, X.shape)
            parts["idiosyncratic"] = data - explained
        user = self._user_positions()
        columns = pd.MultiIndex.from_product(
            [list(parts), list(self.feature_names_in_)], names=["component" if by == "component" else "part", "series"]
        )
        values = np.hstack([np.asarray(part, dtype=np.float64)[:, user] for part in parts.values()])
        return self._emit(pd.DataFrame(values, index=self.factors_.index, columns=columns))

    def _saved_data_spec(self) -> dict:
        """``data_spec.json``: the base spec plus ``internal_feature_names_`` and ``freq_``."""
        spec = super()._saved_data_spec()
        spec["internal_feature_names_"] = list(self.internal_feature_names_)
        spec["freq_"] = dict(self.freq_)
        return spec

    def _restore_data_attributes(self, spec: Mapping) -> None:
        super()._restore_data_attributes(spec)
        self.internal_feature_names_ = np.asarray(list(spec["internal_feature_names_"]), dtype=object)
        # JSON sorts the keys; freq_ follows the user's order.
        self.freq_ = {name: spec["freq_"][name] for name in self.feature_names_in_}

    def _save_state(self, path: Path) -> None:
        """Write the DFM part of the saved run (spec section 7).

        - ``state.npz``: ``ssm/A``, ``ssm/C``, ``ssm/Q``, ``ssm/R``,
          ``ssm/Z_0``, ``ssm/V_0``; ``mixed_freq/constraint_matrix`` and
          ``mixed_freq/constraint_vector`` (when set) and
          ``mixed_freq/idio_indicator``; ``structure/blocks`` and
          ``structure/r``; ``data/data_processed`` (the scaled data the filter
          reads, update rows included), ``data/smoothed_state`` (the engine's
          smoothed state, update rows included, which ``update`` extends),
          ``data/last_smoothed_state`` (its last row) and
          ``data/last_smoothed_cov`` (the last filtered state covariance
          P_T|T, from the filter call ``update`` starts from; as a safeguard,
          all NaN, with a ``RuntimeWarning``, if that filter is not finite
          with the fitted parameters: ``update`` then raises on the fitted
          and on the loaded model alike, and ``load`` does not read it); plus
          the ``scaler/`` entries. Every array keeps
          its memory order.
        - ``state.json``: the engine config dict, ``num_factors``, ``p``,
          ``n_slower_freq``, ``n_clock_freq``, ``max_lag_size``,
          ``mixed_freq``, ``tent_weights_dict`` (lists), the series
          ``frequencies`` in the internal order, the block order, the
          engine's ``num_iter``, ``converged`` and ``loglik`` and
          ``conditioning_index``.
        - ``result.npz`` / ``result.json``: ``result_`` without
          ``target_scaler`` (always None for DFM).
        """
        engine = self.engine_
        state = engine.training_state
        smoothed = engine._result.Z
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS, logging.ERROR):
            # The filter call update() starts from: its last covariance is P_T|T.
            try:
                _, filtered_cov = engine._create_kalman_filter().filter(engine.data_processed)
                last_cov = filtered_cov[-1]
            except (NumericalError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                # The fitted parameters can make this filter non-finite (EM on some
                # mixed-frequency panels); update() then raises the same error on the
                # fitted and the loaded model. The covariance is informational (load
                # does not read it), so the run is still saved.
                state_dim = engine.training_state.A.shape[0]
                last_cov = np.full((state_dim, state_dim), np.nan)
                warnings.warn(
                    f"DFM.save: the Kalman filter over the training data is not finite with the fitted parameters "
                    f"({type(exc).__name__}: {exc}); data/last_smoothed_cov is saved as NaN. Forecasts and the "
                    "result are saved unchanged; update() runs this filter and raises the same error, on the fitted and "
                    "on the loaded model.",
                    RuntimeWarning,
                    stacklevel=5,  # _save_state < write < atomic_save_dir < BaseFactorModel.save < caller
                )
        arrays = {
            **{f"ssm/{name}": getattr(state, name) for name in _SSM_FIELDS},
            **{f"mixed_freq/{name}": getattr(state, name) for name in _MIXED_FREQ_ARRAYS
               if getattr(state, name) is not None},
            "structure/blocks": engine.blocks,
            "structure/r": engine.r,
            "data/data_processed": engine.data_processed,
            "data/smoothed_state": smoothed,
            "data/last_smoothed_state": smoothed[-1],
            "data/last_smoothed_cov": last_cov,
        }
        # Keep each array's memory order (data_processed is column-major), so
        # a loaded model filters and forecasts bitwise like the fitted one.
        arrays = {key: np.array(value, order="K", copy=True) for key, value in arrays.items()}
        tent = state.tent_weights_dict
        meta = {
            "engine_config": self._engine_config_dict(self.freq_, list(self.internal_feature_names_)),
            "block_order": None if self.blocks is None else list(self.blocks),
            "num_factors": int(state.num_factors),
            "p": int(state.p),
            "n_slower_freq": int(state.n_slower_freq),
            "n_clock_freq": None if state.n_clock_freq is None else int(state.n_clock_freq),
            "max_lag_size": None if state.max_lag_size is None else int(state.max_lag_size),
            "mixed_freq": None if state.mixed_freq is None else bool(state.mixed_freq),
            "tent_weights_dict": None if tent is None else {key: np.asarray(v).tolist() for key, v in tent.items()},
            "frequencies": None if state.frequencies is None else [str(f) for f in state.frequencies],
            "num_iter": int(engine._training_num_iter),
            "converged": bool(engine._training_converged),
            "loglik": float(engine._training_loglik),
            "conditioning_index": index_to_spec(self.conditioning_index_),
        }
        self._write_state(path, arrays, meta)
        self._write_result(path, exclude=_RESULT_REBUILT)

    @classmethod
    def _load_state(cls, est: DFM, path: Path, device: Any, allow_pickle: bool) -> None:
        """Rebuild the engine without data (``DFMEngine._from_saved``) and restore the fitted attributes.

        ``factors_``, ``states_``, ``loadings_``, ``explained_variance_`` and
        the other attributes of :meth:`_set_interpretation` are recomputed from
        the saved result and data, as ``fit`` computed them, so runs saved
        before they existed load too.
        """
        arrays, meta = cls._read_state(path)
        # JSON sorts dict keys; the block order is the order of the factors.
        if est.blocks is not None:
            order = list(meta["block_order"] or [])
            if sorted(order) != sorted(est.blocks):
                raise ConfigurationError(
                    f"{str(path)!r}: state.json lists blocks {order}, but config.json has {sorted(est.blocks)}; the "
                    "run is inconsistent."
                )
            est.blocks = {name: est.blocks[name] for name in order}
        saved_config = meta["engine_config"]
        if "em_acceleration" not in saved_config:
            # Saved before em_acceleration existed (0.8.x): fitted by plain EM.
            est.em_acceleration = None
            saved_config = {**saved_config, "em_acceleration": None}
        internal = list(est.internal_feature_names_)
        engine_config = est._engine_config_dict(est.freq_, internal)
        if engine_config != saved_config:
            raise ConfigurationError(
                f"{str(path)!r}: state.json holds the engine config {meta['engine_config']}, but config.json and "
                f"data_spec.json give {engine_config}; the run is inconsistent."
            )
        tent = meta["tent_weights_dict"]
        frequencies = meta["frequencies"]
        state = DFMModelState(
            num_factors=meta["num_factors"],
            r=arrays["structure/r"],
            p=meta["p"],
            blocks=arrays["structure/blocks"],
            mixed_freq=meta["mixed_freq"],
            constraint_matrix=arrays.get("mixed_freq/constraint_matrix"),
            constraint_vector=arrays.get("mixed_freq/constraint_vector"),
            n_slower_freq=meta["n_slower_freq"],
            n_clock_freq=meta["n_clock_freq"],
            # As get_agg_structure builds them: numpy arrays of the weights.
            tent_weights_dict=None if tent is None else {key: np.array(v) for key, v in tent.items()},
            frequencies=None if frequencies is None else np.array(frequencies),
            idio_indicator=arrays.get("mixed_freq/idio_indicator"),
            max_lag_size=meta["max_lag_size"],
            **{name: arrays[f"ssm/{name}"] for name in _SSM_FIELDS},
        )
        layout = DFMStateLayout.from_state(state)
        if not layout.is_bm14() or np.asarray(state.A).shape[0] != layout.m:
            raise ConfigurationError(
                f"{str(path)!r}: the saved DFM state does not have the current state layout (an AR(1) "
                "idiosyncratic state per clock-frequency series, only a tent chain per slower-frequency series). "
                "It was written by a development build before that layout; fit the model again and save it."
            )
        result = cls._read_result(path, DFMResult, target_scaler=None)
        engine_meta = {
            "training_columns": internal,
            "num_iter": int(meta["num_iter"]),
            "converged": np.bool_(meta["converged"]),
            "loglik": np.float64(meta["loglik"]),
        }
        scaler = vars(est).get("scaler_")
        engine = DFMEngine._from_saved(
            engine_config, state, arrays["data/data_processed"], arrays["data/smoothed_state"], engine_meta,
            scaler, result,
        )
        est.engine_ = engine
        est.result_ = result
        est.scaler_ = scaler
        est.n_iter_ = int(result.num_iter)
        est.converged_ = bool(result.converged)
        est.loglik_ = float(result.loglik)
        est.n_factors_ = int(engine.num_factors)
        est.conditioning_index_ = index_from_spec(meta["conditioning_index"])
        n_rows = result.Z.shape[0]
        est._set_interpretation(est.conditioning_index_[:n_rows] if n_rows != len(est.index_) else est.index_)
