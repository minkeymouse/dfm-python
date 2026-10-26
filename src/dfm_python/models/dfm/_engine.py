"""Linear DFM engine: EM in NumPy with the in-house Kalman filter and smoother.

``DFMEngine`` is the private engine the public
:class:`~dfm_python.models.dfm.estimator.DFM` builds from ``PanelData``
(``dataset=``/``config=`` constructor on a ``DFMDataset`` and a
``_DFMEngineConfig``). It owns the fitted state (``training_state``), the
scaled data the filter reads (``data_processed``, update rows included) and
its cached ``DFMResult``. :meth:`DFMEngine._from_saved` rebuilds a trained
engine for the estimator's ``load`` without ``__init__`` (no dataset).
"""

import copy
import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ...ssm.kalman import DFMKalmanFilter
from ...config import DFMResult
from ...config.schema.model import _DFMEngineConfig
from ...config.schema.params import DFMModelState
from ...config.constants import DEFAULT_DTYPE, MAX_STD_UNSCALED, MIN_STD_SCALED
from ...config.types import to_numpy
from ...logger import get_logger
from ...dataset.dfm_dataset import DFMDataset
from ...utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    ConfigurationError,
    DataValidationError,
)
from ...utils.run_control import RunControl
from ...utils.validation import check_condition
from ...numeric.estimator import forecast_ar1_factors
from ...numeric.validator import (
    validate_and_convert_update_data,
    validate_column_order,
    validate_dfm_initialization,
    validate_horizon,
    validate_no_nan_inf,
    validate_parameters_initialized,
)
from ...functional import em as _em  # looked up at call time: tests patch em.run_em_algorithm
from .initialization import initialize_parameters
from .mixed_freq import setup_mixed_frequency_params
from ...numeric.builder import build_dfm_structure, build_dfm_blocks, log_blocks_diagnostics

_logger = get_logger(__name__)


def _stop_reason(metadata: Dict[str, Any]) -> str:
    """Why ``run_em_algorithm`` stopped, as the estimator's ``stop_reason_``.

    The stop ``control`` asked for (``"max_time"``, ``"callback"``);
    ``"early_stopping"`` when a guard stopped EM and it returned the best
    accepted parameters; else ``"converged"`` or ``"max_iter"``.
    """
    if metadata['halted'] is not None:
        return metadata['halted']
    if metadata['stop_reason'] is not None:
        return "early_stopping"
    return "converged" if metadata['converged'] else "max_iter"


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class DFMEngine:
    """Linear dynamic factor model estimated by EM (NumPy E-step and M-step)."""

    def __init__(
        self,
        dataset: DFMDataset,
        config: Optional[_DFMEngineConfig] = None,
        scaler: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Build the model structure from ``config`` and ``dataset``; parameters are set by ``fit``.

        Parameters
        ----------
        dataset : DFMDataset
            Gives the variables (``fit`` reads them) and the column names.
        config : _DFMEngineConfig, optional
            None builds the config from ``kwargs`` with
            ``_DFMEngineConfig.from_dict``.
        scaler : sklearn transformer, optional
            Fitted in ``fit`` unless already fitted; None fits the data as is.
        **kwargs
            With ``config``, overrides of its attributes (unknown names are
            logged and ignored); without it, the config fields.

        Raises
        ------
        ModelNotInitializedError
            If ``dataset`` is not a ``DFMDataset``.
        ConfigurationError
            Without ``config`` and config kwargs.
        """
        _init_engine_state(self)

        if not isinstance(dataset, DFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DFMDataset, got {type(dataset).__name__}"
            )
        self._dataset = dataset

        if config is None:
            if not kwargs:
                raise ConfigurationError(
                    "DFM: Either config or kwargs must be provided",
                    details="Provide a _DFMEngineConfig object, or pass all config parameters as kwargs (e.g., blocks, threshold, max_iter, etc.)"
                )
            config_kwargs = {k: v for k, v in kwargs.items() if k not in ('dataset', 'scaler')}
            if not config_kwargs:
                raise ConfigurationError(
                    "DFM: No config parameters provided in kwargs",
                    details="Provide _DFMEngineConfig parameters like blocks, threshold, max_iter, etc."
                )
            self._config = _DFMEngineConfig.from_dict(config_kwargs)
            kwargs = {}
        else:
            self._config = config

        for key, value in kwargs.items():
            if key in ('dataset', 'scaler'):
                continue
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                _logger.debug(f"DFM: Overrode config.{key} with {value}")
            else:
                _logger.warning(f"DFM: Unknown parameter '{key}' in kwargs, ignoring")

        # Mixed-frequency attributes, set by fit from setup_mixed_frequency_params.
        self._mixed_freq: Optional[bool] = None
        self._cached_smoothed_factors: Optional[np.ndarray] = None
        self._constraint_matrix = None
        self._constraint_vector = None
        self._n_slower_freq = 0
        self._tent_weights_dict = None
        self._frequencies = None
        self._idio_indicator = None

        # A config without frequencies binds late to the dataset columns (all at the clock).
        dataset_columns = list(self._dataset.variables.columns) if self._dataset.variables is not None else None
        self.blocks, self.r, self.num_factors, self.p = build_dfm_structure(self._config, columns=dataset_columns)

        # Structure only; the state-space parameters are set by fit.
        self.training_state = DFMModelState(
            num_factors=self.num_factors,
            r=self.r,
            p=self.p,
            blocks=self.blocks,
            mixed_freq=None,
            constraint_matrix=None,
            constraint_vector=None,
            n_slower_freq=0,
            n_clock_freq=None,
            tent_weights_dict=None,
            frequencies=None,
            idio_indicator=None,
            max_lag_size=None,
            A=None,
            C=None,
            Q=None,
            R=None,
            Z_0=None,
            V_0=None
        )

        self.data_processed: Optional[np.ndarray] = None
        self.scaler: Optional[Any] = scaler

    @classmethod
    def _from_saved(
        cls,
        engine_config: Dict[str, Any],
        state: DFMModelState,
        data_processed: np.ndarray,
        smoothed_state: np.ndarray,
        meta: Dict[str, Any],
        scaler: Optional[Any],
        result: DFMResult,
    ) -> 'DFMEngine':
        """Rebuild a trained engine without ``__init__`` (no dataset).

        Sets what ``__init__`` and ``fit`` set: the config from the dict the
        estimator built it from; the structure and mixed-frequency attributes
        from ``state`` (a complete ``DFMModelState``); ``data_processed`` (the
        scaled data, update rows included), ``scaler``, ``_dataset=None``;
        the training metadata from ``meta`` (keys ``training_columns``,
        ``num_iter``, ``converged``, ``loglik``); and ``_result``, a copy of
        ``result`` with ``Z=smoothed_state`` (the state ``update`` extends)
        and, when ``smoothed_state`` has more rows than ``result.Z``,
        ``x_sm = smoothed_state @ C.T`` as ``update`` computes it.
        """
        engine = cls.__new__(cls)
        _init_engine_state(engine)
        engine._config = _DFMEngineConfig.from_dict(dict(engine_config))
        engine._dataset = None
        engine.training_state = state
        engine.blocks = state.blocks
        engine.r = state.r
        engine.num_factors = state.num_factors
        engine.p = state.p
        engine.data_processed = data_processed
        engine.scaler = scaler
        engine._cached_smoothed_factors = None
        engine._mixed_freq = state.mixed_freq
        engine._constraint_matrix = state.constraint_matrix
        engine._constraint_vector = state.constraint_vector
        engine._n_slower_freq = state.n_slower_freq
        engine._n_clock_freq = state.n_clock_freq
        engine._tent_weights_dict = state.tent_weights_dict
        engine._frequencies = state.frequencies
        engine._idio_indicator = state.idio_indicator
        engine._max_lag_size = state.max_lag_size
        engine._training_columns = list(meta["training_columns"])
        engine._training_num_iter = meta["num_iter"]
        engine._training_converged = meta["converged"]
        engine._training_loglik = meta["loglik"]
        result = copy.deepcopy(result)
        if smoothed_state.shape[0] == result.Z.shape[0]:
            x_sm = result.x_sm
        else:
            x_sm = smoothed_state @ result.C.T
        engine._result = dataclasses.replace(result, Z=smoothed_state, x_sm=x_sm)
        return engine

    @property
    def A(self) -> Optional[np.ndarray]:
        """Transition matrix (m x m)."""
        return self.training_state.A if self.training_state else None

    @property
    def C(self) -> Optional[np.ndarray]:
        """Observation matrix (N x m), the loadings."""
        return self.training_state.C if self.training_state else None

    @property
    def Q(self) -> Optional[np.ndarray]:
        """Process noise covariance (m x m)."""
        return self.training_state.Q if self.training_state else None

    @property
    def R(self) -> Optional[np.ndarray]:
        """Observation noise covariance (N x N), diagonal."""
        return self.training_state.R if self.training_state else None

    @property
    def Z_0(self) -> Optional[np.ndarray]:
        """Initial state mean (m,)."""
        return self.training_state.Z_0 if self.training_state else None

    @property
    def V_0(self) -> Optional[np.ndarray]:
        """Initial state covariance (m x m)."""
        return self.training_state.V_0 if self.training_state else None

    def _check_parameters_initialized(self) -> None:
        """Raise ``ModelNotInitializedError`` unless every state-space parameter is set."""
        validate_parameters_initialized(
            {
                'A': self.A, 'C': self.C, 'Q': self.Q,
                'R': self.R, 'Z_0': self.Z_0, 'V_0': self.V_0
            },
            model_name=self.__class__.__name__
        )

    def _update_parameters(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
                          R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray) -> None:
        """Set the state-space parameters of ``training_state`` as ``DEFAULT_DTYPE`` arrays."""
        self.training_state.A = np.asarray(A, dtype=DEFAULT_DTYPE) if A is not None else None
        self.training_state.C = np.asarray(C, dtype=DEFAULT_DTYPE) if C is not None else None
        self.training_state.Q = np.asarray(Q, dtype=DEFAULT_DTYPE) if Q is not None else None
        self.training_state.R = np.asarray(R, dtype=DEFAULT_DTYPE) if R is not None else None
        self.training_state.Z_0 = np.asarray(Z_0, dtype=DEFAULT_DTYPE) if Z_0 is not None else None
        self.training_state.V_0 = np.asarray(V_0, dtype=DEFAULT_DTYPE) if V_0 is not None else None

    def fit(
        self,
        X: Optional[Union[np.ndarray, Any]] = None,
        control: Optional[RunControl] = None
    ) -> DFMModelState:
        """Fit by EM: scale the data, initialize the parameters, run ``run_em_algorithm``.

        EM runs ``config.max_iter`` E-steps at most, plain or accelerated
        (``config.em_acceleration``); ``_training_num_iter`` is the number of
        E-steps run and ``_training_stop_reason`` why EM stopped
        (:func:`_stop_reason`).

        Parameters
        ----------
        X : np.ndarray or DataFrame, optional
            Data (T x N) in the dataset's column order; None uses
            ``dataset.variables``. Scaled with ``scaler`` (fitted here unless
            already fitted); the scaled series must have standard deviations
            in ``[MIN_STD_SCALED, MAX_STD_UNSCALED]``.
        control : RunControl, optional
            The time budget and the per-E-step callback, passed to the EM loop.

        Returns
        -------
        DFMModelState
            The fitted ``training_state``.
        """
        self._cached_smoothed_factors = None
        self._result = None
        self._config._cached_blocks = None

        dataset = self._dataset
        if X is None:
            data_df = dataset.variables
            X_np = data_df.values
            columns = list(data_df.columns)
        elif isinstance(X, np.ndarray):
            X_np = X
            columns = list(dataset.variables.columns)
            if X_np.shape[1] != len(columns):
                raise DataValidationError(
                    f"Provided X has {X_np.shape[1]} columns but dataset has {len(columns)} columns. "
                    f"Column count must match.",
                    details=f"X shape: {X_np.shape}, dataset columns: {columns[:5]}..."
                )
        else:
            X_np = np.asarray(X)
            columns = list(X.columns) if hasattr(X, 'columns') else list(dataset.variables.columns)

        clock = self._config.clock
        self._training_columns = columns  # update() validates against them

        if self.scaler is not None:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(self.scaler)
                is_fitted = True
            except NotFittedError:
                is_fitted = False
            if is_fitted:
                X_np_std = self.scaler.transform(X_np)
            else:
                X_np_std = self.scaler.fit_transform(X_np)
        else:
            X_np_std = X_np

        # Unscaled data (raw levels) makes EM unstable and slow: scaling is a contract.
        data_std = np.nanstd(X_np_std, axis=0)
        unscaled_series = np.where((data_std > MAX_STD_UNSCALED) | (data_std < MIN_STD_SCALED))[0]
        if len(unscaled_series) > 0:
            unscaled_names = [columns[i] for i in unscaled_series[:5]]
            raise DataValidationError(
                f"Input data appears unscaled after preprocessing. Series with problematic std: {unscaled_names}... "
                f"(std range: [{np.min(data_std):.2e}, {np.max(data_std):.2e}]). "
                f"Please apply a scaler (e.g., StandardScaler) before fitting the model. "
                f"Unscaled data causes numerical instability, excessive stabilization, and slow convergence.",
                details=f"Expected std range: [{MIN_STD_SCALED:.2e}, {MAX_STD_UNSCALED:.2e}]. "
                       f"Found {len(unscaled_series)}/{len(columns)} series outside this range. "
                       f"Scaler provided: {self.scaler is not None}"
            )

        N_actual = X_np_std.shape[1]
        self.blocks = build_dfm_blocks(self.blocks, self._config, columns, N_actual)
        log_blocks_diagnostics(self.blocks, columns, N_actual)

        # Raises DataValidationError unless the columns are ordered [clock | slower].
        mf_params = setup_mixed_frequency_params(self._config, clock, columns, N_actual)
        self._mixed_freq        = mf_params['mixed_freq']
        self._constraint_matrix = mf_params['R_mat']
        self._constraint_vector = mf_params['q']
        self._n_slower_freq     = mf_params['n_slower_freq']
        self._n_clock_freq      = mf_params['n_clock_freq']
        self._tent_weights_dict = mf_params['tent_weights_dict']
        self._frequencies       = mf_params['frequencies_np']
        self._idio_indicator    = mf_params['idio_indicator']
        self._max_lag_size      = max(self.p + 1, mf_params['tent_kernel_size'])

        _logger.info("Initializing DFM parameters...")
        _logger.info(f"  Data: {X_np_std.shape[0]} time steps × {X_np_std.shape[1]} series")
        _logger.info(f"  Blocks: {self.blocks.shape[1]}, Factors: {self.num_factors}, "
                    f"Mixed freq: {self._mixed_freq}")
        A_np, C_np, Q_np, R_np, Z_0_np, V_0_np = initialize_parameters(
            X_np_std, self.r, self.p, self.blocks,
            self._constraint_matrix, self._constraint_vector, self._n_slower_freq,
            self._idio_indicator, clock, self._tent_weights_dict
        )

        self.data_processed = X_np_std
        validate_dfm_initialization(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)
        self._update_parameters(A_np, C_np, Q_np, R_np, Z_0_np, V_0_np)

        _logger.info(f"Initialization complete: state_dim={self.A.shape[0]}, obs_dim={self.C.shape[0]}, "
                    f"factors={self.num_factors}, max_lag={self._max_lag_size}, mixed_freq={self._mixed_freq}")

        self.training_state.mixed_freq = self._mixed_freq
        self.training_state.constraint_matrix = self._constraint_matrix
        self.training_state.constraint_vector = self._constraint_vector
        self.training_state.n_slower_freq = self._n_slower_freq
        self.training_state.n_clock_freq = self._n_clock_freq
        self.training_state.tent_weights_dict = self._tent_weights_dict
        self.training_state.frequencies = self._frequencies
        self.training_state.idio_indicator = self._idio_indicator
        self.training_state.max_lag_size = self._max_lag_size

        try:
            final_state, training_metadata = _em.run_em_algorithm(
                X=X_np_std,
                initial_state=self.training_state,
                max_iter=self._config.max_iter,
                threshold=self._config.threshold,
                config=self._config.to_em_config(),
                control=control
            )
        except Exception as e:  # diagnostics only: logs the model's structure and re-raises unchanged
            _logger.error(f"EM algorithm failed: {e}", exc_info=True)
            _logger.error("  Initialization parameters:")
            _logger.error(f"    A shape: {self.A.shape}, C shape: {self.C.shape}")
            _logger.error(f"    Q shape: {self.Q.shape}, R shape: {self.R.shape}")
            _logger.error(f"    Z_0 shape: {self.Z_0.shape}, V_0 shape: {self.V_0.shape}")
            _logger.error(f"    Blocks shape: {self.blocks.shape}, r: {self.r}")
            _logger.error(f"    p: {self.p}, max_lag_size: {self._max_lag_size}")
            raise

        self.training_state = final_state
        self._training_loglik = training_metadata['loglik']
        self._training_num_iter = training_metadata['num_iter']
        self._training_converged = training_metadata['converged']
        self._training_stop_reason = _stop_reason(training_metadata)
        # The smoothed factors of the last EM iteration save get_result a smoother run.
        if training_metadata.get('smoothed_factors') is not None:
            self._cached_smoothed_factors = training_metadata['smoothed_factors']

        return self.training_state

    def _create_kalman_filter(
        self,
        initial_state_mean: Optional[np.ndarray] = None,
        initial_state_covariance: Optional[np.ndarray] = None
    ) -> DFMKalmanFilter:
        """Kalman filter with the fitted parameters; the initial state defaults to ``(Z_0, V_0)``."""
        if initial_state_mean is None:
            initial_state_mean = self.training_state.Z_0
        if initial_state_covariance is None:
            initial_state_covariance = self.training_state.V_0

        return DFMKalmanFilter(
            transition_matrices=self.training_state.A,
            observation_matrices=self.training_state.C,
            transition_covariance=self.training_state.Q,
            observation_covariance=self.training_state.R,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )

    def _compute_smoothed_factors(self) -> np.ndarray:
        """Smoothed states (T x m): the last EM iteration's when cached, else a smoother run."""
        check_condition(
            self.training_state is not None and self.data_processed is not None,
            ModelNotTrainedError,
            "Model not fitted or data not available",
            details="Please call fit() method before computing smoothed factors"
        )

        if self._cached_smoothed_factors is not None:
            _logger.debug("Using cached smoothed factors from last EM iteration (skipping recomputation)")
            return self._cached_smoothed_factors

        # The filter raises NumericalError when the parameters make it non-finite.
        _logger.debug("Computing smoothed factors (cache not available)")
        kalman_final = self._create_kalman_filter()
        smoothed_state_means, _, _, _ = kalman_final.filter_and_smooth(self.data_processed)
        return smoothed_state_means

    def _filter_new_rows(self, data_new: np.ndarray) -> np.ndarray:
        """Filtered states of ``data_new``, continuing the training filter.

        Runs the Kalman filter with the fitted parameters over all rows the
        model conditions on (``data_processed``) followed by ``data_new``, so the
        prior of the first new row is ``(A x_T|T, A P_T|T A' + Q)``, and returns
        the filtered means ``x_t|t`` of the new rows (T_new x m).
        """
        X_all = np.vstack([self.data_processed, data_new])
        filtered_means, _ = self._create_kalman_filter().filter(X_all)
        return np.asarray(filtered_means[-data_new.shape[0]:], dtype=DEFAULT_DTYPE)

    def get_result(self) -> DFMResult:
        """The ``DFMResult`` of the fitted model (cached; ``x_sm = Z @ C.T``)."""
        if self._result is not None:
            return self._result

        Z = self._compute_smoothed_factors()
        state = self.training_state
        result = DFMResult(
            x_sm=Z @ state.C.T, Z=Z, C=state.C, R=state.R, A=state.A, Q=state.Q,
            target_scaler=None,  # the engine's scaler is ``scaler``
            Z_0=state.Z_0, V_0=state.V_0, r=self.r, p=self.p,
            converged=getattr(self, '_training_converged', False),
            num_iter=getattr(self, '_training_num_iter', 0),
            loglik=getattr(self, '_training_loglik', 0.0),
            training_max_iter=self._config.max_iter,
            training_threshold=self._config.threshold,
            training_regularization_scale=(self._config.regularization or {}).get("scale"),
        )
        self._result = result
        return result

    def _ensure_result(self) -> DFMResult:
        """The cached result, computed if needed; ``ModelNotTrainedError`` before ``fit``."""
        if self._result is None:
            if self.training_state is None:
                raise ModelNotTrainedError("Model has not been trained yet")
            self._result = self.get_result()
        return self._result

    def update(self, data: Union[np.ndarray, Any], rescale: bool = False, retrain: bool = False, **kwargs) -> None:
        """Condition the model on the new rows ``data`` (T_new x N, preprocessed, not scaled).

        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New rows, columns in the training order.
        rescale : bool, default False
            Refit the scaler on ``data`` before transforming it.
        retrain : bool, default False
            Run EM (``run_em_algorithm``, the driver ``fit`` uses) on the
            training rows plus the new rows, starting from the fitted
            parameters; the reported loglik is the exact loglik of the returned
            parameters on all rows, and an iteration that would lower it is not
            kept. Otherwise the training filter continues over the new rows
            (one path for one and for many rows) and their filtered states are
            appended to ``result.Z``; the parameters are unchanged.
        **kwargs
            With ``retrain``: ``max_iter`` (default 1: one EM map, the same
            with and without ``config.em_acceleration``) and ``threshold``
            (default the config's).
        """
        validate_column_order(data, self.scaler)
        data_new = validate_and_convert_update_data(
            data,
            self.data_processed,
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        if self.scaler is not None:
            if rescale:
                self.scaler.fit(data_new)
            data_new = self.scaler.transform(data_new)

        if retrain:
            X_combined = np.vstack([self.data_processed, data_new])
            final_state, training_metadata = _em.run_em_algorithm(
                X=X_combined,
                initial_state=self.training_state,
                max_iter=kwargs.get('max_iter', 1),
                threshold=kwargs.get('threshold', self._config.threshold),
                config=self._config.to_em_config(),
            )
            self.data_processed = X_combined
            self.training_state = final_state
            self._training_loglik = training_metadata['loglik']
            self._training_num_iter = training_metadata['num_iter']
            self._training_converged = training_metadata['converged']
            self._training_stop_reason = _stop_reason(training_metadata)
            # The parameters changed: the cached result and smoothed factors are stale.
            self._result = None
            self._cached_smoothed_factors = None
            _logger.debug("Invalidated cached smoothed factors due to retrain")
        else:
            result = self._ensure_result()
            Z_new = self._filter_new_rows(data_new)
            result.Z = np.vstack([result.Z, Z_new])
            self.data_processed = np.vstack([self.data_processed, data_new])
            result.x_sm = result.Z @ result.C.T
            # result.Z was extended: the cached smoothed factors are stale.
            self._cached_smoothed_factors = None
            self._result = result
            _logger.debug("Invalidated cached smoothed factors after update() - result.Z was mutated")

    def predict(
        self,
        horizon: int,
        *,
        data: Optional[Union[np.ndarray, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast ``horizon`` periods ahead from the last state.

        Parameters
        ----------
        horizon : int
            Periods ahead (``validate_horizon``).
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame, optional
            New rows (T_new x N, preprocessed, not scaled): the training
            filter continues over them and the forecast starts from the last
            filtered state; the model is unchanged. None forecasts from the
            last row of ``result.Z``.

        Returns
        -------
        X_forecast : np.ndarray
            (horizon x N) series forecast in data units (``scaler`` inverted).
        Z_forecast : np.ndarray
            (horizon x m) state forecast.
        """
        check_condition(
            self.training_state is not None,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() first"
        )
        self._check_parameters_initialized()
        result = self._ensure_result()
        check_condition(
            result.Z is not None,
            ModelNotTrainedError,
            "DFM prediction failed: result.Z is not available"
        )
        horizon = validate_horizon(horizon)
        A = result.A
        C = result.C

        if data is not None:
            data_new = validate_and_convert_update_data(
                data,
                self.data_processed,
                dtype=DEFAULT_DTYPE,
                model_name=self.__class__.__name__
            )
            if self.scaler is not None:
                data_new = self.scaler.transform(data_new)
            Z_last = self._filter_new_rows(data_new)[-1]
        elif result.Z.shape[0] > 0:
            Z_last = result.Z[-1, :]
        else:
            Z_last = np.zeros(A.shape[0], dtype=DEFAULT_DTYPE)

        validate_no_nan_inf(Z_last, name="factor state Z_last")
        validate_no_nan_inf(A, name="transition matrix A")
        validate_no_nan_inf(C, name="observation matrix C")

        Z_forecast = forecast_ar1_factors(Z_last, A, horizon, dtype=DEFAULT_DTYPE)
        X_forecast_std = Z_forecast @ C.T

        max_abs_value = np.abs(X_forecast_std).max()
        if max_abs_value > 5.0:
            _logger.warning(
                f"DFM predictions in standardized space are extreme (max abs: {max_abs_value:.2f}). "
                f"This indicates model instability or poor training."
            )

        if self.scaler is not None:
            X_forecast = self.scaler.inverse_transform(X_forecast_std)
        else:
            X_forecast = X_forecast_std
        X_forecast = to_numpy(X_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(X_forecast, name="forecast X_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        return X_forecast, Z_forecast
