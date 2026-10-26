"""Deep Dynamic Factor Model (DDFM) engine: MCMC-denoising training of an autoencoder.

``DDFMEngine`` is the private engine behind
:class:`dfm_python.models.ddfm.estimator.DDFM`, which builds the engine's
``DDFMDataset`` from ``PanelData`` and passes every hyperparameter explicitly.

``fit`` pre-trains the autoencoder on the complete rows, then iterates until
the reconstruction MSE converges: estimate the AR(1) idiosyncratic residuals,
denoise the targets, draw ``n_mc_samples`` corrupted copies, train on them,
predict and impute the missing targets. ``build_state_space`` fits the AR(1)
factor transition and takes the decoder's last linear layer as the loadings;
``predict``/``_forecast`` iterate the transition and map the factors to the
targets, inverse-transformed with ``scaler`` when it is set.

``_update(new_rows)`` takes the rows after the conditioning data, in data
units, scales them with the scalers fitted at fit time, interpolates them as
fit interpolates its data and appends their encoded factors. ``_from_saved``
rebuilds a trained engine for the estimator's ``load`` without a dataset or
seeding. ``get_result`` reports ``C = H`` (series x factors, the
``DFMResult.C`` orientation).

``max_epoch_pre_train`` and ``min_target_interporate_ratio`` are stored but
never read (pre-training runs ``n_mc_samples`` epochs) and ``lags_input`` is
always 0; the golden manifests record them.
"""

import inspect
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Tuple
import pandas as pd

from ...logger import get_logger
from ...numeric.stability import convergence_checker
from ...numeric.builder import build_ddfm_state_space
from ...numeric.estimator import forecast_ar1_factors
from ...numeric.statistic import compute_variance_mean, average_3d_array
from ...numeric.builder import build_ddfm_optimizer
from .encoder import SimpleAutoencoder
from .sampling import run_mcmc_iteration
from ...config.schema.params import DDFMModelState
from ...config.schema.results import DDFMResult
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_DTYPE,
    DEFAULT_DDFM_LEARNING_RATE,
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_DDFM_WINDOW_SIZE,
    DEFAULT_MAX_MCMC_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_DISP,
    DEFAULT_SEED,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_LOSS_LOG_PRECISION,
    DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
)
from ...utils.errors import (
    ModelNotTrainedError, ModelNotInitializedError, ConfigurationError, DataValidationError
)
from ...utils.run_control import RunControl
from ...utils.validation import check_condition
from ...numeric.validator import validate_no_nan_inf, validate_horizon
from ...utils.helper import interpolate_dataframe
from ...config.types import to_tensor, to_numpy

from ...dataset.ddfm_dataset import DDFMDataset

_logger = get_logger(__name__)


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class DDFMEngine(nn.Module):
    """Deep Dynamic Factor Model using PyTorch."""

    def __init__(
        self,
        dataset: DDFMDataset,
        encoder_size: Optional[tuple] = None,
        decoder_type: str = "linear",
        seed: int = DEFAULT_SEED,
        activation: str = 'relu',
        learning_rate: float = DEFAULT_DDFM_LEARNING_RATE,
        optimizer: str = 'Adam',
        n_mc_samples: int = DEFAULT_N_MC_SAMPLES,
        window_size: int = DEFAULT_DDFM_WINDOW_SIZE,
        max_epoch_pre_train: int = DEFAULT_PRETRAIN_EPOCHS,
        max_iter: int = DEFAULT_MAX_MCMC_ITER,
        tolerance: float = DEFAULT_TOLERANCE,
        disp: int = DEFAULT_DISP,
        min_target_interporate_ratio: Optional[float] = DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
        interpolation_method: str = 'linear',
        interpolation_limit: Optional[int] = 10,
        interpolation_limit_direction: str = 'both',
        device: Optional[Union[str, torch.device]] = None,
    ):
        _init_engine_state(self)
        nn.Module.__init__(self)

        if not isinstance(dataset, DDFMDataset):
            raise ModelNotInitializedError(
                f"dataset must be an instance of DDFMDataset, got {type(dataset).__name__}"
            )

        self._dataset = dataset
        self.target_series = list(dataset.target_series)
        self.covariates = list(dataset.covariates)
        self._set_hyperparameters(
            encoder_size=encoder_size, decoder_type=decoder_type, seed=seed, activation=activation,
            learning_rate=learning_rate, optimizer=optimizer, n_mc_samples=n_mc_samples,
            window_size=window_size, max_epoch_pre_train=max_epoch_pre_train, max_iter=max_iter,
            tolerance=tolerance, disp=disp, min_target_interporate_ratio=min_target_interporate_ratio,
            interpolation_method=interpolation_method, interpolation_limit=interpolation_limit,
            interpolation_limit_direction=interpolation_limit_direction,
        )
        self.scaler = dataset.scaler
        self.covariate_scaler = dataset.covariate_scaler

        # The encoder reads every column; the decoder reconstructs the targets only.
        self.num_series = self._dataset.data_shape[1]
        self.input_dim = self._dataset.data_shape[1]
        self.output_dim = self._dataset.target_shape[1]

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.training_state = DDFMModelState()

    def _set_hyperparameters(
        self, *, encoder_size, decoder_type, seed, activation, learning_rate, optimizer, n_mc_samples,
        window_size, max_epoch_pre_train, max_iter, tolerance, disp, min_target_interporate_ratio,
        interpolation_method, interpolation_limit, interpolation_limit_direction,
    ) -> None:
        """Set the hyperparameter attributes from ``__init__``'s keyword arguments."""
        if encoder_size is None:
            encoder_size = tuple(DEFAULT_ENCODER_LAYERS)
        self.encoder_size = encoder_size
        if decoder_type not in ("linear", "mlp"):
            raise ConfigurationError(f"decoder_type must be 'linear' or 'mlp', got '{decoder_type}'")
        self.decoder_type = decoder_type
        # The MLP decoder mirrors the encoder's hidden layers.
        self.decoder_size = None
        if decoder_type == "mlp" and len(encoder_size) > 1:
            self.decoder_size = tuple(reversed(encoder_size[:-1]))

        self.activation = activation
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_epoch_pre_train = max_epoch_pre_train
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        self.min_target_interporate_ratio = min_target_interporate_ratio
        self.interpolation_method = interpolation_method
        self.interpolation_limit = interpolation_limit
        self.interpolation_limit_direction = interpolation_limit_direction

        self.initializer_seed = seed
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None  # built by _build_optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.lags_input = 0

    @classmethod
    def _from_saved(cls, params: dict, dims: dict, device: Any) -> 'DDFMEngine':
        """Rebuild a trained engine without ``__init__``: no dataset, no seeding.

        ``params`` holds ``__init__``'s keyword arguments (a keyword left out
        takes its default) and ``dims`` what ``__init__`` reads from the
        dataset: ``input_dim``, ``output_dim``, ``num_series``,
        ``target_series`` and ``covariates``. ``scaler`` and
        ``covariate_scaler`` are None (the caller sets the fitted ones). The
        autoencoder is built as ``fit`` builds it but unseeded, inside
        ``torch.random.fork_rng`` (its initialization draws from the global
        torch RNG, which is restored), moved to ``device`` and put in eval
        mode. There is no optimizer, scheduler or dataset, and
        ``training_state`` is empty: the caller loads the saved weights,
        factors and state-space matrices over the engine.
        """
        defaults = {
            name: p.default for name, p in inspect.signature(cls.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        defaults.pop("device")  # a separate argument
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            raise TypeError(f"DDFMEngine._from_saved got unexpected engine parameters {unknown}.")

        self = cls.__new__(cls)
        _init_engine_state(self)
        nn.Module.__init__(self)

        self.target_series = list(dims["target_series"])
        self.covariates = list(dims["covariates"])
        self._set_hyperparameters(**{**defaults, **params})
        self.scaler = None
        self.covariate_scaler = None

        self.num_series = int(dims["num_series"])
        self.input_dim = int(dims["input_dim"])
        self.output_dim = int(dims["output_dim"])

        self.device = torch.device(device)
        self.training_state = DDFMModelState()

        # The layer initializers draw from the global torch RNG (the CUDA one too on CUDA).
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.autoencoder = SimpleAutoencoder.build(
                input_dim=self.input_dim,
                encoder_size=self.encoder_size,
                decoder_size=self.decoder_size,
                decoder_type=self.decoder_type,
                output_dim=self.output_dim,
                activation=self.activation,
                seed=None,
            )
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        return self

    @property
    def _has_factors(self) -> bool:
        return getattr(self, 'factors', None) is not None

    def _impute_missing_targets(self, y_pred_full: np.ndarray) -> None:
        """Fill the missing targets of ``data_imputed`` with ``y_pred_full`` (T x number of targets)."""
        missing_y = self._dataset.missing_y
        if not missing_y.any():
            return

        # Fill a copy and write it back through the frame: under pandas
        # copy-on-write (the default from pandas 3) ``.values`` is read-only.
        imputed = self.data_imputed.to_numpy(copy=True)
        targets = self._dataset.target_indices
        block = imputed[:, targets]
        block[missing_y] = y_pred_full[missing_y]
        imputed[:, targets] = block
        self.data_imputed.iloc[:, :] = imputed

    def _update_imputed_and_eps(self, y_pred_full: np.ndarray) -> None:
        """Impute the missing targets with ``y_pred_full`` and recompute ``eps`` (targets minus predictions)."""
        self._impute_missing_targets(y_pred_full)
        if self._dataset.all_columns_are_targets:
            self.eps = (self.data_imputed.values - y_pred_full)[:, self.target_indices]
        else:
            self.eps = self.data_imputed.values[:, self.target_indices] - y_pred_full

    def _update_previous_predictions(
        self, y_pred: np.ndarray, y_pred_full: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Copies of the predictions for the next iteration's convergence check."""
        return (
            y_pred.copy(),
            y_pred_full.copy() if self._dataset.all_columns_are_targets else None
        )

    def _build_optimizer(self) -> None:
        self.optimizer, self.scheduler = build_ddfm_optimizer(
            model=self.autoencoder,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            n_mc_samples=self.n_mc_samples
        )

    def _build_inputs_for_pretrain(self, interpolate: bool = True) -> pd.DataFrame:
        """A copy of the dataset's frame, its missing values interpolated when ``interpolate``."""
        full_input_data = self._dataset.data.copy()
        if interpolate and full_input_data.isna().sum().sum() > 0:
            full_input_data = interpolate_dataframe(
                full_input_data,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        return full_input_data

    def _extract_target_predictions(self, y_pred_full_tensor: torch.Tensor) -> torch.Tensor:
        """Identity: the decoder's output_dim is the number of targets, so predictions are target-only."""
        return y_pred_full_tensor

    def _initialize_mcmc_state(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Interpolate the data, predict it with the pre-trained autoencoder and set the first ``eps``.

        Returns the ``(y_pred_prev, y_pred_prev_full)`` state of the convergence check.
        """
        self.data_denoised_interpolated = interpolate_dataframe(
            self.data_denoised,
            method=self.interpolation_method,
            limit=self.interpolation_limit,
            limit_direction=self.interpolation_limit_direction
        )
        self.data_imputed = self.data_denoised_interpolated.copy()

        full_input = to_numpy(self._build_inputs_for_pretrain(interpolate=True))
        full_input_tensor = to_tensor(full_input, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_pred_full = to_numpy(self.autoencoder.predict(full_input_tensor))
        self._impute_missing_targets(y_pred_full)

        # Residuals on the interpolated targets, so that eps holds no NaN: NaN
        # would reach the AR-idio estimate and the state-space SVD.
        if self._dataset.all_columns_are_targets:
            eps_full = to_numpy(self.data_denoised_interpolated) - y_pred_full
            self.eps = eps_full[:, self.target_indices]
        else:
            self.eps = to_numpy(self.data_denoised_interpolated[self._dataset.target_series]) - y_pred_full
        return self._update_previous_predictions(y_pred_full, y_pred_full)

    def fit(self, control: Optional[RunControl] = None) -> None:
        """Build the autoencoder, pre-train it on the complete rows and run the MCMC loop.

        The loop stops when the reconstruction MSE converges (``tolerance``),
        after ``max_iter`` iterations, or when ``control`` (optional; called
        after every iteration with that iteration's reconstruction MSE) asks
        for a stop before either; ``_stop_reason`` records which.
        """
        start_time = time.time()
        self.autoencoder = SimpleAutoencoder.build(
            input_dim=self.input_dim,
            encoder_size=self.encoder_size,
            decoder_size=self.decoder_size,
            decoder_type=self.decoder_type,
            output_dim=self.output_dim,
            activation=self.activation,
            seed=self.initializer_seed
        )
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.autoencoder.to(self.device)
        self._build_optimizer()

        # Pre-train with the plain MSE loss on the complete rows when there are
        # at least 50 of them, else with the masked loss on the interpolated data.
        data_pre_train_dropped = self._build_inputs_for_pretrain(interpolate=False).dropna()
        use_mse_loss = len(data_pre_train_dropped) >= 50
        if not use_mse_loss:
            data_pre_train_dropped = self._build_inputs_for_pretrain(interpolate=True).dropna()
        full_input_pre_train = data_pre_train_dropped.values
        if self._dataset.all_columns_are_targets:
            y_pre_train = full_input_pre_train
        else:
            y_pre_train = data_pre_train_dropped[self._dataset.target_series].values

        full_input_tensor = torch.tensor(full_input_pre_train, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        y_tensor = torch.tensor(y_pre_train, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        final_epoch_losses = self.autoencoder.pretrain(
            full_input=full_input_tensor,
            y=y_tensor,
            epochs=self.n_mc_samples,
            batch_size=self.window_size,
            optimizer=self.optimizer,
            use_mse_loss=use_mse_loss
        )
        if final_epoch_losses:
            _logger.info(f'Pre-training completed: final loss={final_epoch_losses[-1]:.{DEFAULT_LOSS_LOG_PRECISION}f}')

        self.data = self._dataset.data.copy()
        self.data_denoised = self.data.copy()
        self.target_indices = self._dataset.target_indices
        self.rng = np.random.RandomState(self.initializer_seed)
        y_pred_prev, y_pred_prev_full = self._initialize_mcmc_state()
        # Interpolated, so that the convergence check and the diagnostics see no NaN.
        if self._dataset.all_columns_are_targets:
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:]
        else:
            self.y_actual = self.data_denoised_interpolated.values[self.lags_input:, self.target_indices]

        converged = False
        halted = None
        self._num_iter = 0
        self.prediction_std = None
        self.factor_std = None
        while not converged and halted is None and self._num_iter < self.max_iter:
            (
                self.factors, y_pred, y_pred_full, self.prediction_std,
                self.factor_std, self.data_denoised_interpolated, autoencoder_datasets
            ) = run_mcmc_iteration(
                eps=self.eps,
                data_imputed=self.data_imputed,
                data_denoised=self.data_denoised,
                dataset=self._dataset,
                encoder=self.encoder,
                decoder=self.decoder,
                autoencoder=self.autoencoder,
                y_actual=self.y_actual,
                lags_input=self.lags_input,
                n_mc_samples=self.n_mc_samples,
                window_size=self.window_size,
                learning_rate=self.learning_rate,
                optimizer_type=self.optimizer_type,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                extract_target_predictions=self._extract_target_predictions,
                interpolation_method=self.interpolation_method,
                interpolation_limit=self.interpolation_limit,
                interpolation_limit_direction=self.interpolation_limit_direction,
                target_scaler=self.scaler,
                num_iter=self._num_iter,
                disp=self.disp,
                device=self.device,
                rng=self.rng
            )

            self._update_imputed_and_eps(y_pred_full)

            if self._num_iter > 1:
                if self._dataset.all_columns_are_targets:
                    y_pred_prev, y_pred = y_pred_prev_full, y_pred_full
                delta, self.loss_now = convergence_checker(y_pred_prev, y_pred, self.y_actual)

                _logger.info(f'iteration: {self._num_iter} - delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f} - loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}')

                if self._num_iter % self.disp == 0:
                    prediction_std_mean = compute_variance_mean(self.prediction_std)
                    factor_std_mean = compute_variance_mean(self.factor_std) if self.factor_std is not None else None
                    log_parts = [f'iteration: {self._num_iter}', f'loss: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f}', f'delta: {delta:.{DEFAULT_LOSS_LOG_PRECISION}f}']
                    if prediction_std_mean is not None:
                        log_parts.append(f'pred_std: {prediction_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    if factor_std_mean is not None:
                        log_parts.append(f'factor_std: {factor_std_mean:.{DEFAULT_LOSS_LOG_PRECISION}f}')
                    _logger.info(' - '.join(log_parts))

                if delta < self.tolerance:
                    converged = True
                    self._converged = True
                    _logger.info(f'Convergence achieved in {self._num_iter + 1} iterations')

            y_pred_prev, y_pred_prev_full = self._update_previous_predictions(y_pred, y_pred_full)

            # The MLP decoder's last_neurons are read off the last iteration's datasets.
            if self.decoder_type == "mlp":
                self._last_iter_datasets = autoencoder_datasets

            self._num_iter += 1
            if control is not None:
                # The reconstruction MSE of the convergence check (loss_now from the third iteration on).
                target_pred = y_pred_full if self._dataset.all_columns_are_targets else y_pred
                loss = convergence_checker(target_pred, target_pred, self.y_actual)[1]
                halt = control.step(self._num_iter, loss)
                if not converged and self._num_iter < self.max_iter:
                    halted = halt
        self._stop_reason = halted or ("converged" if converged else "max_iter")

        # last_neurons: the factors for the linear decoder, the MLP decoder's
        # last hidden layer otherwise.
        if self.decoder_type == "linear":
            self.last_neurons = self.factors
        else:
            decoder_intermediate = self.decoder.get_intermediate()
            self.autoencoder.eval()
            with torch.no_grad():
                self.last_neurons = np.array([
                    to_numpy(decoder_intermediate(self.encoder(ae_dataset.full_input)))
                    for ae_dataset in self._last_iter_datasets
                ])

        self._training_time = time.time() - start_time

    def build_state_space(self) -> None:
        """Fit the AR(1) state space on the MC-averaged factors and the decoder loadings into ``training_state``."""
        linear_layer = self.decoder.get_last_linear_layer()
        F, Q, mu_0, Sigma_0, H, R = build_ddfm_state_space(
            factors=average_3d_array(self.factors, axis=0),
            eps=self.eps,
            decoder_weight=to_numpy(linear_layer.weight.data),
            observed_y=self._dataset.observed_y,
            model_name=self.__class__.__name__
        )
        self.training_state.F = F
        self.training_state.Q = Q
        self.training_state.mu_0 = mu_0
        self.training_state.Sigma_0 = Sigma_0
        self.training_state.H = H
        self.training_state.R = R

    def predict(self, horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast the targets and factors ``horizon`` steps ahead (default ``DEFAULT_FORECAST_HORIZON``).

        Requires ``fit``, ``build_state_space`` and a scaler; returns
        :meth:`_forecast`'s ``(y_forecast, Z_forecast)``. To condition on new
        rows, call ``_update(new_rows)`` first.
        """
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} prediction failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        check_condition(
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} prediction failed: state-space model has not been built",
            details="Please call build_state_space() after training to enable prediction"
        )
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        horizon = validate_horizon(horizon)
        if self.scaler is None:
            raise ConfigurationError(
                f"{self.__class__.__name__} forecast failed: scaler is None",
                details="Dataset must provide scaler for proper forecast scaling"
            )
        return self._forecast(horizon)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """The trained decoder (bias and hidden layers included) on factors ``Z`` (n, r): scaled targets (n, N), float64."""
        decoder = self.decoder
        param = next(decoder.parameters())
        was_training = decoder.training
        decoder.eval()
        try:
            with torch.no_grad():
                out = decoder(torch.as_tensor(np.asarray(Z), dtype=param.dtype, device=param.device))
        finally:
            decoder.train(was_training)
        return out.detach().cpu().numpy().astype(np.float64)

    def _forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast the targets and factors ``horizon`` steps ahead.

        Pure: reads the state-space parameters and the last factor state and
        changes nothing. Returns ``(y_forecast, Z_forecast)`` of shapes
        ``(horizon, num_target_series)`` and ``(horizon, num_factors)``;
        ``y_forecast`` is on the original scale when ``self.scaler`` is set.
        """
        # Unstable training can leave NaN/Inf in the state space and the factors;
        # they are zeroed so that the forecast still runs.
        F = np.nan_to_num(self.training_state.F, nan=0.0, posinf=0.0, neginf=0.0)
        factors_avg = average_3d_array(self.factors, axis=0)
        Z_last = factors_avg[-1, :] if len(factors_avg) > 0 else self.training_state.mu_0
        Z_last = np.nan_to_num(Z_last, nan=0.0, posinf=0.0, neginf=0.0)

        validate_no_nan_inf(Z_last, name="last factor state Z_last")
        validate_no_nan_inf(F, name="transition matrix F")

        Z_forecast = forecast_ar1_factors(Z_last, F, horizon, dtype=DEFAULT_DTYPE)
        y_forecast_std = self.decode(Z_forecast)  # (horizon, num_target_series), the trained decoder
        if self.scaler is not None:
            y_forecast = self.scaler.inverse_transform(y_forecast_std)
        else:
            y_forecast = y_forecast_std
        y_forecast = np.nan_to_num(y_forecast, nan=0.0, posinf=0.0, neginf=0.0)

        y_forecast = np.asarray(y_forecast, dtype=DEFAULT_DTYPE)
        validate_no_nan_inf(y_forecast, name="forecast y_forecast")
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        return y_forecast, Z_forecast

    def get_result(self):
        """The ``DDFMResult`` of the trained model (requires ``fit`` and ``build_state_space``)."""
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} get_result failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        check_condition(
            self.training_state.F is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} get_result failed: state-space model has not been built",
            details="Please call build_state_space() after training"
        )
        st = self.training_state
        Z = average_3d_array(self.factors, axis=0)
        return DDFMResult(
            x_sm=self.decode(Z),  # (T, num_target_series): the targets only, through the trained decoder
            Z=Z,
            C=st.H,  # (num_target_series x num_factors), the DFMResult.C orientation
            R=st.R,
            A=st.F,
            Q=st.Q,
            Z_0=st.mu_0,
            V_0=st.Sigma_0,
            r=np.array([Z.shape[1]]),  # one block of num_factors
            p=DEFAULT_FACTOR_ORDER,
            target_scaler=self.scaler,
            converged=getattr(self, '_converged', False),
            num_iter=getattr(self, '_num_iter', self.max_iter),
            loglik=-float('inf'),  # DDFM computes no log-likelihood
            training_loss=getattr(self, 'loss_now', None),
            encoder_layers=list(self.encoder_size[:-1]),  # the hidden widths, as the estimator's encoder_layers
            training_max_iter=self.max_iter,
            training_tolerance=self.tolerance,
            training_n_mc_samples=self.n_mc_samples,
            training_window_size=self.window_size,
            training_learning_rate=self.learning_rate,
            training_optimizer=self.optimizer_type,
            training_encoder_size=tuple(self.encoder_size),
            training_decoder_size=None if self.decoder_size is None else tuple(self.decoder_size),
        )

    def _scale_new_rows(self, new_rows: pd.DataFrame) -> pd.DataFrame:
        """``new_rows`` (data units, columns ``[covariates | targets]``) scaled with the fitted scalers.

        ``self.scaler`` scales the targets and ``self.covariate_scaler`` the
        covariates; neither is refitted. Without a scaler the values pass
        through. Missing values stay missing.
        """
        values = new_rows.to_numpy(dtype=np.float64, copy=True)
        scaled = pd.DataFrame(values, columns=list(new_rows.columns))
        if self.scaler is not None:
            if self.covariates and self.covariate_scaler is None:
                raise ModelNotInitializedError(
                    f"{self.__class__.__name__} update failed: the covariate scaler fitted at fit time is missing",
                    details="Refit the model; runs saved before the covariate scaler was kept cannot be updated",
                )
            targets = list(self.target_series)
            scaled[targets] = self.scaler.transform(scaled[targets].to_numpy())
            if self.covariates:
                covariates = list(self.covariates)
                scaled[covariates] = self.covariate_scaler.transform(scaled[covariates].to_numpy())
        return scaled

    def _update(self, new_rows: pd.DataFrame) -> None:
        """Condition the factors on ``new_rows``, the rows after the conditioning data, in data units.

        ``new_rows`` has the columns of ``self.data`` (``[covariates |
        targets]``, in that order). They are scaled with the fitted scalers
        (:meth:`_scale_new_rows`) and appended to ``self.data``, the scaled
        conditioning frame. Missing values are filled as fit fills them:
        ``interpolate_dataframe`` over the whole conditioning frame, so the
        new rows interpolate across the boundary with the earlier ones. The
        encoder maps the filled new rows to factors, appended to
        ``self.factors`` (repeated over the MC samples when 3-D); the forecast
        then starts from the last new row. The network, the scalers and the
        state space are unchanged.
        """
        check_condition(
            self._has_factors,
            ModelNotTrainedError,
            f"{self.__class__.__name__} update failed: model has not been trained yet",
            details="Please call fit() or train() first"
        )
        check_condition(
            getattr(self, 'autoencoder', None) is not None,
            ModelNotInitializedError,
            f"{self.__class__.__name__} update failed: model has not been built",
            details="Please call fit() first"
        )

        columns = list(self.data.columns)
        if list(new_rows.columns) != columns:
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): the new rows have columns {list(new_rows.columns)[:5]}..., "
                f"expected {columns[:5]}... (the training columns [covariates | targets], in that order)."
            )
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(new_rows[col])]
        if non_numeric:
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): non-numeric columns {non_numeric[:5]}; all columns must be numeric."
            )
        n_new = len(new_rows)
        if n_new < 1:
            raise DataValidationError(f"{self.__class__.__name__}._update(): no new rows.")

        scaled = self._scale_new_rows(new_rows)
        conditioning = pd.concat([self.data.reset_index(drop=True), scaled], axis=0, ignore_index=True)
        filled = conditioning
        if conditioning.isna().to_numpy().any():
            filled = interpolate_dataframe(
                conditioning,
                method=self.interpolation_method,
                limit=self.interpolation_limit,
                limit_direction=self.interpolation_limit_direction
            )
        new_input = filled.to_numpy(dtype=np.float64)[-n_new:]
        if not np.isfinite(new_input).all():
            raise DataValidationError(
                f"{self.__class__.__name__}._update(): the new rows hold infinite values (after scaling)."
            )

        new_input_tensor = to_tensor(new_input, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            new_factors_np = to_numpy(self.encoder(new_input_tensor))  # (T_new, num_factors)

        if self.factors.ndim == 3:
            new_factors_expanded = np.repeat(np.expand_dims(new_factors_np, axis=0), self.factors.shape[0], axis=0)
            self.factors = np.concatenate([self.factors, new_factors_expanded], axis=1)
        else:
            self.factors = np.vstack([self.factors, new_factors_np])
        self.data = conditioning
