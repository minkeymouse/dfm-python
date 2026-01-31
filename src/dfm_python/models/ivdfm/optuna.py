"""Optuna-based hyperparameter optimization for iVDFM.

Uses a subset of the latest windows for fast trials: T_subset is chosen so that
(max_window + (n_windows_min - 1) * min_stride + horizon) time steps are enough
to train on n_windows_min windows and hold out the last `horizon` steps for the
forecasting objective (sMSE or sMAE). This keeps each trial fast while giving
a stable metric (n_windows_min ~22 is a rule-of-thumb for normal approximation).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ...dataset.ivdfm_dataset import iVDFMDataset
from ...logger import get_logger
from ..ivdfm import iVDFM

_logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Subset sampling (latest windows)
# -----------------------------------------------------------------------------

def get_optuna_subset(
    data: Union[np.ndarray, pd.DataFrame],
    max_window: int,
    n_windows_min: int,
    min_stride: int,
    horizon: int,
    *,
    subset_ratio: Optional[float] = None,
    subset_max_steps: Optional[int] = None,
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """Take the latest chunk of data and split into train / holdout for forecast metric.

    T_subset is computed as max_window + (n_windows_min - 1) * min_stride + horizon, then optionally
    expanded by subset_ratio (fraction of T_total) or subset_max_steps so trials use more data.

    Parameters
    ----------
    data : np.ndarray (T, N) or pd.DataFrame
        Full time series.
    max_window : int
        Upper bound on window size (used to size the subset).
    n_windows_min : int
        Minimum number of windows to train on (e.g. 22 for stable metric).
    min_stride : int
        Minimum stride (used to size the subset).
    horizon : int
        Forecast horizon for holdout (last horizon steps = test).
    subset_ratio : float, optional
        If set in (0, 1], use at least this fraction of T_total for the subset (larger subset, slower trials).
    subset_max_steps : int, optional
        If set, use at least this many time steps for the subset (capped by T_total).

    Returns
    -------
    train_data : same type as data, shape (T_subset - horizon, N)
        Training portion (all but last horizon steps of the subset).
    test_true : np.ndarray, shape (horizon, N)
        Last horizon steps (ground truth for forecast metric).
    """
    if isinstance(data, pd.DataFrame):
        # Use only numeric columns (same as iVDFMDataset: exclude time index from variable set)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = list(data.columns)
        arr = data[numeric_cols].values
        is_df = True
        columns = numeric_cols
    else:
        arr = np.asarray(data)
        is_df = False
        columns = None

    T_total, N = arr.shape[0], arr.shape[1]
    min_stride = max(1, min_stride)
    n_windows_min = max(1, n_windows_min)
    horizon = max(1, horizon)

    T_subset = max_window + (n_windows_min - 1) * min_stride + horizon
    if subset_ratio is not None and 0 < subset_ratio <= 1:
        T_subset = max(T_subset, min(T_total, int(T_total * subset_ratio)))
    if subset_max_steps is not None and subset_max_steps > 0:
        T_subset = max(T_subset, min(T_total, subset_max_steps))
    T_subset = min(T_subset, T_total)
    if T_subset < horizon + 2:
        raise ValueError(
            f"Data too short: T_total={T_total}, T_subset={T_subset}; need at least horizon+2."
        )

    # Last T_subset time steps
    subset_arr = arr[-T_subset:, :]
    train_end = T_subset - horizon
    train_arr = subset_arr[:train_end, :]
    test_true = np.asarray(subset_arr[train_end:, :], dtype=np.float64)

    if is_df and columns is not None:
        train_data = pd.DataFrame(train_arr, columns=columns)
    else:
        train_data = train_arr

    return train_data, test_true


# -----------------------------------------------------------------------------
# Forecast metrics (sMSE / sMAE)
# -----------------------------------------------------------------------------

def compute_forecast_metric(
    preds: np.ndarray,
    trues: np.ndarray,
    metric: str = "sMSE",
) -> float:
    """Compute multivariate sMSE or sMAE (per-variable normalized then averaged).

    preds, trues: (horizon, N) or (n_origins, horizon, N); flattened to (n_points, N).
    """
    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    n = preds_flat.shape[1]
    if n == 0:
        return float("nan")

    out_per_var: List[float] = []
    for j in range(n):
        p_j = preds_flat[:, j]
        t_j = trues_flat[:, j]
        var_j = float(np.var(t_j))
        std_j = float(np.std(t_j)) if var_j >= 1e-20 else float("nan")
        if metric.upper() == "SMSE":
            if np.isfinite(var_j) and var_j >= 1e-10:
                mse_j = float(np.mean((p_j - t_j) ** 2))
                out_per_var.append(mse_j / var_j)
            else:
                out_per_var.append(float("nan"))
        elif metric.upper() == "SMAE":
            if np.isfinite(std_j) and std_j >= 1e-10:
                mae_j = float(np.mean(np.abs(p_j - t_j)))
                out_per_var.append(mae_j / std_j)
            else:
                out_per_var.append(float("nan"))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'sMSE' or 'sMAE'.")

    return float(np.nanmean(out_per_var)) if out_per_var else float("nan")


# -----------------------------------------------------------------------------
# Optuna suggest (hyperparameters)
# -----------------------------------------------------------------------------

def suggest_ivdfm_hyperparameters(
    trial: "optuna.Trial",
    max_window: int,
    max_regimes: int = 7,
    metric: str = "sMSE",
    fixed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Suggest iVDFM hyperparameters for a trial. Returns a dict suitable for iVDFM(**d) and dataset (window, stride)."""
    import optuna

    fixed = dict(fixed or {})

    def get(key: str, default: Any) -> Any:
        if key in fixed:
            return fixed[key]
        return default

    params: Dict[str, Any] = {}

    # Structural
    params["num_factors"] = trial.suggest_int("num_factors", 2, min(8, max(2, max_window // 50)), step=1)
    params["window"] = trial.suggest_int("window", 50, max(50, max_window), step=10)
    params["stride"] = trial.suggest_int("stride", 5, max(5, params["window"] // 5), step=5)
    params["num_regimes"] = trial.suggest_int("num_regimes", 1, max(1, max_regimes), step=1)
    params["regime_temperature"] = trial.suggest_float("regime_temperature", 0.1, 1.0)
    params["mixing"] = trial.suggest_categorical("mixing", [True, False])
    params["factor_order"] = trial.suggest_int("factor_order", 1, 5, step=1)

    # Encoder / decoder / prior
    params["encoder_hidden_dim"] = trial.suggest_categorical("encoder_hidden_dim", [64, 128, 200, 256])
    params["encoder_n_hidden_layers"] = trial.suggest_int("encoder_n_hidden_layers", 1, 2)
    params["decoder_hidden_dim"] = trial.suggest_categorical("decoder_hidden_dim", [64, 128, 200, 256])
    params["decoder_n_hidden_layers"] = trial.suggest_int("decoder_n_hidden_layers", 1, 2)
    params["prior_hidden_dim"] = trial.suggest_categorical("prior_hidden_dim", [32, 64, 96, 128])
    params["prior_n_hidden_layers"] = trial.suggest_int("prior_n_hidden_layers", 1, 2)

    # Training & regularization
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    params["max_epochs"] = trial.suggest_int("max_epochs", 5, 30, step=5)
    params["decoder_var"] = trial.suggest_float("decoder_var", 0.01, 0.1)
    params["beta_kl"] = trial.suggest_float("beta_kl", 0.8, 1.2)
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)

    # Fixed / passed through (exclude meta keys that are not iVDFM args)
    _exclude = {"max_regimes", "max_window", "n_windows_min", "min_stride"}
    for k, v in fixed.items():
        if k not in params and k not in _exclude:
            params[k] = v

    return params


# -----------------------------------------------------------------------------
# Objective and study
# -----------------------------------------------------------------------------

# Meta keys: used for Optuna/subsets only, not passed to iVDFM
_META_KEYS = {"max_regimes", "max_window", "n_windows_min", "min_stride"}


def create_objective(
    data: Union[np.ndarray, pd.DataFrame],
    device: Optional[torch.device],
    max_window: int,
    n_windows_min: int,
    min_stride: int,
    horizon: int,
    metric: str = "sMSE",
    max_regimes: int = 7,
    max_epochs_per_trial: Optional[int] = None,
    fixed: Optional[Dict[str, Any]] = None,
    seed: int = 2026,
    subset_ratio: Optional[float] = None,
    subset_max_steps: Optional[int] = None,
) -> Callable[["optuna.Trial"], float]:
    """Build an Optuna objective: suggest params, subset data, train iVDFM, predict, return metric."""

    def objective(trial: "optuna.Trial") -> float:
        import optuna

        try:
            params = suggest_ivdfm_hyperparameters(
                trial, max_window=max_window, max_regimes=max_regimes, metric=metric, fixed=fixed
            )
        except Exception as e:
            _logger.warning(f"Suggest failed: {e}")
            raise optuna.TrialPruned()

        train_data, test_true = get_optuna_subset(
            data,
            max_window=max_window,
            n_windows_min=n_windows_min,
            min_stride=min_stride,
            horizon=horizon,
            subset_ratio=subset_ratio,
            subset_max_steps=subset_max_steps,
        )
        T_train = len(train_data) if hasattr(train_data, "__len__") else train_data.shape[0]
        window = min(int(params.pop("window")), T_train - 1)
        window = max(2, window)
        stride = max(1, int(params.pop("stride")))

        try:
            dataset = iVDFMDataset(
                data=train_data,
                window=window,
                stride=stride,
                time_context=params.get("time_context", 2),
                device=device,
            )
        except Exception as e:
            _logger.warning(f"Dataset build failed: {e}")
            raise optuna.TrialPruned()

        data_dim = dataset.target_length
        context_dim = dataset.context_length
        if data_dim is None or data_dim < 1:
            raise optuna.TrialPruned()
        if context_dim is None or context_dim < 1:
            context_dim = 1

        max_epochs = max_epochs_per_trial if max_epochs_per_trial is not None else params.get("max_epochs", 15)
        params["max_epochs"] = max_epochs
        params["data_dim"] = data_dim
        params["context_dim"] = context_dim
        params["window"] = window
        params["device"] = device
        params["seed"] = seed
        # Match model time_context to dataset so predict() generates correct context dim
        params["time_context"] = context_dim

        dropout = params.pop("dropout", 0.0)
        if dropout is not None and dropout > 0:
            params["dropout"] = dropout

        # Drop meta keys so iVDFM does not receive them
        model_kwargs = {k: v for k, v in params.items() if v is not None and k not in _META_KEYS}
        try:
            model = iVDFM(**model_kwargs)
        except Exception as e:
            _logger.warning(f"Model build failed: {e}")
            raise optuna.TrialPruned()

        try:
            model.set_dataset(dataset)
            model.fit()
        except Exception as e:
            _logger.warning(f"Fit failed: {e}")
            raise optuna.TrialPruned()

        try:
            preds = model.predict(horizon=horizon, deterministic=True)
        except Exception as e:
            _logger.warning(f"Predict failed: {e}")
            raise optuna.TrialPruned()

        preds = np.asarray(preds)
        if preds.shape[0] != horizon:
            preds = preds[:horizon]
        if preds.shape[1] != test_true.shape[1]:
            test_true = test_true[:, : preds.shape[1]]

        value = compute_forecast_metric(preds, test_true, metric=metric)
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        return float(value)

    return objective


def run_hyperparameter_optimization(
    dataset: iVDFMDataset,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    max_window: int = 500,
    max_regimes: int = 7,
    n_windows_min: int = 22,
    min_stride: int = 1,
    horizon: int = 96,
    metric: str = "sMSE",
    max_epochs_per_trial: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 2026,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = False,
    pruner: Optional[Any] = None,
    fixed: Optional[Dict[str, Any]] = None,
    subset_ratio: Optional[float] = None,
    subset_max_steps: Optional[int] = None,
    min_batch_size: Optional[int] = None,
) -> Tuple[Dict[str, Any], "optuna.Study"]:
    """Run Optuna study to minimize forecast metric (sMSE or sMAE) on a subset of latest windows.

    Parameters
    ----------
    dataset : iVDFMDataset
        Pre-built dataset. Its target array is used for subsetting; time_context from the dataset
        is passed as fixed so per-trial datasets match.
    n_trials : int
        Number of trials.
    timeout : float, optional
        Total time budget in seconds.
    max_window : int
        Upper bound on window (and used to size subset).
    max_regimes : int
        Upper bound on num_regimes to suggest.
    n_windows_min : int
        Minimum number of training windows (subset size rule-of-thumb, e.g. 22).
    min_stride : int
        Minimum stride for subset sizing.
    horizon : int
        Forecast horizon for holdout metric.
    metric : str
        'sMSE' or 'sMAE' (minimize).
    max_epochs_per_trial : int, optional
        Cap epochs per trial; if None, use suggested max_epochs.
    device : torch.device, optional
        Device for training.
    seed : int
        Random seed.
    study_name : str, optional
        Optuna study name.
    storage : str, optional
        Optuna storage URL (e.g. sqlite).
    load_if_exists : bool
        If True, load existing study when name/storage match.
    pruner : optuna pruner, optional
        e.g. MedianPruner() or HyperbandPruner().
    fixed : dict, optional
        Fixed hyperparameters (not suggested).
    subset_ratio : float, optional
        If in (0, 1], use at least this fraction of train length for the subset (larger = slower trials, may transfer better).
    subset_max_steps : int, optional
        If set, use at least this many time steps for the subset (capped by data length).
    min_batch_size : int, optional
        If set, use this as batch_size for all trials (faster epochs than default 32).

    Returns
    -------
    best_params : dict
        Best hyperparameters (can be passed to iVDFM and dataset).
    study : optuna.Study
        Completed study.
    """
    import optuna

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.asarray(dataset.target)
    fixed = dict(fixed or {})
    fixed.setdefault("time_context", dataset.context_length)
    if min_batch_size is not None:
        fixed["batch_size"] = min_batch_size

    objective = create_objective(
        data=data,
        device=device,
        max_window=max_window,
        n_windows_min=n_windows_min,
        min_stride=min_stride,
        horizon=horizon,
        metric=metric,
        max_regimes=max_regimes,
        max_epochs_per_trial=max_epochs_per_trial,
        fixed=fixed,
        seed=seed,
        subset_ratio=subset_ratio,
        subset_max_steps=subset_max_steps,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name or "ivdfm_optuna",
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    try:
        best_params = dict(study.best_params)
    except ValueError:
        # No completed trials (all pruned or failed)
        _logger.warning("No completed trials; returning empty best_params.")
        best_params = {}
    if fixed:
        for k, v in fixed.items():
            if k not in _META_KEYS:
                best_params.setdefault(k, v)
    return best_params, study
