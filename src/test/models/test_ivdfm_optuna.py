"""Tests for models.ivdfm.optuna module and iVDFM.hyperparameter_optimization."""

import pytest
import numpy as np
import pandas as pd
import torch
import optuna
from unittest.mock import MagicMock

from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
from dfm_python.models.ivdfm.optuna import (
    get_optuna_subset,
    compute_forecast_metric,
    suggest_ivdfm_hyperparameters,
    create_objective,
    run_hyperparameter_optimization,
)
from dfm_python.models.ivdfm import iVDFM


def _make_mock_trial(suggest_ints=None, suggest_floats=None, suggest_categoricals=None):
    """Return a mock Optuna trial that records suggested values."""
    suggest_ints = suggest_ints or {}
    suggest_floats = suggest_floats or {}
    suggest_categoricals = suggest_categoricals or {}

    def suggest_int(name, low, high, step=1):
        return suggest_ints.get(name, low)

    def suggest_float(name, low, high, log=False):
        return suggest_floats.get(name, (low + high) / 2)

    def suggest_categorical(name, choices):
        return suggest_categoricals.get(name, choices[0])

    trial = MagicMock()
    trial.suggest_int = suggest_int
    trial.suggest_float = suggest_float
    trial.suggest_categorical = suggest_categorical
    return trial


class TestGetOptunaSubset:
    """Tests for get_optuna_subset."""

    def test_get_optuna_subset_array(self):
        """Subset from array: shapes and split are correct."""
        np.random.seed(42)
        T, N = 200, 5
        data = np.random.randn(T, N).astype(np.float32)
        max_window, min_stride, horizon = 50, 10, 12  # min_stride >= 10 in get_optuna_subset
        num_windows = (T - max_window) // min_stride + 1  # 16
        train_window_ratio = 10 / num_windows  # use 10 windows
        train_data, test_true = get_optuna_subset(
            data, max_window=max_window, min_stride=min_stride, horizon=horizon,
            train_window_ratio=train_window_ratio,
        )
        T_subset = (10 - 1) * min_stride + max_window + horizon  # 152
        assert train_data.shape[0] == T_subset - horizon
        assert train_data.shape[1] == N
        assert test_true.shape[0] == horizon
        assert test_true.shape[1] == N
        np.testing.assert_array_almost_equal(
            test_true,
            data[-horizon:, :],
        )

    def test_get_optuna_subset_dataframe(self):
        """Subset from DataFrame: numeric columns only, shapes correct."""
        np.random.seed(42)
        T, N = 200, 4
        data = pd.DataFrame(
            np.random.randn(T, N),
            columns=[f"x{i}" for i in range(N)],
        )
        max_window, min_stride, horizon = 40, 10, 10  # min_stride >= 10 in get_optuna_subset
        num_windows = (T - max_window) // min_stride + 1  # 17
        train_window_ratio = 8 / num_windows  # use 8 windows
        train_data, test_true = get_optuna_subset(
            data, max_window=max_window, min_stride=min_stride, horizon=horizon,
            train_window_ratio=train_window_ratio,
        )
        assert isinstance(train_data, pd.DataFrame)
        assert list(train_data.columns) == [f"x{i}" for i in range(N)]
        T_subset = (8 - 1) * min_stride + max_window + horizon  # 130
        assert len(train_data) == T_subset - horizon
        assert test_true.shape == (horizon, N)

    def test_get_optuna_subset_raises_short_data(self):
        """Too short data raises ValueError."""
        data = np.random.randn(5, 3)  # T=5, horizon+2 would need at least 3+2=5
        with pytest.raises(ValueError, match="Data too short"):
            get_optuna_subset(
                data, max_window=100, min_stride=10, horizon=10, train_window_ratio=1.0,
            )


class TestComputeForecastMetric:
    """Tests for compute_forecast_metric."""

    def test_compute_forecast_metric_smse(self):
        """sMSE returns finite value and is scale-normalized."""
        np.random.seed(42)
        horizon, N = 10, 4
        preds = np.random.randn(horizon, N)
        trues = np.random.randn(horizon, N)
        smse = compute_forecast_metric(preds, trues, metric="sMSE")
        assert np.isfinite(smse)
        assert smse >= 0.0

    def test_compute_forecast_metric_smae(self):
        """sMAE returns finite value."""
        np.random.seed(42)
        horizon, N = 10, 4
        preds = np.random.randn(horizon, N)
        trues = np.random.randn(horizon, N)
        smae = compute_forecast_metric(preds, trues, metric="sMAE")
        assert np.isfinite(smae)
        assert smae >= 0.0

    def test_compute_forecast_metric_perfect_prediction(self):
        """Perfect prediction gives sMSE ~0, sMAE ~0."""
        horizon, N = 10, 4
        trues = np.random.randn(horizon, N)
        preds = trues.copy()
        smse = compute_forecast_metric(preds, trues, metric="sMSE")
        smae = compute_forecast_metric(preds, trues, metric="sMAE")
        assert abs(smse) < 1e-10
        assert abs(smae) < 1e-10

    def test_compute_forecast_metric_unknown_raises(self):
        """Unknown metric raises ValueError."""
        preds = np.random.randn(5, 2)
        trues = np.random.randn(5, 2)
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_forecast_metric(preds, trues, metric="invalid")


class TestSuggestIvdfmHyperparameters:
    """Tests for suggest_ivdfm_hyperparameters."""

    def test_suggest_ivdfm_hyperparameters_returns_dict(self):
        """Suggested params contain expected keys and types."""
        trial = _make_mock_trial()
        params = suggest_ivdfm_hyperparameters(trial, max_window=200, max_regimes=5)
        assert "num_factors" in params
        assert "window" in params
        assert "stride" in params
        assert "num_regimes" in params
        assert "regime_temperature" in params
        assert "mixing" in params
        assert "factor_order" in params
        assert "encoder_hidden_dim" in params
        assert "decoder_hidden_dim" in params
        assert "learning_rate" in params
        assert "max_epochs" in params
        assert "decoder_var" in params
        assert "beta_kl" in params
        assert "dropout" in params
        assert isinstance(params["num_factors"], int)
        assert isinstance(params["window"], int)
        assert isinstance(params["mixing"], bool)

    def test_suggest_ivdfm_hyperparameters_respects_fixed(self):
        """Fixed params are in output; meta keys are not passed through."""
        trial = _make_mock_trial()
        fixed = {"innovation_distribution": "laplace", "max_regimes": 3}
        params = suggest_ivdfm_hyperparameters(trial, max_window=100, max_regimes=7, fixed=fixed)
        assert params.get("innovation_distribution") == "laplace"
        assert "max_regimes" not in params


class TestCreateObjective:
    """Tests for create_objective and one full trial."""

    @pytest.fixture
    def small_ts(self):
        """Small time series for fast trial."""
        np.random.seed(42)
        T, N = 150, 4
        return np.random.randn(T, N).astype(np.float32)

    def test_create_objective_one_trial_returns_finite(self, small_ts):
        """One Optuna trial runs and returns a finite metric."""
        import optuna
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        objective = create_objective(
            data=small_ts,
            device=device,
            max_window=60,
            train_window_ratio=0.5,
            min_stride=10,
            horizon=10,
            metric="sMSE",
            max_regimes=2,
            max_epochs_per_trial=2,
            fixed={"num_regimes": 1},
            seed=42,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
        if study.trials[0].state == optuna.trial.TrialState.COMPLETE:
            assert np.isfinite(study.trials[0].value)


class TestRunHyperparameterOptimization:
    """Tests for run_hyperparameter_optimization."""

    @pytest.fixture
    def small_ts(self):
        np.random.seed(43)
        return np.random.randn(120, 3).astype(np.float32)

    def test_run_hyperparameter_optimization_one_trial(self, small_ts):
        """run_hyperparameter_optimization with n_trials=1 returns best_params and study."""
        dataset = iVDFMDataset(data=small_ts, window=40, stride=2, time_context=1)
        best_params, study = run_hyperparameter_optimization(
            dataset=dataset,
            n_trials=1,
            max_window=40,
            train_window_ratio=0.25,
            min_stride=10,
            horizon=8,
            metric="sMSE",
            max_epochs_per_trial=2,
            max_regimes=1,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed=43,
        )
        assert isinstance(best_params, dict)
        assert len(study.trials) == 1
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            assert "num_factors" in best_params or "window" in best_params


class TestIvdfmHyperparameterOptimization:
    """Tests for iVDFM.hyperparameter_optimization method."""

    @pytest.fixture
    def small_ts(self):
        np.random.seed(44)
        return np.random.randn(100, 3).astype(np.float32)

    def test_ivdfm_hyperparameter_optimization_returns_best_params(self, small_ts):
        """model.set_dataset(dataset); model.hyperparameter_optimization(n_trials=1) returns dict of best params."""
        dataset = iVDFMDataset(data=small_ts, window=30, stride=2, time_context=2)
        model = iVDFM(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.set_dataset(dataset)
        best_params = model.hyperparameter_optimization(
            n_trials=1,
            max_window=30,
            train_window_ratio=0.5,
            min_stride=10,
            horizon=6,
            metric="sMSE",
            max_epochs_per_trial=2,
            max_regimes=1,
        )
        assert isinstance(best_params, dict)
        assert len(best_params) >= 1
