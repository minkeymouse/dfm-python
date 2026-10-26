"""The top-level ``dfm_python`` namespace."""

import pytest
import torch

import dfm_python
import dfm_python.config
from dfm_python.base import BaseFactorModel
from dfm_python.dataset.panel import PanelData
from dfm_python.io.loading import load
from dfm_python.logger import get_verbosity, set_verbosity
from dfm_python.models.afm.estimator import AFM
from dfm_python.models.ivdfm.estimator import iVDFM


def test_new_api_names_are_exported():
    assert dfm_python.PanelData is PanelData
    assert dfm_python.load is load
    assert dfm_python.set_verbosity is set_verbosity and dfm_python.get_verbosity is get_verbosity
    assert dfm_python.BaseFactorModel is BaseFactorModel
    assert dfm_python.AFM is AFM
    assert dfm_python.iVDFM is iVDFM
    for name in ("PanelData", "load", "set_verbosity", "get_verbosity", "BaseFactorModel", "AFM", "iVDFM"):
        assert name in dfm_python.__all__, name


def test_evaluation_module_is_exported():
    import dfm_python.evaluation as evaluation

    assert dfm_python.evaluation is evaluation and "evaluation" in dfm_python.__all__
    assert evaluation.__all__ == ["backtest", "BacktestResult"]
    assert callable(evaluation.backtest) and isinstance(evaluation.BacktestResult, type)


def test_every_name_in_all_exists():
    missing = [name for name in dfm_python.__all__ if not hasattr(dfm_python, name)]
    assert not missing


def test_afm_dataset_is_not_public():
    # The AFM dataset is the estimator's private adapter (dfm_python.dataset.afm_dataset).
    assert "AFMDataset" not in dfm_python.__all__
    assert not hasattr(dfm_python, "AFMDataset")


def test_afm_model_state_is_not_public():
    # AFMModelState is used only by the engine and AFM's save/load.
    assert "AFMModelState" not in dfm_python.config.__all__
    assert not hasattr(dfm_python.config, "AFMModelState")


def test_ivdfm_is_the_estimator():
    from dfm_python.models.ivdfm._engine import iVDFMEngine

    assert dfm_python.iVDFM is iVDFM and "iVDFM" in dfm_python.__all__
    assert issubclass(dfm_python.iVDFM, BaseFactorModel) and dfm_python.iVDFM is not iVDFMEngine


def test_ddfm_is_the_estimator():
    import dfm_python.models
    import dfm_python.models.ddfm as ddfm_pkg
    from dfm_python.models.ddfm._engine import DDFMEngine
    from dfm_python.models.ddfm.estimator import DDFM

    assert dfm_python.DDFM is DDFM and "DDFM" in dfm_python.__all__
    assert dfm_python.models.DDFM is DDFM and "DDFM" in dfm_python.models.__all__
    assert ddfm_pkg.DDFM is DDFM and "DDFM" in ddfm_pkg.__all__
    assert issubclass(DDFM, BaseFactorModel) and DDFM is not DDFMEngine
    for namespace in (dfm_python, dfm_python.models, ddfm_pkg):
        assert "DDFMEngine" not in getattr(namespace, "__all__", ()), namespace.__name__
        assert not hasattr(namespace, "DDFMEngine"), namespace.__name__


def test_dfm_is_the_estimator():
    import dfm_python.models
    import dfm_python.models.dfm as dfm_pkg
    from dfm_python.models.dfm._engine import DFMEngine
    from dfm_python.models.dfm.estimator import DFM

    assert dfm_python.DFM is DFM and "DFM" in dfm_python.__all__
    assert dfm_python.models.DFM is DFM and "DFM" in dfm_python.models.__all__
    assert dfm_pkg.DFM is DFM and "DFM" in dfm_pkg.__all__
    assert issubclass(DFM, BaseFactorModel) and DFM is not DFMEngine
    for namespace in (dfm_python, dfm_python.models, dfm_pkg):
        assert "DFMEngine" not in getattr(namespace, "__all__", ()), namespace.__name__
        assert not hasattr(namespace, "DFMEngine"), namespace.__name__


def test_ivdfm_dataset_is_not_public():
    # The iVDFM dataset is the estimator's private adapter base (dfm_python.dataset.ivdfm_dataset).
    import dfm_python.dataset

    assert "iVDFMDataset" not in dfm_python.__all__
    assert not hasattr(dfm_python, "iVDFMDataset")
    assert "iVDFMDataset" not in dfm_python.dataset.__all__
    assert not hasattr(dfm_python.dataset, "iVDFMDataset")


def test_ivdfm_model_state_is_not_public():
    # iVDFMModelState is used only by the engine and iVDFM's save/load.
    import dfm_python.config.schema as schema

    assert "iVDFMModelState" not in schema.__all__ and not hasattr(schema, "iVDFMModelState")
    assert "iVDFMModelState" not in dfm_python.config.__all__
    assert not hasattr(dfm_python.config, "iVDFMModelState")
    assert "iVDFMModelState" not in dfm_python.__all__ and not hasattr(dfm_python, "iVDFMModelState")


def test_ivdfm_engine_and_engine_config_are_not_public():
    import dfm_python.config.schema as schema
    import dfm_python.models
    import dfm_python.models.ivdfm as ivdfm_pkg

    for namespace in (dfm_python, dfm_python.config, schema, dfm_python.models, ivdfm_pkg):
        for name in ("iVDFMEngine", "_iVDFMEngineConfig"):
            assert name not in getattr(namespace, "__all__", ()), (namespace.__name__, name)
        assert not hasattr(namespace, "iVDFMEngine"), namespace.__name__
    assert not hasattr(dfm_python, "_iVDFMEngineConfig") and not hasattr(dfm_python.config, "_iVDFMEngineConfig")
    assert ivdfm_pkg.iVDFM is iVDFM and "iVDFM" in ivdfm_pkg.__all__


def test_ivdfm_optuna_module_is_gone():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dfm_python.models.ivdfm.optuna")


def test_ivdfm_config_is_the_estimator_mirror():
    from dfm_python.config import EstimatorConfig

    assert dfm_python.iVDFMConfig is dfm_python.config.iVDFMConfig is iVDFM._config_class
    assert issubclass(dfm_python.iVDFMConfig, EstimatorConfig)


def test_ddfm_dataset_is_not_public():
    # The DDFM dataset is the estimator's private adapter (dfm_python.dataset.ddfm_dataset).
    import dfm_python.dataset

    for namespace in (dfm_python, dfm_python.dataset):
        assert "DDFMDataset" not in namespace.__all__, namespace.__name__
        assert not hasattr(namespace, "DDFMDataset"), namespace.__name__


def test_ddfm_model_state_is_not_public():
    # DDFMModelState is the engine's training_state and the DDFMResult.fit_params type only;
    # its aliases DDFMTrainingState and DDFMStateSpaceParams are gone.
    import dfm_python.config.schema as schema
    import dfm_python.config.schema.params as params

    for name in ("DDFMModelState", "DDFMTrainingState", "DDFMStateSpaceParams"):
        for namespace in (dfm_python, dfm_python.config, schema):
            assert name not in namespace.__all__, (namespace.__name__, name)
            assert not hasattr(namespace, name), (namespace.__name__, name)
    assert hasattr(params, "DDFMModelState")
    assert not hasattr(params, "DDFMTrainingState") and not hasattr(params, "DDFMStateSpaceParams")


def test_ddfm_engine_is_not_public_and_has_no_old_entry_points():
    import dfm_python.models.ddfm as ddfm_pkg
    from dfm_python.models.ddfm._engine import DDFMEngine

    for namespace in (dfm_python, dfm_python.models, ddfm_pkg, dfm_python.config):
        assert "DDFMEngine" not in getattr(namespace, "__all__", ()), namespace.__name__
        assert not hasattr(namespace, "DDFMEngine"), namespace.__name__
    for name in ("save", "load", "update"):
        assert not hasattr(DDFMEngine, name), name
    assert not issubclass(DDFMEngine, BaseFactorModel) and DDFMEngine.__bases__ == (torch.nn.Module,)


def test_ddfm_logger_is_gone():
    import importlib

    import dfm_python.logger

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dfm_python.logger.ddfm_logger")
    for name in ("DDFMTrainLogger", "DDFMInferenceLogger"):
        assert name not in dfm_python.logger.__all__ and not hasattr(dfm_python.logger, name)


def test_ddfm_config_is_the_estimator_mirror():
    from dfm_python.config import EstimatorConfig
    from dfm_python.models.ddfm.estimator import DDFM

    assert dfm_python.DDFMConfig is dfm_python.config.DDFMConfig is DDFM._config_class
    assert issubclass(dfm_python.DDFMConfig, EstimatorConfig)


def test_dfm_config_is_the_estimator_mirror():
    from dfm_python.config import EstimatorConfig
    from dfm_python.config.schema.model import _DFMEngineConfig
    from dfm_python.models.dfm.estimator import DFM

    assert dfm_python.DFMConfig is dfm_python.config.DFMConfig is DFM._config_class
    assert issubclass(dfm_python.DFMConfig, EstimatorConfig)
    # The engine's config is private.
    assert dfm_python.DFMConfig is not _DFMEngineConfig
    for namespace in (dfm_python, dfm_python.config, dfm_python.config.schema):
        assert "_DFMEngineConfig" not in getattr(namespace, "__all__", ()), namespace.__name__


def test_dfm_dataset_is_not_public():
    # The DFM dataset is the estimator's private adapter (dfm_python.dataset.dfm_dataset).
    import dfm_python.dataset
    from dfm_python.dataset.dfm_dataset import DFMDataset

    for namespace in (dfm_python, dfm_python.dataset):
        assert "DFMDataset" not in namespace.__all__, namespace.__name__
        assert not hasattr(namespace, "DFMDataset"), namespace.__name__
    assert DFMDataset.__module__ == "dfm_python.dataset.dfm_dataset"


def test_dfm_state_space_params_alias_is_gone():
    import dfm_python.config.schema as schema
    import dfm_python.config.schema.params as params

    for namespace in (dfm_python, dfm_python.config, schema):
        assert "DFMStateSpaceParams" not in namespace.__all__, namespace.__name__
        assert not hasattr(namespace, "DFMStateSpaceParams"), namespace.__name__
    assert not hasattr(params, "DFMStateSpaceParams") and hasattr(params, "DFMModelState")


def test_dfm_engine_is_not_public_and_has_no_old_entry_points():
    import dfm_python.models
    import dfm_python.models.dfm as dfm_pkg
    from dfm_python.models.dfm._engine import DFMEngine

    for namespace in (dfm_python, dfm_python.models, dfm_pkg, dfm_python.config):
        assert "DFMEngine" not in getattr(namespace, "__all__", ()), namespace.__name__
        assert not hasattr(namespace, "DFMEngine"), namespace.__name__
    for name in ("save", "load", "load_config", "reset", "result"):
        assert not hasattr(DFMEngine, name), name
    assert not issubclass(DFMEngine, BaseFactorModel) and DFMEngine.__bases__ == (object,)
    # The DFM package exports the estimator only; the engine helpers stay in their submodules.
    assert dfm_pkg.__all__ == ["DFM"]
    from dfm_python.models.dfm.initialization import initialize_parameters  # noqa: F401
    from dfm_python.models.dfm.mixed_freq import find_slower_frequency, setup_mixed_frequency_params  # noqa: F401
    for name in ("find_slower_frequency", "setup_mixed_frequency_params", "initialize_parameters",
                 "build_dfm_structure", "log_blocks_diagnostics"):
        assert name not in dfm_pkg.__all__, name


# -- the final cleanup (spec section 13, items 14 and 15) -----------------------------


_DATASETS = ("DFMDataset", "DDFMDataset", "iVDFMDataset", "AFMDataset")


@pytest.mark.parametrize("name", _DATASETS)
def test_no_model_dataset_is_public(name):
    import dfm_python.dataset

    for namespace in (dfm_python, dfm_python.dataset):
        assert name not in namespace.__all__, (namespace.__name__, name)
        assert not hasattr(namespace, name), (namespace.__name__, name)


def test_dataset_package_exports_the_container_only():
    import dfm_python.dataset
    from dfm_python.dataset.time import TimeIndex

    assert dfm_python.dataset.__all__ == ["PanelData", "TimeIndex"]
    assert dfm_python.dataset.PanelData is PanelData and dfm_python.dataset.TimeIndex is TimeIndex
    # The datasets live on as the estimators' private adapters.
    from dfm_python.dataset.afm_dataset import AFMDataset  # noqa: F401
    from dfm_python.dataset.ddfm_dataset import DDFMDataset  # noqa: F401
    from dfm_python.dataset.dfm_dataset import DFMDataset  # noqa: F401
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset  # noqa: F401


@pytest.mark.parametrize(
    "name", ["ConfigSource", "YamlSource", "DictSource", "make_config_source", "detect_config_type", "BaseModelConfig"]
)
def test_config_sources_and_engine_config_base_are_not_exported(name):
    import dfm_python.config.schema as schema
    import dfm_python.utils

    for namespace in (dfm_python, dfm_python.config, schema, dfm_python.utils):
        assert name not in getattr(namespace, "__all__", ()), (namespace.__name__, name)
        assert not hasattr(namespace, name), (namespace.__name__, name)


@pytest.mark.parametrize("name", ["ConfigSource", "YamlSource", "DictSource", "make_config_source"])
def test_config_sources_are_deleted(name):
    import dfm_python.config.adapter as adapter

    assert not hasattr(adapter, name)


def test_config_adapter_keeps_the_engine_config_helpers():
    import dfm_python.config.adapter as adapter
    import dfm_python.config.schema.model as model

    for name in ("_raise_config_error", "_is_dict_like", "_convert_series_to_frequency_dict",
                 "_extract_frequency_dict", "_normalize_blocks_dict", "detect_config_type"):
        assert callable(getattr(adapter, name)), name
    assert (adapter.MODEL_TYPE_DFM, adapter.MODEL_TYPE_DDFM) == ("dfm", "ddfm")
    assert isinstance(model.BaseModelConfig, type)


@pytest.mark.parametrize("module", ["train_logger", "inference_logger", "ivdfm_logger", "dfm_logger", "ddfm_logger"])
def test_old_logger_modules_are_gone(module):
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"dfm_python.logger.{module}")


def test_logger_package_is_logger_and_progress():
    from pathlib import Path

    import dfm_python.logger
    import dfm_python.logger.progress as progress

    package = Path(dfm_python.logger.__file__).parent
    assert sorted(p.name for p in package.glob("*.py")) == ["__init__.py", "logger.py", "progress.py"]
    assert dfm_python.logger.__all__ == [
        "get_logger", "setup_logging", "configure_logging", "set_verbosity", "get_verbosity",
        "fit_verbosity", "quiet_engine_loggers",
    ]
    for name in ("BaseTrainLogger", "DFMTrainLogger", "BaseInferenceLogger", "DFMInferenceLogger",
                 "iVDFMInferenceLogger", "log_training_start", "log_training_step", "log_training_end",
                 "log_training_epoch", "log_inference_start", "log_inference_step", "log_inference_end",
                 "log_prediction"):
        assert not hasattr(dfm_python.logger, name), name
        assert not hasattr(progress, name), name
    # What the engines call moved to progress.py.
    for name in ("log_em_iteration", "log_convergence", "iVDFMTrainLogger"):
        assert hasattr(progress, name), name


def test_engines_import_their_progress_reporting_from_progress():
    import dfm_python.functional.em as em
    import dfm_python.logger.progress as progress
    import dfm_python.models.dfm._engine as dfm_engine
    import dfm_python.models.ivdfm._engine as ivdfm_engine
    import dfm_python.numeric.builder as builder

    assert em.log_em_iteration is progress.log_em_iteration and em.log_convergence is progress.log_convergence
    assert ivdfm_engine.iVDFMTrainLogger is progress.iVDFMTrainLogger
    assert dfm_engine.log_blocks_diagnostics is builder.log_blocks_diagnostics


def test_models_package_exports_the_estimators_and_results():
    import dfm_python.models

    assert dfm_python.models.__all__ == ["DFM", "DDFM", "iVDFM", "AFM", "BaseResult", "DFMResult", "DDFMResult"]
    for name in ("DFM", "DDFM", "iVDFM", "AFM"):
        assert getattr(dfm_python.models, name) is getattr(dfm_python, name), name


def test_estimate_var_is_defined_once():
    import inspect

    import dfm_python.numeric.estimator as estimator

    source = inspect.getsource(estimator)
    assert source.count("\ndef estimate_var(") == 1
    assert list(inspect.signature(estimator.estimate_var).parameters)[:2] == ["y", "x"]
