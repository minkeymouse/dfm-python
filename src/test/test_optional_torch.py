"""torch is optional: the numpy DFM imports and runs without the ``deep`` extra.

Each test runs a fresh interpreter. ``_NO_DEEP`` installs a meta path finder
that makes ``torch`` and ``einops`` unimportable, as in an install without the
extra; the tests without it check that the DFM workflow does not import torch
even where it is installed.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import dfm_python

# The interpreter must import this dfm_python, whether installed or from the source tree.
_SOURCE_ROOT = str(Path(dfm_python.__file__).resolve().parents[1])

_NO_DEEP = """
import sys

class _NoDeep:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("torch", "einops"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _NoDeep())
"""

_DFM_RUN = """
import sys
import tempfile
from pathlib import Path

import numpy as np

import dfm_python
from dfm_python import DFM
from dfm_python.testing import make_panel

panel = make_panel(DFM)
train, new = panel.loc(None, panel.index[-7]), panel.loc(panel.index[-6], None)
model = DFM(**DFM.get_test_params()[0]).fit(train)
forecast = model.predict(3)
assert forecast.shape == (3, 3) and np.isfinite(forecast.to_numpy()).all()
assert model.predict_interval(3, coverage=0.8).shape == (3, 6)
assert model.predict_factors(3).shape[0] == 3
assert model.decompose(by="factor").shape[0] == len(train.index)
with tempfile.TemporaryDirectory() as tmp:
    model.save(Path(tmp) / "run")
    for loaded in (DFM.load(Path(tmp) / "run"), dfm_python.load(Path(tmp) / "run")):
        assert loaded.predict(3).equals(forecast)
assert model.update(new).predict(3).index[0] > new.index[-1]
assert "torch" not in sys.modules and "einops" not in sys.modules, "the DFM workflow imported torch"
"""


def _run(code: str, *, no_deep: bool) -> None:
    """Run ``code`` in a fresh interpreter (without torch and einops if ``no_deep``); fail on a non-zero exit."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_SOURCE_ROOT, env.get("PYTHONPATH")) if p)
    script = (_NO_DEEP if no_deep else "") + textwrap.dedent(code)
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr


def test_import_does_not_import_torch():
    _run(
        """
        import sys

        import dfm_python

        assert "torch" not in sys.modules and "einops" not in sys.modules, "import dfm_python imported torch"

        from dfm_python import AFM, DDFM, DFM, PanelData, iVDFM, load
        import dfm_python.models, dfm_python.ssm, dfm_python.testing, dfm_python.tuning
        from dfm_python.config import *
        from dfm_python.utils import *
        from dfm_python.registry import all_estimators

        assert [name for name, _ in all_estimators()] == ["AFM", "DDFM", "DFM", "iVDFM"]
        for cls in (AFM, DDFM, DFM, iVDFM):
            cls().get_params()
        loaded = sorted({"torch", "einops"} & set(sys.modules))
        assert not loaded, f"the public modules or a constructor imported {loaded}"
        """,
        no_deep=False,
    )


@pytest.mark.parametrize("no_deep", [True, False], ids=["without_torch", "with_torch"])
def test_dfm_round_trip_does_not_need_torch(no_deep):
    _run(_DFM_RUN, no_deep=no_deep)


def test_dfm_passes_the_contract_checks_without_torch():
    _run(
        """
        from dfm_python import DFM
        from dfm_python.testing import check_estimator

        assert "save_load" in check_estimator(DFM)
        """,
        no_deep=True,
    )


def test_torch_models_build_without_torch_and_name_the_extra_when_used():
    _run(
        """
        import importlib.util
        import json
        import tempfile
        from pathlib import Path

        import numpy as np
        from sklearn.base import clone

        import dfm_python
        import dfm_python.models.ddfm
        import dfm_python.ssm
        from dfm_python import AFM, DDFM, iVDFM
        from dfm_python.testing import check_estimator
        from dfm_python.tuning import tune_hyperparameters

        def raises_naming_the_extra(call, what):
            try:
                call()
            except ImportError as err:
                assert 'pip install "dfm-python[deep]"' in str(err) and what in str(err), str(err)
            else:
                raise AssertionError(f"{what} did not raise ImportError")

        data = np.random.default_rng(0).standard_normal((40, 3))
        for cls in (DDFM, iVDFM, AFM):
            model = cls()
            assert clone(model).get_params() == model.get_params() == model.get_config().to_dict()
            raises_naming_the_extra(lambda: model.fit(data), f"{cls.__name__}.fit")
            assert not hasattr(model, "result_")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"format_version": 1, "model_type": "DDFM", "params": {}}))
            raises_naming_the_extra(lambda: dfm_python.load(run), "DDFM.load")
        raises_naming_the_extra(lambda: check_estimator(DDFM), "check_estimator")
        if importlib.util.find_spec("optuna") is not None:  # without it, the tune extra is named first
            raises_naming_the_extra(lambda: tune_hyperparameters(iVDFM(), data), "tune_hyperparameters")
        raises_naming_the_extra(lambda: dfm_python.models.ddfm.DDFMEncoder, "DDFMEncoder")
        raises_naming_the_extra(lambda: dfm_python.ssm.iVDFMCompanionSSM, "iVDFMCompanionSSM")
        """,
        no_deep=True,
    )


def test_torch_stand_in_resolves_attributes_and_ignores_dunders():
    torch = pytest.importorskip("torch")
    from dfm_python.utils.optional import torch as deferred

    assert deferred.float32 is torch.float32 and deferred.device("cpu") == torch.device("cpu")
    assert not hasattr(deferred, "__wrapped__")
