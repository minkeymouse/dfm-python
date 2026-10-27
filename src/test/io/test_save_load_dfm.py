"""Saving and loading DFM runs: ``save``, ``DFM.load`` and ``dfm_python.load``."""

import warnings

import dfm_python
from dfm_python import DFM
from dfm_python.testing import make_panel


def _fitted():
    model = DFM(**DFM.get_test_params()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(make_panel(DFM))


def test_package_load_dispatches_to_the_model(tmp_path):
    model = _fitted()
    model.save(tmp_path / "run")
    loaded = dfm_python.load(tmp_path / "run")
    assert type(loaded) is DFM
    assert loaded.get_params() == model.get_params()


def test_round_trip_reproduces_the_outputs(tmp_path):
    model = _fitted()
    model.save(tmp_path / "run")
    loaded = DFM.load(tmp_path / "run")
    assert loaded.index_.equals(model.index_) and list(loaded.feature_names_in_) == list(model.feature_names_in_)
    assert loaded.stop_reason_ == model.stop_reason_ and loaded.n_iter_ == model.n_iter_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert loaded.predict(3).to_numpy().tobytes() == model.predict(3).to_numpy().tobytes()
