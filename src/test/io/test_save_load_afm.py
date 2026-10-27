"""Saving and loading AFM runs: ``save``, ``AFM.load`` and ``dfm_python.load``."""

import warnings

import dfm_python
from dfm_python import AFM
from dfm_python.testing import make_panel


def _fitted():
    model = AFM(**AFM.get_test_params()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(make_panel(AFM))


def test_package_load_dispatches_to_the_model(tmp_path):
    model = _fitted()
    model.save(tmp_path / "run")
    loaded = dfm_python.load(tmp_path / "run")
    assert type(loaded) is AFM
    assert loaded.get_params() == model.get_params()


def test_round_trip_reproduces_the_outputs(tmp_path):
    model = _fitted()
    model.save(tmp_path / "run")
    loaded = AFM.load(tmp_path / "run", device="cpu")
    assert loaded.index_.equals(model.index_) and list(loaded.feature_names_in_) == list(model.feature_names_in_)
    assert loaded.stop_reason_ == model.stop_reason_ and loaded.n_iter_ == model.n_iter_
    data = make_panel(AFM)  # a loaded AFM keeps no training data: allocate on the panel
    assert loaded.allocate(data=data).to_numpy().tobytes() == model.allocate(data=data).to_numpy().tobytes()
