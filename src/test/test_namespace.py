"""``dfm_python._namespace.Namespace``: read-only, attribute access, dir, repr, lazy caching and pickling."""

import copy
import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from dfm_python._namespace import Field, Namespace

CALLS = {"factors": 0}


def _factors(data):
    CALLS["factors"] += 1
    return pd.DataFrame(data["f"], index=data["index"], columns=pd.Index(data["names"], name="factor"))


def _factors_shape(data):
    return data["f"].shape


def _common(data):
    return pd.DataFrame(data["f"] @ data["L"].T, index=data["index"])


def _make():
    index = pd.date_range("2020-01-31", periods=5, freq="ME", name="date")
    f = np.arange(10.0).reshape(5, 2)
    data = {
        "f": f,
        "L": np.ones((3, 2)),
        "index": index,
        "names": ["factor_0", "factor_1"],
        "loadings": pd.DataFrame(np.ones((3, 2)), index=["a", "b", "c"]),
        "contributions": None,
    }
    fields = [
        Field("factors", "E[f_t given the data]", "factor units", build=_factors, shape=_factors_shape),
        Field("loadings", "data units per factor unit", "data units"),
        Field("common", "the common component", "data units", build=_common),
        Field("contributions", "per-factor parts; None here", "data units"),
    ]
    return Namespace("Model.states_", fields, data)


def test_is_a_read_only_mapping():
    ns = _make()
    assert isinstance(ns, Mapping)
    assert list(ns) == ["factors", "loadings", "common", "contributions"]
    assert len(ns) == 4 and "factors" in ns and "nope" not in ns
    assert ns["loadings"] is ns.loadings
    with pytest.raises(KeyError):
        ns["nope"]
    with pytest.raises(AttributeError, match="read-only"):
        ns.factors = None
    with pytest.raises(AttributeError, match="read-only"):
        ns.new_field = 1
    with pytest.raises(AttributeError, match="read-only"):
        del ns.loadings
    with pytest.raises(TypeError):
        ns["factors"] = None
    assert not hasattr(ns, "__dict__")


def test_attribute_access_and_closest_name():
    ns = _make()
    pd.testing.assert_frame_equal(ns.factors, ns["factors"])
    with pytest.raises(AttributeError, match="Did you mean 'factors'"):
        _ = ns.factor
    assert not hasattr(ns, "regimes")


def test_dir_lists_the_fields():
    names = dir(_make())
    for name in ("factors", "loadings", "common", "contributions", "keys", "items"):
        assert name in names
    assert "_data" not in names


def test_field_that_does_not_apply_is_none():
    ns = _make()
    assert ns.contributions is None and ns["contributions"] is None
    assert dict(ns.items())["contributions"] is None


def test_lazy_fields_are_built_once_and_cached():
    ns = _make()
    before = CALLS["factors"]
    repr(ns)  # repr does not build an uncomputed field
    assert CALLS["factors"] == before
    first = ns.factors
    assert ns.factors is first and ns["factors"] is first
    assert CALLS["factors"] == before + 1
    assert first.columns.name == "factor" and first.index.name == "date"


def test_repr_has_one_line_per_field():
    ns = _make()
    text = repr(ns)
    lines = text.splitlines()
    assert lines[0] == "Model.states_ (4 fields)"
    assert len(lines) == 5
    assert "factors" in lines[1] and "DataFrame[5 x 2]" in lines[1] and "factor units" in lines[1]
    assert "E[f_t given the data]" in lines[1]
    assert "loadings" in lines[2] and "DataFrame[3 x 2]" in lines[2]
    assert "common" in lines[3] and "not built yet" in lines[3]
    assert "None (does not apply)" in lines[4]
    assert ns.common is not None  # builds the field
    assert "DataFrame[5 x 3]" in repr(ns).splitlines()[3]


def test_pickling_keeps_the_arrays_not_the_cache():
    ns = _make()
    built = (ns.factors, ns.common)
    assert all(isinstance(frame, pd.DataFrame) for frame in built)
    state = ns.__getstate__()
    assert set(state) == {"title", "fields", "data"}
    assert not any(isinstance(v, pd.DataFrame) and k in ("factors", "common") for k, v in state["data"].items())
    back = pickle.loads(pickle.dumps(ns))
    assert isinstance(back, Namespace) and back._cache == {}
    pd.testing.assert_frame_equal(back.factors, ns.factors)
    pd.testing.assert_frame_equal(back.common, ns.common)
    assert back.contributions is None


def test_deepcopy_is_independent():
    ns = _make()
    clone = copy.deepcopy(ns)
    assert clone is not ns and clone._data["f"] is not ns._data["f"]
    np.testing.assert_array_equal(clone.factors.to_numpy(), ns.factors.to_numpy())


def test_equality_is_identity():
    ns = _make()
    assert ns == ns and ns != _make()
    assert isinstance(hash(ns), int)


@pytest.mark.parametrize(
    "fields, match",
    [
        ([Field("a", "x"), Field("a", "y")], "unique"),
        ([Field("_a", "x")], "public identifiers"),
        ([Field("not ok", "x")], "public identifiers"),
        ([Field("keys", "x")], "shadow"),
        ([Field("missing", "x")], "no value"),
    ],
)
def test_invalid_fields(fields, match):
    with pytest.raises(ValueError, match=match):
        Namespace("bad", fields, {"a": 1, "_a": 1, "not ok": 1, "keys": 1})
