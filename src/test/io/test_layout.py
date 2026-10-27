"""config.json checks (formats 1 and 2), the single-file error and the index and frequency specs."""

import json

import pandas as pd
import pytest

from dfm_python.io.codecs import write_json
from dfm_python.io.layout import (
    FORMAT_2,
    FORMAT_VERSION,
    REQUIRED_FILES_2,
    index_from_spec,
    index_to_spec,
    offset_from_spec,
    offset_to_spec,
    read_config,
    read_config_2,
    saved_format,
)
from dfm_python.utils.errors import ConfigurationError


def _month_end(n, **kwargs):
    return pd.date_range("2000-01-31", periods=n, freq="ME", **kwargs)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 10, name="t"),
        pd.RangeIndex(5, 50, 5),
        _month_end(12, name="date"),
        pd.date_range("2020-01-01", periods=5, freq="D", tz="US/Eastern", name="ts"),
        pd.date_range("2020-01-01", periods=5, freq="h", unit="s"),
        pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-02-11"], name="irregular"),
        pd.period_range("2000Q1", periods=8, freq="Q", name="quarter"),
        pd.period_range("2000-01", periods=6, freq="M"),
        pd.RangeIndex(0, 4, name=7),
    ],
    ids=lambda i: f"{type(i).__name__}-{getattr(i, 'freqstr', None)}-{i.name}",
)
def test_index_round_trip(index):
    spec = index_to_spec(index)
    spec = json.loads(json.dumps(spec))
    back = index_from_spec(spec)
    assert type(back) is type(index) and back.equals(index) and back.name == index.name
    assert back.dtype == index.dtype
    assert getattr(back, "freq", None) == getattr(index, "freq", None)
    assert spec["length"] == len(index)
    assert spec["first"] == (int(index[0]) if isinstance(index, pd.RangeIndex) else str(index[0]))
    assert spec["last"] == (int(index[-1]) if isinstance(index, pd.RangeIndex) else str(index[-1]))


def test_index_spec_fields():
    index = _month_end(3, name="date")
    spec = index_to_spec(index)
    assert spec["kind"] == "datetime" and spec["freq"] == "ME" and spec["name"] == "date"
    assert spec["first"] == "2000-01-31 00:00:00" and spec["last"] == "2000-03-31 00:00:00"
    # The index's own resolution: ns under pandas 2, us for date_range under pandas 3.
    assert spec["unit"] == index.unit and spec["tz"] is None and len(spec["values"]) == 3
    assert index_to_spec(pd.RangeIndex(2, 8, 2))["kind"] == "range"


def test_index_spec_refuses_other_indexes():
    with pytest.raises(TypeError, match="Range, Datetime or Period"):
        index_to_spec(pd.Index([0.5, 1.5]))
    with pytest.raises(ConfigurationError, match="unknown index kind"):
        index_from_spec({"kind": "interval"})


def _config(**overrides):
    return {"format_version": FORMAT_VERSION, "model_type": "X", "params": {}, **overrides}


def test_read_config(tmp_path):
    write_json(tmp_path / "config.json", _config(params={"a": 1}))
    assert read_config(tmp_path)["params"] == {"a": 1}


@pytest.mark.parametrize("name", ["model.pkl", "model.pt", "run"])
def test_single_file_checkpoints_are_refused(tmp_path, name):
    (tmp_path / name).write_bytes(b"\x80\x04old checkpoint")
    with pytest.raises(ConfigurationError, match="(?s)single-file checkpoints.*release of dfm-python that wrote them"):
        read_config(tmp_path / name)


def test_directory_without_config_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(ConfigurationError, match="has no config.json.*single-file checkpoints"):
        read_config(tmp_path / "run")


def test_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_config(tmp_path / "nothing")


@pytest.mark.parametrize(
    "content, message",
    [
        (_config(format_version=2), "format_version 2"),
        ({"model_type": "X", "params": {}}, "lacks 'format_version'"),
        (_config(params=[1]), "'params' is not a JSON object"),
        ([1, 2], "does not hold a JSON object"),
    ],
)
def test_bad_configs(tmp_path, content, message):
    write_json(tmp_path / "config.json", content)
    with pytest.raises(ConfigurationError, match=message):
        read_config(tmp_path)


# ---------------------------------------------------------------------------
# Format 2
# ---------------------------------------------------------------------------


def test_integer_index_and_custom_business_day_round_trip():
    ints = pd.Index([0, 5], name="t")
    back = index_from_spec(json.loads(json.dumps(index_to_spec(ints))))
    assert back.equals(ints) and back.name == "t" and back.dtype == ints.dtype
    cbd = pd.offsets.CustomBusinessDay(holidays=["2024-01-15"])
    days = pd.date_range("2024-01-08", periods=8, freq=cbd, name="day")
    spec = json.loads(json.dumps(index_to_spec(days)))
    assert spec["freq"]["holidays"] == ["2024-01-15"]
    back = index_from_spec(spec)
    assert back.equals(days) and back.freq == cbd
    assert offset_from_spec(offset_to_spec(cbd)) == cbd
    assert offset_to_spec(None) is None and offset_from_spec(None) is None
    assert offset_from_spec(offset_to_spec(pd.offsets.MonthEnd())) == pd.offsets.MonthEnd()


def test_read_config_2(tmp_path):
    write_json(tmp_path / "config.json", _config(format_version=FORMAT_2, params={"a": {"spec": "X", "params": {}}}))
    config = read_config_2(tmp_path)
    assert config["params"] == {"a": {"spec": "X", "params": {}}}  # as written: decode_params rebuilds specs
    assert saved_format(tmp_path) == FORMAT_2


@pytest.mark.parametrize(
    "version, message",
    [(FORMAT_VERSION, r"saved by dfm-python 0.8 \(format 1\); load it with that version"), (3, "format_version 3")],
)
def test_read_config_2_refuses_other_formats(tmp_path, version, message):
    write_json(tmp_path / "config.json", _config(format_version=version))
    with pytest.raises(ConfigurationError, match=message):
        read_config_2(tmp_path)


def test_format_2_file_names():
    assert FORMAT_2 == 2
    assert REQUIRED_FILES_2 == ("config.json", "data.npz", "data.json", "fitted.npz", "fitted.json")
