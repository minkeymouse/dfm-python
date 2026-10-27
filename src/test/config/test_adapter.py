"""Tests for the config.adapter parsing helpers and their replacements for the removed config sources.

``ConfigSource``, ``YamlSource``, ``DictSource`` and ``make_config_source`` were
removed. Their tests are ported:

- ``DictSource(mapping).load()`` called ``_DFMEngineConfig.from_dict(mapping)``
  for every mapping tested here; those tests now call it directly, which runs
  the same helpers (``_extract_frequency_dict``, ``_normalize_blocks_dict``).
- ``YamlSource(path).load()`` read a flat file. Its tests now read the
  ``model:``/``data:`` layout with ``load_yaml`` and ``DFMConfig.from_yaml``;
  the frequencies live in ``data: freq``, which ``PanelData`` takes in the
  per-series form only.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import dfm_python
import dfm_python.config
import dfm_python.config.adapter as adapter
from dfm_python import DFM, DFMConfig, PanelData
from dfm_python.config import EstimatorConfig, load_yaml
from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.utils.errors import ConfigurationError, DataValidationError

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


class TestEngineConfigFromDict:
    """The former DictSource tests: ``_DFMEngineConfig.from_dict`` on the same mappings."""

    def test_from_dict_keeps_the_mapping(self):
        mapping = {
            'frequency': {'series1': 'd', 'series2': 'w'},
            'blocks': {'block1': {'num_factors': 2, 'series': ['series1', 'series2']}},
            'clock': 'd',
        }
        config = _DFMEngineConfig.from_dict(mapping)
        assert config.frequency == {'series1': 'd', 'series2': 'w'}
        assert config.blocks == {'block1': {'num_factors': 2, 'series': ['series1', 'series2']}}

    def test_from_dict_loading(self):
        mapping = {
            'frequency': {'series1': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'max_iter': 20
        }
        config = _DFMEngineConfig.from_dict(mapping)
        assert config.max_iter == 20
        assert 'block1' in config.blocks
        assert config.frequency == {'series1': 'm'}

    def test_grouped_frequency(self):
        mapping = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series3', 'series4'],
                'q': ['series5']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1', 'series3', 'series5']}},
            'clock': 'w',
            'max_iter': 20
        }
        config = _DFMEngineConfig.from_dict(mapping)
        assert config.frequency == {
            'series1': 'w', 'series2': 'w', 'series3': 'm', 'series4': 'm', 'series5': 'q',
        }
        assert config.max_iter == 20

    def test_grouped_frequency_duplicate_series_error(self):
        mapping = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series2', 'series3']  # series2 appears twice
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        with pytest.raises(ConfigurationError, match="appears in multiple frequency groups"):
            _DFMEngineConfig.from_dict(mapping)

    def test_grouped_frequency_invalid_freq_code(self):
        # Invalid frequency keys make the mapping the per-series form; validation then rejects it.
        mapping = {
            'frequency': {
                'invalid_freq': ['series1'],
                'm': ['series2']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        with pytest.raises(ConfigurationError):
            _DFMEngineConfig.from_dict(mapping)

    def test_per_series_frequency(self):
        mapping = {
            'frequency': {'series1': 'w', 'series2': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'clock': 'w'
        }
        config = _DFMEngineConfig.from_dict(mapping)
        assert config.frequency == {'series1': 'w', 'series2': 'm'}

    def test_grouped_frequency_empty_list(self):
        mapping = {
            'frequency': {
                'w': [],
                'm': ['series1']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        config = _DFMEngineConfig.from_dict(mapping)
        assert config.frequency == {'series1': 'm'}


class TestHelpers:
    """The private helpers ``_DFMEngineConfig`` uses, called directly."""

    def test_extract_frequency_dict_forms(self):
        grouped = {'frequency': {'m': ['a', 'b'], 'q': ['c']}}
        assert adapter._extract_frequency_dict(grouped) == {'a': 'm', 'b': 'm', 'c': 'q'}
        assert adapter._extract_frequency_dict({'frequency': {'a': 'm'}}) == {'a': 'm'}
        legacy = {'series': [{'series_id': 'a', 'frequency': 'q'}, {'series_id': 'b'}]}
        assert adapter._extract_frequency_dict(legacy, clock='w') == {'a': 'q', 'b': 'w'}
        assert adapter._extract_frequency_dict({}) is None

    def test_extract_frequency_dict_rejects_non_string_values(self):
        with pytest.raises(ConfigurationError, match="must be strings"):
            adapter._extract_frequency_dict({'frequency': {'a': 3}})

    def test_normalize_blocks_dict(self):
        blocks = {'g': {'factors': 2, 'series': ['a']}, 'h': {'num_factors': 1, 'series': 'a'}}
        assert adapter._normalize_blocks_dict(blocks) == {
            'g': {'num_factors': 2, 'series': ['a']},
            'h': {'num_factors': 1, 'series': []},
        }
        with pytest.raises(ConfigurationError, match="must be a dict"):
            adapter._normalize_blocks_dict({'g': 1})

    def test_detect_config_type_rejects_ddfm_parameters_in_from_dict(self):
        assert adapter.detect_config_type({'blocks': {'b': {}}, 'encoder_layers': [8]}) == adapter.MODEL_TYPE_DFM
        assert adapter.detect_config_type({'encoder_layers': [8]}) == adapter.MODEL_TYPE_DDFM
        assert adapter.detect_config_type({'model_type': 'deep'}) == adapter.MODEL_TYPE_DDFM
        with pytest.raises(ConfigurationError):
            adapter.detect_config_type([1, 2])
        with pytest.raises(ConfigurationError):
            _DFMEngineConfig.from_dict({'encoder_layers': [8], 'epochs': 2})


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
class TestYamlLayout:
    """The former YamlSource tests, on the model:/data: layout."""

    def test_yaml_loading(self, tmp_path):
        path = _write(tmp_path / "dfm.yaml", (
            "model:\n"
            "  blocks:\n    block1: {n_factors: 1, series: [series1]}\n"
            "  max_iter: 20\n"
            "data:\n  freq: {series1: m}\n"
        ))
        config = DFMConfig.from_yaml(path)
        assert config.max_iter == 20
        assert config.blocks == {'block1': {'n_factors': 1, 'series': ['series1']}}
        assert load_yaml(path)["data"] == {"freq": {"series1": "m"}}

    def test_yaml_per_series_frequency(self, tmp_path):
        path = _write(tmp_path / "dfm.yaml", (
            "model:\n"
            "  blocks:\n    block1: {n_factors: 2, series: [A001, KOEQUIPTE, KOGCFCNSD]}\n"
            "  clock: w\n  max_iter: 30\n"
            "data:\n"
            "  freq:\n"
            "    A001: w\n    GSCITOT: w\n"
            "    KOEQUIPTE: m\n    KOWRCCNSE: m\n    KOIPALL.G: m\n"
            "    KOGCFCNSD: q\n"
        ))
        config = DFMConfig.from_yaml(path)
        assert config.clock == 'w' and config.max_iter == 30
        names = ["A001", "GSCITOT", "KOEQUIPTE", "KOWRCCNSE", "KOIPALL.G", "KOGCFCNSD"]
        y = pd.DataFrame(np.zeros((3, len(names))), columns=names)
        panel = PanelData(y, freq=load_yaml(path)["data"]["freq"])
        assert panel.freq == {
            'A001': 'w', 'GSCITOT': 'w', 'KOEQUIPTE': 'm', 'KOWRCCNSE': 'm', 'KOIPALL.G': 'm', 'KOGCFCNSD': 'q',
        }

    def test_yaml_grouped_frequency_is_rejected(self, tmp_path):
        path = _write(tmp_path / "dfm.yaml", (
            "model:\n  clock: m\n"
            "data:\n  freq: {w: [], m: [series1, series2], q: []}\n"
        ))
        freq = load_yaml(path)["data"]["freq"]
        y = pd.DataFrame(np.zeros((3, 2)), columns=["series1", "series2"])
        with pytest.raises(DataValidationError, match="maps each series to its frequency code"):
            PanelData(y, freq=freq)

    def test_flat_file_is_refused(self, tmp_path):
        # The flat layout YamlSource read is not supported; the error says where the keys go.
        path = _write(tmp_path / "flat.yaml", "frequency: {series1: m}\nmax_iter: 20\n")
        with pytest.raises(ConfigurationError, match="model:"):
            DFMConfig.from_yaml(path)


class TestConfigSourcesRemoved:
    """The former ConfigSource test: the names are gone; the config mirror is the replacement."""

    @pytest.mark.parametrize("name", ["ConfigSource", "YamlSource", "DictSource", "make_config_source"])
    def test_names_are_gone(self, name):
        for namespace in (adapter, dfm_python.config, dfm_python):
            assert not hasattr(namespace, name), (namespace.__name__, name)
            assert name not in getattr(namespace, "__all__", ()), (namespace.__name__, name)

    @pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
    def test_config_mirror_builds_the_estimator(self, tmp_path):
        path = _write(tmp_path / "dfm.yaml", (
            "model:\n"
            "  blocks:\n    block1: {n_factors: 1, series: [series1]}\n"
        ))
        config = DFMConfig.from_yaml(path)
        assert isinstance(config, EstimatorConfig)
        model = DFM.from_config(config)
        assert isinstance(model, DFM) and model.blocks == {'block1': {'n_factors': 1, 'series': ['series1']}}
