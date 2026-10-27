"""Unit tests for the golden-suite helpers (pinning, fingerprint, comparison)."""

import json
import sys

import numpy as np
import pytest
import threadpoolctl
import torch

from . import _golden
from ._cases import CASES
from ._cases_common import Case
from ._golden import (
    TOLERANCES,
    GoldenToleranceWarning,
    assert_golden,
    fingerprint,
    load_golden,
    matches_capture_platform,
    pinned,
    skip_unless_comparable,
)


def _write_fixture(directory, name, arrays, fp):
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    (directory / f"{name}.json").write_text(json.dumps({"name": name, "fingerprint": fp}))


@pytest.fixture
def fixture_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_golden, "FIXTURES_DIR", tmp_path)
    return tmp_path


# No registered case opts out of the cross-platform comparison since the DFM
# numerical-stability rewrite (the former dfm_macro_weekly_monthly and
# dfm_macro_monthly_quarterly_update opt-outs agree within 1e-6 relative under
# forced AVX2/Sandybridge/SSE kernels). The mechanism is tested on a stand-in case.
PINNED_TO_CAPTURE_PLATFORM: set[str] = set()
PINNED_STAND_IN = "pinned_stand_in"
RTOL_F32, ATOL_F32 = TOLERANCES[np.dtype(np.float32)]
RTOL_F64, ATOL_F64 = TOLERANCES[np.dtype(np.float64)]

ARRAYS = {
    "f64": np.array([1.0, 2.0, 3.0]),
    "f32": np.array([1.0, 2.0], dtype=np.float32),
    "i": np.int64(7),
}


class TestPinned:
    def test_sets_and_restores(self):
        torch_threads = torch.get_num_threads()
        det = torch.are_deterministic_algorithms_enabled()
        with pinned():
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled()
            assert all(lib["num_threads"] == 1 for lib in threadpoolctl.threadpool_info())
        assert torch.get_num_threads() == torch_threads
        assert torch.are_deterministic_algorithms_enabled() == det

    def test_restores_on_error(self):
        torch_threads = torch.get_num_threads()
        det = torch.are_deterministic_algorithms_enabled()
        with pytest.raises(RuntimeError), pinned():
            raise RuntimeError("boom")
        assert torch.get_num_threads() == torch_threads
        assert torch.are_deterministic_algorithms_enabled() == det

    def test_fixture(self, pinned):
        assert torch.get_num_threads() == 1
        assert torch.are_deterministic_algorithms_enabled()


class TestFingerprint:
    def test_keys(self):
        fp = fingerprint()
        assert set(fp) == {"machine", "processor", "cpu_model", "cpu_capability", "python", "numpy", "torch",
                           "blas"}
        assert fp["numpy"] == np.__version__
        assert fp["torch"] == torch.__version__
        assert fp["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
        assert fp["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
        assert isinstance(fp["cpu_model"], str)
        assert all(e["internal_api"] for e in fp["blas"])

    def test_independent_of_thread_count(self):
        outside = fingerprint()
        with pinned():
            assert fingerprint() == outside


class TestAssertGolden:
    def test_load_round_trip(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        manifest, arrays = load_golden("case")
        assert manifest["name"] == "case"
        assert set(arrays) == set(ARRAYS)

    def test_bitwise_pass(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        assert_golden(dict(ARRAYS), "case")

    def test_bitwise_detects_last_bit(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        actual = dict(ARRAYS, f64=np.nextafter(ARRAYS["f64"], np.inf))
        with pytest.raises(AssertionError):
            assert_golden(actual, "case")

    def test_fallback_warns_and_uses_tolerance(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        actual = dict(ARRAYS, f64=ARRAYS["f64"] * (1 + RTOL_F64 / 2), f32=ARRAYS["f32"] * np.float32(1 + RTOL_F32 / 2))
        with pytest.warns(GoldenToleranceWarning, match="assert_allclose"):
            assert_golden(actual, "case")

    @pytest.mark.parametrize("key, factor", [("f64", 1 + 2 * RTOL_F64), ("f32", np.float32(1 + 2 * RTOL_F32))])
    def test_fallback_rejects_beyond_tolerance(self, fixture_dir, key, factor):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        actual = dict(ARRAYS, **{key: ARRAYS[key] * factor})
        with pytest.warns(GoldenToleranceWarning), pytest.raises(AssertionError):
            assert_golden(actual, "case")

    @pytest.mark.parametrize("dtype, atol", [(np.float64, ATOL_F64), (np.float32, ATOL_F32)])
    def test_fallback_atol_covers_entries_near_zero(self, fixture_dir, dtype, atol):
        key = "f64" if dtype is np.float64 else "f32"
        arrays = dict(ARRAYS, **{key: np.array([0.0, 1e-9, 1.0], dtype=dtype)})
        _write_fixture(fixture_dir, "case", arrays, {"machine": "elsewhere"})
        with pytest.warns(GoldenToleranceWarning):
            assert_golden(dict(arrays, **{key: np.array([atol / 2, -1e-9, 1.0], dtype=dtype)}), "case")
        with pytest.warns(GoldenToleranceWarning), pytest.raises(AssertionError):
            assert_golden(dict(arrays, **{key: np.array([2 * atol, 1e-9, 1.0], dtype=dtype)}), "case")

    def test_fallback_keeps_integers_exact(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, {"machine": "elsewhere"})
        with pytest.warns(GoldenToleranceWarning), pytest.raises(AssertionError):
            assert_golden(dict(ARRAYS, i=np.int64(8)), "case")

    @pytest.mark.parametrize("actual", [
        {k: v for k, v in ARRAYS.items() if k != "i"},               # missing key
        dict(ARRAYS, extra=np.zeros(1)),                              # unexpected key
        dict(ARRAYS, f64=ARRAYS["f64"].astype(np.float32)),           # dtype
        dict(ARRAYS, f64=ARRAYS["f64"][None, :]),                     # shape
    ], ids=["missing", "extra", "dtype", "shape"])
    def test_structure_mismatch(self, fixture_dir, actual):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        with pytest.raises(AssertionError):
            assert_golden(actual, "case")

    def test_exclude_leaves_fixture_outputs_out(self, fixture_dir):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        partial = {k: v for k, v in ARRAYS.items() if k != "i"}
        assert_golden(partial, "case", exclude=["i"])
        with pytest.raises(AssertionError):  # the remaining outputs are still compared
            assert_golden(dict(partial, f64=np.nextafter(ARRAYS["f64"], np.inf)), "case", exclude=["i"])

    @pytest.mark.parametrize("exclude", [["nope"], ["i"]], ids=["not-a-fixture-key", "also-given"])
    def test_exclude_must_name_absent_fixture_outputs(self, fixture_dir, exclude):
        _write_fixture(fixture_dir, "case", ARRAYS, fingerprint())
        with pytest.raises(AssertionError, match="excluded"):
            assert_golden(dict(ARRAYS), "case", exclude=exclude)


@pytest.fixture
def pinned_case(monkeypatch):
    """Register a stand-in case with ``cross_platform=False`` for the duration of a test."""
    monkeypatch.setitem(CASES, PINNED_STAND_IN, Case(PINNED_STAND_IN, "dfm", lambda: None, cross_platform=False))
    return PINNED_STAND_IN


class TestCrossPlatformOptOut:
    def test_registered_opt_outs(self):
        assert {name for name, case in CASES.items() if not case.cross_platform} == PINNED_TO_CAPTURE_PLATFORM

    def test_matches_capture_platform(self, fixture_dir):
        _write_fixture(fixture_dir, "here", ARRAYS, fingerprint())
        _write_fixture(fixture_dir, "elsewhere", ARRAYS, dict(fingerprint(), cpu_model="another CPU"))
        assert matches_capture_platform("here")
        assert not matches_capture_platform("elsewhere")

    def test_skips_off_the_capture_platform(self, fixture_dir, pinned_case):
        name = pinned_case
        _write_fixture(fixture_dir, name, ARRAYS, dict(fingerprint(), python="0.0"))
        with pytest.raises(pytest.skip.Exception, match=f"golden '{name}' is pinned bitwise on the capture platform"):
            skip_unless_comparable(name)
        with pytest.raises(pytest.skip.Exception):  # assert_golden skips instead of comparing
            assert_golden(dict(ARRAYS, f64=ARRAYS["f64"] + 1), name)

    def test_compared_bitwise_on_the_capture_platform(self, fixture_dir, pinned_case):
        name = pinned_case
        _write_fixture(fixture_dir, name, ARRAYS, fingerprint())
        skip_unless_comparable(name)  # no skip
        with pytest.raises(AssertionError):
            assert_golden(dict(ARRAYS, f64=np.nextafter(ARRAYS["f64"], np.inf)), name)

    def test_other_cases_and_names_are_never_skipped(self, fixture_dir):
        for name in ("dfm_synthetic", "dfm_macro_weekly_monthly", "dfm_macro_monthly_quarterly_update", "case"):
            _write_fixture(fixture_dir, name, ARRAYS, {"machine": "elsewhere"})
            skip_unless_comparable(name)
