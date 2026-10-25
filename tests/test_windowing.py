"""
tests/test_windowing.py
=======================
Unit tests for src/windowing.py and scripts/build_windows.py smoke.

Covers:
* canonical_window: shape, central selection, reflect-pad, exclude, empty
* dev_windows: stride, count, shape, no cross-capture
* sliding_windows: generic stride, tail window, short capture
* reflect_pad: symmetric padding, edge mirroring
* Determinism: same input → identical output
* build_windows smoke (subprocess)
"""

from __future__ import annotations

import csv
import subprocess
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from src.config import WindowConfig
from src.windowing import (
    canonical_window,
    count_windows,
    dev_windows,
    reflect_pad,
    sliding_windows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(
    length: int = 150,
    stride: int = 50,
    policy: str = "reflect_pad",
) -> WindowConfig:
    return WindowConfig(
        length=length,
        n_subcarriers=256,
        dev_stride=stride,
        val_stride=length,
        short_capture_policy=policy,
    )


def _make_arr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 256)).astype(np.float32)


# ---------------------------------------------------------------------------
# reflect_pad
# ---------------------------------------------------------------------------

class TestReflectPad:
    def test_pads_to_correct_length(self) -> None:
        arr = _make_arr(100)
        padded = reflect_pad(arr, 150)
        assert padded.shape == (150, 256)

    def test_no_pad_needed(self) -> None:
        arr = _make_arr(150)
        padded = reflect_pad(arr, 150)
        assert np.array_equal(padded, arr)

    def test_short_arr_padded(self) -> None:
        arr = _make_arr(50)
        padded = reflect_pad(arr, 150)
        assert padded.shape == (150, 256)

    def test_single_row(self) -> None:
        arr = _make_arr(1)
        padded = reflect_pad(arr, 10)
        assert padded.shape == (10, 256)

    def test_empty_raises(self) -> None:
        arr = np.zeros((0, 256), dtype=np.float32)
        with pytest.raises(ValueError):
            reflect_pad(arr, 150)

    def test_dtype_preserved(self) -> None:
        arr = _make_arr(80)
        padded = reflect_pad(arr, 150)
        assert padded.dtype == np.float32

    def test_longer_than_length_truncated(self) -> None:
        arr = _make_arr(200)
        padded = reflect_pad(arr, 150)
        assert padded.shape == (150, 256)
        # Should be first 150 rows
        assert np.array_equal(padded, arr[:150])

    def test_reflect_not_repeat(self) -> None:
        """reflect mode should mirror without repeating edge values.
        For a 2-row array padded to 4, reflect gives [B, A, B] pattern
        (not [A, A, B, B])."""
        arr = np.zeros((2, 256), dtype=np.float32)
        arr[0, 0] = 10.0  # row 0
        arr[1, 0] = 20.0  # row 1
        padded = reflect_pad(arr, 4)
        # reflect of [10, 20] → [20, 10, 20, 10] (mirror without edge repeat)
        # numpy reflect: [1,2] padded by 1 on each side → [2, 1, 2, 1]
        assert padded[0, 0] == 20.0  # mirrored from row 1
        assert padded[1, 0] == 10.0  # original row 0
        assert padded[2, 0] == 20.0  # original row 1
        assert padded[3, 0] == 10.0  # mirrored from row 0


# ---------------------------------------------------------------------------
# canonical_window
# ---------------------------------------------------------------------------

class TestCanonicalWindow:
    def test_shape_150_256(self) -> None:
        arr = _make_arr(200)
        w = canonical_window(arr, _cfg())
        assert w.shape == (150, 256)

    def test_dtype_float32(self) -> None:
        arr = _make_arr(200)
        w = canonical_window(arr, _cfg())
        assert w.dtype == np.float32

    def test_exact_length_passthrough(self) -> None:
        arr = _make_arr(150, seed=5)
        w = canonical_window(arr, _cfg())
        assert np.array_equal(w, arr)

    def test_central_selection(self) -> None:
        """The central 150 rows of a 300-row array should be rows 75–224."""
        arr = _make_arr(300, seed=9)
        w = canonical_window(arr, _cfg())
        expected = arr[75:225]
        assert np.array_equal(w, expected)

    def test_short_capture_reflect_pad(self) -> None:
        arr = _make_arr(100, seed=3)
        w = canonical_window(arr, _cfg(policy="reflect_pad"))
        assert w.shape == (150, 256)
        assert w.dtype == np.float32

    def test_short_capture_exclude(self) -> None:
        arr = _make_arr(100, seed=3)
        w = canonical_window(arr, _cfg(policy="exclude"))
        assert w is None

    def test_empty_array_exclude(self) -> None:
        arr = np.zeros((0, 256), dtype=np.float32)
        w = canonical_window(arr, _cfg(policy="exclude"))
        assert w is None

    def test_empty_array_reflect_pad(self) -> None:
        arr = np.zeros((0, 256), dtype=np.float32)
        w = canonical_window(arr, _cfg(policy="reflect_pad"))
        assert w is None  # cannot pad empty

    def test_wrong_n_subcarriers_raises(self) -> None:
        arr = np.ones((200, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="256"):
            canonical_window(arr, _cfg())

    def test_determinism(self) -> None:
        arr = _make_arr(250, seed=7)
        w1 = canonical_window(arr, _cfg())
        w2 = canonical_window(arr.copy(), _cfg())
        assert np.array_equal(w1, w2)

    def test_central_selection_odd_difference(self) -> None:
        """301 rows → central 150 = rows 75..225 (left-biased)."""
        arr = _make_arr(301, seed=11)
        w = canonical_window(arr, _cfg())
        start = (301 - 150) // 2  # 75
        assert np.array_equal(w, arr[start: start + 150])


# ---------------------------------------------------------------------------
# dev_windows
# ---------------------------------------------------------------------------

class TestDevWindows:
    def test_shape(self) -> None:
        arr = _make_arr(200)
        windows = dev_windows(arr, _cfg())
        for _, w in windows:
            assert w.shape == (150, 256)

    def test_count(self) -> None:
        arr = _make_arr(200)
        windows = dev_windows(arr, _cfg(stride=50))
        # starts: 0, 50 → 2 strided windows + possibly tail
        # 0+150=150 < 200, 50+150=200 == 200 → 2 windows, no tail needed
        assert len(windows) == 2

    def test_stride_values(self) -> None:
        arr = _make_arr(200)
        windows = dev_windows(arr, _cfg(stride=50))
        starts = [s for s, _ in windows]
        assert starts == [0, 50]

    def test_large_capture_multiple_windows(self) -> None:
        arr = _make_arr(500)
        windows = dev_windows(arr, _cfg(stride=50))
        assert len(windows) > 5

    def test_short_capture_reflect_pad(self) -> None:
        arr = _make_arr(80)
        windows = dev_windows(arr, _cfg(policy="reflect_pad"))
        assert len(windows) == 1
        assert windows[0][1].shape == (150, 256)

    def test_short_capture_exclude(self) -> None:
        arr = _make_arr(80)
        windows = dev_windows(arr, _cfg(policy="exclude"))
        assert windows == []

    def test_empty_capture(self) -> None:
        arr = np.zeros((0, 256), dtype=np.float32)
        windows = dev_windows(arr, _cfg())
        assert windows == []

    def test_exact_fit_no_tail(self) -> None:
        """300 rows, stride 50 → starts 0,50,100,150 → 4 windows, no tail."""
        arr = _make_arr(300)
        windows = dev_windows(arr, _cfg(stride=50))
        starts = [s for s, _ in windows]
        # 0+150=150, 50+150=200, 100+150=250, 150+150=300
        # last strided window ends at 300 = n → no tail
        assert starts == [0, 50, 100, 150]

    def test_tail_window_when_data_remains(self) -> None:
        """301 rows, stride 50 → last strided start=150, end=300 < 301 → tail at 151."""
        arr = _make_arr(301)
        windows = dev_windows(arr, _cfg(stride=50))
        starts = [s for s, _ in windows]
        # strided: 0, 50, 100, 150 → end=300 < 301 → tail at 301-150=151
        assert 151 in starts  # tail window
        assert starts[-1] == 151

    def test_windows_never_cross_captures(self) -> None:
        """If we process capture A and B separately, no window contains
        data from the other capture."""
        arr_a = _make_arr(200, seed=1)  # values 0-100
        arr_b = _make_arr(200, seed=2)  # different random values
        windows_a = dev_windows(arr_a, _cfg())
        windows_b = dev_windows(arr_b, _cfg())
        # Each window from A must come from arr_a only
        for start, w in windows_a:
            assert np.array_equal(w, arr_a[start: start + 150])
        for start, w in windows_b:
            assert np.array_equal(w, arr_b[start: start + 150])

    def test_determinism(self) -> None:
        arr = _make_arr(250, seed=7)
        w1 = dev_windows(arr, _cfg())
        w2 = dev_windows(arr.copy(), _cfg())
        assert len(w1) == len(w2)
        for (s1, a1), (s2, a2) in zip(w1, w2):
            assert s1 == s2
            assert np.array_equal(a1, a2)

    def test_dtype_float32(self) -> None:
        arr = _make_arr(200).astype(np.float64)
        windows = dev_windows(arr, _cfg())
        for _, w in windows:
            assert w.dtype == np.float32


# ---------------------------------------------------------------------------
# sliding_windows
# ---------------------------------------------------------------------------

class TestSlidingWindows:
    def test_shape(self) -> None:
        arr = _make_arr(300)
        windows = sliding_windows(arr, _cfg(), stride=100)
        for _, w in windows:
            assert w.shape == (150, 256)

    def test_stride_150_non_overlapping(self) -> None:
        arr = _make_arr(450)
        windows = sliding_windows(arr, _cfg(), stride=150)
        starts = [s for s, _ in windows]
        assert starts == [0, 150, 300]

    def test_stride_1(self) -> None:
        arr = _make_arr(152)
        windows = sliding_windows(arr, _cfg(), stride=1)
        assert len(windows) == 3  # starts 0, 1, 2

    def test_short_capture_single_padded_window(self) -> None:
        arr = _make_arr(50)
        windows = sliding_windows(arr, _cfg(policy="reflect_pad"), stride=50)
        assert len(windows) == 1
        assert windows[0][1].shape == (150, 256)

    def test_empty_returns_empty(self) -> None:
        arr = np.zeros((0, 256), dtype=np.float32)
        assert sliding_windows(arr, _cfg(), stride=50) == []

    def test_wrong_subcarriers_raises(self) -> None:
        arr = np.ones((200, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="256"):
            sliding_windows(arr, _cfg(), stride=50)


# ---------------------------------------------------------------------------
# count_windows
# ---------------------------------------------------------------------------

class TestCountWindows:
    def test_exact_fit(self) -> None:
        # n=300, length=150, stride=50 → 4 windows (0,50,100,150), no tail
        assert count_windows(300, 150, 50) == 4

    def test_with_tail(self) -> None:
        # n=301, length=150, stride=50 → 4 strided + 1 tail = 5
        assert count_windows(301, 150, 50) == 5

    def test_short_capture(self) -> None:
        # n=50 < 150 → 1 (reflect_pad)
        assert count_windows(50, 150, 50) == 1

    def test_empty(self) -> None:
        assert count_windows(0, 150, 50) == 0

    def test_non_overlapping(self) -> None:
        # n=450, length=150, stride=150 → 3
        assert count_windows(450, 150, 150) == 3


# ---------------------------------------------------------------------------
# build_windows smoke
# ---------------------------------------------------------------------------

class TestBuildWindowsSmoke:
    def test_smoke_runs_without_error(self) -> None:
        import sys
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "scripts/build_windows.py", "--smoke"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "smoke checks passed" in result.stdout


# ---------------------------------------------------------------------------
# Real-data integration (skipped if derived/ absent)
# ---------------------------------------------------------------------------

DERIVED_MANIFEST = Path("derived/manifest.csv")
DERIVED_INTERMEDIATE = Path("derived/intermediate")


@pytest.mark.skipif(
    not DERIVED_MANIFEST.exists() or not DERIVED_INTERMEDIATE.exists(),
    reason="Derived intermediate data not built",
)
class TestRealDataWindowing:
    def test_canonical_window_on_real_capture(self) -> None:
        from src.manifest import read_manifest
        from src.config import load_config

        cfg = load_config("configs/experiment.yaml")
        rows = read_manifest(DERIVED_MANIFEST)
        # Pick first val capture
        val_rows = [r for r in rows if r.split == "val"]
        assert len(val_rows) > 0, "No validation captures in manifest"

        row = val_rows[0]
        arr = np.load(row.intermediate_path)
        w = canonical_window(arr, cfg.window)
        assert w is not None
        assert w.shape == (150, 256)
        assert w.dtype == np.float32

    def test_dev_windows_on_real_capture(self) -> None:
        from src.manifest import read_manifest
        from src.config import load_config

        cfg = load_config("configs/experiment.yaml")
        rows = read_manifest(DERIVED_MANIFEST)
        dev_rows = [r for r in rows if r.split == "dev"]
        assert len(dev_rows) > 0

        row = dev_rows[0]
        arr = np.load(row.intermediate_path)
        windows = dev_windows(arr, cfg.window)
        assert len(windows) > 0
        for _, w in windows:
            assert w.shape == (150, 256)
