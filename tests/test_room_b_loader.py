"""
tests/test_room_b_loader.py
===========================
Unit tests for src/room_b_loader.py.

All tests use tmp_path; no real dataset files are required unless the
real-data smoke class runs (skipped automatically when the dataset is absent).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.room_b_loader import (
    N_SUBCARRIERS,
    PREFIX_TO_ACTIVITY,
    discover_room_b,
    load_capture,
    load_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, n_rows: int = 10,
               header: str = "named", fill: float = 1.0) -> None:
    """Write a synthetic 256-column CSV to *path*."""
    if header == "named":
        cols = [f"Subcarrier_{i+1}" for i in range(256)]
    else:  # zero-indexed
        cols = [str(i) for i in range(256)]
    data = np.full((n_rows, 256), fill, dtype=np.float32)
    pd.DataFrame(data, columns=cols).to_csv(path, index=False)


def _make_dir(tmp: Path) -> Path:
    """Create a fake Room-B directory with a few files per activity."""
    d = tmp / "room_b"
    d.mkdir()

    files = {
        "Empty_Amp_1.csv":    ("empty",    "named"),
        "Empty_Amp_2.csv":    ("empty",    "named"),
        "Empty_11.csv":       ("empty",    "zero"),
        "Lying_Amp_1.csv":    ("lying",    "named"),
        "Lying_0.csv":        ("lying",    "zero"),
        "Sitting_Amp_1.csv":  ("sitting",  "named"),
        "Sitting_Amp_2.csv":  ("sitting",  "named"),
        "Standing_Amp_1.csv": ("standing", "named"),
        "Walking_1.csv":      ("walking",  "named"),
        "Walking_2.csv":      ("walking",  "named"),
        "Walking_0.csv":      ("walking",  "zero"),
        "model_plot_1.png":   None,   # should be skipped
    }
    for fname, info in files.items():
        p = d / fname
        if info is None:
            p.touch()
        else:
            _, hdr = info
            _write_csv(p, n_rows=5, header=hdr)
    return d


# ---------------------------------------------------------------------------
# discover_room_b
# ---------------------------------------------------------------------------

class TestDiscoverRoomB:
    def test_returns_all_five_activities(self, tmp_path: Path) -> None:
        d = _make_dir(tmp_path)
        groups = discover_room_b(d)
        assert set(groups.keys()) == {"empty", "lying", "sit", "stand", "walk"}

    def test_counts_correct(self, tmp_path: Path) -> None:
        d = _make_dir(tmp_path)
        groups = discover_room_b(d)
        assert len(groups["empty"]) == 3   # Amp_1, Amp_2, 11.csv
        assert len(groups["lying"]) == 2   # Amp_1, 0.csv
        assert len(groups["sit"])   == 2   # Amp_1, Amp_2
        assert len(groups["stand"]) == 1   # Amp_1
        assert len(groups["walk"])  == 3   # 0, 1, 2

    def test_skips_non_csv_files(self, tmp_path: Path) -> None:
        d = _make_dir(tmp_path)
        groups = discover_room_b(d)
        for paths in groups.values():
            assert all(p.suffix.lower() == ".csv" for p in paths)

    def test_sorted_deterministically(self, tmp_path: Path) -> None:
        d = _make_dir(tmp_path)
        g1 = discover_room_b(d)
        g2 = discover_room_b(d)
        for act in g1:
            assert g1[act] == g2[act]

    def test_all_paths_exist(self, tmp_path: Path) -> None:
        d = _make_dir(tmp_path)
        groups = discover_room_b(d)
        for paths in groups.values():
            for p in paths:
                assert p.exists()

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_room_b(tmp_path / "nonexistent")

    def test_empty_dir_returns_empty_lists(self, tmp_path: Path) -> None:
        d = tmp_path / "empty_dir"
        d.mkdir()
        groups = discover_room_b(d)
        assert all(len(v) == 0 for v in groups.values())


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_returns_float32_array(self, tmp_path: Path) -> None:
        p = tmp_path / "test.csv"
        _write_csv(p, n_rows=20, header="named")
        arr = load_csv(p)
        assert arr.dtype == np.float32

    def test_shape_256_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "test.csv"
        _write_csv(p, n_rows=15, header="named")
        arr = load_csv(p)
        assert arr.shape == (15, N_SUBCARRIERS)

    def test_zero_indexed_header_reads_correctly(self, tmp_path: Path) -> None:
        p = tmp_path / "anomalous.csv"
        _write_csv(p, n_rows=8, header="zero", fill=42.0)
        arr = load_csv(p)
        assert arr.shape == (8, 256)
        assert np.allclose(arr, 42.0)

    def test_named_header_values_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "named.csv"
        _write_csv(p, n_rows=5, header="named", fill=3.14)
        arr = load_csv(p)
        assert np.allclose(arr, 3.14, rtol=1e-4)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nope.csv")

    def test_wrong_column_count_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        # Write with only 10 columns
        pd.DataFrame(np.ones((5, 10))).to_csv(p, index=False)
        with pytest.raises(ValueError, match="256"):
            load_csv(p)


# ---------------------------------------------------------------------------
# load_capture
# ---------------------------------------------------------------------------

class TestLoadCapture:
    def test_returns_array_and_manifest_row(self, tmp_path: Path) -> None:
        p = tmp_path / "Empty_Amp_1.csv"
        _write_csv(p, n_rows=10)
        arr, row = load_capture(p, activity="empty", split="dev")
        assert arr.shape == (10, 256)
        assert row.room == "B"
        assert row.activity == "empty"
        assert row.split == "dev"

    def test_capture_id_format(self, tmp_path: Path) -> None:
        p = tmp_path / "Sitting_Amp_42.csv"
        _write_csv(p)
        _, row = load_capture(p, activity="sit", split="val")
        assert row.capture_id == "B__sit__Sitting_Amp_42"

    def test_n_packets_equals_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "Walking_7.csv"
        _write_csv(p, n_rows=300)
        arr, row = load_capture(p, activity="walk", split="dev")
        assert row.n_packets_raw == 300
        assert row.n_packets_clean == 300

    def test_source_file_is_absolute_string(self, tmp_path: Path) -> None:
        p = tmp_path / "Lying_Amp_1.csv"
        _write_csv(p)
        _, row = load_capture(p, activity="lying", split="dev")
        assert str(p) == row.source_file

    def test_val_split_label(self, tmp_path: Path) -> None:
        p = tmp_path / "Standing_Amp_1.csv"
        _write_csv(p)
        _, row = load_capture(p, activity="stand", split="val")
        assert row.split == "val"

    def test_intermediate_path_passthrough(self, tmp_path: Path) -> None:
        p = tmp_path / "Empty_Amp_1.csv"
        _write_csv(p)
        _, row = load_capture(p, activity="empty", split="dev",
                              intermediate_path="derived/intermediate/B/x.npy")
        assert row.intermediate_path == "derived/intermediate/B/x.npy"


# ---------------------------------------------------------------------------
# Real-data smoke tests
# ---------------------------------------------------------------------------

REAL_CSV_DIR = Path("dataset/Experiment-3/Data")


@pytest.mark.skipif(not REAL_CSV_DIR.exists(), reason="Dataset not available")
class TestRealRoomB:
    def test_discover_returns_expected_counts(self) -> None:
        groups = discover_room_b(REAL_CSV_DIR)
        # Confirmed counts from inspect_data.py
        assert len(groups["empty"]) == 267   # 266 Amp + 1 bare
        assert len(groups["lying"]) == 351   # 350 Amp + 1 bare
        assert len(groups["sit"])   == 153   # 152 Amp + 1 bare
        assert len(groups["stand"]) == 221   # 220 Amp + 1 bare
        assert len(groups["walk"])  == 209   # all Walking_N

    def test_load_csv_first_file_each_activity(self) -> None:
        groups = discover_room_b(REAL_CSV_DIR)
        for act, paths in groups.items():
            arr = load_csv(paths[0])
            assert arr.shape[1] == 256
            assert arr.dtype == np.float32
            assert np.isfinite(arr).all(), f"{act}: non-finite values"

    def test_all_files_have_256_columns(self) -> None:
        groups = discover_room_b(REAL_CSV_DIR)
        for act, paths in groups.items():
            for p in paths:
                arr = load_csv(p)
                assert arr.shape[1] == 256, f"{p.name} has {arr.shape[1]} cols"
