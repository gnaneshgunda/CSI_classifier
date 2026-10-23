"""
tests/test_splits.py
====================
Unit tests for src/splits.py.

Covers:
* make_room_b_split: exactly 25 per class, identical across runs,
  no overlap, deterministic ordering.
* room_a_manifest_rows: correct split labels, activity parsing,
  dev/val separation.
* room_b_manifest_rows: end-to-end manifest row generation.

Real-data tests are skipped when the dataset directory is absent.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    VAL_PER_CLASS,
    VAL_SEED,
    make_room_b_split,
    room_a_manifest_rows,
    room_b_manifest_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, n_rows: int = 5) -> None:
    cols = [f"Subcarrier_{i+1}" for i in range(256)]
    df = pd.DataFrame(np.ones((n_rows, 256), dtype=np.float32), columns=cols)
    df.to_csv(path, index=False)


def _make_room_b_dir(tmp: Path, counts: dict) -> Path:
    """counts: {activity_prefix: n_files}"""
    d = tmp / "room_b"
    d.mkdir(exist_ok=True)
    prefix_map = {
        "empty": "Empty_Amp",
        "lying": "Lying_Amp",
        "sit": "Sitting_Amp",
        "stand": "Standing_Amp",
        "walk": "Walking",
    }
    for act, n in counts.items():
        prefix = prefix_map[act]
        for i in range(1, n + 1):
            p = d / f"{prefix}_{i}.csv"
            _write_csv(p)
    return d


def _make_room_a_dirs(tmp: Path) -> tuple:
    """Create fake pcap_root and val_root directory trees."""
    pcap_root = tmp / "csidata"
    val_root  = tmp / "validation"
    activities = ["fall", "sit", "walk"]
    for act in activities:
        (pcap_root / act).mkdir(parents=True)
        (val_root  / act).mkdir(parents=True)
        # 5 dev, 3 val per activity
        for i in range(5):
            (pcap_root / act / f"{act}{i}.pcap").touch()
        for i in range(3):
            (val_root  / act / f"{act}_v{i}.pcap").touch()
    return pcap_root, val_root


# ---------------------------------------------------------------------------
# make_room_b_split
# ---------------------------------------------------------------------------

class TestMakeRoomBSplit:
    def _make_groups(self, tmp: Path, n_per_act: int = 30):
        from src.room_b_loader import discover_room_b
        d = _make_room_b_dir(tmp, {a: n_per_act
                                    for a in ["empty","lying","sit","stand","walk"]})
        return discover_room_b(d)

    def test_exactly_25_val_per_class(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=30)
        _, val = make_room_b_split(groups, val_per_class=25, seed=42)
        for act, paths in val.items():
            assert len(paths) == 25, f"{act}: {len(paths)} val files"

    def test_dev_count_correct(self, tmp_path: Path) -> None:
        n = 30
        groups = self._make_groups(tmp_path, n_per_act=n)
        dev, val = make_room_b_split(groups, val_per_class=25, seed=42)
        for act in groups:
            assert len(dev[act]) + len(val[act]) == n

    def test_no_overlap_between_dev_and_val(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=30)
        dev, val = make_room_b_split(groups, val_per_class=25, seed=42)
        for act in groups:
            dev_set = set(p.name for p in dev[act])
            val_set = set(p.name for p in val[act])
            assert dev_set.isdisjoint(val_set), f"{act} has overlap"

    def test_identical_across_two_calls(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=30)
        dev1, val1 = make_room_b_split(groups, val_per_class=25, seed=42)
        dev2, val2 = make_room_b_split(groups, val_per_class=25, seed=42)
        for act in groups:
            assert [p.name for p in val1[act]] == [p.name for p in val2[act]]
            assert [p.name for p in dev1[act]] == [p.name for p in dev2[act]]

    def test_different_seed_gives_different_split(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=50)
        _, val42 = make_room_b_split(groups, val_per_class=25, seed=42)
        _, val99 = make_room_b_split(groups, val_per_class=25, seed=99)
        # At least one activity must differ
        diffs = [
            set(p.name for p in val42[a]) != set(p.name for p in val99[a])
            for a in groups
        ]
        assert any(diffs), "Different seeds produced identical splits"

    def test_insufficient_files_raises(self, tmp_path: Path) -> None:
        # Only 10 files per activity but requesting 25 val
        groups = self._make_groups(tmp_path, n_per_act=10)
        with pytest.raises(ValueError, match="25"):
            make_room_b_split(groups, val_per_class=25, seed=42)

    def test_custom_val_per_class(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=30)
        _, val = make_room_b_split(groups, val_per_class=5, seed=42)
        for paths in val.values():
            assert len(paths) == 5

    def test_output_lists_are_sorted(self, tmp_path: Path) -> None:
        groups = self._make_groups(tmp_path, n_per_act=30)
        dev, val = make_room_b_split(groups, val_per_class=25, seed=42)
        for act in groups:
            names_d = [p.name.lower() for p in dev[act]]
            names_v = [p.name.lower() for p in val[act]]
            assert names_d == sorted(names_d)
            assert names_v == sorted(names_v)


# ---------------------------------------------------------------------------
# room_a_manifest_rows
# ---------------------------------------------------------------------------

class TestRoomAManifestRows:
    def test_dev_rows_have_dev_split(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        dev = [r for r in rows if r.split == "dev"]
        assert all(r.split == "dev" for r in dev)

    def test_val_rows_have_val_split(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        val = [r for r in rows if r.split == "val"]
        assert all(r.split == "val" for r in val)

    def test_correct_dev_val_counts(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        dev = [r for r in rows if r.split == "dev"]
        val = [r for r in rows if r.split == "val"]
        # 3 activities × 5 dev each, 3 activities × 3 val each
        assert len(dev) == 15
        assert len(val) == 9

    def test_room_is_A(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        assert all(r.room == "A" for r in rows)

    def test_activity_parsed_from_directory_name(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        activities = {r.activity for r in rows}
        assert activities == {"fall", "sit", "walk"}

    def test_capture_id_format(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        for r in rows:
            assert r.capture_id.startswith("A__")

    def test_no_duplicate_capture_ids(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        ids = [r.capture_id for r in rows]
        assert len(ids) == len(set(ids))

    def test_no_dev_val_overlap(self, tmp_path: Path) -> None:
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        dev_ids = {r.capture_id for r in rows if r.split == "dev"}
        val_ids = {r.capture_id for r in rows if r.split == "val"}
        assert dev_ids.isdisjoint(val_ids)

    def test_validation_subdir_skipped_in_pcap_root(self, tmp_path: Path) -> None:
        """validation/ sub-folder inside pcap_root must not produce dev rows."""
        pcap_root, val_root = _make_room_a_dirs(tmp_path)
        # Create a 'validation' subfolder inside pcap_root with extra files
        fake_val = pcap_root / "validation" / "fall"
        fake_val.mkdir(parents=True)
        (fake_val / "shouldnotappear.pcap").touch()
        rows = list(room_a_manifest_rows(pcap_root, val_root))
        ids = [r.capture_id for r in rows]
        assert not any("shouldnotappear" in cid for cid in ids)

    def test_missing_pcap_root_raises(self, tmp_path: Path) -> None:
        _, val_root = _make_room_a_dirs(tmp_path)
        with pytest.raises(FileNotFoundError):
            list(room_a_manifest_rows(tmp_path / "nope", val_root))

    def test_missing_val_root_raises(self, tmp_path: Path) -> None:
        pcap_root, _ = _make_room_a_dirs(tmp_path)
        with pytest.raises(FileNotFoundError):
            list(room_a_manifest_rows(pcap_root, tmp_path / "nope"))


# ---------------------------------------------------------------------------
# room_b_manifest_rows
# ---------------------------------------------------------------------------

class TestRoomBManifestRows:
    def test_exactly_25_val_per_activity(self, tmp_path: Path) -> None:
        csv_dir = _make_room_b_dir(
            tmp_path, {a: 30 for a in ["empty","lying","sit","stand","walk"]}
        )
        rows = list(room_b_manifest_rows(csv_dir, val_per_class=25, seed=42))
        val_counts = Counter(r.activity for r in rows if r.split == "val")
        for act, cnt in val_counts.items():
            assert cnt == 25, f"{act}: {cnt}"

    def test_no_dev_val_overlap(self, tmp_path: Path) -> None:
        csv_dir = _make_room_b_dir(
            tmp_path, {a: 30 for a in ["empty","lying","sit","stand","walk"]}
        )
        rows = list(room_b_manifest_rows(csv_dir, val_per_class=25, seed=42))
        dev_ids = {r.capture_id for r in rows if r.split == "dev"}
        val_ids = {r.capture_id for r in rows if r.split == "val"}
        assert dev_ids.isdisjoint(val_ids)

    def test_identical_across_two_calls(self, tmp_path: Path) -> None:
        csv_dir = _make_room_b_dir(
            tmp_path, {a: 30 for a in ["empty","lying","sit","stand","walk"]}
        )
        rows1 = sorted(room_b_manifest_rows(csv_dir, 25, 42), key=lambda r: r.capture_id)
        rows2 = sorted(room_b_manifest_rows(csv_dir, 25, 42), key=lambda r: r.capture_id)
        assert [r.capture_id for r in rows1] == [r.capture_id for r in rows2]
        assert [r.split for r in rows1] == [r.split for r in rows2]

    def test_all_room_B(self, tmp_path: Path) -> None:
        csv_dir = _make_room_b_dir(
            tmp_path, {a: 26 for a in ["empty","lying","sit","stand","walk"]}
        )
        rows = list(room_b_manifest_rows(csv_dir, val_per_class=25, seed=42))
        assert all(r.room == "B" for r in rows)

    def test_no_duplicate_capture_ids(self, tmp_path: Path) -> None:
        csv_dir = _make_room_b_dir(
            tmp_path, {a: 30 for a in ["empty","lying","sit","stand","walk"]}
        )
        rows = list(room_b_manifest_rows(csv_dir, 25, 42))
        ids = [r.capture_id for r in rows]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Real-data integration tests
# ---------------------------------------------------------------------------

REAL_PCAP_ROOT = Path(
    "dataset/Experiment-2/Experiment-2/realdata/input_data/30ms/csidata"
)
REAL_VAL_ROOT  = REAL_PCAP_ROOT / "validation"
REAL_CSV_DIR   = Path("dataset/Experiment-3/Data")


@pytest.mark.skipif(
    not REAL_PCAP_ROOT.exists(),
    reason="Room A dataset not available",
)
class TestRealRoomA:
    def test_dev_count(self) -> None:
        rows = list(room_a_manifest_rows(REAL_PCAP_ROOT, REAL_VAL_ROOT))
        dev = [r for r in rows if r.split == "dev"]
        # 177+178+155+116+155+154 = 935
        assert len(dev) == 935

    def test_val_count(self) -> None:
        rows = list(room_a_manifest_rows(REAL_PCAP_ROOT, REAL_VAL_ROOT))
        val = [r for r in rows if r.split == "val"]
        assert len(val) == 150   # 25 × 6 activities

    def test_exactly_25_val_per_activity(self) -> None:
        rows = list(room_a_manifest_rows(REAL_PCAP_ROOT, REAL_VAL_ROOT))
        val_counts = Counter(r.activity for r in rows if r.split == "val")
        for act, cnt in val_counts.items():
            assert cnt == 25, f"{act}: {cnt}"

    def test_no_dev_val_overlap(self) -> None:
        rows = list(room_a_manifest_rows(REAL_PCAP_ROOT, REAL_VAL_ROOT))
        dev_ids = {r.capture_id for r in rows if r.split == "dev"}
        val_ids = {r.capture_id for r in rows if r.split == "val"}
        assert dev_ids.isdisjoint(val_ids)


@pytest.mark.skipif(
    not REAL_CSV_DIR.exists(),
    reason="Room B dataset not available",
)
class TestRealRoomB:
    def test_exactly_25_val_per_activity(self) -> None:
        rows = list(room_b_manifest_rows(REAL_CSV_DIR, 25, 42))
        val_counts = Counter(r.activity for r in rows if r.split == "val")
        assert set(val_counts.keys()) == {"empty", "lying", "sit", "stand", "walk"}
        for act, cnt in val_counts.items():
            assert cnt == 25, f"{act}: {cnt}"

    def test_identical_across_runs(self) -> None:
        rows1 = sorted(room_b_manifest_rows(REAL_CSV_DIR, 25, 42),
                       key=lambda r: r.capture_id)
        rows2 = sorted(room_b_manifest_rows(REAL_CSV_DIR, 25, 42),
                       key=lambda r: r.capture_id)
        assert [r.capture_id for r in rows1] == [r.capture_id for r in rows2]
        assert [r.split     for r in rows1] == [r.split     for r in rows2]

    def test_total_count_matches_file_count(self) -> None:
        from src.room_b_loader import discover_room_b
        groups = discover_room_b(REAL_CSV_DIR)
        total_files = sum(len(v) for v in groups.values())
        rows = list(room_b_manifest_rows(REAL_CSV_DIR, 25, 42))
        assert len(rows) == total_files

    def test_no_dev_val_overlap(self) -> None:
        rows = list(room_b_manifest_rows(REAL_CSV_DIR, 25, 42))
        dev_ids = {r.capture_id for r in rows if r.split == "dev"}
        val_ids = {r.capture_id for r in rows if r.split == "val"}
        assert dev_ids.isdisjoint(val_ids)
