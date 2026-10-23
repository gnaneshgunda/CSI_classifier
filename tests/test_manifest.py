"""
tests/test_manifest.py
======================
Unit tests for src/manifest.py.

All tests use tmp_path fixtures – no real dataset files are required.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest

from src.manifest import (
    ManifestRow,
    append_rows,
    filter_manifest,
    read_manifest,
    write_manifest,
    _COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    capture_id: str = "A__fall__fall10",
    room: str = "A",
    activity: str = "fall",
    split: str = "dev",
    source_file: str = "dataset/Experiment-2/fall/fall10.pcap",
    n_packets_raw: int = 320,
    n_packets_clean: int = 298,
    intermediate_path: str = "derived/intermediate/A/A__fall__fall10.npy",
) -> ManifestRow:
    return ManifestRow(
        capture_id=capture_id,
        room=room,
        activity=activity,
        split=split,
        source_file=source_file,
        n_packets_raw=n_packets_raw,
        n_packets_clean=n_packets_clean,
        intermediate_path=intermediate_path,
    )


def _make_rows(n: int = 3) -> List[ManifestRow]:
    """Return *n* distinct valid rows."""
    activities = ["fall", "sit", "sitdown", "stand", "standup", "walk"]
    rows = []
    for i in range(n):
        act = activities[i % len(activities)]
        cid = f"A__{act}__capture{i:03d}"
        rows.append(
            _make_row(
                capture_id=cid,
                activity=act,
                source_file=f"dataset/Experiment-2/{act}/{cid}.pcap",
                intermediate_path=f"derived/intermediate/A/{cid}.npy",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# ManifestRow construction and validation
# ---------------------------------------------------------------------------

class TestManifestRowConstruction:
    def test_valid_row_constructs(self) -> None:
        row = _make_row()
        assert row.capture_id == "A__fall__fall10"
        assert row.room == "A"

    def test_strings_are_stripped(self) -> None:
        row = _make_row(capture_id="  A__fall__fall10  ", room=" A ")
        assert row.capture_id == "A__fall__fall10"
        assert row.room == "A"

    def test_numerics_coerced_from_strings(self) -> None:
        row = ManifestRow(
            capture_id="A__sit__sit1",
            room="A",
            activity="sit",
            split="dev",
            source_file="x.pcap",
            n_packets_raw="400",  # type: ignore[arg-type]
            n_packets_clean="380",  # type: ignore[arg-type]
            intermediate_path="derived/intermediate/A/A__sit__sit1.npy",
        )
        assert isinstance(row.n_packets_raw, int)
        assert row.n_packets_raw == 400

    def test_room_b_valid(self) -> None:
        row = _make_row(room="B", activity="empty")
        row.validate()  # should not raise

    def test_all_activities_valid(self) -> None:
        activities = ["empty", "fall", "lying", "sit", "sitdown", "stand", "standup", "walk"]
        for act in activities:
            room = "A" if act in ("fall", "sit", "sitdown", "stand", "standup", "walk") else "B"
            row = _make_row(activity=act, room=room)
            row.validate()


class TestManifestRowValidation:
    def test_empty_capture_id(self) -> None:
        with pytest.raises(ValueError, match="capture_id"):
            _make_row(capture_id="").validate()

    def test_bad_room(self) -> None:
        with pytest.raises(ValueError, match="room"):
            _make_row(room="C").validate()

    def test_bad_activity(self) -> None:
        with pytest.raises(ValueError, match="activity"):
            _make_row(activity="jump").validate()

    def test_bad_split(self) -> None:
        with pytest.raises(ValueError, match="split"):
            _make_row(split="test").validate()

    def test_negative_raw_packets(self) -> None:
        with pytest.raises(ValueError, match="n_packets_raw"):
            _make_row(n_packets_raw=-1).validate()

    def test_negative_clean_packets(self) -> None:
        with pytest.raises(ValueError, match="n_packets_clean"):
            _make_row(n_packets_clean=-1).validate()

    def test_clean_exceeds_raw(self) -> None:
        with pytest.raises(ValueError, match="n_packets_clean"):
            _make_row(n_packets_raw=100, n_packets_clean=200).validate()

    def test_clean_equals_raw_is_valid(self) -> None:
        row = _make_row(n_packets_raw=150, n_packets_clean=150)
        row.validate()  # should not raise

    def test_zero_packets_valid(self) -> None:
        """A capture with 0 clean packets is technically valid at row level."""
        row = _make_row(n_packets_raw=50, n_packets_clean=0)
        row.validate()  # should not raise


# ---------------------------------------------------------------------------
# write_manifest / read_manifest round-trip
# ---------------------------------------------------------------------------

class TestWriteReadRoundTrip:
    def test_write_then_read_returns_same_data(self, tmp_path: Path) -> None:
        rows = _make_rows(5)
        path = tmp_path / "manifest.csv"
        write_manifest(rows, path)
        loaded = read_manifest(path)
        assert len(loaded) == 5
        for orig, loaded_row in zip(rows, loaded):
            assert orig.capture_id == loaded_row.capture_id
            assert orig.room == loaded_row.room
            assert orig.activity == loaded_row.activity
            assert orig.split == loaded_row.split
            assert orig.n_packets_raw == loaded_row.n_packets_raw
            assert orig.n_packets_clean == loaded_row.n_packets_clean

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "manifest.csv"
        write_manifest([_make_row()], path)
        assert path.exists()

    def test_csv_has_correct_header(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest([_make_row()], path)
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == _COLUMNS

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest(_make_rows(3), path)
        write_manifest(_make_rows(1), path)
        loaded = read_manifest(path)
        assert len(loaded) == 1

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "nonexistent.csv")

    def test_empty_manifest_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest([], path)
        loaded = read_manifest(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# Collection integrity checks
# ---------------------------------------------------------------------------

class TestCollectionIntegrity:
    def test_duplicate_capture_id_raises(self, tmp_path: Path) -> None:
        rows = [_make_row(capture_id="dup"), _make_row(capture_id="dup")]
        with pytest.raises(ValueError, match="dup"):
            write_manifest(rows, tmp_path / "manifest.csv")

    def test_same_id_in_dev_and_val_raises(self, tmp_path: Path) -> None:
        rows = [
            _make_row(capture_id="A__fall__fall10", split="dev"),
            _make_row(capture_id="A__fall__fall10", split="val"),
        ]
        with pytest.raises(ValueError, match="A__fall__fall10"):
            write_manifest(rows, tmp_path / "manifest.csv")

    def test_different_ids_different_splits_ok(self, tmp_path: Path) -> None:
        rows = [
            _make_row(capture_id="A__fall__fall10", split="dev"),
            _make_row(capture_id="A__fall__fall11", split="val"),
        ]
        path = tmp_path / "manifest.csv"
        write_manifest(rows, path)  # should not raise
        loaded = read_manifest(path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# append_rows
# ---------------------------------------------------------------------------

class TestAppendRows:
    def test_append_to_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        append_rows([_make_row()], path)
        assert read_manifest(path)[0].capture_id == "A__fall__fall10"

    def test_append_grows_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest(_make_rows(2), path)
        extra = _make_row(
            capture_id="A__walk__walk999",
            activity="walk",
            source_file="dataset/Experiment-2/walk/walk999.pcap",
            intermediate_path="derived/intermediate/A/A__walk__walk999.npy",
        )
        append_rows([extra], path)
        loaded = read_manifest(path)
        assert len(loaded) == 3

    def test_append_duplicate_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        write_manifest([_make_row()], path)
        with pytest.raises(ValueError, match="A__fall__fall10"):
            append_rows([_make_row()], path)


# ---------------------------------------------------------------------------
# filter_manifest
# ---------------------------------------------------------------------------

class TestFilterManifest:
    def _mixed_rows(self) -> List[ManifestRow]:
        return [
            _make_row(capture_id="r1", room="A", activity="fall", split="dev"),
            _make_row(capture_id="r2", room="A", activity="sit",  split="val"),
            _make_row(capture_id="r3", room="B", activity="empty", split="dev",
                      source_file="x.csv",
                      intermediate_path="derived/intermediate/B/r3.npy"),
            _make_row(capture_id="r4", room="B", activity="lying", split="val",
                      source_file="y.csv",
                      intermediate_path="derived/intermediate/B/r4.npy"),
        ]

    def test_filter_by_room(self) -> None:
        result = filter_manifest(self._mixed_rows(), room="A")
        assert all(r.room == "A" for r in result)
        assert len(result) == 2

    def test_filter_by_split(self) -> None:
        result = filter_manifest(self._mixed_rows(), split="val")
        assert all(r.split == "val" for r in result)
        assert len(result) == 2

    def test_filter_by_activity(self) -> None:
        result = filter_manifest(self._mixed_rows(), activity="fall")
        assert len(result) == 1
        assert result[0].capture_id == "r1"

    def test_combined_filters(self) -> None:
        result = filter_manifest(self._mixed_rows(), room="B", split="dev")
        assert len(result) == 1
        assert result[0].capture_id == "r3"

    def test_no_filter_returns_all(self) -> None:
        rows = self._mixed_rows()
        assert filter_manifest(rows) == rows

    def test_filter_no_match_returns_empty(self) -> None:
        result = filter_manifest(self._mixed_rows(), activity="standup")
        assert result == []


# ---------------------------------------------------------------------------
# Corrupt CSV edge cases
# ---------------------------------------------------------------------------

class TestCorruptCSV:
    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        # Write a CSV missing the 'room' column
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[c for c in _COLUMNS if c != "room"],
            )
            writer.writeheader()
        with pytest.raises(ValueError, match="room"):
            read_manifest(path)

    def test_bad_room_value_in_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        row = _make_row()
        d = row.to_dict()
        d["room"] = "Z"  # invalid
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerow(d)
        with pytest.raises(ValueError, match="room"):
            read_manifest(path)

    def test_clean_exceeds_raw_in_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        row = _make_row()
        d = row.to_dict()
        d["n_packets_clean"] = 9999
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerow(d)
        with pytest.raises(ValueError, match="n_packets_clean"):
            read_manifest(path)
