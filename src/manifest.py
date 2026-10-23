"""
src/manifest.py
===============
Read and write ``derived/manifest.csv`` – the capture-level registry that
links every original source file to its room, activity, split assignment,
and cleaning statistics.

Schema (one row per original capture)
--------------------------------------
capture_id        str   – unique identifier, e.g. "A__fall__fall10"
room              str   – "A" | "B"
activity          str   – one of the eight canonical activity labels
split             str   – "dev" | "val"
source_file       str   – path to the original source file (relative to repo root)
n_packets_raw     int   – packet/row count before cleaning
n_packets_clean   int   – packet/row count after cleaning
intermediate_path str   – relative path to derived/intermediate/{room}/{capture_id}.npy

Integrity rules (enforced on both read and write)
--------------------------------------------------
1. capture_id values are unique.
2. No capture_id appears in both "dev" and "val" splits.
3. room values are in {"A", "B"}.
4. activity values are in the eight-element canonical set.
5. split values are in {"dev", "val"}.
6. n_packets_clean <= n_packets_raw for every row.
7. n_packets_raw >= 0 and n_packets_clean >= 0.

Usage
-----
    from src.manifest import ManifestRow, read_manifest, write_manifest, append_rows

    # Build rows during ingestion
    row = ManifestRow(
        capture_id="A__fall__fall10",
        room="A",
        activity="fall",
        split="dev",
        source_file="dataset/Experiment-2/.../fall10.pcap",
        n_packets_raw=320,
        n_packets_clean=298,
        intermediate_path="derived/intermediate/A/A__fall__fall10.npy",
    )

    # Persist / reload
    write_manifest([row], Path("derived/manifest.csv"))
    rows = read_manifest(Path("derived/manifest.csv"))
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Allowed vocabulary sets (kept in sync with config.py)
# ---------------------------------------------------------------------------

_ALLOWED_ROOMS: frozenset[str] = frozenset({"A", "B"})
_ALLOWED_SPLITS: frozenset[str] = frozenset({"dev", "val"})
_ALLOWED_ACTIVITIES: frozenset[str] = frozenset(
    {"empty", "fall", "lying", "sit", "sitdown", "stand", "standup", "walk"}
)

# Canonical CSV column order (must match schema docstring above)
_COLUMNS: List[str] = [
    "capture_id",
    "room",
    "activity",
    "split",
    "source_file",
    "n_packets_raw",
    "n_packets_clean",
    "intermediate_path",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ManifestRow:
    """One row in ``manifest.csv``.

    All string fields are stripped of leading/trailing whitespace on
    construction.  Numeric fields are stored as ``int``.
    """

    capture_id: str
    room: str
    activity: str
    split: str
    source_file: str
    n_packets_raw: int
    n_packets_clean: int
    intermediate_path: str

    def __post_init__(self) -> None:
        # Normalise strings
        self.capture_id = str(self.capture_id).strip()
        self.room = str(self.room).strip()
        self.activity = str(self.activity).strip()
        self.split = str(self.split).strip()
        self.source_file = str(self.source_file).strip()
        self.intermediate_path = str(self.intermediate_path).strip()
        # Coerce numerics
        self.n_packets_raw = int(self.n_packets_raw)
        self.n_packets_clean = int(self.n_packets_clean)

    # ------------------------------------------------------------------
    # Per-row validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if this row violates the schema contract."""
        if not self.capture_id:
            raise ValueError("ManifestRow.capture_id must not be empty")
        if self.room not in _ALLOWED_ROOMS:
            raise ValueError(
                f"Row {self.capture_id!r}: room={self.room!r} not in {sorted(_ALLOWED_ROOMS)}"
            )
        if self.activity not in _ALLOWED_ACTIVITIES:
            raise ValueError(
                f"Row {self.capture_id!r}: activity={self.activity!r} "
                f"not in {sorted(_ALLOWED_ACTIVITIES)}"
            )
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(
                f"Row {self.capture_id!r}: split={self.split!r} not in {sorted(_ALLOWED_SPLITS)}"
            )
        if self.n_packets_raw < 0:
            raise ValueError(
                f"Row {self.capture_id!r}: n_packets_raw={self.n_packets_raw} must be >= 0"
            )
        if self.n_packets_clean < 0:
            raise ValueError(
                f"Row {self.capture_id!r}: n_packets_clean={self.n_packets_clean} must be >= 0"
            )
        if self.n_packets_clean > self.n_packets_raw:
            raise ValueError(
                f"Row {self.capture_id!r}: n_packets_clean ({self.n_packets_clean}) "
                f"> n_packets_raw ({self.n_packets_raw})"
            )

    def to_dict(self) -> dict:
        """Return ordered dict suitable for CSV writing."""
        d = asdict(self)
        return {col: d[col] for col in _COLUMNS}


# ---------------------------------------------------------------------------
# Collection-level integrity checks
# ---------------------------------------------------------------------------

def _assert_collection_integrity(rows: List[ManifestRow]) -> None:
    """Run cross-row integrity checks on a complete manifest.

    Checks
    ------
    * capture_id uniqueness.
    * No capture_id appears in both "dev" and "val".

    Raises
    ------
    ValueError on first detected violation.
    """
    seen_ids: dict[str, str] = {}  # capture_id -> split

    for row in rows:
        cid = row.capture_id
        if cid in seen_ids:
            raise ValueError(
                f"Duplicate capture_id {cid!r} found in manifest "
                f"(splits: {seen_ids[cid]!r} and {row.split!r})"
            )
        seen_ids[cid] = row.split

    # Check for cross-split contamination
    # (The above uniqueness check already catches this, but we keep an
    # explicit grouped check for clarity.)
    dev_ids: set[str] = {r.capture_id for r in rows if r.split == "dev"}
    val_ids: set[str] = {r.capture_id for r in rows if r.split == "val"}
    overlap = dev_ids & val_ids
    if overlap:
        raise ValueError(
            f"The following capture_ids appear in both 'dev' and 'val' splits: "
            f"{sorted(overlap)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> List[ManifestRow]:
    """Read ``manifest.csv`` and return a validated list of ``ManifestRow``.

    Parameters
    ----------
    path:
        Path to the manifest CSV file.

    Returns
    -------
    List[ManifestRow]
        All rows, in file order.  Each row has been individually validated
        and the full collection has passed cross-row integrity checks.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any row or the collection fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: List[ManifestRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _check_columns(reader.fieldnames or [], path)
        for i, record in enumerate(reader, start=2):  # start=2: row 1 is header
            try:
                row = ManifestRow(
                    capture_id=record["capture_id"],
                    room=record["room"],
                    activity=record["activity"],
                    split=record["split"],
                    source_file=record["source_file"],
                    n_packets_raw=record["n_packets_raw"],
                    n_packets_clean=record["n_packets_clean"],
                    intermediate_path=record["intermediate_path"],
                )
                row.validate()
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Manifest {path}, CSV line {i}: {exc}") from exc
            rows.append(row)

    _assert_collection_integrity(rows)
    return rows


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> None:
    """Validate *rows* and write them to *path* as a CSV.

    The file is written atomically: we write to a temporary sibling first,
    then rename, so a crash during write does not corrupt an existing manifest.

    Parameters
    ----------
    rows:
        Rows to persist.  Each row is individually validated; the
        collection-level integrity checks are also run before writing.
    path:
        Destination path.  Parent directories are created if absent.

    Raises
    ------
    ValueError
        If any row or the collection fails validation.
    """
    rows = list(rows)
    # Per-row validation
    for row in rows:
        row.validate()
    # Collection validation
    _assert_collection_integrity(rows)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.csv")

    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def append_rows(new_rows: Sequence[ManifestRow], path: Path) -> None:
    """Append *new_rows* to an existing manifest at *path*.

    Reads the existing manifest, merges with *new_rows*, runs full
    collection integrity checks, then writes the result back atomically.

    Parameters
    ----------
    new_rows:
        Rows to add.  Must not introduce duplicate capture_ids.
    path:
        Path to the existing (or not-yet-existing) manifest CSV.

    Raises
    ------
    ValueError
        If the merged collection fails any integrity check.
    """
    existing: List[ManifestRow] = []
    if Path(path).exists():
        existing = read_manifest(path)
    write_manifest(existing + list(new_rows), path)


def filter_manifest(
    rows: List[ManifestRow],
    *,
    room: Optional[str] = None,
    activity: Optional[str] = None,
    split: Optional[str] = None,
) -> List[ManifestRow]:
    """Return the subset of *rows* matching the given filters.

    All supplied keyword arguments are ANDed together.  Passing ``None``
    for a field means "no filter on that field".

    Parameters
    ----------
    rows:
        Source list (typically the output of ``read_manifest``).
    room:
        Keep only rows with this room value.
    activity:
        Keep only rows with this activity value.
    split:
        Keep only rows with this split value.

    Returns
    -------
    List[ManifestRow]
        Filtered subset, preserving original order.
    """
    result = rows
    if room is not None:
        result = [r for r in result if r.room == room]
    if activity is not None:
        result = [r for r in result if r.activity == activity]
    if split is not None:
        result = [r for r in result if r.split == split]
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(actual: List[str], path: Path) -> None:
    """Raise ``ValueError`` if *actual* columns differ from the expected schema."""
    actual_set = set(actual)
    expected_set = set(_COLUMNS)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing or extra:
        raise ValueError(
            f"Manifest {path} has unexpected columns.\n"
            f"  Missing : {sorted(missing)}\n"
            f"  Extra   : {sorted(extra)}\n"
            f"  Expected: {_COLUMNS}"
        )
