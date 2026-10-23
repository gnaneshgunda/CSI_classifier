"""
src/splits.py
=============
Assign every capture in the dataset to either the ``dev`` or ``val`` split,
producing :class:`~src.manifest.ManifestRow` entries for both rooms.

Split rules
-----------
Room A (Experiment-2)
    The dataset already ships a ``validation/`` sub-directory containing
    exactly 25 captures per activity.  Those captures are ``val``; every
    other capture in the ``csidata/`` tree is ``dev``.  No random sampling
    is needed and no seed is consumed.

Room B (Experiment-3)
    No pre-existing validation folder exists.  Validation is created once
    at capture level:

    1. For each activity, collect all CSV paths and sort them
       deterministically by lower-case filename (lexicographic).
    2. Sample exactly 25 paths as ``val`` using
       ``numpy.random.RandomState(seed)`` (default ``seed=42``).
    3. All remaining paths are ``dev``.

    Sorting before sampling guarantees that every team that runs the same
    code against the same data produces identical splits regardless of
    filesystem ordering.

Leakage guarantee
-----------------
* Validation captures are *never* used for normalization fitting, support
  sets, or model training — that is enforced by keeping the split label in
  the manifest and checking it downstream.
* The split is created **once** and is never resampled during a project run.

Public API
----------
``make_room_b_split(groups, val_per_class, seed)``
    Given the output of ``room_b_loader.discover_room_b``, return two
    dicts ``(dev_groups, val_groups)`` mapping activity → path list.

``room_a_manifest_rows(pcap_root, val_root)``
    Walk the Room-A directory tree and yield ``ManifestRow`` objects with
    split labels assigned from the directory structure.

``room_b_manifest_rows(csv_dir, val_per_class, seed)``
    Discover, split, and yield ``ManifestRow`` objects for all Room-B
    captures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np

from src.manifest import ManifestRow
from src.room_b_loader import PREFIX_TO_ACTIVITY, discover_room_b

# ---------------------------------------------------------------------------
# Room B split
# ---------------------------------------------------------------------------

#: Default number of validation captures per activity for Room B.
VAL_PER_CLASS: int = 25

#: Public random seed used for Room B validation sampling.
VAL_SEED: int = 42


def make_room_b_split(
    groups: Dict[str, List[Path]],
    val_per_class: int = VAL_PER_CLASS,
    seed: int = VAL_SEED,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Split Room-B file groups into development and validation sets.

    Parameters
    ----------
    groups:
        Activity → sorted list of CSV paths, as returned by
        ``room_b_loader.discover_room_b``.
    val_per_class:
        Number of captures to hold out per activity.  Must be ≤ the
        smallest activity count.
    seed:
        Random seed for ``numpy.random.RandomState``.  Must be fixed for
        reproducibility; the default (42) is the project-wide public seed.

    Returns
    -------
    dev_groups : dict[str, list[Path]]
        Activity → development paths (sorted, stable order).
    val_groups : dict[str, list[Path]]
        Activity → validation paths (sorted, stable order).

    Raises
    ------
    ValueError
        If any activity has fewer than *val_per_class* captures.
    """
    # Validate feasibility before touching the RNG so a bad call never
    # partially advances the random state.
    for act, paths in groups.items():
        if len(paths) < val_per_class:
            raise ValueError(
                f"Activity '{act}' has only {len(paths)} captures; "
                f"cannot sample {val_per_class} for validation."
            )

    rng = np.random.RandomState(seed)

    dev_groups: Dict[str, List[Path]] = {}
    val_groups: Dict[str, List[Path]] = {}

    # Iterate in a fixed alphabetical order so the RNG is consumed in a
    # deterministic sequence regardless of dict insertion order.
    for act in sorted(groups.keys()):
        paths = list(groups[act])          # already sorted by discover_room_b
        n = len(paths)
        # Sample val_per_class indices without replacement
        val_indices = set(
            rng.choice(n, size=val_per_class, replace=False).tolist()
        )
        dev_paths = sorted(
            [p for i, p in enumerate(paths) if i not in val_indices],
            key=lambda p: p.name.lower(),
        )
        val_paths = sorted(
            [p for i, p in enumerate(paths) if i in val_indices],
            key=lambda p: p.name.lower(),
        )
        dev_groups[act] = dev_paths
        val_groups[act] = val_paths

    return dev_groups, val_groups


# ---------------------------------------------------------------------------
# Room A manifest rows
# ---------------------------------------------------------------------------

def room_a_manifest_rows(
    pcap_root: Path,
    val_root: Path,
) -> Generator[ManifestRow, None, None]:
    """Yield ManifestRow objects for all Room-A captures.

    Room-A split assignment comes from the supplied directory structure:

    * ``pcap_root/{activity}/*.pcap``          → split ``"dev"``
    * ``val_root/{activity}/*.pcap``           → split ``"val"``

    Parameters
    ----------
    pcap_root:
        Directory containing one sub-folder per activity with ``.pcap``
        files (e.g. ``csidata/``).  The ``validation/`` sub-folder is
        ignored if it happens to be present inside *pcap_root*.
    val_root:
        Directory containing one sub-folder per activity with validation
        ``.pcap`` files (e.g. ``csidata/validation/``).

    Yields
    ------
    ManifestRow
        One row per capture, in deterministic order
        (activity then filename, both lexicographic).

    Raises
    ------
    FileNotFoundError
        If either *pcap_root* or *val_root* does not exist.
    """
    pcap_root = Path(pcap_root)
    val_root  = Path(val_root)

    for root, check in [(pcap_root, "pcap_root"), (val_root, "val_root")]:
        if not root.exists():
            raise FileNotFoundError(f"Room A {check} not found: {root}")

    yield from _room_a_rows_from_dir(pcap_root, split="dev", skip_subdir="validation")
    yield from _room_a_rows_from_dir(val_root,  split="val", skip_subdir=None)


def _room_a_rows_from_dir(
    root: Path,
    split: str,
    skip_subdir: str | None,
) -> Generator[ManifestRow, None, None]:
    """Internal helper: walk *root*, yield one ManifestRow per .pcap."""
    for act_dir in sorted(root.iterdir(), key=lambda d: d.name.lower()):
        if not act_dir.is_dir():
            continue
        if skip_subdir and act_dir.name.lower() == skip_subdir.lower():
            continue

        activity = act_dir.name.lower()

        for pcap in sorted(act_dir.glob("*.pcap"), key=lambda p: p.name.lower()):
            capture_id = f"A__{activity}__{pcap.stem}"
            yield ManifestRow(
                capture_id=capture_id,
                room="A",
                activity=activity,
                split=split,
                source_file=str(pcap),
                n_packets_raw=0,      # filled in by build_intermediate.py
                n_packets_clean=0,    # filled in by build_intermediate.py
                intermediate_path="",  # filled in by build_intermediate.py
            )


# ---------------------------------------------------------------------------
# Room B manifest rows
# ---------------------------------------------------------------------------

def room_b_manifest_rows(
    csv_dir: Path,
    val_per_class: int = VAL_PER_CLASS,
    seed: int = VAL_SEED,
) -> Generator[ManifestRow, None, None]:
    """Discover Room-B CSVs, split them, and yield ManifestRow objects.

    Parameters
    ----------
    csv_dir:
        Directory containing all Experiment-3 ``*.csv`` files.
    val_per_class:
        Captures held out per activity for validation (default 25).
    seed:
        RNG seed for the validation split (default 42).

    Yields
    ------
    ManifestRow
        One row per capture with split label, room ``"B"``, canonical
        activity, and source file path.  ``n_packets_raw``,
        ``n_packets_clean``, and ``intermediate_path`` are filled in by
        ``build_intermediate.py`` after the CSV is actually loaded.
    """
    groups = discover_room_b(csv_dir)
    dev_groups, val_groups = make_room_b_split(
        groups, val_per_class=val_per_class, seed=seed
    )

    for split_label, split_groups in [("dev", dev_groups), ("val", val_groups)]:
        for act in sorted(split_groups.keys()):
            for path in split_groups[act]:
                capture_id = f"B__{act}__{path.stem}"
                yield ManifestRow(
                    capture_id=capture_id,
                    room="B",
                    activity=act,
                    split=split_label,
                    source_file=str(path),
                    n_packets_raw=0,
                    n_packets_clean=0,
                    intermediate_path="",
                )
