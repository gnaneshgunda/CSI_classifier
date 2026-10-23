"""
src/room_b_loader.py
====================
Read Experiment-3 (Room B) amplitude CSV files into float32 arrays and
produce :class:`~src.manifest.ManifestRow` entries.

File layout (confirmed by inspect_data.py)
------------------------------------------
All CSVs live flat inside one directory (no sub-folders).

Two header styles exist:

``subcarrier_named``
    Header row is ``Subcarrier_1, Subcarrier_2, …, Subcarrier_256``.
    Used by all ``*_Amp_*.csv`` files and all ``Walking_N.csv`` files
    except ``Walking_0.csv``.

``zero_indexed``
    Header row is ``0, 1, 2, …, 255`` (integer column names).
    Used by ``Walking_0.csv``, ``Empty_11.csv``, ``Lying_0.csv``,
    ``Sitting_0.csv``, ``Standing_0.csv``.

Both styles contain 256 columns of float/integer amplitude values and
a variable number of rows (observed: 152–500).  The column *content*
is the same; only the header string differs.

Filename → activity mapping
----------------------------
=================================  ========  ============
Filename pattern                   Prefix    Canonical activity
=================================  ========  ============
``Empty_Amp_N.csv``                empty     empty
``Lying_Amp_N.csv``                lying     lying
``Sitting_Amp_N.csv``              sitting   sit
``Standing_Amp_N.csv``             standing  stand
``Walking_N.csv``                  walking   walk
``Empty_11.csv``                   empty     empty   (anomalous header)
``Lying_0.csv``                    lying     lying   (anomalous header)
``Sitting_0.csv``                  sitting   sit     (anomalous header)
``Standing_0.csv``                 standing  stand   (anomalous header)
=================================  ========  ============

Non-matching filenames (e.g. ``model_plot_1.png``) are silently skipped.

Public API
----------
``discover_room_b(csv_dir)``
    Return sorted lists of Path objects grouped by canonical activity.

``load_csv(path)``
    Read one CSV → float32 ndarray of shape (n_rows, 256).

``load_capture(path, activity)``
    Load one CSV and return ``(array, ManifestRow-kwargs dict)``.
    The caller supplies the split label when constructing the full row.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.manifest import ManifestRow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected number of subcarrier columns.
N_SUBCARRIERS: int = 256

# Regex patterns for filename discovery (case-insensitive)
_AMP_RE   = re.compile(r"^(empty|lying|sitting|standing)_amp_(\d+)\.csv$", re.I)
_WALK_RE  = re.compile(r"^walking_(\d+)\.csv$", re.I)

# Anomalous files that lack the _Amp_ infix but still contain valid CSI data
_BARE_RE  = re.compile(r"^(empty|lying|sitting|standing|walking)_(\d+)\.csv$", re.I)

#: Maps the lower-case filename prefix to the canonical activity label used
#: throughout the pipeline (must match manifest._ALLOWED_ACTIVITIES).
PREFIX_TO_ACTIVITY: Dict[str, str] = {
    "empty":    "empty",
    "lying":    "lying",
    "sitting":  "sit",
    "standing": "stand",
    "walking":  "walk",
}

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_room_b(
    csv_dir: Path,
) -> Dict[str, List[Path]]:
    """Return a dict mapping canonical activity → sorted list of CSV paths.

    Both standard (``*_Amp_*``) and anomalous (bare-index) filenames are
    included.  Non-CSI files (images, etc.) are silently skipped.

    Parameters
    ----------
    csv_dir:
        Directory containing all Experiment-3 ``*.csv`` files.

    Returns
    -------
    dict[str, list[Path]]
        Keys are canonical activity labels (``empty``, ``lying``, ``sit``,
        ``stand``, ``walk``).  Values are path lists sorted
        deterministically by filename (lexicographic).

    Raises
    ------
    FileNotFoundError
        If *csv_dir* does not exist.
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Room B CSV directory not found: {csv_dir}")

    groups: Dict[str, List[Path]] = {act: [] for act in PREFIX_TO_ACTIVITY.values()}

    for p in csv_dir.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        name_lower = p.name.lower()

        m_amp  = _AMP_RE.match(name_lower)
        m_walk = _WALK_RE.match(name_lower)
        m_bare = _BARE_RE.match(name_lower)

        if m_amp:
            prefix = m_amp.group(1).lower()
        elif m_walk:
            prefix = "walking"
        elif m_bare:
            prefix = m_bare.group(1).lower()
        else:
            continue  # unrecognised filename – skip silently

        act = PREFIX_TO_ACTIVITY.get(prefix)
        if act is None:
            continue
        groups[act].append(p)

    # Sort each list deterministically by filename stem
    for act in groups:
        groups[act].sort(key=lambda p: p.name.lower())

    return groups


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> np.ndarray:
    """Read one Room-B CSV and return a float32 array of shape (n_rows, 256).

    Handles both header styles:

    * ``Subcarrier_1 … Subcarrier_256``  (standard)
    * ``0 … 255``                        (zero-indexed anomalous)

    In both cases the 256 columns are returned in their original left-to-right
    order (no reordering applied — that matches the order used in the standard
    files where Subcarrier_1 is the leftmost column).

    Parameters
    ----------
    path:
        Path to a Room-B CSV file.

    Returns
    -------
    np.ndarray, shape (n_rows, 256), dtype float32

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file does not have exactly 256 columns after parsing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, header=0)

    if df.shape[1] != N_SUBCARRIERS:
        raise ValueError(
            f"{path.name}: expected {N_SUBCARRIERS} columns, got {df.shape[1]}"
        )

    # Convert to float32; non-numeric entries become NaN
    arr = df.to_numpy(dtype=np.float32, na_value=np.nan)
    return arr


# ---------------------------------------------------------------------------
# Capture-level loader (returns array + metadata)
# ---------------------------------------------------------------------------

def load_capture(
    path: Path,
    activity: str,
    split: str,
    intermediate_path: str = "",
) -> Tuple[np.ndarray, ManifestRow]:
    """Load one CSV capture and return (array, ManifestRow).

    Parameters
    ----------
    path:
        Path to the CSV file.
    activity:
        Canonical activity label (e.g. ``"sit"``).
    split:
        ``"dev"`` or ``"val"``.
    intermediate_path:
        Relative path where the derived ``.npy`` file will be written.
        May be left empty at discovery time and filled in later.

    Returns
    -------
    array : np.ndarray, shape (n_rows, 256), dtype float32
    row   : ManifestRow
    """
    arr = load_csv(path)
    n_rows = arr.shape[0]

    # Capture ID: deterministic, derived only from the filename stem
    capture_id = f"B__{activity}__{path.stem}"

    row = ManifestRow(
        capture_id=capture_id,
        room="B",
        activity=activity,
        split=split,
        source_file=str(path),
        n_packets_raw=n_rows,
        n_packets_clean=n_rows,   # Room B CSVs are pre-cleaned by dataset authors
        intermediate_path=intermediate_path,
    )
    return arr, row
