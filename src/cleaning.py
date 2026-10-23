"""
src/cleaning.py
===============
Row-level quality control for CSI amplitude arrays.

Input
-----
A 2-D array of shape ``(n_rows, 256)`` containing either:

* **Real-valued amplitude** (float32, from ``csi_decode.amplitude_batch`` or
  Room-B CSVs), or
* **Complex CSI** (complex64, from ``csi_decode.decode_batch``).

When a complex array is supplied the amplitude ``|H|`` is computed first;
the rest of the pipeline operates on amplitude only.

Output
------
``clean(arr, cfg)`` returns:

* ``cleaned : np.ndarray`` – shape ``(m, 256)`` float32, ``m ≤ n_rows``,
  rows in original time order (no reordering, no interpolation).
* ``report  : CleaningReport`` – dataclass with per-reason rejection counts
  and a boolean mask aligned to the *original* row index.

Cleaning rules (applied in order, all configurable)
----------------------------------------------------
1. **non_finite** – drop rows that contain any NaN or ±Inf value.
2. **near_zero_energy** – drop rows whose total energy
   ``sum(amplitude²)`` is ≤ ``zero_energy_threshold``.
3. **consecutive_duplicate** – drop any row that is an *exact* duplicate
   of its immediately preceding kept row (bit-for-bit equal after float32
   conversion).  Time order is preserved; only adjacent duplicates are
   removed.
4. **gain_spike** – drop rows whose row energy is an outlier under a
   robust z-score: ``|energy - median(energy)| / (MAD + eps) > spike_z``.
   MAD = median absolute deviation.  ``eps`` prevents division by zero on
   constant captures.

Rules are applied sequentially; a row rejected by rule 1 is never tested
by rules 2–4.  The ``valid_mask`` in the report is aligned to the original
row indices so provenance is fully traceable.

Configuration dataclass
-----------------------
``CleaningConfig`` mirrors the ``cleaning:`` section of ``experiment.yaml``
and adds two new fields for the gain-spike rule that are not in the YAML yet
(``spike_z_threshold`` and ``zero_energy_threshold``).  All fields have
sensible defaults so callers that omit them get safe behaviour.

CSV report
----------
``write_report_csv(reports, path)`` writes one row per capture to a CSV
containing capture_id, room, activity, n_raw, n_clean, and per-reason
rejection counts.  Useful for auditing which captures lose the most packets.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CleaningConfig:
    """Parameters controlling every cleaning rule.

    All thresholds correspond to entries (or planned entries) in the
    ``cleaning:`` section of ``experiment.yaml``.

    Attributes
    ----------
    max_nan_fraction_per_packet:
        A row is dropped as *non_finite* if the fraction of non-finite
        values exceeds this threshold.  Setting to 0.0 drops any row with
        at least one NaN/Inf; setting to 1.0 disables the rule.
    zero_energy_threshold:
        Rows with total energy (sum of squared amplitudes) at or below this
        value are dropped as *near_zero_energy*.  Default 0.0 drops only
        all-zero rows.
    drop_consecutive_duplicates:
        If True, exact consecutive duplicate rows (float32 bit-equal) are
        dropped.
    spike_z_threshold:
        Rows whose robust z-score of row energy exceeds this value are
        dropped as *gain_spike*.  Set to ``float('inf')`` to disable.
    spike_mad_eps:
        Small constant added to MAD to prevent division by zero on
        constant-energy captures.
    """
    max_nan_fraction_per_packet: float = 0.10
    zero_energy_threshold: float = 0.0
    drop_consecutive_duplicates: bool = True
    spike_z_threshold: float = 6.0
    spike_mad_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_nan_fraction_per_packet <= 1.0):
            raise ValueError("max_nan_fraction_per_packet must be in [0, 1]")
        if self.zero_energy_threshold < 0:
            raise ValueError("zero_energy_threshold must be >= 0")
        if self.spike_z_threshold <= 0:
            raise ValueError("spike_z_threshold must be > 0")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Per-capture cleaning summary.

    Attributes
    ----------
    capture_id:
        Identifier of the source capture (may be empty string if not known).
    n_raw:
        Number of rows before any cleaning.
    n_clean:
        Number of rows that survived all rules.
    dropped_non_finite:
        Rows dropped by rule 1 (non-finite values).
    dropped_near_zero:
        Rows dropped by rule 2 (near-zero energy).
    dropped_consecutive_dup:
        Rows dropped by rule 3 (consecutive duplicate).
    dropped_gain_spike:
        Rows dropped by rule 4 (gain spike).
    valid_mask:
        Boolean array of shape ``(n_raw,)`` – True for rows that survived.
        Aligned to the original row index.
    """
    capture_id: str = ""
    n_raw: int = 0
    n_clean: int = 0
    dropped_non_finite: int = 0
    dropped_near_zero: int = 0
    dropped_consecutive_dup: int = 0
    dropped_gain_spike: int = 0
    valid_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    @property
    def total_dropped(self) -> int:
        return (
            self.dropped_non_finite
            + self.dropped_near_zero
            + self.dropped_consecutive_dup
            + self.dropped_gain_spike
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a flat dict suitable for CSV writing (excludes valid_mask)."""
        return {
            "capture_id":             self.capture_id,
            "n_raw":                  self.n_raw,
            "n_clean":                self.n_clean,
            "dropped_non_finite":     self.dropped_non_finite,
            "dropped_near_zero":      self.dropped_near_zero,
            "dropped_consecutive_dup": self.dropped_consecutive_dup,
            "dropped_gain_spike":     self.dropped_gain_spike,
            "total_dropped":          self.total_dropped,
        }


# ---------------------------------------------------------------------------
# Column names for the CSV report
# ---------------------------------------------------------------------------

_REPORT_COLUMNS = [
    "capture_id",
    "n_raw",
    "n_clean",
    "dropped_non_finite",
    "dropped_near_zero",
    "dropped_consecutive_dup",
    "dropped_gain_spike",
    "total_dropped",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(
    arr: np.ndarray,
    cfg: Optional[CleaningConfig] = None,
    *,
    capture_id: str = "",
) -> Tuple[np.ndarray, CleaningReport]:
    """Apply all cleaning rules to *arr* and return the cleaned amplitude.

    Parameters
    ----------
    arr:
        Input array of shape ``(n_rows, 256)``.
        * ``float32`` or ``float64`` – treated as amplitude directly.
        * ``complex64`` or ``complex128`` – ``|arr|`` is computed first.
        Other dtypes are cast to float32 before processing.
    cfg:
        Cleaning configuration.  If ``None`` the default ``CleaningConfig``
        is used.
    capture_id:
        Optional identifier written into the ``CleaningReport``.

    Returns
    -------
    cleaned : np.ndarray, shape (m, 256), dtype float32
        Rows that survived all rules, in original order.
    report  : CleaningReport
        Per-reason rejection counts and valid_mask.

    Notes
    -----
    * No interpolation is ever performed.  Dropped rows are simply absent
      from the output.
    * The function is pure; it never modifies *arr* in-place.
    """
    if cfg is None:
        cfg = CleaningConfig()

    n_raw = arr.shape[0]
    report = CleaningReport(capture_id=capture_id, n_raw=n_raw)

    # Convert to real-valued float32 amplitude
    amp = _to_amplitude(arr)   # (n_raw, 256) float32

    if n_raw == 0:
        report.n_clean = 0
        report.valid_mask = np.zeros(0, dtype=bool)
        return amp, report

    # alive[i] = True means row i has not yet been dropped
    alive = np.ones(n_raw, dtype=bool)

    # ── Rule 1: non-finite ──────────────────────────────────────────────────
    non_finite_mask = _non_finite_rows(amp, cfg.max_nan_fraction_per_packet)
    alive &= ~non_finite_mask
    report.dropped_non_finite = int(non_finite_mask.sum())

    # ── Rule 2: near-zero energy ────────────────────────────────────────────
    near_zero_mask = _near_zero_rows(amp, cfg.zero_energy_threshold, alive)
    alive &= ~near_zero_mask
    report.dropped_near_zero = int(near_zero_mask.sum())

    # ── Rule 3: consecutive duplicates ──────────────────────────────────────
    if cfg.drop_consecutive_duplicates:
        dup_mask = _consecutive_duplicate_rows(amp, alive)
        alive &= ~dup_mask
        report.dropped_consecutive_dup = int(dup_mask.sum())

    # ── Rule 4: gain spike ──────────────────────────────────────────────────
    spike_mask = _gain_spike_rows(
        amp, alive, cfg.spike_z_threshold, cfg.spike_mad_eps
    )
    alive &= ~spike_mask
    report.dropped_gain_spike = int(spike_mask.sum())

    report.valid_mask = alive
    report.n_clean = int(alive.sum())

    return amp[alive].copy(), report


# ---------------------------------------------------------------------------
# CSV report writer
# ---------------------------------------------------------------------------

def write_report_csv(
    reports: Sequence[CleaningReport],
    path: Path,
) -> None:
    """Write per-capture cleaning summaries to *path* as CSV.

    Parameters
    ----------
    reports:
        Iterable of ``CleaningReport`` objects (one per capture).
    path:
        Destination CSV path.  Parent directories are created if absent.
        Written atomically via a ``.tmp`` file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_REPORT_COLUMNS)
            writer.writeheader()
            for r in reports:
                writer.writerow(r.to_dict())
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Private rule implementations
# ---------------------------------------------------------------------------

def _to_amplitude(arr: np.ndarray) -> np.ndarray:
    """Return real-valued float32 amplitude from *arr*.

    * Complex input  → ``np.abs(arr)`` cast to float32.
    * Real input     → cast to float32 (values treated as amplitude already).
    """
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float32)
    return arr.astype(np.float32)


def _non_finite_rows(
    amp: np.ndarray,
    max_nan_fraction: float,
) -> np.ndarray:
    """Return boolean mask (shape n_rows) – True where row is non-finite.

    A row is flagged if the fraction of non-finite elements exceeds
    *max_nan_fraction*.  At ``max_nan_fraction=0.0`` any single bad value
    flags the row.
    """
    non_finite = ~np.isfinite(amp)          # (n, 256) bool
    if max_nan_fraction == 0.0:
        return non_finite.any(axis=1)
    frac = non_finite.mean(axis=1)          # (n,) float
    return frac > max_nan_fraction


def _near_zero_rows(
    amp: np.ndarray,
    threshold: float,
    alive: np.ndarray,
) -> np.ndarray:
    """Return boolean mask – True where alive row has energy ≤ threshold.

    Only rows that are still *alive* are flagged; already-dropped rows
    produce False so their drop is not double-counted.
    """
    energy = (amp.astype(np.float64) ** 2).sum(axis=1)   # (n,) float64
    below = (energy <= threshold) & alive
    return below


def _consecutive_duplicate_rows(
    amp: np.ndarray,
    alive: np.ndarray,
) -> np.ndarray:
    """Return boolean mask – True for alive rows that are exact duplicates
    of the nearest preceding alive row.

    Comparison is done on float32 bit patterns (``np.array_equal`` per row),
    so only bit-for-bit identical rows are flagged — small floating-point
    perturbations are preserved.
    """
    n = amp.shape[0]
    dup_mask = np.zeros(n, dtype=bool)
    prev_row: Optional[np.ndarray] = None

    for i in range(n):
        if not alive[i]:
            continue
        row = amp[i]
        if prev_row is not None and np.array_equal(row, prev_row):
            dup_mask[i] = True
        else:
            prev_row = row   # update "last kept row" only on non-dup

    return dup_mask


def _gain_spike_rows(
    amp: np.ndarray,
    alive: np.ndarray,
    z_threshold: float,
    eps: float,
) -> np.ndarray:
    """Return boolean mask – True for alive rows whose row energy is an
    outlier under the robust z-score (median / MAD).

    Only alive rows participate in the median/MAD computation, preventing
    already-rejected rows from distorting the statistics.

    Formula::

        energy_i = sum(amp_i ^ 2)
        z_i = |energy_i - median(energy_alive)| / (MAD(energy_alive) + eps)
        flagged if z_i > z_threshold

    If fewer than 2 alive rows exist the rule is skipped (returns all False).
    """
    n = amp.shape[0]
    spike_mask = np.zeros(n, dtype=bool)

    alive_idx = np.where(alive)[0]
    if len(alive_idx) < 2:
        return spike_mask

    energy = (amp.astype(np.float64) ** 2).sum(axis=1)   # (n,) float64
    alive_energy = energy[alive_idx]

    med = np.median(alive_energy)
    mad = np.median(np.abs(alive_energy - med))
    robust_z = np.abs(energy - med) / (mad + eps)

    spike_mask = (robust_z > z_threshold) & alive
    return spike_mask
