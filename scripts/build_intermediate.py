"""
scripts/build_intermediate.py
==============================
Build ``derived/intermediate/{A,B}/{capture_id}.npy`` for every capture
in both rooms and write ``derived/manifest.csv``.

Pipeline per capture
--------------------
Room A (.pcap):
    pcap_reader.read_pcap()
    → amplitude.frames_to_amplitude()
    → cleaning.clean()
    → save .npy

Room B (.csv):
    room_b_loader.load_csv()
    → cleaning.clean()
    → save .npy

Outputs
-------
``derived/intermediate/A/{capture_id}.npy``
    float32 (n_clean, 256) amplitude array for each Room-A capture.

``derived/intermediate/B/{capture_id}.npy``
    float32 (n_clean, 256) amplitude array for each Room-B capture.

``derived/manifest.csv``
    One row per capture with capture_id, room, activity, split,
    source_file, n_packets_raw, n_packets_clean, intermediate_path.
    Written atomically at the end (not incrementally).

``derived/cleaning_report.csv``
    One row per capture with per-reason rejection counts.

Rules
-----
* Already-built captures are skipped (check .npy existence + --force flag).
* Shape ``[1] != 256`` causes an immediate hard failure.
* ``n_packets_clean < cfg.cleaning.min_valid_packets`` after cleaning:
  the .npy is still written (may be reflect-padded downstream by windowing)
  but a WARNING is printed.
* Capture IDs are derived only from room + activity + filename stem; they
  do not encode label information beyond what is already in the manifest.
* Multiprocessing via ``concurrent.futures.ProcessPoolExecutor`` with
  ``--workers`` flag (default: 1 to avoid scapy fork issues on Windows).
* Progress bar via tqdm.

Usage
-----
    python scripts/build_intermediate.py --config configs/experiment.yaml
    python scripts/build_intermediate.py --config configs/experiment.yaml --force
    python scripts/build_intermediate.py --config configs/experiment.yaml --workers 4
    python scripts/build_intermediate.py --smoke
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the repository root is on sys.path so ``src`` is importable
# regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SUBCARRIERS: int = 256


# ---------------------------------------------------------------------------
# Per-capture work item (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

@dataclass
class CaptureTask:
    """Everything needed to process one capture in a worker process."""
    capture_id: str
    room: str              # "A" | "B"
    activity: str
    split: str
    source_file: str
    out_path: str          # destination .npy path
    # Cleaning parameters (plain Python scalars → picklable)
    max_nan_fraction: float
    zero_energy_threshold: float
    drop_consecutive_duplicates: bool
    spike_z_threshold: float
    spike_mad_eps: float
    min_valid_packets: int


@dataclass
class CaptureResult:
    """Outcome of processing one capture."""
    capture_id: str
    room: str
    activity: str
    split: str
    source_file: str
    out_path: str
    n_packets_raw: int
    n_packets_clean: int
    dropped_non_finite: int
    dropped_near_zero: int
    dropped_consecutive_dup: int
    dropped_gain_spike: int
    skipped: bool = False   # True if already built and not forced
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker function (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _process_capture(task: CaptureTask) -> CaptureResult:
    """Process one capture and return a CaptureResult.

    This function runs in a worker process (or the main process if
    workers=1).  All imports are local to avoid multiprocessing issues.
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    from src.amplitude import frames_to_amplitude
    from src.cleaning import CleaningConfig, clean
    from src.room_b_loader import load_csv

    out_path = Path(task.out_path)

    try:
        # ── Load raw data ───────────────────────────────────────────────────
        if task.room == "A":
            from src.pcap_reader import read_pcap
            frames, skip_stats = read_pcap(Path(task.source_file))
            n_raw = len(frames) + skip_stats.total_skipped
            amp_raw, valid = frames_to_amplitude(frames, fftshift=True)
            # Replace invalid rows with NaN so cleaning rule 1 catches them
            amp_raw[~valid] = np.nan
        else:  # Room B
            amp_raw = load_csv(Path(task.source_file))
            n_raw = amp_raw.shape[0]

        # ── Shape guard ─────────────────────────────────────────────────────
        if amp_raw.shape[1] != N_SUBCARRIERS:
            raise ValueError(
                f"Expected {N_SUBCARRIERS} subcarriers, "
                f"got {amp_raw.shape[1]} in {task.source_file}"
            )

        # ── Clean ───────────────────────────────────────────────────────────
        cfg = CleaningConfig(
            max_nan_fraction_per_packet=task.max_nan_fraction,
            zero_energy_threshold=task.zero_energy_threshold,
            drop_consecutive_duplicates=task.drop_consecutive_duplicates,
            spike_z_threshold=task.spike_z_threshold,
            spike_mad_eps=task.spike_mad_eps,
        )
        cleaned, report = clean(amp_raw, cfg, capture_id=task.capture_id)

        # ── Save ────────────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), cleaned.astype(np.float32))

        return CaptureResult(
            capture_id=task.capture_id,
            room=task.room,
            activity=task.activity,
            split=task.split,
            source_file=task.source_file,
            out_path=str(out_path),
            n_packets_raw=n_raw,
            n_packets_clean=report.n_clean,
            dropped_non_finite=report.dropped_non_finite,
            dropped_near_zero=report.dropped_near_zero,
            dropped_consecutive_dup=report.dropped_consecutive_dup,
            dropped_gain_spike=report.dropped_gain_spike,
        )

    except Exception as exc:
        return CaptureResult(
            capture_id=task.capture_id,
            room=task.room,
            activity=task.activity,
            split=task.split,
            source_file=task.source_file,
            out_path=str(out_path),
            n_packets_raw=0,
            n_packets_clean=0,
            dropped_non_finite=0,
            dropped_near_zero=0,
            dropped_consecutive_dup=0,
            dropped_gain_spike=0,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_tasks(
    cfg,             # ExperimentConfig
    force: bool,
) -> List[CaptureTask]:
    """Enumerate all captures and return one CaptureTask per capture."""
    from src.splits import room_a_manifest_rows, room_b_manifest_rows

    intermediate_dir = Path(cfg.paths.intermediate_dir)

    clean_kwargs = dict(
        max_nan_fraction=cfg.cleaning.max_nan_fraction_per_packet,
        zero_energy_threshold=0.0,    # drop all-zero rows
        drop_consecutive_duplicates=True,
        spike_z_threshold=6.0,        # sensible default; not in yaml yet
        spike_mad_eps=1e-6,
        min_valid_packets=cfg.cleaning.min_valid_packets,
    )

    tasks: List[CaptureTask] = []

    # Room A
    for row in room_a_manifest_rows(
        Path(cfg.paths.room_a_pcap_dir),
        Path(cfg.paths.room_a_val_dir),
    ):
        out_path = intermediate_dir / "A" / f"{row.capture_id}.npy"
        if not force and out_path.exists():
            continue
        tasks.append(CaptureTask(
            capture_id=row.capture_id,
            room="A",
            activity=row.activity,
            split=row.split,
            source_file=row.source_file,
            out_path=str(out_path),
            **clean_kwargs,
        ))

    # Room B
    for row in room_b_manifest_rows(
        Path(cfg.paths.room_b_csv_dir),
        val_per_class=cfg.splits.val_captures_per_class,
        seed=cfg.splits.val_seed,
    ):
        out_path = intermediate_dir / "B" / f"{row.capture_id}.npy"
        if not force and out_path.exists():
            continue
        tasks.append(CaptureTask(
            capture_id=row.capture_id,
            room="B",
            activity=row.activity,
            split=row.split,
            source_file=row.source_file,
            out_path=str(out_path),
            **clean_kwargs,
        ))

    return tasks


def _all_manifest_rows(cfg):
    """Yield ALL manifest rows (including already-built captures) so we can
    write a complete manifest.csv even when some captures were skipped."""
    from src.splits import room_a_manifest_rows, room_b_manifest_rows

    yield from room_a_manifest_rows(
        Path(cfg.paths.room_a_pcap_dir),
        Path(cfg.paths.room_a_val_dir),
    )
    yield from room_b_manifest_rows(
        Path(cfg.paths.room_b_csv_dir),
        val_per_class=cfg.splits.val_captures_per_class,
        seed=cfg.splits.val_seed,
    )


def _write_manifest(cfg, results: List[CaptureResult]) -> None:
    """Write derived/manifest.csv merging build results with any pre-existing
    intermediate files that were skipped (already built)."""
    from src.manifest import ManifestRow, write_manifest
    from src.splits import room_a_manifest_rows, room_b_manifest_rows

    intermediate_dir = Path(cfg.paths.intermediate_dir)

    # Index results by capture_id
    result_map = {r.capture_id: r for r in results}

    rows: List[ManifestRow] = []
    for seed_row in _all_manifest_rows(cfg):
        cid = seed_row.capture_id
        room = seed_row.room
        out_path = intermediate_dir / room / f"{cid}.npy"

        if cid in result_map:
            res = result_map[cid]
            n_raw   = res.n_packets_raw
            n_clean = res.n_packets_clean
        elif out_path.exists():
            # Already built on a previous run — load packet counts from npy
            try:
                arr = np.load(str(out_path))
                n_clean = arr.shape[0]
                n_raw   = n_clean   # raw count not cached; use clean as proxy
            except Exception:
                n_raw = n_clean = 0
        else:
            n_raw = n_clean = 0

        rows.append(ManifestRow(
            capture_id=cid,
            room=seed_row.room,
            activity=seed_row.activity,
            split=seed_row.split,
            source_file=seed_row.source_file,
            n_packets_raw=n_raw,
            n_packets_clean=n_clean,
            intermediate_path=str(out_path),
        ))

    manifest_path = Path(cfg.paths.manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, manifest_path)
    log.info("Manifest written → %s  (%d rows)", manifest_path, len(rows))


def _write_cleaning_report(cfg, results: List[CaptureResult]) -> None:
    """Write per-capture cleaning statistics to derived/cleaning_report.csv."""
    from src.cleaning import CleaningReport, write_report_csv

    reports = [
        CleaningReport(
            capture_id=r.capture_id,
            n_raw=r.n_packets_raw,
            n_clean=r.n_packets_clean,
            dropped_non_finite=r.dropped_non_finite,
            dropped_near_zero=r.dropped_near_zero,
            dropped_consecutive_dup=r.dropped_consecutive_dup,
            dropped_gain_spike=r.dropped_gain_spike,
        )
        for r in results
        if not r.skipped and r.error is None
    ]
    report_path = Path(cfg.paths.derived_dir) / "cleaning_report.csv"
    write_report_csv(reports, report_path)
    log.info("Cleaning report → %s  (%d captures)", report_path, len(reports))


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

def _run_smoke(args: argparse.Namespace) -> None:
    """End-to-end smoke test on tiny synthetic data (no real files needed)."""
    import tempfile, shutil

    print("[smoke] Building synthetic intermediate data ...")
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inter_dir = tmp / "intermediate"
        manifest_path = tmp / "manifest.csv"

        # Synthetic amplitude arrays
        captures = [
            ("A__fall__synth0", "A", "fall", "dev"),
            ("A__sit__synth1",  "A", "sit",  "val"),
            ("B__empty__synth2","B", "empty","dev"),
        ]
        results: List[CaptureResult] = []
        for cid, room, act, split in captures:
            n = rng.integers(20, 40)
            amp = rng.uniform(100, 2000, size=(n, 256)).astype(np.float32)

            # Inject one dirty row
            amp[0, :] = 0.0

            from src.cleaning import CleaningConfig, clean
            cleaned, report = clean(amp, capture_id=cid)

            assert cleaned.shape[1] == N_SUBCARRIERS, \
                f"Shape guard failed: {cleaned.shape}"

            out = inter_dir / room / f"{cid}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), cleaned)

            results.append(CaptureResult(
                capture_id=cid, room=room, activity=act, split=split,
                source_file=f"synthetic/{cid}.pcap",
                out_path=str(out),
                n_packets_raw=int(n),
                n_packets_clean=report.n_clean,
                dropped_non_finite=report.dropped_non_finite,
                dropped_near_zero=report.dropped_near_zero,
                dropped_consecutive_dup=report.dropped_consecutive_dup,
                dropped_gain_spike=report.dropped_gain_spike,
            ))

        # Write manifest
        from src.manifest import ManifestRow, write_manifest
        from src.cleaning import write_report_csv, CleaningReport

        rows = [
            ManifestRow(
                capture_id=r.capture_id, room=r.room,
                activity=r.activity, split=r.split,
                source_file=r.source_file,
                n_packets_raw=r.n_packets_raw,
                n_packets_clean=r.n_packets_clean,
                intermediate_path=r.out_path,
            )
            for r in results
        ]
        write_manifest(rows, manifest_path)

        # Verify
        from src.manifest import read_manifest
        loaded = read_manifest(manifest_path)
        assert len(loaded) == 3
        for row in loaded:
            arr = np.load(row.intermediate_path)
            assert arr.shape[1] == N_SUBCARRIERS, \
                f"Loaded shape guard failed: {arr.shape}"
            assert arr.dtype == np.float32

        print(f"[smoke] {len(results)} captures processed")
        print(f"[smoke] Manifest verified: {len(loaded)} rows")
        print("[smoke] Shape guard (256 subcarriers): OK")
        print("[smoke] All smoke checks passed ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build derived/intermediate/*.npy from raw pcap/csv captures."
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process captures even if the .npy already exists.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of worker processes (default: 1). "
            "Use 1 on Windows to avoid scapy fork issues."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run on synthetic data only; no real files needed.",
    )
    parser.add_argument(
        "--room", choices=["A", "B", "both"], default="both",
        help="Restrict processing to one room (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        _run_smoke(args)
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):   # type: ignore[misc]
            return it

    from src.config import load_config
    cfg = load_config(args.config)

    tasks = _build_tasks(cfg, force=args.force)

    n_skip = 0
    # Count pre-built captures (those not in tasks)
    intermediate_dir = Path(cfg.paths.intermediate_dir)
    all_rows_list = list(_all_manifest_rows(cfg))
    n_total = len(all_rows_list)
    n_skip = n_total - len(tasks)

    if n_skip:
        log.info("Skipping %d already-built captures (use --force to rebuild)", n_skip)

    if not tasks:
        log.info("Nothing to build.")
    else:
        log.info("Processing %d captures with %d worker(s) ...", len(tasks), args.workers)

    results: List[CaptureResult] = []
    errors:  List[CaptureResult] = []

    if args.workers == 1 or len(tasks) == 0:
        # Sequential path (avoids Windows fork/scapy issues)
        for task in tqdm(tasks, desc="Building", unit="capture"):
            res = _process_capture(task)
            results.append(res)
            if res.error:
                log.error("FAILED %s: %s", res.capture_id, res.error.splitlines()[0])
                errors.append(res)
            else:
                below_min = res.n_packets_clean < cfg.cleaning.min_valid_packets
                if below_min:
                    log.warning(
                        "Capture %s has only %d clean packets (< min %d)",
                        res.capture_id, res.n_packets_clean,
                        cfg.cleaning.min_valid_packets,
                    )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {pool.submit(_process_capture, t): t for t in tasks}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(tasks),
                desc="Building",
                unit="capture",
            ):
                res = fut.result()
                results.append(res)
                if res.error:
                    log.error(
                        "FAILED %s: %s",
                        res.capture_id, res.error.splitlines()[0],
                    )
                    errors.append(res)

    # Write manifest (includes skipped captures loaded from disk)
    _write_manifest(cfg, results)

    # Write cleaning report for newly-processed captures
    if results:
        _write_cleaning_report(cfg, results)

    # Summary
    n_ok = len(results) - len(errors)
    log.info(
        "Done. Built: %d  Skipped: %d  Errors: %d",
        n_ok, n_skip, len(errors),
    )

    if errors:
        log.error(
            "%d capture(s) failed. First error:\n%s",
            len(errors), errors[0].error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
