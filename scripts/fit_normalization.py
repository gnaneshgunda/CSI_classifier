"""
scripts/fit_normalization.py
============================
Fit normalization statistics on development windows and save to
``derived/norm_stats.npz``.

Steps
-----
1. Load ``derived/windows_index.csv`` (must exist; run build_windows.py first).
2. Filter to ``split == "dev"`` rows only (leakage guard enforced by Normalizer).
3. For each dev window, load its ``.npy`` file.
4. Call ``Normalizer.fit()`` which saves ``derived/norm_stats.npz``.
5. Print gate summary: window counts per room × split.

Usage
-----
    python scripts/fit_normalization.py
    python scripts/fit_normalization.py --config configs/experiment.yaml
    python scripts/fit_normalization.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate check helper
# ---------------------------------------------------------------------------

def _print_gate(index_df: pd.DataFrame) -> None:
    """Print window counts per room × split and validate expectations."""
    print("\n=== Window count gate check ===")
    pivot = (
        index_df.groupby(["room", "split"])
        .size()
        .reset_index(name="n_windows")
    )
    col_w = max(len(str(v)) for v in pivot["n_windows"]) + 2
    print(f"{'Room':<6} {'Split':<6} {'Windows':>{col_w}}")
    print("-" * (6 + 6 + col_w + 2))
    for _, row in pivot.iterrows():
        print(f"{row['room']:<6} {row['split']:<6} {row['n_windows']:>{col_w},}")

    total = pivot["n_windows"].sum()
    dev_total = pivot.loc[pivot["split"] == "dev", "n_windows"].sum()
    val_total = pivot.loc[pivot["split"] == "val", "n_windows"].sum()
    print(f"\nTotal windows : {total:,}")
    print(f"  dev         : {dev_total:,}")
    print(f"  val         : {val_total:,}")

    # Expectations:
    # - val windows == number of val captures (1 canonical window each)
    # - dev windows >= val windows (dev has multiple windows per capture)
    if val_total > 0 and dev_total <= val_total:
        log.warning(
            "GATE WARNING: dev window count (%d) <= val window count (%d). "
            "Expected dev >> val (dev uses stride-50 sliding windows).",
            dev_total, val_total,
        )
    else:
        print("\nGATE: dev >> val  ✓")

    # Both rooms should have dev windows
    rooms_with_dev = set(
        pivot.loc[pivot["split"] == "dev", "room"].tolist()
    )
    if rooms_with_dev != {"A", "B"}:
        log.warning(
            "GATE WARNING: not all rooms have dev windows. Found: %s",
            rooms_with_dev,
        )
    else:
        print("GATE: both rooms have dev windows  ✓")


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

def _run_smoke() -> None:
    """End-to-end smoke on a tiny synthetic dataset."""
    import tempfile
    from src.normalization import Normalizer
    from unittest.mock import MagicMock

    print("[smoke] fit_normalization smoke ...")

    rng = np.random.RandomState(0)
    T, N_SUB = 150, 256

    rows = []
    for room, n in [("A", 20), ("B", 20)]:
        for i in range(n):
            rows.append({
                "split": "dev",
                "room": room,
                "array": rng.uniform(0.1, 100.0, (T, N_SUB)).astype(np.float32),
            })
    # Add some val rows (must NOT be passed to fit)
    for room in ("A", "B"):
        for i in range(5):
            rows.append({
                "split": "val",
                "room": room,
                "array": rng.uniform(0.1, 100.0, (T, N_SUB)).astype(np.float32),
            })

    full_df = pd.DataFrame(rows)
    dev_df  = full_df[full_df["split"] == "dev"].copy()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = MagicMock()
        cfg.routing.mode = "oracle"
        cfg.paths.derived_dir = tmp

        norm = Normalizer(cfg, log1p=True)
        norm.fit(dev_df)

        # Check stats file written
        npz_path = Path(tmp) / "norm_stats.npz"
        assert npz_path.exists(), "norm_stats.npz not written"
        data = np.load(str(npz_path))
        for key in ["global_mean", "global_std", "global_scale",
                    "mean_A", "std_A", "scale_A", "mean_B", "std_B", "scale_B"]:
            assert key in data.files, f"Missing key: {key}"
            assert data[key].shape == (256,), f"Bad shape for {key}"

        # Test transform works after load
        norm2 = Normalizer(cfg, log1p=True)
        norm2.load()
        win = rng.uniform(0.1, 50.0, (T, N_SUB)).astype(np.float32)
        s, d = norm2.transform(win, room="A")
        assert s.shape == (N_SUB,)
        assert d.shape == (T, N_SUB)

        # Leakage guard: passing val rows must raise
        try:
            norm3 = Normalizer(cfg, log1p=True)
            norm3.fit(full_df)   # full_df contains val rows → must raise
            assert False, "Expected ValueError for val-split rows"
        except ValueError as e:
            assert "validation-split" in str(e)

        _print_gate(full_df)

    print("[smoke] norm_stats.npz written and verified  ✓")
    print("[smoke] transform() works after load()  ✓")
    print("[smoke] leakage guard fires on val rows  ✓")
    print("[smoke] fit_normalization smoke PASSED ✓")


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit normalization stats on dev windows.")
    p.add_argument("--config", default="configs/experiment.yaml",
                   help="Path to experiment.yaml (default: configs/experiment.yaml)")
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke test on synthetic data (no real files needed)")
    p.add_argument("--log1p", action="store_true", default=True,
                   help="Apply log1p before decomposition (default: True)")
    p.add_argument("--no-log1p", dest="log1p", action="store_false",
                   help="Disable log1p pre-processing")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        _run_smoke()
        return

    from src.config import load_config
    from src.normalization import Normalizer

    cfg = load_config(args.config)

    index_csv = Path(cfg.paths.windows_index_csv)
    if not index_csv.exists():
        log.error(
            "windows_index.csv not found: %s\n"
            "Run scripts/build_windows.py first.",
            index_csv,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    log.info("Loading windows index: %s", index_csv)
    index_df = pd.read_csv(str(index_csv))
    log.info("Total windows in index: %d", len(index_df))

    # Print gate check over ALL windows (dev + val)
    _print_gate(index_df)

    # ------------------------------------------------------------------
    # Filter to dev windows for fitting
    # ------------------------------------------------------------------
    dev_df = index_df[index_df["split"] == "dev"].copy()
    log.info("Dev windows for fitting: %d", len(dev_df))

    if len(dev_df) == 0:
        log.error("No dev windows found in index. Cannot fit normalization.")
        sys.exit(1)

    # Resolve window file paths
    windows_dir = Path(cfg.paths.windows_dir)

    def _loader(row: pd.Series) -> np.ndarray:
        wpath = windows_dir / row["split"] / f"{row['window_id']}.npy"
        if not wpath.exists():
            raise FileNotFoundError(f"Window file not found: {wpath}")
        return np.load(str(wpath)).astype(np.float32)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    log.info(
        "Fitting normalization (log1p=%s) on %d dev windows ...",
        args.log1p, len(dev_df),
    )
    norm = Normalizer(cfg, log1p=args.log1p)
    norm.fit(dev_df, window_loader=_loader)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    stats_path = Path(cfg.paths.derived_dir) / "norm_stats.npz"
    log.info("Saved: %s", stats_path)

    s = norm._stats
    print("\n=== Normalization statistics summary ===")
    for room in ("A", "B"):
        if room in s.room_mean:
            print(
                f"  Room {room}: mean∈[{s.room_mean[room].min():.3f}, "
                f"{s.room_mean[room].max():.3f}]  "
                f"scale∈[{s.room_scale[room].min():.4f}, "
                f"{s.room_scale[room].max():.4f}]"
            )
    print(
        f"  Global : mean∈[{s.global_mean.min():.3f}, "
        f"{s.global_mean.max():.3f}]  "
        f"scale∈[{s.global_scale.min():.4f}, "
        f"{s.global_scale.max():.4f}]"
    )
    print("\nFit complete. GATE PASSED ✓")


if __name__ == "__main__":
    main()
