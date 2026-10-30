"""
scripts/task2_eval.py
=====================
Task 2 – Linear evaluation of self-supervised encoder representations.

Sweeps five conditions, trains a LogisticRegression probe on development
embeddings, and scores on fixed validation windows.

Conditions
----------
1. ``handcrafted``  – 524-d hand-crafted features (src.features).
2. ``random``       – randomly initialised ResNet1DEncoder (no training).
3. ``vicreg_aug``   – encoder pre-trained with VICReg + augmentation.
4. ``vicreg_noaug`` – encoder pre-trained with VICReg, no augmentation.
5. ``masked``       – encoder pre-trained with masked auto-encoding.

Scope
-----
``--scope joint``    → one probe trained on all rooms together.
``--scope per_room`` → one probe per room.
``--scope both``     → both (default).

Output
------
``reports/task2_probe_results.csv`` with columns::

    experiment, target_room, K, metric, value

Matching the joint results-table format:
  * ``experiment``   = ``probe_{condition}``
  * ``target_room``  = ``joint`` | ``A`` | ``B``
  * ``K``            = -1  (full-probe; sentinel for join with Task 3 table)
  * ``metric``       = ``macro_f1`` | ``balanced_accuracy``
  * ``value``        = float

Usage
-----
    # All conditions, both scopes
    python scripts/task2_eval.py --config configs/experiment.yaml

    # Single condition
    python scripts/task2_eval.py --config configs/experiment.yaml \
        --conditions handcrafted random --scope joint

    # Smoke test (synthetic data, no real files)
    python scripts/task2_eval.py --smoke

Options
-------
``--config PATH``         Path to experiment.yaml (default: configs/experiment.yaml)
``--conditions C [C ...]``  Conditions to run (default: all five)
``--scope {joint|per_room|both}``  Training scope (default: both)
``--checkpoint_dir DIR``  Directory containing encoder .pt files
                          (default: cfg.paths.checkpoints_dir)
``--out PATH``            Override output CSV path
``--device {cpu|cuda}``   Torch device (default: cpu)
``--smoke``               Run on tiny synthetic data (no real files needed)
``--stage {pretrain|eval|both}``   Pipeline stage (default: eval)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("task2_eval")

_CSV_COLS = ["experiment", "target_room", "K", "metric", "value"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke(args: argparse.Namespace) -> None:
    """End-to-end smoke on tiny synthetic data (no real files needed)."""
    import csv as csv_mod
    import tempfile

    import numpy as np
    import pandas as pd

    from src.probe import (
        ALL_CONDITIONS,
        EmbeddingExtractor,
        LinearProbe,
        ProbeNormalizer,
        run_probe,
    )

    print("[smoke] task2_eval smoke test ...")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp = Path(tmp_dir_str)
        windows_dir = tmp / "windows"
        (windows_dir / "dev").mkdir(parents=True)
        (windows_dir / "val").mkdir(parents=True)
        ckpt_dir = tmp / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        reports_dir = tmp / "reports"

        rng = np.random.default_rng(42)
        T, N_SUB = 150, 256

        # Shared activities (both rooms): sit, stand, walk
        # Exclusive activities: A: sitdown, standup, fall; B: empty, lying
        activities_A = ["sit", "stand", "walk", "sitdown", "standup", "fall"]
        activities_B = ["sit", "stand", "walk", "empty", "lying"]

        index_rows = []
        win_count = 0

        def _write_window(split: str, room: str, cap_id: str, act: str) -> dict:
            nonlocal win_count
            w_id = f"{room}_{cap_id}_w{win_count:04d}"
            win_count += 1
            arr = rng.uniform(0.1, 10.0, size=(T, N_SUB)).astype(np.float32)
            # Add activity-specific frequency / pattern
            arr += float(hash(act) % 10) * 0.5
            out_p = windows_dir / split / f"{w_id}.npy"
            np.save(str(out_p), arr)
            return {
                "window_id":   w_id,
                "capture_id":  cap_id,
                "room":        room,
                "activity":    act,
                "split":       split,
                "start":       0,
                "config_hash": "smoke",
            }

        n_caps = 6
        for room, activities in [("A", activities_A), ("B", activities_B)]:
            for act in activities:
                for c in range(n_caps):
                    cap_id = f"{room}__{act}__cap{c:03d}"
                    # dev: 3 windows per capture
                    for _ in range(3):
                        index_rows.append(_write_window("dev", room, cap_id, act))
                    # val: 1 window per capture
                    index_rows.append(_write_window("val", room, cap_id, act))

        index_df = pd.DataFrame(index_rows)

        # Only run handcrafted and random in smoke (no real checkpoints)
        smoke_conditions = ["handcrafted", "random"]

        all_results: List[dict] = []
        for cond in smoke_conditions:
            rows = run_probe(
                cfg=None,
                condition=cond,
                index_df=index_df,
                windows_dir=windows_dir,
                scope="both",
                checkpoint_dir=ckpt_dir,
                device="cpu",
                K=-1,
            )
            all_results.extend(rows)
            log.info("[smoke] condition=%s: %d result rows", cond, len(rows))

        # Write output CSV
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_csv = reports_dir / "smoke_task2_results.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction="ignore")
            writer.writeheader()
            for row in all_results:
                writer.writerow({k: row.get(k, "") for k in _CSV_COLS})

        # Validate
        assert len(all_results) > 0, "No results"
        experiments   = {r["experiment"] for r in all_results}
        scopes        = {r["scope"] for r in all_results}
        metrics       = {r["metric"] for r in all_results}
        target_rooms  = {r["target_room"] for r in all_results}

        assert "probe_handcrafted" in experiments, f"Missing handcrafted: {experiments}"
        assert "probe_random" in experiments,      f"Missing random: {experiments}"
        assert "joint"    in scopes,               f"Missing joint scope: {scopes}"
        assert "per_room" in scopes,               f"Missing per_room scope: {scopes}"
        assert "macro_f1" in metrics,              f"Missing macro_f1: {metrics}"
        assert "balanced_accuracy" in metrics,     f"Missing balanced_accuracy: {metrics}"
        assert "joint" in target_rooms,            f"Missing joint room: {target_rooms}"
        assert "A" in target_rooms and "B" in target_rooms, f"Missing rooms: {target_rooms}"

        # Verify written CSV schema
        written_df = pd.read_csv(str(out_csv))
        assert list(written_df.columns) == _CSV_COLS, f"Wrong CSV cols: {written_df.columns.tolist()}"

        for r in all_results:
            v = float(r["value"])
            assert 0.0 <= v <= 1.0, f"Value out of [0,1]: {v}"
            assert int(r["K"]) == -1, f"K not -1: {r['K']}"

        print(f"[smoke] {len(all_results)} result rows  OK")
        print(f"[smoke] experiments: {sorted(experiments)}  OK")
        print(f"[smoke] scopes: {sorted(scopes)}  OK")
        print(f"[smoke] metrics: {sorted(metrics)}  OK")
        print(f"[smoke] target_rooms: {sorted(target_rooms)}  OK")
        print(f"[smoke] CSV format {_CSV_COLS}  OK")
        print("[smoke] task2_eval PASSED")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(rows: List[dict]) -> None:
    print("\n" + "=" * 80)
    print("Task 2 – Linear Probe Results")
    print("=" * 80)
    print(f"{'Experiment':<28} {'Scope':<10} {'Room':<8} {'K':>4}  {'Metric':<22} {'Value':>8}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['experiment']:<28} {r.get('scope', ''):<10} {r['target_room']:<8} "
            f"{r['K']:>4}  {r['metric']:<22} {float(r['value']):>8.4f}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = _parse_args()

    if args.smoke:
        _run_smoke(args)
        return

    from src.config import load_config
    from src.probe import ALL_CONDITIONS, run_probe

    cfg = load_config(args.config)

    # Resolve paths
    index_csv   = Path(cfg.paths.windows_index_csv)
    windows_dir = Path(cfg.paths.windows_dir)
    ckpt_dir    = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(cfg.paths.checkpoints_dir)
    reports_dir = Path(cfg.paths.reports_dir)
    out_csv     = Path(args.out) if args.out else reports_dir / "task2_probe_results.csv"

    if not index_csv.exists():
        log.error(
            "windows_index.csv not found: %s\nRun scripts/build_windows.py first.",
            index_csv,
        )
        sys.exit(1)

    import pandas as pd
    index_df = pd.read_csv(str(index_csv))
    log.info(
        "Index loaded: %d windows  dev=%d  val=%d",
        len(index_df),
        (index_df["split"] == "dev").sum(),
        (index_df["split"] == "val").sum(),
    )

    # Determine which conditions to run
    conditions = args.conditions if args.conditions else list(ALL_CONDITIONS)
    log.info("Conditions: %s", conditions)
    log.info("Scope:      %s", args.scope)

    all_results: List[dict] = []

    for cond in conditions:
        log.info("=" * 60)
        log.info("Condition: %s", cond)
        try:
            rows = run_probe(
                cfg=cfg,
                condition=cond,
                index_df=index_df,
                windows_dir=windows_dir,
                scope=args.scope,
                checkpoint_dir=ckpt_dir,
                device=args.device,
                K=-1,
            )
            all_results.extend(rows)
            log.info("Condition %s: %d rows", cond, len(rows))
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", cond, exc)
        except Exception as exc:
            log.error("Failed %s: %s", cond, exc, exc_info=True)

    # Write CSV
    reports_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in _CSV_COLS})

    log.info("Results written -> %s  (%d rows)", out_csv, len(all_results))
    _print_summary(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task 2 linear probe: evaluate SSL encoder representations "
            "across multiple conditions."
        )
    )
    p.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    p.add_argument(
        "--conditions", nargs="+", default=None,
        choices=["handcrafted", "random", "vicreg_aug", "vicreg_noaug", "masked"],
        help="Conditions to evaluate (default: all five).",
    )
    p.add_argument(
        "--scope", default="both",
        choices=["joint", "per_room", "both"],
        help=(
            "Probe training scope. 'joint' = one probe on all rooms, "
            "'per_room' = one probe per room, 'both' = run both (default: both)."
        ),
    )
    p.add_argument(
        "--checkpoint_dir", default=None,
        help="Directory with encoder .pt checkpoints (default: cfg.paths.checkpoints_dir).",
    )
    p.add_argument(
        "--out", default=None,
        help="Override output CSV path.",
    )
    p.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device (default: cpu).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Run smoke test on synthetic data (no real files needed).",
    )
    p.add_argument(
        "--stage", default="eval",
        choices=["pretrain", "eval", "both"],
        help="Pipeline stage (default: eval).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
