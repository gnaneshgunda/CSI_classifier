"""
scripts/sweep.py
================
Task 3 – Cross-room few-shot adaptation sweep.

Sweeps K in [1, 5, 10, 20] across 10 random seeds for both transfer directions:
- A -> B (pre-trained on Room A, adapted to Room B)
- B -> A (pre-trained on Room B, adapted to Room A)

Results are aggregated and written to reports/task3_sweep.csv.

Usage
-----
    python scripts/sweep.py --config configs/experiment.yaml
    python scripts/sweep.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_episode import (
    extract_embeddings_for_df,
    load_encoder,
    load_normalizer,
    load_router,
    run_episode,
)
from src.align import EmbeddingAligner
from src.config import ExperimentConfig, load_config
from src.fewshot import (
    EXCLUSIVE_ACTIVITIES,
    OPPOSITE_ROOM,
    ROOM_A_ACTIVITIES,
    ROOM_A_EXCLUSIVE,
    ROOM_B_ACTIVITIES,
    ROOM_B_EXCLUSIVE,
    SHARED_ACTIVITIES,
    VALID_ACTIVITIES,
)
from src.manifest import ManifestRow, read_manifest

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("sweep")


def run_sweep(
    cfg: ExperimentConfig,
    K_list: Sequence[int],
    seeds: Sequence[int],
    directions: Sequence[str] = ("A->B", "B->A"),
    out_csv: Optional[Path | str] = None,
    capture_avg: bool = False,
    condition: str = "vicreg_aug",
    checkpoint_dir: Optional[Path | str] = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Execute the full few-shot sweep across K values, seeds, and directions."""
    windows_dir = Path(cfg.paths.windows_dir)
    manifest_csv = Path(cfg.paths.manifest_csv)
    index_csv = Path(cfg.paths.windows_index_csv)

    valid_directions = {"A->B", "B->A"}
    for d in directions:
        if d not in valid_directions:
            raise ValueError(f"Invalid direction: {d!r}. Allowed directions: ['A->B', 'B->A']")

    if not K_list:
        raise ValueError("K_list cannot be empty")
    for k in K_list:
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")

    if not seeds:
        raise ValueError("seeds cannot be empty")

    index_df = pd.read_csv(str(index_csv))
    manifest_rows = read_manifest(manifest_csv)

    log.info("Loading encoder and extracting window embeddings ...")
    encoder = load_encoder(cfg, checkpoint_dir=checkpoint_dir, condition=condition, device=device)
    dev_df = index_df[index_df["split"] == "dev"].copy()
    normalizer = load_normalizer(cfg, dev_df=dev_df)
    router = load_router(cfg)

    # Pre-extract all embeddings once
    # Room A
    A_dev_df = index_df[(index_df["room"] == "A") & (index_df["split"] == "dev")]
    X_A_dev, y_A_dev, cids_A_dev, _ = extract_embeddings_for_df(
        A_dev_df, windows_dir, encoder, normalizer, router, device=device
    )

    A_val_df = index_df[(index_df["room"] == "A") & (index_df["split"] == "val")]
    X_A_val, y_A_val, cids_A_val, wids_A_val = extract_embeddings_for_df(
        A_val_df, windows_dir, encoder, normalizer, router, device=device
    )

    # Room B
    B_dev_df = index_df[(index_df["room"] == "B") & (index_df["split"] == "dev")]
    X_B_dev, y_B_dev, cids_B_dev, _ = extract_embeddings_for_df(
        B_dev_df, windows_dir, encoder, normalizer, router, device=device
    )

    B_val_df = index_df[(index_df["room"] == "B") & (index_df["split"] == "val")]
    X_B_val, y_B_val, cids_B_val, wids_B_val = extract_embeddings_for_df(
        B_val_df, windows_dir, encoder, normalizer, router, device=device
    )

    # Domain alignment if configured
    if hasattr(cfg, "align") and cfg.align is not None and getattr(cfg.align, "method", "none") != "none":
        aligner = EmbeddingAligner(cfg)
        try:
            aligner.load()
        except Exception:
            all_dev_embs = np.vstack([X_A_dev, X_B_dev])
            all_dev_rooms = ["A"] * len(X_A_dev) + ["B"] * len(X_B_dev)
            aligner.fit(all_dev_embs, all_dev_rooms)

        X_A_dev = aligner.transform(X_A_dev, ["A"] * len(X_A_dev))
        X_A_val = aligner.transform(X_A_val, ["A"] * len(X_A_val))
        X_B_dev = aligner.transform(X_B_dev, ["B"] * len(X_B_dev))
        X_B_val = aligner.transform(X_B_val, ["B"] * len(X_B_val))

    # Package cache for directions
    cached_transfer: Dict[str, Dict[str, Any]] = {
        "A->B": {
            "X_src_dev": X_A_dev,
            "y_src_dev": y_A_dev,
            "cids_src_dev": cids_A_dev,
            "X_tgt_dev": X_B_dev,
            "y_tgt_dev": y_B_dev,
            "cids_tgt_dev": cids_B_dev,
            "X_tgt_val": X_B_val,
            "y_tgt_val": y_B_val,
            "cids_tgt_val": cids_B_val,
            "wids_tgt_val": wids_B_val,
            "manifest_rows": manifest_rows,
        },
        "B->A": {
            "X_src_dev": X_B_dev,
            "y_src_dev": y_B_dev,
            "cids_src_dev": cids_B_dev,
            "X_tgt_dev": X_A_dev,
            "y_tgt_dev": y_A_dev,
            "cids_tgt_dev": cids_A_dev,
            "X_tgt_val": X_A_val,
            "y_tgt_val": y_A_val,
            "cids_tgt_val": cids_A_val,
            "wids_tgt_val": wids_A_val,
            "manifest_rows": manifest_rows,
        },
    }

    all_rows: List[Dict[str, Any]] = []
    total_episodes = len(directions) * len(K_list) * len(seeds)
    ep_count = 0

    log.info(
        "Starting sweep: directions=%s K_list=%s n_seeds=%d (total %d episodes)",
        list(directions),
        list(K_list),
        len(seeds),
        total_episodes,
    )

    for direction in directions:
        target_room = "B" if direction == "A->B" else "A"
        cached = cached_transfer[direction]

        for K in K_list:
            for seed in seeds:
                ep_count += 1
                rows, _ = run_episode(
                    cfg=cfg,
                    target_room=target_room,
                    K=K,
                    seed=seed,
                    capture_avg=capture_avg,
                    condition=condition,
                    checkpoint_dir=checkpoint_dir,
                    device=device,
                    cached_data=cached,
                )
                for r in rows:
                    all_rows.append({
                        "experiment": r["experiment"],
                        "direction": direction,
                        "target_room": target_room,
                        "K": K,
                        "seed": seed,
                        "metric": r["metric"],
                        "value": r["value"],
                    })

    df_results = pd.DataFrame(all_rows)

    if out_csv is not None:
        out_p = Path(out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Write clean CSV matching required columns
        df_results.to_csv(str(out_p), index=False)
        log.info("Saved Task 3 sweep results to %s", out_p)

    # Print summary table
    print("\n" + "=" * 80)
    print("TASK 3 FEW-SHOT SWEEP SUMMARY (Mean ± Std over seeds)")
    print("=" * 80)
    val_float = df_results["value"].astype(float)
    summary_df = df_results.copy()
    summary_df["val_float"] = val_float
    summary = (
        summary_df.groupby(["direction", "target_room", "K", "metric"])["val_float"]
        .agg(["mean", "std"])
        .reset_index()
    )
    for _, row in summary.iterrows():
        std_val = row["std"] if not np.isnan(row["std"]) else 0.0
        print(
            f"{row['direction']:<6} room={row['target_room']}  K={row['K']:<2}  "
            f"{row['metric']:<25} {row['mean']:.4f} ± {std_val:.4f}"
        )
    print("=" * 80 + "\n")

    return df_results


def _run_smoke(args: argparse.Namespace) -> None:
    """Run fast smoke test on synthetic data."""
    import tempfile
    from src.config import ActivitiesConfig, FewshotConfig, PathsConfig, RoomsConfig, SeedsConfig, SplitsConfig

    print("[smoke] Running sweep.py smoke test ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        windows_dir = tmp / "windows"
        reports_dir = tmp / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        T, D = 150, 256
        index_rows: List[dict] = []
        manifest_rows: List[ManifestRow] = []

        activities_A = list(ROOM_A_ACTIVITIES)
        activities_B = list(ROOM_B_ACTIVITIES)

        for room, acts in [("A", activities_A), ("B", activities_B)]:
            for act in acts:
                for c in range(4):
                    cid = f"{room}__{act}__cap{c:03d}"
                    split = "dev" if c < 3 else "val"
                    manifest_rows.append(
                        ManifestRow(
                            capture_id=cid,
                            room=room,
                            activity=act,
                            split=split,
                            source_file="synth",
                            n_packets_raw=T,
                            n_packets_clean=T,
                            intermediate_path="synth",
                        )
                    )
                    n_wins = 2 if split == "dev" else 1
                    for w in range(n_wins):
                        wid = f"{cid}__{w:05d}"
                        out_dir = windows_dir / split
                        out_dir.mkdir(parents=True, exist_ok=True)
                        np.save(str(out_dir / f"{wid}.npy"), np.random.randn(T, D).astype(np.float32))
                        index_rows.append({
                            "window_id": wid,
                            "capture_id": cid,
                            "room": room,
                            "activity": act,
                            "split": split,
                            "start": 0,
                            "config_hash": "smoke",
                        })

        index_csv = tmp / "windows_index.csv"
        pd.DataFrame(index_rows).to_csv(str(index_csv), index=False)

        manifest_csv = tmp / "manifest.csv"
        from src.manifest import write_manifest
        write_manifest(manifest_rows, manifest_csv)

        cfg = ExperimentConfig(
            paths=PathsConfig(
                room_a_pcap_dir=tmp,
                room_a_val_dir=tmp,
                room_b_csv_dir=tmp,
                derived_dir=tmp,
                intermediate_dir=tmp,
                windows_dir=windows_dir,
                manifest_csv=manifest_csv,
                windows_index_csv=index_csv,
                checkpoints_dir=tmp,
                reports_dir=reports_dir,
            ),
            seeds=SeedsConfig(global_seed=42, fewshot_seeds=[0, 1]),
            window=None,
            cleaning=None,
            normalization=None,
            ssl=None,
            augment=None,
            align=None,
            routing=argparse.Namespace(mode="oracle", confidence_threshold=0.6),
            fewshot=FewshotConfig(
                K_list=[1, 2],
                n_seeds=2,
                distance_metric="cosine",
                shrinkage="ledoit_wolf",
                classifier="ncm",
            ),
            activities=ActivitiesConfig(
                room_a=list(ROOM_A_ACTIVITIES),
                room_b=list(ROOM_B_ACTIVITIES),
                shared=list(SHARED_ACTIVITIES),
                room_a_exclusive=list(ROOM_A_EXCLUSIVE),
                room_b_exclusive=list(ROOM_B_EXCLUSIVE),
                all=list(ROOM_A_ACTIVITIES + ROOM_B_EXCLUSIVE),
            ),
            rooms=RoomsConfig(all=["A", "B"]),
            splits=SplitsConfig(all=["dev", "val"], val_seed=42, val_captures_per_class=1),
        )

        out_csv = reports_dir / "smoke_sweep.csv"
        df_res = run_sweep(
            cfg=cfg,
            K_list=[1],
            seeds=[0, 1],
            directions=["A->B", "B->A"],
            out_csv=out_csv,
            capture_avg=True,
            condition="random",
        )

        assert out_csv.exists(), "Sweep output CSV was not created!"
        assert not df_res.empty, "Sweep results DataFrame is empty!"
        print("[smoke] sweep smoke test passed successfully ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep few-shot K and seeds across both transfer directions.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--out", "--output", type=str, default=None, help="Output CSV path (default: reports/task3_sweep.csv)")
    parser.add_argument("--K_list", type=int, nargs="+", default=None, help="K values to sweep (e.g. 1 5 10 20)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to sweep (e.g. 0 1 2 3 ...)")
    parser.add_argument("--directions", type=str, nargs="+", default=["A->B", "B->A"], help="Directions to evaluate")
    parser.add_argument("--capture_avg", action="store_true", help="Average probabilities across windows per capture")
    parser.add_argument("--condition", type=str, default="vicreg_aug", help="Encoder condition (default: vicreg_aug)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoints directory")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on synthetic data")

    args = parser.parse_args()

    if args.smoke:
        _run_smoke(args)
        return

    cfg = load_config(args.config)

    K_list = args.K_list
    if K_list is None:
        K_list = cfg.fewshot.K_list if hasattr(cfg, "fewshot") else [1, 5, 10, 20]

    seeds = args.seeds
    if seeds is None:
        seeds = cfg.seeds.fewshot_seeds if hasattr(cfg, "seeds") else list(range(10))

    out_csv = args.out
    if out_csv is None:
        out_csv = Path(cfg.paths.reports_dir) / "task3_sweep.csv"

    run_sweep(
        cfg=cfg,
        K_list=K_list,
        seeds=seeds,
        directions=args.directions,
        out_csv=out_csv,
        capture_avg=args.capture_avg,
        condition=args.condition,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()