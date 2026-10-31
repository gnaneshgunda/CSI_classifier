"""
run_episode.py
==============
Run a single few-shot domain adaptation episode for Task 3.

Overview
--------
Given target_room, K, seed, and config:
1. Load cached windows and encoder (or feature extractor).
2. Apply configured normalization, routing, and alignment.
3. Build K-shot support set from target development captures (exclusive classes only).
4. Fit PrototypeHead with source prototypes and hierarchical target prototypes.
5. Predict target VALIDATION windows (with optional --capture_avg across windows of each capture).
6. Output results rows:
     experiment,target_room,K,metric,value
   and predictions:
     window_id,room,label,confidence

Usage
-----
    python run_episode.py \
        --target_room A \
        --K 5 \
        --seed 42 \
        --config configs/experiment.yaml \
        --out reports/episode_results.csv \
        --predictions_out reports/episode_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.align import EmbeddingAligner
from src.config import ExperimentConfig, load_config
from src.encoder import ResNet1DEncoder
from src.fewshot import (
    EXCLUSIVE_ACTIVITIES, ROOM_A_EXCLUSIVE, ROOM_B_EXCLUSIVE,
    OPPOSITE_ROOM,
    ROOM_A_ACTIVITIES,
    ROOM_B_ACTIVITIES,
    SHARED_ACTIVITIES,
    VALID_ACTIVITIES,
    PrototypeHead,
    sample_support,
)
from src.manifest import ManifestRow, read_manifest
from src.metrics import compute_all_metrics
from src.normalization import Normalizer
from src.routing import RoomRouter

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("run_episode")


# ---------------------------------------------------------------------------
# Helpers: Encoder Loading & Window Embedding Extraction
# ---------------------------------------------------------------------------

def load_encoder(
    cfg: ExperimentConfig,
    checkpoint_dir: Optional[Path] = None,
    condition: str = "vicreg_aug",
    device: str = "cpu",
) -> ResNet1DEncoder:
    """Load the trained encoder or initialize a random one if checkpoint not found."""
    emb_dim = 256
    if hasattr(cfg, "ssl") and hasattr(cfg.ssl, "embedding_dim"):
        emb_dim = cfg.ssl.embedding_dim

    enc = ResNet1DEncoder(
        embedding_dim=emb_dim,
        random_init=(condition == "random"),
        verbose=False,
    )

    if condition != "random":
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(cfg.paths.checkpoints_dir)
        candidate_names = [
            f"{condition}.pt",
            f"encoder_{condition}.pt",
            "vicreg_aug.pt",
            "best_encoder.pt",
        ]
        loaded = False
        for name in candidate_names:
            ckpt_path = ckpt_dir / name
            if ckpt_path.exists():
                try:
                    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    if isinstance(state, dict) and "encoder" in state:
                        state = state["encoder"]
                    enc.load_state_dict(state, strict=False)
                    log.info("Loaded encoder checkpoint from %s", ckpt_path)
                    loaded = True
                    break
                except Exception as exc:
                    log.warning("Failed to load checkpoint %s: %s", ckpt_path, exc)
        if not loaded:
            log.warning(
                "No checkpoint found in %s for condition '%s'. Using randomly initialized encoder.",
                ckpt_dir,
                condition,
            )

    enc.to(torch.device(device))
    enc.eval()
    return enc


def load_normalizer(cfg: ExperimentConfig, dev_df: Optional[pd.DataFrame] = None) -> Normalizer:
    """Load fitted Normalizer or fit on dev_df if stats file does not exist."""
    norm = Normalizer(cfg, log1p=True)
    stats_path = Path(cfg.paths.derived_dir) / "norm_stats.npz"
    if stats_path.exists():
        try:
            norm.load()
            return norm
        except Exception:
            pass

    if dev_df is not None and len(dev_df) > 0:
        try:
            norm.fit(dev_df)
        except Exception as exc:
            log.warning("Could not fit normalizer on dev_df: %s", exc)
    return norm


def load_router(cfg: ExperimentConfig) -> RoomRouter:
    """Load room router according to cfg.routing.mode."""
    mode = cfg.routing.mode
    if mode in ("oracle", "none"):
        return RoomRouter(None, cfg)

    # Try loading pre-trained room classifier
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    clf = None
    # room_model.pkl is the canonical name from src/room_model.py RoomModel.save()
    for clf_name in ("room_model.pkl", "room_model_lr.pkl", "room_model_mlp.pkl"):
        p = ckpt_dir / clf_name
        if p.exists():
            try:
                import joblib
                payload = joblib.load(str(p))
                # room_model.pkl stores a dict payload; extract the best clf
                if isinstance(payload, dict) and "clfs" in payload:
                    from src.room_model import RoomModel
                    rm = RoomModel.__new__(RoomModel)
                    rm._clfs     = payload["clfs"]
                    rm._scalers  = payload["scalers"]
                    rm.best_name = payload.get("best_name", list(payload["clfs"].keys())[0])
                    rm._is_fit   = True
                    rm.rooms     = payload.get("rooms", ["A", "B"])
                    rm.classes_  = __import__("numpy").array(rm.rooms)
                    from src.normalization import Normalizer
                    rm._extractor = __import__("src.room_model", fromlist=["RoomFeatureExtractor"]).RoomFeatureExtractor(
                        log1p=True,
                        norm_stats_path=__import__("pathlib").Path(cfg.paths.derived_dir) / "norm_stats.npz"
                        if __import__("pathlib").Path(cfg.paths.derived_dir + "/norm_stats.npz").exists() else None
                    )
                    clf = rm
                else:
                    clf = payload
                log.info("Loaded room classifier from %s", p)
                break
            except Exception as exc:
                log.warning("Failed to load room classifier %s: %s", p, exc)

    if clf is None:
        log.warning("No room classifier checkpoint found; falling back to oracle mode for routing.")
        from dataclasses import replace
        new_routing = replace(cfg.routing, mode="oracle")
        return RoomRouter(None, replace(cfg, routing=new_routing))

    return RoomRouter(clf, cfg)


def extract_embeddings_for_df(
    df: pd.DataFrame,
    windows_dir: Path,
    encoder: ResNet1DEncoder,
    normalizer: Normalizer,
    router: RoomRouter,
    device: str = "cpu",
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract normalized and encoded embeddings for a DataFrame of windows.

    Returns:
        (embeddings, labels, capture_ids, window_ids)
    """
    if df.empty:
        dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else 256
        return (
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )

    X_list: List[np.ndarray] = []
    labels_list: List[str] = []
    cids_list: List[str] = []
    wids_list: List[str] = []

    batch_wins: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, str, str]] = []
    dev_obj = torch.device(device)

    def _flush_batch():
        if not batch_wins:
            return
        t = torch.stack(batch_wins, dim=0).to(dev_obj)
        with torch.no_grad():
            embs = encoder.encode(t).cpu().numpy()
        X_list.append(embs)
        for wid, cid, act in batch_meta:
            wids_list.append(wid)
            cids_list.append(cid)
            labels_list.append(act)
        batch_wins.clear()
        batch_meta.clear()

    for _, row in df.iterrows():
        split = str(row["split"])
        wid = str(row["window_id"])
        cid = str(row["capture_id"])
        act = str(row["activity"])
        room = str(row["room"]).upper()

        wpath = windows_dir / split / f"{wid}.npy"
        if not wpath.exists():
            continue

        try:
            win = np.load(str(wpath)).astype(np.float32)
        except Exception:
            continue

        # Route and normalize
        try:
            route_res = router.route(win.mean(axis=0), true_room=room)
            norm_static, norm_dynamic = normalizer.transform(
                win, room=route_res.room, room_probs=route_res.room_probs
            )
            norm_win = (norm_dynamic + norm_static[np.newaxis, :]).astype(np.float32)
        except Exception:
            # Fallback to direct window standardization if normalizer lacks stats
            mu = win.mean(axis=0, keepdims=True)
            std = win.std(axis=0, keepdims=True) + 1e-6
            norm_win = ((win - mu) / std).astype(np.float32)

        batch_wins.append(torch.from_numpy(norm_win))
        batch_meta.append((wid, cid, act))

        if len(batch_wins) >= batch_size:
            _flush_batch()

    _flush_batch()

    if not X_list:
        dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else 256
        return (
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )

    X = np.vstack(X_list).astype(np.float32)
    return (
        X,
        np.array(labels_list, dtype=object),
        np.array(cids_list, dtype=object),
        np.array(wids_list, dtype=object),
    )


# ---------------------------------------------------------------------------
# Core Episode Execution
# ---------------------------------------------------------------------------

def run_episode(
    cfg: ExperimentConfig,
    target_room: str,
    K: int,
    seed: int,
    out: Optional[Path | str] = None,
    predictions_out: Optional[Path | str] = None,
    capture_avg: bool = False,
    condition: str = "vicreg_aug",
    checkpoint_dir: Optional[Path | str] = None,
    device: str = "cpu",
    cached_data: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Execute a single few-shot evaluation episode.

    Returns:
        (results_rows, predictions_df)
    """
    # Set seeds for deterministic execution
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_room = str(target_room).strip().upper()
    if target_room not in ("A", "B"):
        raise ValueError(f"Invalid target_room: {target_room!r}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    source_room = OPPOSITE_ROOM[target_room]

    windows_dir = Path(cfg.paths.windows_dir)
    manifest_csv = Path(cfg.paths.manifest_csv)
    index_csv = Path(cfg.paths.windows_index_csv)

    if cached_data is not None:
        # Use pre-extracted embeddings and metadata for ultra-fast sweeps
        X_src_dev = cached_data["X_src_dev"]
        y_src_dev = cached_data["y_src_dev"]
        cids_src_dev = cached_data["cids_src_dev"]

        X_tgt_dev = cached_data["X_tgt_dev"]
        y_tgt_dev = cached_data["y_tgt_dev"]
        cids_tgt_dev = cached_data["cids_tgt_dev"]

        X_tgt_val = cached_data["X_tgt_val"]
        y_tgt_val = cached_data["y_tgt_val"]
        cids_tgt_val = cached_data["cids_tgt_val"]
        wids_tgt_val = cached_data["wids_tgt_val"]

        manifest_rows = cached_data["manifest_rows"]
    else:
        index_df = pd.read_csv(str(index_csv))
        manifest_rows = read_manifest(manifest_csv)

        encoder = load_encoder(cfg, checkpoint_dir=checkpoint_dir, condition=condition, device=device)
        dev_df = index_df[index_df["split"] == "dev"].copy()
        normalizer = load_normalizer(cfg, dev_df=dev_df)
        router = load_router(cfg)

        # Source dev
        src_dev_df = index_df[(index_df["room"] == source_room) & (index_df["split"] == "dev")]
        X_src_dev, y_src_dev, cids_src_dev, _ = extract_embeddings_for_df(
            src_dev_df, windows_dir, encoder, normalizer, router, device=device
        )

        # Target dev
        tgt_dev_df = index_df[(index_df["room"] == target_room) & (index_df["split"] == "dev")]
        X_tgt_dev, y_tgt_dev, cids_tgt_dev, _ = extract_embeddings_for_df(
            tgt_dev_df, windows_dir, encoder, normalizer, router, device=device
        )

        # Target val
        tgt_val_df = index_df[(index_df["room"] == target_room) & (index_df["split"] == "val")]
        X_tgt_val, y_tgt_val, cids_tgt_val, wids_tgt_val = extract_embeddings_for_df(
            tgt_val_df, windows_dir, encoder, normalizer, router, device=device
        )

        # Domain alignment if configured
        if getattr(cfg, "align", None) is not None and getattr(cfg.align, "method", "none") != "none":
            aligner = EmbeddingAligner(cfg)
            try:
                aligner.load()
            except Exception:
                all_dev_embs = np.vstack([X_src_dev, X_tgt_dev])
                all_dev_rooms = [source_room] * len(X_src_dev) + [target_room] * len(X_tgt_dev)
                aligner.fit(all_dev_embs, all_dev_rooms)

            X_src_dev = aligner.transform(X_src_dev, [source_room] * len(X_src_dev))
            X_tgt_dev = aligner.transform(X_tgt_dev, [target_room] * len(X_tgt_dev))
            X_tgt_val = aligner.transform(X_tgt_val, [target_room] * len(X_tgt_val))

    # 1. Sample K-shot support set from target development captures (exclusive activities only)
    support = sample_support(manifest_rows, target_room=target_room, K=K, seed=seed)
    sampled_cids = set(support.capture_ids)

    # Filter target dev windows to only those from sampled support captures
    support_mask = np.isin(cids_tgt_dev, list(sampled_cids))
    X_support = X_tgt_dev[support_mask]
    y_support = y_tgt_dev[support_mask]
    cids_support = cids_tgt_dev[support_mask]

    # 2. Fit PrototypeHead
    head = PrototypeHead(target_room=target_room, cfg=cfg)
    head.fit(
        source_embeddings=X_src_dev,
        source_labels=y_src_dev,
        target_support_embeddings=X_support,
        target_support_labels=y_support,
        target_support_capture_ids=cids_support,
    )

    # 3. Predict on target validation windows
    if len(X_tgt_val) == 0:
        log.warning("No validation windows found for target room %s", target_room)
        preds_df = pd.DataFrame(columns=["window_id", "room", "label", "confidence"])
        results_rows = []
        return results_rows, preds_df

    if capture_avg:
        # Average predicted probabilities across all sliding windows of each capture
        unique_cids = np.unique(cids_tgt_val)
        cid_predictions: Dict[str, Tuple[str, float]] = {}

        for cid in unique_cids:
            mask = cids_tgt_val == cid
            probs_w = head.predict_proba(X_tgt_val[mask])  # (W, C)
            mean_probs = np.mean(probs_w, axis=0)          # (C,)
            best_idx = int(np.argmax(mean_probs))
            best_label = head.classes_[best_idx]
            best_conf = float(mean_probs[best_idx])
            cid_predictions[str(cid)] = (best_label, best_conf)

        y_pred_list: List[str] = []
        conf_list: List[float] = []
        for cid in cids_tgt_val:
            bl, cf = cid_predictions[str(cid)]
            y_pred_list.append(bl)
            conf_list.append(cf)

        y_pred = np.array(y_pred_list, dtype=object)
        confs = np.array(conf_list, dtype=np.float32)
    else:
        probs = head.predict_proba(X_tgt_val)
        best_indices = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1).astype(np.float32)
        y_pred = np.array([head.classes_[i] for i in best_indices], dtype=object)

    # Build predictions DataFrame: (window_id, room, label, confidence)
    preds_df = pd.DataFrame({
        "window_id": wids_tgt_val,
        "room": [target_room] * len(wids_tgt_val),
        "label": y_pred,
        "confidence": confs,
    })

    # 4. Compute benchmark metrics
    metrics = compute_all_metrics(
        y_true=y_tgt_val,
        y_pred=y_pred,
        target_room=target_room,
        labels=VALID_ACTIVITIES[target_room],
    )

    exp_name = f"fewshot_{source_room}to{target_room}"
    if capture_avg:
        exp_name += "_capavg"

    results_rows: List[Dict[str, Any]] = []
    for metric_name, val in metrics.items():
        results_rows.append({
            "experiment": exp_name,
            "target_room": target_room,
            "K": K,
            "metric": metric_name,
            "value": f"{val:.6f}",
        })

    log.info(
        "[%s] K=%d seed=%d -> macro_f1=%.4f bal_acc=%.4f excl_f1=%.4f shared_f1=%.4f",
        exp_name,
        K,
        seed,
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["target_exclusive_macro_f1"],
        metrics["shared_class_macro_f1"],
    )

    # 5. Write outputs if paths specified
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_df = pd.DataFrame(results_rows)
        res_df.to_csv(str(out_path), index=False)
        log.info("Wrote episode results to %s", out_path)

    if predictions_out is not None:
        p_out = Path(predictions_out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(str(p_out), index=False)
        log.info("Wrote episode predictions to %s", p_out)

    return results_rows, preds_df


# ---------------------------------------------------------------------------
# CLI & Smoke Testing
# ---------------------------------------------------------------------------

def _run_smoke(args: argparse.Namespace) -> None:
    """Run end-to-end smoke test on synthetic data without real files."""
    import tempfile
    from src.config import ActivitiesConfig, FewshotConfig, PathsConfig, RoomsConfig, SeedsConfig, SplitsConfig

    print("[smoke] Running run_episode smoke test ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        windows_dir = tmp / "windows"
        reports_dir = tmp / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic windows
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
                    # Add to manifest
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
                    # Create 2 windows for dev, 1 for val
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

        # Mock config
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
            seeds=SeedsConfig(global_seed=42, fewshot_seeds=[0]),
            window=None,
            cleaning=None,
            normalization=None,
            ssl=None,
            augment=None,
            align=None,
            routing=argparse.Namespace(mode="oracle", confidence_threshold=0.6),
            fewshot=FewshotConfig(
                K_list=[1],
                n_seeds=1,
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

        res_path = reports_dir / "smoke_results.csv"
        pred_path = reports_dir / "smoke_predictions.csv"

        rows, preds = run_episode(
            cfg=cfg,
            target_room="A",
            K=1,
            seed=42,
            out=res_path,
            predictions_out=pred_path,
            capture_avg=True,
            condition="random",
        )

        assert res_path.exists(), "Results CSV not written!"
        assert pred_path.exists(), "Predictions CSV not written!"
        assert len(rows) == 4, f"Expected 4 metrics, got {len(rows)}"
        assert not preds.empty, "Predictions DataFrame is empty!"
        print("[smoke] run_episode smoke test passed successfully âœ“")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single few-shot adaptation episode (Task 3).")
    parser.add_argument("--target_room", type=str, default="A", help="Target room (A or B)")
    parser.add_argument("--K", type=int, default=5, help="Shots per target-exclusive activity")
    parser.add_argument("--seed", type=int, default=42, help="Seed for support sampling")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config file")
    parser.add_argument("--out", "--output", type=str, default=None, help="Output path for results rows")
    parser.add_argument("--predictions_out", "--pred_out", type=str, default=None, help="Output path for predictions CSV")
    parser.add_argument("--capture_avg", action="store_true", help="Average probabilities over windows per capture")
    parser.add_argument("--condition", type=str, default="vicreg_aug", help="Encoder condition")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--episode_dir", type=str, default=None, help="Optional challenge episode directory")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on synthetic data")

    args = parser.parse_args()

    if args.smoke:
        _run_smoke(args)
        return

    cfg = load_config(args.config)

    # Derive default output paths if not explicitly given
    out_path = args.out
    if out_path is None:
        out_path = Path(cfg.paths.reports_dir) / f"episode_results_{args.target_room}_K{args.K}.csv"

    pred_path = args.predictions_out
    if pred_path is None:
        pred_path = Path(cfg.paths.reports_dir) / "episode_predictions.csv"

    run_episode(
        cfg=cfg,
        target_room=args.target_room,
        K=args.K,
        seed=args.seed,
        out=out_path,
        predictions_out=pred_path,
        capture_avg=args.capture_avg,
        condition=args.condition,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()