"""
tests/test_run_episode.py
=========================
Unit tests for run_episode.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from run_episode import _run_smoke, main, run_episode
from src.config import (
    ActivitiesConfig,
    ExperimentConfig,
    FewshotConfig,
    PathsConfig,
    RoomsConfig,
    SeedsConfig,
    SplitsConfig,
)
from src.fewshot import (
    ROOM_A_ACTIVITIES,
    ROOM_A_EXCLUSIVE,
    ROOM_B_ACTIVITIES,
    ROOM_B_EXCLUSIVE,
    SHARED_ACTIVITIES,
)
from src.manifest import ManifestRow, write_manifest


def _setup_synthetic_env(tmp: Path) -> ExperimentConfig:
    """Helper creating synthetic windows and manifest for testing."""
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
                # 3 windows for dev, 2 windows for val (to test capture_avg)
                n_wins = 3 if split == "dev" else 2
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
                        "start": w * 50,
                        "config_hash": "synth",
                    })

    index_csv = tmp / "windows_index.csv"
    pd.DataFrame(index_rows).to_csv(str(index_csv), index=False)

    manifest_csv = tmp / "manifest.csv"
    write_manifest(manifest_rows, manifest_csv)

    import argparse
    return ExperimentConfig(
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


class TestRunEpisode:
    def test_run_episode_outputs_and_schema(self, tmp_path: Path) -> None:
        """Verify results rows and predictions schema match specification."""
        cfg = _setup_synthetic_env(tmp_path)
        out_res = tmp_path / "results.csv"
        out_pred = tmp_path / "preds.csv"

        results_rows, preds_df = run_episode(
            cfg=cfg,
            target_room="A",
            K=1,
            seed=42,
            out=out_res,
            predictions_out=out_pred,
            capture_avg=False,
            condition="random",
        )

        # 1. Results rows checks
        assert out_res.exists()
        df_res = pd.read_csv(out_res)
        assert list(df_res.columns) == ["experiment", "target_room", "K", "metric", "value"]
        assert (df_res["target_room"] == "A").all()
        assert (df_res["K"] == 1).all()

        metrics_found = set(df_res["metric"])
        assert "macro_f1" in metrics_found
        assert "balanced_accuracy" in metrics_found
        assert "target_exclusive_macro_f1" in metrics_found
        assert "shared_class_macro_f1" in metrics_found

        # 2. Predictions checks
        assert out_pred.exists()
        df_pred = pd.read_csv(out_pred)
        assert list(df_pred.columns) == ["window_id", "room", "label", "confidence"]
        assert (df_pred["room"] == "A").all()
        assert df_pred["label"].isin(ROOM_A_ACTIVITIES).all()
        assert (df_pred["confidence"] >= 0.0).all()
        assert (df_pred["confidence"] <= 1.0).all()

    def test_run_episode_capture_avg(self, tmp_path: Path) -> None:
        """Verify capture_avg assigns identical capture-level predictions to all windows of a capture."""
        cfg = _setup_synthetic_env(tmp_path)
        _, preds_df = run_episode(
            cfg=cfg,
            target_room="B",
            K=1,
            seed=42,
            capture_avg=True,
            condition="random",
        )

        assert not preds_df.empty
        assert (preds_df["room"] == "B").all()
        assert preds_df["label"].isin(ROOM_B_ACTIVITIES).all()

        # Each validation capture has 2 windows in synthetic setup.
        # Verify both windows of each capture have identical label and confidence.
        preds_df["cid"] = preds_df["window_id"].apply(lambda x: x.rsplit("__", 1)[0])
        for cid, grp in preds_df.groupby("cid"):
            assert len(grp["label"].unique()) == 1, f"Capture {cid} windows have inconsistent labels!"
            assert len(grp["confidence"].unique()) == 1, f"Capture {cid} windows have inconsistent confidences!"

    def test_run_episode_validation_only(self, tmp_path: Path) -> None:
        """Verify that ONLY target validation windows are evaluated and predicted."""
        cfg = _setup_synthetic_env(tmp_path)
        index_df = pd.read_csv(cfg.paths.windows_index_csv)

        # Target room A validation window IDs
        val_wids_A = set(
            index_df[(index_df["room"] == "A") & (index_df["split"] == "val")]["window_id"]
        )
        dev_wids_A = set(
            index_df[(index_df["room"] == "A") & (index_df["split"] == "dev")]["window_id"]
        )

        _, preds_df = run_episode(
            cfg=cfg,
            target_room="A",
            K=1,
            seed=42,
            condition="random",
        )

        pred_wids = set(preds_df["window_id"])
        # Must match validation windows exactly
        assert pred_wids == val_wids_A
        # Must NEVER contain development windows
        assert pred_wids.isdisjoint(dev_wids_A)

    def test_run_episode_deterministic(self, tmp_path: Path) -> None:
        """Verify run_episode is deterministic when called with the same seed."""
        cfg = _setup_synthetic_env(tmp_path)

        rows1, preds1 = run_episode(cfg=cfg, target_room="A", K=1, seed=42, condition="random")
        rows2, preds2 = run_episode(cfg=cfg, target_room="A", K=1, seed=42, condition="random")

        df1 = pd.DataFrame(rows1)
        df2 = pd.DataFrame(rows2)
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(preds1, preds2)

    def test_run_episode_invalid_inputs(self, tmp_path: Path) -> None:
        """Verify that invalid target_room or K raises ValueError."""
        cfg = _setup_synthetic_env(tmp_path)

        # Invalid room
        with pytest.raises(ValueError, match="Invalid target_room"):
            run_episode(cfg=cfg, target_room="C", K=1, seed=42)

        # Non-positive K
        with pytest.raises(ValueError, match="K must be positive"):
            run_episode(cfg=cfg, target_room="A", K=0, seed=42)

        with pytest.raises(ValueError, match="K must be positive"):
            run_episode(cfg=cfg, target_room="A", K=-5, seed=42)

    def test_run_episode_cli_smoke(self) -> None:
        """Verify run_episode smoke test runs and succeeds."""
        import argparse
        _run_smoke(argparse.Namespace())
