"""
tests/test_sweep.py
===================
Unit tests for scripts/sweep.py.

Tests cover:
- Sweeping multiple K values (1, 5, etc.)
- Both room transfer directions (A->B and B->A)
- Seed coverage (multiple seeds and 10 seeds)
- Correct episode invocation (mock inspection)
- Result aggregation and output CSV schema
- Capture averaging behavior
- Invalid input validation
- CLI smoke test execution
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from run_episode import load_encoder
from scripts.sweep import _run_smoke, main, run_sweep
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
    """Helper creating minimal synthetic windows and manifest for sweep testing."""
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
            # 4 captures per activity: 3 dev, 1 val
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
                        "start": w * 50,
                        "config_hash": "synth",
                    })

    index_csv = tmp / "windows_index.csv"
    pd.DataFrame(index_rows).to_csv(str(index_csv), index=False)

    manifest_csv = tmp / "manifest.csv"
    write_manifest(manifest_rows, manifest_csv)

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
        seeds=SeedsConfig(global_seed=42, fewshot_seeds=list(range(10))),
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


class TestSweep:
    def test_sweep_k_values(self, tmp_path: Path) -> None:
        """Verify sweep correctly evaluates multiple K values."""
        cfg = _setup_synthetic_env(tmp_path)
        K_list = [1, 2]
        seeds = [0]
        directions = ["A->B"]

        df = run_sweep(
            cfg=cfg,
            K_list=K_list,
            seeds=seeds,
            directions=directions,
            condition="random",
        )

        assert not df.empty
        assert set(df["K"].unique()) == {1, 2}
        # 1 direction * 2 K * 1 seed * 4 metrics = 8 rows
        assert len(df) == 8

    def test_sweep_both_room_directions(self, tmp_path: Path) -> None:
        """Verify sweep runs both transfer directions A->B and B->A with proper target rooms."""
        cfg = _setup_synthetic_env(tmp_path)
        K_list = [1]
        seeds = [0]
        directions = ["A->B", "B->A"]

        df = run_sweep(
            cfg=cfg,
            K_list=K_list,
            seeds=seeds,
            directions=directions,
            condition="random",
        )

        assert set(df["direction"].unique()) == {"A->B", "B->A"}

        # Direction A->B adapts to room B
        ab_rows = df[df["direction"] == "A->B"]
        assert (ab_rows["target_room"] == "B").all()

        # Direction B->A adapts to room A
        ba_rows = df[df["direction"] == "B->A"]
        assert (ba_rows["target_room"] == "A").all()

        # 2 directions * 1 K * 1 seed * 4 metrics = 8 rows
        assert len(df) == 8

    def test_sweep_seed_coverage(self, tmp_path: Path) -> None:
        """Verify sweep loops through all requested seeds and aggregates them."""
        cfg = _setup_synthetic_env(tmp_path)
        seeds = [0, 1, 2]

        df = run_sweep(
            cfg=cfg,
            K_list=[1],
            seeds=seeds,
            directions=["A->B"],
            condition="random",
        )

        assert set(df["seed"].unique()) == {0, 1, 2}
        # 1 direction * 1 K * 3 seeds * 4 metrics = 12 rows
        assert len(df) == 12

    def test_sweep_10_seeds_coverage(self, tmp_path: Path) -> None:
        """Verify sweep executes full 10 seeds without error."""
        cfg = _setup_synthetic_env(tmp_path)
        seeds = list(range(10))

        df = run_sweep(
            cfg=cfg,
            K_list=[1],
            seeds=seeds,
            directions=["A->B"],
            condition="random",
        )

        assert set(df["seed"].unique()) == set(range(10))
        # 1 direction * 1 K * 10 seeds * 4 metrics = 40 rows
        assert len(df) == 40

    def test_sweep_correct_episode_invocation(self, tmp_path: Path) -> None:
        """Verify run_sweep calls run_episode with exact expected parameters."""
        cfg = _setup_synthetic_env(tmp_path)

        dummy_rows = [
            {"experiment": "fewshot_AtoB", "target_room": "B", "K": 1, "metric": "macro_f1", "value": "0.800000"},
            {"experiment": "fewshot_AtoB", "target_room": "B", "K": 1, "metric": "balanced_accuracy", "value": "0.750000"},
        ]
        dummy_preds = pd.DataFrame(columns=["window_id", "room", "label", "confidence"])

        with patch("scripts.sweep.run_episode", return_value=(dummy_rows, dummy_preds)) as mock_ep:
            df = run_sweep(
                cfg=cfg,
                K_list=[1, 2],
                seeds=[10, 20],
                directions=["A->B", "B->A"],
                capture_avg=True,
                condition="random",
            )

            # Total invocations: 2 directions * 2 K * 2 seeds = 8 calls
            assert mock_ep.call_count == 8

            # Check calls for (direction, target_room, K, seed)
            call_kwargs_list = [c.kwargs for c in mock_ep.call_args_list]

            # Verify target rooms match directions
            first_call = call_kwargs_list[0]
            assert first_call["target_room"] == "B"
            assert first_call["K"] == 1
            assert first_call["seed"] == 10
            assert first_call["capture_avg"] is True
            assert "cached_data" in first_call

            last_call = call_kwargs_list[-1]
            assert last_call["target_room"] == "A"
            assert last_call["K"] == 2
            assert last_call["seed"] == 20
            assert last_call["capture_avg"] is True

    def test_sweep_result_aggregation_and_output(self, tmp_path: Path) -> None:
        """Verify output DataFrame and saved CSV match the required columns and format."""
        cfg = _setup_synthetic_env(tmp_path)
        out_csv = tmp_path / "task3_sweep_test.csv"

        df = run_sweep(
            cfg=cfg,
            K_list=[1],
            seeds=[0, 1],
            directions=["A->B", "B->A"],
            out_csv=out_csv,
            condition="random",
        )

        assert out_csv.exists()
        saved_df = pd.read_csv(out_csv)

        expected_cols = ["experiment", "direction", "target_room", "K", "seed", "metric", "value"]
        assert list(saved_df.columns) == expected_cols
        assert list(df.columns) == expected_cols

        # Verify all 4 required metrics are present
        expected_metrics = {
            "macro_f1",
            "balanced_accuracy",
            "target_exclusive_macro_f1",
            "shared_class_macro_f1",
        }
        assert set(saved_df["metric"].unique()) == expected_metrics

        # Verify values can be parsed as float
        values = saved_df["value"].astype(float)
        assert not values.isna().any()

    def test_sweep_capture_avg(self, tmp_path: Path) -> None:
        """Verify capture_avg=True appends _capavg to experiment names."""
        cfg = _setup_synthetic_env(tmp_path)

        df = run_sweep(
            cfg=cfg,
            K_list=[1],
            seeds=[0],
            directions=["A->B"],
            capture_avg=True,
            condition="random",
        )

        assert (df["experiment"].str.endswith("_capavg")).all()

    def test_sweep_invalid_inputs(self, tmp_path: Path) -> None:
        """Verify invalid directions, non-positive K, or empty lists raise ValueError."""
        cfg = _setup_synthetic_env(tmp_path)

        # Invalid direction
        with pytest.raises(ValueError, match="Invalid direction"):
            run_sweep(cfg=cfg, K_list=[1], seeds=[0], directions=["invalid"])

        with pytest.raises(ValueError, match="Invalid direction"):
            run_sweep(cfg=cfg, K_list=[1], seeds=[0], directions=["A->C"])

        # Non-positive K
        with pytest.raises(ValueError, match="K must be positive"):
            run_sweep(cfg=cfg, K_list=[0], seeds=[0], directions=["A->B"])

        with pytest.raises(ValueError, match="K must be positive"):
            run_sweep(cfg=cfg, K_list=[-1], seeds=[0], directions=["A->B"])

        # Empty K_list
        with pytest.raises(ValueError, match="K_list cannot be empty"):
            run_sweep(cfg=cfg, K_list=[], seeds=[0], directions=["A->B"])

        # Empty seeds
        with pytest.raises(ValueError, match="seeds cannot be empty"):
            run_sweep(cfg=cfg, K_list=[1], seeds=[], directions=["A->B"])

    def test_sweep_smoke_cli(self) -> None:
        """Verify sweep CLI smoke test executes successfully."""
        _run_smoke(argparse.Namespace())
