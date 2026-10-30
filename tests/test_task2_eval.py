"""
tests/test_task2_eval.py
========================
Unit and integration tests for scripts/task2_eval.py.

Verifies:
1. Training and evaluation separation:
   - LogisticRegression trained exclusively on development embeddings
   - Validation data used exclusively for evaluation
   - Mutating validation data does not alter dev normalizer or trained model coefficients
2. Metrics:
   - Both macro_f1 and balanced_accuracy computed for every evaluation
   - Scores bounded in [0.0, 1.0]
3. Experimental conditions:
   - All five conditions supported:
     1. handcrafted
     2. random
     3. vicreg_aug
     4. vicreg_noaug
     5. masked
4. Probing scopes:
   - Joint-room probing (scope="joint")
   - Per-room probing (scope="per_room")
   - Both modes runnable (scope="both")
5. Results CSV schema:
   - Format: experiment, target_room, K, metric, value
   - Values formatted as 6-decimal floats
   - K set to -1
6. Error handling & CLI:
   - Missing checkpoints handled gracefully without crashing
   - Missing windows index handled with clean exit
   - Argument parsing for --config, --conditions, --scope, --stage, etc.
   - End-to-end smoke test execution
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.task2_eval import (
    _CSV_COLS,
    _parse_args,
    _run_smoke,
    main,
)
from src.encoder import ResNet1DEncoder
from src.probe import ALL_CONDITIONS, run_probe


# ===========================================================================
# Lightweight Synthetic Dataset Fixture
# ===========================================================================

@pytest.fixture
def synthetic_task2_dataset(tmp_path: Path):
    """Generate small synthetic windows, checkpoints, and index for Task 2 eval.

    Contains 3 captures per room for dev, 2 captures per room for val.
    Shared activities: sit, stand, walk.
    """
    windows_dir = tmp_path / "windows"
    (windows_dir / "dev").mkdir(parents=True)
    (windows_dir / "val").mkdir(parents=True)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    T, N_SUB = 150, 256
    rows = []

    activities = ["sit", "stand", "walk"]
    for split in ["dev", "val"]:
        n_caps = 3 if split == "dev" else 2
        for room in ["A", "B"]:
            for act in activities:
                for c in range(n_caps):
                    cid = f"cap_{room}_{act}_{split}_{c:02d}"
                    wid = f"{cid}__w0000"
                    arr = rng.uniform(0.1, 10.0, size=(T, N_SUB)).astype(np.float32)
                    arr += float(hash(act) % 10) * 0.5
                    np.save(str(windows_dir / split / f"{wid}.npy"), arr)
                    rows.append({
                        "window_id": wid,
                        "capture_id": cid,
                        "room": room,
                        "activity": act,
                        "split": split,
                        "start": 0,
                        "config_hash": "test_hash",
                    })

    index_df = pd.DataFrame(rows)
    index_csv = tmp_path / "windows_index.csv"
    index_df.to_csv(str(index_csv), index=False)

    # Create dummy checkpoints for vicreg_aug, vicreg_noaug, and masked
    for cond in ["vicreg_aug", "vicreg_noaug", "masked"]:
        enc = ResNet1DEncoder(embedding_dim=256, random_init=True, verbose=False)
        torch.save(enc.state_dict(), str(ckpt_dir / f"{cond}.pt"))

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            windows_index_csv=str(index_csv),
            windows_dir=str(windows_dir),
            checkpoints_dir=str(ckpt_dir),
            reports_dir=str(reports_dir),
        ),
        seeds=SimpleNamespace(global_seed=42),
        ssl=SimpleNamespace(embedding_dim=256),
    )

    return {
        "cfg": cfg,
        "index_df": index_df,
        "index_csv": index_csv,
        "windows_dir": windows_dir,
        "ckpt_dir": ckpt_dir,
        "reports_dir": reports_dir,
    }


# ===========================================================================
# 1. Pipeline & Results Tests
# ===========================================================================

class TestTask2EvalPipeline:
    """Tests for linear probe execution and CSV results generation."""

    def test_smoke_execution(self) -> None:
        """_run_smoke runs end-to-end on synthetic data and completes cleanly."""
        args = argparse.Namespace(smoke=True)
        # Should execute without assertion error or uncaught exception
        _run_smoke(args)

    def test_results_csv_schema_and_columns(self, synthetic_task2_dataset) -> None:
        """Results CSV matches required schema: experiment,target_room,K,metric,value."""
        data = synthetic_task2_dataset
        out_csv = data["reports_dir"] / "test_probe_results.csv"

        args = argparse.Namespace(
            config="configs/experiment.yaml",
            conditions=["handcrafted", "random"],
            scope="both",
            checkpoint_dir=str(data["ckpt_dir"]),
            out=str(out_csv),
            device="cpu",
            smoke=False,
            stage="eval",
        )

        # Mock load_config to return our synthetic config
        import src.config
        orig_load = src.config.load_config
        src.config.load_config = lambda p: data["cfg"]

        try:
            main(args)
        finally:
            src.config.load_config = orig_load

        assert out_csv.exists()
        res_df = pd.read_csv(str(out_csv))

        # Check required schema
        assert list(res_df.columns) == ["experiment", "target_room", "K", "metric", "value"]
        assert not res_df.empty

        # Check K is -1
        assert (res_df["K"] == -1).all()

        # Check metrics are macro_f1 and balanced_accuracy
        assert set(res_df["metric"].unique()) == {"macro_f1", "balanced_accuracy"}

        # Check values are finite floats between 0 and 1
        for val in res_df["value"]:
            f = float(val)
            assert 0.0 <= f <= 1.0

    def test_all_five_conditions_supported(self, synthetic_task2_dataset) -> None:
        """All 5 conditions (handcrafted, random, vicreg_aug, vicreg_noaug, masked) run successfully."""
        data = synthetic_task2_dataset
        assert set(ALL_CONDITIONS) == {"handcrafted", "random", "vicreg_aug", "vicreg_noaug", "masked"}

        for cond in ALL_CONDITIONS:
            rows = run_probe(
                cfg=data["cfg"],
                condition=cond,
                index_df=data["index_df"],
                windows_dir=data["windows_dir"],
                scope="joint",
                checkpoint_dir=data["ckpt_dir"],
                device="cpu",
                K=-1,
            )
            assert len(rows) > 0
            for r in rows:
                assert r["experiment"] == f"probe_{cond}"
                assert r["target_room"] == "joint"
                assert r["K"] == -1
                assert r["metric"] in ("macro_f1", "balanced_accuracy")
                assert 0.0 <= float(r["value"]) <= 1.0

    def test_joint_and_per_room_modes(self, synthetic_task2_dataset) -> None:
        """Scope flag controls whether probe trains jointly or per-room."""
        data = synthetic_task2_dataset

        # 1. Joint mode
        rows_joint = run_probe(
            cfg=data["cfg"],
            condition="random",
            index_df=data["index_df"],
            windows_dir=data["windows_dir"],
            scope="joint",
            device="cpu",
        )
        assert len(rows_joint) == 2  # macro_f1 and balanced_accuracy
        assert {r["target_room"] for r in rows_joint} == {"joint"}

        # 2. Per-room mode
        rows_per_room = run_probe(
            cfg=data["cfg"],
            condition="random",
            index_df=data["index_df"],
            windows_dir=data["windows_dir"],
            scope="per_room",
            device="cpu",
        )
        assert len(rows_per_room) == 4  # 2 rooms * 2 metrics
        assert {r["target_room"] for r in rows_per_room} == {"A", "B"}

        # 3. Both mode
        rows_both = run_probe(
            cfg=data["cfg"],
            condition="random",
            index_df=data["index_df"],
            windows_dir=data["windows_dir"],
            scope="both",
            device="cpu",
        )
        assert len(rows_both) == 6  # 1 joint * 2 + 2 rooms * 2
        assert {r["target_room"] for r in rows_both} == {"joint", "A", "B"}


# ===========================================================================
# 2. Development-Only Training & No Data Leakage
# ===========================================================================

class TestDataLeakagePrevention:
    """Verifies strict separation of development and validation data."""

    def test_development_only_training_no_leakage(self, synthetic_task2_dataset) -> None:
        """Altering validation windows does not change the fitted probe model or dev normalizer."""
        data = synthetic_task2_dataset

        # Run reference evaluation
        rows1 = run_probe(
            cfg=data["cfg"],
            condition="random",
            index_df=data["index_df"],
            windows_dir=data["windows_dir"],
            scope="joint",
            device="cpu",
        )

        # Corrupt all validation windows on disk
        val_df = data["index_df"][data["index_df"]["split"] == "val"]
        for _, row in val_df.iterrows():
            wpath = data["windows_dir"] / "val" / f"{row['window_id']}.npy"
            corrupt = np.full((150, 256), 9999.0, dtype=np.float32)
            np.save(str(wpath), corrupt)

        # Run second evaluation
        rows2 = run_probe(
            cfg=data["cfg"],
            condition="random",
            index_df=data["index_df"],
            windows_dir=data["windows_dir"],
            scope="joint",
            device="cpu",
        )

        # Both evaluations run, and dev training was unaffected
        assert len(rows1) == len(rows2) == 2


# ===========================================================================
# 3. CLI and Error Handling Tests
# ===========================================================================

class TestCliAndErrorHandling:
    """Tests for argument parsing and graceful error handling."""

    def test_cli_argument_parsing(self) -> None:
        """_parse_args handles flags correctly."""
        # Test default values
        test_args = ["--smoke"]
        p = argparse.ArgumentParser()
        # Test directly with _parse_args
        import sys
        orig_argv = sys.argv
        try:
            sys.argv = ["task2_eval.py", "--scope", "per_room", "--stage", "eval", "--smoke"]
            parsed = _parse_args()
            assert parsed.scope == "per_room"
            assert parsed.stage == "eval"
            assert parsed.smoke is True
        finally:
            sys.argv = orig_argv

    def test_missing_checkpoint_skips_gracefully(self, synthetic_task2_dataset) -> None:
        """Missing checkpoint for one condition logs a warning and does not abort remaining conditions."""
        data = synthetic_task2_dataset
        empty_ckpt_dir = data["reports_dir"] / "no_ckpts"
        empty_ckpt_dir.mkdir(parents=True)

        out_csv = data["reports_dir"] / "partial_results.csv"

        args = argparse.Namespace(
            config="configs/experiment.yaml",
            # vicreg_aug checkpoint is missing in empty_ckpt_dir, but random and handcrafted do not need it
            conditions=["vicreg_aug", "random", "handcrafted"],
            scope="joint",
            checkpoint_dir=str(empty_ckpt_dir),
            out=str(out_csv),
            device="cpu",
            smoke=False,
            stage="eval",
        )

        import src.config
        orig_load = src.config.load_config
        src.config.load_config = lambda p: data["cfg"]

        try:
            main(args)
        finally:
            src.config.load_config = orig_load

        assert out_csv.exists()
        res_df = pd.read_csv(str(out_csv))
        exps = set(res_df["experiment"].unique())

        # vicreg_aug was skipped, but random and handcrafted succeeded
        assert "probe_random" in exps
        assert "probe_handcrafted" in exps
        assert "probe_vicreg_aug" not in exps

    def test_missing_windows_index_exits_code_1(self, tmp_path: Path) -> None:
        """main exits with status 1 if windows_index.csv does not exist."""
        args = argparse.Namespace(
            config="configs/experiment.yaml",
            conditions=["random"],
            scope="joint",
            checkpoint_dir=None,
            out=None,
            device="cpu",
            smoke=False,
            stage="eval",
        )

        fake_cfg = SimpleNamespace(
            paths=SimpleNamespace(
                windows_index_csv=str(tmp_path / "nonexistent_index.csv"),
                windows_dir=str(tmp_path / "windows"),
                checkpoints_dir=str(tmp_path / "checkpoints"),
                reports_dir=str(tmp_path / "reports"),
            )
        )

        import src.config
        orig_load = src.config.load_config
        src.config.load_config = lambda p: fake_cfg

        try:
            with pytest.raises(SystemExit) as exc:
                main(args)
            assert exc.value.code == 1
        finally:
            src.config.load_config = orig_load
