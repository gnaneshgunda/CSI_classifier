"""Tests for src/config.py and src/manifest.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import ALLOWED_ACTIVITIES, Config, load_config
from src.manifest import (
    MANIFEST_COLUMNS,
    _assign_splits,
    _validate,
    build_manifest,
    read_manifest,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_load_config_default():
    cfg = load_config("configs/experiment.yaml")
    assert cfg.window.length == 150
    assert cfg.window.dev_stride == 50
    assert cfg.fewshot.K_list == [1, 5, 10, 20]
    assert cfg.routing.mode in {"predicted_hard", "predicted_soft", "oracle", "none"}


def test_config_invalid_activity(tmp_path):
    import yaml
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["activities"].append("Running")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump(raw))
    with pytest.raises(AssertionError, match="Unknown activities"):
        load_config(bad_yaml)


def test_config_invalid_routing(tmp_path):
    import yaml
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["routing"]["mode"] = "magic"
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump(raw))
    with pytest.raises(AssertionError):
        load_config(bad_yaml)


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

def _make_fake_csvs(root: Path, n_per_class: int = 3) -> None:
    header = ",".join(str(i) for i in range(256))
    for act in ALLOWED_ACTIVITIES:
        for idx in range(n_per_class):
            rows = [header] + [",".join(["100"] * 256)] * 200
            (root / f"{act}_{idx}.csv").write_text("\n".join(rows))


def test_assign_splits_no_leakage():
    records = [
        {"capture_id": f"Walking_{i}", "activity": "Walking", "room": "room1",
         "csv_path": "", "n_packets": 200}
        for i in range(20)
    ]
    records = _assign_splits(records, train_frac=0.7, val_frac=0.15, seed=0)
    df = pd.DataFrame(records)
    # Each capture_id appears exactly once
    assert df["capture_id"].nunique() == len(df)
    # No capture in two splits (trivially true here, but validates logic)
    dup = df.groupby("capture_id")["split"].nunique()
    assert (dup == 1).all()


def test_validate_rejects_duplicate_split():
    df = pd.DataFrame([
        {"capture_id": "Walking_0", "activity": "Walking", "room": "room1",
         "split": "train", "csv_path": "", "n_packets": 200},
        {"capture_id": "Walking_0", "activity": "Walking", "room": "room1",
         "split": "val", "csv_path": "", "n_packets": 200},
    ])
    with pytest.raises(AssertionError, match="multiple splits"):
        _validate(df)


def test_validate_rejects_unknown_activity():
    df = pd.DataFrame([
        {"capture_id": "Running_0", "activity": "Running", "room": "room1",
         "split": "train", "csv_path": "", "n_packets": 200},
    ])
    with pytest.raises(AssertionError, match="Unknown activities"):
        _validate(df)


def test_build_and_roundtrip(tmp_path):
    _make_fake_csvs(tmp_path)
    cfg = load_config("configs/experiment.yaml")
    # Override dataset root to tmp_path
    cfg.paths.dataset_root = str(tmp_path)
    cfg.paths.manifest_csv = str(tmp_path / "manifest.csv")

    df = build_manifest(cfg)
    assert set(df.columns) == set(MANIFEST_COLUMNS)
    assert set(df["activity"]).issubset(ALLOWED_ACTIVITIES)

    write_manifest(df, cfg.paths.manifest_csv)
    df2 = read_manifest(cfg.paths.manifest_csv)
    assert len(df2) == len(df)


def test_no_val_data_used_for_normalization():
    """Conceptual guard: val/test capture_ids must not appear in train."""
    records = [
        {"capture_id": f"Sitting_{i}", "activity": "Sitting", "room": "room1",
         "csv_path": "", "n_packets": 200}
        for i in range(10)
    ]
    records = _assign_splits(records, train_frac=0.7, val_frac=0.15, seed=42)
    df = pd.DataFrame(records)
    train_ids = set(df[df["split"] == "train"]["capture_id"])
    val_ids = set(df[df["split"] == "val"]["capture_id"])
    test_ids = set(df[df["split"] == "test"]["capture_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
