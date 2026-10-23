"""
tests/test_config.py
====================
Unit tests for src/config.py.

Tests do NOT depend on the real dataset.  A minimal valid YAML fixture is
built in-memory using tmp_path; every test that expects a failure patches
exactly one field so the error is attributable to a single rule.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import (
    ExperimentConfig,
    load_config,
    ALLOWED_ACTIVITIES,
    ALLOWED_ROOMS,
    ALLOWED_SPLITS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    paths:
      room_a_pcap_dir:   "dataset/Experiment-2/Experiment-2/realdata/input_data/30ms/csidata"
      room_a_val_dir:    "dataset/Experiment-2/Experiment-2/realdata/input_data/30ms/csidata/validation"
      room_b_csv_dir:    "dataset/Experiment-3/Data"
      derived_dir:       "derived"
      intermediate_dir:  "derived/intermediate"
      windows_dir:       "derived/windows"
      manifest_csv:      "derived/manifest.csv"
      windows_index_csv: "derived/windows_index.csv"
      checkpoints_dir:   "checkpoints"
      reports_dir:       "reports"

    seeds:
      global_seed: 42
      fewshot_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    window:
      length: 150
      n_subcarriers: 256
      dev_stride: 50
      val_stride: 150
      short_capture_policy: "reflect_pad"

    cleaning:
      min_valid_packets: 10
      max_nan_fraction_per_packet: 0.10
      amplitude_clip_max: 5000.0
      amplitude_floor: 0.0
      max_zero_subcarrier_fraction: 0.50

    normalization:
      method: "per_capture"
      clip_after_norm: 10.0

    ssl:
      encoder: "cnn1d"
      projection_dim: 128
      embedding_dim: 256
      temperature: 0.07
      epochs: 50
      batch_size: 64
      lr: 1.0e-3
      weight_decay: 1.0e-4
      lr_min: 1.0e-5
      n_negatives: null

    augment:
      gaussian_noise_std: 0.05
      time_crop_min_frac: 0.70
      subcarrier_mask_frac: 0.10
      amplitude_scale_min: 0.80
      amplitude_scale_max: 1.20

    align:
      method: "none"

    routing:
      mode: "predicted_hard"
      confidence_threshold: 0.60

    fewshot:
      K_list: [1, 5, 10, 20]
      n_seeds: 10
      distance_metric: "cosine"
      shrinkage: "ledoit_wolf"
      classifier: "ncm"

    activities:
      room_a: ["fall", "sit", "sitdown", "stand", "standup", "walk"]
      room_b: ["empty", "lying", "sit", "stand", "walk"]
      shared: ["sit", "stand", "walk"]
      room_a_exclusive: ["fall", "sitdown", "standup"]
      room_b_exclusive: ["empty", "lying"]
      all: ["empty", "fall", "lying", "sit", "sitdown", "stand", "standup", "walk"]

    rooms:
      all: ["A", "B"]

    splits:
      all: ["dev", "val"]
      val_seed: 42
      val_captures_per_class: 25
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temp yaml file and return its path."""
    p = tmp_path / "experiment.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _patch_yaml(base: str, section: str, key: str, value: object) -> str:
    """Return *base* YAML with raw[section][key] replaced by *value*."""
    data = yaml.safe_load(base)
    data[section][key] = value
    return yaml.dump(data, default_flow_style=False)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadConfigValid:
    def test_returns_experiment_config(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert isinstance(cfg, ExperimentConfig)

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert isinstance(cfg.paths.manifest_csv, Path)
        assert isinstance(cfg.paths.room_a_pcap_dir, Path)

    def test_window_length_is_150(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.window.length == 150

    def test_n_subcarriers_is_256(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.window.n_subcarriers == 256

    def test_global_seed(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.seeds.global_seed == 42

    def test_fewshot_k_list(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.fewshot.K_list == [1, 5, 10, 20]

    def test_ssl_n_negatives_none(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.ssl.n_negatives is None

    def test_normalization_clip_none(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "normalization", "clip_after_norm", None)
        cfg = load_config(_write_yaml(tmp_path, content))
        assert cfg.normalization.clip_after_norm is None

    def test_config_is_frozen(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        with pytest.raises((AttributeError, TypeError)):
            cfg.window.length = 999  # type: ignore[misc]

    def test_config_hash_is_16_hex_chars(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        h = cfg.config_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_deterministic(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _VALID_YAML)
        h1 = load_config(p).config_hash()
        h2 = load_config(p).config_hash()
        assert h1 == h2

    def test_activity_to_room_exclusive(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert cfg.activity_to_room("fall") == "A"
        assert cfg.activity_to_room("empty") == "B"
        assert cfg.activity_to_room("lying") == "B"

    def test_activity_to_room_shared_returns_none(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        for act in ("sit", "stand", "walk"):
            assert cfg.activity_to_room(act) is None

    def test_activities_all_covers_both_rooms(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        all_set = set(cfg.activities.all)
        assert set(cfg.activities.room_a) <= all_set
        assert set(cfg.activities.room_b) <= all_set

    def test_shared_plus_exclusive_equals_room(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _VALID_YAML))
        assert set(cfg.activities.shared) | set(cfg.activities.room_a_exclusive) == set(cfg.activities.room_a)
        assert set(cfg.activities.shared) | set(cfg.activities.room_b_exclusive) == set(cfg.activities.room_b)


# ---------------------------------------------------------------------------
# File-not-found
# ---------------------------------------------------------------------------

class TestLoadConfigMissing:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Validation – window section
# ---------------------------------------------------------------------------

class TestWindowValidation:
    def test_bad_short_capture_policy(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "window", "short_capture_policy", "truncate")
        with pytest.raises(ValueError, match="short_capture_policy"):
            load_config(_write_yaml(tmp_path, content))

    def test_zero_window_length(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "window", "length", 0)
        with pytest.raises(ValueError):
            load_config(_write_yaml(tmp_path, content))

    def test_zero_dev_stride(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "window", "dev_stride", 0)
        with pytest.raises(ValueError):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – cleaning section
# ---------------------------------------------------------------------------

class TestCleaningValidation:
    def test_nan_fraction_above_one(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "cleaning", "max_nan_fraction_per_packet", 1.5)
        with pytest.raises(ValueError, match="max_nan_fraction_per_packet"):
            load_config(_write_yaml(tmp_path, content))

    def test_clip_max_below_floor(self, tmp_path: Path) -> None:
        data = yaml.safe_load(_VALID_YAML)
        data["cleaning"]["amplitude_clip_max"] = 0.0
        data["cleaning"]["amplitude_floor"] = 100.0
        content = yaml.dump(data)
        with pytest.raises(ValueError, match="amplitude_clip_max"):
            load_config(_write_yaml(tmp_path, content))

    def test_negative_min_valid_packets(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "cleaning", "min_valid_packets", -1)
        with pytest.raises(ValueError):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – normalization section
# ---------------------------------------------------------------------------

class TestNormalizationValidation:
    def test_bad_method(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "normalization", "method", "l2")
        with pytest.raises(ValueError, match="normalization.method"):
            load_config(_write_yaml(tmp_path, content))

    def test_negative_clip(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "normalization", "clip_after_norm", -1.0)
        with pytest.raises(ValueError, match="clip_after_norm"):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – SSL section
# ---------------------------------------------------------------------------

class TestSSLValidation:
    def test_bad_encoder(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "ssl", "encoder", "lstm")
        with pytest.raises(ValueError, match="ssl.encoder"):
            load_config(_write_yaml(tmp_path, content))

    def test_zero_batch_size(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "ssl", "batch_size", 0)
        with pytest.raises(ValueError):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – augment section
# ---------------------------------------------------------------------------

class TestAugmentValidation:
    def test_scale_min_exceeds_max(self, tmp_path: Path) -> None:
        data = yaml.safe_load(_VALID_YAML)
        data["augment"]["amplitude_scale_min"] = 1.5
        data["augment"]["amplitude_scale_max"] = 0.5
        content = yaml.dump(data)
        with pytest.raises(ValueError, match="amplitude_scale"):
            load_config(_write_yaml(tmp_path, content))

    def test_time_crop_out_of_range(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "augment", "time_crop_min_frac", 1.5)
        with pytest.raises(ValueError, match="time_crop_min_frac"):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – routing / fewshot / align sections
# ---------------------------------------------------------------------------

class TestRoutingValidation:
    def test_bad_mode(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "routing", "mode", "random")
        with pytest.raises(ValueError, match="routing.mode"):
            load_config(_write_yaml(tmp_path, content))

    def test_confidence_above_one(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "routing", "confidence_threshold", 1.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            load_config(_write_yaml(tmp_path, content))


class TestFewshotValidation:
    def test_bad_classifier(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "fewshot", "classifier", "svm")
        with pytest.raises(ValueError, match="fewshot.classifier"):
            load_config(_write_yaml(tmp_path, content))

    def test_zero_k_in_list(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "fewshot", "K_list", [0, 5])
        with pytest.raises(ValueError, match="K_list"):
            load_config(_write_yaml(tmp_path, content))


class TestAlignValidation:
    def test_bad_method(self, tmp_path: Path) -> None:
        content = _patch_yaml(_VALID_YAML, "align", "method", "pca")
        with pytest.raises(ValueError, match="align.method"):
            load_config(_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# Validation – activities consistency
# ---------------------------------------------------------------------------

class TestActivitiesValidation:
    def test_unknown_activity_in_room_a(self, tmp_path: Path) -> None:
        data = yaml.safe_load(_VALID_YAML)
        data["activities"]["room_a"].append("jump")
        data["activities"]["all"].append("jump")
        data["activities"]["room_a_exclusive"].append("jump")
        content = yaml.dump(data)
        with pytest.raises(ValueError, match="jump"):
            load_config(_write_yaml(tmp_path, content))

    def test_exclusive_not_covering_room(self, tmp_path: Path) -> None:
        """shared ∪ room_a_exclusive must equal room_a exactly."""
        data = yaml.safe_load(_VALID_YAML)
        # Remove 'fall' from exclusive so the union no longer equals room_a
        data["activities"]["room_a_exclusive"] = ["sitdown", "standup"]
        content = yaml.dump(data)
        with pytest.raises(ValueError, match="room_a"):
            load_config(_write_yaml(tmp_path, content))
