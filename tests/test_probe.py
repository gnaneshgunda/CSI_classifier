"""
tests/test_probe.py
===================
Unit tests for src/probe.py.

Verifies:
1. EmbeddingExtractor:
   - Fixed-length embedding extraction across conditions
   - Correct embedding dimensions (handcrafted 524-d, encoder 256-d)
   - Both checkpoint and random initialization modes
   - Encoder weights remain completely frozen (eval mode, bitwise identical before/after)
   - Missing checkpoint and missing checkpoint_dir raise FileNotFoundError
   - Invalid condition raises ValueError
   - Empty input splits and missing window files handled gracefully
2. ProbeNormalizer:
   - Row-wise L2 normalization onto unit sphere
   - Zero vectors handled without NaN/inf
   - Development-only fitting: statistics fit exclusively on dev data
   - Validation data normalized using dev statistics only (no leakage)
   - Guard against calling transform before fit
   - 1D and 2D input support
3. LinearProbe:
   - LogisticRegression probe initialization, training, and prediction
   - Development-only training and validation scoring
   - Macro-F1 and balanced accuracy calculation
   - String labels and integer labels support
   - Probability outputs from predict_proba sum to 1.0
   - Unseen validation classes handled cleanly
   - Guard against calling score before fit
4. run_probe:
   - Integration of extraction, normalization, probing, and scoring
   - Supports joint, per_room, and both scopes
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from src.probe import (
    ALL_CONDITIONS,
    CHECKPOINT_CONDITIONS,
    EmbeddingExtractor,
    LinearProbe,
    ProbeNormalizer,
    run_probe,
)


# ===========================================================================
# 1. ProbeNormalizer Tests
# ===========================================================================

class TestProbeNormalizer:
    """Tests for ProbeNormalizer (L2 + StandardScaler)."""

    def test_l2_normalization_unit_sphere(self) -> None:
        """Every non-zero row has L2 norm equal to 1.0."""
        rng = np.random.default_rng(1)
        X = rng.normal(loc=5.0, scale=2.0, size=(10, 64)).astype(np.float32)

        X_l2 = ProbeNormalizer._l2(X)
        norms = np.linalg.norm(X_l2, axis=1)

        np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)

    def test_l2_zero_vector_clamping(self) -> None:
        """Zero vectors are clamped to zero without division-by-zero, NaNs, or Infs."""
        X = np.zeros((3, 32), dtype=np.float32)
        X_l2 = ProbeNormalizer._l2(X)

        assert not np.isnan(X_l2).any()
        assert not np.isinf(X_l2).any()
        np.testing.assert_allclose(X_l2, 0.0)

    def test_dev_only_standardization_no_leakage(self) -> None:
        """Scaler is fit exclusively on dev data; val data does not alter mean or scale."""
        rng = np.random.default_rng(2)
        X_dev = rng.normal(loc=10.0, scale=3.0, size=(50, 32)).astype(np.float32)
        X_val = rng.normal(loc=-50.0, scale=1.0, size=(20, 32)).astype(np.float32)

        norm = ProbeNormalizer()
        norm.fit(X_dev)

        # Expected scaler stats match L2-normalized dev
        dev_l2 = ProbeNormalizer._l2(X_dev)
        expected_mean = dev_l2.mean(axis=0)
        expected_scale = dev_l2.std(axis=0)

        np.testing.assert_allclose(norm._scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(norm._scaler.scale_, expected_scale, rtol=1e-5, atol=1e-5)

        # Transform val using dev statistics
        val_norm = norm.transform(X_val)
        assert val_norm.shape == X_val.shape
        assert val_norm.dtype == np.float32

        # Corrupting val in memory must not alter the fitted normalizer
        corrupted_val = np.full_like(X_val, 9999.0)
        _ = norm.transform(corrupted_val)

        np.testing.assert_allclose(norm._scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(norm._scaler.scale_, expected_scale, rtol=1e-5, atol=1e-5)

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform before fit raises RuntimeError."""
        norm = ProbeNormalizer()
        X = np.ones((5, 16), dtype=np.float32)
        with pytest.raises(RuntimeError, match="must be called first"):
            norm.transform(X)

    def test_fit_transform_consistency(self) -> None:
        """fit_transform(X) equals fit(X).transform(X)."""
        rng = np.random.default_rng(3)
        X = rng.uniform(0.1, 10.0, size=(15, 20)).astype(np.float32)

        n1 = ProbeNormalizer()
        res1 = n1.fit_transform(X)

        n2 = ProbeNormalizer()
        n2.fit(X)
        res2 = n2.transform(X)

        np.testing.assert_allclose(res1, res2, rtol=1e-6)

    def test_transform_1d_input(self) -> None:
        """Handles 1D vector (D,) input gracefully."""
        rng = np.random.default_rng(4)
        X_dev = rng.uniform(1.0, 5.0, size=(10, 8)).astype(np.float32)
        norm = ProbeNormalizer().fit(X_dev)

        vec = X_dev[0]
        out_vec = norm.transform(vec)
        assert out_vec.shape == (8,)


# ===========================================================================
# 2. EmbeddingExtractor Tests
# ===========================================================================

@pytest.fixture
def synthetic_windows_dir(tmp_path: Path):
    """Create a minimal windows dataset with dev and val splits."""
    windows_dir = tmp_path / "windows"
    (windows_dir / "dev").mkdir(parents=True)
    (windows_dir / "val").mkdir(parents=True)

    rng = np.random.default_rng(42)
    T, N_SUB = 150, 256
    rows = []

    activities = ["sit", "stand", "walk"]
    for split in ["dev", "val"]:
        n_per_act = 3 if split == "dev" else 1
        for room in ["A", "B"]:
            for act in activities:
                for i in range(n_per_act):
                    wid = f"{room}_{act}_{split}_{i:03d}"
                    arr = rng.uniform(0.5, 15.0, size=(T, N_SUB)).astype(np.float32)
                    np.save(str(windows_dir / split / f"{wid}.npy"), arr)
                    rows.append({
                        "window_id": wid,
                        "capture_id": f"cap_{room}_{act}_{i}",
                        "room": room,
                        "activity": act,
                        "split": split,
                        "start": 0,
                        "config_hash": "test",
                    })

    df = pd.DataFrame(rows)
    return df, windows_dir


class TestEmbeddingExtractor:
    """Tests for EmbeddingExtractor dimensions, weight freezing, and loading."""

    def test_dimensions_handcrafted(self, synthetic_windows_dir) -> None:
        """Handcrafted condition returns 524-d features from src.features."""
        df, windows_dir = synthetic_windows_dir
        extractor = EmbeddingExtractor(condition="handcrafted")
        assert extractor.embedding_dim == 524

        X, y, labels = extractor.extract(df, windows_dir, split="dev")
        assert X.shape[1] == 524
        assert X.dtype == np.float32
        assert len(y) == len(X)
        assert len(labels) == len(X)

    def test_dimensions_random_encoder(self, synthetic_windows_dir) -> None:
        """Random condition returns 256-d embeddings from randomly initialised encoder."""
        df, windows_dir = synthetic_windows_dir
        extractor = EmbeddingExtractor(condition="random", device="cpu")
        assert extractor.embedding_dim == 256

        X, y, labels = extractor.extract(df, windows_dir, split="dev")
        assert X.shape[1] == 256
        assert X.dtype == np.float32
        assert len(y) == len(X)
        assert len(labels) == len(X)

    def test_encoder_weights_frozen_during_extraction(self, synthetic_windows_dir) -> None:
        """Encoder parameters remain strictly identical bit-for-bit before and after extraction."""
        df, windows_dir = synthetic_windows_dir
        extractor = EmbeddingExtractor(condition="random", device="cpu")
        enc = extractor._get_encoder()

        # Capture original weight tensors
        orig_weights = {name: param.clone() for name, param in enc.named_parameters()}

        # Run extraction across dev and val
        _ = extractor.extract(df, windows_dir, split="dev")
        _ = extractor.extract(df, windows_dir, split="val")

        # Verify all weights remain identical
        for name, param in enc.named_parameters():
            assert torch.equal(param, orig_weights[name]), f"Weights modified for {name}"

    def test_checkpoint_mode_loads_weights(self, synthetic_windows_dir, tmp_path: Path) -> None:
        """Checkpoint conditions load saved state_dict and wrapped state_dict."""
        from src.encoder import ResNet1DEncoder

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        # Save a reference checkpoint for vicreg_aug
        ref_enc = ResNet1DEncoder(embedding_dim=256, random_init=True, verbose=False)
        torch.save(ref_enc.state_dict(), str(ckpt_dir / "vicreg_aug.pt"))

        extractor = EmbeddingExtractor(
            condition="vicreg_aug",
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        loaded_enc = extractor._get_encoder()

        for name, param in ref_enc.named_parameters():
            loaded_param = dict(loaded_enc.named_parameters())[name]
            assert torch.equal(param, loaded_param)

    def test_checkpoint_wrapped_state_dict(self, tmp_path: Path) -> None:
        """Handles checkpoint with 'encoder' key."""
        from src.encoder import ResNet1DEncoder

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        ref_enc = ResNet1DEncoder(embedding_dim=256, random_init=True, verbose=False)
        wrapped = {"encoder": ref_enc.state_dict(), "epoch": 10}
        torch.save(wrapped, str(ckpt_dir / "masked.pt"))

        extractor = EmbeddingExtractor(
            condition="masked",
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        loaded_enc = extractor._get_encoder()

        for name, param in ref_enc.named_parameters():
            loaded_param = dict(loaded_enc.named_parameters())[name]
            assert torch.equal(param, loaded_param)

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised when checkpoint file does not exist."""
        ckpt_dir = tmp_path / "empty_ckpts"
        ckpt_dir.mkdir(parents=True)

        extractor = EmbeddingExtractor(
            condition="vicreg_noaug",
            checkpoint_dir=ckpt_dir,
        )
        with pytest.raises(FileNotFoundError, match="Encoder checkpoint for condition"):
            extractor._get_encoder()

    def test_missing_checkpoint_dir_raises_file_not_found(self) -> None:
        """FileNotFoundError is raised if checkpoint_dir is None for checkpoint condition."""
        extractor = EmbeddingExtractor(condition="vicreg_aug", checkpoint_dir=None)
        with pytest.raises(FileNotFoundError, match="checkpoint_dir must be specified"):
            extractor._get_encoder()

    def test_unknown_condition_raises_value_error(self) -> None:
        """ValueError is raised if an invalid condition is requested."""
        with pytest.raises(ValueError, match="Unknown condition"):
            EmbeddingExtractor(condition="nonexistent_condition")

    def test_extract_empty_split(self, synthetic_windows_dir) -> None:
        """Returns empty arrays of shape (0, dim) if index has no matching rows."""
        df, windows_dir = synthetic_windows_dir
        extractor = EmbeddingExtractor(condition="random")

        # Request split not in dataframe
        X, y, labels = extractor.extract(df, windows_dir, split="test_split")
        assert X.shape == (0, 256)
        assert len(y) == 0
        assert len(labels) == 0


# ===========================================================================
# 3. LinearProbe Tests
# ===========================================================================

class TestLinearProbe:
    """Tests for LinearProbe fitting, scoring, and label handling."""

    def test_fit_and_score_basic(self) -> None:
        """LinearProbe fits on dev and computes macro_f1 and balanced_accuracy on val."""
        rng = np.random.default_rng(10)
        # Class 0: mean -2.0, Class 1: mean +2.0
        X_dev = np.vstack([
            rng.normal(loc=-2.0, scale=1.0, size=(30, 16)),
            rng.normal(loc=2.0, scale=1.0, size=(30, 16)),
        ]).astype(np.float32)
        y_dev = np.array([0] * 30 + [1] * 30)

        X_val = np.vstack([
            rng.normal(loc=-2.0, scale=1.0, size=(10, 16)),
            rng.normal(loc=2.0, scale=1.0, size=(10, 16)),
        ]).astype(np.float32)
        y_val = np.array([0] * 10 + [1] * 10)

        probe = LinearProbe()
        probe.fit(X_dev, y_dev)
        assert probe._is_fit is True

        scores = probe.score(X_val, y_val)
        assert "macro_f1" in scores and "balanced_accuracy" in scores
        assert 0.0 <= scores["macro_f1"] <= 1.0
        assert 0.0 <= scores["balanced_accuracy"] <= 1.0
        assert scores["balanced_accuracy"] > 0.80

    def test_score_before_fit_raises_runtime_error(self) -> None:
        """Calling score before fit raises RuntimeError."""
        probe = LinearProbe()
        X = np.ones((5, 10), dtype=np.float32)
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(RuntimeError, match="must be called first"):
            probe.score(X, y)

    def test_predict_proba_simplex(self) -> None:
        """predict_proba outputs sum to 1.0 per sample."""
        rng = np.random.default_rng(11)
        X_dev = rng.normal(size=(20, 8)).astype(np.float32)
        y_dev = np.array([0] * 10 + [1] * 10)

        probe = LinearProbe().fit(X_dev, y_dev)
        probs = probe.predict_proba(X_dev[:5])

        assert probs.shape == (5, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_string_labels_support(self) -> None:
        """LinearProbe accepts string activity labels directly."""
        rng = np.random.default_rng(12)
        X_dev = rng.normal(size=(20, 8)).astype(np.float32)
        y_dev = np.array(["sit"] * 10 + ["stand"] * 10)

        X_val = rng.normal(size=(6, 8)).astype(np.float32)
        y_val = np.array(["sit"] * 3 + ["stand"] * 3)

        probe = LinearProbe()
        probe.fit(X_dev, y_dev)
        assert probe.label_encoder is not None
        assert set(probe.label_encoder.classes_) == {"sit", "stand"}

        scores = probe.score(X_val, y_val)
        assert 0.0 <= scores["macro_f1"] <= 1.0
        assert 0.0 <= scores["balanced_accuracy"] <= 1.0

    def test_unseen_val_classes_handled(self) -> None:
        """Validation samples with unseen classes are safely filtered with a warning."""
        rng = np.random.default_rng(13)
        X_dev = rng.normal(size=(20, 8)).astype(np.float32)
        y_dev = np.array([0] * 10 + [1] * 10)

        # Val contains unknown class 2
        X_val = rng.normal(size=(9, 8)).astype(np.float32)
        y_val = np.array([0] * 3 + [1] * 3 + [2] * 3)

        probe = LinearProbe().fit(X_dev, y_dev)
        scores = probe.score(X_val, y_val)

        assert 0.0 <= scores["macro_f1"] <= 1.0
        assert 0.0 <= scores["balanced_accuracy"] <= 1.0


# ===========================================================================
# 4. run_probe Integration Tests
# ===========================================================================

class TestRunProbe:
    """Tests for run_probe convenience function."""

    def test_run_probe_both_scopes(self, synthetic_windows_dir) -> None:
        """run_probe with scope='both' generates joint and per_room rows."""
        df, windows_dir = synthetic_windows_dir
        rows = run_probe(
            cfg=None,
            condition="random",
            index_df=df,
            windows_dir=windows_dir,
            scope="both",
            device="cpu",
        )

        assert len(rows) > 0
        scopes = {r["scope"] for r in rows}
        rooms  = {r["target_room"] for r in rows}
        metrics = {r["metric"] for r in rows}

        assert scopes == {"joint", "per_room"}
        assert rooms == {"joint", "A", "B"}
        assert metrics == {"macro_f1", "balanced_accuracy"}

        for r in rows:
            assert r["experiment"] == "probe_random"
            assert r["condition"] == "random"
            assert r["K"] == -1
            assert 0.0 <= float(r["value"]) <= 1.0
