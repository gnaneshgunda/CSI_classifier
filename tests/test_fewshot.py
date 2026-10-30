"""
tests/test_fewshot.py
=====================
Unit tests for src/fewshot.py.

Verifies:
1. sample_support:
   - support never contains validation captures (split == "dev" strictly)
   - support never contains shared activity labels ("sit", "stand", "walk")
   - support only contains target-exclusive activities for the target room
   - one capture counts as one shot (distinct capture_id, independent of window count)
   - deterministic given seed
   - raises ValueError on invalid room or insufficient dev captures
2. PrototypeHead:
   - source prototypes (mean embedding per class from source development labels)
   - target-exclusive prototypes (average windows within capture first, then across captures)
   - one capture counts as one shot hierarchical weighting
   - masking predictions strictly to target room's valid classes
   - Gaussian classifier with pooled shrunken covariance (alpha configurable, ledoit_wolf)
   - cosine variant
   - predict -> (label, confidence) for single vector and batch
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.manifest import ManifestRow
from src.fewshot import (
    EXCLUSIVE_ACTIVITIES,
    ROOM_A_ACTIVITIES,
    ROOM_A_EXCLUSIVE,
    ROOM_B_ACTIVITIES,
    ROOM_B_EXCLUSIVE,
    SHARED_ACTIVITIES,
    VALID_ACTIVITIES,
    PrototypeHead,
    SupportRow,
    SupportSet,
    compute_pooled_covariance,
    compute_source_prototypes,
    compute_target_exclusive_prototypes,
    sample_support,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_mock_manifest() -> List[ManifestRow]:
    """Create a realistic mock manifest covering both rooms, splits, and all activities."""
    rows: List[ManifestRow] = []

    # Room A: "fall", "sit", "sitdown", "stand", "standup", "walk"
    # Create 10 dev and 5 val captures per activity
    for act in ROOM_A_ACTIVITIES:
        for i in range(10):
            cid = f"A__{act}__dev_{i:03d}"
            rows.append(
                ManifestRow(
                    capture_id=cid,
                    room="A",
                    activity=act,
                    split="dev",
                    source_file=f"pcap/A/{act}/{cid}.pcap",
                    n_packets_raw=200,
                    n_packets_clean=200,
                    intermediate_path=f"derived/intermediate/A/{cid}.npy",
                )
            )
        for i in range(5):
            cid = f"A__{act}__val_{i:03d}"
            rows.append(
                ManifestRow(
                    capture_id=cid,
                    room="A",
                    activity=act,
                    split="val",
                    source_file=f"pcap/A/{act}/{cid}.pcap",
                    n_packets_raw=200,
                    n_packets_clean=200,
                    intermediate_path=f"derived/intermediate/A/{cid}.npy",
                )
            )

    # Room B: "empty", "lying", "sit", "stand", "walk"
    # Create 10 dev and 5 val captures per activity
    for act in ROOM_B_ACTIVITIES:
        for i in range(10):
            cid = f"B__{act}__dev_{i:03d}"
            rows.append(
                ManifestRow(
                    capture_id=cid,
                    room="B",
                    activity=act,
                    split="dev",
                    source_file=f"csv/B/{act}/{cid}.csv",
                    n_packets_raw=200,
                    n_packets_clean=200,
                    intermediate_path=f"derived/intermediate/B/{cid}.npy",
                )
            )
        for i in range(5):
            cid = f"B__{act}__val_{i:03d}"
            rows.append(
                ManifestRow(
                    capture_id=cid,
                    room="B",
                    activity=act,
                    split="val",
                    source_file=f"csv/B/{act}/{cid}.csv",
                    n_packets_raw=200,
                    n_packets_clean=200,
                    intermediate_path=f"derived/intermediate/B/{cid}.npy",
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Tests: sample_support
# ---------------------------------------------------------------------------

class TestSampleSupport:
    def test_support_never_contains_validation(self) -> None:
        """Requirement: support set must NEVER contain validation captures."""
        manifest = _make_mock_manifest()

        for room in ("A", "B"):
            for K in (1, 3, 5):
                support = sample_support(manifest, target_room=room, K=K, seed=42)
                for row in support:
                    assert row.split == "dev", f"Found non-dev capture in support: {row}"
                    assert row.split != "val", f"Found val capture in support: {row}"
                    assert row["split"] == "dev"

    def test_support_never_contains_shared_activities(self) -> None:
        """Requirement: support set must NEVER contain shared activities (sit, stand, walk)."""
        manifest = _make_mock_manifest()

        for room in ("A", "B"):
            support = sample_support(manifest, target_room=room, K=3, seed=42)
            activities_in_support = set(support["activity"])
            for shared_act in SHARED_ACTIVITIES:
                assert shared_act not in activities_in_support, (
                    f"Shared activity {shared_act!r} found in support set for room {room}!"
                )

    def test_support_contains_only_target_exclusive_activities(self) -> None:
        """Room A exclusive: fall, sitdown, standup; Room B exclusive: empty, lying."""
        manifest = _make_mock_manifest()

        # Target Room A
        support_a = sample_support(manifest, target_room="A", K=3, seed=42)
        assert set(support_a["activity"]) == set(ROOM_A_EXCLUSIVE)
        for r in support_a:
            assert r.room == "A"
            assert r.activity in ROOM_A_EXCLUSIVE

        # Target Room B
        support_b = sample_support(manifest, target_room="B", K=3, seed=42)
        assert set(support_b["activity"]) == set(ROOM_B_EXCLUSIVE)
        for r in support_b:
            assert r.room == "B"
            assert r.activity in ROOM_B_EXCLUSIVE

    def test_one_capture_counts_as_one_shot(self) -> None:
        """Requirement: one capture counts as one shot (distinct capture_ids)."""
        manifest = _make_mock_manifest()

        # Add duplicate window entries for certain captures to simulate window-level records
        duplicated_manifest = list(manifest)
        for row in manifest[:20]:
            # Duplicate the row 4 times
            for _ in range(4):
                duplicated_manifest.append(row)

        for K in (1, 2, 5):
            support = sample_support(duplicated_manifest, target_room="A", K=K, seed=42)
            # Distinct capture_ids
            unique_cids = set(support.capture_ids)
            assert len(unique_cids) == K * len(ROOM_A_EXCLUSIVE)
            assert len(support) == K * len(ROOM_A_EXCLUSIVE)

            # Per activity count
            by_act = {}
            for r in support:
                by_act[r.activity] = by_act.get(r.activity, 0) + 1
            for act in ROOM_A_EXCLUSIVE:
                assert by_act[act] == K

    def test_deterministic_sampling_with_seed(self) -> None:
        manifest = _make_mock_manifest()
        s1 = sample_support(manifest, "A", K=3, seed=123)
        s2 = sample_support(manifest, "A", K=3, seed=123)
        s3 = sample_support(manifest, "A", K=3, seed=456)

        assert [r.capture_id for r in s1] == [r.capture_id for r in s2]
        assert [r.capture_id for r in s1] != [r.capture_id for r in s3]

    def test_insufficient_captures_raises_value_error(self) -> None:
        manifest = _make_mock_manifest()
        # Mock manifest has 10 dev captures per class. Asking for K=15 must raise.
        with pytest.raises(ValueError, match="Cannot sample K=15"):
            sample_support(manifest, "A", K=15, seed=42)

    def test_invalid_target_room_raises_value_error(self) -> None:
        manifest = _make_mock_manifest()
        with pytest.raises(ValueError, match="Invalid target_room"):
            sample_support(manifest, "C", K=1, seed=42)

    def test_dataframe_input_and_output(self) -> None:
        import pandas as pd
        manifest = _make_mock_manifest()
        df_manifest = pd.DataFrame([r.to_dict() for r in manifest])

        # Input as DataFrame, output as SupportSet
        support = sample_support(df_manifest, "B", K=2, seed=42)
        assert isinstance(support, SupportSet)
        assert len(support) == 2 * len(ROOM_B_EXCLUSIVE)

        # Output as DataFrame
        df_support = sample_support(df_manifest, "B", K=2, seed=42, as_dataframe=True)
        assert isinstance(df_support, pd.DataFrame)
        assert len(df_support) == 2 * len(ROOM_B_EXCLUSIVE)
        assert (df_support["split"] == "dev").all()


# ---------------------------------------------------------------------------
# Tests: PrototypeHead
# ---------------------------------------------------------------------------

class TestPrototypeHead:
    def test_source_prototypes_mean_calculation(self) -> None:
        """Source prototypes: exact mean embedding per class."""
        X_src = np.array([
            [1.0, 2.0],
            [3.0, 4.0],  # class 'sit' -> mean [2.0, 3.0]
            [10.0, 20.0],
            [30.0, 40.0],  # class 'stand' -> mean [20.0, 30.0]
        ], dtype=np.float32)
        y_src = ["sit", "sit", "stand", "stand"]

        protos = compute_source_prototypes(X_src, y_src)
        np.testing.assert_allclose(protos["sit"], [2.0, 3.0])
        np.testing.assert_allclose(protos["stand"], [20.0, 30.0])

    def test_target_exclusive_hierarchical_averaging(self) -> None:
        """Target-exclusive prototypes: average windows within each capture first, then across captures."""
        # Class 'fall' has 2 captures:
        # Capture A: 10 windows with value 0.0
        # Capture B: 2 windows with value 10.0
        #
        # Naive window average: (10 * 0 + 2 * 10) / 12 = 1.6667
        # Hierarchical capture average: (mean(Capture A) + mean(Capture B)) / 2
        #                             = (0.0 + 10.0) / 2 = 5.0
        X_cap_a = np.zeros((10, 2), dtype=np.float32)
        X_cap_b = np.full((2, 2), 10.0, dtype=np.float32)

        X_tgt = np.vstack([X_cap_a, X_cap_b])
        y_tgt = ["fall"] * 12
        cids = ["cap_a"] * 10 + ["cap_b"] * 2

        protos = compute_target_exclusive_prototypes(X_tgt, y_tgt, capture_ids=cids)
        np.testing.assert_allclose(
            protos["fall"],
            [5.0, 5.0],
            err_msg="Hierarchical capture average must weight each capture equally (one capture = one shot)!",
        )

    def test_masking_predictions_to_target_room_classes(self) -> None:
        """Requirement: mask predictions strictly to target room's valid classes."""
        dim = 8
        # Source room B data has: empty, lying, sit, stand, walk
        source_labels = ["empty", "lying", "sit", "stand", "walk"]
        source_embs = np.random.RandomState(42).randn(len(source_labels), dim).astype(np.float32)

        # Target room A support has: fall, sitdown, standup
        target_labels = ["fall", "sitdown", "standup"]
        target_embs = np.random.RandomState(43).randn(len(target_labels), dim).astype(np.float32)
        target_cids = ["c1", "c2", "c3"]

        head_a = PrototypeHead(target_room="A", metric="gaussian")
        head_a.fit(
            source_embeddings=source_embs,
            source_labels=source_labels,
            target_support_embeddings=target_embs,
            target_support_labels=target_labels,
            target_support_capture_ids=target_cids,
        )

        # Classes must be strictly the 6 valid classes for Room A
        assert set(head_a.classes_) == set(ROOM_A_ACTIVITIES)
        assert "empty" not in head_a.classes_
        assert "lying" not in head_a.classes_

        # Feed query exactly equal to the 'empty' source embedding
        empty_query = source_embs[0]
        pred_label, conf = head_a.predict(empty_query)
        assert pred_label in ROOM_A_ACTIVITIES
        assert pred_label not in ("empty", "lying")
        assert 0.0 <= conf <= 1.0

        # Now test target Room B
        # Source room A data has: fall, sitdown, standup, sit, stand, walk
        source_labels_b = ["fall", "sitdown", "standup", "sit", "stand", "walk"]
        source_embs_b = np.random.RandomState(44).randn(len(source_labels_b), dim).astype(np.float32)

        # Target room B support has: empty, lying
        target_labels_b = ["empty", "lying"]
        target_embs_b = np.random.RandomState(45).randn(len(target_labels_b), dim).astype(np.float32)
        target_cids_b = ["cb1", "cb2"]

        head_b = PrototypeHead(target_room="B", metric="cosine")
        head_b.fit(
            source_embeddings=source_embs_b,
            source_labels=source_labels_b,
            target_support_embeddings=target_embs_b,
            target_support_labels=target_labels_b,
            target_support_capture_ids=target_cids_b,
        )

        assert set(head_b.classes_) == set(ROOM_B_ACTIVITIES)
        assert "fall" not in head_b.classes_
        assert "sitdown" not in head_b.classes_
        assert "standup" not in head_b.classes_

        fall_query = source_embs_b[0]
        pred_label_b, conf_b = head_b.predict(fall_query)
        assert pred_label_b in ROOM_B_ACTIVITIES
        assert pred_label_b not in ("fall", "sitdown", "standup")
        assert 0.0 <= conf_b <= 1.0

    def test_gaussian_classifier_with_alpha_covariance(self) -> None:
        """Gaussian classifier with pooled shrunken covariance (alpha configurable)."""
        rng = np.random.RandomState(42)
        dim = 16
        n_samples = 40

        source_labels = ["sit", "stand", "walk"] * (n_samples // 3)
        source_embs = rng.randn(len(source_labels), dim).astype(np.float32)

        target_labels = ["fall", "sitdown", "standup"] * 2
        target_embs = rng.randn(len(target_labels), dim).astype(np.float32)
        target_cids = [f"cid_{i}" for i in range(len(target_labels))]

        # Test varying alpha values
        for alpha in (0.0, 0.25, 0.75, 1.0):
            head = PrototypeHead(target_room="A", metric="gaussian", alpha=alpha)
            head.fit(
                source_embeddings=source_embs,
                source_labels=source_labels,
                target_support_embeddings=target_embs,
                target_support_labels=target_labels,
                target_support_capture_ids=target_cids,
            )
            assert head.cov_ is not None
            assert head.cov_.shape == (dim, dim)

            # Predict query
            query = rng.randn(dim).astype(np.float32)
            label, conf = head.predict(query)
            assert label in ROOM_A_ACTIVITIES
            assert 0.0 <= conf <= 1.0

        # Test Ledoit-Wolf shrinkage
        head_lw = PrototypeHead(target_room="A", metric="gaussian", shrinkage="ledoit_wolf")
        head_lw.fit(
            source_embeddings=source_embs,
            source_labels=source_labels,
            target_support_embeddings=target_embs,
            target_support_labels=target_labels,
            target_support_capture_ids=target_cids,
        )
        assert head_lw.cov_ is not None

    def test_cosine_variant(self) -> None:
        """Cosine variant nearest centroid classification."""
        dim = 4
        # Prototypes aligned with canonical axes
        prototypes = {
            "sit": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "stand": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "walk": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "fall": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "sitdown": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
            "standup": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
        }

        head = PrototypeHead(target_room="A", metric="cosine", prototypes=prototypes)

        # Query pointing along axis 3 (fall)
        q_fall = np.array([0.1, 0.1, 0.1, 0.9], dtype=np.float32)
        label, conf = head.predict(q_fall)
        assert label == "fall"
        assert conf > 0.3

        # Query pointing along axis 0 (sit)
        q_sit = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        label, conf = head.predict(q_sit)
        assert label == "sit"

    def test_predict_single_and_batch(self) -> None:
        """predict(X) -> (label, confidence) for single vector or batch."""
        dim = 8
        prototypes = {act: np.random.randn(dim).astype(np.float32) for act in ROOM_A_ACTIVITIES}
        head = PrototypeHead(target_room="A", metric="gaussian", prototypes=prototypes)

        # 1D single query
        q_1d = np.random.randn(dim).astype(np.float32)
        label, conf = head.predict(q_1d)
        assert isinstance(label, str)
        assert isinstance(conf, float)
        assert label in ROOM_A_ACTIVITIES
        assert 0.0 <= conf <= 1.0

        # 2D batch query
        q_2d = np.random.randn(5, dim).astype(np.float32)
        labels, confs = head.predict(q_2d)
        assert isinstance(labels, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert len(labels) == 5
        assert len(confs) == 5
        for l, c in zip(labels, confs):
            assert l in ROOM_A_ACTIVITIES
            assert 0.0 <= c <= 1.0

        # Probabilities sum to 1.0
        probs = head.predict_proba(q_2d)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5)

    def test_sample_support_k_zero(self) -> None:
        """K=0 returns an empty support set without error."""
        manifest = _make_mock_manifest()
        support = sample_support(manifest, target_room="A", K=0, seed=42)
        assert len(support) == 0
        assert isinstance(support, SupportSet)

    def test_sample_support_negative_k_raises(self) -> None:
        """K < 0 raises ValueError."""
        manifest = _make_mock_manifest()
        with pytest.raises(ValueError, match="K must be non-negative"):
            sample_support(manifest, target_room="A", K=-1, seed=42)

    def test_sample_support_empty_manifest_raises(self) -> None:
        """Empty manifest with K > 0 raises ValueError (no dev captures available)."""
        with pytest.raises(ValueError, match="Cannot sample K=1 captures"):
            sample_support([], target_room="A", K=1, seed=42)


class TestEdgeCasesAndValidation:
    """Tests for edge cases and invalid inputs across fewshot module."""

    def test_prototype_head_predict_before_fit_raises(self) -> None:
        """Calling predict or predict_proba before fit raises RuntimeError."""
        head = PrototypeHead(target_room="A")
        q = np.ones((8,), dtype=np.float32)

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            head.predict_proba(q)

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            head.predict(q)

    def test_prototype_head_invalid_room_raises(self) -> None:
        """Initializing PrototypeHead with invalid target_room raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_room"):
            PrototypeHead(target_room="INVALID_ROOM")

    def test_prototype_head_feature_dim_mismatch_raises(self) -> None:
        """Query with mismatched feature dimension raises ValueError."""
        dim = 8
        prototypes = {act: np.ones(dim, dtype=np.float32) for act in ROOM_A_ACTIVITIES}
        head = PrototypeHead(target_room="A", prototypes=prototypes)

        # Query has dimension 16 instead of 8
        bad_query = np.ones((16,), dtype=np.float32)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            head.predict(bad_query)

    def test_prototype_head_3d_query_raises(self) -> None:
        """3D query input raises ValueError."""
        dim = 8
        prototypes = {act: np.ones(dim, dtype=np.float32) for act in ROOM_A_ACTIVITIES}
        head = PrototypeHead(target_room="A", prototypes=prototypes)

        bad_query = np.ones((2, 5, dim), dtype=np.float32)
        with pytest.raises(ValueError, match="X must be 1D or 2D"):
            head.predict(bad_query)

    def test_compute_source_prototypes_mismatched_lengths_raises(self) -> None:
        """Mismatched labels and embeddings lengths raise ValueError."""
        X = np.ones((5, 4), dtype=np.float32)
        y = ["sit", "stand"]  # length 2 != 5
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_source_prototypes(X, y)

    def test_compute_source_prototypes_1d_embeddings_raises(self) -> None:
        """1D embeddings input raises ValueError."""
        X = np.ones(4, dtype=np.float32)
        y = ["sit"]
        with pytest.raises(ValueError, match="embeddings must be 2D"):
            compute_source_prototypes(X, y)

    def test_compute_target_exclusive_prototypes_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths in target exclusive prototypes raise ValueError."""
        X = np.ones((4, 4), dtype=np.float32)
        y = ["fall", "fall"]
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_target_exclusive_prototypes(X, y)

        y_corr = ["fall"] * 4
        cids_bad = ["c1", "c2"]
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_target_exclusive_prototypes(X, y_corr, capture_ids=cids_bad)

