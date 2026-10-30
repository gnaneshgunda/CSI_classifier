"""
tests/test_baseline_har.py
==========================
Unit tests for scripts/baseline_har.py.

Verifies:
1. Packet rate helper:
   - Derives packet rate from config path string ("30ms" -> 33.33 Hz, "10ms" -> 100 Hz, etc.)
   - Respects explicit packet_rate configuration attributes when present
   - Graceful fallback on unrecognized paths or missing config attributes
2. Classifier factory:
   - LogisticRegression, SVM (RBF), and RandomForestClassifier instantiated with correct parameters
   - Classifiers can fit and predict on synthetic features
3. Scoring and metrics:
   - macro_f1 and balanced_accuracy calculated correctly
   - Zero-division cases handled gracefully
4. Dev-only training and validation evaluation (no data leakage):
   - StandardScaler fitted strictly on dev data, applied to val data
   - Model fitted strictly on dev (X_dev, y_dev), evaluated strictly on val (X_val, y_val)
   - Val labels with unseen classes safely filtered out without crashing
   - Empty dev or empty val handled gracefully (returns empty list)
5. Loading windows for split:
   - Filtering by split ("dev" vs "val") and optional room ("A", "B", or None)
   - Missing window files on disk skipped with warning rather than crashing
   - Empty queries return properly shaped empty arrays
6. End-to-end evaluation:
   - run_evaluation creates CSV report with exact schema:
     experiment, target_room, K, metric, value
   - Covers per-room ("A", "B") and joint ("joint") evaluations across all three models
   - K is always -1 (sentinel)
   - All metric values are valid floats in [0, 1]
   - Custom output path support with automatic parent directory creation
7. Built-in smoke test and summary display:
   - _run_smoke executes end-to-end on synthetic data
   - _print_summary prints formatted output table
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baseline_har import (
    _DEFAULT_PACKET_RATE_HZ,
    _load_windows_for_split,
    _make_classifiers,
    _print_summary,
    _run_one_room,
    _run_smoke,
    _score,
    cfg_packet_rate,
    run_evaluation,
)
from src.features import FEATURE_DIM


# ===========================================================================
# 1. Packet Rate Helper
# ===========================================================================

class TestPacketRateHelper:
    """Tests for cfg_packet_rate."""

    def test_packet_rate_30ms(self) -> None:
        """Paths containing '30ms' infer 1000/30 ~ 33.333 Hz."""
        cfg = SimpleNamespace(paths=SimpleNamespace(room_a_pcap_dir="/data/capture_30ms/pcap"))
        rate = cfg_packet_rate(cfg)
        assert rate == pytest.approx(1000.0 / 30.0, rel=1e-5)

    def test_packet_rate_10ms(self) -> None:
        """Paths containing '10ms' infer 100.0 Hz."""
        cfg = SimpleNamespace(paths=SimpleNamespace(room_a_pcap_dir="/data/capture_10ms/pcap"))
        rate = cfg_packet_rate(cfg)
        assert rate == pytest.approx(100.0, rel=1e-5)

    def test_packet_rate_50ms(self) -> None:
        """Paths containing '50ms' infer 20.0 Hz."""
        cfg = SimpleNamespace(paths=SimpleNamespace(room_a_pcap_dir="/data/capture_50ms/pcap"))
        rate = cfg_packet_rate(cfg)
        assert rate == pytest.approx(20.0, rel=1e-5)

    def test_packet_rate_explicit_attribute(self) -> None:
        """Explicit cfg.packet_rate overrides path inspection."""
        cfg = SimpleNamespace(
            packet_rate=50.0,
            paths=SimpleNamespace(room_a_pcap_dir="/data/capture_30ms/pcap"),
        )
        assert cfg_packet_rate(cfg) == pytest.approx(50.0)

    def test_packet_rate_section_attribute(self) -> None:
        """Explicit cfg.features.packet_rate overrides path inspection."""
        cfg = SimpleNamespace(
            features=SimpleNamespace(packet_rate=60.0),
            paths=SimpleNamespace(room_a_pcap_dir="/data/capture_30ms/pcap"),
        )
        assert cfg_packet_rate(cfg) == pytest.approx(60.0)

    def test_packet_rate_fallback_unrecognized(self) -> None:
        """Paths without 10ms/30ms/50ms fallback to default ~33.333 Hz."""
        cfg = SimpleNamespace(paths=SimpleNamespace(room_a_pcap_dir="/data/custom_rate/pcap"))
        rate = cfg_packet_rate(cfg)
        assert rate == pytest.approx(_DEFAULT_PACKET_RATE_HZ, rel=1e-5)

    def test_packet_rate_fallback_exception(self) -> None:
        """Missing or malformed cfg objects return default rate safely."""
        assert cfg_packet_rate(None) == pytest.approx(_DEFAULT_PACKET_RATE_HZ, rel=1e-5)
        assert cfg_packet_rate(object()) == pytest.approx(_DEFAULT_PACKET_RATE_HZ, rel=1e-5)


# ===========================================================================
# 2. Classifier Factory
# ===========================================================================

class TestClassifierFactory:
    """Tests for _make_classifiers instantiation and training."""

    def test_make_classifiers_keys_and_types(self) -> None:
        """_make_classifiers returns LR, SVM, and RF estimators."""
        clfs = _make_classifiers(n_jobs=2)
        assert set(clfs.keys()) == {"LR", "SVM", "RF"}

        lr = clfs["LR"]
        assert isinstance(lr, LogisticRegression)
        assert lr.C == 1.0
        assert lr.max_iter == 2000
        assert lr.solver == "lbfgs"

        svm = clfs["SVM"]
        assert isinstance(svm, SVC)
        assert svm.kernel == "rbf"
        assert svm.C == 10.0
        assert svm.gamma == "scale"

        rf = clfs["RF"]
        assert isinstance(rf, RandomForestClassifier)
        assert rf.n_estimators == 300
        assert rf.n_jobs == 2

    def test_classifiers_fit_and_predict(self) -> None:
        """All three classifiers can fit and predict on synthetic feature matrices."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, FEATURE_DIM)).astype(np.float32)
        y = np.array(["sit"] * 10 + ["stand"] * 10, dtype=object)

        clfs = _make_classifiers(n_jobs=1)
        for name, clf in clfs.items():
            clf.fit(X, y)
            preds = clf.predict(X[:4])
            assert len(preds) == 4
            assert all(p in {"sit", "stand"} for p in preds)


# ===========================================================================
# 3. Scoring and Metrics
# ===========================================================================

class TestScoring:
    """Tests for _score calculation."""

    def test_score_perfect_predictions(self) -> None:
        """Identical predictions give macro_f1 = 1.0 and balanced_accuracy = 1.0."""
        y_true = np.array(["fall", "sit", "walk"], dtype=object)
        y_pred = np.array(["fall", "sit", "walk"], dtype=object)

        scores = _score(y_true, y_pred)
        assert scores["macro_f1"] == pytest.approx(1.0)
        assert scores["balanced_accuracy"] == pytest.approx(1.0)

    def test_score_imbalanced_predictions(self) -> None:
        """Checks balanced accuracy computation on imbalanced test set."""
        y_true = np.array(["sit", "sit", "sit", "walk"], dtype=object)
        y_pred = np.array(["sit", "sit", "sit", "sit"], dtype=object)

        scores = _score(y_true, y_pred)
        # Recall: sit=1.0, walk=0.0 -> balanced accuracy = 0.5
        assert scores["balanced_accuracy"] == pytest.approx(0.5)
        assert 0.0 < scores["macro_f1"] < 1.0

    def test_score_zero_division(self) -> None:
        """Completely incorrect predictions handle zero division without error."""
        y_true = np.array(["sit", "sit"], dtype=object)
        y_pred = np.array(["walk", "walk"], dtype=object)

        scores = _score(y_true, y_pred)
        assert scores["macro_f1"] == pytest.approx(0.0)
        assert scores["balanced_accuracy"] == pytest.approx(0.0)


# ===========================================================================
# 4. Dev-Only Training and Validation Evaluation
# ===========================================================================

class TestRunOneRoomEvaluation:
    """Tests for _run_one_room logic, leakage prevention, and error resilience."""

    @pytest.fixture
    def synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate small distinct dev and val datasets."""
        rng = np.random.default_rng(101)
        # Dev: 16 samples, 2 classes
        X_dev = rng.normal(loc=1.0, scale=2.0, size=(16, FEATURE_DIM)).astype(np.float32)
        y_dev = np.array(["sit"] * 8 + ["walk"] * 8, dtype=object)

        # Val: 8 samples, same 2 classes
        X_val = rng.normal(loc=1.5, scale=2.0, size=(8, FEATURE_DIM)).astype(np.float32)
        y_val = np.array(["sit"] * 4 + ["walk"] * 4, dtype=object)
        return X_dev, y_dev, X_val, y_val

    def test_run_one_room_returns_expected_rows(
        self, synthetic_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """_run_one_room returns 2 rows (macro_f1 and balanced_accuracy) matching schema."""
        X_dev, y_dev, X_val, y_val = synthetic_data
        clf = LogisticRegression(max_iter=100, random_state=42)

        rows = _run_one_room(
            name="LR",
            clf=clf,
            X_dev=X_dev,
            y_dev=y_dev,
            X_val=X_val,
            y_val=y_val,
            target_room="A",
        )

        assert len(rows) == 2
        metrics = {r["metric"] for r in rows}
        assert metrics == {"macro_f1", "balanced_accuracy"}

        for r in rows:
            assert r["experiment"] == "LR"
            assert r["target_room"] == "A"
            assert r["K"] == -1
            val = float(r["value"])
            assert 0.0 <= val <= 1.0

    def test_dev_only_training_no_leakage(
        self, synthetic_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Scaling and model parameters are fit ONLY on dev data."""
        X_dev, y_dev, X_val, y_val = synthetic_data

        # Train model with val1
        clf1 = LogisticRegression(max_iter=100, random_state=42)
        _run_one_room("LR", clf1, X_dev, y_dev, X_val, y_val, "A")

        # Train model with completely different val2 (same dev)
        X_val_extreme = X_val * 1000.0 + 500.0
        clf2 = LogisticRegression(max_iter=100, random_state=42)
        _run_one_room("LR", clf2, X_dev, y_dev, X_val_extreme, y_val, "A")

        # The coefficients of clf1 and clf2 must be strictly identical because
        # training happens exclusively on X_dev
        np.testing.assert_allclose(clf1.coef_, clf2.coef_, rtol=1e-5)
        np.testing.assert_allclose(clf1.intercept_, clf2.intercept_, rtol=1e-5)

    def test_unseen_val_labels_filtered_safely(
        self, synthetic_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """When validation contains labels not in dev, unseen labels are excluded without crashing."""
        X_dev, y_dev, X_val, y_val = synthetic_data

        # Add unseen label "fall" to validation
        y_val_with_unseen = np.copy(y_val)
        y_val_with_unseen[0] = "fall"

        clf = LogisticRegression(max_iter=100, random_state=42)
        rows = _run_one_room(
            name="LR",
            clf=clf,
            X_dev=X_dev,
            y_dev=y_dev,
            X_val=X_val,
            y_val=y_val_with_unseen,
            target_room="A",
        )

        assert len(rows) == 2
        for r in rows:
            val = float(r["value"])
            assert 0.0 <= val <= 1.0

    def test_empty_dev_skips_gracefully(self) -> None:
        """If X_dev is empty, returns empty list without error."""
        clf = LogisticRegression()
        rows = _run_one_room(
            name="LR",
            clf=clf,
            X_dev=np.empty((0, FEATURE_DIM), dtype=np.float32),
            y_dev=np.empty(0, dtype=object),
            X_val=np.ones((5, FEATURE_DIM), dtype=np.float32),
            y_val=np.array(["sit"] * 5, dtype=object),
            target_room="A",
        )
        assert rows == []

    def test_empty_val_skips_gracefully(self) -> None:
        """If X_val is empty, returns empty list without error."""
        clf = LogisticRegression()
        rows = _run_one_room(
            name="LR",
            clf=clf,
            X_dev=np.ones((5, FEATURE_DIM), dtype=np.float32),
            y_dev=np.array(["sit"] * 5, dtype=object),
            X_val=np.empty((0, FEATURE_DIM), dtype=np.float32),
            y_val=np.empty(0, dtype=object),
            target_room="A",
        )
        assert rows == []

    def test_all_unseen_val_skips_gracefully(self) -> None:
        """If all validation labels are unseen in dev, returns empty list without crashing."""
        clf = LogisticRegression()
        rows = _run_one_room(
            name="LR",
            clf=clf,
            X_dev=np.ones((10, FEATURE_DIM), dtype=np.float32),
            y_dev=np.array(["sit"] * 5 + ["walk"] * 5, dtype=object),
            X_val=np.ones((4, FEATURE_DIM), dtype=np.float32),
            y_val=np.array(["empty"] * 4, dtype=object),  # entirely unseen in dev
            target_room="A",
        )
        assert rows == []


# ===========================================================================
# 5. Loading Windows for Split
# ===========================================================================

class TestLoadWindowsForSplit:
    """Tests for _load_windows_for_split."""

    @pytest.fixture
    def mock_windows_dataset(self, tmp_path: Path) -> Tuple[pd.DataFrame, Path]:
        """Create a mock filesystem with windows and index dataframe."""
        windows_dir = tmp_path / "windows"
        dev_dir = windows_dir / "dev"
        val_dir = windows_dir / "val"
        dev_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        rows = []
        for i in range(4):
            wid = f"win_{i}"
            split = "dev" if i < 2 else "val"
            room = "A" if i % 2 == 0 else "B"
            act = "sit" if i < 2 else "walk"

            # Create dummy window (150, 256)
            arr = np.full((150, 256), float(i + 1), dtype=np.float32)
            np.save(str(windows_dir / split / f"{wid}.npy"), arr)

            rows.append({
                "window_id": wid,
                "capture_id": f"cap_{i}",
                "room": room,
                "activity": act,
                "split": split,
            })

        df = pd.DataFrame(rows)
        return df, windows_dir

    def test_load_all_for_split(self, mock_windows_dataset: Tuple[pd.DataFrame, Path]) -> None:
        """Loads all windows belonging to split when room=None."""
        df, windows_dir = mock_windows_dataset
        X, y, rooms = _load_windows_for_split(
            index_df=df,
            split="dev",
            windows_dir=windows_dir,
            packet_rate=30.0,
            room=None,
        )

        assert X.shape == (2, FEATURE_DIM)
        assert len(y) == 2
        assert len(rooms) == 2
        assert list(rooms) == ["A", "B"]

    def test_load_filter_by_room(self, mock_windows_dataset: Tuple[pd.DataFrame, Path]) -> None:
        """Filters to a single room when room is specified."""
        df, windows_dir = mock_windows_dataset
        X, y, rooms = _load_windows_for_split(
            index_df=df,
            split="dev",
            windows_dir=windows_dir,
            packet_rate=30.0,
            room="A",
        )

        assert X.shape == (1, FEATURE_DIM)
        assert len(y) == 1
        assert rooms == ["A"]

    def test_missing_files_skipped_gracefully(self, mock_windows_dataset: Tuple[pd.DataFrame, Path]) -> None:
        """If a window file listed in index doesn't exist on disk, skips without crashing."""
        df, windows_dir = mock_windows_dataset
        # Delete one window file
        (windows_dir / "dev" / "win_0.npy").unlink()

        X, y, rooms = _load_windows_for_split(
            index_df=df,
            split="dev",
            windows_dir=windows_dir,
            packet_rate=30.0,
            room=None,
        )

        assert X.shape == (1, FEATURE_DIM)
        assert len(y) == 1
        assert rooms == ["B"]

    def test_empty_result_on_no_match(self, mock_windows_dataset: Tuple[pd.DataFrame, Path]) -> None:
        """Returns empty float32 array with shape (0, FEATURE_DIM) when no match found."""
        df, windows_dir = mock_windows_dataset
        X, y, rooms = _load_windows_for_split(
            index_df=df,
            split="dev",
            windows_dir=windows_dir,
            packet_rate=30.0,
            room="NON_EXISTENT",
        )

        assert X.shape == (0, FEATURE_DIM)
        assert X.dtype == np.float32
        assert len(y) == 0
        assert rooms == []


# ===========================================================================
# 6. End-to-End Evaluation & CSV Output
# ===========================================================================

class TestRunEvaluationEndToEnd:
    """Tests for run_evaluation end-to-end execution, CSV schema, and per-room/joint rows."""

    @pytest.fixture
    def synthetic_har_environment(self, tmp_path: Path) -> Tuple[Path, Path, Path]:
        """Build small synthetic windows, index CSV, and reports directory."""
        windows_dir = tmp_path / "windows"
        reports_dir = tmp_path / "reports"
        index_csv = tmp_path / "windows_index.csv"

        (windows_dir / "dev").mkdir(parents=True)
        (windows_dir / "val").mkdir(parents=True)

        rng = np.random.default_rng(202)
        T, N_SUB = 32, 256  # small T for fast test

        # 2 rooms, 2 activities each, 3 dev + 2 val per activity
        index_rows = []
        wid = 0

        setup = [
            ("A", ["sit", "walk"]),
            ("B", ["sit", "lying"]),
        ]

        for room, activities in setup:
            for act in activities:
                # Dev
                for _ in range(3):
                    wname = f"w_{wid:04d}"
                    arr = rng.uniform(5.0, 50.0, size=(T, N_SUB)).astype(np.float32)
                    np.save(str(windows_dir / "dev" / f"{wname}.npy"), arr)
                    index_rows.append({
                        "window_id": wname,
                        "capture_id": f"c_{wid}",
                        "room": room,
                        "activity": act,
                        "split": "dev",
                        "start": 0,
                        "config_hash": "testhash",
                    })
                    wid += 1

                # Val
                for _ in range(2):
                    wname = f"w_{wid:04d}"
                    arr = rng.uniform(5.0, 50.0, size=(T, N_SUB)).astype(np.float32)
                    np.save(str(windows_dir / "val" / f"{wname}.npy"), arr)
                    index_rows.append({
                        "window_id": wname,
                        "capture_id": f"c_{wid}",
                        "room": room,
                        "activity": act,
                        "split": "val",
                        "start": 0,
                        "config_hash": "testhash",
                    })
                    wid += 1

        pd.DataFrame(index_rows).to_csv(str(index_csv), index=False)
        return index_csv, windows_dir, reports_dir

    def test_run_evaluation_complete(
        self, synthetic_har_environment: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_evaluation generates CSV with LR, SVM, RF across Room A, Room B, and joint."""
        index_csv, windows_dir, reports_dir = synthetic_har_environment

        # Monkeypatch _make_classifiers to use ultra-fast estimators
        def _fast_classifiers(n_jobs: int = 1) -> Dict[str, object]:
            return {
                "LR": LogisticRegression(C=1.0, max_iter=50, random_state=42),
                "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
                "RF": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=n_jobs),
            }

        monkeypatch.setattr("scripts.baseline_har._make_classifiers", _fast_classifiers)

        out_csv = run_evaluation(
            index_csv=index_csv,
            windows_dir=windows_dir,
            packet_rate=30.0,
            reports_dir=reports_dir,
            n_jobs=1,
        )

        assert out_csv.exists()
        assert out_csv == reports_dir / "baseline_har_results.csv"

        # Read CSV
        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Expected schema: experiment, target_room, K, metric, value
        assert reader.fieldnames == ["experiment", "target_room", "K", "metric", "value"]

        # Models: 3 ("LR", "SVM", "RF")
        # Rooms: 3 ("A", "B", "joint")
        # Metrics: 2 ("macro_f1", "balanced_accuracy")
        # Total rows = 3 * 3 * 2 = 18
        assert len(rows) == 18

        experiments = {r["experiment"] for r in rows}
        rooms = {r["target_room"] for r in rows}
        metrics = {r["metric"] for r in rows}

        assert experiments == {"LR", "SVM", "RF"}
        assert rooms == {"A", "B", "joint"}
        assert metrics == {"macro_f1", "balanced_accuracy"}

        for r in rows:
            assert int(r["K"]) == -1
            score = float(r["value"])
            assert 0.0 <= score <= 1.0

    def test_run_evaluation_custom_out_csv(
        self,
        synthetic_har_environment: Tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Custom out_csv argument writes directly to target path, creating parent dir if needed."""
        index_csv, windows_dir, reports_dir = synthetic_har_environment

        def _fast_classifiers(n_jobs: int = 1) -> Dict[str, object]:
            return {
                "LR": LogisticRegression(max_iter=50, random_state=42),
            }

        monkeypatch.setattr("scripts.baseline_har._make_classifiers", _fast_classifiers)

        custom_path = tmp_path / "custom_nested_dir" / "my_results.csv"
        out_csv = run_evaluation(
            index_csv=index_csv,
            windows_dir=windows_dir,
            packet_rate=30.0,
            reports_dir=reports_dir,
            out_csv=custom_path,
            n_jobs=1,
        )

        assert out_csv == custom_path
        assert custom_path.exists()


# ===========================================================================
# 7. Smoke and Summary Display
# ===========================================================================

class TestSmokeAndSummaryDisplay:
    """Tests for _run_smoke CLI functionality and _print_summary table formatting."""

    def test_print_summary(self, capsys: pytest.CaptureFixture) -> None:
        """_print_summary formats and prints rows without exception."""
        sample_rows = [
            {"experiment": "LR", "target_room": "A", "K": -1, "metric": "macro_f1", "value": "0.8500"},
            {"experiment": "LR", "target_room": "joint", "K": -1, "metric": "balanced_accuracy", "value": "0.9000"},
        ]
        _print_summary(sample_rows)
        captured = capsys.readouterr()
        assert "Baseline HAR results" in captured.out
        assert "LR" in captured.out
        assert "macro_f1" in captured.out
        assert "0.8500" in captured.out

    def test_run_smoke_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_run_smoke runs the built-in synthetic smoke test cleanly."""
        # Monkeypatch RF to be fast so test finishes in <1s
        original_make = _make_classifiers

        def _fast_make(n_jobs: int = 1):
            clfs = original_make(n_jobs=n_jobs)
            clfs["RF"].n_estimators = 5
            clfs["LR"].max_iter = 100
            return clfs

        monkeypatch.setattr("scripts.baseline_har._make_classifiers", _fast_make)

        args = argparse.Namespace(smoke=True, workers=1)
        # Should execute and complete without error
        _run_smoke(args)
