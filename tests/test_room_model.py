"""
tests/test_room_model.py
========================
Unit tests for src/room_model.py.

Verifies:
1. Feature extraction:
   - 512-dimensional output (256 mean + 256 std)
   - Numerical correctness with and without log1p
   - Central cropping for length truncation
   - Batch transformation consistency
   - Shape guards and ValueError on invalid dimensions
2. Classifiers and initialization:
   - LogisticRegression and MLPClassifier default parameters
   - RoomModel initialization, default best model, room classes
   - Guard against calling predict before fitting
3. Training & Grouped CV (no data leakage):
   - Training strictly on development split windows
   - Scaler and model parameters fit exclusively on dev data
   - Grouped CV using GroupKFold grouped by capture_id
   - Error handling when dev windows are missing
4. Validation scoring & Headline metrics:
   - Accuracy and balanced_accuracy evaluation on val windows
   - Headline result reported at window length 150
5. Window-length study:
   - Sweeps all five required lengths: 10, 25, 50, 100, 150
   - Both LR and MLP evaluated at each length
   - Standalone window_length_study module helper
6. Predict_proba and RoomRouter compatibility:
   - Single window (T, 256), pre-extracted (512,), and batch inputs
   - Output probability distribution sums to 1.0
   - Integration with RoomRouter for downstream routing
7. Persistence:
   - save / export to checkpoints_dir
   - load restores exact predict_proba outputs
   - FileNotFoundError on missing checkpoint
8. Reporting & Visualization:
   - CSV results format: model, split, window_length, metric, value
   - Figure generation with plot_window_length_study
   - train_and_evaluate end-to-end pipeline execution
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.room_model import (
    LENGTH_STUDY_VALUES,
    ROOMS,
    RoomFeatureExtractor,
    RoomModel,
    _make_lr,
    _make_mlp,
    plot_window_length_study,
    train_and_evaluate,
    window_length_study,
)


# ===========================================================================
# 1. Feature Extractor
# ===========================================================================

class TestRoomFeatureExtractor:
    """Tests for RoomFeatureExtractor dimensions, numerical values, and slicing."""

    def test_dimensions_single_window(self) -> None:
        """transform on (T, 256) returns float32 vector of shape (512,)."""
        ext = RoomFeatureExtractor(log1p=True)
        assert ext.DIM == 512

        rng = np.random.default_rng(1)
        win = rng.uniform(0.0, 50.0, size=(150, 256)).astype(np.float32)
        feat = ext.transform(win)

        assert isinstance(feat, np.ndarray)
        assert feat.shape == (512,)
        assert feat.dtype == np.float32
        assert np.isfinite(feat).all()

    def test_dimensions_batch(self) -> None:
        """transform_batch on (N, T, 256) returns (N, 512) float32 matching row-by-row."""
        ext = RoomFeatureExtractor(log1p=True)
        rng = np.random.default_rng(2)
        batch = rng.uniform(1.0, 30.0, size=(4, 80, 256)).astype(np.float32)

        bf = ext.transform_batch(batch)
        assert bf.shape == (4, 512)
        assert bf.dtype == np.float32

        for i in range(4):
            single = ext.transform(batch[i])
            np.testing.assert_allclose(bf[i], single, rtol=1e-6, atol=1e-6)

    def test_numerical_correctness_without_log1p(self) -> None:
        """Verify exact mean and population standard deviation without log1p."""
        ext = RoomFeatureExtractor(log1p=False)
        # Create a T=2 window where row 0 is 2.0 and row 1 is 4.0
        # Mean across time is 3.0, population std is 1.0
        win = np.zeros((2, 256), dtype=np.float32)
        win[0, :] = 2.0
        win[1, :] = 4.0

        feat = ext.transform(win)
        means = feat[:256]
        stds  = feat[256:]

        np.testing.assert_allclose(means, 3.0, rtol=1e-6)
        np.testing.assert_allclose(stds, 1.0, rtol=1e-6)

    def test_numerical_correctness_with_log1p(self) -> None:
        """Verify exact mean and zero std for constant amplitude with log1p."""
        ext = RoomFeatureExtractor(log1p=True)
        c = 3.0
        win = np.full((50, 256), c, dtype=np.float32)

        feat = ext.transform(win)
        expected_log = math.log1p(c)  # ln(4.0)
        means = feat[:256]
        stds  = feat[256:]

        np.testing.assert_allclose(means, expected_log, rtol=1e-6)
        np.testing.assert_allclose(stds, 0.0, atol=1e-7)

    def test_transform_truncated_central_crop(self) -> None:
        """transform_truncated slices exactly the central `length` rows."""
        ext = RoomFeatureExtractor(log1p=False)
        T = 100
        # Each row t has value t
        row_vals = np.arange(T, dtype=np.float32)[:, None]
        win = np.repeat(row_vals, 256, axis=1)

        # For length=50 on T=100: start is (100 - 50) // 2 = 25
        # Expected rows are 25 to 74
        feat_trunc = ext.transform_truncated(win, length=50)
        feat_slice = ext.transform(win[25:75])

        np.testing.assert_allclose(feat_trunc, feat_slice, rtol=1e-6)

    def test_transform_truncated_clamping(self) -> None:
        """When length >= T, uses the full window without error."""
        ext = RoomFeatureExtractor(log1p=True)
        win = np.ones((40, 256), dtype=np.float32)

        feat_trunc = ext.transform_truncated(win, length=150)
        feat_full  = ext.transform(win)
        np.testing.assert_allclose(feat_trunc, feat_full, rtol=1e-6)

    def test_invalid_input_shapes_raise_error(self) -> None:
        """ValueError raised on invalid window dimensions."""
        ext = RoomFeatureExtractor()

        # 1D input
        with pytest.raises(ValueError, match="Expected \\(T, 256\\)"):
            ext.transform(np.zeros(256, dtype=np.float32))

        # 3D input to transform
        with pytest.raises(ValueError, match="Expected \\(T, 256\\)"):
            ext.transform(np.zeros((2, 100, 256), dtype=np.float32))

        # Subcarrier count != 256
        with pytest.raises(ValueError, match="Expected \\(T, 256\\)"):
            ext.transform(np.zeros((100, 128), dtype=np.float32))

        # 2D input to transform_batch
        with pytest.raises(ValueError, match="Expected \\(N, T, 256\\)"):
            ext.transform_batch(np.zeros((100, 256), dtype=np.float32))


# ===========================================================================
# 2. Classifiers & Initialization
# ===========================================================================

class TestClassifiersAndInitialization:
    """Tests for model constructors and RoomModel state initialization."""

    def test_make_lr_parameters(self) -> None:
        """_make_lr returns LogisticRegression with specified hyperparameters."""
        lr = _make_lr(random_state=123)
        assert isinstance(lr, LogisticRegression)
        assert lr.C == 1.0
        assert lr.max_iter == 2000
        assert lr.solver == "lbfgs"
        assert lr.random_state == 123

    def test_make_mlp_parameters(self) -> None:
        """_make_mlp returns MLPClassifier with specified architecture."""
        mlp = _make_mlp(random_state=456)
        assert isinstance(mlp, MLPClassifier)
        assert mlp.hidden_layer_sizes == (256, 128)
        assert mlp.activation == "relu"
        assert mlp.solver == "adam"
        assert mlp.early_stopping is True
        assert mlp.random_state == 456

    def test_room_model_init_defaults(self) -> None:
        """RoomModel initializes with default room lists and unfitted state."""
        model = RoomModel()
        assert model.rooms == ["A", "B"]
        assert list(model.classes_) == ["A", "B"]
        assert model.best_name == "LR"
        assert model._is_fit is False

    def test_predict_before_fit_raises(self) -> None:
        """predict and predict_proba raise RuntimeError when called before fit."""
        model = RoomModel()
        win = np.zeros((150, 256), dtype=np.float32)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(win)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict_proba(win)


# ===========================================================================
# 3. Training & Grouped Cross-Validation (Synthetic Fixture)
# ===========================================================================

@pytest.fixture
def synthetic_room_dataset(tmp_path: Path):
    """Generate small synthetic windows for Room A and Room B across dev and val splits.

    Contains 5 captures per room for dev (3 windows each = 30 dev windows, 10 groups).
    Contains 3 captures per room for val (1 canonical window each = 6 val windows).
    Also writes corresponding intermediate arrays for the window-length study.
    """
    windows_dir = tmp_path / "windows"
    inter_dir   = tmp_path / "intermediate"
    (windows_dir / "dev").mkdir(parents=True)
    (windows_dir / "val").mkdir(parents=True)
    (inter_dir / "A").mkdir(parents=True)
    (inter_dir / "B").mkdir(parents=True)

    rng = np.random.default_rng(42)
    T, N_SUB = 150, 256
    rows = []

    # Room A captures: elevated mean amplitude ~ 150
    # Room B captures: lower mean amplitude ~ 30
    room_amplitudes = {"A": 150.0, "B": 30.0}

    # Dev captures: 5 captures per room, 3 windows each
    for room, base in room_amplitudes.items():
        for c in range(5):
            cid = f"{room}_cap_{c:03d}"
            # Intermediate array (220, 256)
            inter_arr = rng.normal(loc=base, scale=5.0, size=(220, N_SUB)).astype(np.float32)
            np.save(str(inter_dir / room / f"{cid}.npy"), inter_arr)

            for w in range(3):
                wid = f"{cid}__w{w:04d}"
                win_arr = inter_arr[w * 25: w * 25 + T].copy()
                np.save(str(windows_dir / "dev" / f"{wid}.npy"), win_arr)
                rows.append({
                    "window_id": wid,
                    "capture_id": cid,
                    "room": room,
                    "activity": "walk",
                    "split": "dev",
                    "start": w * 25,
                    "config_hash": "testhash",
                })

    # Val captures: 3 captures per room, 1 canonical window each
    for room, base in room_amplitudes.items():
        for c in range(3):
            cid = f"{room}_valcap_{c:03d}"
            inter_arr = rng.normal(loc=base, scale=5.0, size=(180, N_SUB)).astype(np.float32)
            np.save(str(inter_dir / room / f"{cid}.npy"), inter_arr)

            wid = f"{cid}__w0000"
            win_arr = inter_arr[:T].copy()
            np.save(str(windows_dir / "val" / f"{wid}.npy"), win_arr)
            rows.append({
                "window_id": wid,
                "capture_id": cid,
                "room": room,
                "activity": "walk",
                "split": "val",
                "start": 0,
                "config_hash": "testhash",
            })

    df = pd.DataFrame(rows)
    return df, windows_dir, inter_dir


class TestFitAndGroupedCV:
    """Tests for dev-only training, data leakage prevention, and grouped CV."""

    def test_fit_and_dev_only_training(self, synthetic_room_dataset) -> None:
        """RoomModel fits exclusively on dev split windows."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)

        model.fit(df, windows_dir, random_state=42)
        assert model._is_fit is True
        assert "LR" in model._clfs
        assert "MLP" in model._clfs
        assert "LR" in model._scalers
        assert "MLP" in model._scalers
        assert model.best_name in ("LR", "MLP")

    def test_no_validation_leakage(self, synthetic_room_dataset) -> None:
        """Altering validation windows does not change the fitted dev scaler or model."""
        df, windows_dir, _ = synthetic_room_dataset

        model1 = RoomModel(log1p=True)
        model1.fit(df, windows_dir, random_state=42)

        # Perturb all validation files on disk
        val_df = df[df["split"] == "val"]
        for _, r in val_df.iterrows():
            wpath = windows_dir / "val" / f"{r['window_id']}.npy"
            corrupt = np.full((150, 256), 9999.0, dtype=np.float32)
            np.save(str(wpath), corrupt)

        model2 = RoomModel(log1p=True)
        model2.fit(df, windows_dir, random_state=42)

        # LR coefficients and scaler statistics must remain strictly identical
        np.testing.assert_allclose(
            model1._scalers["LR"].mean_,
            model2._scalers["LR"].mean_,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model1._clfs["LR"].coef_,
            model2._clfs["LR"].coef_,
            rtol=1e-6,
        )

    def test_grouped_cv_respects_capture_id(self, synthetic_room_dataset) -> None:
        """Internal CV correctly groups windows by capture_id so windows from the same capture stay together."""
        df, windows_dir, _ = synthetic_room_dataset
        dev_df = df[df["split"] == "dev"]

        model = RoomModel(log1p=True)
        X, y, groups = model._load_split(dev_df, windows_dir, split="dev")

        # Verify group indices map 1-to-1 to capture_id
        unique_cids = dev_df["capture_id"].unique()
        assert len(np.unique(groups)) == len(unique_cids) == 10
        # Each capture has 3 windows, so each group should have 3 entries
        for g in np.unique(groups):
            assert (groups == g).sum() == 3

    def test_empty_dev_raises_error(self, synthetic_room_dataset) -> None:
        """Raises ValueError if no development windows are present in the index."""
        df, windows_dir, _ = synthetic_room_dataset
        no_dev_df = df[df["split"] != "dev"].copy()

        model = RoomModel()
        with pytest.raises(ValueError, match="No development windows found"):
            model.fit(no_dev_df, windows_dir)


# ===========================================================================
# 4. Prediction, Probabilities & Routing Integration
# ===========================================================================

class TestPredictionAndRouting:
    """Tests for predict, predict_proba, batch inputs, and RoomRouter compatibility."""

    def test_predict_proba_single_window(self, synthetic_room_dataset) -> None:
        """predict_proba on (150, 256) returns valid probability vector shape (2,)."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        test_win = np.random.default_rng(99).normal(loc=150.0, scale=5.0, size=(150, 256)).astype(np.float32)
        proba = model.predict_proba(test_win)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (2,)
        assert proba.dtype == np.float32
        assert np.isclose(float(proba.sum()), 1.0, atol=1e-5)
        # Should be confidently Room A (index 0)
        assert proba[0] > proba[1]

        pred = model.predict(test_win)
        assert pred == "A"

    def test_predict_proba_batch_windows(self, synthetic_room_dataset) -> None:
        """predict_proba on (N, T, 256) returns shape (N, 2) float32 probabilities."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        rng = np.random.default_rng(88)
        batch = np.zeros((3, 150, 256), dtype=np.float32)
        batch[0] = rng.normal(loc=150.0, scale=5.0, size=(150, 256)).astype(np.float32)  # Room A
        batch[1] = rng.normal(loc=30.0,  scale=5.0, size=(150, 256)).astype(np.float32)  # Room B
        batch[2] = rng.normal(loc=150.0, scale=5.0, size=(150, 256)).astype(np.float32)  # Room A

        proba = model.predict_proba(batch)
        assert proba.shape == (3, 2)
        for i in range(3):
            assert np.isclose(float(proba[i].sum()), 1.0, atol=1e-5)

        preds = model.predict(batch)
        assert list(preds) == ["A", "B", "A"]

    def test_predict_proba_pre_extracted_feature(self, synthetic_room_dataset) -> None:
        """predict_proba supports precomputed 512-d feature vector input."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        rng = np.random.default_rng(77)
        test_win = rng.normal(loc=30.0, scale=5.0, size=(150, 256)).astype(np.float32)
        feat = model._extractor.transform(test_win)  # (512,)

        proba_feat = model.predict_proba(feat)
        proba_win  = model.predict_proba(test_win)
        np.testing.assert_allclose(proba_feat, proba_win, atol=1e-5)

    def test_room_router_compatibility(self, synthetic_room_dataset) -> None:
        """Verify RoomRouter from src/routing can wrap fitted RoomModel directly."""
        from src.routing import RoomRouter

        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        cfg = SimpleNamespace(
            routing=SimpleNamespace(
                mode="predicted_hard",
                confidence_threshold=0.60,
            )
        )
        router = RoomRouter(classifier=model, cfg=cfg)

        rng = np.random.default_rng(66)
        test_win_a = rng.normal(loc=150.0, scale=5.0, size=(150, 256)).astype(np.float32)
        feat_a = model._extractor.transform(test_win_a)

        res_a = router.route(feat_a)
        assert res_a.room == "A"
        assert res_a.mode_used == "predicted_hard"
        assert res_a.room_probs is not None
        assert res_a.room_probs[0] > 0.60

        test_win_b = rng.normal(loc=30.0, scale=5.0, size=(150, 256)).astype(np.float32)
        feat_b = model._extractor.transform(test_win_b)

        res_b = router.route(feat_b)
        assert res_b.room == "B"
        assert res_b.mode_used == "predicted_hard"
        assert res_b.room_probs[1] > 0.60

        # Batch routing test
        batch_feats = np.stack([feat_a, feat_b], axis=0)
        batch_results = router.route_batch(batch_feats)
        assert len(batch_results) == 2
        assert batch_results[0].room == "A"
        assert batch_results[1].room == "B"


# ===========================================================================
# 5. Validation Scoring & Window-Length Study
# ===========================================================================

class TestValidationAndWindowLengthStudy:
    """Tests for score_split and window_length_study on all five window lengths."""

    def test_score_split_val(self, synthetic_room_dataset) -> None:
        """score_split computes accuracy and balanced_accuracy for LR and MLP on val."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        scores = model.score_split(df, windows_dir, split="val")
        assert "LR" in scores and "MLP" in scores
        for name in ("LR", "MLP"):
            m = scores[name]
            assert "accuracy" in m and "balanced_accuracy" in m
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["balanced_accuracy"] <= 1.0

    def test_all_five_window_lengths_evaluated(self, synthetic_room_dataset) -> None:
        """window_length_study evaluates all 5 lengths {10, 25, 50, 100, 150} for LR & MLP."""
        df, _, inter_dir = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, _, random_state=42)

        assert LENGTH_STUDY_VALUES == [10, 25, 50, 100, 150]

        length_df = model.window_length_study(df, inter_dir)
        assert isinstance(length_df, pd.DataFrame)
        assert not length_df.empty

        expected_cols = {"model", "window_length", "accuracy", "balanced_accuracy"}
        assert set(length_df.columns) == expected_cols

        # All 5 lengths must be present
        lengths_present = sorted(length_df["window_length"].unique())
        assert lengths_present == [10, 25, 50, 100, 150]

        # Both models present for each length
        assert set(length_df["model"].unique()) == {"LR", "MLP"}
        assert len(length_df) == 2 * 5  # 10 rows

        for _, r in length_df.iterrows():
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["balanced_accuracy"] <= 1.0

    def test_standalone_window_length_study_function(self, synthetic_room_dataset, tmp_path: Path) -> None:
        """Standalone window_length_study module function runs and returns DataFrame."""
        df, windows_dir, inter_dir = synthetic_room_dataset
        index_csv = tmp_path / "windows_index.csv"
        df.to_csv(str(index_csv), index=False)

        cfg = SimpleNamespace(
            paths=SimpleNamespace(
                windows_index_csv=str(index_csv),
                intermediate_dir=str(inter_dir),
            )
        )

        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        ldf = window_length_study(cfg, model, lengths=[10, 50, 150])
        assert isinstance(ldf, pd.DataFrame)
        assert sorted(ldf["window_length"].unique()) == [10, 50, 150]

    def test_headline_accuracy_at_length_150(self, synthetic_room_dataset, tmp_path: Path) -> None:
        """Headline metric is evaluated at canonical window length 150."""
        df, windows_dir, inter_dir = synthetic_room_dataset
        index_csv = tmp_path / "windows_index.csv"
        df.to_csv(str(index_csv), index=False)

        cfg = SimpleNamespace(
            paths=SimpleNamespace(
                windows_index_csv=str(index_csv),
                windows_dir=str(windows_dir),
                intermediate_dir=str(inter_dir),
                checkpoints_dir=str(tmp_path / "checkpoints"),
                reports_dir=str(tmp_path / "reports"),
            ),
            seeds=SimpleNamespace(global_seed=42),
        )

        csv_path = train_and_evaluate(cfg)
        res_df = pd.read_csv(str(csv_path))

        val_150 = res_df[(res_df["split"] == "val") & (res_df["window_length"] == 150)]
        assert not val_150.empty
        # Accuracy at 150 is the headline metric
        assert "accuracy" in val_150["metric"].values
        assert "balanced_accuracy" in val_150["metric"].values


# ===========================================================================
# 6. Persistence (Save, Export, Load)
# ===========================================================================

class TestPersistence:
    """Tests for model checkpointing and loading."""

    def test_save_and_load_roundtrip(self, synthetic_room_dataset, tmp_path: Path) -> None:
        """Fitted model saved and loaded restores identical predictions and proba."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        ckpt_dir = tmp_path / "checkpoints"
        model.save(ckpt_dir)
        assert (ckpt_dir / "room_model.pkl").exists()

        loaded = RoomModel().load(ckpt_dir)
        assert loaded._is_fit is True
        assert loaded.best_name == model.best_name
        assert loaded.rooms == model.rooms

        test_win = np.full((150, 256), 75.0, dtype=np.float32)
        p_orig   = model.predict_proba(test_win)
        p_loaded = loaded.predict_proba(test_win)
        np.testing.assert_allclose(p_orig, p_loaded, atol=1e-6)

    def test_export_alias(self, synthetic_room_dataset, tmp_path: Path) -> None:
        """export() alias creates valid checkpoint."""
        df, windows_dir, _ = synthetic_room_dataset
        model = RoomModel(log1p=True)
        model.fit(df, windows_dir)

        ckpt_dir = tmp_path / "exported"
        model.export(ckpt_dir)
        assert (ckpt_dir / "room_model.pkl").exists()

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path) -> None:
        """load raises FileNotFoundError if checkpoint does not exist."""
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            RoomModel().load(tmp_path / "empty_dir")


# ===========================================================================
# 7. Visualization & End-to-End Pipeline
# ===========================================================================

class TestPlotAndPipeline:
    """Tests for plot_window_length_study and train_and_evaluate end-to-end execution."""

    def test_plot_window_length_study_creates_png(self, tmp_path: Path) -> None:
        """plot_window_length_study writes a non-empty PNG figure."""
        sample_df = pd.DataFrame([
            {"model": "LR",  "window_length": 10,  "accuracy": 0.80, "balanced_accuracy": 0.79},
            {"model": "LR",  "window_length": 150, "accuracy": 0.98, "balanced_accuracy": 0.97},
            {"model": "MLP", "window_length": 10,  "accuracy": 0.82, "balanced_accuracy": 0.81},
            {"model": "MLP", "window_length": 150, "accuracy": 0.99, "balanced_accuracy": 0.98},
        ])

        fig_path = tmp_path / "room_window_length.png"
        plot_window_length_study(sample_df, fig_path)

        assert fig_path.exists()
        assert fig_path.stat().st_size > 0

    def test_train_and_evaluate_end_to_end(self, synthetic_room_dataset, tmp_path: Path) -> None:
        """train_and_evaluate creates results CSV with headline results at length 150 and figure."""
        df, windows_dir, inter_dir = synthetic_room_dataset
        index_csv = tmp_path / "windows_index.csv"
        df.to_csv(str(index_csv), index=False)

        reports_dir = tmp_path / "reports"
        ckpt_dir    = tmp_path / "checkpoints"

        cfg = SimpleNamespace(
            paths=SimpleNamespace(
                windows_index_csv=str(index_csv),
                windows_dir=str(windows_dir),
                intermediate_dir=str(inter_dir),
                checkpoints_dir=str(ckpt_dir),
                reports_dir=str(reports_dir),
            ),
            seeds=SimpleNamespace(global_seed=42),
        )

        res_csv = train_and_evaluate(cfg)
        assert res_csv.exists()
        assert res_csv == reports_dir / "room_model_results.csv"

        results_df = pd.read_csv(str(res_csv))
        assert list(results_df.columns) == ["model", "split", "window_length", "metric", "value"]

        # Verify headline result for val at window_length == 150
        headline = results_df[
            (results_df["window_length"] == 150) &
            (results_df["split"] == "val")
        ]
        assert not headline.empty
        assert set(headline["model"].unique()) == {"LR", "MLP"}
        assert set(headline["metric"].unique()) == {"accuracy", "balanced_accuracy"}

        # Verify window length study entries present in CSV
        study_entries = results_df[results_df["split"] == "val_length_study"]
        assert not study_entries.empty
        assert set(study_entries["window_length"].unique()) == {10, 25, 50, 100, 150}

        # Verify figure created
        fig_path = reports_dir / "room_window_length.png"
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0

        # Verify model saved
        assert (ckpt_dir / "room_model.pkl").exists()
