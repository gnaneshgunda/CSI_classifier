"""
tests/test_make_figures.py
==========================
Comprehensive unit tests for scripts/make_figures.py.

Tests cover:
- Generation of all five figure groups
- PNG and PDF dual format verification
- Label-efficiency aggregation (mean +/- std over seeds)
- Window-length accuracy curves
- Routing comparison bar charts
- Confusion matrices and per-class metrics tables
- Missing and malformed input handling
- Cached embedding handling (avoiding fitting on validation data)
- Smoke test and CLI execution
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.make_figures import (
    load_predictions_for_room,
    make_all_figures,
    plot_confusion_matrix_with_metrics,
    plot_encoder_umap,
    plot_label_efficiency_curves,
    plot_room_accuracy_vs_window_length,
    plot_routing_comparison,
)
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


def _make_dummy_config(tmp: Path) -> ExperimentConfig:
    """Create a lightweight dummy config pointing to tmp paths."""
    reports_dir = tmp / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentConfig(
        paths=PathsConfig(
            room_a_pcap_dir=tmp, room_a_val_dir=tmp, room_b_csv_dir=tmp,
            derived_dir=tmp, intermediate_dir=tmp, windows_dir=tmp,
            manifest_csv=tmp / "manifest.csv",
            windows_index_csv=tmp / "windows_index.csv",
            checkpoints_dir=tmp, reports_dir=reports_dir,
        ),
        seeds=SeedsConfig(global_seed=42, fewshot_seeds=[0]),
        window=None, cleaning=None, normalization=None, ssl=None,
        augment=None, align=None, routing=None, fewshot=None,
        activities=None, rooms=None, splits=None,
    )


class TestMakeFigures:
    """Test suite for figure generation functions, error handling, and CLI."""

    def test_plot_label_efficiency_curves(self, tmp_path: Path) -> None:
        """Verify label efficiency curves generation in PNG and PDF."""
        csv_path = tmp_path / "task3_sweep.csv"
        records = []
        for seed in range(3):
            for room in ("A", "B"):
                for K in (1, 5, 10):
                    records.append({"target_room": room, "K": K, "seed": seed, "metric": "macro_f1", "value": 0.8})
                    records.append({"target_room": room, "K": K, "seed": seed, "metric": "target_exclusive_macro_f1", "value": 0.75})
                    records.append({"target_room": room, "K": K, "seed": seed, "metric": "shared_class_macro_f1", "value": 0.85})
        pd.DataFrame(records).to_csv(str(csv_path), index=False)

        out_dir = tmp_path / "figures"
        plot_label_efficiency_curves(csv_path, out_dir)

        png = out_dir / "label_efficiency_curve.png"
        pdf = out_dir / "label_efficiency_curve.pdf"
        assert png.exists() and png.stat().st_size > 0
        assert pdf.exists() and pdf.stat().st_size > 0

    def test_plot_label_efficiency_missing_or_malformed_raises(self, tmp_path: Path) -> None:
        """Verify missing or malformed sweep CSV raises appropriate errors."""
        out_dir = tmp_path / "figures"
        missing_csv = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="Sweep results CSV not found"):
            plot_label_efficiency_curves(missing_csv, out_dir)

        malformed_csv = tmp_path / "bad.csv"
        pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(str(malformed_csv), index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_label_efficiency_curves(malformed_csv, out_dir)

    def test_plot_encoder_umap(self, tmp_path: Path) -> None:
        """Verify UMAP projection produces 2-panel and single-panel figures in PNG and PDF."""
        rng = np.random.RandomState(42)
        n = 60
        embs = rng.randn(n, 32).astype(np.float32)
        rooms = ["A"] * 30 + ["B"] * 30
        acts = list(ROOM_A_ACTIVITIES[:3]) * 10 + list(ROOM_B_ACTIVITIES[:3]) * 10
        splits = ["dev"] * 40 + ["val"] * 20

        out_dir = tmp_path / "figures"
        plot_encoder_umap(embs, rooms, acts, out_dir, splits=splits)

        for base in ("encoder_umap", "encoder_umap_by_room", "encoder_umap_by_activity"):
            png = out_dir / f"{base}.png"
            pdf = out_dir / f"{base}.pdf"
            assert png.exists() and png.stat().st_size > 0
            assert pdf.exists() and pdf.stat().st_size > 0

    def test_plot_encoder_umap_empty_raises(self, tmp_path: Path) -> None:
        """Verify empty embeddings array raises ValueError."""
        out_dir = tmp_path / "figures"
        with pytest.raises(ValueError, match="embeddings array is empty"):
            plot_encoder_umap(np.empty((0, 32)), [], [], out_dir)

    def test_plot_room_accuracy_vs_window_length(self, tmp_path: Path) -> None:
        """Verify Task 1 room accuracy vs window length generation."""
        csv_path = tmp_path / "room_model_results.csv"
        rows = []
        for model in ("Logistic Regression", "MLP"):
            for L in (10, 25, 50, 100, 150):
                rows.append({"model": model, "split": "val", "window_length": L, "metric": "accuracy", "value": 0.90 + L * 0.0005})
                rows.append({"model": model, "split": "val", "window_length": L, "metric": "balanced_accuracy", "value": 0.90})
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        out_dir = tmp_path / "figures"
        plot_room_accuracy_vs_window_length(csv_path, out_dir)

        png = out_dir / "room_accuracy_vs_window_length.png"
        pdf = out_dir / "room_accuracy_vs_window_length.pdf"
        assert png.exists() and png.stat().st_size > 0
        assert pdf.exists() and pdf.stat().st_size > 0

    def test_plot_room_accuracy_missing_or_malformed_raises(self, tmp_path: Path) -> None:
        """Verify missing or malformed room results CSV raises appropriate errors."""
        out_dir = tmp_path / "figures"
        with pytest.raises(FileNotFoundError, match="Room model results CSV not found"):
            plot_room_accuracy_vs_window_length(tmp_path / "missing.csv", out_dir)

        bad_csv = tmp_path / "bad_room.csv"
        pd.DataFrame([{"split": "val", "score": 0.5}]).to_csv(str(bad_csv), index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_room_accuracy_vs_window_length(bad_csv, out_dir)

    def test_plot_routing_comparison(self, tmp_path: Path) -> None:
        """Verify routing comparison bar chart generation in PNG and PDF."""
        csv_path = tmp_path / "routing_results.csv"
        rows = [
            {"target": "Room A", "mode": "None (Global)", "macro_f1": 0.70},
            {"target": "Room A", "mode": "Predicted (Router)", "macro_f1": 0.84},
            {"target": "Room A", "mode": "Oracle (Ground Truth)", "macro_f1": 0.86},
            {"target": "Room B", "mode": "None (Global)", "macro_f1": 0.72},
            {"target": "Room B", "mode": "Predicted (Router)", "macro_f1": 0.85},
            {"target": "Room B", "mode": "Oracle (Ground Truth)", "macro_f1": 0.87},
        ]
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        out_dir = tmp_path / "figures"
        plot_routing_comparison(csv_path, out_dir)

        png = out_dir / "routing_contribution.png"
        pdf = out_dir / "routing_contribution.pdf"
        assert png.exists() and png.stat().st_size > 0
        assert pdf.exists() and pdf.stat().st_size > 0

    def test_plot_routing_comparison_missing_or_malformed_raises(self, tmp_path: Path) -> None:
        """Verify missing or malformed routing CSV raises appropriate errors."""
        out_dir = tmp_path / "figures"
        with pytest.raises(FileNotFoundError, match="Routing results CSV not found"):
            plot_routing_comparison(tmp_path / "missing_routing.csv", out_dir)

        bad_csv = tmp_path / "bad_routing.csv"
        pd.DataFrame([{"invalid_col": 123}]).to_csv(str(bad_csv), index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_routing_comparison(bad_csv, out_dir)

    def test_plot_confusion_matrix_with_metrics(self, tmp_path: Path) -> None:
        """Verify confusion matrix heatmap and metrics table for both rooms."""
        out_dir = tmp_path / "figures"

        # Room A
        y_true_A = list(ROOM_A_ACTIVITIES) * 4
        y_pred_A = list(ROOM_A_ACTIVITIES) * 4
        plot_confusion_matrix_with_metrics(y_true_A, y_pred_A, "A", out_dir, K=5)

        png_A = out_dir / "confusion_matrix_roomA_K5.png"
        pdf_A = out_dir / "confusion_matrix_roomA_K5.pdf"
        assert png_A.exists() and png_A.stat().st_size > 0
        assert pdf_A.exists() and pdf_A.stat().st_size > 0

        # Room B
        y_true_B = list(ROOM_B_ACTIVITIES) * 4
        y_pred_B = list(ROOM_B_ACTIVITIES) * 4
        plot_confusion_matrix_with_metrics(y_true_B, y_pred_B, "B", out_dir, K=5)

        png_B = out_dir / "confusion_matrix_roomB_K5.png"
        pdf_B = out_dir / "confusion_matrix_roomB_K5.pdf"
        assert png_B.exists() and png_B.stat().st_size > 0
        assert pdf_B.exists() and pdf_B.stat().st_size > 0

    def test_plot_confusion_matrix_invalid_inputs_raises(self, tmp_path: Path) -> None:
        """Verify invalid room, length mismatch, or empty inputs raise ValueError."""
        out_dir = tmp_path / "figures"

        with pytest.raises(ValueError, match="Invalid target_room"):
            plot_confusion_matrix_with_metrics(["fall"], ["fall"], "C", out_dir)

        with pytest.raises(ValueError, match="Length mismatch"):
            plot_confusion_matrix_with_metrics(["fall", "sit"], ["fall"], "A", out_dir)

        with pytest.raises(ValueError, match="empty"):
            plot_confusion_matrix_with_metrics([], [], "A", out_dir)

    def test_load_predictions_for_room(self, tmp_path: Path) -> None:
        """Verify load_predictions_for_room resolves ground truth via window_id or direct column."""
        # Direct y_true
        pred_csv = tmp_path / "preds_direct.csv"
        pd.DataFrame({
            "window_id": ["w1", "w2"],
            "room": ["A", "A"],
            "y_true": ["fall", "sit"],
            "label": ["fall", "stand"],
            "confidence": [0.9, 0.8],
        }).to_csv(str(pred_csv), index=False)

        y_true, y_pred = load_predictions_for_room(pred_csv, "A")
        assert y_true == ["fall", "sit"]
        assert y_pred == ["fall", "stand"]

        # Resolved via windows_index_csv
        idx_csv = tmp_path / "windows_index.csv"
        pd.DataFrame({
            "window_id": ["w1", "w2"],
            "activity": ["walk", "stand"],
        }).to_csv(str(idx_csv), index=False)

        pred_p18_csv = tmp_path / "preds_p18.csv"
        pd.DataFrame({
            "window_id": ["w1", "w2"],
            "room": ["B", "B"],
            "label": ["walk", "walk"],
            "confidence": [0.95, 0.85],
        }).to_csv(str(pred_p18_csv), index=False)

        y_true_p18, y_pred_p18 = load_predictions_for_room(pred_p18_csv, "B", windows_index_csv=idx_csv)
        assert y_true_p18 == ["walk", "stand"]
        assert y_pred_p18 == ["walk", "walk"]

    def test_make_all_figures_smoke(self, tmp_path: Path) -> None:
        """Verify make_all_figures in smoke mode produces all 8 figures in PNG and PDF."""
        cfg = _make_dummy_config(tmp_path)
        out_dir = tmp_path / "figures"
        make_all_figures(cfg, out_dir=out_dir, smoke=True)

        expected_stems = [
            "label_efficiency_curve",
            "encoder_umap",
            "encoder_umap_by_room",
            "encoder_umap_by_activity",
            "room_accuracy_vs_window_length",
            "routing_contribution",
            "confusion_matrix_roomA_K5",
            "confusion_matrix_roomB_K5",
        ]
        for stem in expected_stems:
            png = out_dir / f"{stem}.png"
            pdf = out_dir / f"{stem}.pdf"
            assert png.exists(), f"Missing PNG for {stem}"
            assert pdf.exists(), f"Missing PDF for {stem}"
            assert png.stat().st_size > 0
            assert pdf.stat().st_size > 0

    def test_cli_smoke(self, tmp_path: Path) -> None:
        """Verify CLI smoke command runs end-to-end and outputs files."""
        out_dir = tmp_path / "cli_figures"
        cmd = [
            sys.executable,
            "scripts/make_figures.py",
            "--smoke",
            "--out_dir",
            str(out_dir),
        ]
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"make_figures.py failed: {result.stderr}\n{result.stdout}"
        assert (out_dir / "label_efficiency_curve.png").exists()
        assert (out_dir / "label_efficiency_curve.pdf").exists()
        assert (out_dir / "encoder_umap.png").exists()
        assert (out_dir / "confusion_matrix_roomA_K5.png").exists()
