"""
tests/test_metrics.py
=====================
Unit tests for src/metrics.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    balanced_accuracy,
    compute_all_metrics,
    confusion_matrix,
    confusion_matrix_df,
    macro_f1,
    per_class_metrics_df,
    per_class_precision_recall_f1,
    shared_class_macro_f1,
    target_exclusive_macro_f1,
)


class TestMetrics:
    def test_macro_f1_perfect(self) -> None:
        y_true = ["sit", "stand", "walk"]
        y_pred = ["sit", "stand", "walk"]
        assert macro_f1(y_true, y_pred) == 1.0

    def test_balanced_accuracy_imbalanced(self) -> None:
        y_true = ["sit", "sit", "sit", "stand"]
        y_pred = ["sit", "sit", "sit", "sit"]
        # sit recall = 1.0, stand recall = 0.0 -> balanced acc = 0.5
        assert balanced_accuracy(y_true, y_pred) == 0.5

    def test_target_exclusive_macro_f1_room_a(self) -> None:
        # Room A exclusive: fall, sitdown, standup
        # Shared: sit
        y_true = ["fall", "sitdown", "standup", "sit"]
        y_pred = ["fall", "sitdown", "walk",    "sit"]  # standup wrong, sit correct

        # Target-exclusive should only evaluate fall, sitdown, standup
        # fall: f1=1.0, sitdown: f1=1.0, standup: f1=0.0 -> macro = 2/3
        score = target_exclusive_macro_f1(y_true, y_pred, target_room="A")
        assert pytest.approx(score, rel=1e-4) == 2.0 / 3.0

    def test_target_exclusive_macro_f1_room_b(self) -> None:
        # Room B exclusive: empty, lying
        y_true = ["empty", "lying", "sit"]
        y_pred = ["empty", "sit",   "sit"]  # lying wrong

        # empty: f1=1.0, lying: f1=0.0 -> macro = 0.5
        score = target_exclusive_macro_f1(y_true, y_pred, target_room="B")
        assert pytest.approx(score, rel=1e-4) == 0.5

    def test_target_exclusive_invalid_room_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid target_room"):
            target_exclusive_macro_f1(["fall"], ["fall"], target_room="X")

    def test_shared_class_macro_f1(self) -> None:
        # Shared: sit, stand, walk
        y_true = ["sit", "stand", "walk", "fall"]
        y_pred = ["sit", "stand", "sit",  "fall"]  # walk wrong, fall ignored
        # sit: true=1, pred=2 -> p=1/2, r=1 -> f1 = 2/3
        # stand: f1=1.0
        # walk: f1=0.0
        # shared macro = (2/3 + 1.0 + 0.0) / 3 = 5/9
        score = shared_class_macro_f1(y_true, y_pred)
        assert pytest.approx(score, rel=1e-4) == (2.0 / 3.0 + 1.0 + 0.0) / 3.0

    def test_confusion_matrix_and_df(self) -> None:
        y_true = ["sit", "stand", "sit"]
        y_pred = ["sit", "sit",   "stand"]
        labels = ["sit", "stand"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # true sit, pred sit
        assert cm[0, 1] == 1  # true sit, pred stand
        assert cm[1, 0] == 1  # true stand, pred sit
        assert cm[1, 1] == 0  # true stand, pred stand

        cm_df = confusion_matrix_df(y_true, y_pred, labels=labels)
        assert isinstance(cm_df, pd.DataFrame)
        assert list(cm_df.index) == ["true_sit", "true_stand"]
        assert list(cm_df.columns) == ["pred_sit", "pred_stand"]

    def test_per_class_precision_recall_f1(self) -> None:
        y_true = ["sit", "stand", "sit"]
        y_pred = ["sit", "sit",   "stand"]
        metrics = per_class_precision_recall_f1(y_true, y_pred)
        assert "sit" in metrics
        assert "stand" in metrics
        assert metrics["sit"]["support"] == 2
        assert metrics["stand"]["support"] == 1

        df = per_class_metrics_df(y_true, y_pred)
        assert isinstance(df, pd.DataFrame)
        assert "activity" in df.columns
        assert "f1" in df.columns

    def test_compute_all_metrics(self) -> None:
        y_true = ["fall", "sit", "stand", "walk"]
        y_pred = ["fall", "sit", "stand", "walk"]
        res = compute_all_metrics(y_true, y_pred, target_room="A")
        assert "macro_f1" in res
        assert "balanced_accuracy" in res
        assert "target_exclusive_macro_f1" in res
        assert "shared_class_macro_f1" in res
        assert res["macro_f1"] == 1.0
        assert res["balanced_accuracy"] == 1.0

    def test_numpy_array_inputs(self) -> None:
        y_true = np.array(["sit", "stand", "walk"])
        y_pred = np.array(["sit", "stand", "walk"])
        assert macro_f1(y_true, y_pred) == 1.0
        assert balanced_accuracy(y_true, y_pred) == 1.0
