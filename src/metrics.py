"""
src/metrics.py
==============
Evaluation metrics for human activity recognition (HAR) and few-shot adaptation.

Metrics implemented:
- macro_f1: Macro-averaged F1 score across all classes.
- balanced_accuracy: Balanced accuracy score across classes.
- target_exclusive_macro_f1: Macro-F1 computed strictly on target-exclusive classes.
- shared_class_macro_f1: Macro-F1 computed strictly on classes shared between rooms.
- confusion_matrix: Array and DataFrame confusion matrix (rows=true, cols=pred).
- per_class_precision_recall_f1: Precision, recall, and F1 score per activity class.
- compute_all_metrics: Helper evaluating all summary metrics for an episode.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.fewshot import (
    EXCLUSIVE_ACTIVITIES,
    ROOM_A_ACTIVITIES,
    ROOM_B_ACTIVITIES,
    SHARED_ACTIVITIES,
    VALID_ACTIVITIES,
)


def macro_f1(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> float:
    """Compute macro-averaged F1 score."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def balanced_accuracy(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
) -> float:
    """Compute balanced accuracy score."""
    return float(balanced_accuracy_score(y_true, y_pred))


def target_exclusive_macro_f1(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    target_room: str,
) -> float:
    """Compute macro-F1 strictly across target-exclusive classes for target_room."""
    room = str(target_room).strip().upper()
    if room not in EXCLUSIVE_ACTIVITIES:
        raise ValueError(
            f"Invalid target_room: {target_room!r}. Allowed: ['A', 'B']"
        )
    exclusive_labels = list(EXCLUSIVE_ACTIVITIES[room])
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=exclusive_labels,
            average="macro",
            zero_division=0,
        )
    )


def shared_class_macro_f1(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
) -> float:
    """Compute macro-F1 strictly across shared classes ('sit', 'stand', 'walk')."""
    shared_labels = list(SHARED_ACTIVITIES)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=shared_labels,
            average="macro",
            zero_division=0,
        )
    )


def confusion_matrix(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute confusion matrix as a numpy array."""
    return sk_confusion_matrix(y_true, y_pred, labels=labels)


def confusion_matrix_df(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute confusion matrix formatted as a pandas DataFrame."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )


def per_class_precision_recall_f1(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, and F1 per class."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    results = {}
    for cls, p, r, f, s in zip(labels, prec, rec, f1, supp):
        results[str(cls)] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
    return results


def per_class_metrics_df(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return per-class precision, recall, f1, support as a DataFrame."""
    metrics = per_class_precision_recall_f1(y_true, y_pred, labels=labels)
    rows = [{"activity": k, **v} for k, v in metrics.items()]
    return pd.DataFrame(rows)


def compute_all_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    target_room: str,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute standard benchmark metrics for cross-room HAR evaluation."""
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels=labels),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "target_exclusive_macro_f1": target_exclusive_macro_f1(
            y_true, y_pred, target_room=target_room
        ),
        "shared_class_macro_f1": shared_class_macro_f1(y_true, y_pred),
    }