"""
scripts/make_figures.py
=======================
Generate all primary publication figures for cross-room WiFi CSI human activity recognition.

Figures generated (both .png and .pdf in reports/figures/):
1. label_efficiency_curve: Macro-F1 vs K for Room A and Room B with separate curves
   for target-exclusive and transferred shared activities (mean +/- std over seeds).
2. encoder_umap: 2D latent space projection (UMAP / PCA fallback) colored by room and activity
   (avoiding fitting on validation data), plus single-panel exports (encoder_umap_by_room,
   encoder_umap_by_activity).
3. room_accuracy_vs_window_length: Task 1 room identification accuracy as a function
   of observation window length L in {10, 25, 50, 100, 150}.
4. routing_contribution: Grouped bar chart comparing HAR downstream Macro-F1 across
   routing strategies (None vs Predicted vs Oracle).
5. confusion_matrix_roomA_K5 & confusion_matrix_roomB_K5: Heatmaps paired with per-class
   Precision, Recall, and F1 performance tables at K=5.

Usage:
    python scripts/make_figures.py --config configs/experiment.yaml
    python scripts/make_figures.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import ExperimentConfig, load_config
from src.fewshot import (
    EXCLUSIVE_ACTIVITIES,
    ROOM_A_ACTIVITIES,
    ROOM_B_ACTIVITIES,
    SHARED_ACTIVITIES,
    VALID_ACTIVITIES,
)
from src.metrics import confusion_matrix, per_class_precision_recall_f1

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("make_figures")

# Distinct, publication-grade color scheme
_COLOR_OVERALL = "#1f77b4"     # Steel blue
_COLOR_EXCLUSIVE = "#d95f02"   # Ochre / vermilion
_COLOR_SHARED = "#2ca02c"      # Green
_COLOR_ROOM_A = "#e74c3c"      # Coral red
_COLOR_ROOM_B = "#2980b9"      # Blue

_ACTIVITY_PALETTE = {
    "empty": "#7f7f7f",
    "fall": "#d62728",
    "lying": "#8c564b",
    "sit": "#1f77b4",
    "sitdown": "#ff7f0e",
    "stand": "#2ca02c",
    "standup": "#9467bd",
    "walk": "#17becf",
}


# ---------------------------------------------------------------------------
# Helper: Save Figure in both PNG and PDF formats
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, out_dir: Path, base_name: str, dpi: int = 300) -> None:
    """Save matplotlib figure to out_dir as both .png and .pdf atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{base_name}.png"
    pdf_path = out_dir / f"{base_name}.pdf"

    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    log.info("Saved %s and %s", png_path.name, pdf_path.name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Label-Efficiency Curves
# ---------------------------------------------------------------------------

def plot_label_efficiency_curves(
    task3_csv: Path,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    """Generate Macro-F1 vs K curves for Target Rooms A and B with separate lines
    for target-exclusive and transferred shared activities (mean +/- std over seeds).
    """
    if not task3_csv.exists():
        raise FileNotFoundError(f"Sweep results CSV not found: {task3_csv}")

    df = pd.read_csv(str(task3_csv))
    required_cols = {"target_room", "K", "metric", "value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"Malformed sweep CSV in {task3_csv}. Missing required columns: {sorted(list(missing))}"
        )

    df = df.copy()
    df["target_room"] = (
        df["target_room"]
        .astype(str)
        .str.upper()
        .str.replace("ROOM ", "")
        .str.strip()
    )
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["K"].isna().any() or df["value"].isna().any():
        raise ValueError(f"Malformed numerical values in {task3_csv} for K or value.")
    df["K"] = df["K"].astype(int)
    df["value"] = df["value"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    target_configs = [
        ("A", "Target: Room A (Transfer: B → A)", axes[0]),
        ("B", "Target: Room B (Transfer: A → B)", axes[1]),
    ]

    metric_styles = [
        ("macro_f1", ["macro_f1", "macro-f1", "overall_macro_f1"], "Overall Macro-F1", _COLOR_OVERALL, "o", "-"),
        ("target_exclusive_macro_f1", ["target_exclusive_macro_f1", "exclusive_macro_f1", "target_exclusive_f1", "exclusive_f1"], "Target-Exclusive Classes", _COLOR_EXCLUSIVE, "s", "--"),
        ("shared_class_macro_f1", ["shared_class_macro_f1", "shared_macro_f1", "shared_class_f1", "shared_f1"], "Transferred Shared Classes", _COLOR_SHARED, "^", ":"),
    ]

    for room, title, ax in target_configs:
        sub_df = df[df["target_room"] == room]
        if sub_df.empty:
            continue

        K_vals = sorted(sub_df["K"].unique())

        for canon_name, aliases, m_label, col, mark, lstyle in metric_styles:
            m_sub = sub_df[sub_df["metric"].str.lower().isin([a.lower() for a in aliases])]
            if m_sub.empty:
                continue

            stats = m_sub.groupby("K")["value"].agg(["mean", "std"]).reindex(K_vals)
            means = stats["mean"].values
            stds = stats["std"].fillna(0.0).values

            ax.plot(
                K_vals,
                means,
                label=m_label,
                color=col,
                marker=mark,
                linestyle=lstyle,
                linewidth=2.2,
                markersize=7,
            )
            ax.fill_between(
                K_vals,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=col,
                alpha=0.15,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Shots K (Target Dev Captures / Exclusive Class)", fontsize=11, labelpad=8)
        ax.set_xticks(K_vals)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10, loc="lower right")

    axes[0].set_ylabel("Macro-F1 Score", fontsize=12, labelpad=8)
    fig.suptitle("Label Efficiency: Few-Shot Adaptation Performance vs Support Size", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_fig(fig, out_dir, "label_efficiency_curve", dpi=dpi)


# ---------------------------------------------------------------------------
# 2. UMAP Projection Colored by Room and Activity
# ---------------------------------------------------------------------------

def plot_encoder_umap(
    embeddings: np.ndarray,
    rooms: Sequence[str],
    activities: Sequence[str],
    out_dir: Path,
    splits: Optional[Sequence[str]] = None,
    dpi: int = 300,
) -> None:
    """Generate 2D latent space projection (UMAP with PCA fallback) colored by room and activity.
    Avoids fitting on validation data by fitting solely on dev data when splits are provided.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rooms = np.asarray([str(r).upper() for r in rooms])
    activities = np.asarray([str(a).lower() for a in activities])
    if splits is not None:
        splits = np.asarray([str(s).lower() for s in splits])

    if len(embeddings) == 0:
        raise ValueError("Cannot generate UMAP projection: embeddings array is empty.")
    if len(embeddings) != len(rooms) or len(embeddings) != len(activities):
        raise ValueError(
            f"Length mismatch: embeddings={len(embeddings)}, rooms={len(rooms)}, activities={len(activities)}"
        )

    # Avoid fitting on validation data: fit reducer only on development embeddings
    if splits is not None and "dev" in splits:
        dev_mask = (splits == "dev")
        if np.any(dev_mask):
            fit_embs = embeddings[dev_mask]
        else:
            fit_embs = embeddings
    else:
        fit_embs = embeddings

    proj_2d: Optional[np.ndarray] = None
    method_name = "UMAP"
    try:
        import umap
        n_neighbors = min(15, max(2, len(fit_embs) - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42)
        reducer.fit(fit_embs)
        proj_2d = reducer.transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        reducer.fit(fit_embs)
        proj_2d = reducer.transform(embeddings)
        method_name = "PCA (2D)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: By Room
    ax1 = axes[0]
    for r, col, name in [("A", _COLOR_ROOM_A, "Room A"), ("B", _COLOR_ROOM_B, "Room B")]:
        mask = rooms == r
        if mask.any():
            ax1.scatter(
                proj_2d[mask, 0],
                proj_2d[mask, 1],
                c=col,
                label=name,
                alpha=0.65,
                s=28,
                edgecolors="none",
            )
    ax1.set_title("(a) Latent Space by Environment (Room)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax1.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=10.5, loc="upper right")

    # Panel 2: By Activity
    ax2 = axes[1]
    present_acts = sorted(list(set(activities)))
    for act in present_acts:
        mask = activities == act
        col = _ACTIVITY_PALETTE.get(act, "#333333")
        ax2.scatter(
            proj_2d[mask, 0],
            proj_2d[mask, 1],
            c=col,
            label=act,
            alpha=0.7,
            s=28,
            edgecolors="none",
        )
    ax2.set_title("(b) Latent Space by Activity Label", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=9.5, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(f"Frozen Encoder Embedding Manifold ({method_name})", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()

    save_fig(fig, out_dir, "encoder_umap", dpi=dpi)

    # Save separate single-panel figures matching RUNBOOK.md naming
    fig_room, ax_r = plt.subplots(figsize=(7, 6))
    for r, col, name in [("A", _COLOR_ROOM_A, "Room A"), ("B", _COLOR_ROOM_B, "Room B")]:
        mask = rooms == r
        if mask.any():
            ax_r.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=col, label=name, alpha=0.65, s=28)
    ax_r.set_title("Latent Space by Environment (Room)", fontsize=13, fontweight="bold")
    ax_r.set_xlabel(f"{method_name} Dimension 1")
    ax_r.set_ylabel(f"{method_name} Dimension 2")
    ax_r.grid(True, linestyle="--", alpha=0.4)
    ax_r.legend(frameon=True)
    fig_room.tight_layout()
    save_fig(fig_room, out_dir, "encoder_umap_by_room", dpi=dpi)

    fig_act, ax_a = plt.subplots(figsize=(8, 6))
    for act in present_acts:
        mask = activities == act
        col = _ACTIVITY_PALETTE.get(act, "#333333")
        ax_a.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=col, label=act, alpha=0.7, s=28)
    ax_a.set_title("Latent Space by Activity Label", fontsize=13, fontweight="bold")
    ax_a.set_xlabel(f"{method_name} Dimension 1")
    ax_a.set_ylabel(f"{method_name} Dimension 2")
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.legend(frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig_act.tight_layout()
    save_fig(fig_act, out_dir, "encoder_umap_by_activity", dpi=dpi)


# ---------------------------------------------------------------------------
# 3. Room Accuracy vs Window Length
# ---------------------------------------------------------------------------

def plot_room_accuracy_vs_window_length(
    room_results_csv: Path,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    """Plot Task 1 room identification accuracy as a function of observation window length."""
    if not room_results_csv.exists():
        raise FileNotFoundError(f"Room model results CSV not found: {room_results_csv}")

    df = pd.read_csv(str(room_results_csv))
    required_cols = {"model", "window_length", "value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"Malformed room results CSV in {room_results_csv}. Missing required columns: {sorted(list(missing))}"
        )

    df = df.copy()
    if "split" in df.columns:
        val_df = df[df["split"].astype(str).str.lower() == "val"]
        if not val_df.empty:
            df = val_df
    if "metric" in df.columns:
        acc_df = df[df["metric"].astype(str).str.lower() == "accuracy"]
        if not acc_df.empty:
            df = acc_df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["window_length"] = pd.to_numeric(df["window_length"], errors="coerce")
    if df["value"].isna().any() or df["window_length"].isna().any():
        raise ValueError(f"Malformed numerical values in {room_results_csv}.")
    df["value"] = df["value"].astype(float)
    df["window_length"] = df["window_length"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    models = df["model"].unique()
    colors = ["#2b5c8f", "#d95f02", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "D"]

    for idx, model in enumerate(models):
        m_df = df[df["model"] == model].groupby("window_length")["value"].mean().reset_index()
        m_df = m_df.sort_values("window_length")
        col = colors[idx % len(colors)]
        mark = markers[idx % len(markers)]
        ax.plot(
            m_df["window_length"],
            m_df["value"],
            label=str(model),
            color=col,
            marker=mark,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_title("Task 1: Room Identification Accuracy vs Window Length", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Observation Window Length L (Packets / Time-steps)", fontsize=11, labelpad=8)
    ax.set_ylabel("Validation Accuracy", fontsize=11, labelpad=8)
    ax.set_xticks(sorted(df["window_length"].unique()))
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    fig.tight_layout()

    save_fig(fig, out_dir, "room_accuracy_vs_window_length", dpi=dpi)


# ---------------------------------------------------------------------------
# 4. Routing Comparison (Predicted vs Oracle vs None)
# ---------------------------------------------------------------------------

def plot_routing_comparison(
    routing_csv: Path,
    out_dir: Path,
    dpi: int = 300,
) -> None:
    """Bar chart comparing HAR downstream Macro-F1 across routing modes: None vs Predicted vs Oracle."""
    if not routing_csv.exists():
        raise FileNotFoundError(f"Routing results CSV not found: {routing_csv}")

    df = pd.read_csv(str(routing_csv))
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("target", "target_room", "room"):
            col_map[c] = "target"
        elif cl in ("mode", "routing_mode", "routing"):
            col_map[c] = "mode"
        elif cl in ("macro_f1", "f1", "value"):
            col_map[c] = "macro_f1"
    df = df.rename(columns=col_map)

    required = {"target", "mode", "macro_f1"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Malformed routing CSV in {routing_csv}. Missing required columns: {sorted(list(missing))}"
        )

    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    if df["macro_f1"].isna().any():
        raise ValueError(f"Malformed macro_f1 numbers in {routing_csv}.")
    df["macro_f1"] = df["macro_f1"].astype(float)

    # Normalize target names
    df["target"] = df["target"].astype(str).str.strip()
    df["target"] = df["target"].replace({"A": "Room A", "B": "Room B", "joint": "Overall", "all": "Overall"})

    # Normalize mode names
    def _clean_mode(m: str) -> str:
        ml = str(m).lower()
        if "none" in ml or "global" in ml:
            return "None (Global)"
        if "pred" in ml or "router" in ml:
            return "Predicted (Router)"
        if "oracle" in ml:
            return "Oracle (Ground Truth)"
        return str(m)

    df["mode"] = df["mode"].apply(_clean_mode)

    targets = ["Room A", "Room B", "Overall"]
    modes = ["None (Global)", "Predicted (Router)", "Oracle (Ground Truth)"]
    mode_colors = ["#95a5a6", "#2980b9", "#27ae60"]

    x = np.arange(len(targets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, (m, col) in enumerate(zip(modes, mode_colors)):
        sub = df[df["mode"] == m]
        vals = []
        for t in targets:
            match = sub[sub["target"].str.lower() == t.lower()]
            if not match.empty:
                vals.append(match["macro_f1"].mean())
            else:
                if t == "Overall":
                    room_vals = [
                        sub[sub["target"].str.lower() == r.lower()]["macro_f1"].mean()
                        for r in ("Room A", "Room B")
                    ]
                    vals.append(float(np.nanmean(room_vals)) if room_vals else 0.0)
                else:
                    vals.append(0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, vals, width, label=m, color=col, edgecolor="black", linewidth=0.8, alpha=0.9)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Routing Contribution to Cross-Room Human Activity Recognition", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Validation Macro-F1 Score", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    fig.tight_layout()

    save_fig(fig, out_dir, "routing_contribution", dpi=dpi)


# ---------------------------------------------------------------------------
# 5. Confusion Matrices per Target Room at K=5 with Per-Class Metrics
# ---------------------------------------------------------------------------

def plot_confusion_matrix_with_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    target_room: str,
    out_dir: Path,
    K: int = 5,
    dpi: int = 300,
) -> None:
    """Generate normalized confusion matrix heatmap paired with per-class Precision, Recall, and F1 table."""
    room = str(target_room).strip().upper()
    if room not in ("A", "B"):
        raise ValueError(f"Invalid target_room: {target_room!r}. Allowed: ['A', 'B']")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("Cannot plot confusion matrix: true/pred sequences are empty.")

    valid_classes = VALID_ACTIVITIES[room]
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm.astype(float) / np.maximum(row_sums, 1), 0.0)

    per_class = per_class_precision_recall_f1(y_true, y_pred, labels=valid_classes)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.25)

    # Left: Confusion Matrix Heatmap
    ax_cm = fig.add_subplot(gs[0, 0])
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall Ratio", rotation=-90, va="bottom", fontsize=10)

    n_cls = len(valid_classes)
    for i in range(n_cls):
        for j in range(n_cls):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_col = "white" if cm_norm[i, j] > 0.55 else "black"
            ax_cm.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", color=text_col, fontsize=9, fontweight="bold"
            )

    ax_cm.set_xticks(range(n_cls))
    ax_cm.set_yticks(range(n_cls))
    ax_cm.set_xticklabels(valid_classes, rotation=35, ha="right", fontsize=10)
    ax_cm.set_yticklabels(valid_classes, fontsize=10)
    ax_cm.set_xlabel("Predicted Activity", fontsize=11, fontweight="bold", labelpad=8)
    ax_cm.set_ylabel("True Activity", fontsize=11, fontweight="bold", labelpad=8)
    ax_cm.set_title(f"Target Room {room} (K={K}): Confusion Matrix", fontsize=12, fontweight="bold", pad=10)

    # Right: Per-Class Precision, Recall, F1 Table
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    table_data = []
    for cls in valid_classes:
        stats = per_class.get(cls, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        is_excl = cls in EXCLUSIVE_ACTIVITIES[room]
        tag = "(excl)" if is_excl else "(shared)"
        table_data.append([
            f"{cls} {tag}",
            f"{stats['precision']:.3f}",
            f"{stats['recall']:.3f}",
            f"{stats['f1']:.3f}",
            str(stats['support']),
        ])

    col_labels = ["Activity", "Precision", "Recall", "F1", "Support"]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.7)

    for col_idx in range(len(col_labels)):
        cell = tbl[(0, col_idx)]
        cell.set_facecolor("#2b5c8f")
        cell.set_text_props(color="white", fontweight="bold")

    ax_tbl.set_title(f"Per-Class Performance Summary (K={K})", fontsize=12, fontweight="bold", pad=12)
    fig.suptitle(f"Few-Shot Classification Diagnosis: Target Room {room} (K={K})", fontsize=14, fontweight="bold", y=0.98)

    save_fig(fig, out_dir, f"confusion_matrix_room{room}_K{K}", dpi=dpi)


# ---------------------------------------------------------------------------
# Helper: Load Predictions with windows_index ground truth join
# ---------------------------------------------------------------------------

def load_predictions_for_room(
    predictions_path: Path,
    target_room: str,
    windows_index_csv: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Load prediction file, extracting true activity labels and predicted labels."""
    pdf = pd.read_csv(str(predictions_path))
    if "room" in pdf.columns:
        pdf = pdf[pdf["room"].astype(str).str.upper() == target_room.upper()]

    if "label" not in pdf.columns:
        raise ValueError(f"Predictions CSV {predictions_path} missing 'label' column.")

    y_pred = list(pdf["label"].astype(str))

    # Determine ground-truth labels
    if "y_true" in pdf.columns:
        y_true = list(pdf["y_true"].astype(str))
    elif "true_label" in pdf.columns:
        y_true = list(pdf["true_label"].astype(str))
    elif "activity" in pdf.columns:
        y_true = list(pdf["activity"].astype(str))
    elif "window_id" in pdf.columns and windows_index_csv is not None and windows_index_csv.exists():
        index_df = pd.read_csv(str(windows_index_csv))
        wid_to_act = dict(zip(index_df["window_id"].astype(str), index_df["activity"].astype(str)))
        y_true = [wid_to_act.get(str(w), "") for w in pdf["window_id"]]
        if any(not act for act in y_true):
            raise ValueError(f"Some window_ids in {predictions_path} could not be resolved in {windows_index_csv}.")
    else:
        raise ValueError(
            f"Could not resolve ground-truth labels for {predictions_path}. "
            "File must contain 'y_true', 'activity', or valid 'window_id' resolvable via windows_index_csv."
        )

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Main Routine: Generate All Figures
# ---------------------------------------------------------------------------

def make_all_figures(
    cfg: ExperimentConfig,
    out_dir: Optional[Path | str] = None,
    dpi: int = 300,
    smoke: bool = False,
    task3_csv: Optional[Path | str] = None,
    room_csv: Optional[Path | str] = None,
    routing_csv: Optional[Path | str] = None,
    embeddings_path: Optional[Path | str] = None,
    predictions_dir: Optional[Path | str] = None,
) -> None:
    """Generate all five requested publication figures and write PNG & PDF outputs."""
    if out_dir is None:
        out_dir = Path(cfg.paths.reports_dir) / "figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating publication figures into %s ...", out_dir)

    if smoke:
        _make_all_figures_smoke(out_dir=out_dir, dpi=dpi)
        return

    # 1. Label efficiency curve
    if task3_csv is None:
        task3_csv = Path(cfg.paths.reports_dir) / "task3_sweep.csv"
    plot_label_efficiency_curves(Path(task3_csv), out_dir, dpi=dpi)

    # 2. UMAP Projection
    loaded_embs = False
    if embeddings_path is None:
        for candidate in (
            Path(cfg.paths.derived_dir) / "cached_embeddings.npz",
            Path(cfg.paths.reports_dir) / "cached_embeddings.npz",
        ):
            if candidate.exists():
                embeddings_path = candidate
                break

    if embeddings_path is not None and Path(embeddings_path).exists():
        emb_file = Path(embeddings_path)
        npz = np.load(str(emb_file), allow_pickle=True)
        emb_key = next((k for k in ("embeddings", "X", "z") if k in npz), None)
        room_key = next((k for k in ("rooms", "room", "target_room") if k in npz), None)
        act_key = next((k for k in ("activities", "labels", "y", "activity") if k in npz), None)
        split_key = next((k for k in ("splits", "split") if k in npz), None)
        if emb_key and room_key and act_key:
            splits_arr = npz[split_key] if split_key else None
            plot_encoder_umap(
                npz[emb_key], npz[room_key], npz[act_key], out_dir, splits=splits_arr, dpi=dpi
            )
            loaded_embs = True

    if not loaded_embs:
        windows_dir = Path(cfg.paths.windows_dir)
        index_csv = Path(cfg.paths.windows_index_csv)
        if index_csv.exists() and windows_dir.exists():
            from run_episode import extract_embeddings_for_df, load_encoder, load_normalizer, load_router
            index_df = pd.read_csv(str(index_csv))
            # Sample dev and val windows
            sample_df = index_df.groupby(["room", "activity", "split"]).head(15).copy()
            encoder = load_encoder(cfg)
            normalizer = load_normalizer(cfg, dev_df=index_df[index_df["split"] == "dev"])
            router = load_router(cfg)

            X_emb, y_labels, _, _ = extract_embeddings_for_df(
                sample_df, windows_dir, encoder, normalizer, router
            )
            plot_encoder_umap(
                X_emb,
                sample_df["room"].values,
                y_labels,
                out_dir,
                splits=sample_df["split"].values,
                dpi=dpi,
            )
            loaded_embs = True
        else:
            raise FileNotFoundError(
                f"Cached embeddings not found at {embeddings_path} and windows index not available at {index_csv}."
            )

    # 3. Room accuracy vs window length
    if room_csv is None:
        room_csv = Path(cfg.paths.reports_dir) / "room_model_results.csv"
    plot_room_accuracy_vs_window_length(Path(room_csv), out_dir, dpi=dpi)

    # 4. Routing comparison
    if routing_csv is None:
        routing_csv = Path(cfg.paths.reports_dir) / "routing_results.csv"
    plot_routing_comparison(Path(routing_csv), out_dir, dpi=dpi)

    # 5. Confusion matrices at K=5
    rep_dir = Path(predictions_dir) if predictions_dir else Path(cfg.paths.reports_dir)
    index_csv = Path(cfg.paths.windows_index_csv) if hasattr(cfg.paths, "windows_index_csv") else None

    for room in ("A", "B"):
        candidate_paths = [
            rep_dir / f"predictions_room{room}_K5.csv",
            rep_dir / f"predictions_{room}_K5.csv",
            rep_dir / f"task3_predictions_room{room}_K5.csv",
            rep_dir / f"episode_predictions.csv",
        ]
        found = False
        for cpath in candidate_paths:
            if cpath.exists():
                try:
                    y_true_list, y_pred_list = load_predictions_for_room(cpath, room, index_csv)
                    if y_true_list and y_pred_list:
                        plot_confusion_matrix_with_metrics(
                            y_true_list, y_pred_list, room, out_dir, K=5, dpi=dpi
                        )
                        found = True
                        break
                except Exception as exc:
                    log.warning("Could not extract predictions from %s: %s", cpath, exc)

        if not found:
            raise FileNotFoundError(
                f"K=5 predictions for Room {room} not found in {rep_dir}. "
                f"Run 'python run_episode.py --target_room {room} --K 5' first."
            )

    log.info("All publication figures successfully generated in %s", out_dir)


def _make_all_figures_smoke(out_dir: Path, dpi: int = 150) -> None:
    """Generate complete synthetic figure suite for smoke testing in < 15 seconds."""
    print("[smoke] Running make_figures smoke test on synthetic data ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # 1. Label efficiency curve synthetic data
    records = []
    for seed in range(5):
        srng = np.random.RandomState(seed)
        for room, base_ov in [("A", 0.50), ("B", 0.55)]:
            for K in (1, 5, 10, 20):
                gain = np.log2(K) * 0.10
                records.append({"target_room": room, "K": K, "seed": seed, "metric": "macro_f1", "value": min(0.95, base_ov + gain + srng.normal(0, 0.02))})
                records.append({"target_room": room, "K": K, "seed": seed, "metric": "target_exclusive_macro_f1", "value": min(0.95, base_ov + gain - 0.05 + srng.normal(0, 0.02))})
                records.append({"target_room": room, "K": K, "seed": seed, "metric": "shared_class_macro_f1", "value": min(0.95, base_ov + gain + 0.05 + srng.normal(0, 0.02))})

    synth_task3_csv = out_dir / "_smoke_task3.csv"
    pd.DataFrame(records).to_csv(str(synth_task3_csv), index=False)
    plot_label_efficiency_curves(synth_task3_csv, out_dir, dpi=dpi)
    synth_task3_csv.unlink()

    # 2. UMAP synthetic data
    all_acts = list(_ACTIVITY_PALETTE.keys())
    act_centers = {act: rng.randn(32) * 2.0 for act in all_acts}
    room_offsets = {"A": rng.randn(32) * 0.8, "B": rng.randn(32) * -0.8}
    syn_embs, syn_rooms, syn_acts, syn_splits = [], [], [], []
    for r in ("A", "B"):
        valid = ROOM_A_ACTIVITIES if r == "A" else ROOM_B_ACTIVITIES
        for act in valid:
            for s_idx in range(6):
                vec = act_centers[act] + room_offsets[r] + rng.randn(32) * 0.3
                syn_embs.append(vec)
                syn_rooms.append(r)
                syn_acts.append(act)
                syn_splits.append("dev" if s_idx < 4 else "val")

    plot_encoder_umap(
        np.array(syn_embs, dtype=np.float32),
        syn_rooms,
        syn_acts,
        out_dir,
        splits=syn_splits,
        dpi=dpi,
    )

    # 3. Room accuracy synthetic data
    synth_room_csv = out_dir / "_smoke_room.csv"
    r_rows = []
    for model in ("Logistic Regression", "MLP Classifier"):
        for L in (10, 25, 50, 100, 150):
            r_rows.append({"model": model, "split": "val", "window_length": L, "metric": "accuracy", "value": 0.85 + (L / 150.0) * 0.14})
    pd.DataFrame(r_rows).to_csv(str(synth_room_csv), index=False)
    plot_room_accuracy_vs_window_length(synth_room_csv, out_dir, dpi=dpi)
    synth_room_csv.unlink()

    # 4. Routing comparison synthetic data
    synth_routing_csv = out_dir / "_smoke_routing.csv"
    rt_rows = [
        {"target": "Room A", "mode": "None (Global)", "macro_f1": 0.70},
        {"target": "Room A", "mode": "Predicted (Router)", "macro_f1": 0.84},
        {"target": "Room A", "mode": "Oracle (Ground Truth)", "macro_f1": 0.86},
        {"target": "Room B", "mode": "None (Global)", "macro_f1": 0.72},
        {"target": "Room B", "mode": "Predicted (Router)", "macro_f1": 0.85},
        {"target": "Room B", "mode": "Oracle (Ground Truth)", "macro_f1": 0.87},
    ]
    pd.DataFrame(rt_rows).to_csv(str(synth_routing_csv), index=False)
    plot_routing_comparison(synth_routing_csv, out_dir, dpi=dpi)
    synth_routing_csv.unlink()

    # 5. Confusion matrices at K=5
    for room in ("A", "B"):
        valid = VALID_ACTIVITIES[room]
        y_t, y_p = [], []
        for act in valid:
            for _ in range(25):
                y_t.append(act)
                y_p.append(act if rng.rand() < 0.85 else rng.choice(valid))
        plot_confusion_matrix_with_metrics(y_t, y_p, room, out_dir, K=5, dpi=dpi)

    print("[smoke] make_figures smoke test passed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from results CSVs and cached embeddings.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--task3_csv", "--results_csv", type=str, default=None, help="Path to task3 sweep CSV")
    parser.add_argument("--room_csv", type=str, default=None, help="Path to room model results CSV")
    parser.add_argument("--routing_csv", type=str, default=None, help="Path to routing comparison CSV")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to cached embeddings .npz file")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Directory with prediction CSVs")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: reports/figures)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else None
    if cfg is None:
        from src.config import ActivitiesConfig, FewshotConfig, PathsConfig, RoomsConfig, SeedsConfig, SplitsConfig
        tmp = Path("reports")
        cfg = ExperimentConfig(
            paths=PathsConfig(
                room_a_pcap_dir=tmp, room_a_val_dir=tmp, room_b_csv_dir=tmp, derived_dir=tmp,
                intermediate_dir=tmp, windows_dir=tmp, manifest_csv=tmp / "manifest.csv",
                windows_index_csv=tmp / "windows_index.csv", checkpoints_dir=tmp, reports_dir=tmp,
            ),
            seeds=SeedsConfig(global_seed=42, fewshot_seeds=[0]),
            window=None, cleaning=None, normalization=None, ssl=None, augment=None, align=None, routing=None,
            fewshot=None, activities=None, rooms=None, splits=None,
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.reports_dir) / "figures"
    make_all_figures(
        cfg=cfg,
        out_dir=out_dir,
        dpi=args.dpi,
        smoke=args.smoke,
        task3_csv=args.task3_csv,
        room_csv=args.room_csv,
        routing_csv=args.routing_csv,
        embeddings_path=args.embeddings,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()
