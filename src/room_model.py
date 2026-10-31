"""
src/room_model.py
=================
Room-identification model for the CSI-HAR pipeline.

Overview
--------
Given a CSI amplitude window ``(T, 256)`` float32, predict which room
(``"A"`` or ``"B"``) it was recorded in.  This is **Task 1** – the routing
layer that feeds Tasks 2 and 3.

Features
--------
Per-subcarrier mean and standard deviation over the time axis (512-d):

    static = window.mean(axis=0)   # (256,)  time-averaged spectrum
    std    = window.std(axis=0)    # (256,)  temporal variability per subcarrier
    x      = concat([static, std]) # (512,)  room-discriminative feature

This 512-d vector is log1p-transformed before feeding to the classifiers,
matching the normalisation used in the SSL pipeline.

Classifiers
-----------
Two models are trained and evaluated:

``LogisticRegression`` (LR)
    C=1.0, max_iter=2000, solver='lbfgs', L2 regularisation.

``MLPClassifier`` (MLP)
    Architecture: 512 → 256 → 128 → 2.
    relu activations, dropout-equivalent alpha=1e-4 L2,
    adam optimiser, early stopping on 10 % internal validation.
    max_iter=200, random_state from cfg.seeds.global_seed.

Both use ``StandardScaler`` fitted on dev windows only (no leakage).

Window-length study
-------------------
Each validation capture has a full ``(n_packets_clean, 256)`` intermediate
array.  We slice the *central* ``L`` rows for L in {10, 25, 50, 100, 150}
and recompute the 512-d feature.  This tests how accuracy degrades with
shorter observations without re-training (the model is always trained at
the full length 150).

Cross-validation during internal dev fitting is grouped by ``capture_id``
so windows from the same physical capture never appear in both train and
validation folds (``GroupKFold``).

Persistence
-----------
Trained models are saved to ``checkpoints/room_model.pkl`` via
``joblib``.  The public ``predict_proba(window)`` method loads on demand.

Outputs
-------
``reports/room_model_results.csv``
    columns: model, split, window_length, metric, value

``reports/room_window_length.png``
    Line plot of validation accuracy vs window length for both models.

Public API
----------
``RoomFeatureExtractor``
    ``transform(window) -> np.ndarray``   shape (512,) float32

``RoomModel(cfg)``
    ``.fit(index_df, windows_dir)``
    ``.predict(window) -> str``
    ``.predict_proba(window) -> np.ndarray``   shape (2,) [p_A, p_B]
    ``.save(checkpoints_dir)``
    ``.load(checkpoints_dir)``
    ``.rooms`` – list: ['A', 'B']

``train_and_evaluate(cfg) -> Path``
    Full pipeline: fit, score on val, window-length study, save CSV + figure.

``window_length_study(cfg, model) -> pd.DataFrame``
    Standalone sweep at lengths {10, 25, 50, 100, 150} on val captures.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

log = logging.getLogger(__name__)

# Ordered room list – position 0 = "A", position 1 = "B"
ROOMS: List[str] = ["A", "B"]

# Window lengths swept in the length study
LENGTH_STUDY_VALUES: List[int] = [10, 25, 50, 100, 150]


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class RoomFeatureExtractor:
    """Extract the 512-d room-discriminative feature from a CSI window.

    Feature: ``log1p(concat([mean_over_time, std_over_time]))``.

    Parameters
    ----------
    log1p:
        Whether to apply log1p to the raw amplitudes before computing
        mean/std.  Default True (matches normalisation in the SSL pipeline).
    norm_stats_path:
        Optional path to ``norm_stats.npz``.  When provided, applies
        global z-score (log1p → subtract global_mean, divide by global_scale)
        before computing mean/std features.  This removes the amplitude-scale
        difference between data collection setups so the classifier operates
        on a common numerical representation.
    """

    DIM: int = 512   # 256 mean + 256 std

    def __init__(self, log1p: bool = True, norm_stats_path=None) -> None:
        self.log1p = log1p
        self._global_mean  = None
        self._global_scale = None
        if norm_stats_path is not None:
            stats = np.load(str(norm_stats_path))
            self._global_mean  = stats["global_mean"].astype(np.float64)   # (256,)
            self._global_scale = stats["global_scale"].astype(np.float64)  # (256,)

    def transform(self, window: np.ndarray) -> np.ndarray:
        """Extract feature from a single window.

        Parameters
        ----------
        window:
            Array of shape ``(T, 256)`` float32.

        Returns
        -------
        np.ndarray shape ``(512,)`` float32.

        Raises
        ------
        ValueError
            If ``window.shape[1] != 256`` or ``window.ndim != 2``.
        """
        if window.ndim != 2 or window.shape[1] != 256:
            raise ValueError(
                f"Expected (T, 256) window, got {window.shape}"
            )
        w = window.astype(np.float64)
        if self.log1p:
            w = np.log1p(np.clip(w, 0.0, None))
        if self._global_mean is not None:
            w = (w - self._global_mean) / (self._global_scale + 1e-8)
        mean = w.mean(axis=0).astype(np.float32)   # (256,)
        std  = w.std(axis=0,  ddof=0).astype(np.float32)  # (256,)
        return np.concatenate([mean, std])

    def transform_batch(self, windows: np.ndarray) -> np.ndarray:
        """Vectorised transform for a batch of windows.

        Parameters
        ----------
        windows:
            Array of shape ``(N, T, 256)``.

        Returns
        -------
        np.ndarray shape ``(N, 512)`` float32.
        """
        if windows.ndim != 3 or windows.shape[2] != 256:
            raise ValueError(f"Expected (N, T, 256), got {windows.shape}")
        w = windows.astype(np.float64)
        if self.log1p:
            w = np.log1p(np.clip(w, 0.0, None))
        if self._global_mean is not None:
            w = (w - self._global_mean) / (self._global_scale + 1e-8)
        mean = w.mean(axis=1).astype(np.float32)   # (N, 256)
        std  = w.std(axis=1,  ddof=0).astype(np.float32)  # (N, 256)
        return np.concatenate([mean, std], axis=1)

    def transform_truncated(self, window: np.ndarray, length: int) -> np.ndarray:
        """Extract feature from the central *length* rows of *window*.

        Used by the window-length study to simulate shorter observations.

        Parameters
        ----------
        window:
            Full window of shape ``(T, 256)``.
        length:
            Number of rows to use (central crop).  Clamped to T.

        Returns
        -------
        np.ndarray shape ``(512,)`` float32.
        """
        T = window.shape[0]
        length = min(length, T)
        start  = (T - length) // 2
        return self.transform(window[start: start + length])


# ---------------------------------------------------------------------------
# Classifier pair
# ---------------------------------------------------------------------------

def _make_lr(random_state: int = 42):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )


def _make_mlp(random_state: int = 42):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15,
        random_state=random_state,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# RoomModel
# ---------------------------------------------------------------------------

class RoomModel:
    """Trained room-identification model with ``predict_proba`` for routing.

    Contains two sub-models (LR and MLP) trained on the same dev features.
    The primary model used for routing is the one with higher validation
    accuracy (stored in ``self.best_name``).

    Parameters
    ----------
    cfg:
        Loaded ``ExperimentConfig``.  If ``None`` (smoke mode), sensible
        defaults are used.
    log1p:
        Whether to log1p-transform amplitudes before feature extraction.
    """

    def __init__(self, cfg=None, log1p: bool = True) -> None:
        self.cfg = cfg
        # Use global norm stats if available so features are on a common scale
        norm_stats_path = None
        if cfg is not None:
            p = Path(cfg.paths.derived_dir) / "norm_stats.npz"
            if p.exists():
                norm_stats_path = p
        self._extractor = RoomFeatureExtractor(log1p=log1p, norm_stats_path=norm_stats_path)
        self._scalers: Dict[str, object]   = {}
        self._clfs:   Dict[str, object]   = {}
        self._is_fit: bool = False
        self.best_name: str = "LR"
        self.rooms: List[str] = list(ROOMS)
        self.classes_: np.ndarray = np.array(ROOMS)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        index_df,                   # pandas DataFrame from windows_index.csv
        windows_dir: Path,
        *,
        random_state: int = 42,
    ) -> "RoomModel":
        """Fit LR and MLP on all development windows.

        Parameters
        ----------
        index_df:
            Full windows index DataFrame (all splits).
        windows_dir:
            Root of the ``derived/windows/`` tree.
        random_state:
            RNG seed for reproducibility.

        Returns
        -------
        self
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold, cross_val_score

        dev_df = index_df[index_df["split"] == "dev"].copy()
        if dev_df.empty:
            raise ValueError("No development windows found in index.")

        log.info("Extracting features for %d dev windows ...", len(dev_df))
        X, y, groups = self._load_split(dev_df, windows_dir, split="dev")
        log.info("Dev features: X=%s  classes=%s", X.shape, sorted(set(y)))

        # Encode labels: "A" -> 0, "B" -> 1
        y_enc = np.array([ROOMS.index(r) for r in y], dtype=int)

        models = {
            "LR":  _make_lr(random_state),
            "MLP": _make_mlp(random_state),
        }

        cv_scores: Dict[str, float] = {}
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)

        for name, clf in models.items():
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)

            # Grouped CV (no capture leakage)
            if n_splits >= 2:
                gkf = GroupKFold(n_splits=n_splits)
                try:
                    cv = cross_val_score(
                        clf, X_sc, y_enc,
                        cv=gkf, groups=groups,
                        scoring="balanced_accuracy",
                        n_jobs=1,
                    )
                    cv_mean = float(cv.mean())
                    log.info(
                        "[%s] Grouped %d-fold CV balanced_accuracy: %.4f ± %.4f",
                        name, n_splits, cv_mean, float(cv.std()),
                    )
                except Exception as exc:
                    log.warning("[%s] CV failed (%s); fitting without CV.", name, exc)
                    cv_mean = 0.0
            else:
                log.warning("[%s] Not enough groups (%d) for GroupKFold; fitting without CV.", name, n_groups)
                cv_mean = 0.0

            # Re-fit on full dev set
            scaler2 = StandardScaler()
            X_sc2   = scaler2.fit_transform(X)
            clf.fit(X_sc2, y_enc)

            self._scalers[name] = scaler2
            self._clfs[name]    = clf
            cv_scores[name]     = cv_mean

        # Select best model for routing
        self.best_name = max(cv_scores, key=cv_scores.__getitem__)
        log.info("Best model for routing: %s (CV bal_acc=%.4f)",
                 self.best_name, cv_scores[self.best_name])

        self._is_fit = True
        return self

    # ---------------------------------------------------------- predict API

    def predict_proba(
        self,
        window: np.ndarray,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Return room probabilities ``[p_A, p_B]`` for routing.

        Parameters
        ----------
        window:
            CSI amplitude window ``(T, 256)`` float32,
            or precomputed feature ``(512,)``,
            or batch of features ``(N, 512)``,
            or batch of windows ``(N, T, 256)``.
        model:
            ``"LR"``, ``"MLP"``, or ``None`` (use ``self.best_name``).

        Returns
        -------
        np.ndarray shape ``(2,)`` or ``(N, 2)`` float32, sums to ~1.
        """
        self._check_fit()
        name = model or self.best_name
        arr  = np.asarray(window, dtype=np.float32)

        is_batch = False
        if arr.ndim == 1 and arr.shape[0] == self._extractor.DIM:
            feat = arr.reshape(1, -1)
        elif arr.ndim == 2 and arr.shape[1] == 256:
            feat = self._extractor.transform(arr).reshape(1, -1)
        elif arr.ndim == 2 and arr.shape[1] == self._extractor.DIM:
            feat = arr
            is_batch = True
        elif arr.ndim == 3 and arr.shape[2] == 256:
            feat = self._extractor.transform_batch(arr)
            is_batch = True
        else:
            raise ValueError(
                f"Unsupported input shape for predict_proba: {arr.shape}. "
                f"Expected (T, 256) window, (512,) feature, (N, 512) features, or (N, T, 256) windows."
            )

        feat_sc = self._scalers[name].transform(feat)
        proba   = self._clfs[name].predict_proba(feat_sc)
        if not is_batch:
            return proba[0].astype(np.float32)
        return proba.astype(np.float32)

    def predict(
        self,
        window: np.ndarray,
        model: Optional[str] = None,
    ):
        """Return the predicted room label (``"A"`` or ``"B"``).

        Parameters
        ----------
        window:
            CSI amplitude window or batch of windows/features.
        model:
            ``"LR"``, ``"MLP"``, or ``None`` (use ``self.best_name``).

        Returns
        -------
        str or np.ndarray of str: ``"A"`` or ``"B"``.
        """
        proba = self.predict_proba(window, model=model)
        if proba.ndim == 1:
            return ROOMS[int(np.argmax(proba))]
        return np.array([ROOMS[int(i)] for i in np.argmax(proba, axis=1)])

    def score_split(
        self,
        index_df,
        windows_dir: Path,
        split: str = "val",
    ) -> Dict[str, Dict[str, float]]:
        """Score both models on a given split.

        Parameters
        ----------
        index_df:
            Full index DataFrame.
        windows_dir:
            Root of windows directory.
        split:
            ``"dev"`` or ``"val"``.

        Returns
        -------
        dict mapping model name to dict of metric -> value.
        E.g. ``{"LR": {"accuracy": 0.95, "balanced_accuracy": 0.94}, ...}``
        """
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        self._check_fit()
        df = index_df[index_df["split"] == split].copy()
        if df.empty:
            log.warning("No windows for split=%s", split)
            return {}

        X, y, _ = self._load_split(df, windows_dir, split=split)
        y_enc   = np.array([ROOMS.index(r) for r in y], dtype=int)

        results: Dict[str, Dict[str, float]] = {}
        for name in self._clfs:
            feat_sc = self._scalers[name].transform(X)
            y_pred  = self._clfs[name].predict(feat_sc)
            results[name] = {
                "accuracy":          float(accuracy_score(y_enc, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_enc, y_pred)),
            }
            log.info(
                "[%s] split=%s  acc=%.4f  bal_acc=%.4f",
                name, split,
                results[name]["accuracy"],
                results[name]["balanced_accuracy"],
            )

        return results

    # ------------------------------------------------------- window-length study

    def window_length_study(
        self,
        index_df,
        intermediate_dir: Path,
        lengths: Optional[List[int]] = None,
    ):
        """Sweep validation accuracy vs window length.

        Uses the intermediate (full-length) arrays so each validation capture
        can be truncated to any target length in a single pass.

        Parameters
        ----------
        index_df:
            Full index DataFrame (val rows are used).
        intermediate_dir:
            Path to ``derived/intermediate/`` directory.
        lengths:
            Window lengths to try.  Defaults to ``LENGTH_STUDY_VALUES``.

        Returns
        -------
        pandas.DataFrame with columns:
            model, window_length, accuracy, balanced_accuracy
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        self._check_fit()
        if lengths is None:
            lengths = LENGTH_STUDY_VALUES

        # Collect unique val captures with their intermediate paths
        val_df = index_df[index_df["split"] == "val"].copy()
        if val_df.empty:
            log.warning("No val windows for length study.")
            return pd.DataFrame()

        # Group by capture to get one entry per capture
        # Val windows are canonical (one per capture), so groupby is safe
        cap_info = (
            val_df
            .groupby("capture_id")
            .agg(room=("room", "first"))
            .reset_index()
        )

        rows = []
        for length in lengths:
            X_list: List[np.ndarray] = []
            y_list: List[int]        = []

            for _, cap_row in cap_info.iterrows():
                cid  = cap_row["capture_id"]
                room = cap_row["room"]
                # Try to find the intermediate file
                inter_path = (
                    intermediate_dir / room / f"{cid}.npy"
                )
                if not inter_path.exists():
                    log.warning(
                        "Intermediate file not found (length study): %s", inter_path
                    )
                    continue
                try:
                    arr  = np.load(str(inter_path)).astype(np.float32)
                    feat = self._extractor.transform_truncated(arr, length)
                    X_list.append(feat)
                    y_list.append(ROOMS.index(room))
                except Exception as exc:
                    log.warning("Skipping %s (length=%d): %s", cid, length, exc)

            if not X_list:
                log.warning("No data for length=%d, skipping.", length)
                continue

            X  = np.vstack(X_list)
            y  = np.array(y_list, dtype=int)

            for name in self._clfs:
                X_sc   = self._scalers[name].transform(X)
                y_pred = self._clfs[name].predict(X_sc)
                rows.append({
                    "model":             name,
                    "window_length":     length,
                    "accuracy":          float(accuracy_score(y, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
                })
                log.info(
                    "[%s] length=%3d  acc=%.4f  bal_acc=%.4f",
                    name, length,
                    rows[-1]["accuracy"], rows[-1]["balanced_accuracy"],
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------- persistence

    def save(self, checkpoints_dir: Path) -> None:
        """Serialise fitted scalers + classifiers to disk.

        Writes ``{checkpoints_dir}/room_model.pkl``.

        Parameters
        ----------
        checkpoints_dir:
            Destination directory (created if absent).
        """
        import joblib

        self._check_fit()
        checkpoints_dir = Path(checkpoints_dir)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "rooms":      self.rooms,
            "best_name":  self.best_name,
            "log1p":      self._extractor.log1p,
            "scalers":    self._scalers,
            "clfs":       self._clfs,
        }
        out_path = checkpoints_dir / "room_model.pkl"
        joblib.dump(payload, str(out_path), compress=3)
        log.info("Room model saved -> %s", out_path)

    # Alias for export requirement
    export = save

    def load(self, checkpoints_dir: Path) -> "RoomModel":
        """Reload a previously saved room model.

        Parameters
        ----------
        checkpoints_dir:
            Directory containing ``room_model.pkl``.

        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        import joblib

        ckpt_path = Path(checkpoints_dir) / "room_model.pkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Room model checkpoint not found: {ckpt_path}\n"
                "Run train_and_evaluate() first."
            )
        payload = joblib.load(str(ckpt_path))
        self.rooms       = payload["rooms"]
        self.classes_    = np.array(self.rooms)
        self.best_name   = payload["best_name"]
        self._extractor  = RoomFeatureExtractor(log1p=payload["log1p"])
        self._scalers    = payload["scalers"]
        self._clfs       = payload["clfs"]
        self._is_fit     = True
        log.info("Room model loaded from %s  (best=%s)", ckpt_path, self.best_name)
        return self

    # ------------------------------------------------------- internal helpers

    def _load_split(
        self,
        df,
        windows_dir: Path,
        split: str,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load windows from disk, extract 512-d features.

        Returns
        -------
        X      : (N, 512) float32
        y      : list[str] of room labels ('A' or 'B')
        groups : (N,) int – capture index for GroupKFold
        """
        from sklearn.preprocessing import LabelEncoder

        X_list: List[np.ndarray] = []
        y_list: List[str]        = []
        cap_ids: List[str]       = []
        skipped = 0

        for _, row in df.iterrows():
            wid   = row["window_id"]
            room  = row["room"]
            wpath = windows_dir / split / f"{wid}.npy"
            if not wpath.exists():
                skipped += 1
                continue
            try:
                win  = np.load(str(wpath)).astype(np.float32)
                feat = self._extractor.transform(win)
                X_list.append(feat)
                y_list.append(str(room))
                cap_ids.append(str(row["capture_id"]))
            except Exception as exc:
                log.warning("Skipping window %s: %s", wid, exc)
                skipped += 1

        if skipped:
            log.warning("Skipped %d windows in split=%s", skipped, split)

        if not X_list:
            return (
                np.empty((0, self._extractor.DIM), dtype=np.float32),
                [],
                np.empty((0,), dtype=int),
            )

        X      = np.vstack(X_list).astype(np.float32)
        le     = LabelEncoder()
        groups = le.fit_transform(cap_ids).astype(int)
        return X, y_list, groups

    def _check_fit(self) -> None:
        if not self._is_fit or not self._clfs:
            raise RuntimeError(
                "RoomModel has not been fitted. Call fit() or load() first."
            )


# ---------------------------------------------------------------------------
# Module-level window length study helper
# ---------------------------------------------------------------------------

def window_length_study(
    cfg,
    model: RoomModel,
    lengths: Optional[List[int]] = None,
):
    """Standalone sweep at lengths {10, 25, 50, 100, 150} on val captures.

    Parameters
    ----------
    cfg:
        Loaded ``ExperimentConfig``.
    model:
        Fitted ``RoomModel``.
    lengths:
        Optional list of window lengths. Defaults to ``LENGTH_STUDY_VALUES``.

    Returns
    -------
    pandas.DataFrame with columns: model, window_length, accuracy, balanced_accuracy.
    """
    import pandas as pd
    index_csv = Path(cfg.paths.windows_index_csv)
    inter_dir = Path(cfg.paths.intermediate_dir)
    index_df  = pd.read_csv(str(index_csv))
    return model.window_length_study(index_df, inter_dir, lengths=lengths)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_window_length_study(
    length_df,        # pandas DataFrame from window_length_study()
    out_path: Path,
) -> None:
    """Save a line plot of accuracy vs window length for LR and MLP.

    Parameters
    ----------
    length_df:
        DataFrame with columns: model, window_length, accuracy, balanced_accuracy.
    out_path:
        Destination PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced Accuracy")]

    colors = {"LR": "#2563eb", "MLP": "#dc2626"}
    markers = {"LR": "o", "MLP": "s"}

    for ax, (metric, title) in zip(axes, metrics):
        for model in length_df["model"].unique():
            sub = length_df[length_df["model"] == model].sort_values("window_length")
            ax.plot(
                sub["window_length"], sub[metric],
                label=model,
                color=colors.get(model, None),
                marker=markers.get(model, "o"),
                linewidth=2,
                markersize=7,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Window length (packets)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xticks(sorted(length_df["window_length"].unique()))
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Room Identification: Accuracy vs Window Length",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def train_and_evaluate(cfg) -> Path:
    """Full pipeline: fit room models, score, run length study, save outputs.

    Parameters
    ----------
    cfg:
        Loaded ``ExperimentConfig``.

    Returns
    -------
    Path to the written results CSV.
    """
    import csv as csv_mod
    import pandas as pd

    index_csv   = Path(cfg.paths.windows_index_csv)
    windows_dir = Path(cfg.paths.windows_dir)
    inter_dir   = Path(cfg.paths.intermediate_dir)
    ckpt_dir    = Path(cfg.paths.checkpoints_dir)
    reports_dir = Path(cfg.paths.reports_dir)
    seed        = cfg.seeds.global_seed

    if not index_csv.exists():
        raise FileNotFoundError(
            f"windows_index.csv not found: {index_csv}\n"
            "Run scripts/build_windows.py first."
        )

    log.info("Loading windows index: %s", index_csv)
    index_df = pd.read_csv(str(index_csv))
    log.info(
        "Total: %d windows  dev=%d  val=%d",
        len(index_df),
        (index_df["split"] == "dev").sum(),
        (index_df["split"] == "val").sum(),
    )

    # Fit
    model = RoomModel(cfg=cfg, log1p=True)
    model.fit(index_df, windows_dir, random_state=seed)

    # Score on val (headline)
    log.info("Scoring on val windows ...")
    val_scores = model.score_split(index_df, windows_dir, split="val")

    # Score on dev (sanity)
    log.info("Scoring on dev windows ...")
    dev_scores = model.score_split(index_df, windows_dir, split="dev")

    # Window-length study
    log.info("Running window-length study ...")
    length_df = model.window_length_study(index_df, inter_dir)

    # Save model
    model.save(ckpt_dir)

    # Assemble results CSV
    result_rows = []
    for clf_name, metrics in val_scores.items():
        for metric, value in metrics.items():
            result_rows.append({
                "model":         clf_name,
                "split":         "val",
                "window_length": 150,
                "metric":        metric,
                "value":         f"{value:.6f}",
            })
    for clf_name, metrics in dev_scores.items():
        for metric, value in metrics.items():
            result_rows.append({
                "model":         clf_name,
                "split":         "dev",
                "window_length": 150,
                "metric":        metric,
                "value":         f"{value:.6f}",
            })
    for _, r in length_df.iterrows():
        result_rows.append({
            "model":         r["model"],
            "split":         "val_length_study",
            "window_length": int(r["window_length"]),
            "metric":        "accuracy",
            "value":         f"{r['accuracy']:.6f}",
        })
        result_rows.append({
            "model":         r["model"],
            "split":         "val_length_study",
            "window_length": int(r["window_length"]),
            "metric":        "balanced_accuracy",
            "value":         f"{r['balanced_accuracy']:.6f}",
        })

    reports_dir.mkdir(parents=True, exist_ok=True)
    results_csv = reports_dir / "room_model_results.csv"
    _CSV_COLS   = ["model", "split", "window_length", "metric", "value"]
    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=_CSV_COLS)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)
    log.info("Results CSV -> %s  (%d rows)", results_csv, len(result_rows))

    # Figure
    if not length_df.empty:
        fig_path = reports_dir / "room_window_length.png"
        try:
            plot_window_length_study(length_df, fig_path)
        except Exception as exc:
            log.warning("Could not save figure: %s", exc)

    # Print headline
    print("\n=== Room Identification – Headline Results (window_length=150) ===")
    print(f"{'Model':<8} {'Split':<6} {'Metric':<22} {'Value':>8}")
    print("-" * 50)
    for r in result_rows:
        if r["window_length"] == 150 and r["split"] in ("val", "dev"):
            print(
                f"{r['model']:<8} {r['split']:<6} {r['metric']:<22} "
                f"{float(r['value']):>8.4f}"
            )
    print(f"\nBest model for routing: {model.best_name}")

    return results_csv


# ---------------------------------------------------------------------------
# CLI & Smoke test
# ---------------------------------------------------------------------------

def _run_smoke() -> None:
    """Run end-to-end self-test on tiny synthetic data in <30 seconds."""
    import tempfile
    import csv as csv_mod
    import numpy as np

    print("[smoke] src/room_model.py self-test ...")

    rng = np.random.default_rng(42)
    T, N_SUB = 150, 256

    # ---- test feature extractor ----
    ext  = RoomFeatureExtractor(log1p=True)
    win  = rng.uniform(0, 100, (T, N_SUB)).astype(np.float32)
    feat = ext.transform(win)
    assert feat.shape == (512,), f"Feat shape {feat.shape}"
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()

    feat_trunc = ext.transform_truncated(win, 50)
    assert feat_trunc.shape == (512,)

    batch = rng.uniform(0, 100, (8, T, N_SUB)).astype(np.float32)
    bf    = ext.transform_batch(batch)
    assert bf.shape == (8, 512)

    # ---- test RoomModel fit / predict ----
    with tempfile.TemporaryDirectory() as tmp:
        tmp         = __import__("pathlib").Path(tmp)
        windows_dir = tmp / "windows"
        inter_dir   = tmp / "intermediate"
        ckpt_dir    = tmp / "checkpoints"
        reports_dir = tmp / "reports"

        index_rows = []
        wi         = 0

        def _w(split, room, capture_num, win_num):
            global wi
            arr     = rng.uniform(
                (100 if room == "A" else 10),
                (200 if room == "A" else 50),
                (T, N_SUB),
            ).astype(np.float32)
            cap_id  = f"{room}__cap{capture_num:03d}"
            win_id  = f"{cap_id}__{win_num:05d}"
            out_dir = windows_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / f"{win_id}.npy"), arr)
            # Also save intermediate
            inter_out = inter_dir / room
            inter_out.mkdir(parents=True, exist_ok=True)
            np.save(str(inter_out / f"{cap_id}.npy"), arr)
            return {
                "window_id":   win_id,
                "capture_id":  cap_id,
                "room":        room,
                "activity":    "walk",
                "split":       split,
                "start":       0,
                "config_hash": "smoke",
            }

        n_caps = 10
        for room in ("A", "B"):
            for c in range(n_caps):
                for w in range(3):
                    index_rows.append(_w("dev", room, c, w))
                index_rows.append(_w("val", room, c, 0))

        import pandas as pd
        index_df = pd.DataFrame(index_rows)

        model = RoomModel(cfg=None, log1p=True)
        model.fit(index_df, windows_dir)

        # predict / predict_proba
        test_win = rng.uniform(100, 200, (T, N_SUB)).astype(np.float32)
        pred  = model.predict(test_win)
        proba = model.predict_proba(test_win)
        assert pred in ("A", "B")
        assert proba.shape == (2,)
        assert abs(float(proba.sum()) - 1.0) < 1e-5

        # score_split
        scores = model.score_split(index_df, windows_dir, split="val")
        assert "LR" in scores and "MLP" in scores
        for name, m in scores.items():
            assert "accuracy" in m and "balanced_accuracy" in m

        # save / load
        model.save(ckpt_dir)
        assert (ckpt_dir / "room_model.pkl").exists()

        model2 = RoomModel()
        model2.load(ckpt_dir)
        proba2 = model2.predict_proba(test_win)
        assert np.allclose(proba, proba2, atol=1e-5), \
            f"Saved/loaded proba mismatch: {proba} vs {proba2}"

        # window-length study
        length_df = model.window_length_study(
            index_df, inter_dir, lengths=[10, 50, 150]
        )
        assert not length_df.empty
        assert set(length_df["window_length"]).issuperset({10, 50, 150})

        # figure (skip if matplotlib unavailable)
        try:
            fig_path = reports_dir / "test_fig.png"
            plot_window_length_study(length_df, fig_path)
            assert fig_path.exists(), "Figure not written"
            print("[smoke] figure saved  OK")
        except ImportError:
            print("[smoke] matplotlib not available - figure skipped")

    print(f"[smoke] feat shape (512,) float32 finite  OK")
    print(f"[smoke] batch transform (8, 512)  OK")
    print(f"[smoke] fit LR + MLP on synthetic data  OK")
    print(f"[smoke] predict / predict_proba  OK")
    print(f"[smoke] score_split on val  OK")
    print(f"[smoke] save / load checkpoint + proba match  OK")
    print(f"[smoke] window-length study ({len(length_df)} rows)  OK")
    print("[smoke] src/room_model.py PASSED")


def main() -> None:
    """CLI entry point for src/room_model.py."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="P15: Room identification model (Task 1) - train LR & MLP, window-length study, and export model."
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run end-to-end smoke test on tiny synthetic data in under 30 seconds",
    )
    args = parser.parse_args()

    if args.smoke:
        _run_smoke()
        sys.exit(0)
    else:
        from src.config import load_config
        cfg = load_config(args.config)
        train_and_evaluate(cfg)


if __name__ == "__main__":
    main()
