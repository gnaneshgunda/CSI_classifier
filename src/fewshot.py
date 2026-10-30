"""
src/fewshot.py
==============
Few-shot domain adaptation for Task 3 – Cross-room human activity recognition.

Overview
--------
In Task 3, a model pre-trained on source-room data is adapted to a target room
using a small support set of K labeled captures per target-exclusive activity:

- Room A exclusive: "fall", "sitdown", "standup"
- Room B exclusive: "empty", "lying"
- Shared activities: "sit", "stand", "walk"

Key Components
--------------
1. ``sample_support(manifest, target_room, K, seed)``:
   Selects exactly K distinct development captures for each target-exclusive
   activity in the target room. Validation captures and shared activities
   are NEVER included. One capture counts as one shot.

2. ``PrototypeHead``:
   - Source prototypes: computed as the mean embedding per class across all
     source development labels.
   - Target-exclusive prototypes: computed hierarchically by first averaging
     windows within each capture, then averaging across captures (ensuring
     one capture counts as one shot regardless of window count).
   - Classifier:
     * Gaussian: Class-conditional Gaussian with pooled shrunken covariance
       (regularized with configurable alpha or Ledoit-Wolf shrinkage).
     * Cosine: Nearest class centroid using cosine similarity with optional
       temperature scaling.
   - Room Masking: Predictions are strictly masked to the valid activity
     classes of the target room (non-target classes cannot be predicted).
   - Prediction: ``predict(X) -> (label, confidence)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.covariance import ledoit_wolf

from src.manifest import ManifestRow, read_manifest

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activity and Room Constants
# ---------------------------------------------------------------------------

SHARED_ACTIVITIES: Tuple[str, ...] = ("sit", "stand", "walk")

ROOM_A_ACTIVITIES: Tuple[str, ...] = (
    "fall",
    "sit",
    "sitdown",
    "stand",
    "standup",
    "walk",
)

ROOM_B_ACTIVITIES: Tuple[str, ...] = (
    "empty",
    "lying",
    "sit",
    "stand",
    "walk",
)

ROOM_A_EXCLUSIVE: Tuple[str, ...] = ("fall", "sitdown", "standup")
ROOM_B_EXCLUSIVE: Tuple[str, ...] = ("empty", "lying")

EXCLUSIVE_ACTIVITIES: Dict[str, Tuple[str, ...]] = {
    "A": ROOM_A_EXCLUSIVE,
    "B": ROOM_B_EXCLUSIVE,
}

VALID_ACTIVITIES: Dict[str, Tuple[str, ...]] = {
    "A": ROOM_A_ACTIVITIES,
    "B": ROOM_B_ACTIVITIES,
}

OPPOSITE_ROOM: Dict[str, str] = {
    "A": "B",
    "B": "A",
}


# ---------------------------------------------------------------------------
# Support Data Structures
# ---------------------------------------------------------------------------

class SupportRow(ManifestRow):
    """ManifestRow subclass that supports both attribute and dict-like key access."""

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class SupportSet(list):
    """Specialized list of support captures with column indexing and helpers."""

    def __getitem__(self, key):
        if isinstance(key, str):
            return [
                getattr(item, key) if hasattr(item, key) else item[key]
                for item in self
            ]
        return super().__getitem__(key)

    @property
    def capture_ids(self) -> List[str]:
        return [r.capture_id for r in self]

    @property
    def activities(self) -> List[str]:
        return [r.activity for r in self]

    @property
    def splits(self) -> List[str]:
        return [r.split for r in self]

    @property
    def rooms(self) -> List[str]:
        return [r.room for r in self]

    def to_dataframe(self):
        import pandas as pd

        records = []
        for r in self:
            if hasattr(r, "to_dict"):
                records.append(r.to_dict())
            elif hasattr(r, "__dict__"):
                records.append(vars(r))
            else:
                records.append(dict(r))
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Support Sampling
# ---------------------------------------------------------------------------

def sample_support(
    manifest: Union[Sequence[Union[ManifestRow, dict]], Path, str, object],
    target_room: str,
    K: int,
    seed: int,
    as_dataframe: bool = False,
) -> Union[SupportSet, object]:
    """Sample K distinct target DEVELOPMENT captures per target-exclusive activity.

    Parameters
    ----------
    manifest:
        List of ManifestRow, path to manifest.csv, or pandas DataFrame.
    target_room:
        "A" or "B" (case-insensitive).
    K:
        Number of distinct captures to sample per target-exclusive activity.
        One capture counts as one shot.
    seed:
        RNG seed for reproducibility.
    as_dataframe:
        If True, return a pandas DataFrame instead of SupportSet.

    Returns
    -------
    SupportSet
        A list of SupportRow objects representing the selected captures.
        Guaranteed:
        - NEVER contains validation captures (split == "dev" strictly).
        - NEVER contains shared activities ("sit", "stand", "walk").
        - Only contains target-exclusive activities for target_room.
        - Exactly K distinct captures per exclusive activity (unless K=0).

    Raises
    ------
    ValueError:
        If target_room is invalid, K < 0, or fewer than K dev captures exist.
    """
    target_room = str(target_room).strip().upper()
    if target_room not in ("A", "B"):
        raise ValueError(
            f"Invalid target_room: {target_room!r}. Allowed rooms: ['A', 'B']"
        )

    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")

    # Load / parse manifest rows
    if isinstance(manifest, (str, Path)):
        raw_rows = read_manifest(Path(manifest))
    elif hasattr(manifest, "itertuples"):  # DataFrame
        raw_rows = []
        for row in manifest.itertuples(index=False):
            d = row._asdict()
            raw_rows.append(
                SupportRow(
                    capture_id=str(d.get("capture_id", "")),
                    room=str(d.get("room", "")),
                    activity=str(d.get("activity", "")),
                    split=str(d.get("split", "")),
                    source_file=str(d.get("source_file", "")),
                    n_packets_raw=int(d.get("n_packets_raw", 0)),
                    n_packets_clean=int(d.get("n_packets_clean", 0)),
                    intermediate_path=str(d.get("intermediate_path", "")),
                )
            )
    else:
        raw_rows = list(manifest)

    # Standardize rows into SupportRow objects
    standardized_rows: List[SupportRow] = []
    for r in raw_rows:
        if isinstance(r, SupportRow):
            standardized_rows.append(r)
        elif isinstance(r, ManifestRow):
            standardized_rows.append(
                SupportRow(
                    capture_id=r.capture_id,
                    room=r.room,
                    activity=r.activity,
                    split=r.split,
                    source_file=r.source_file,
                    n_packets_raw=r.n_packets_raw,
                    n_packets_clean=r.n_packets_clean,
                    intermediate_path=r.intermediate_path,
                )
            )
        elif isinstance(r, dict):
            standardized_rows.append(
                SupportRow(
                    capture_id=str(r.get("capture_id", "")),
                    room=str(r.get("room", "")),
                    activity=str(r.get("activity", "")),
                    split=str(r.get("split", "")),
                    source_file=str(r.get("source_file", "")),
                    n_packets_raw=int(r.get("n_packets_raw", 0)),
                    n_packets_clean=int(r.get("n_packets_clean", 0)),
                    intermediate_path=str(r.get("intermediate_path", "")),
                )
            )
        else:
            standardized_rows.append(
                SupportRow(
                    capture_id=str(getattr(r, "capture_id", "")),
                    room=str(getattr(r, "room", "")),
                    activity=str(getattr(r, "activity", "")),
                    split=str(getattr(r, "split", "")),
                    source_file=str(getattr(r, "source_file", "")),
                    n_packets_raw=int(getattr(r, "n_packets_raw", 0)),
                    n_packets_clean=int(getattr(r, "n_packets_clean", 0)),
                    intermediate_path=str(getattr(r, "intermediate_path", "")),
                )
            )

    exclusive_activities = EXCLUSIVE_ACTIVITIES[target_room]

    # Filter strictly:
    # 1. room == target_room
    # 2. split == "dev" (NEVER validation)
    # 3. activity in exclusive_activities (NEVER shared activities)
    filtered = [
        r for r in standardized_rows
        if r.room == target_room
        and r.split == "dev"
        and r.activity in exclusive_activities
        and r.activity not in SHARED_ACTIVITIES
    ]

    # Deduplicate by capture_id so one capture counts as one shot
    captures_by_activity: Dict[str, Dict[str, SupportRow]] = {
        act: {} for act in exclusive_activities
    }
    for r in filtered:
        if r.capture_id not in captures_by_activity[r.activity]:
            captures_by_activity[r.activity][r.capture_id] = r

    rng = np.random.RandomState(seed)
    sampled_rows: List[SupportRow] = []

    # Iterate activities in sorted order for deterministic RNG consumption
    for act in sorted(exclusive_activities):
        candidates = list(captures_by_activity[act].values())
        candidates.sort(key=lambda x: x.capture_id)
        n_avail = len(candidates)
        if K > n_avail:
            raise ValueError(
                f"Cannot sample K={K} captures for activity {act!r} in room {target_room!r}: "
                f"only {n_avail} distinct dev captures available."
            )
        if K > 0:
            chosen_indices = rng.choice(n_avail, size=K, replace=False)
            for idx in sorted(chosen_indices):
                sampled_rows.append(candidates[idx])

    result = SupportSet(sampled_rows)
    if as_dataframe:
        return result.to_dataframe()
    return result


# ---------------------------------------------------------------------------
# Prototype and Covariance Calculation
# ---------------------------------------------------------------------------

def compute_source_prototypes(
    embeddings: np.ndarray,
    labels: Sequence[str] | np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute mean embedding per class across all source development labels."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(labels) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(embeddings)} embeddings vs {len(labels)} labels"
        )
    prototypes = {}
    for cls in np.unique(labels):
        mask = labels == cls
        prototypes[str(cls)] = np.mean(embeddings[mask], axis=0)
    return prototypes


def compute_target_exclusive_prototypes(
    embeddings: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    capture_ids: Optional[Sequence[str] | np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute target-exclusive prototypes by averaging windows within each capture first,
    then averaging across captures (one capture counts as one shot)."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(labels) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(embeddings)} embeddings vs {len(labels)} labels"
        )
    if capture_ids is not None:
        capture_ids = np.asarray(capture_ids)
        if len(capture_ids) != len(embeddings):
            raise ValueError("Length mismatch between capture_ids and embeddings")

    prototypes = {}
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_emb = embeddings[cls_mask]

        if capture_ids is None:
            prototypes[str(cls)] = np.mean(cls_emb, axis=0)
        else:
            cls_cids = capture_ids[cls_mask]
            unique_cids = np.unique(cls_cids)
            capture_means = []
            for cid in unique_cids:
                cid_mask = cls_cids == cid
                cid_mean = np.mean(cls_emb[cid_mask], axis=0)
                capture_means.append(cid_mean)
            prototypes[str(cls)] = np.mean(capture_means, axis=0)

    return prototypes


def compute_pooled_covariance(
    embeddings: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    alpha: Optional[float] = 0.1,
    shrinkage: str = "shrunken",
) -> np.ndarray:
    """Compute pooled within-class covariance matrix with optional shrinkage."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    N, D = embeddings.shape

    unique_labels = np.unique(labels)
    residuals = []
    for cls in unique_labels:
        mask = labels == cls
        X_c = embeddings[mask]
        mu_c = np.mean(X_c, axis=0, keepdims=True)
        residuals.append(X_c - mu_c)

    tilde_X = np.vstack(residuals) if residuals else np.zeros((0, D), dtype=np.float32)

    if shrinkage == "ledoit_wolf" or (alpha is None and shrinkage != "none"):
        if tilde_X.shape[0] > 1:
            cov, _ = ledoit_wolf(tilde_X)
            return cov.astype(np.float32)
        return np.eye(D, dtype=np.float32)

    dof = max(1, N - len(unique_labels))
    cov = (tilde_X.T @ tilde_X) / dof

    if alpha is None or alpha <= 0.0 or shrinkage == "none":
        cov += 1e-5 * np.eye(D, dtype=np.float32)
        return cov.astype(np.float32)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    nu = float(np.trace(cov) / D)
    if nu <= 0 or not np.isfinite(nu):
        nu = 1.0
    target = nu * np.eye(D, dtype=np.float32)
    shrunk_cov = (1.0 - alpha) * cov + alpha * target + 1e-5 * np.eye(D, dtype=np.float32)
    return shrunk_cov.astype(np.float32)


# ---------------------------------------------------------------------------
# PrototypeHead
# ---------------------------------------------------------------------------

class PrototypeHead:
    """Prototypical classifier head for few-shot cross-room activity recognition.

    Supports:
    - Source prototypes (mean embedding per class across source development labels)
    - Target-exclusive prototypes (averaging windows within capture first, then across captures)
    - Masking predictions strictly to target room's valid classes
    - Gaussian classifier with pooled shrunken covariance (alpha configurable)
    - Cosine similarity nearest centroid classifier
    - predict(X) -> (label, confidence)
    """

    def __init__(
        self,
        target_room: str,
        metric: str = "gaussian",
        alpha: Optional[float] = 0.1,
        shrinkage: str = "shrunken",
        temperature: float = 1.0,
        cfg: Optional[object] = None,
        source_embeddings: Optional[np.ndarray] = None,
        source_labels: Optional[Sequence[str] | np.ndarray] = None,
        target_support_embeddings: Optional[np.ndarray] = None,
        target_support_labels: Optional[Sequence[str] | np.ndarray] = None,
        target_support_capture_ids: Optional[Sequence[str] | np.ndarray] = None,
        prototypes: Optional[Dict[str, np.ndarray]] = None,
        covariance: Optional[np.ndarray] = None,
    ):
        self.target_room = str(target_room).strip().upper()
        if self.target_room not in ("A", "B"):
            raise ValueError(
                f"Invalid target_room: {target_room!r}. Allowed rooms: ['A', 'B']"
            )

        if cfg is not None and hasattr(cfg, "fewshot"):
            fs_cfg = cfg.fewshot
            if hasattr(fs_cfg, "distance_metric") and metric == "gaussian":
                if fs_cfg.distance_metric == "cosine":
                    metric = "cosine"
            if hasattr(fs_cfg, "shrinkage"):
                if fs_cfg.shrinkage == "ledoit_wolf":
                    shrinkage = "ledoit_wolf"
            if hasattr(fs_cfg, "classifier") and metric == "gaussian":
                if fs_cfg.classifier == "ncm":
                    metric = "cosine"

        self.metric = metric.lower()
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.temperature = float(temperature)

        self.classes_: List[str] = sorted(VALID_ACTIVITIES[self.target_room])
        self.prototypes_: Dict[str, np.ndarray] = {}
        self.cov_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._M_norm: Optional[np.ndarray] = None

        if prototypes is not None or source_embeddings is not None:
            self.fit(
                source_embeddings=source_embeddings,
                source_labels=source_labels,
                target_support_embeddings=target_support_embeddings,
                target_support_labels=target_support_labels,
                target_support_capture_ids=target_support_capture_ids,
                prototypes=prototypes,
                covariance=covariance,
            )

    def fit(
        self,
        source_embeddings: Optional[np.ndarray] = None,
        source_labels: Optional[Sequence[str] | np.ndarray] = None,
        target_support_embeddings: Optional[np.ndarray] = None,
        target_support_labels: Optional[Sequence[str] | np.ndarray] = None,
        target_support_capture_ids: Optional[Sequence[str] | np.ndarray] = None,
        *,
        prototypes: Optional[Dict[str, np.ndarray]] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> "PrototypeHead":
        """Fit prototypes and covariance model.

        Parameters
        ----------
        source_embeddings:
            Embeddings from source development set (shape N_src, D).
        source_labels:
            Class labels for source development set (shape N_src,).
        target_support_embeddings:
            Embeddings from target support captures (shape N_tgt, D).
        target_support_labels:
            Class labels for target support captures (shape N_tgt,).
        target_support_capture_ids:
            Capture IDs for target support captures (shape N_tgt,).
        prototypes:
            Optional pre-computed prototype dictionary {class_name: vector}.
        covariance:
            Optional pre-computed covariance matrix (shape D, D).
        """
        combined_prototypes: Dict[str, np.ndarray] = {}

        if prototypes is not None:
            for k, v in prototypes.items():
                combined_prototypes[k] = np.asarray(v, dtype=np.float32)

        # 1. Source prototypes: mean embedding per class from source development labels
        if source_embeddings is not None and source_labels is not None:
            src_protos = compute_source_prototypes(source_embeddings, source_labels)
            for k, v in src_protos.items():
                if k not in combined_prototypes:
                    combined_prototypes[k] = v

        # 2. Target-exclusive prototypes: average windows within capture first, then across captures
        if target_support_embeddings is not None and target_support_labels is not None:
            tgt_protos = compute_target_exclusive_prototypes(
                target_support_embeddings,
                target_support_labels,
                target_support_capture_ids,
            )
            for k, v in tgt_protos.items():
                combined_prototypes[k] = v

        # Mask predictions strictly to target room's valid classes
        valid_set = set(VALID_ACTIVITIES[self.target_room])
        self.prototypes_ = {
            k: v for k, v in combined_prototypes.items() if k in valid_set
        }

        present_classes = sorted(self.prototypes_.keys())
        if not present_classes:
            raise ValueError(
                f"No prototypes available for valid classes of room {self.target_room}. "
                f"Valid classes: {sorted(valid_set)}"
            )
        self.classes_ = present_classes

        # 3. Covariance estimation
        D = next(iter(self.prototypes_.values())).shape[0]
        if covariance is not None:
            self.cov_ = np.asarray(covariance, dtype=np.float32)
        elif source_embeddings is not None and source_labels is not None:
            self.cov_ = compute_pooled_covariance(
                source_embeddings,
                source_labels,
                alpha=self.alpha,
                shrinkage=self.shrinkage,
            )
        else:
            self.cov_ = np.eye(D, dtype=np.float32)

        try:
            self.cov_inv_ = np.linalg.pinv(self.cov_)
        except Exception:
            self.cov_inv_ = np.eye(D, dtype=np.float32)

        M = np.stack([self.prototypes_[c] for c in self.classes_], axis=0)  # (C, D)
        self._W = (self.cov_inv_ @ M.T)  # (D, C)
        self._b = -0.5 * np.sum(M * (self.cov_inv_ @ M.T).T, axis=1)  # (C,)

        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._M_norm = M / norms  # (C, D)

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        metric: Optional[str] = None,
    ) -> np.ndarray:
        """Compute posterior probabilities masked to target room's valid classes.

        Returns (N, len(self.classes_)) float array.
        """
        if not self.prototypes_:
            raise RuntimeError("PrototypeHead must be fitted before predict.")

        X = np.asarray(X, dtype=np.float32)
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")

        D = self._W.shape[0] if self._W is not None else self._M_norm.shape[1]
        if X.shape[1] != D:
            raise ValueError(f"Feature dimension mismatch: expected {D}, got {X.shape[1]}")

        active_metric = (metric or self.metric).lower()

        if active_metric in ("cosine", "ncm"):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            X_norm = X / norms
            sims = X_norm @ self._M_norm.T
            logits = sims / max(1e-6, self.temperature)
        else:
            logits = X @ self._W + self._b

        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(
        self,
        X: np.ndarray,
        metric: Optional[str] = None,
    ) -> Union[Tuple[str, float], Tuple[np.ndarray, np.ndarray]]:
        """Predict activity label and confidence.

        Predictions are strictly masked to the target room's valid classes.

        Parameters
        ----------
        X:
            Query embedding(s), shape (D,) or (N, D).
        metric:
            Optional override: "gaussian" or "cosine".

        Returns
        -------
        If X is 1D (single query):
            (label: str, confidence: float)
        If X is 2D (batch):
            (labels: np.ndarray[str], confidences: np.ndarray[float])
        """
        X_arr = np.asarray(X, dtype=np.float32)
        single = X_arr.ndim == 1

        probs = self.predict_proba(X_arr, metric=metric)
        best_indices = np.argmax(probs, axis=1)
        best_confs = np.max(probs, axis=1)

        labels = np.array([self.classes_[i] for i in best_indices], dtype=object)

        if single:
            return str(labels[0]), float(best_confs[0])
        return labels, best_confs.astype(np.float32)