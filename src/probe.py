"""
src/probe.py
============
Linear evaluation probe for Task 2 – Self-supervised representation quality.

Overview
--------
A frozen encoder (or hand-crafted features) produces fixed embeddings for
every window; a ``LogisticRegression`` probe is fitted on the DEVELOPMENT
embeddings and scored on the VALIDATION embeddings.

Conditions swept by ``scripts/task2_eval.py``
---------------------------------------------
1. ``handcrafted``  – 524-d features from ``src.features`` (no encoder needed).
2. ``random``       – ``ResNet1DEncoder(random_init=True)``; random weights,
                      no training.
3. ``vicreg_aug``   – encoder trained with VICReg + augmentation.
4. ``vicreg_noaug`` – encoder trained with VICReg, no augmentation.
5. ``masked``       – encoder trained with masked auto-encoding.

For conditions 2-5 the encoder is **frozen** (``eval()`` mode, no gradients).

Embedding normalisation
-----------------------
*After* extracting embeddings from the frozen encoder:
  1. L2-normalise each embedding vector (unit sphere).
  2. Fit a ``StandardScaler`` on the DEVELOPMENT embeddings.
  3. Apply the fitted scaler to both dev and val embeddings.

Both steps use development statistics only (no leakage).

Probe
-----
``LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')``.

Metrics
-------
- ``macro_f1``          – macro-averaged F1 across activity classes.
- ``balanced_accuracy`` – balanced accuracy (sklearn).

Scope of training
-----------------
The probe can be trained **jointly** (both rooms together) or **per room**.
``scripts/task2_eval.py`` controls this via ``--scope {joint|per_room|both}``.

Public API
----------
``EmbeddingExtractor(cfg, condition, checkpoint_dir)``
    ``extract(index_df, windows_dir, split, room=None) -> (np.ndarray, np.ndarray, np.ndarray)``
        Returns ``(X, y, labels)`` where ``X.shape = (N, emb_dim)``,
        ``y`` contains integer-encoded activity labels, and ``labels`` contains strings.

``LinearProbe(cfg=None, C=1.0)``
    ``fit(X_dev, y_dev)``
    ``score(X_val, y_val) -> dict``
    ``predict_proba(X) -> np.ndarray``
    ``label_encoder`` – fitted ``LabelEncoder`` (activity strings ↔ ints)

``run_probe(cfg, condition, index_df, windows_dir, scope, checkpoint_dir)``
    Convenience function: extract → normalise → fit probe → score.
    Returns a list of result-row dicts (ready for CSV).

``ProbeNormalizer``
    Handles L2 + StandardScaler in a single object.

Notes
-----
* If the encoder checkpoint for a condition does not exist,
  a ``FileNotFoundError`` is raised with a clear message.
* ``handcrafted`` uses ``src.features.extract_features`` directly and never
  touches the ``ResNet1DEncoder``.
* Torch is imported lazily so the module remains importable even when
  torch is not installed (handcrafted path only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Conditions that need an encoder checkpoint
CHECKPOINT_CONDITIONS = frozenset({"vicreg_aug", "vicreg_noaug", "masked"})

# All valid condition names
ALL_CONDITIONS = ("handcrafted", "random", "vicreg_aug", "vicreg_noaug", "masked")

# Checkpoint filename for each condition (matches ssl_train.py run_name convention)
_CKPT_NAME: Dict[str, str] = {
    "vicreg_aug":   "vicreg_aug.pt",
    "vicreg_noaug": "vicreg_noaug.pt",
    "masked":       "masked.pt",
}


# ---------------------------------------------------------------------------
# Probe normaliser
# ---------------------------------------------------------------------------

class ProbeNormalizer:
    """L2-normalise then StandardScale embeddings.

    Fit on development embeddings only; applied identically to val.
    """

    def __init__(self) -> None:
        self._scaler = None
        self._is_fit = False

    def fit(self, X: np.ndarray) -> "ProbeNormalizer":
        """Fit on development embeddings ``X`` shape ``(N, D)``."""
        from sklearn.preprocessing import StandardScaler
        X_arr = np.asarray(X, dtype=np.float32)
        X_l2  = self._l2(X_arr)
        self._scaler = StandardScaler()
        self._scaler.fit(X_l2)
        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """L2-normalise then StandardScale ``X``."""
        if not self._is_fit:
            raise RuntimeError("ProbeNormalizer.fit() must be called first.")
        X_arr = np.asarray(X, dtype=np.float32)
        was_1d = (X_arr.ndim == 1)
        X_l2 = self._l2(X_arr)
        res = self._scaler.transform(X_l2).astype(np.float32)
        return res[0] if was_1d else res

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation (unit sphere)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (X / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding extractor
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """Extract fixed-length embeddings from a given condition.

    Parameters
    ----------
    cfg:
        Loaded ``ExperimentConfig`` (can be ``None`` in smoke mode).
    condition:
        One of ``ALL_CONDITIONS``.
    checkpoint_dir:
        Directory containing ``*.pt`` encoder weight files.
    batch_size:
        Windows per forward-pass batch (for encoder conditions).
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        cfg=None,
        condition: str = "random",
        checkpoint_dir: Optional[Path] = None,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        if condition not in ALL_CONDITIONS:
            raise ValueError(
                f"Unknown condition {condition!r}. "
                f"Valid: {ALL_CONDITIONS}"
            )
        self.cfg            = cfg
        self.condition      = condition
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.batch_size     = batch_size
        self.device         = device
        self._encoder       = None  # lazy-loaded

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""
        if self.condition == "handcrafted":
            from src.features import FEATURE_DIM
            return FEATURE_DIM
        emb_dim = 256  # default
        if self.cfg is not None:
            try:
                emb_dim = self.cfg.ssl.embedding_dim
            except Exception:
                pass
        return emb_dim

    def _get_encoder(self):
        """Return (and cache) the loaded/initialised encoder."""
        if self._encoder is not None:
            return self._encoder

        import torch
        from src.encoder import ResNet1DEncoder

        emb_dim = self.embedding_dim
        enc = ResNet1DEncoder(
            embedding_dim=emb_dim,
            random_init=(self.condition == "random"),
            verbose=False,
        )

        if self.condition in CHECKPOINT_CONDITIONS:
            if self.checkpoint_dir is None:
                raise FileNotFoundError(
                    f"checkpoint_dir must be specified for checkpoint condition '{self.condition}'."
                )
            ckpt_file = _CKPT_NAME[self.condition]
            ckpt_path = self.checkpoint_dir / ckpt_file
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Encoder checkpoint for condition '{self.condition}' "
                    f"not found: {ckpt_path}\n"
                    f"Run: python src/ssl_train.py --mode "
                    f"{'vicreg' if 'vicreg' in self.condition else 'masked'} "
                    f"--run_name {self.condition}"
                )
            state = torch.load(str(ckpt_path), map_location="cpu")
            # Handle both raw state_dict and wrapped checkpoints
            if isinstance(state, dict) and "encoder" in state:
                state = state["encoder"]
            enc.load_state_dict(state)
            log.info("Loaded encoder from %s", ckpt_path)

        enc.eval()
        enc.to(torch.device(self.device))
        self._encoder = enc
        return enc

    def extract(
        self,
        index_df,
        windows_dir: Path,
        split: str,
        room: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract embeddings for a given split (and optionally room).

        Parameters
        ----------
        index_df:
            Full windows index DataFrame.
        windows_dir:
            Root of ``derived/windows/`` directory.
        split:
            ``"dev"`` or ``"val"``.
        room:
            If provided, restrict to this room only.

        Returns
        -------
        X      : np.ndarray shape ``(N, embedding_dim)`` float32
        y      : np.ndarray shape ``(N,)`` int (activity label indices)
        labels : np.ndarray shape ``(N,)`` object (activity strings)
        """
        df = index_df[index_df["split"] == split].copy()
        if room is not None:
            df = df[df["room"] == room]

        if df.empty:
            dim = self.embedding_dim
            return (
                np.empty((0, dim), dtype=np.float32),
                np.empty((0,),     dtype=int),
                np.empty((0,),     dtype=object),
            )

        if self.condition == "handcrafted":
            return self._extract_handcrafted(df, windows_dir, split)
        else:
            return self._extract_encoder(df, windows_dir, split)

    def _extract_handcrafted(self, df, windows_dir: Path, split: str):
        from src.features import extract_features, FEATURE_DIM
        X_list, lab_list = [], []
        skipped = 0
        for _, row in df.iterrows():
            wpath = windows_dir / split / f"{row['window_id']}.npy"
            if not wpath.exists():
                skipped += 1
                continue
            try:
                win  = np.load(str(wpath)).astype(np.float32)
                feat = extract_features(win)
                X_list.append(feat)
                lab_list.append(str(row["activity"]))
            except Exception as exc:
                log.warning("Skip %s: %s", row["window_id"], exc)
                skipped += 1
        if skipped:
            log.warning("Skipped %d windows (split=%s)", skipped, split)
        if not X_list:
            return (
                np.empty((0, FEATURE_DIM), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=object),
            )
        X = np.vstack(X_list).astype(np.float32)
        labels = np.array(lab_list, dtype=object)
        y, _ = _encode_labels(labels)
        return X, y, labels

    def _extract_encoder(self, df, windows_dir: Path, split: str):
        import torch
        enc = self._get_encoder()
        device = torch.device(self.device)

        X_list, lab_list = [], []
        skipped = 0

        rows_list = list(df.iterrows())
        batch_wins, batch_labs = [], []

        def _flush():
            if not batch_wins:
                return
            t = torch.stack(batch_wins, dim=0).to(device)   # (B, T, 256)
            with torch.no_grad():
                emb = enc.encode(t).cpu().numpy()            # (B, D)
            X_list.append(emb)
            lab_list.extend(batch_labs)
            batch_wins.clear()
            batch_labs.clear()

        for _, row in rows_list:
            wpath = windows_dir / split / f"{row['window_id']}.npy"
            if not wpath.exists():
                skipped += 1
                continue
            try:
                win = np.load(str(wpath)).astype(np.float32)
                batch_wins.append(torch.from_numpy(win))
                batch_labs.append(str(row["activity"]))
            except Exception as exc:
                log.warning("Skip %s: %s", row["window_id"], exc)
                skipped += 1

            if len(batch_wins) >= self.batch_size:
                _flush()

        _flush()

        if skipped:
            log.warning("Skipped %d windows (split=%s, condition=%s)",
                        skipped, split, self.condition)

        if not X_list:
            dim = self.embedding_dim
            return (
                np.empty((0, dim), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=object),
            )

        X      = np.vstack(X_list).astype(np.float32)
        labels = np.array(lab_list, dtype=object)
        y, _   = _encode_labels(labels)
        return X, y, labels


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe:
    """LogisticRegression probe on fixed embeddings.

    Parameters
    ----------
    cfg:
        Loaded ``ExperimentConfig`` (used for random_state; can be None).
    C:
        Regularisation strength (LR inverse).
    """

    def __init__(self, cfg=None, C: float = 1.0) -> None:
        self.cfg           = cfg
        self.C             = C
        self._clf          = None
        self.label_encoder = None
        self._is_fit       = False

    def fit(
        self,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> "LinearProbe":
        """Fit probe on development embeddings.

        Parameters
        ----------
        X_dev:
            Shape ``(N, D)`` already normalised.
        y_dev:
            Integer activity labels ``(N,)`` or string labels.

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression
        X_arr = np.asarray(X_dev, dtype=np.float32)
        y_arr = np.asarray(y_dev)

        if y_arr.dtype.kind in ("U", "O", "S"):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y_arr = self.label_encoder.fit_transform(y_arr)

        seed = 42
        if self.cfg is not None:
            try:
                seed = self.cfg.seeds.global_seed
            except Exception:
                pass
        self._clf = LogisticRegression(
            C=self.C,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
        self._clf.fit(X_arr, y_arr)
        self._is_fit = True
        return self

    def score(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Score probe on validation embeddings.

        Parameters
        ----------
        X_val:
            Shape ``(N, D)`` already normalised with dev statistics.
        y_val:
            Integer activity labels ``(N,)`` or string labels.

        Returns
        -------
        dict with keys ``macro_f1`` and ``balanced_accuracy``.
        """
        from sklearn.metrics import f1_score, balanced_accuracy_score
        if not self._is_fit:
            raise RuntimeError("LinearProbe.fit() must be called first.")

        X_arr = np.asarray(X_val, dtype=np.float32)
        y_arr = np.asarray(y_val)

        if y_arr.dtype.kind in ("U", "O", "S"):
            if self.label_encoder is None:
                raise ValueError(
                    "String labels passed to score(), but probe was fitted with integer labels."
                )
            known = set(self.label_encoder.classes_)
            mask = np.isin(y_arr, list(known))
            X_arr = X_arr[mask]
            y_arr = self.label_encoder.transform(y_arr[mask])

        # Restrict to labels seen during training
        train_classes = set(self._clf.classes_)
        mask = np.isin(y_arr, list(train_classes))
        if not mask.all():
            n_unseen = (~mask).sum()
            log.warning("%d val samples have unseen labels – excluded.", n_unseen)
        X_v = X_arr[mask]
        y_v = y_arr[mask]

        if len(y_v) == 0:
            return {"macro_f1": 0.0, "balanced_accuracy": 0.0}

        y_pred = self._clf.predict(X_v)
        return {
            "macro_f1":          float(f1_score(y_v, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_v, y_pred)),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix ``(N, n_classes)``."""
        if not self._is_fit:
            raise RuntimeError("LinearProbe.fit() must be called first.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr[np.newaxis, :]
        return self._clf.predict_proba(X_arr)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_probe(
    cfg,
    condition: str,
    index_df,
    windows_dir: Path,
    scope: str,                    # "joint" | "per_room" | "both"
    checkpoint_dir: Optional[Path] = None,
    device: str = "cpu",
    K: int = -1,
) -> List[Dict]:
    """Extract, normalise, probe, and return result rows.

    Parameters
    ----------
    cfg:
        ``ExperimentConfig`` or None.
    condition:
        One of ``ALL_CONDITIONS``.
    index_df:
        Full windows index DataFrame.
    windows_dir:
        Root of windows directory.
    scope:
        ``"joint"``    – one probe trained on all rooms.
        ``"per_room"`` – one probe per room.
        ``"both"``     – run both modes.
    checkpoint_dir:
        Directory for encoder ``.pt`` files.
    device:
        Torch device.
    K:
        Few-shot K value for the result rows (use -1 for the full probe).

    Returns
    -------
    List of dicts with keys:
        experiment, condition, scope, target_room, K, metric, value
    """
    extractor = EmbeddingExtractor(
        cfg=cfg,
        condition=condition,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    scopes = []
    if scope in ("joint", "both"):
        scopes.append("joint")
    if scope in ("per_room", "both"):
        scopes.append("per_room")

    rooms = ["A", "B"]
    result_rows: List[Dict] = []

    for sc in scopes:
        if sc == "joint":
            rows = _probe_joint(cfg, extractor, index_df, windows_dir, condition, K)
            result_rows.extend(rows)
        else:
            for room in rooms:
                rows = _probe_per_room(
                    cfg, extractor, index_df, windows_dir, condition, room, K
                )
                result_rows.extend(rows)

    return result_rows


def _probe_joint(cfg, extractor, index_df, windows_dir, condition, K):
    log.info("[%s] Extracting joint dev embeddings ...", condition)
    X_dev, _, lab_dev = extractor.extract(index_df, windows_dir, "dev")
    log.info("[%s] Extracting joint val embeddings ...", condition)
    X_val, _, lab_val = extractor.extract(index_df, windows_dir, "val")

    if X_dev.shape[0] == 0 or X_val.shape[0] == 0:
        log.warning("[%s] joint: no data for dev or val.", condition)
        return []

    norm = ProbeNormalizer()
    X_dev_n = norm.fit_transform(X_dev)
    X_val_n = norm.transform(X_val)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_dev_enc = le.fit_transform(lab_dev)

    probe = LinearProbe(cfg=cfg)
    probe.label_encoder = le
    probe.fit(X_dev_n, y_dev_enc)

    known = set(le.classes_)
    mask  = np.isin(lab_val, list(known))
    if not mask.all():
        log.warning(
            "[%s] joint: %d val samples have unseen labels – excluded.",
            condition, (~mask).sum(),
        )
    X_val_filt = X_val_n[mask]
    y_val_enc  = le.transform(lab_val[mask])

    scores = probe.score(X_val_filt, y_val_enc)

    log.info(
        "[%s] joint  macro_f1=%.4f  bal_acc=%.4f  dev=%d  val=%d",
        condition, scores["macro_f1"], scores["balanced_accuracy"],
        X_dev.shape[0], X_val_filt.shape[0],
    )
    return _make_rows(condition, "joint", "joint", K, scores)


def _probe_per_room(cfg, extractor, index_df, windows_dir, condition, room, K):
    log.info("[%s] room=%s: Extracting dev embeddings ...", condition, room)
    X_dev, _, lab_dev = extractor.extract(index_df, windows_dir, "dev", room=room)
    log.info("[%s] room=%s: Extracting val embeddings ...", condition, room)
    X_val, _, lab_val = extractor.extract(index_df, windows_dir, "val", room=room)

    if X_dev.shape[0] == 0 or X_val.shape[0] == 0:
        log.warning("[%s] room=%s: no data for dev or val.", condition, room)
        return []

    norm = ProbeNormalizer()
    X_dev_n = norm.fit_transform(X_dev)
    X_val_n = norm.transform(X_val)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_dev_enc = le.fit_transform(lab_dev)

    probe = LinearProbe(cfg=cfg)
    probe.label_encoder = le
    probe.fit(X_dev_n, y_dev_enc)

    known = set(le.classes_)
    mask  = np.isin(lab_val, list(known))
    if not mask.all():
        log.warning(
            "[%s] room=%s: %d val samples have unseen labels – excluded.",
            condition, room, (~mask).sum(),
        )
    X_val_filt = X_val_n[mask]
    y_val_enc  = le.transform(lab_val[mask])

    scores = probe.score(X_val_filt, y_val_enc)

    log.info(
        "[%s] room=%-2s  macro_f1=%.4f  bal_acc=%.4f  dev=%d  val=%d",
        condition, room, scores["macro_f1"], scores["balanced_accuracy"],
        X_dev.shape[0], X_val_filt.shape[0],
    )
    return _make_rows(condition, "per_room", room, K, scores)


def _make_rows(condition, scope, target_room, K, scores):
    rows = []
    for metric, value in scores.items():
        rows.append({
            "experiment":  f"probe_{condition}",
            "condition":   condition,
            "scope":       scope,
            "target_room": target_room,
            "K":           K,
            "metric":      metric,
            "value":       f"{value:.6f}",
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, object]:
    """Encode string activity labels to integers.

    Returns
    -------
    y     : np.ndarray int
    le    : fitted LabelEncoder
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return y.astype(int), le
