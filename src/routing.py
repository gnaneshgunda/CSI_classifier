"""
src/routing.py
==============
Environment routing for the CSI-HAR pipeline.

Overview
--------
Before computing embeddings (or before feeding embeddings into the activity
classifier), the pipeline needs to decide **which room's normalisation
statistics** to use for a given unlabelled test window.

``RoomRouter`` wraps a trained sklearn room-classifier (any object that
exposes ``predict_proba(X) -> (N, n_classes)`` and
``classes_ -> array of room labels``) and implements four routing strategies
controlled by ``cfg.routing.mode``:

``"predicted_hard"``
    Hard argmax routing: choose the room with the highest probability from
    the classifier.  If the winning probability is below
    ``cfg.routing.confidence_threshold``, fall back to ``"none"`` (global
    statistics) rather than committing to a low-confidence prediction.

``"predicted_soft"``
    Soft / probabilistic routing: return the full probability vector so that
    downstream code (e.g. ``Normalizer.transform``) can compute a weighted
    mixture of per-room statistics.  No confidence threshold applied here —
    the caller receives the raw probabilities.

``"oracle"``
    Return the ground-truth room label directly.  This is the upper-bound
    ablation; it requires the true room to be provided at call time.

``"none"``
    Skip routing entirely — return ``room_probs=None`` and ``room=None`` so
    that ``Normalizer.transform`` falls back to global (all-room) statistics.

Design goals
------------
* **Mode is a config switch** so ablations require no code changes.
* A single ``route(features, true_room)`` method covers all modes.
* ``RoomRouter`` is stateless beyond the wrapped sklearn classifier and the
  config; it does not modify either.
* Works for single samples (``features.ndim == 1``) or batches.

Public API
----------
``RoomRouter(classifier, cfg)``
    ``route(features, true_room=None)``
        → ``RoutingResult(room, room_probs, mode_used)``
    ``route_batch(features_batch, true_rooms=None)``
        → ``list[RoutingResult]``

``RoutingResult``
    Named dataclass with fields:
    - ``room``       : str | None  – chosen room (``"A"`` or ``"B"``), or None.
    - ``room_probs`` : ndarray (n_rooms,) | None – probability vector, or None.
    - ``mode_used``  : str – actual mode applied (may differ from config when
                       confidence fallback triggers).

Notes
-----
* ``features`` passed to ``route()`` must match whatever feature space the
  sklearn classifier was trained on (e.g. raw normalised windows flattened, or
  embeddings).
* The classifier's ``classes_`` attribute is used to map probability columns
  to room labels; so the classifier must expose it (all sklearn estimators do).
* The room-probability vector is always ordered ``["A", "B"]`` regardless of
  the classifier's internal class ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config import ExperimentConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOMS: Tuple[str, ...] = ("A", "B")


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Result of a single routing decision.

    Attributes
    ----------
    room : str or None
        Chosen room label (``"A"`` or ``"B"``), or ``None`` when
        ``mode_used == "none"``.
    room_probs : ndarray shape (2,) or None
        Probability vector ordered ``[P(A), P(B)]``, or ``None`` when
        ``mode_used in {"oracle", "none"}``.
    mode_used : str
        The routing mode that was actually applied.  May differ from the
        configured mode when the confidence-fallback fires
        (``"predicted_hard"`` → ``"none"``).
    """
    room:       Optional[str]
    room_probs: Optional[np.ndarray]
    mode_used:  str


# ---------------------------------------------------------------------------
# RoomRouter
# ---------------------------------------------------------------------------

class RoomRouter:
    """Wraps a sklearn room-classifier and implements routing modes.

    Parameters
    ----------
    classifier:
        A **fitted** sklearn estimator (or compatible object) that exposes:
          - ``.predict_proba(X)`` → ``(N, n_classes)`` float array
          - ``.classes_``         → array-like of room labels (``"A"``, ``"B"``)
        Pass ``None`` only when ``cfg.routing.mode`` is ``"oracle"`` or
        ``"none"`` (classifier is not called in those modes).
    cfg:
        Full ``ExperimentConfig``.  Reads ``cfg.routing.mode`` and
        ``cfg.routing.confidence_threshold``.
    """

    def __init__(self, classifier, cfg: ExperimentConfig) -> None:
        self._clf              = classifier
        self._mode             = cfg.routing.mode
        self._conf_threshold   = cfg.routing.confidence_threshold
        self._cfg              = cfg

        # Validate classifier is present when needed
        if self._mode in ("predicted_hard", "predicted_soft") and classifier is None:
            raise ValueError(
                f"routing.mode='{self._mode}' requires a fitted classifier, "
                "but classifier=None was passed."
            )
        if self._mode not in ("predicted_hard", "predicted_soft", "oracle", "none"):
            raise ValueError(
                f"Unknown routing.mode {self._mode!r}. "
                "Expected one of: 'predicted_hard', 'predicted_soft', 'oracle', 'none'."
            )

        # Pre-compute the index mapping from classifier.classes_ → [A, B] order
        self._room_to_idx: Optional[dict] = None
        if classifier is not None and hasattr(classifier, "classes_"):
            classes = [str(c).upper() for c in classifier.classes_]
            self._room_to_idx = {r: classes.index(r) for r in _ROOMS if r in classes}

    # ------------------------------------------------------------------
    # Single-sample routing
    # ------------------------------------------------------------------

    def route(
        self,
        features: np.ndarray,
        true_room: Optional[str] = None,
    ) -> RoutingResult:
        """Route a single sample and return a ``RoutingResult``.

        Parameters
        ----------
        features:
            1-D feature vector ``(F,)`` or 2-D ``(1, F)``.  Used by the
            classifier in ``"predicted_hard"`` and ``"predicted_soft"`` modes.
            Ignored in ``"oracle"`` and ``"none"`` modes.
        true_room:
            Ground-truth room label.  Required for ``"oracle"`` mode.
            Ignored (but safe to pass) in other modes.

        Returns
        -------
        RoutingResult

        Raises
        ------
        ValueError
            If ``mode == "oracle"`` and ``true_room`` is ``None``.
        """
        mode = self._mode

        # ---- oracle -------------------------------------------------------
        if mode == "oracle":
            if true_room is None:
                raise ValueError(
                    "routing.mode='oracle' requires true_room to be provided."
                )
            room = str(true_room).upper()
            if room not in _ROOMS:
                raise ValueError(
                    f"Unknown room {true_room!r}; expected one of {_ROOMS}."
                )
            return RoutingResult(room=room, room_probs=None, mode_used="oracle")

        # ---- none ---------------------------------------------------------
        if mode == "none":
            return RoutingResult(room=None, room_probs=None, mode_used="none")

        # ---- classifier-based (predicted_hard / predicted_soft) -----------
        probs_ab = self._classify_single(features)     # shape (2,)  [P(A), P(B)]

        if mode == "predicted_soft":
            return RoutingResult(
                room=None,            # caller uses probs for soft mixture
                room_probs=probs_ab,
                mode_used="predicted_soft",
            )

        # predicted_hard
        best_idx  = int(np.argmax(probs_ab))
        best_prob = float(probs_ab[best_idx])
        best_room = _ROOMS[best_idx]

        if best_prob < self._conf_threshold:
            # Low confidence: fall back to global statistics
            log.debug(
                "[routing] hard: best_prob=%.3f < threshold=%.3f → fallback to none",
                best_prob, self._conf_threshold,
            )
            return RoutingResult(room=None, room_probs=probs_ab, mode_used="none")

        return RoutingResult(
            room=best_room,
            room_probs=probs_ab,
            mode_used="predicted_hard",
        )

    # ------------------------------------------------------------------
    # Batch routing
    # ------------------------------------------------------------------

    def route_batch(
        self,
        features_batch: np.ndarray,
        true_rooms: Optional[Sequence[Optional[str]]] = None,
    ) -> List[RoutingResult]:
        """Route a batch of samples.

        Parameters
        ----------
        features_batch:
            2-D array ``(N, F)``.  For ``"oracle"`` / ``"none"`` modes the
            features are ignored.
        true_rooms:
            Sequence of N room labels.  Required for ``"oracle"`` mode.

        Returns
        -------
        list of RoutingResult, length N.
        """
        N = features_batch.shape[0]
        if true_rooms is None:
            true_rooms = [None] * N

        mode = self._mode

        # ---- oracle -------------------------------------------------------
        if mode == "oracle":
            return [
                self.route(features_batch[i], true_rooms[i])
                for i in range(N)
            ]

        # ---- none ---------------------------------------------------------
        if mode == "none":
            return [RoutingResult(room=None, room_probs=None, mode_used="none")] * N

        # ---- classifier-based: single predict_proba call for the whole batch
        probs_all = self._classify_batch(features_batch)   # (N, 2) in [A, B] order

        results: List[RoutingResult] = []

        for i in range(N):
            probs_ab = probs_all[i]

            if mode == "predicted_soft":
                results.append(RoutingResult(
                    room=None,
                    room_probs=probs_ab,
                    mode_used="predicted_soft",
                ))
                continue

            # predicted_hard
            best_idx  = int(np.argmax(probs_ab))
            best_prob = float(probs_ab[best_idx])
            best_room = _ROOMS[best_idx]

            if best_prob < self._conf_threshold:
                results.append(RoutingResult(
                    room=None,
                    room_probs=probs_ab,
                    mode_used="none",
                ))
            else:
                results.append(RoutingResult(
                    room=best_room,
                    room_probs=probs_ab,
                    mode_used="predicted_hard",
                ))

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Configured routing mode."""
        return self._mode

    @property
    def confidence_threshold(self) -> float:
        """Confidence threshold for ``"predicted_hard"`` fallback."""
        return self._conf_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_single(self, features: np.ndarray) -> np.ndarray:
        """Run predict_proba on a single sample; return (2,) array [P(A), P(B)]."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]                        # (1, F)
        raw = self._clf.predict_proba(x)[0]             # (n_classes,)
        return self._reorder_probs(raw)

    def _classify_batch(self, features: np.ndarray) -> np.ndarray:
        """Run predict_proba on a batch; return (N, 2) array [P(A), P(B)]."""
        x   = np.asarray(features, dtype=np.float32)
        raw = self._clf.predict_proba(x)                 # (N, n_classes)
        return np.stack([self._reorder_probs(r) for r in raw], axis=0)

    def _reorder_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map classifier output columns → canonical [P(A), P(B)] order."""
        if self._room_to_idx is None:
            # Fallback: assume classifier outputs [A, B] in that order already
            if len(raw_probs) == 2:
                return raw_probs.astype(np.float32)
            raise RuntimeError(
                "Classifier has no .classes_ attribute; cannot determine "
                "room-to-probability mapping."
            )
        probs = np.array(
            [raw_probs[self._room_to_idx[r]] for r in _ROOMS],
            dtype=np.float32,
        )
        # Normalise to simplex (guard against numerical drift)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs


# ---------------------------------------------------------------------------
# Smoke test  (python -m src.routing)
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Self-test: synthetic classifier + all four routing modes."""
    from unittest.mock import MagicMock

    # ---- synthetic sklearn-style classifier --------------------------------
    clf = MagicMock()
    clf.classes_ = np.array(["A", "B"])

    def _predict_proba_confident(X):
        # Always confident: 90 % A
        return np.tile([0.9, 0.1], (len(X), 1))

    def _predict_proba_uncertain(X):
        # Low-confidence: 55 % A (below default threshold 0.60)
        return np.tile([0.55, 0.45], (len(X), 1))

    features = np.random.rand(5).astype(np.float32)

    for mode in ("predicted_hard", "predicted_soft", "oracle", "none"):
        cfg = MagicMock()
        cfg.routing.mode                 = mode
        cfg.routing.confidence_threshold = 0.60

        router = RoomRouter(clf, cfg)

        if mode == "oracle":
            res = router.route(features, true_room="B")
            assert res.room == "B", f"oracle: expected B, got {res.room}"
            assert res.room_probs is None
            assert res.mode_used == "oracle"
            print(f"[smoke] mode={mode:<16}  room={res.room}  probs={res.room_probs}  ✓")
            continue

        if mode == "none":
            res = router.route(features)
            assert res.room is None
            assert res.room_probs is None
            assert res.mode_used == "none"
            print(f"[smoke] mode={mode:<16}  room={res.room}  probs={res.room_probs}  ✓")
            continue

        if mode == "predicted_soft":
            clf.predict_proba = _predict_proba_confident
            router = RoomRouter(clf, cfg)
            res = router.route(features)
            assert res.room is None,                        "predicted_soft: room must be None"
            assert res.room_probs is not None,              "predicted_soft: probs must not be None"
            assert abs(res.room_probs[0] - 0.9) < 1e-4,   "predicted_soft: P(A) should be ~0.9"
            assert res.mode_used == "predicted_soft"
            print(f"[smoke] mode={mode:<16}  probs={res.room_probs}  ✓")
            continue

        # predicted_hard – confident case
        clf.predict_proba = _predict_proba_confident
        router = RoomRouter(clf, cfg)
        res = router.route(features)
        assert res.room == "A", f"predicted_hard confident: expected A, got {res.room}"
        assert res.mode_used == "predicted_hard"
        assert res.room_probs is not None and len(res.room_probs) == 2
        print(f"[smoke] mode={mode:<16} (confident)  room={res.room}  probs={res.room_probs}  ✓")

        # predicted_hard – low-confidence fallback
        clf.predict_proba = _predict_proba_uncertain
        router = RoomRouter(clf, cfg)
        res = router.route(features)
        assert res.room is None,         "predicted_hard fallback: room should be None"
        assert res.mode_used == "none",  "predicted_hard fallback: mode_used should be none"
        print(f"[smoke] mode={mode:<16} (fallback)   room={res.room}  mode_used={res.mode_used}  ✓")

    # ---- batch routing smoke -----------------------------------------------
    clf.predict_proba = _predict_proba_confident
    cfg = MagicMock()
    cfg.routing.mode                 = "predicted_hard"
    cfg.routing.confidence_threshold = 0.60
    router = RoomRouter(clf, cfg)
    batch_feats = np.random.rand(10, 128).astype(np.float32)
    batch_res   = router.route_batch(batch_feats)
    assert len(batch_res) == 10
    assert all(r.room == "A" for r in batch_res)
    print(f"[smoke] batch route_batch (N=10)  all rooms=A  ✓")

    print("[smoke] routing.py PASSED ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke()
