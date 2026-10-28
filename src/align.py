"""
src/align.py
============
Per-room embedding alignment for the CSI-HAR pipeline.

Overview
--------
After SSL pre-training the encoder produces embeddings that may carry a large
room-specific bias (different channel conditions, antenna geometry, etc.).
``EmbeddingAligner`` corrects this **in embedding space** before any downstream
classifier is trained, using only *unlabelled* development embeddings.

Two correction methods are supported (config: ``align.method``):

``"recentre"``
    Subtract each room's mean embedding so both rooms share a common centroid
    at the origin.  Fast, parameter-free, zero hyperparameters.

``"coral"``
    CORAL (Sun & Saenko, 2016) second-order statistics alignment.
    Room B is whitened (zero-mean, identity covariance) then re-coloured to
    match Room A's covariance:

        z_B' = (z_B − μ_B) @ W_B_inv @ W_A + μ_A

    where W_r = Σ_r^{+½}  (symmetric square root, via eigen-decomposition).
    Room A is the reference domain; Room B is aligned to it.  Both are then
    globally re-centred to the origin.

``"none"``
    Pass-through — embeddings are returned unchanged.

All statistics are estimated from **dev-split** embeddings only (no val
leakage).  The fitted transform is stored and reloaded for inference.

Persistence
-----------
``fit()`` writes statistics to ``derived/align_stats.npz`` and calls ``save()``
automatically.  ``load()`` reloads them.

Public API
----------
``EmbeddingAligner(cfg)``
    ``fit(embeddings, rooms)``       – compute statistics from dev embeddings.
    ``transform(embeddings, rooms)`` – apply the alignment.
    ``load()``                       – reload from disk.
    ``save()``                       – write to disk (called by fit()).

Notes
-----
* ``embeddings`` : ``np.ndarray`` shape ``(N, D)``, dtype float32.
* ``rooms``      : length-N sequence of ``"A"`` / ``"B"`` (case-insensitive).
* CORAL covariance is regularised: Σ_reg = Σ + ε·I  (ε = 1e-5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.config import ExperimentConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOMS: Tuple[str, ...] = ("A", "B")
_CORAL_REG: float = 1e-5           # regularisation added to cov diagonal
_MIN_ROOM_SAMPLES: int = 2         # minimum samples per room to fit stats


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlignStats:
    """Fitted alignment statistics.

    Attributes
    ----------
    method : str
        Alignment method these stats were fitted with.
    room_mean : dict[str, ndarray]
        Per-room mean embedding, shape ``(D,)``.
    global_mean : ndarray
        Mean over all dev embeddings, shape ``(D,)``.
    coral_W_A : ndarray or None
        Symmetric square root of Room A's covariance, ``(D, D)``.
        Present only when ``method == "coral"``.
    coral_W_B_inv : ndarray or None
        Symmetric inverse square root of Room B's covariance, ``(D, D)``.
        Present only when ``method == "coral"``.
    """
    method:        str
    room_mean:     Dict[str, np.ndarray] = field(default_factory=dict)
    global_mean:   np.ndarray            = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    coral_W_A:     Optional[np.ndarray]  = None
    coral_W_B_inv: Optional[np.ndarray]  = None


# ---------------------------------------------------------------------------
# Helper: symmetric matrix square root and inverse square root
# ---------------------------------------------------------------------------

def _sym_sqrt_and_inv(
    mat: np.ndarray,
    reg: float = _CORAL_REG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(M^{+1/2}, M^{-1/2})`` via eigen-decomposition.

    Parameters
    ----------
    mat:
        Symmetric PSD matrix, shape ``(D, D)``.
    reg:
        Ridge added to all eigenvalues before square-rooting.

    Returns
    -------
    sqrt_mat : ndarray ``(D, D)``
    inv_sqrt_mat : ndarray ``(D, D)``
    """
    mat = (mat + mat.T) * 0.5                             # symmetrise
    eigvals, eigvecs = np.linalg.eigh(mat)                # ascending order
    eigvals = np.maximum(eigvals, 0.0) + reg              # clip + regularise

    sqrt_vals     = np.sqrt(eigvals)
    inv_sqrt_vals = 1.0 / sqrt_vals

    sqrt_mat     = (eigvecs * sqrt_vals[np.newaxis, :])     @ eigvecs.T
    inv_sqrt_mat = (eigvecs * inv_sqrt_vals[np.newaxis, :]) @ eigvecs.T

    return sqrt_mat.astype(np.float32), inv_sqrt_mat.astype(np.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmbeddingAligner:
    """Per-room embedding aligner (re-centring and/or CORAL).

    Parameters
    ----------
    cfg:
        Full ``ExperimentConfig``.  Reads ``cfg.align.method`` and
        ``cfg.paths.derived_dir``.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self._cfg    = cfg
        self._method = cfg.align.method          # "recentre" | "coral" | "none"
        self._stats: Optional[AlignStats] = None

        derived = Path(cfg.paths.derived_dir)
        self._stats_path = derived / "align_stats.npz"

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        rooms: Sequence[str],
    ) -> "EmbeddingAligner":
        """Estimate alignment statistics from **dev** embeddings.

        Parameters
        ----------
        embeddings:
            Float32 array ``(N, D)`` — encoder output for all dev windows.
        rooms:
            N room labels (``"A"`` or ``"B"``, case-insensitive).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``_MIN_ROOM_SAMPLES`` is not met for any room, or if an unknown
            ``align.method`` is configured.
        """
        emb       = np.asarray(embeddings, dtype=np.float64)
        rooms_arr = np.array([r.upper() for r in rooms])

        # ------------------------------------------------------------------ #
        # "none" – trivial stats                                              #
        # ------------------------------------------------------------------ #
        if self._method == "none":
            D = emb.shape[1]
            self._stats = AlignStats(
                method="none",
                room_mean={r: np.zeros(D, dtype=np.float32) for r in _ROOMS},
                global_mean=np.zeros(D, dtype=np.float32),
            )
            self._save()
            log.info("[align] method=none – no statistics fitted")
            return self

        # ------------------------------------------------------------------ #
        # Per-room means                                                      #
        # ------------------------------------------------------------------ #
        if self._method not in ("recentre", "coral"):
            raise ValueError(
                f"Unknown align.method {self._method!r}. "
                "Expected one of: 'recentre', 'coral', 'none'."
            )

        room_mean: Dict[str, np.ndarray] = {}
        room_emb:  Dict[str, np.ndarray] = {}

        for r in _ROOMS:
            mask = rooms_arr == r
            n    = int(mask.sum())
            if n < _MIN_ROOM_SAMPLES:
                raise ValueError(
                    f"Room {r} has only {n} dev embeddings; "
                    f"need ≥ {_MIN_ROOM_SAMPLES} to fit alignment statistics."
                )
            room_emb[r]  = emb[mask]
            room_mean[r] = room_emb[r].mean(axis=0)
            log.info(
                "[align] Room %s: %d embeddings  mean_norm=%.4f  emb_std=%.4f",
                r, n,
                float(np.linalg.norm(room_mean[r])),
                float(room_emb[r].std(axis=0).mean()),
            )

        global_mean = emb.mean(axis=0)

        # ------------------------------------------------------------------ #
        # CORAL: symmetric (inverse) square roots of per-room covariances     #
        # ------------------------------------------------------------------ #
        coral_W_A: Optional[np.ndarray]     = None
        coral_W_B_inv: Optional[np.ndarray] = None

        if self._method == "coral":
            for r in _ROOMS:
                centred = room_emb[r] - room_mean[r][np.newaxis, :]
                n_r     = centred.shape[0]
                cov     = (centred.T @ centred) / max(n_r - 1, 1)

                if r == "A":
                    coral_W_A, _ = _sym_sqrt_and_inv(cov)
                    log.info(
                        "[align] CORAL W_A computed  D=%d  cond=%.2e",
                        cov.shape[0],
                        float(np.linalg.cond(cov + _CORAL_REG * np.eye(cov.shape[0]))),
                    )
                else:
                    _, coral_W_B_inv = _sym_sqrt_and_inv(cov)
                    log.info(
                        "[align] CORAL W_B_inv computed  D=%d  cond=%.2e",
                        cov.shape[0],
                        float(np.linalg.cond(cov + _CORAL_REG * np.eye(cov.shape[0]))),
                    )

        self._stats = AlignStats(
            method=self._method,
            room_mean={r: room_mean[r].astype(np.float32) for r in _ROOMS},
            global_mean=global_mean.astype(np.float32),
            coral_W_A=coral_W_A,
            coral_W_B_inv=coral_W_B_inv,
        )

        self._save()
        log.info(
            "[align] method=%s  fit complete – stats saved to %s",
            self._method, self._stats_path,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(
        self,
        embeddings: np.ndarray,
        rooms: Sequence[str],
    ) -> np.ndarray:
        """Apply the fitted alignment transform.

        Parameters
        ----------
        embeddings:
            Float32 array ``(N, D)``.
        rooms:
            N room labels (``"A"`` or ``"B"``).

        Returns
        -------
        np.ndarray
            Aligned embeddings, shape ``(N, D)``, dtype float32.

        Raises
        ------
        RuntimeError
            If ``fit()`` or ``load()`` has not been called first.
        """
        stats     = self._require_stats()
        emb       = np.asarray(embeddings, dtype=np.float32)
        rooms_arr = np.array([r.upper() for r in rooms])
        out       = emb.copy()

        if stats.method == "none":
            return out

        if stats.method == "recentre":
            # Subtract per-room centroid
            for r in _ROOMS:
                mask = rooms_arr == r
                if mask.any():
                    out[mask] -= stats.room_mean[r][np.newaxis, :]
            # Globally re-centre to origin
            out -= stats.global_mean[np.newaxis, :]
            return out

        if stats.method == "coral":
            if stats.coral_W_A is None or stats.coral_W_B_inv is None:
                raise RuntimeError(
                    "CORAL matrices are None – was fit() called with method='coral'?"
                )
            # Room A: re-centre only (A is the reference domain)
            mask_a = rooms_arr == "A"
            if mask_a.any():
                out[mask_a] = (
                    out[mask_a].astype(np.float64)
                    - stats.room_mean["A"].astype(np.float64)[np.newaxis, :]
                ).astype(np.float32)

            # Room B: whiten → re-colour to match A's covariance
            mask_b = rooms_arr == "B"
            if mask_b.any():
                z_b     = (
                    out[mask_b].astype(np.float64)
                    - stats.room_mean["B"].astype(np.float64)[np.newaxis, :]
                )
                W_B_inv = stats.coral_W_B_inv.astype(np.float64)
                W_A     = stats.coral_W_A.astype(np.float64)
                out[mask_b] = (z_b @ W_B_inv @ W_A).astype(np.float32)

            # Global re-centring
            out -= stats.global_mean[np.newaxis, :]
            return out

        raise ValueError(f"Unknown method {stats.method!r} stored in AlignStats.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write fitted statistics to ``derived/align_stats.npz``."""
        self._save()

    def load(self) -> "EmbeddingAligner":
        """Reload previously fitted statistics from ``derived/align_stats.npz``.

        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If the stats file does not exist.
        """
        if not self._stats_path.exists():
            raise FileNotFoundError(
                f"Alignment stats not found at {self._stats_path}. "
                "Run EmbeddingAligner.fit() first."
            )
        data   = np.load(str(self._stats_path), allow_pickle=False)
        method = str(data["method"].item())

        room_mean = {
            r: data[f"room_mean_{r}"].astype(np.float32)
            for r in _ROOMS
            if f"room_mean_{r}" in data
        }
        D = next(iter(room_mean.values())).shape[0] if room_mean else 1
        global_mean = (
            data["global_mean"].astype(np.float32) if "global_mean" in data
            else np.zeros(D, dtype=np.float32)
        )
        coral_W_A     = data["coral_W_A"].astype(np.float32)     if "coral_W_A"     in data else None
        coral_W_B_inv = data["coral_W_B_inv"].astype(np.float32) if "coral_W_B_inv" in data else None

        self._stats = AlignStats(
            method=method,
            room_mean=room_mean,
            global_mean=global_mean,
            coral_W_A=coral_W_A,
            coral_W_B_inv=coral_W_B_inv,
        )
        log.info("[align] Loaded stats from %s  (method=%s)", self._stats_path, method)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Configured alignment method (from ``cfg.align.method``)."""
        return self._method

    @property
    def stats(self) -> Optional[AlignStats]:
        """Fitted ``AlignStats``, or ``None`` before ``fit()``/``load()``."""
        return self._stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        stats = self._require_stats()
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "method":      np.array(stats.method),
            "global_mean": stats.global_mean,
        }
        for r in _ROOMS:
            if r in stats.room_mean:
                arrays[f"room_mean_{r}"] = stats.room_mean[r]
        if stats.coral_W_A     is not None:
            arrays["coral_W_A"]     = stats.coral_W_A
        if stats.coral_W_B_inv is not None:
            arrays["coral_W_B_inv"] = stats.coral_W_B_inv

        np.savez(str(self._stats_path), **arrays)

    def _require_stats(self) -> AlignStats:
        if self._stats is None:
            raise RuntimeError(
                "EmbeddingAligner has no fitted statistics. "
                "Call fit() or load() first."
            )
        return self._stats


# ---------------------------------------------------------------------------
# Smoke test  (python -m src.align)
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Self-test on synthetic two-room embeddings."""
    from unittest.mock import MagicMock

    D, N = 32, 300
    rng = np.random.default_rng(0)

    # Room A centred at origin; Room B shifted by +3 with 2× variance
    emb_A = rng.standard_normal((N, D)).astype(np.float32)
    emb_B = (rng.standard_normal((N, D)) * 2.0 + 3.0).astype(np.float32)

    embeddings = np.vstack([emb_A, emb_B])
    rooms      = ["A"] * N + ["B"] * N

    for method in ("recentre", "coral", "none"):
        cfg = MagicMock()
        cfg.align.method      = method
        cfg.paths.derived_dir = "/tmp"

        aligner = EmbeddingAligner(cfg)
        aligner.fit(embeddings, rooms)
        out = aligner.transform(embeddings, rooms)

        assert out.shape == embeddings.shape, f"[{method}] shape mismatch"
        assert out.dtype == np.float32,       f"[{method}] dtype != float32"

        out_A = out[:N]
        out_B = out[N:]
        if method in ("recentre", "coral"):
            gap = float(np.linalg.norm(out_A.mean(0) - out_B.mean(0)))
            assert gap < 0.5, (
                f"[{method}] centroid gap still large after alignment: {gap:.4f}"
            )

        print(
            f"[smoke] method={method:<10}  "
            f"A_centroid_norm={np.linalg.norm(out_A.mean(0)):.4f}  "
            f"B_centroid_norm={np.linalg.norm(out_B.mean(0)):.4f}  ✓"
        )

    print("[smoke] align.py PASSED ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke()
