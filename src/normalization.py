"""
src/normalization.py
====================
Window-level normalization for the CSI-HAR pipeline.

Overview
--------
Each raw window is a ``(T, 256)`` float32 array (T = window length, e.g. 150).
The normaliser decomposes every window into two parts:

* **Static vector** ``(256,)`` – the time-mean across the T rows.
* **Dynamic matrix** ``(T, 256)`` – residuals after subtracting the mean,
  divided by a *robust* per-subcarrier scale (IQR / 1.349, the normal-sigma
  equivalent).

``fit()`` collects the static vectors from **development windows only**,
computes per-room statistics (mean & std of static vectors), and writes them to
``derived/norm_stats.npz``.  The file is keyed by room (``mean_A``, ``std_A``,
``scale_A``, ``mean_B``, etc.).

``transform(window, room, room_probs)`` applies the normalisation and selects
statistics according to one of four *routing modes*:

* ``"predicted_hard"``  – argmax of *room_probs* selects one room's stats.
* ``"predicted_soft"``  – weighted mixture of per-room stats.
* ``"oracle"``          – use the ground-truth room label.
* ``"none"``            – use global (all-room) statistics.

Leakage guarantee
-----------------
``fit()`` raises ``ValueError`` if any supplied window row has
``split == "val"``.  Callers must filter the manifest/index before calling fit.

Public API
----------
``NormStats``
    Dataclass holding per-room/global statistics.

``Normalizer(cfg, log1p=True)``
    ``fit(windows_df)``        – compute & save stats (dev rows only).
    ``transform(window, ...)`` – return ``(static, dynamic)`` pair.
    ``load()``                 – reload stats from disk.

``_robust_scale(arr)``
    Module-level helper: per-column IQR/1.349 robust scale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ExperimentConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Rooms understood by the pipeline (ordered: A first, B second).
_ROOMS: Tuple[str, ...] = ("A", "B")

#: IQR -> sigma conversion factor
#: scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25) ≈ 1.3490
_IQR_TO_SIGMA: float = 1.3489795003921634

#: Minimum robust scale – guards against divide-by-zero on constant subcarriers.
_MIN_SCALE: float = 1e-6


# ---------------------------------------------------------------------------
# NormStats dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Fitted normalisation statistics.

    Attributes
    ----------
    room_mean : dict[str, ndarray]
        Per-room mean of static (time-mean) vectors.  Shape ``(256,)`` per room.
    room_std : dict[str, ndarray]
        Per-room std of static vectors.  Shape ``(256,)`` per room.
    room_scale : dict[str, ndarray]
        Per-room robust scale of the dynamic residuals.  Shape ``(256,)`` per room.
    global_mean : ndarray, shape (256,)
        Mean of static vectors across *all* dev windows (both rooms).
    global_std : ndarray, shape (256,)
        Std of static vectors across *all* dev windows.
    global_scale : ndarray, shape (256,)
        Robust scale of dynamic residuals across *all* dev windows.
    """

    room_mean:    Dict[str, np.ndarray] = field(default_factory=dict)
    room_std:     Dict[str, np.ndarray] = field(default_factory=dict)
    room_scale:   Dict[str, np.ndarray] = field(default_factory=dict)
    global_mean:  np.ndarray = field(
        default_factory=lambda: np.zeros(256, dtype=np.float32)
    )
    global_std:   np.ndarray = field(
        default_factory=lambda: np.ones(256, dtype=np.float32)
    )
    global_scale: np.ndarray = field(
        default_factory=lambda: np.ones(256, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """Fit and apply window-level normalisation.

    Parameters
    ----------
    cfg:
        Full experiment configuration (paths + routing mode).
    log1p:
        If ``True`` (default), apply ``np.log1p`` to raw amplitude values
        *before* the static/dynamic decomposition.  Toggle off for ablations.

    Examples
    --------
    >>> norm = Normalizer(cfg, log1p=True)
    >>> norm.fit(dev_df)                          # dev_df must have split='dev'
    >>> static, dynamic = norm.transform(win, room="A")
    """

    def __init__(self, cfg: ExperimentConfig, *, log1p: bool = True) -> None:
        self.cfg = cfg
        self.log1p = log1p
        self._stats: Optional[NormStats] = None
        self._stats_path: Path = Path(cfg.paths.derived_dir) / "norm_stats.npz"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        windows_df: pd.DataFrame,
        *,
        window_loader=None,
    ) -> "Normalizer":
        """Compute per-room normalisation statistics from development windows.

        Each row of *windows_df* must contain at minimum:

        * ``"split"``  – **must** be ``"dev"`` for every row (leakage guard).
        * ``"room"``   – ``"A"`` or ``"B"``.
        * One of:
          - ``"array"`` column holding a pre-loaded ``np.ndarray``, **or**
          - ``"path"`` / ``"window_path"`` column with a path to a ``.npy`` file.

        Alternatively, pass a *window_loader* callable ``(row) -> np.ndarray``.

        Parameters
        ----------
        windows_df:
            DataFrame of development windows to fit on.
        window_loader:
            Optional callable used to load the window array for each row.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            * If *windows_df* contains any row with ``split == "val"``
              (leakage guard).
            * If *windows_df* is empty.
            * If the ``"split"`` column is missing.
        """
        # ------------------------------------------------------------------ #
        # Leakage guard – must be the very first check                        #
        # ------------------------------------------------------------------ #
        if "split" not in windows_df.columns:
            raise ValueError(
                "fit() requires a 'split' column in windows_df to enforce "
                "the no-leakage guarantee."
            )

        val_mask = windows_df["split"] == "val"
        if val_mask.any():
            n_val = int(val_mask.sum())
            raise ValueError(
                f"fit() received {n_val} validation-split row(s). "
                "Normalization statistics must be computed on development data "
                "only. Filter out val-split rows before calling fit()."
            )

        if len(windows_df) == 0:
            raise ValueError(
                "fit() received an empty DataFrame; nothing to fit on."
            )

        # ------------------------------------------------------------------ #
        # Accumulate static vectors & dynamic residuals per room              #
        # ------------------------------------------------------------------ #
        room_statics:  Dict[str, List[np.ndarray]] = {r: [] for r in _ROOMS}
        room_dynamics: Dict[str, List[np.ndarray]] = {r: [] for r in _ROOMS}
        all_statics:   List[np.ndarray] = []
        all_dynamics:  List[np.ndarray] = []

        for _, row in windows_df.iterrows():
            win = _load_window_row(row, window_loader)
            static, dynamic = self._decompose(win)

            room = str(row.get("room", "")).upper()
            if room in _ROOMS:
                room_statics[room].append(static)
                room_dynamics[room].append(dynamic)

            all_statics.append(static)
            all_dynamics.append(dynamic)

        # ------------------------------------------------------------------ #
        # Global statistics                                                   #
        # ------------------------------------------------------------------ #
        g_static_arr  = np.stack(all_statics, axis=0)           # (N, 256)
        g_dynamic_arr = np.concatenate(all_dynamics, axis=0)    # (N*T, 256)

        global_mean  = g_static_arr.mean(axis=0).astype(np.float32)
        global_std   = g_static_arr.std(axis=0).astype(np.float32)
        global_scale = _robust_scale(g_dynamic_arr)

        # ------------------------------------------------------------------ #
        # Per-room statistics                                                 #
        # ------------------------------------------------------------------ #
        r_mean:  Dict[str, np.ndarray] = {}
        r_std:   Dict[str, np.ndarray] = {}
        r_scale: Dict[str, np.ndarray] = {}

        for room in _ROOMS:
            if room_statics[room]:
                s_arr = np.stack(room_statics[room], axis=0)          # (N_r, 256)
                d_arr = np.concatenate(room_dynamics[room], axis=0)   # (N_r*T, 256)
                r_mean[room]  = s_arr.mean(axis=0).astype(np.float32)
                r_std[room]   = s_arr.std(axis=0).astype(np.float32)
                r_scale[room] = _robust_scale(d_arr)
            else:
                warnings.warn(
                    f"No development windows found for room '{room}'; "
                    "falling back to global statistics for that room.",
                    UserWarning,
                    stacklevel=2,
                )
                r_mean[room]  = global_mean.copy()
                r_std[room]   = global_std.copy()
                r_scale[room] = global_scale.copy()

        self._stats = NormStats(
            room_mean=r_mean,
            room_std=r_std,
            room_scale=r_scale,
            global_mean=global_mean,
            global_std=global_std,
            global_scale=global_scale,
        )
        self._save()
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(
        self,
        window: np.ndarray,
        room: Optional[str] = None,
        room_probs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise *window* and return ``(static_vec, dynamic_mat)``.

        Routing mode is read from ``cfg.routing.mode``:

        ``"oracle"``
            Use statistics for the ground-truth *room* (must be supplied).
        ``"predicted_hard"``
            Use statistics for ``argmax(room_probs)``.  Falls back to oracle
            if *room_probs* is None but *room* is supplied.
        ``"predicted_soft"``
            Soft mixture of per-room statistics weighted by *room_probs*.
        ``"none"``
            Use global (all-room) statistics.

        Parameters
        ----------
        window:
            Raw amplitude array of shape ``(T, 256)``, dtype float32.
        room:
            Ground-truth room label (``"A"`` or ``"B"``).  Required for
            ``"oracle"`` mode; optional hint for ``"predicted_hard"``.
        room_probs:
            Softmax room probabilities, shape ``(2,)`` ordered
            ``[P(A), P(B)]``.  Required for ``"predicted_hard"`` and
            ``"predicted_soft"`` modes.

        Returns
        -------
        static : ndarray, shape (256,)
            Z-scored time-mean of the window.
        dynamic : ndarray, shape (T, 256)
            Residuals divided by robust scale.

        Raises
        ------
        RuntimeError
            If ``fit()`` / ``load()`` has not been called yet.
        ValueError
            If required routing arguments are missing.
        """
        stats = self._require_stats()
        mode  = self.cfg.routing.mode

        mean, std, scale = self._select_stats(stats, mode, room, room_probs)

        win = window.astype(np.float32, copy=False)
        if self.log1p:
            win = np.log1p(win)

        raw_static  = win.mean(axis=0)                  # (256,)
        raw_dynamic = win - raw_static[np.newaxis, :]   # (T, 256)

        safe_std   = np.where(std   > _MIN_SCALE, std,   1.0).astype(np.float32)
        safe_scale = np.where(scale > _MIN_SCALE, scale, 1.0).astype(np.float32)

        norm_static  = ((raw_static - mean) / safe_std).astype(np.float32)
        norm_dynamic = (raw_dynamic / safe_scale[np.newaxis, :]).astype(np.float32)

        return norm_static, norm_dynamic

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write fitted statistics to ``derived/norm_stats.npz``."""
        stats = self._require_stats()
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "global_mean":  stats.global_mean,
            "global_std":   stats.global_std,
            "global_scale": stats.global_scale,
        }
        for room in _ROOMS:
            arrays[f"mean_{room}"]  = stats.room_mean[room]
            arrays[f"std_{room}"]   = stats.room_std[room]
            arrays[f"scale_{room}"] = stats.room_scale[room]

        np.savez(str(self._stats_path), **arrays)

    def load(self) -> "Normalizer":
        """Reload previously fitted statistics from ``derived/norm_stats.npz``.

        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If the stats file does not exist (run ``fit()`` first).
        """
        if not self._stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found at {self._stats_path}. "
                "Run Normalizer.fit() first."
            )
        data = np.load(str(self._stats_path))
        self._stats = NormStats(
            global_mean=data["global_mean"].astype(np.float32),
            global_std=data["global_std"].astype(np.float32),
            global_scale=data["global_scale"].astype(np.float32),
            room_mean={r: data[f"mean_{r}"].astype(np.float32) for r in _ROOMS},
            room_std={r: data[f"std_{r}"].astype(np.float32) for r in _ROOMS},
            room_scale={r: data[f"scale_{r}"].astype(np.float32) for r in _ROOMS},
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose(self, win: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply optional log1p then split into ``(time_mean, residuals)``."""
        w = win.astype(np.float32, copy=False)
        if self.log1p:
            w = np.log1p(w)
        static  = w.mean(axis=0)                    # (256,)
        dynamic = w - static[np.newaxis, :]         # (T, 256)
        return static, dynamic

    def _require_stats(self) -> NormStats:
        if self._stats is None:
            raise RuntimeError(
                "Normalizer has no fitted statistics. "
                "Call fit() or load() first."
            )
        return self._stats

    def _select_stats(
        self,
        stats: NormStats,
        mode: str,
        room: Optional[str],
        room_probs: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, std, scale)`` for the chosen routing mode."""

        if mode == "oracle":
            if room is None:
                raise ValueError(
                    "routing.mode='oracle' requires the 'room' argument "
                    "(ground-truth room label)."
                )
            r = room.upper()
            if r not in _ROOMS:
                raise ValueError(
                    f"Unknown room {room!r}; expected one of {_ROOMS}."
                )
            return stats.room_mean[r], stats.room_std[r], stats.room_scale[r]

        elif mode == "predicted_hard":
            if room_probs is None:
                # Graceful fallback: use oracle if ground-truth room given
                if room is not None:
                    return self._select_stats(stats, "oracle", room, None)
                raise ValueError(
                    "routing.mode='predicted_hard' requires 'room_probs'."
                )
            probs = np.asarray(room_probs, dtype=np.float32)
            if probs.shape[0] != len(_ROOMS):
                raise ValueError(
                    f"room_probs must have {len(_ROOMS)} elements, "
                    f"got {probs.shape[0]}."
                )
            pred = _ROOMS[int(np.argmax(probs))]
            return (
                stats.room_mean[pred],
                stats.room_std[pred],
                stats.room_scale[pred],
            )

        elif mode == "predicted_soft":
            if room_probs is None:
                raise ValueError(
                    "routing.mode='predicted_soft' requires 'room_probs'."
                )
            probs = np.asarray(room_probs, dtype=np.float32)
            if probs.shape[0] != len(_ROOMS):
                raise ValueError(
                    f"room_probs must have {len(_ROOMS)} elements, "
                    f"got {probs.shape[0]}."
                )
            # Re-normalise to guard against non-simplex inputs
            probs = probs / (probs.sum() + 1e-12)
            mean  = sum(
                float(probs[i]) * stats.room_mean[r]
                for i, r in enumerate(_ROOMS)
            )
            std   = sum(
                float(probs[i]) * stats.room_std[r]
                for i, r in enumerate(_ROOMS)
            )
            scale = sum(
                float(probs[i]) * stats.room_scale[r]
                for i, r in enumerate(_ROOMS)
            )
            return (
                np.asarray(mean,  dtype=np.float32),
                np.asarray(std,   dtype=np.float32),
                np.asarray(scale, dtype=np.float32),
            )

        elif mode == "none":
            return stats.global_mean, stats.global_std, stats.global_scale

        else:
            raise ValueError(
                f"Unknown routing mode {mode!r}. "
                "Expected one of: 'oracle', 'predicted_hard', "
                "'predicted_soft', 'none'."
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _robust_scale(arr: np.ndarray) -> np.ndarray:
    """Per-column robust scale: IQR / 1.349 (normal-sigma equivalent).

    Parameters
    ----------
    arr:
        2-D array ``(N, 256)``; each column is one subcarrier.

    Returns
    -------
    ndarray of shape ``(256,)``, dtype float32.
    Values below ``_MIN_SCALE`` are clamped to ``_MIN_SCALE``.
    """
    q75 = np.percentile(arr, 75, axis=0).astype(np.float64)
    q25 = np.percentile(arr, 25, axis=0).astype(np.float64)
    iqr = q75 - q25
    scale = (iqr / _IQR_TO_SIGMA).astype(np.float32)
    return np.where(scale > _MIN_SCALE, scale, _MIN_SCALE).astype(np.float32)


def _load_window_row(row: pd.Series, loader) -> np.ndarray:
    """Load the window array for *row*, using *loader* if provided."""
    if loader is not None:
        return loader(row)
    if "array" in row.index:
        arr = row["array"]
        return np.asarray(arr, dtype=np.float32)
    path = row.get("path", row.get("window_path", None))
    if path is None:
        raise ValueError(
            "Cannot load window: row has no 'array', 'path', or "
            "'window_path' column and no window_loader was supplied."
        )
    return np.load(str(path)).astype(np.float32)
