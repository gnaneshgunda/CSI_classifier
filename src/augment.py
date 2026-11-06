"""
src/augment.py
==============
CSI-valid augmentations and the SSL pair-dataset for self-supervised pre-training.

Augmentations
-------------
All operate on a single ``(T, 256)`` float32 **numpy** array and return the
same shape.  Each is individually switchable via its strength parameter
(set to 0 or False to disable):

``gaussian_jitter(x, std)``
    Add i.i.d. Gaussian noise scaled by ``std * x.std()``.
    CSI-valid: small perturbation, does not change temporal structure.

``global_gain(x, scale_min, scale_max)``
    Multiply the whole window by a scalar drawn from ``U[scale_min, scale_max]``.
    CSI-valid: models antenna/distance amplitude variation.

``subcarrier_band_dropout(x, frac)``
    Zero out a contiguous *band* of subcarriers (fraction ``frac`` of 256).
    CSI-valid: models narrowband interference / dead subcarriers.

``circular_time_shift(x, max_shift)``
    Circularly shift along the time axis by up to ``max_shift`` steps.
    CSI-valid: small temporal misalignment.  max_shift ≤ 10 (enforced).
    No time reversal.  No strong time-warp.

``compose(x, cfg, rng)``
    Apply all four augmentations in order (each individually disabled if its
    config parameter makes it a no-op).

NOT included (per spec):
    - Time reversal
    - Strong time-warp / speed perturbation

SSL Dataset
-----------
``SSLPairDataset``
    PyTorch Dataset.  Each item is a ``(view1, view2)`` tuple of float32
    tensors of shape ``(T, 256)`` — two **different** windows from the
    **same development capture**, each independently augmented.

    * Mixes both rooms (Room A + Room B dev windows).
    * Returns **no labels**.
    * Normalisation uses ``Normalizer.transform`` with the capture's TRUE
      room (oracle mode) so pretraining uses only dev-split leakage-free stats.
    * Augmentation can be globally disabled by setting ``augment=False``.

Public API
----------
``gaussian_jitter(x, std, rng)``
``global_gain(x, scale_min, scale_max, rng)``
``subcarrier_band_dropout(x, frac, rng)``
``circular_time_shift(x, max_shift, rng)``
``compose(x, cfg, rng)``

``SSLPairDataset(index_df, windows_dir, normalizer, cfg, augment=True, seed=0)``
    ``.pairs``              – list of (window_id_1, window_id_2) per capture
    ``__len__()``           – number of capture groups
    ``__getitem__(i)``      – ``(view1, view2)`` float32 tensors ``(T, 256)``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import ExperimentConfig, AugmentConfig
from src.normalization import Normalizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum allowed circular shift in time steps (hard cap per spec).
_MAX_SHIFT_CAP: int = 10

#: Window shape expected from files.
_N_SUB: int = 256


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def gaussian_jitter(
    x: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to *x*.

    Noise standard deviation is ``std * x.std()``.  If ``std <= 0`` the
    input is returned unchanged.

    Parameters
    ----------
    x:
        Float32 array ``(T, 256)``.
    std:
        Noise scale (fraction of per-window std).  Set to 0 to disable.
    rng:
        NumPy ``Generator`` for reproducibility.

    Returns
    -------
    np.ndarray of same shape as *x*, dtype float32.
    """
    if std <= 0:
        return x
    noise = rng.standard_normal(x.shape).astype(np.float32)
    return x + std * float(x.std()) * noise


def global_gain(
    x: np.ndarray,
    scale_min: float,
    scale_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multiply *x* by a scalar drawn from ``U[scale_min, scale_max]``.

    If ``scale_min >= scale_max`` or ``scale_min <= 0``, or if
    ``scale_min == scale_max == 1``, the input is returned unchanged.

    Parameters
    ----------
    x:
        Float32 array ``(T, 256)``.
    scale_min, scale_max:
        Range for the amplitude multiplier.
    rng:
        NumPy ``Generator``.

    Returns
    -------
    np.ndarray of same shape as *x*, dtype float32.
    """
    if scale_min <= 0 or scale_min >= scale_max:
        return x
    if scale_min == 1.0 and scale_max == 1.0:
        return x
    factor = float(rng.uniform(scale_min, scale_max))
    return (x * factor).astype(np.float32)


def subcarrier_band_dropout(
    x: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Zero out a contiguous random band of subcarriers.

    The band width is ``round(frac * 256)`` subcarriers.  The start index is
    chosen uniformly so the band fits within [0, 256).  If ``frac <= 0`` the
    input is returned unchanged.

    Parameters
    ----------
    x:
        Float32 array ``(T, 256)``.
    frac:
        Fraction of subcarriers to zero out (0 ≤ frac < 1).
    rng:
        NumPy ``Generator``.

    Returns
    -------
    np.ndarray of same shape as *x*, dtype float32.
    """
    if frac <= 0:
        return x
    n_sub = x.shape[1]
    band_width = max(1, round(frac * n_sub))
    band_width = min(band_width, n_sub)
    max_start = n_sub - band_width
    start = int(rng.integers(0, max_start + 1))
    out = x.copy()
    out[:, start: start + band_width] = 0.0
    return out


def circular_time_shift(
    x: np.ndarray,
    max_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circularly shift *x* along the time axis by a random amount.

    The shift is drawn from ``{-max_shift, ..., max_shift}`` (uniform,
    excluding 0 shift with low probability but 0 is possible).
    ``max_shift`` is clamped to ``_MAX_SHIFT_CAP`` (10) to avoid strong
    temporal distortion.  If ``max_shift <= 0`` the input is returned unchanged.

    No time reversal, no strong time-warp.

    Parameters
    ----------
    x:
        Float32 array ``(T, 256)``.
    max_shift:
        Maximum shift in time steps (≤ 10 enforced).
    rng:
        NumPy ``Generator``.

    Returns
    -------
    np.ndarray of same shape as *x*, dtype float32.
    """
    if max_shift <= 0:
        return x
    max_shift = min(int(max_shift), _MAX_SHIFT_CAP)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return x
    return np.roll(x, shift, axis=0).astype(np.float32)


def compose(
    x: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply all augmentations in order, each individually switchable.

    Order: gaussian_jitter → global_gain → subcarrier_band_dropout →
    circular_time_shift.

    Parameters
    ----------
    x:
        Float32 array ``(T, 256)``.
    cfg:
        ``AugmentConfig`` from ``experiment.yaml``.
    rng:
        NumPy ``Generator`` (shared across the four calls).

    Returns
    -------
    np.ndarray ``(T, 256)`` float32.
    """
    x = gaussian_jitter(x, cfg.gaussian_noise_std, rng)
    x = global_gain(x, cfg.amplitude_scale_min, cfg.amplitude_scale_max, rng)
    x = subcarrier_band_dropout(x, cfg.subcarrier_mask_frac, rng)
    # circular_time_shift — derive max_shift from time_crop_min_frac:
    # crop_frac gives fraction kept, so max_shift ≈ (1-crop_frac)*T/2, capped at 10
    x = circular_time_shift(x, _MAX_SHIFT_CAP, rng)
    return x


# ---------------------------------------------------------------------------
# SSL Pair Dataset
# ---------------------------------------------------------------------------

class SSLPairDataset(Dataset):
    """Returns two independently-augmented views for self-supervised learning.

    Two pairing modes are supported, controlled by ``cross_room_frac``:

    **Same-capture pairs** (``cross_room_frac=0.0``, default legacy behaviour):
        Both views come from different windows of the *same* development
        capture.  The invariance loss teaches the encoder to ignore intra-
        capture variation (slight temporal shifts, noise).

    **Cross-room pairs** (``cross_room_frac > 0.0``):
        A fraction ``cross_room_frac`` of pairs are drawn cross-room — one
        window from Room A and one from Room B, selected at random with no
        activity label used.  The invariance loss then pushes the encoder to
        produce similar embeddings for both rooms, which directly improves
        cross-room transfer without violating the no-label SSL constraint.
        The remaining ``1 - cross_room_frac`` fraction are same-capture pairs
        as before.

    Each item ``(view1, view2)`` contains:

    * Two windows (same-capture or cross-room depending on pair type).
    * Each window is independently augmented (via ``compose``).
    * Normalisation applied with the capture's TRUE room (oracle routing).
    * No labels returned.

    Parameters
    ----------
    index_df:
        DataFrame with columns ``window_id, capture_id, room, activity,
        split, start``.  **Must contain only ``split == "dev"`` rows**
        (enforced at construction).
    windows_dir:
        Base directory containing ``dev/`` sub-folder with ``.npy`` files.
    normalizer:
        A fitted ``Normalizer`` instance (must have called ``fit()`` or
        ``load()`` before constructing this dataset).
    cfg:
        Full ``ExperimentConfig`` (uses ``cfg.augment``).
    augment:
        If ``True`` (default), apply ``compose`` augmentation to each view.
        If ``False``, views are returned as-is (post-normalisation only).
    seed:
        Base RNG seed for augmentation.  Each ``__getitem__`` call advances
        a per-item sub-seed so results are reproducible given a fixed seed.
    min_windows_per_capture:
        Minimum number of dev windows a capture must have to be included.
        Captures with only 1 window are silently dropped (need ≥ 2 for a pair).
    cross_room_frac:
        Fraction of pairs that should be cross-room (one window from Room A,
        one from Room B).  Must be in [0.0, 1.0].  Default 0.0 gives the
        original same-capture-only behaviour.  A value of 0.5 means half the
        pairs are cross-room, half are same-capture.

    Raises
    ------
    ValueError
        If ``index_df`` contains any ``split == "val"`` rows.
    RuntimeError
        If no captures with ≥ 2 dev windows are found.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        windows_dir: Path,
        normalizer: Normalizer,
        cfg: ExperimentConfig,
        *,
        augment: bool = True,
        seed: int = 0,
        min_windows_per_capture: int = 2,
        cross_room_frac: float = 0.0,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate: no val rows                                               #
        # ------------------------------------------------------------------ #
        if "split" in index_df.columns:
            val_mask = index_df["split"] == "val"
            if val_mask.any():
                raise ValueError(
                    f"SSLPairDataset received {int(val_mask.sum())} val-split "
                    "rows. Only dev windows may be used for SSL pre-training."
                )

        if not 0.0 <= cross_room_frac <= 1.0:
            raise ValueError(
                f"cross_room_frac must be in [0.0, 1.0], got {cross_room_frac}"
            )

        self._windows_dir    = Path(windows_dir)
        self._normalizer     = normalizer
        self._cfg            = cfg
        self._augment        = augment
        self._seed           = seed
        self._cross_room_frac = float(cross_room_frac)
        self._cache: dict    = {}   # window_id -> np.ndarray, populated after pairs built

        # ------------------------------------------------------------------ #
        # Build per-capture window lists                                      #
        # ------------------------------------------------------------------ #
        # Group by capture_id; store (room, [window_id, ...]) per capture
        self._captures: List[Tuple[str, List[str]]] = []

        grouped = index_df.groupby("capture_id", sort=False)
        for capture_id, grp in grouped:
            win_ids = grp["window_id"].tolist()
            if len(win_ids) < min_windows_per_capture:
                continue
            room = str(grp["room"].iloc[0]).upper()
            self._captures.append((room, win_ids))

        if not self._captures:
            raise RuntimeError(
                f"No captures with ≥ {min_windows_per_capture} dev windows "
                "found in index_df. Cannot build SSL pairs."
            )

        # ------------------------------------------------------------------ #
        # Pre-build same-capture pair index                                   #
        # Each capture contributes C(n,2) pairs.                             #
        # pairs: list of (room_a, window_id_a, room_b, window_id_b)          #
        # For same-capture pairs room_a == room_b.                           #
        # ------------------------------------------------------------------ #
        same_pairs: List[Tuple[str, str, str, str]] = []
        for room, win_ids in self._captures:
            n = len(win_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    same_pairs.append((room, win_ids[i], room, win_ids[j]))

        # ------------------------------------------------------------------ #
        # Build cross-room pairs (no labels used)                             #
        # Strategy: for each same-capture pair we may want to replace, pick  #
        # a random window from the *other* room.  We pre-build a flat list   #
        # of all windows per room so cross-room sampling is O(1).            #
        # ------------------------------------------------------------------ #
        # Flat per-room window lists for cross-room sampling
        self._room_windows: Dict[str, List[str]] = {"A": [], "B": []}
        for room, win_ids in self._captures:
            r = room.upper()
            if r in self._room_windows:
                self._room_windows[r].extend(win_ids)

        cross_pairs: List[Tuple[str, str, str, str]] = []
        if cross_room_frac > 0.0:
            rng_build = np.random.default_rng([seed, 999])
            n_cross = int(round(len(same_pairs) * cross_room_frac))
            wins_A = self._room_windows["A"]
            wins_B = self._room_windows["B"]
            if not wins_A or not wins_B:
                warnings.warn(
                    "cross_room_frac > 0 but one room has no dev windows. "
                    "Falling back to same-capture pairs only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                idxA = rng_build.integers(0, len(wins_A), size=n_cross)
                idxB = rng_build.integers(0, len(wins_B), size=n_cross)
                for ia, ib in zip(idxA, idxB):
                    cross_pairs.append(("A", wins_A[ia], "B", wins_B[ib]))

        # Mix: keep (1 - frac) of same-capture pairs + all cross-room pairs
        if cross_room_frac > 0.0 and same_pairs:
            rng_mix = np.random.default_rng([seed, 998])
            n_keep = len(same_pairs) - int(round(len(same_pairs) * cross_room_frac))
            n_keep = max(0, n_keep)
            keep_idx = rng_mix.choice(len(same_pairs), size=n_keep, replace=False)
            kept_same = [same_pairs[i] for i in sorted(keep_idx)]
        else:
            kept_same = same_pairs

        self._pairs: List[Tuple[str, str, str, str]] = kept_same + cross_pairs

        import random as _random
        _random.seed(seed)
        _random.shuffle(self._pairs)

        # ------------------------------------------------------------------ #
        # Preload all unique windows into RAM once                            #
        # ------------------------------------------------------------------ #
        all_ids: set = set()
        for ra, wid1, rb, wid2 in self._pairs:
            all_ids.add(wid1)
            all_ids.add(wid2)
        for wid in all_ids:
            self._cache[wid] = self._load_npy(wid, split="dev")

        n_cross_actual = sum(1 for ra, _, rb, _ in self._pairs if ra != rb)
        import logging as _logging
        _logging.getLogger(__name__).info(
            "SSLPairDataset: %d total pairs  (%d same-capture, %d cross-room)  "
            "cross_room_frac=%.2f",
            len(self._pairs), len(self._pairs) - n_cross_actual,
            n_cross_actual, cross_room_frac,
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(view1, view2)`` — two independently augmented windows.

        For same-capture pairs both views come from the same room/capture.
        For cross-room pairs view1 is from Room A and view2 from Room B.

        Parameters
        ----------
        idx:
            Index into the pair list.

        Returns
        -------
        view1, view2 : torch.Tensor
            Shape ``(T, 256)``, dtype ``float32``.
        """
        room1, wid1, room2, wid2 = self._pairs[idx]

        # Load from cache (preloaded at construction) instead of disk
        raw1 = self._cache[wid1].copy()
        raw2 = self._cache[wid2].copy()

        # Normalise each view with its own room's statistics
        win1 = self._normalizer.transform(raw1[np.newaxis], room=room1)[0]
        win2 = self._normalizer.transform(raw2[np.newaxis], room=room2)[0]

        # Per-item reproducible RNG (different seed per view so views differ)
        rng1 = np.random.default_rng([self._seed, idx, 0])
        rng2 = np.random.default_rng([self._seed, idx, 1])

        if self._augment:
            win1 = compose(win1, self._cfg.augment, rng1)
            win2 = compose(win2, self._cfg.augment, rng2)

        return (
            torch.from_numpy(win1.astype(np.float32)),
            torch.from_numpy(win2.astype(np.float32)),
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _load_npy(self, window_id: str, split: str = "dev") -> np.ndarray:
        path = self._windows_dir / split / f"{window_id}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Window file not found: {path}"
            )
        return np.load(str(path)).astype(np.float32)

    @property
    def pairs(self) -> List[Tuple[str, str, str, str]]:
        """List of ``(room_1, window_id_1, room_2, window_id_2)`` pairs."""
        return list(self._pairs)

    @property
    def n_captures(self) -> int:
        """Number of development captures with ≥ 2 windows."""
        return len(self._captures)
