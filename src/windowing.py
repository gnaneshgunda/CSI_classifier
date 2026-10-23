"""
src/windowing.py
================
Tile cleaned CSI amplitude captures into fixed ``(150, 256)`` windows.

Three windowing strategies
--------------------------
``canonical_window(arr, cfg)``
    Return **one** deterministic window per capture — the central 150
    observations (or reflect-padded if the capture is shorter).  Used for
    validation captures and for any single-window inference path.

``dev_windows(arr, cfg)``
    Sliding windows with ``cfg.window.dev_stride`` (default 50) for
    development captures.  Returns a list of ``(start, window)`` pairs.
    Never crosses capture boundaries.

``sliding_windows(arr, cfg, stride)``
    Generic sliding-window generator with configurable stride.  Used for
    optional capture-level inference averaging.

Short-capture handling
----------------------
When a capture has fewer than ``cfg.window.length`` observations:

* ``reflect_pad`` — mirror-pad the array at both ends to reach ``length``.
    The padding uses ``np.pad(mode='reflect')`` which mirrors without
    repeating edge values.
* ``exclude`` — return an empty list / None so the caller can skip the
    capture entirely.

Determinism
-----------
All windowing is purely deterministic — no randomness, no seeds.  The same
input array + config always produces the same output.  This is critical
for provenance: ``windows_index.csv`` records the ``start`` index of each
window so any window can be traced back to its source capture.

Public API
----------
    canonical_window(arr, cfg) -> Optional[np.ndarray]        # (150, 256) or None
    dev_windows(arr, cfg)      -> List[Tuple[int, np.ndarray]] # [(start, (150,256)), ...]
    sliding_windows(arr, cfg, stride) -> List[Tuple[int, np.ndarray]]
    reflect_pad(arr, length)   -> np.ndarray                   # (length, 256)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config import WindowConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical window length (matches cfg.window.length but available at
#: import time for assertions).
_DEFAULT_LENGTH: int = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reflect_pad(arr: np.ndarray, length: int) -> np.ndarray:
    """Mirror-pad *arr* along axis 0 to exactly *length* rows.

    Uses ``np.pad(mode='reflect')`` which mirrors without repeating the
    edge element (e.g. ``[1,2,3]`` → ``[3,2,1,2,3,2,1]``).

    Parameters
    ----------
    arr:
        Array of shape ``(n, 256)`` where ``n < length``.
    length:
        Target number of rows.

    Returns
    -------
    np.ndarray of shape ``(length, 256)``.

    Raises
    ------
    ValueError
        If *arr* has more rows than *length*, or if *arr* is empty and
        padding is impossible.
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Cannot reflect-pad an empty array.")
    if n >= length:
        return arr[:length].copy()
    pad_total = length - n
    # Split padding between left and right, favouring left so the original
    # data is centered in the output.
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(
        arr,
        ((pad_left, pad_right), (0, 0)),
        mode="reflect",
    )


def _extract_central(arr: np.ndarray, length: int) -> np.ndarray:
    """Return the central *length* rows of *arr*.

    If ``arr.shape[0] == length`` the array is returned as-is.  For
    ``arr.shape[0] > length`` the middle *length* rows are selected
    symmetrically (left-biased on odd remainders).
    """
    n = arr.shape[0]
    if n <= length:
        return arr.copy()
    start = (n - length) // 2
    return arr[start: start + length].copy()


# ---------------------------------------------------------------------------
# Public API: canonical_window
# ---------------------------------------------------------------------------

def canonical_window(
    arr: np.ndarray,
    cfg: WindowConfig,
) -> Optional[np.ndarray]:
    """Return one deterministic (150, 256) window from *arr*.

    The window is the **central** ``cfg.window.length`` observations.  If
    the capture is shorter than ``length``:

    * ``reflect_pad`` — mirror-pad to reach ``length``.
    * ``exclude``     — return ``None`` (caller should skip the capture).

    Parameters
    ----------
    arr:
        Cleaned amplitude array of shape ``(n, n_subcarriers)``.
    cfg:
        Window configuration from ``experiment.yaml``.

    Returns
    -------
    np.ndarray of shape ``(length, n_subcarriers)`` dtype float32, or None.

    Raises
    ------
    ValueError
        If ``arr.shape[1] != cfg.window.n_subcarriers``.
    """
    length = cfg.length
    n_sub = cfg.n_subcarriers
    n = arr.shape[0]

    if arr.shape[1] != n_sub:
        raise ValueError(
            f"Expected {n_sub} subcarriers, got {arr.shape[1]}"
        )

    arr = arr.astype(np.float32, copy=False)

    if n == 0:
        if cfg.short_capture_policy == "exclude":
            return None
        # reflect_pad on empty raises; return None for safety
        return None

    if n < length:
        if cfg.short_capture_policy == "exclude":
            return None
        return reflect_pad(arr, length)

    return _extract_central(arr, length)


# ---------------------------------------------------------------------------
# Public API: dev_windows
# ---------------------------------------------------------------------------

def dev_windows(
    arr: np.ndarray,
    cfg: WindowConfig,
) -> List[Tuple[int, np.ndarray]]:
    """Sliding windows for development captures (SSL training, probes).

    Uses ``cfg.window.dev_stride`` (default 50).  Each window is
    ``(length, n_subcarriers)``.  Windows never cross capture boundaries.

    Parameters
    ----------
    arr:
        Cleaned amplitude array of shape ``(n, n_subcarriers)``.
    cfg:
        Window configuration.

    Returns
    -------
    List of ``(start_index, window)`` tuples.  Empty list if the capture
    is too short and ``short_capture_policy == "exclude"``.
    """
    return sliding_windows(arr, cfg, stride=cfg.dev_stride)


# ---------------------------------------------------------------------------
# Public API: sliding_windows
# ---------------------------------------------------------------------------

def sliding_windows(
    arr: np.ndarray,
    cfg: WindowConfig,
    stride: int,
) -> List[Tuple[int, np.ndarray]]:
    """Generic sliding-window generator.

    Parameters
    ----------
    arr:
        Cleaned amplitude array of shape ``(n, n_subcarriers)``.
    cfg:
        Window configuration (uses ``length``, ``n_subcarriers``,
        ``short_capture_policy``).
    stride:
        Step size between consecutive window starts.

    Returns
    -------
    List of ``(start, window)`` tuples where ``window`` has shape
    ``(length, n_subcarriers)`` and dtype float32.  The ``start`` index
    is the row offset in the original array (0-based, before any padding).

    Notes
    -----
    * If ``n < length`` and policy is ``reflect_pad``: returns a single
      window with ``start=0`` (the padded array).
    * If ``n < length`` and policy is ``exclude``: returns ``[]``.
    * If ``n >= length``: windows start at 0, stride, 2*stride, … up to
      ``n - length``.  The last window always ends at ``n``.
    """
    length = cfg.length
    n_sub = cfg.n_subcarriers
    n = arr.shape[0]

    if arr.shape[1] != n_sub:
        raise ValueError(
            f"Expected {n_sub} subcarriers, got {arr.shape[1]}"
        )

    arr = arr.astype(np.float32, copy=False)

    if n == 0:
        return []

    if n < length:
        if cfg.short_capture_policy == "exclude":
            return []
        padded = reflect_pad(arr, length)
        return [(0, padded)]

    windows: List[Tuple[int, np.ndarray]] = []
    start = 0
    while start + length <= n:
        windows.append((start, arr[start: start + length].copy()))
        start += stride

    # If the last window doesn't reach the end and we have remaining data,
    # add a final non-overlapping window anchored at the end.
    # This ensures no data at the tail is silently dropped.
    last_end = windows[-1][0] + length if windows else 0
    if last_end < n:
        final_start = n - length
        if final_start > windows[-1][0]:  # avoid duplicate
            windows.append((final_start, arr[final_start: final_start + length].copy()))

    return windows


# ---------------------------------------------------------------------------
# Convenience: count windows without generating them
# ---------------------------------------------------------------------------

def count_windows(n_rows: int, length: int, stride: int) -> int:
    """Return how many sliding windows fit in *n_rows* rows.

    Accounts for the tail-anchored final window (see ``sliding_windows``).
    """
    if n_rows < length:
        return 1 if n_rows > 0 else 0  # reflect_pad gives 1; empty gives 0
    n_strided = (n_rows - length) // stride + 1
    last_end = (n_strided - 1) * stride + length
    if last_end < n_rows:
        return n_strided + 1  # tail window
    return n_strided
