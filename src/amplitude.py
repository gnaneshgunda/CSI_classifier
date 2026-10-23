"""
src/amplitude.py
================
Convert complex CSI or raw NexmonFrame lists to float32 amplitude arrays.

This module is a thin adapter layer between ``src/pcap_reader.py`` (which
returns ``NexmonFrame`` objects whose ``csi_raw`` field is an ``(N, 2)``
int16 array of packed-float words) and ``src/csi_decode.py`` (which
operates on raw UDP payload bytes).

The key responsibility is **reconstructing the synthetic payload** that
``amplitude_batch`` needs from ``NexmonFrame.csi_raw``:

    header (18 zero bytes) + csi_raw.tobytes()   →  1042-byte payload

The 18-byte header is zeroed because only the 1024-byte CSI data region is
used by ``amplitude_batch``; the Nexmon header fields (RSSI, chanspec, etc.)
are not needed for amplitude extraction.

Public API
----------
``frames_to_amplitude(frames)``
    NexmonFrame list → float32 (n, 256) amplitude array + validity mask.

``complex_to_amplitude(csi)``
    complex64 (n, 256) array → float32 (n, 256) amplitude array.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.csi_decode import CSI_DATA_OFFSET, amplitude_batch
from src.pcap_reader import NexmonFrame

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of CSI subcarriers (matches csi_decode.N_SUBCARRIERS).
N_SUBCARRIERS: int = 256

# Zero-padded header bytes prepended to each csi_raw block to form the
# synthetic payload expected by amplitude_batch.
_ZERO_HEADER: bytes = bytes(CSI_DATA_OFFSET)  # 18 zero bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frames_to_amplitude(
    frames: List[NexmonFrame],
    *,
    fftshift: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of NexmonFrame objects to float32 amplitude.

    Each frame's ``csi_raw`` (shape ``(256, 2)`` int16) is reinterpreted as
    256 packed-float uint32 words by concatenating the two int16 columns
    back into consecutive bytes, then prepending 18 zero bytes to form the
    synthetic UDP payload that ``csi_decode.amplitude_batch`` expects.

    Parameters
    ----------
    frames:
        List of ``NexmonFrame`` objects in capture order.
    fftshift:
        Passed through to ``amplitude_batch`` (applies fftshift along the
        subcarrier axis).  Default True.

    Returns
    -------
    amp : np.ndarray, shape (n, 256), dtype float32
        Per-subcarrier amplitude. Rows for invalid frames contain NaN.
    valid : np.ndarray, shape (n,), dtype bool
        True for frames that decoded successfully.
    """
    if not frames:
        return (
            np.zeros((0, N_SUBCARRIERS), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

    # Build synthetic payloads: 18 zero-header bytes + 1024 CSI bytes
    payloads = [
        _ZERO_HEADER + f.csi_raw.astype("<i2").tobytes()
        for f in frames
    ]
    return amplitude_batch(payloads, fftshift=fftshift)


def complex_to_amplitude(csi: np.ndarray) -> np.ndarray:
    """Compute per-subcarrier amplitude from a complex CSI array.

    Parameters
    ----------
    csi:
        Complex array of shape ``(n, 256)``, dtype complex64 or complex128.

    Returns
    -------
    np.ndarray, shape (n, 256), dtype float32
        ``|csi|`` per element.

    Raises
    ------
    ValueError
        If *csi* does not have exactly 256 columns.
    """
    if csi.ndim != 2 or csi.shape[1] != N_SUBCARRIERS:
        raise ValueError(
            f"complex_to_amplitude expects shape (n, {N_SUBCARRIERS}), "
            f"got {csi.shape}"
        )
    return np.abs(csi).astype(np.float32)
