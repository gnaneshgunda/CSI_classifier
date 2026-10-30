"""
src/features.py
===============
Hand-crafted feature extraction from CSI amplitude windows.

Each window is a ``(T, 256)`` float32 array (T = cfg.window.length = 150,
256 subcarriers).  ``extract_features`` converts it to a **fixed-length
float32 vector** that aggregates both frequency-domain (subcarrier-wise) and
time-domain (temporal) information.

Feature groups
--------------
1. **Per-subcarrier statistics** (2 * 256 = 512 dims)
   - Mean and standard deviation computed along the time axis for each of
   the 256 subcarriers.

2. **Band energies - spatial** (4 dims)
   - The 256 subcarriers are partitioned into 4 equal-width bands
   (sub-bands 0-63, 64-127, 128-191, 192-255).  The mean squared amplitude
   within each band is returned.

3. **STFT spectral energy in Doppler bands** (5 dims)
   - A short-time Fourier transform is applied along the *time axis* for
   each subcarrier separately (n_fft=32, hop=8).
   The magnitude spectrogram (averaged over all subcarriers) is then binned
   into 5 Doppler frequency bands:
     * Static         : 0 - 0.5 Hz
     * Very slow      : 0.5 - 1 Hz
     * Slow motion    : 1 - 3 Hz
     * Body motion    : 3 - 8 Hz
     * Fast / noise   : 8 Hz - Nyquist
   Each bin contains the mean magnitude across the STFT frames and
   subcarriers that fall into that band.

4. **Temporal statistics** (3 dims)
   - Per-packet L2 norm (across subcarriers) gives a 1-D power envelope
   series of length T.  From this we compute:
     * Temporal variance
     * Temporal entropy  (Shannon, treating normalised power as PMF)
     * Temporal range    (max - min of power envelope)

Total feature vector: 512 + 4 + 5 + 3 = **524 dimensions**.

Public API
----------
``FEATURE_DIM``
    Module-level int constant (524).

``extract_features(window, packet_rate) -> np.ndarray``
    Shape ``(FEATURE_DIM,)`` float32.  Pure function - no side effects.

``extract_features_batch(windows, packet_rate) -> np.ndarray``
    Shape ``(N, FEATURE_DIM)`` float32.

``feature_names() -> list[str]``
    Human-readable name per feature dimension.

Notes
-----
* Uses only NumPy (np.fft.rfft); no SciPy dependency.
* ``packet_rate`` is in Hz. Room A: 100/3 ~ 33.3 Hz (30 ms inter-packet).
* Pure function - no side effects, fully deterministic.
* ``FEATURE_DIM`` = 524.
"""

from __future__ import annotations

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_SPATIAL_BANDS: int = 4

# Doppler band edges in Hz; last element is treated as Nyquist (open upper bound).
_DOPPLER_EDGES_HZ = (0.0, 0.5, 1.0, 3.0, 8.0, float("inf"))
_N_DOPPLER_BANDS: int = len(_DOPPLER_EDGES_HZ) - 1   # 5

_SUBCARRIER_STAT_DIM: int = 256 * 2                   # 512

#: Total feature vector length (512 + 4 + 5 + 3).
FEATURE_DIM: int = _SUBCARRIER_STAT_DIM + _N_SPATIAL_BANDS + _N_DOPPLER_BANDS + 3  # 524


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subcarrier_stats(window: np.ndarray) -> np.ndarray:
    """Per-subcarrier mean + std concatenated -> shape (512,) float32."""
    mean = window.mean(axis=0)        # (256,)
    std  = window.std(axis=0, ddof=0) # (256,) population std
    return np.concatenate([mean, std]).astype(np.float32)


def _spatial_band_energies(window: np.ndarray, n_bands: int = _N_SPATIAL_BANDS) -> np.ndarray:
    """Mean squared amplitude in equal-width subcarrier bands.

    Parameters
    ----------
    window:
        Shape ``(T, 256)``.
    n_bands:
        Number of equal-width bands.

    Returns
    -------
    np.ndarray shape ``(n_bands,)`` float32.
    """
    n_sub     = window.shape[1]
    band_size = n_sub // n_bands
    sq        = window.astype(np.float64) ** 2
    energies  = np.empty(n_bands, dtype=np.float32)
    for i in range(n_bands):
        start         = i * band_size
        end           = (i + 1) * band_size if i < n_bands - 1 else n_sub
        energies[i]   = float(sq[:, start:end].mean())
    return energies


def _stft_doppler_energies(
    window: np.ndarray,
    packet_rate: float,
    n_fft: int = 32,
    hop: int = 8,
) -> np.ndarray:
    """Mean STFT magnitude in Doppler frequency bands.

    An STFT is computed along the **time axis** for every subcarrier
    (using a Hann window of length ``n_fft``).  The per-subcarrier magnitude
    spectrograms are averaged across all 256 subcarriers, then binned by
    physical frequency (Hz) into ``_N_DOPPLER_BANDS`` bins.

    Parameters
    ----------
    window:
        Shape ``(T, 256)`` float32.
    packet_rate:
        Sampling rate in Hz (packets per second).
    n_fft:
        FFT size. Clamped to T if necessary.
    hop:
        Hop length between STFT frames (samples).

    Returns
    -------
    np.ndarray shape ``(_N_DOPPLER_BANDS,)`` float32.
    """
    T, n_sub   = window.shape
    n_fft      = min(n_fft, T)
    hann       = np.hanning(n_fft)                              # (n_fft,)
    n_freqs    = n_fft // 2 + 1
    freqs_hz   = np.fft.rfftfreq(n_fft, d=1.0 / packet_rate)   # (n_freqs,)

    arr          = window.astype(np.float64)                    # (T, 256)
    frame_starts = list(range(0, T - n_fft + 1, hop)) or [0]
    n_frames     = len(frame_starts)

    # Accumulate magnitude spectrogram averaged over all subcarriers
    mean_mag = np.zeros((n_freqs, n_frames), dtype=np.float64)
    for sub_idx in range(n_sub):
        sig = arr[:, sub_idx]
        for fi, s in enumerate(frame_starts):
            spectrum          = np.abs(np.fft.rfft(sig[s: s + n_fft] * hann))
            mean_mag[:, fi]  += spectrum
    mean_mag /= n_sub

    # Bin into Doppler bands
    edges         = _DOPPLER_EDGES_HZ
    band_energies = np.empty(_N_DOPPLER_BANDS, dtype=np.float32)
    for b in range(_N_DOPPLER_BANDS):
        mask             = (freqs_hz >= edges[b]) & (freqs_hz < edges[b + 1])
        band_energies[b] = float(mean_mag[mask, :].mean()) if mask.any() else 0.0
    return band_energies


def _temporal_features(window: np.ndarray) -> np.ndarray:
    """Temporal variance, entropy, and range of the per-packet power envelope.

    The L2 norm of each packet (row) across the 256 subcarriers gives a
    1-D ``power`` series of length T.

    Returns
    -------
    np.ndarray shape ``(3,)`` float32: [variance, entropy, range].
    """
    power    = np.linalg.norm(window.astype(np.float64), axis=1)  # (T,)
    variance = float(np.var(power, ddof=0))

    s = power.sum()
    if s > 0.0:
        p       = np.clip(power / s, 1e-12, None)
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = 0.0

    range_ = float(power.max() - power.min()) if len(power) > 0 else 0.0

    return np.array([variance, entropy, range_], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    window: np.ndarray,
    packet_rate: float = 100.0 / 3.0,
) -> np.ndarray:
    """Extract a fixed-length feature vector from a single CSI window.

    Parameters
    ----------
    window:
        Array of shape ``(T, 256)`` dtype float32.
        T is typically 150 (``cfg.window.length``).
    packet_rate:
        Packet transmission rate in Hz.  Defaults to 100/3 ~ 33.3 Hz
        (30 ms inter-packet interval for Room A).

    Returns
    -------
    np.ndarray shape ``(FEATURE_DIM,)`` float32.

    Feature layout (contiguous)::

        [sub_mean x256 | sub_std x256 | spatial_band x4 |
         doppler_band x5 | temporal_variance | temporal_entropy | temporal_range]

    Raises
    ------
    ValueError
        If ``window.ndim != 2`` or ``window.shape[1] != 256``.
    """
    if window.ndim != 2 or window.shape[1] != 256:
        raise ValueError(f"Expected window shape (T, 256), got {window.shape}")
    if window.shape[0] == 0:
        raise ValueError("Window must have at least one time step.")

    sub_stats = _subcarrier_stats(window)
    spatial   = _spatial_band_energies(window)
    doppler   = _stft_doppler_energies(window, packet_rate)
    temporal  = _temporal_features(window)

    feat = np.concatenate([sub_stats, spatial, doppler, temporal]).astype(np.float32)
    assert feat.shape == (FEATURE_DIM,), f"BUG: got {feat.shape[0]}, expected {FEATURE_DIM}"
    return feat


def extract_features_batch(
    windows: np.ndarray,
    packet_rate: float = 100.0 / 3.0,
) -> np.ndarray:
    """Extract features for a batch of windows.

    Parameters
    ----------
    windows:
        Array of shape ``(N, T, 256)``.
    packet_rate:
        Packet rate in Hz (see ``extract_features``).

    Returns
    -------
    np.ndarray shape ``(N, FEATURE_DIM)`` float32.
    """
    if windows.ndim != 3 or windows.shape[2] != 256:
        raise ValueError(f"Expected (N, T, 256), got {windows.shape}")
    N   = windows.shape[0]
    out = np.empty((N, FEATURE_DIM), dtype=np.float32)
    for i in range(N):
        out[i] = extract_features(windows[i], packet_rate=packet_rate)
    return out


def feature_names() -> List[str]:
    """Return human-readable feature names, one per dimension.

    Returns
    -------
    list[str] of length ``FEATURE_DIM``.
    """
    names: List[str] = []
    names += [f"sub{i:03d}_mean" for i in range(256)]
    names += [f"sub{i:03d}_std"  for i in range(256)]
    band_w = 256 // _N_SPATIAL_BANDS
    for b in range(_N_SPATIAL_BANDS):
        lo = b * band_w
        hi = (b + 1) * band_w - 1
        names.append(f"band_energy_sub{lo}-{hi}")
    names += [
        "doppler_static_0-0.5Hz",
        "doppler_veryslow_0.5-1Hz",
        "doppler_slow_1-3Hz",
        "doppler_body_3-8Hz",
        "doppler_fast_8Hz+",
    ]
    names += ["temporal_variance", "temporal_entropy", "temporal_range"]
    assert len(names) == FEATURE_DIM, f"Name count {len(names)} != FEATURE_DIM {FEATURE_DIM}"
    return names


# ---------------------------------------------------------------------------
# Smoke test (python -m src.features  or  python src/features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("[smoke] src/features.py self-test ...")

    rng = np.random.default_rng(0)
    win = rng.uniform(0, 100, (150, 256)).astype(np.float32)

    feat = extract_features(win, packet_rate=100.0 / 3.0)
    assert feat.shape == (FEATURE_DIM,), f"Shape mismatch: {feat.shape}"
    assert feat.dtype == np.float32,     f"Dtype mismatch: {feat.dtype}"
    assert np.isfinite(feat).all(),      "Non-finite values in feature vector"

    feat2 = extract_features(win.copy(), packet_rate=100.0 / 3.0)
    assert np.array_equal(feat, feat2), "Feature extraction is not deterministic"

    batch    = rng.uniform(0, 100, (8, 150, 256)).astype(np.float32)
    bf       = extract_features_batch(batch)
    assert bf.shape == (8, FEATURE_DIM) and bf.dtype == np.float32

    names = feature_names()
    assert len(names) == FEATURE_DIM
    assert names[0]   == "sub000_mean"
    assert names[256] == "sub000_std"
    assert "temporal_entropy" in names

    try:
        extract_features(rng.uniform(0, 1, (150, 128)).astype(np.float32))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    print(f"[smoke] FEATURE_DIM = {FEATURE_DIM}")
    print(f"[smoke] feature vector: shape={feat.shape}  dtype={feat.dtype}  finite=OK  deterministic=OK")
    print(f"[smoke] batch API: shape={bf.shape}  OK")
    print(f"[smoke] feature_names(): {len(names)} names  OK")
    print(f"[smoke] error-handling check  OK")
    print("[smoke] src/features.py PASSED")
    sys.exit(0)
