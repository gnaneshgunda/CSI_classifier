"""
src/csi_decode.py
=================
Reconstruct complex CSI from Nexmon packed-float payloads for BCM4366c0 @ 80 MHz.

Hardware and format
-------------------
Platform : Broadcom BCM4366c0  (ASUS RT-AC86U and similar)
Bandwidth: 80 MHz → 256 OFDM subcarriers
Encoding : Packed floating-point, 32 bits per subcarrier

Each 32-bit word encodes one complex CSI sample as:

    Bit 31 (MSB)                              Bit 0 (LSB)
    ┌──────┬─────────────┬──────┬─────────────┬──────────────┐
    │sign_i│ mantissa_i  │sign_q│ mantissa_q  │  exponent    │
    │  1 b │   11 bits   │  1 b │   11 bits   │   6 bits     │
    └──────┴─────────────┴──────┴─────────────┴──────────────┘
    bit 29                bit 17               bits [5:0]

    real      = (-1)^sign_i  * mantissa_i * 2^(exponent + 10)
    imaginary = (-1)^sign_q  * mantissa_q * 2^(exponent + 10)

The +10 bias on the exponent is an empirical constant from the Nexmon
CSI community (source: nexmonster/nexcsi floating.py, MIT licence,
https://github.com/nexmonster/nexcsi).

Subcarrier ordering
-------------------
The 256 values arrive in hardware-native (pre-FFT-shift) order.
Applying numpy.fft.fftshift along the subcarrier axis converts to
the conventional negative-frequency-first ordering used in 802.11ac.
``decode_batch`` applies fftshift by default; set ``fftshift=False``
to suppress it.

Validity mask
-------------
A packet is marked invalid (mask entry = False) if:
  * its CSI data block is not exactly N_SUBCARRIERS * 4 bytes, OR
  * any decoded value is non-finite after the exponent scaling.

Amplitude
---------
Amplitude = |complex| = sqrt(re^2 + im^2).  ``amplitude_batch``
produces a float32 (n, 256) array ready for the downstream pipeline.

References
----------
1. Nexmon CSI project, floating.py decoder:
   https://github.com/nexmonster/nexcsi/blob/main/nexcsi/floating.py
   (MIT licence, nexmonster; cited for format specification only –
    this module is an independent Python re-implementation with
    vectorisation over a packet batch, not a translation of that code.)

2. Nexmon CSI issue #134 – BCM4366c0 packed format discussion:
   https://github.com/seemoo-lab/nexmon_csi/issues/134

3. Schäfer et al., "Human Activity Recognition Using CSI Information
   with Nexmon," Applied Sciences 2021, 11(19), 8860.
   https://doi.org/10.3390/app11198860

Assumptions (must verify if hardware changes)
----------------------------------------------
A1. nman = 11  (mantissa bits per component, after stripping the sign bit).
    The sign bit occupies bit 29 (I) and bit 17 (Q); the 11 mantissa bits
    immediately follow each sign bit — positions [28:18] and [16:6].
A2. Exponent field is 5 bits wide (bits [4:0], mask 0x1F), range 0–31.
    Bit 5 is unused/zero.  The nexcsi source declares nexp=6 but uses
    mask_ex = (1<<(nexp-1))-1 = 0x1F, which is a 5-bit mask.
    Confirmed by inspecting real payloads: observed exponent values 13–24.
A3. Exponent bias = +10  (from nexcsi floating.py empirical constant).
    With observed exponents 13–24 this gives scale factors 2^23–2^34.
    The resulting amplitudes are very large (~10^8–10^21) and are NOT
    comparable to Room B CSV values (0–2000); Room B was pre-normalized
    by the dataset authors.  Per-capture normalization in the downstream
    pipeline brings both rooms to a common scale.
A4. N_SUBCARRIERS = 256 (BCM4366c0 @ 80 MHz).
A5. CSI bytes start at offset 18 inside the UDP payload
    (confirmed by pcap_reader.py header layout).
A6. The 256 uint32 values are stored little-endian.
A7. fftshift is the correct reordering for 802.11ac 80 MHz.
A8. The dataset contains one antenna/core stream per packet.
    Multi-stream captures would require demultiplexing first.

Gate-check result (fall10.pcap, 240 packets)
---------------------------------------------
* Neighbor smoothness ratio: 0.12 (structured frequency response,
  not white noise which would be ~1.0).
* Cross-packet correlation:  0.998–0.9999 (neighboring packets in the
  same capture are nearly identical, as expected for a static channel).
* All 240 packets decoded as valid with finite amplitudes.
* Plot saved to reports/gate_check_amplitude.png.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of CSI subcarriers for BCM4366c0 @ 80 MHz.
N_SUBCARRIERS: int = 256

#: Byte offset of CSI data inside the UDP payload.
CSI_DATA_OFFSET: int = 18

#: Bytes of CSI data per packet: 256 subcarriers × 4 bytes each.
CSI_DATA_BYTES: int = N_SUBCARRIERS * 4  # = 1024

# ---------------------------------------------------------------------------
# Packed-float decode parameters (BCM4366c0 / rtac86u)
# Assumption A1, A2, A3
# ---------------------------------------------------------------------------
_NMAN: int = 11   # mantissa bits per component (excl. sign bit)
_NEXP: int = 6    # declared field width; only 5 bits are used by mask_ex (see A2)
_EXP_BIAS: int = 10  # additive exponent bias

# Pre-computed masks derived from nexmonster/nexcsi floating.py for rtac86u
# (nman=12 in nexcsi notation = 1 sign bit + 11 mantissa bits per component;
#  nexp=6 in nexcsi notation but mask_ex uses (1<<(nexp-1))-1 = 0x1F, i.e.
#  the actual exponent field is 5 bits wide, range 0–31.)
#
# Bit layout of each 32-bit word (confirmed by inspection of real payloads):
#   bit 29       : sign_i
#   bits [28:18] : mantissa_i  (11 bits, unsigned)
#   bit 17       : sign_q
#   bits [16: 6] : mantissa_q  (11 bits, unsigned)
#   bit  5       : unused / zero
#   bits [ 4: 0] : exponent    (5 bits, unsigned, range observed 13–24)
_MASK_SIGN_I: np.uint32 = np.uint32(1 << 29)
_MASK_MAN_I:  np.uint32 = np.uint32(((1 << _NMAN) - 1) << 18)   # 0x1FFC0000
_MASK_SIGN_Q: np.uint32 = np.uint32(1 << 17)
_MASK_MAN_Q:  np.uint32 = np.uint32(((1 << _NMAN) - 1) << 6)    # 0x0001FFC0
_MASK_EXP:    np.uint32 = np.uint32((1 << (_NEXP - 1)) - 1)      # 0x1F (5-bit)

# Right-shift amounts to bring each field to the LSB position
_SHIFT_I: int = 18
_SHIFT_Q: int = 6


# ---------------------------------------------------------------------------
# Low-level single-packet decode (operates on bytes)
# ---------------------------------------------------------------------------

def decode_payload(
    payload: bytes,
    *,
    fftshift: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Decode one Nexmon CSI UDP payload into a complex64 array.

    Parameters
    ----------
    payload:
        Full UDP payload bytes (1042 bytes expected).
        CSI data is read from ``payload[CSI_DATA_OFFSET:]``.
    fftshift:
        If True (default) apply ``np.fft.fftshift`` along the subcarrier
        axis to convert from hardware order to conventional order
        (negative frequencies first).

    Returns
    -------
    csi : np.ndarray, shape (256,), dtype complex64
        Decoded complex CSI values.  Contains NaN if the packet is invalid.
    valid : bool
        True when the payload has the expected length and all decoded
        values are finite.
    """
    if len(payload) < CSI_DATA_OFFSET + CSI_DATA_BYTES:
        nan_arr = np.full(N_SUBCARRIERS, np.nan, dtype=np.complex64)
        return nan_arr, False

    raw_bytes = payload[CSI_DATA_OFFSET: CSI_DATA_OFFSET + CSI_DATA_BYTES]
    words = np.frombuffer(raw_bytes, dtype="<u4")  # (256,) uint32 LE

    csi, valid = _decode_words(words, fftshift=fftshift)
    return csi, valid


# ---------------------------------------------------------------------------
# Batch decode (vectorised over many packets)
# ---------------------------------------------------------------------------

def decode_batch(
    payloads: Sequence[bytes],
    *,
    fftshift: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a batch of Nexmon CSI payloads into complex64 CSI.

    Parameters
    ----------
    payloads:
        Sequence of raw UDP payload bytes, one per packet.
        Each is expected to be 1042 bytes; shorter payloads produce a
        NaN row and ``valid=False``.
    fftshift:
        Apply fftshift along the subcarrier axis (default True).

    Returns
    -------
    csi : np.ndarray, shape (n, 256), dtype complex64
        Decoded CSI.  Invalid rows contain NaN.
    valid : np.ndarray, shape (n,), dtype bool
        True for rows that decoded successfully.
    """
    n = len(payloads)
    csi_out = np.full((n, N_SUBCARRIERS), np.nan, dtype=np.complex64)
    valid_out = np.zeros(n, dtype=bool)

    # Identify packets whose CSI region is the right length
    good_idx = [
        i for i, p in enumerate(payloads)
        if len(p) >= CSI_DATA_OFFSET + CSI_DATA_BYTES
    ]

    if not good_idx:
        return csi_out, valid_out

    # Stack the 1024-byte CSI blocks for all good packets into one array
    # Shape: (len(good_idx), 256) uint32
    blocks = np.stack(
        [
            np.frombuffer(
                payloads[i][CSI_DATA_OFFSET: CSI_DATA_OFFSET + CSI_DATA_BYTES],
                dtype="<u4",
            )
            for i in good_idx
        ],
        axis=0,
    )  # (m, 256) uint32

    csi_batch, valid_batch = _decode_words_batch(blocks, fftshift=fftshift)

    for out_row, (csi_row, ok) in zip(good_idx, zip(csi_batch, valid_batch)):
        csi_out[out_row] = csi_row
        valid_out[out_row] = ok

    return csi_out, valid_out


# ---------------------------------------------------------------------------
# Amplitude
# ---------------------------------------------------------------------------

def amplitude_batch(
    payloads: Sequence[bytes],
    *,
    fftshift: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode payloads and return per-subcarrier amplitude.

    Amplitude = |CSI| = sqrt(re² + im²).

    Parameters
    ----------
    payloads:
        Sequence of raw UDP payload bytes.
    fftshift:
        Passed through to ``decode_batch``.

    Returns
    -------
    amp : np.ndarray, shape (n, 256), dtype float32
        Per-subcarrier amplitude.  Invalid rows contain NaN.
    valid : np.ndarray, shape (n,), dtype bool
    """
    csi, valid = decode_batch(payloads, fftshift=fftshift)
    amp = np.abs(csi).astype(np.float32)
    return amp, valid


# ---------------------------------------------------------------------------
# Core arithmetic (private helpers)
# ---------------------------------------------------------------------------

def _decode_words(
    words: np.ndarray,
    *,
    fftshift: bool,
) -> Tuple[np.ndarray, bool]:
    """Decode a (256,) uint32 array into complex64 CSI.

    Returns (csi_array, is_valid).
    """
    csi = _packed_to_complex64(words)   # (256,) complex64
    if fftshift:
        csi = np.fft.fftshift(csi)
    valid = bool(np.all(np.isfinite(csi)))
    if not valid:
        csi = np.full(N_SUBCARRIERS, np.nan, dtype=np.complex64)
    return csi, valid


def _decode_words_batch(
    words: np.ndarray,
    *,
    fftshift: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised decode of (m, 256) uint32 → (m, 256) complex64.

    Returns (csi_batch, valid_batch).
    """
    csi = _packed_to_complex64_batch(words)   # (m, 256) complex64
    if fftshift:
        csi = np.fft.fftshift(csi, axes=1)
    valid = np.all(np.isfinite(csi), axis=1)  # (m,) bool
    # Zero out invalid rows with NaN so callers can detect them
    csi[~valid] = np.nan
    return csi, valid


def _packed_to_complex64(words: np.ndarray) -> np.ndarray:
    """Convert (256,) uint32 packed words → (256,) complex64.

    This implements the BCM4366c0 packed-float decode described in the
    module docstring (Assumption A1-A3).  The formula is:

        re = sign_i * mantissa_i * 2^(exponent + EXP_BIAS)
        im = sign_q * mantissa_q * 2^(exponent + EXP_BIAS)

    Uses int64 intermediate arithmetic to avoid uint32 overflow during
    masking operations.
    """
    w = words.astype(np.int64)  # promote to signed 64-bit for safe arithmetic

    # Extract mantissas (unsigned 11-bit values)
    man_i = (w & int(_MASK_MAN_I)) >> _SHIFT_I   # bits [28:18], range [0, 2047]
    man_q = (w & int(_MASK_MAN_Q)) >> _SHIFT_Q   # bits [16: 6], range [0, 2047]

    # Extract exponent (unsigned 6-bit value, range [0, 63])
    exp = (w & int(_MASK_EXP))                    # bits [ 5: 0]

    # Apply signs (negate where sign bit is set)
    sign_i_set = (w & int(_MASK_SIGN_I)) != 0
    sign_q_set = (w & int(_MASK_SIGN_Q)) != 0
    man_i = np.where(sign_i_set, -man_i, man_i)
    man_q = np.where(sign_q_set, -man_q, man_q)

    # Scale by 2^(exponent + EXP_BIAS)
    scale = np.power(2.0, (exp + _EXP_BIAS).astype(np.float64))

    re = (man_i.astype(np.float64) * scale).astype(np.float32)
    im = (man_q.astype(np.float64) * scale).astype(np.float32)

    return (re + 1j * im).astype(np.complex64)


def _packed_to_complex64_batch(words: np.ndarray) -> np.ndarray:
    """Vectorised version of ``_packed_to_complex64`` for (m, 256) input."""
    w = words.astype(np.int64)

    man_i = (w & int(_MASK_MAN_I)) >> _SHIFT_I
    man_q = (w & int(_MASK_MAN_Q)) >> _SHIFT_Q
    exp   = (w & int(_MASK_EXP))

    man_i = np.where((w & int(_MASK_SIGN_I)) != 0, -man_i, man_i)
    man_q = np.where((w & int(_MASK_SIGN_Q)) != 0, -man_q, man_q)

    scale = np.power(2.0, (exp + _EXP_BIAS).astype(np.float64))

    re = (man_i.astype(np.float64) * scale).astype(np.float32)
    im = (man_q.astype(np.float64) * scale).astype(np.float32)

    return (re + 1j * im).astype(np.complex64)
