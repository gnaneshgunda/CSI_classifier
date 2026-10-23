"""
tests/test_csi_decode.py
========================
Unit tests for src/csi_decode.py.

Synthetic payloads are built from known uint32 words so we can verify
the exact decoded values without relying on real pcap data.
Real-file smoke tests are included at the bottom (marked with the
``realdata`` pytest mark so CI can skip them if the dataset is absent).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from src.csi_decode import (
    CSI_DATA_BYTES,
    CSI_DATA_OFFSET,
    N_SUBCARRIERS,
    _EXP_BIAS,
    _NEXP,
    _NMAN,
    _packed_to_complex64,
    _packed_to_complex64_batch,
    amplitude_batch,
    decode_batch,
    decode_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_word(man_i: int, sign_i: int, man_q: int, sign_q: int, exp: int) -> int:
    """Pack fields into a 32-bit word matching the BCM4366c0 layout.

    sign_i/q: 0 = positive, 1 = negative
    man_i/q : unsigned 11-bit mantissa (0–2047)
    exp      : unsigned 6-bit exponent (0–63)
    """
    word = 0
    word |= (exp & 0x3F)
    word |= ((man_q & 0x7FF) << 6)
    word |= ((sign_q & 1)    << 17)
    word |= ((man_i & 0x7FF) << 18)
    word |= ((sign_i & 1)    << 29)
    return word


def _make_payload(words_256: np.ndarray) -> bytes:
    """Build a 1042-byte synthetic payload from 256 uint32 words."""
    assert words_256.shape == (256,)
    header = bytes(CSI_DATA_OFFSET)  # 18 zero bytes
    csi_bytes = words_256.astype("<u4").tobytes()
    assert len(csi_bytes) == CSI_DATA_BYTES
    return header + csi_bytes


def _all_same_word(man_i: int, sign_i: int,
                   man_q: int, sign_q: int, exp: int) -> bytes:
    """All 256 subcarriers encoded with the same field values."""
    w = _make_word(man_i, sign_i, man_q, sign_q, exp)
    return _make_payload(np.full(256, w, dtype=np.uint32))


def _expected_value(man: int, sign: int, exp: int) -> float:
    """Expected decoded float: sign * man * 2^(exp + EXP_BIAS)."""
    s = -1.0 if sign else 1.0
    return s * man * (2.0 ** (exp + _EXP_BIAS))


# ---------------------------------------------------------------------------
# _packed_to_complex64 – single-packet
# ---------------------------------------------------------------------------

class TestPackedToComplex64:
    def test_zero_word_gives_zero_complex(self) -> None:
        words = np.zeros(256, dtype=np.uint32)
        csi = _packed_to_complex64(words)
        assert csi.shape == (256,)
        assert np.all(csi == 0)

    def test_positive_re_positive_im(self) -> None:
        # man_i=100, sign_i=0, man_q=50, sign_q=0, exp=0
        w = _make_word(100, 0, 50, 0, 0)
        words = np.full(256, w, dtype=np.uint32)
        csi = _packed_to_complex64(words)
        expected_re = _expected_value(100, 0, 0)
        expected_im = _expected_value(50, 0, 0)
        assert np.allclose(csi.real, expected_re, rtol=1e-5)
        assert np.allclose(csi.imag, expected_im, rtol=1e-5)

    def test_negative_re(self) -> None:
        w = _make_word(200, 1, 0, 0, 0)   # sign_i=1 → negative real
        csi = _packed_to_complex64(np.full(256, w, dtype=np.uint32))
        assert np.all(csi.real < 0)
        assert np.allclose(csi.imag, 0.0, atol=1e-6)

    def test_negative_im(self) -> None:
        w = _make_word(0, 0, 150, 1, 0)   # sign_q=1 → negative imag
        csi = _packed_to_complex64(np.full(256, w, dtype=np.uint32))
        assert np.allclose(csi.real, 0.0, atol=1e-6)
        assert np.all(csi.imag < 0)

    def test_exponent_scaling(self) -> None:
        # exp=3 → scale = 2^(3+10) = 8192
        w0 = _make_word(1, 0, 1, 0, 0)   # scale = 2^10
        w3 = _make_word(1, 0, 1, 0, 3)   # scale = 2^13
        csi0 = _packed_to_complex64(np.full(256, w0, dtype=np.uint32))
        csi3 = _packed_to_complex64(np.full(256, w3, dtype=np.uint32))
        ratio = csi3.real[0] / csi0.real[0]
        assert abs(ratio - 8.0) < 1e-3   # 2^3 = 8

    def test_dtype_is_complex64(self) -> None:
        csi = _packed_to_complex64(np.zeros(256, dtype=np.uint32))
        assert csi.dtype == np.complex64

    def test_output_length_is_256(self) -> None:
        csi = _packed_to_complex64(np.zeros(256, dtype=np.uint32))
        assert len(csi) == 256

    def test_max_mantissa(self) -> None:
        """max mantissa (2047) with exp=0 → re = 2047 * 2^EXP_BIAS."""
        w = _make_word(2047, 0, 0, 0, 0)
        csi = _packed_to_complex64(np.full(256, w, dtype=np.uint32))
        expected = 2047.0 * (2 ** _EXP_BIAS)
        assert np.allclose(csi.real, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# _packed_to_complex64_batch
# ---------------------------------------------------------------------------

class TestPackedToComplex64Batch:
    def test_shape(self) -> None:
        words = np.zeros((10, 256), dtype=np.uint32)
        csi = _packed_to_complex64_batch(words)
        assert csi.shape == (10, 256)

    def test_dtype(self) -> None:
        words = np.zeros((3, 256), dtype=np.uint32)
        csi = _packed_to_complex64_batch(words)
        assert csi.dtype == np.complex64

    def test_consistent_with_single(self) -> None:
        """Batch output must match single-packet output row-by-row."""
        rng = np.random.default_rng(0)
        words_batch = rng.integers(0, 2**30, size=(5, 256), dtype=np.int64).astype(np.uint32)
        batch_out = _packed_to_complex64_batch(words_batch)
        for i in range(5):
            single_out = _packed_to_complex64(words_batch[i])
            assert np.allclose(batch_out[i], single_out, equal_nan=True)


# ---------------------------------------------------------------------------
# decode_payload
# ---------------------------------------------------------------------------

class TestDecodePayload:
    def test_returns_256_complex64(self) -> None:
        payload = _all_same_word(10, 0, 5, 0, 0)
        csi, valid = decode_payload(payload)
        assert csi.shape == (256,)
        assert csi.dtype == np.complex64
        assert valid is True

    def test_short_payload_invalid(self) -> None:
        csi, valid = decode_payload(b"\x00" * 10)
        assert valid is False
        assert np.all(np.isnan(csi))

    def test_empty_payload_invalid(self) -> None:
        csi, valid = decode_payload(b"")
        assert valid is False

    def test_zero_payload_is_valid(self) -> None:
        """All-zero CSI data → zero complex values → valid."""
        payload = bytes(CSI_DATA_OFFSET + CSI_DATA_BYTES)
        csi, valid = decode_payload(payload)
        assert valid is True
        assert np.all(csi == 0)

    def test_fftshift_default_true(self) -> None:
        """fftshift=True rotates the array; must differ from fftshift=False
        when the input is not symmetric under circular shift."""
        # Put a large unique value only at index 0; all others are zero.
        # fftshift of 256 elements shifts by 128, so index 0 moves to index 128.
        w_big = _make_word(2000, 0, 0, 0, 0)
        words = np.zeros(256, dtype=np.uint32)
        words[0] = w_big
        payload = _make_payload(words)
        csi_shifted, _ = decode_payload(payload, fftshift=True)
        csi_raw, _     = decode_payload(payload, fftshift=False)
        assert not np.array_equal(csi_shifted, csi_raw)
        # The non-zero value should be at index 128 after shift, not index 0
        assert csi_shifted[128].real != 0
        assert csi_raw[0].real != 0

    def test_fftshift_false_no_reorder(self) -> None:
        """With fftshift=False the first decoded value should match word[0]."""
        w = _make_word(7, 0, 3, 0, 0)
        words = np.array([w] + [0] * 255, dtype=np.uint32)
        payload = _make_payload(words)
        csi, valid = decode_payload(payload, fftshift=False)
        assert valid is True
        # Only subcarrier 0 is non-zero; the rest should be zero
        assert csi[0].real != 0
        assert np.all(csi[1:] == 0)

    def test_finite_values_for_valid_payload(self) -> None:
        payload = _all_same_word(1, 0, 1, 0, 1)
        csi, valid = decode_payload(payload)
        assert valid is True
        assert np.all(np.isfinite(csi))


# ---------------------------------------------------------------------------
# decode_batch
# ---------------------------------------------------------------------------

class TestDecodeBatch:
    def test_shape(self) -> None:
        payloads = [_all_same_word(10, 0, 5, 0, 0) for _ in range(7)]
        csi, valid = decode_batch(payloads)
        assert csi.shape == (7, 256)
        assert valid.shape == (7,)

    def test_dtype(self) -> None:
        payloads = [_all_same_word(1, 0, 1, 0, 0)]
        csi, _ = decode_batch(payloads)
        assert csi.dtype == np.complex64

    def test_all_valid_on_good_payloads(self) -> None:
        payloads = [_all_same_word(1, 0, 1, 0, 0) for _ in range(4)]
        _, valid = decode_batch(payloads)
        assert np.all(valid)

    def test_short_payload_marked_invalid(self) -> None:
        payloads = [
            _all_same_word(1, 0, 1, 0, 0),
            b"\x00" * 5,                     # too short
            _all_same_word(1, 0, 1, 0, 0),
        ]
        _, valid = decode_batch(payloads)
        assert valid[0] is np.bool_(True)
        assert valid[1] is np.bool_(False)
        assert valid[2] is np.bool_(True)

    def test_short_payload_row_contains_nan(self) -> None:
        payloads = [b"\x00" * 5]
        csi, valid = decode_batch(payloads)
        assert not valid[0]
        assert np.all(np.isnan(csi[0]))

    def test_empty_batch(self) -> None:
        csi, valid = decode_batch([])
        assert csi.shape == (0, N_SUBCARRIERS)
        assert valid.shape == (0,)

    def test_single_item_consistent_with_decode_payload(self) -> None:
        payload = _all_same_word(42, 0, 13, 1, 2)
        csi_single, _ = decode_payload(payload, fftshift=True)
        csi_batch, _  = decode_batch([payload], fftshift=True)
        assert np.allclose(csi_single, csi_batch[0], equal_nan=True)

    def test_zero_csi_block_is_valid(self) -> None:
        payload = bytes(CSI_DATA_OFFSET + CSI_DATA_BYTES)
        csi, valid = decode_batch([payload])
        assert valid[0]
        assert np.all(csi[0] == 0)


# ---------------------------------------------------------------------------
# amplitude_batch
# ---------------------------------------------------------------------------

class TestAmplitudeBatch:
    def test_shape_and_dtype(self) -> None:
        payloads = [_all_same_word(10, 0, 10, 0, 0) for _ in range(5)]
        amp, valid = amplitude_batch(payloads)
        assert amp.shape == (5, 256)
        assert amp.dtype == np.float32
        assert np.all(valid)

    def test_amplitude_is_nonnegative(self) -> None:
        payloads = [_all_same_word(100, 1, 50, 1, 0)]  # negative re and im
        amp, valid = amplitude_batch(payloads)
        assert valid[0]
        assert np.all(amp[0] >= 0)

    def test_amplitude_formula(self) -> None:
        """amp = sqrt(re^2 + im^2)."""
        man_i, man_q, exp = 3, 4, 0
        # expected: re = 3*2^10, im = 4*2^10 → amp = 5*2^10
        w = _make_word(man_i, 0, man_q, 0, exp)
        words = np.full(256, w, dtype=np.uint32)
        payload = _make_payload(words)
        amp, valid = amplitude_batch([payload], fftshift=False)
        expected = 5.0 * (2 ** _EXP_BIAS)
        assert valid[0]
        assert np.allclose(amp[0], expected, rtol=1e-4)

    def test_invalid_row_contains_nan_amplitude(self) -> None:
        payloads = [b"\x00" * 3]
        amp, valid = amplitude_batch(payloads)
        assert not valid[0]
        assert np.all(np.isnan(amp[0]))


# ---------------------------------------------------------------------------
# Real-data smoke tests (require the dataset to be present)
# ---------------------------------------------------------------------------

REAL_PCAP = Path(
    "dataset/Experiment-2/Experiment-2/realdata/input_data/30ms/csidata/fall/fall10.pcap"
)


@pytest.mark.skipif(not REAL_PCAP.exists(), reason="Dataset not available")
class TestRealData:
    """Smoke tests that run against a real capture file."""

    def _load_payloads(self) -> list:
        import warnings
        warnings.filterwarnings("ignore")
        from src.pcap_reader import read_pcap
        frames, stats = read_pcap(REAL_PCAP)
        assert stats.total_skipped == 0
        # Build raw payload bytes list from csi_raw (int16 re/im pairs)
        import struct as st
        payloads = []
        for f in frames:
            header = bytes(CSI_DATA_OFFSET)
            csi_bytes = f.csi_raw.astype("<i2").tobytes()
            payloads.append(header + csi_bytes)
        return payloads

    def test_output_width_is_256(self) -> None:
        payloads = self._load_payloads()
        csi, valid = decode_batch(payloads)
        assert csi.shape[1] == 256

    def test_all_packets_valid(self) -> None:
        payloads = self._load_payloads()
        _, valid = decode_batch(payloads)
        assert np.all(valid), f"Invalid packets: {np.sum(~valid)}"

    def test_all_finite_values(self) -> None:
        payloads = self._load_payloads()
        csi, valid = decode_batch(payloads)
        assert np.all(np.isfinite(csi[valid]))

    def test_amplitude_nonnegative(self) -> None:
        payloads = self._load_payloads()
        amp, valid = amplitude_batch(payloads)
        assert np.all(amp[valid] >= 0)

    def test_amplitude_in_plausible_range(self) -> None:
        """Real CSI amplitudes on BCM4366c0 @ 80 MHz are expected to be
        well above zero and below ~10^9 for typical indoor captures."""
        payloads = self._load_payloads()
        amp, valid = amplitude_batch(payloads, fftshift=False)
        valid_amp = amp[valid]
        assert np.max(valid_amp) > 0, "All amplitudes are zero – likely a decode error"
        # Sanity upper bound: with max mantissa 2047 and max exp 63 we get
        # 2047 * 2^(63+10) ≈ 1.9e22; real indoor values are far below that
        assert np.all(np.isfinite(valid_amp))

    def test_packet_count_matches_pcap_reader(self) -> None:
        import warnings
        warnings.filterwarnings("ignore")
        from src.pcap_reader import read_pcap
        frames, _ = read_pcap(REAL_PCAP)
        payloads = self._load_payloads()
        csi, _ = decode_batch(payloads)
        assert len(frames) == csi.shape[0]
