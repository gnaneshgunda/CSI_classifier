"""
tests/test_pcap_reader.py
=========================
Unit tests for src/pcap_reader.py.

All tests use tiny synthetic .pcap files written with scapy in tmp_path.
No real dataset files are required.

Synthetic pcap helpers
----------------------
``_make_payload(magic, length, csi_re_val)``  – build a raw UDP payload
``_write_pcap(packets, path)``                – write a .pcap with scapy
``_valid_payload()``                          – canonical 1042-byte valid frame
"""

from __future__ import annotations
import sys

import io
import struct
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from src.pcap_reader import (
    CSI_DATA_BYTES,
    CSI_DATA_OFFSET,
    N_SUBCARRIERS,
    NEXMON_MAGIC,
    NEXMON_PAYLOAD_BYTES,
    NEXMON_UDP_PORT,
    NexmonFrame,
    NexmonHeader,
    SkipStats,
    _decode_csi,
    _decode_header,
    interval_stats,
    iter_csi_frames,
    read_pcap,
)


# ---------------------------------------------------------------------------
# Synthetic pcap helpers
# ---------------------------------------------------------------------------

def _valid_payload(
    rssi: int = -70,
    chanspec: int = 0x9696,
    chip_ver: int = 0x6540,
    csi_fill: int = 100,
) -> bytes:
    """Build a canonical 1042-byte Nexmon CSI payload.

    CSI data is filled with a fixed int16 value (*csi_fill*) in the real
    part and zero in the imaginary part for all 256 subcarriers.
    """
    # 18-byte header: magic(2) rssi(b) fc(B) seq_fn(B) pad1(B)
    #                 chanspec(H) chip_ver(H) n_sub(H) core_info(H)
    #                 rssi2(h) pad2(B) pad3(B)
    header = struct.pack(
        "<BBbBBBHHHHhBB",
        0x11, 0x11,          # magic
        rssi,                # rssi int8
        0xDC,                # fc
        0x71,                # seq_fn
        0x00,                # pad1
        chanspec,
        chip_ver,
        0xFFFF,              # n_sub (unreliable field)
        0x0000,              # core_info
        rssi,                # rssi2
        0x00, 0x00,          # pad2, pad3
    )
    assert len(header) == CSI_DATA_OFFSET, len(header)

    # CSI: 256 × (re=csi_fill, im=0) as int16 LE
    csi_data = struct.pack(f"<{N_SUBCARRIERS * 2}h", *([csi_fill, 0] * N_SUBCARRIERS))
    assert len(csi_data) == CSI_DATA_BYTES

    payload = header + csi_data
    assert len(payload) == NEXMON_PAYLOAD_BYTES
    return payload


def _bad_magic_payload() -> bytes:
    """Valid-length payload but wrong magic bytes."""
    p = bytearray(_valid_payload())
    p[0] = 0xAA
    p[1] = 0xBB
    return bytes(p)


def _short_payload() -> bytes:
    """Payload that is one byte too short."""
    return _valid_payload()[:-1]


def _write_pcap(
    udp_packets: List[Tuple[bytes, int, int]],
    path: Path,
) -> None:
    """Write a minimal .pcap containing UDP packets.

    Parameters
    ----------
    udp_packets:
        List of ``(payload_bytes, sport, dport)`` tuples.
    path:
        Destination .pcap path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scapy.utils import wrpcap
        from scapy.layers.l2 import Ether
        from scapy.layers.inet import IP, UDP
        from scapy.packet import Raw

    pkts = []
    ts_base = 1_700_000_000.0
    for i, (payload, sport, dport) in enumerate(udp_packets):
        pkt = (
            Ether()
            / IP(src="192.168.1.1", dst="192.168.1.2")
            / UDP(sport=sport, dport=dport)
            / Raw(load=payload)
        )
        pkt.time = ts_base + i * 0.030  # ~33 Hz
        pkts.append(pkt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrpcap(str(path), pkts)


def _write_valid_pcap(path: Path, n: int = 5, csi_fill: int = 100) -> None:
    """Write a .pcap with *n* valid Nexmon CSI frames."""
    packets = [
        (_valid_payload(csi_fill=csi_fill + i), NEXMON_UDP_PORT, NEXMON_UDP_PORT)
        for i in range(n)
    ]
    _write_pcap(packets, path)


# ---------------------------------------------------------------------------
# _decode_header
# ---------------------------------------------------------------------------

class TestDecodeHeader:
    def test_magic_bytes_not_in_header_fields(self) -> None:
        h = _decode_header(_valid_payload(rssi=-50, chanspec=0x9696, chip_ver=0x6540))
        # magic bytes are consumed but not stored in NexmonHeader
        assert h.rssi == -50
        assert h.chanspec == 0x9696
        assert h.chip_ver == 0x6540

    def test_rssi_is_signed(self) -> None:
        h = _decode_header(_valid_payload(rssi=-90))
        assert h.rssi == -90

    def test_seq_fn_properties(self) -> None:
        payload = bytearray(_valid_payload())
        # set seq_fn byte (offset 4) to 0xA3 → seq=0xA, fn=0x3
        payload[4] = 0xA3
        h = _decode_header(bytes(payload))
        assert h.seq_fn == 0xA3
        assert h.seq_num == 0xA
        assert h.frag_num == 0x3

    def test_n_sub_field_unreliable_value(self) -> None:
        """n_sub reads 0xFFFF on BCM4366c0; confirm we parse but don't trust it."""
        h = _decode_header(_valid_payload())
        assert h.n_sub == 0xFFFF


# ---------------------------------------------------------------------------
# _decode_csi
# ---------------------------------------------------------------------------

class TestDecodeCsi:
    def test_shape(self) -> None:
        arr = _decode_csi(_valid_payload(csi_fill=42))
        assert arr.shape == (N_SUBCARRIERS, 2)

    def test_dtype(self) -> None:
        arr = _decode_csi(_valid_payload())
        assert arr.dtype == np.int16

    def test_values(self) -> None:
        arr = _decode_csi(_valid_payload(csi_fill=777))
        # real parts should all be 777, imaginary parts 0
        assert np.all(arr[:, 0] == 777)
        assert np.all(arr[:, 1] == 0)

    def test_negative_values_preserved(self) -> None:
        arr = _decode_csi(_valid_payload(csi_fill=-500))
        assert np.all(arr[:, 0] == -500)

    def test_distinct_subcarriers(self) -> None:
        """Verify each subcarrier is independently accessible."""
        payload = bytearray(_valid_payload(csi_fill=0))
        # Set subcarrier[10] real part to 1234
        offset = CSI_DATA_OFFSET + 10 * 4
        struct.pack_into("<h", payload, offset, 1234)
        arr = _decode_csi(bytes(payload))
        assert arr[10, 0] == 1234
        assert arr[9, 0] == 0
        assert arr[11, 0] == 0


# ---------------------------------------------------------------------------
# iter_csi_frames – valid frames
# ---------------------------------------------------------------------------

class TestIterCsiFramesValid:
    def test_yields_correct_count(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=8)
        frames = list(iter_csi_frames(p))
        assert len(frames) == 8

    def test_yields_tuples_of_index_and_frame(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=3)
        results = list(iter_csi_frames(p))
        for idx, frame in results:
            assert isinstance(idx, int)
            assert isinstance(frame, NexmonFrame)

    def test_indices_are_monotone_increasing(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=5)
        indices = [idx for idx, _ in iter_csi_frames(p)]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    def test_timestamps_are_monotone_increasing(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=5)
        timestamps = [f.timestamp for _, f in iter_csi_frames(p)]
        assert all(t2 > t1 for t1, t2 in zip(timestamps, timestamps[1:]))

    def test_csi_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=2)
        _, frame = next(iter_csi_frames(p))
        assert frame.csi_raw.shape == (N_SUBCARRIERS, 2)
        assert frame.csi_raw.dtype == np.int16

    def test_csi_values_match_fill(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=1, csi_fill=321)
        _, frame = next(iter_csi_frames(p))
        assert np.all(frame.csi_raw[:, 0] == 321)
        assert np.all(frame.csi_raw[:, 1] == 0)

    def test_no_skips_on_clean_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=10)
        stats = SkipStats()
        list(iter_csi_frames(p, skip_stats=stats))
        assert stats.total_skipped == 0

    def test_max_packets_limits_total_read(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=20)
        frames = list(iter_csi_frames(p, max_packets=5))
        assert len(frames) <= 5

    def test_empty_pcap_yields_nothing(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.pcap"
        _write_pcap([], p)
        frames = list(iter_csi_frames(p))
        assert frames == []


# ---------------------------------------------------------------------------
# iter_csi_frames – skip reasons
# ---------------------------------------------------------------------------

class TestIterCsiFramesSkips:
    def test_skips_wrong_port(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        # port 9999 instead of 5500
        _write_pcap([(_valid_payload(), 9999, 9999)], p)
        stats = SkipStats()
        frames = list(iter_csi_frames(p, skip_stats=stats))
        assert frames == []
        assert stats.wrong_port == 1

    def test_skips_wrong_length(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_pcap([(_short_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT)], p)
        stats = SkipStats()
        frames = list(iter_csi_frames(p, skip_stats=stats))
        assert frames == []
        assert stats.wrong_length == 1

    def test_skips_wrong_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_pcap([(_bad_magic_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT)], p)
        stats = SkipStats()
        frames = list(iter_csi_frames(p, skip_stats=stats))
        assert frames == []
        assert stats.wrong_magic == 1

    def test_mixed_valid_and_invalid(self, tmp_path: Path) -> None:
        """3 valid + 1 wrong-port + 1 wrong-magic → 3 frames, 2 skipped."""
        packets = [
            (_valid_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),   # valid
            (_valid_payload(), 9999, 9999),                          # wrong port
            (_valid_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),   # valid
            (_bad_magic_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),# wrong magic
            (_valid_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),   # valid
        ]
        p = tmp_path / "mixed.pcap"
        _write_pcap(packets, p)
        stats = SkipStats()
        frames = list(iter_csi_frames(p, skip_stats=stats))
        assert len(frames) == 3
        assert stats.wrong_port == 1
        assert stats.wrong_magic == 1
        assert stats.total_skipped == 2

    def test_indices_skip_over_invalid_packets(self, tmp_path: Path) -> None:
        """Packet indices must reflect position in the file, not just valid frames."""
        packets = [
            (_valid_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),   # pkt 0 → valid
            (_valid_payload(), 9999, 9999),                          # pkt 1 → skipped
            (_valid_payload(), NEXMON_UDP_PORT, NEXMON_UDP_PORT),   # pkt 2 → valid
        ]
        p = tmp_path / "gaps.pcap"
        _write_pcap(packets, p)
        results = list(iter_csi_frames(p))
        assert results[0][0] == 0
        assert results[1][0] == 2   # pkt 1 was skipped

    def test_skip_stats_none_does_not_crash(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_pcap([(_valid_payload(), 9999, 9999)], p)
        frames = list(iter_csi_frames(p, skip_stats=None))
        assert frames == []


# ---------------------------------------------------------------------------
# read_pcap
# ---------------------------------------------------------------------------

class TestReadPcap:
    def test_returns_list_and_stats(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=4)
        frames, stats = read_pcap(p)
        assert isinstance(frames, list)
        assert isinstance(stats, SkipStats)
        assert len(frames) == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_pcap(tmp_path / "does_not_exist.pcap")

    def test_max_packets_respected(self, tmp_path: Path) -> None:
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=20)
        frames, _ = read_pcap(p, max_packets=3)
        assert len(frames) <= 3

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.pcap"
        _write_pcap([], p)
        frames, stats = read_pcap(p)
        assert frames == []
        assert stats.total_skipped == 0


# ---------------------------------------------------------------------------
# interval_stats
# ---------------------------------------------------------------------------

class TestIntervalStats:
    def _make_frames(self, timestamps_s: List[float]) -> List[NexmonFrame]:
        dummy_header = NexmonHeader(
            rssi=-70, fc=0xDC, seq_fn=0x71,
            chanspec=0x9696, chip_ver=0x6540,
            n_sub=0xFFFF, core_info=0, rssi2=-70,
        )
        dummy_csi = np.zeros((N_SUBCARRIERS, 2), dtype=np.int16)
        return [
            NexmonFrame(index=i, timestamp=t, header=dummy_header, csi_raw=dummy_csi)
            for i, t in enumerate(timestamps_s)
        ]

    def test_single_frame_returns_nan(self) -> None:
        stats = interval_stats(self._make_frames([1.0]))
        assert stats["n_intervals"] == 0
        assert np.isnan(stats["median_ms"])

    def test_empty_returns_nan(self) -> None:
        stats = interval_stats([])
        assert np.isnan(stats["median_ms"])

    def test_uniform_30ms(self) -> None:
        # 11 frames → 10 intervals of exactly 30 ms
        ts = [i * 0.030 for i in range(11)]
        stats = interval_stats(self._make_frames(ts))
        assert stats["n_intervals"] == 10
        assert abs(stats["median_ms"] - 30.0) < 0.01
        assert abs(stats["estimated_fps"] - 1000.0 / 30.0) < 0.1

    def test_keys_present(self) -> None:
        ts = [i * 0.030 for i in range(5)]
        s = interval_stats(self._make_frames(ts))
        for key in ("n_intervals", "min_ms", "p5_ms", "median_ms",
                    "mean_ms", "p95_ms", "max_ms", "estimated_fps"):
            assert key in s


# ---------------------------------------------------------------------------
# SkipStats
# ---------------------------------------------------------------------------

class TestSkipStats:
    def test_initial_all_zero(self) -> None:
        s = SkipStats()
        assert s.total_skipped == 0
        for v in s.as_dict().values():
            assert v == 0

    def test_total_is_sum(self) -> None:
        s = SkipStats(not_udp=3, wrong_port=1, wrong_length=2)
        assert s.total_skipped == 6

    def test_as_dict_contains_total(self) -> None:
        s = SkipStats(wrong_magic=5)
        d = s.as_dict()
        assert d["total_skipped"] == 5
        assert "wrong_magic" in d


# ---------------------------------------------------------------------------
# NexmonHeader properties
# ---------------------------------------------------------------------------

class TestNexmonHeaderProperties:
    def test_seq_num_and_frag_num(self) -> None:
        h = NexmonHeader(rssi=-70, fc=0, seq_fn=0xB2,
                         chanspec=0, chip_ver=0, n_sub=0, core_info=0, rssi2=0)
        assert h.seq_num == 0xB
        assert h.frag_num == 0x2

    def test_seq_fn_zero(self) -> None:
        h = NexmonHeader(rssi=0, fc=0, seq_fn=0,
                         chanspec=0, chip_ver=0, n_sub=0, core_info=0, rssi2=0)
        assert h.seq_num == 0
        assert h.frag_num == 0


# ---------------------------------------------------------------------------
# CLI smoke (subprocess)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_runs_and_exits_zero(self, tmp_path: Path) -> None:
        import subprocess, os
        p = tmp_path / "test.pcap"
        _write_valid_pcap(p, n=5)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m", "src.pcap_reader", str(p)],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "Valid CSI frames" in result.stdout
        assert "5" in result.stdout

    def test_cli_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "src.pcap_reader", str(tmp_path / "nope.pcap")],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0
