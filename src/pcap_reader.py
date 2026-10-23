"""
src/pcap_reader.py
==================
Iterate a Nexmon CSI .pcap file and yield validated CSI frames.

Hardware context (confirmed from data inspection)
--------------------------------------------------
Platform  : Broadcom BCM4366c0
Bandwidth : 80 MHz
Subcarriers: 256 complex int16 pairs
UDP port  : 5500 (src == dst)
Payload   : 1042 bytes total
  [0:2]   magic     = 0x1111  (little-endian uint16)
  [2:3]   rssi      int8
  [3:4]   fc        uint8  (frame-control low byte)
  [4:5]   seq_fn    uint8  (sequence/fragment number)
  [5:7]   chanspec  uint16 LE
  [7:9]   chip_ver  uint16 LE  (BCM4366c0 → 0x6540)
  [9:11]  n_sub     uint16 LE  (reads 0xFFFF on this chip; not reliable)
  [11:13] core_info uint16 LE
  [13:15] rssi2     int16  LE
  [15:18] pad       3 bytes
  [18:]   csi_data  1024 bytes = 256 × (int16 re, int16 im)

NOTE: The n_sub field is not reliable on BCM4366c0 (often reads 0xFFFF).
      We use the fixed payload-length criterion (1042 bytes) and verify
      the magic bytes instead.

Skip reasons tracked
--------------------
  not_udp          – packet has no UDP layer
  wrong_port       – UDP port ≠ 5500
  wrong_length     – UDP payload length ≠ 1042 bytes
  wrong_magic      – first two bytes ≠ 0x11 0x11
  payload_too_short– raw payload shorter than 18 + 1024 = 1042 bytes
                     (redundant with wrong_length; kept as belt-and-suspenders)

Public API
----------
  NexmonFrame      – named container for one decoded frame
  SkipStats        – skip-reason counter dataclass
  iter_csi_frames  – generator: yields (index, NexmonFrame); also accumulates SkipStats
  read_pcap        – convenience: returns (list[NexmonFrame], SkipStats)

CLI
---
  python -m src.pcap_reader FILE [--max N]

  Prints packet counts, skip breakdown, and inter-packet interval
  statistics (min / median / mean / max / p5 / p95).
"""

from __future__ import annotations

import argparse
import struct
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (confirmed from live data inspection)
# ---------------------------------------------------------------------------

#: Expected UDP payload length for a Nexmon CSI frame on BCM4366c0 @ 80 MHz.
NEXMON_PAYLOAD_BYTES: int = 1042

#: Nexmon CSI magic word (first two bytes of every valid payload).
NEXMON_MAGIC: bytes = b"\x11\x11"

#: UDP port used by Nexmon CSI injection (source and destination).
NEXMON_UDP_PORT: int = 5500

#: Number of CSI subcarriers on BCM4366c0 @ 80 MHz.
N_SUBCARRIERS: int = 256

#: Byte offset where CSI int16 pairs begin inside the UDP payload.
CSI_DATA_OFFSET: int = 18

#: Total bytes of raw CSI data (256 subcarriers × 2 int16 × 2 bytes).
CSI_DATA_BYTES: int = N_SUBCARRIERS * 4  # = 1024

# Struct for the 18-byte Nexmon header (little-endian)
# B  magic[0]    uint8
# B  magic[1]    uint8
# b  rssi        int8
# B  fc          uint8
# B  seq_fn      uint8   (seq_num << 4 | frag_num)
# B  pad1        uint8
# H  chanspec    uint16
# H  chip_ver    uint16
# H  n_sub       uint16  (unreliable on BCM4366c0; treat as informational)
# H  core_info   uint16
# h  rssi2       int16
# B  pad2        uint8
# B  pad3        uint8
_HEADER_FMT: str = "<BBbBBBHHHHhBB"
_HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)  # == 18

assert _HEADER_SIZE == CSI_DATA_OFFSET, (
    f"Header struct size {_HEADER_SIZE} ≠ CSI_DATA_OFFSET {CSI_DATA_OFFSET}"
)

# CSI data: 256 pairs of int16 (re, im), little-endian
_CSI_FMT: str = f"<{N_SUBCARRIERS * 2}h"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NexmonHeader:
    """Decoded Nexmon CSI frame header fields.

    All values are extracted as-is from the payload bytes; no gain
    correction or subcarrier-index mapping is applied here.
    """
    rssi: int          # int8 RSSI value
    fc: int            # frame-control low byte
    seq_fn: int        # raw seq/frag byte
    chanspec: int      # channel specification
    chip_ver: int      # chip version (0x6540 for BCM4366c0)
    n_sub: int         # subcarrier count field (unreliable on BCM4366c0)
    core_info: int     # core/stream bitmask
    rssi2: int         # secondary RSSI estimate

    @property
    def seq_num(self) -> int:
        """Sequence number extracted from seq_fn (upper 4 bits)."""
        return (self.seq_fn >> 4) & 0xF

    @property
    def frag_num(self) -> int:
        """Fragment number extracted from seq_fn (lower 4 bits)."""
        return self.seq_fn & 0xF


@dataclass(frozen=True)
class NexmonFrame:
    """One validated Nexmon CSI frame extracted from a .pcap packet.

    Attributes
    ----------
    index : int
        Zero-based position of this packet in the pcap file (counting
        only packets, not filtered ones).
    timestamp : float
        Packet capture timestamp in seconds (from pcap record header).
    header : NexmonHeader
        Decoded 18-byte Nexmon header.
    csi_raw : np.ndarray
        Shape ``(256, 2)`` int16 array.  Column 0 is real part, column 1
        is imaginary part.  No amplitude conversion is done here.
    """
    index: int
    timestamp: float
    header: NexmonHeader
    csi_raw: np.ndarray   # shape (N_SUBCARRIERS, 2), dtype int16

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NexmonFrame):
            return NotImplemented
        return (
            self.index == other.index
            and self.timestamp == other.timestamp
            and self.header == other.header
            and np.array_equal(self.csi_raw, other.csi_raw)
        )

    def __hash__(self) -> int:  # frozen dataclass with ndarray needs explicit hash
        return hash((self.index, self.timestamp, self.header))


@dataclass
class SkipStats:
    """Counts of packets skipped during pcap iteration, broken down by reason.

    All counters start at zero and are incremented in-place by
    ``iter_csi_frames``.
    """
    not_udp: int = 0
    wrong_port: int = 0
    wrong_length: int = 0
    wrong_magic: int = 0
    payload_too_short: int = 0

    @property
    def total_skipped(self) -> int:
        """Sum of all skip counters."""
        return (
            self.not_udp
            + self.wrong_port
            + self.wrong_length
            + self.wrong_magic
            + self.payload_too_short
        )

    def as_dict(self) -> dict:
        """Return skip counts as an ordered dict (useful for reporting)."""
        return {
            "not_udp": self.not_udp,
            "wrong_port": self.wrong_port,
            "wrong_length": self.wrong_length,
            "wrong_magic": self.wrong_magic,
            "payload_too_short": self.payload_too_short,
            "total_skipped": self.total_skipped,
        }


# ---------------------------------------------------------------------------
# Core parsing helpers
# ---------------------------------------------------------------------------

def _decode_header(payload: bytes) -> NexmonHeader:
    """Unpack the 18-byte Nexmon header from *payload*.

    Parameters
    ----------
    payload:
        Full UDP payload bytes (length must be >= CSI_DATA_OFFSET).

    Returns
    -------
    NexmonHeader with all fields populated.
    """
    (
        _magic0, _magic1,
        rssi, fc, seq_fn, _pad1,
        chanspec, chip_ver, n_sub, core_info,
        rssi2,
        _pad2, _pad3,
    ) = struct.unpack_from(_HEADER_FMT, payload, 0)
    return NexmonHeader(
        rssi=rssi,
        fc=fc,
        seq_fn=seq_fn,
        chanspec=chanspec,
        chip_ver=chip_ver,
        n_sub=n_sub,
        core_info=core_info,
        rssi2=rssi2,
    )


def _decode_csi(payload: bytes) -> np.ndarray:
    """Unpack 256 complex int16 pairs from *payload[CSI_DATA_OFFSET:]*.

    Returns
    -------
    np.ndarray of shape (256, 2), dtype int16.
    Column 0 = real part, column 1 = imaginary part.
    """
    flat = struct.unpack_from(_CSI_FMT, payload, CSI_DATA_OFFSET)
    return np.array(flat, dtype=np.int16).reshape(N_SUBCARRIERS, 2)


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def iter_csi_frames(
    path: Path,
    *,
    skip_stats: Optional[SkipStats] = None,
    max_packets: Optional[int] = None,
) -> Generator[Tuple[int, NexmonFrame], None, None]:
    """Iterate a .pcap file and yield ``(packet_index, NexmonFrame)`` tuples.

    Packets are yielded in capture order.  Non-CSI packets are silently
    skipped; skip reasons are accumulated in *skip_stats* if provided.

    Parameters
    ----------
    path:
        Path to the .pcap file.
    skip_stats:
        If supplied, skip-reason counters are incremented in-place as
        packets are filtered.  Pass a fresh ``SkipStats()`` instance to
        collect statistics; pass ``None`` to ignore them.
    max_packets:
        Stop after reading this many *total* packets (before filtering).
        ``None`` means read the whole file.

    Yields
    ------
    (packet_index, NexmonFrame)
        *packet_index* is the zero-based position of the packet in the
        file (counting ALL packets, including skipped ones).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If scapy is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    try:
        from scapy.utils import PcapReader
        from scapy.layers.inet import UDP
    except ImportError as exc:
        raise ImportError(
            "scapy is required for PCAP reading. "
            "Install with: pip install scapy"
        ) from exc

    stats = skip_stats  # may be None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress scapy libpcap warnings
        with PcapReader(str(path)) as reader:
            for pkt_index, pkt in enumerate(reader):
                if max_packets is not None and pkt_index >= max_packets:
                    break

                # ── Filter 1: must have UDP layer ──────────────────────────
                if not pkt.haslayer(UDP):
                    if stats is not None:
                        stats.not_udp += 1
                    continue

                udp_layer = pkt[UDP]

                # ── Filter 2: port must be 5500 ────────────────────────────
                if udp_layer.dport != NEXMON_UDP_PORT and udp_layer.sport != NEXMON_UDP_PORT:
                    if stats is not None:
                        stats.wrong_port += 1
                    continue

                raw: bytes = bytes(udp_layer.payload)

                # ── Filter 3: payload length must be exactly 1042 bytes ────
                if len(raw) != NEXMON_PAYLOAD_BYTES:
                    if stats is not None:
                        stats.wrong_length += 1
                    continue

                # ── Filter 4: belt-and-suspenders length check ─────────────
                if len(raw) < CSI_DATA_OFFSET + CSI_DATA_BYTES:
                    if stats is not None:
                        stats.payload_too_short += 1
                    continue

                # ── Filter 5: magic bytes must be 0x11 0x11 ───────────────
                if raw[0:2] != NEXMON_MAGIC:
                    if stats is not None:
                        stats.wrong_magic += 1
                    continue

                # ── Decode ─────────────────────────────────────────────────
                header = _decode_header(raw)
                csi_raw = _decode_csi(raw)
                frame = NexmonFrame(
                    index=pkt_index,
                    timestamp=float(pkt.time),
                    header=header,
                    csi_raw=csi_raw,
                )
                yield pkt_index, frame


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def read_pcap(
    path: Path,
    *,
    max_packets: Optional[int] = None,
) -> Tuple[List[NexmonFrame], SkipStats]:
    """Read a full .pcap and return all valid frames plus skip statistics.

    Parameters
    ----------
    path:
        Path to the .pcap file.
    max_packets:
        Optional hard limit on total packets read (before filtering).

    Returns
    -------
    frames : List[NexmonFrame]
        All valid CSI frames in capture order.
    stats : SkipStats
        Counts of skipped packets, broken down by reason.
    """
    stats = SkipStats()
    frames: List[NexmonFrame] = [
        frame
        for _, frame in iter_csi_frames(path, skip_stats=stats, max_packets=max_packets)
    ]
    return frames, stats


# ---------------------------------------------------------------------------
# Inter-packet interval helper
# ---------------------------------------------------------------------------

def interval_stats(frames: List[NexmonFrame]) -> dict:
    """Compute inter-frame timestamp delta statistics.

    Parameters
    ----------
    frames:
        List of ``NexmonFrame`` in capture order.

    Returns
    -------
    dict with keys: n_intervals, min_ms, p5_ms, median_ms, mean_ms,
    p95_ms, max_ms, estimated_fps.  All ``*_ms`` values are in
    milliseconds.  Returns zeros/nan if fewer than 2 frames.
    """
    if len(frames) < 2:
        return {
            "n_intervals": 0,
            "min_ms": float("nan"),
            "p5_ms": float("nan"),
            "median_ms": float("nan"),
            "mean_ms": float("nan"),
            "p95_ms": float("nan"),
            "max_ms": float("nan"),
            "estimated_fps": float("nan"),
        }

    deltas_ms = np.diff([f.timestamp for f in frames]) * 1000.0
    return {
        "n_intervals": int(len(deltas_ms)),
        "min_ms": float(np.min(deltas_ms)),
        "p5_ms": float(np.percentile(deltas_ms, 5)),
        "median_ms": float(np.median(deltas_ms)),
        "mean_ms": float(np.mean(deltas_ms)),
        "p95_ms": float(np.percentile(deltas_ms, 95)),
        "max_ms": float(np.max(deltas_ms)),
        "estimated_fps": float(1000.0 / np.median(deltas_ms)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    # Suppress scapy's "No libpcap provider" warning on Windows so the
    # process exits with code 0 when the actual work succeeds.
    import logging
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        prog="python -m src.pcap_reader",
        description="Inspect a Nexmon CSI .pcap file.",
    )
    parser.add_argument("file", type=Path, help="Path to .pcap file")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N total packets (default: read all)",
    )
    args = parser.parse_args()

    frames, stats = read_pcap(args.file, max_packets=args.max)
    total_read = len(frames) + stats.total_skipped

    print(f"\nFile : {args.file}")
    print("-" * 50)
    print(f"  Total packets read   : {total_read}")
    print(f"  Valid CSI frames     : {len(frames)}")
    print(f"  Skipped (total)      : {stats.total_skipped}")
    if stats.total_skipped:
        for reason, count in stats.as_dict().items():
            if reason == "total_skipped" or count == 0:
                continue
            print(f"    {reason:<22}: {count}")

    if frames:
        iv = interval_stats(frames)
        print(f"\nInter-frame interval (ms)  [n={iv['n_intervals']}]")
        print(f"  min    : {iv['min_ms']:8.3f}")
        print(f"  p5     : {iv['p5_ms']:8.3f}")
        print(f"  median : {iv['median_ms']:8.3f}")
        print(f"  mean   : {iv['mean_ms']:8.3f}")
        print(f"  p95    : {iv['p95_ms']:8.3f}")
        print(f"  max    : {iv['max_ms']:8.3f}")
        print(f"  est FPS: {iv['estimated_fps']:8.1f}")

        h0 = frames[0].header
        print(f"\nFirst frame header")
        print(f"  rssi      : {h0.rssi}")
        print(f"  chanspec  : 0x{h0.chanspec:04x}")
        print(f"  chip_ver  : 0x{h0.chip_ver:04x}")
        print(f"  core_info : 0x{h0.core_info:04x}")
        print(f"  csi shape : {frames[0].csi_raw.shape}")
    print()


if __name__ == "__main__":
    _cli()
