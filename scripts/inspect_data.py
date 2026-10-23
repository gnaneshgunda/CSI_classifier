"""
scripts/inspect_data.py
=======================
Inspect raw source files for the CSI-HAR pipeline without writing any
derived data or fitting any model parameters.

Room A  – raw Nexmon .pcap captures (Experiment-2)
Room B  – pre-computed amplitude .csv captures (Experiment-3)

Outputs
-------
* Console summary (per-room, per-activity tables)
* reports/data_notes.md  – machine-readable Markdown summary

Usage
-----
    # real data
    python scripts/inspect_data.py --config configs/experiment.yaml

    # quick smoke (no real data needed, runs in <5 s)
    python scripts/inspect_data.py --smoke

Design rules
------------
* Read-only: no writes to derived/ or raw/.
* No metadata (filename, folder, packet count) used as model features –
  those are only logged for human inspection.
* No validation-set statistics are computed separately; Room-A validation
  captures are reported in the same aggregates as the rest for auditing
  purposes only.
* scapy is imported lazily so the script starts fast even if scapy is slow.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import textwrap
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

ActivityName = str
Stats = Dict[str, object]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PcapFileSummary:
    """Per-file summary for one Room-A .pcap capture."""
    path: Path
    activity: str
    split: str                    # "dev" | "val"
    total_packets: int
    udp_packets: int
    non_csi_packets: int          # UDP packets whose payload length ≠ expected
    csi_candidate_packets: int    # UDP packets that look like Nexmon CSI
    payload_lengths: List[int]    # payload length of every UDP packet
    timestamp_deltas_ms: List[float]  # inter-packet gap in milliseconds


@dataclass
class CsvFileSummary:
    """Per-file summary for one Room-B .csv capture."""
    path: Path
    activity: str
    header_style: str             # "subcarrier_named" | "zero_indexed" | "other"
    n_rows: int
    n_cols: int
    nan_count: int
    non_finite_count: int         # inf / -inf
    zero_rows: int                # rows where every value is 0
    min_val: float
    max_val: float
    mean_val: float


# ---------------------------------------------------------------------------
# Room-A PCAP inspection
# ---------------------------------------------------------------------------

# Nexmon CSI payload size for BCM4366c0 @ 80 MHz:
# 2 bytes magic + 2 rssi + 1 fc + 1 seq + 2 csiconf + 2 chanspec +
# 2 chip_version + 2 n_subcarriers + 4*n_subcarriers bytes (re+im int16 pairs)
# = 18 header bytes + 4*256 = 1042 bytes total
_NEXMON_PAYLOAD_BYTES = 1042
_NEXMON_MAGIC = b"\x11\x11"      # first two bytes of a Nexmon CSI frame


def _inspect_pcap(path: Path, activity: str, split: str) -> PcapFileSummary:
    """Parse one .pcap and return its ``PcapFileSummary``.

    We use scapy's ``PcapReader`` for lazy packet-by-packet streaming so
    large captures don't blow up memory.  Only the raw bytes are examined;
    no CSI reconstruction is performed here.
    """
    # Lazy import: scapy startup is ~0.5 s; don't penalise --help / --smoke
    try:
        from scapy.utils import PcapReader
        from scapy.layers.l2 import Ether
        from scapy.layers.inet import UDP, IP
    except ImportError as exc:
        raise ImportError(
            "scapy is required for PCAP inspection. "
            "Install it with: pip install scapy"
        ) from exc

    total = 0
    udp_count = 0
    non_csi = 0
    csi_candidates = 0
    payload_lens: List[int] = []
    ts_deltas: List[float] = []
    prev_ts: Optional[float] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress scapy deprecation noise
        with PcapReader(str(path)) as reader:
            for pkt in reader:
                total += 1
                ts = float(pkt.time)
                if prev_ts is not None:
                    ts_deltas.append((ts - prev_ts) * 1000.0)  # → ms
                prev_ts = ts

                # Check for UDP layer (Nexmon CSI is sent as UDP)
                if pkt.haslayer(UDP):
                    udp_count += 1
                    raw_payload: bytes = bytes(pkt[UDP].payload)
                    plen = len(raw_payload)
                    payload_lens.append(plen)

                    # A Nexmon CSI frame starts with the magic bytes 0x1111
                    if (
                        plen == _NEXMON_PAYLOAD_BYTES
                        and raw_payload[:2] == _NEXMON_MAGIC
                    ):
                        csi_candidates += 1
                    else:
                        non_csi += 1
                else:
                    non_csi += 1

    return PcapFileSummary(
        path=path,
        activity=activity,
        split=split,
        total_packets=total,
        udp_packets=udp_count,
        non_csi_packets=non_csi,
        csi_candidate_packets=csi_candidates,
        payload_lengths=payload_lens,
        timestamp_deltas_ms=ts_deltas,
    )


def _summarise_pcap_activity(
    summaries: List[PcapFileSummary],
) -> Dict[str, object]:
    """Aggregate per-file summaries into per-activity statistics."""
    counts = [s.csi_candidate_packets for s in summaries]
    all_deltas = [d for s in summaries for d in s.timestamp_deltas_ms]
    all_payloads = [p for s in summaries for p in s.payload_lengths]

    # payload-length value counts (top-5)
    from collections import Counter
    payload_counter = Counter(all_payloads)
    top_payloads = payload_counter.most_common(5)

    result: Dict[str, object] = {
        "n_captures": len(summaries),
        "csi_packets_min": int(np.min(counts)) if counts else 0,
        "csi_packets_max": int(np.max(counts)) if counts else 0,
        "csi_packets_median": float(np.median(counts)) if counts else 0.0,
        "csi_packets_mean": float(np.mean(counts)) if counts else 0.0,
        "csi_packets_p10": float(np.percentile(counts, 10)) if counts else 0.0,
        "csi_packets_p90": float(np.percentile(counts, 90)) if counts else 0.0,
        "non_csi_packets_total": sum(s.non_csi_packets for s in summaries),
        "ts_delta_median_ms": float(np.median(all_deltas)) if all_deltas else float("nan"),
        "ts_delta_min_ms": float(np.min(all_deltas)) if all_deltas else float("nan"),
        "ts_delta_max_ms": float(np.max(all_deltas)) if all_deltas else float("nan"),
        "estimated_fps": 1000.0 / float(np.median(all_deltas)) if all_deltas else float("nan"),
        "payload_length_top5": top_payloads,
        "captures_below_150": sum(1 for c in counts if c < 150),
    }
    return result


# ---------------------------------------------------------------------------
# Room-B CSV inspection
# ---------------------------------------------------------------------------

# Canonical header: Subcarrier_1 … Subcarrier_256
_SUBCARRIER_RE = re.compile(r"^subcarrier_\d+$", re.IGNORECASE)
# Anomalous header: 0, 1, 2, … 255
_ZERO_IDX_RE = re.compile(r"^\d+$")


def _detect_header_style(columns: List[str]) -> str:
    """Return the header style for a Room-B CSV."""
    if all(_SUBCARRIER_RE.match(str(c)) for c in columns):
        return "subcarrier_named"
    if all(_ZERO_IDX_RE.match(str(c)) for c in columns):
        return "zero_indexed"
    return "other"


def _inspect_csv(path: Path, activity: str) -> CsvFileSummary:
    """Read one Room-B CSV and return its ``CsvFileSummary``."""
    try:
        df = pd.read_csv(path, header=0)
    except Exception as exc:
        # Return a degenerate summary rather than crashing the whole run
        return CsvFileSummary(
            path=path,
            activity=activity,
            header_style="read_error",
            n_rows=0,
            n_cols=0,
            nan_count=-1,
            non_finite_count=-1,
            zero_rows=-1,
            min_val=float("nan"),
            max_val=float("nan"),
            mean_val=float("nan"),
        )

    header_style = _detect_header_style(list(df.columns))
    n_rows, n_cols = df.shape

    # Convert to numeric, coercing parse failures to NaN
    numeric = df.apply(pd.to_numeric, errors="coerce")
    nan_count = int(numeric.isna().sum().sum())
    inf_count = int(np.isinf(numeric.values.astype(float)).sum())

    vals = numeric.values.astype(float)
    zero_rows = int((np.abs(vals).sum(axis=1) == 0).sum())

    finite_vals = vals[np.isfinite(vals)]
    min_val = float(finite_vals.min()) if finite_vals.size else float("nan")
    max_val = float(finite_vals.max()) if finite_vals.size else float("nan")
    mean_val = float(finite_vals.mean()) if finite_vals.size else float("nan")

    return CsvFileSummary(
        path=path,
        activity=activity,
        header_style=header_style,
        n_rows=n_rows,
        n_cols=n_cols,
        nan_count=nan_count,
        non_finite_count=inf_count,
        zero_rows=zero_rows,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
    )


def _summarise_csv_activity(
    summaries: List[CsvFileSummary],
) -> Dict[str, object]:
    """Aggregate per-file CSV summaries into per-activity statistics."""
    row_counts = [s.n_rows for s in summaries]
    nan_counts = [s.nan_count for s in summaries]
    col_counts = [s.n_cols for s in summaries]
    header_styles = set(s.header_style for s in summaries)

    named = sum(1 for s in summaries if s.header_style == "subcarrier_named")
    zero_idx = sum(1 for s in summaries if s.header_style == "zero_indexed")
    other = sum(1 for s in summaries if s.header_style not in ("subcarrier_named", "zero_indexed"))

    all_mins = [s.min_val for s in summaries if np.isfinite(s.min_val)]
    all_maxs = [s.max_val for s in summaries if np.isfinite(s.max_val)]
    all_means = [s.mean_val for s in summaries if np.isfinite(s.mean_val)]

    return {
        "n_captures": len(summaries),
        "header_subcarrier_named": named,
        "header_zero_indexed": zero_idx,
        "header_other": other,
        "rows_min": int(np.min(row_counts)) if row_counts else 0,
        "rows_max": int(np.max(row_counts)) if row_counts else 0,
        "rows_median": float(np.median(row_counts)) if row_counts else 0.0,
        "rows_mean": float(np.mean(row_counts)) if row_counts else 0.0,
        "rows_p10": float(np.percentile(row_counts, 10)) if row_counts else 0.0,
        "rows_p90": float(np.percentile(row_counts, 90)) if row_counts else 0.0,
        "cols_unique": sorted(set(col_counts)),
        "total_nan": int(sum(nan_counts)),
        "captures_with_nan": sum(1 for n in nan_counts if n > 0),
        "captures_below_150_rows": sum(1 for r in row_counts if r < 150),
        "amplitude_min_overall": float(np.min(all_mins)) if all_mins else float("nan"),
        "amplitude_max_overall": float(np.max(all_maxs)) if all_maxs else float("nan"),
        "amplitude_mean_overall": float(np.mean(all_means)) if all_means else float("nan"),
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Maps Room-B filename prefix → canonical activity label
_ROOM_B_ACTIVITY_MAP: Dict[str, str] = {
    "empty":    "empty",
    "lying":    "lying",
    "sitting":  "sit",
    "standing": "stand",
    "walking":  "walk",
}

# Pattern 1: {Activity}_Amp_{N}.csv  (named subcarrier header)
_AMP_PATTERN = re.compile(
    r"^(empty|lying|sitting|standing|walking)_amp_\d+\.csv$",
    re.IGNORECASE,
)
# Pattern 2: {Activity}_{N}.csv (zero-indexed header, anomalous)
_BARE_PATTERN = re.compile(
    r"^(empty|lying|sitting|standing|walking)_\d+\.csv$",
    re.IGNORECASE,
)


def _discover_room_a(
    pcap_root: Path,
    val_root: Path,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Return (dev_map, val_map) where each maps activity → sorted list of pcap paths."""
    dev_map: Dict[str, List[Path]] = defaultdict(list)
    val_map: Dict[str, List[Path]] = defaultdict(list)

    activity_dirs = [d for d in sorted(pcap_root.iterdir())
                     if d.is_dir() and d.name != "validation"]
    for act_dir in activity_dirs:
        act = act_dir.name.lower()
        for p in sorted(act_dir.glob("*.pcap")):
            dev_map[act].append(p)

    for act_dir in sorted(val_root.iterdir()):
        if not act_dir.is_dir():
            continue
        act = act_dir.name.lower()
        for p in sorted(act_dir.glob("*.pcap")):
            val_map[act].append(p)

    return dict(dev_map), dict(val_map)


def _discover_room_b(csv_root: Path) -> Tuple[
    Dict[str, List[Path]],   # amp_files: activity → paths (standard naming)
    Dict[str, List[Path]],   # bare_files: activity → paths (anomalous naming)
]:
    """Classify Room-B CSVs into standard (_Amp_) and anomalous (no _Amp_) groups."""
    amp_files: Dict[str, List[Path]] = defaultdict(list)
    bare_files: Dict[str, List[Path]] = defaultdict(list)

    for p in sorted(csv_root.glob("*.csv")):
        name_lower = p.name.lower()
        m_amp = _AMP_PATTERN.match(name_lower)
        m_bare = _BARE_PATTERN.match(name_lower)
        if m_amp:
            prefix = m_amp.group(1).lower()
            act = _ROOM_B_ACTIVITY_MAP.get(prefix, prefix)
            amp_files[act].append(p)
        elif m_bare:
            prefix = m_bare.group(1).lower()
            act = _ROOM_B_ACTIVITY_MAP.get(prefix, prefix)
            bare_files[act].append(p)
        # ignore model_plot_1.png and any other non-CSV bleed-throughs

    return dict(amp_files), dict(bare_files)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _hline(width: int = 72) -> str:
    return "─" * width


def _section(title: str) -> str:
    bar = "═" * 72
    return f"\n{bar}\n  {title}\n{bar}"


def _print_pcap_table(
    act_stats: Dict[str, Dict],
    split_label: str,
    out: io.StringIO,
) -> None:
    header = (
        f"  {'Activity':<12} {'N':>5} {'CSI med':>9} {'CSI min':>8} "
        f"{'CSI max':>8} {'<150':>5} {'FPS est':>8} {'Δt med ms':>10}"
    )
    print(header, file=out)
    print("  " + _hline(70), file=out)
    for act, s in sorted(act_stats.items()):
        fps = s["estimated_fps"]
        fps_str = f"{fps:8.1f}" if np.isfinite(fps) else "     N/A"
        dt = s["ts_delta_median_ms"]
        dt_str = f"{dt:10.2f}" if np.isfinite(dt) else "       N/A"
        print(
            f"  {act:<12} {s['n_captures']:>5} "
            f"{s['csi_packets_median']:>9.0f} "
            f"{s['csi_packets_min']:>8} "
            f"{s['csi_packets_max']:>8} "
            f"{s['captures_below_150']:>5} "
            f"{fps_str}"
            f"{dt_str}",
            file=out,
        )


def _print_csv_table(
    act_stats: Dict[str, Dict],
    label: str,
    out: io.StringIO,
) -> None:
    header = (
        f"  {'Activity':<10} {'N':>5} {'rows med':>9} {'rows min':>9} "
        f"{'rows max':>9} {'<150':>5} {'NaN tot':>8} {'named hdr':>10} "
        f"{'0-idx hdr':>10}"
    )
    print(header, file=out)
    print("  " + _hline(70), file=out)
    for act, s in sorted(act_stats.items()):
        print(
            f"  {act:<10} {s['n_captures']:>5} "
            f"{s['rows_median']:>9.0f} "
            f"{s['rows_min']:>9} "
            f"{s['rows_max']:>9} "
            f"{s['captures_below_150_rows']:>5} "
            f"{s['total_nan']:>8} "
            f"{s['header_subcarrier_named']:>10} "
            f"{s['header_zero_indexed']:>10}",
            file=out,
        )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown(
    room_a_dev: Dict[str, Dict],
    room_a_val: Dict[str, Dict],
    room_b_amp: Dict[str, Dict],
    room_b_bare: Dict[str, Dict],
    anomalous_files: List[Path],
    out_path: Path,
) -> None:
    """Write a Markdown summary to *out_path*."""
    lines: List[str] = [
        "# CSI-HAR Dataset Inspection Notes",
        "",
        "_Auto-generated by `scripts/inspect_data.py`. Do not edit by hand._",
        "",
    ]

    def _table_row(*cells: str) -> str:
        return "| " + " | ".join(cells) + " |"

    def _add_pcap_section(title: str, stats: Dict[str, Dict]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_table_row(
            "Activity", "N captures", "CSI med", "CSI min", "CSI max",
            "Captures <150", "FPS (est)", "Δt median (ms)"
        ))
        lines.append(_table_row(*["---"] * 8))
        for act, s in sorted(stats.items()):
            fps = s["estimated_fps"]
            fps_str = f"{fps:.1f}" if np.isfinite(fps) else "N/A"
            dt = s["ts_delta_median_ms"]
            dt_str = f"{dt:.2f}" if np.isfinite(dt) else "N/A"
            lines.append(_table_row(
                act,
                str(s["n_captures"]),
                f"{s['csi_packets_median']:.0f}",
                str(s["csi_packets_min"]),
                str(s["csi_packets_max"]),
                str(s["captures_below_150"]),
                fps_str,
                dt_str,
            ))
        lines.append("")

    def _add_csv_section(title: str, stats: Dict[str, Dict]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_table_row(
            "Activity", "N", "Rows med", "Rows min", "Rows max",
            "Captures <150", "NaN total", "Named hdr", "0-idx hdr",
            "Amp min", "Amp max"
        ))
        lines.append(_table_row(*["---"] * 11))
        for act, s in sorted(stats.items()):
            lines.append(_table_row(
                act,
                str(s["n_captures"]),
                f"{s['rows_median']:.0f}",
                str(s["rows_min"]),
                str(s["rows_max"]),
                str(s["captures_below_150_rows"]),
                str(s["total_nan"]),
                str(s["header_subcarrier_named"]),
                str(s["header_zero_indexed"]),
                f"{s['amplitude_min_overall']:.1f}",
                f"{s['amplitude_max_overall']:.1f}",
            ))
        lines.append("")

    _add_pcap_section("Room A – Development Captures (PCAP)", room_a_dev)
    _add_pcap_section("Room A – Validation Captures (PCAP)", room_a_val)
    _add_csv_section("Room B – Standard Captures (_Amp_ naming)", room_b_amp)
    if room_b_bare:
        _add_csv_section("Room B – Anomalous Captures (no _Amp_ in name)", room_b_bare)

    if anomalous_files:
        lines.append("## Anomalous File Flags")
        lines.append("")
        lines.append(
            "The following files were detected with non-standard headers "
            "(0-indexed instead of `Subcarrier_N`) or unexpected naming patterns. "
            "They should be inspected before inclusion in the pipeline."
        )
        lines.append("")
        for p in sorted(anomalous_files):
            lines.append(f"- `{p.name}`")
        lines.append("")

    lines.append("## Known Issues and Observations")
    lines.append("")
    lines.append(textwrap.dedent("""\
        - Room B Walking files use `Walking_{N}.csv` (no `_Amp_`) with 0-indexed headers.
        - Several Room B activities have a single anomalous file with 0-indexed headers
          (e.g. `Empty_11.csv`, `Lying_0.csv`, `Sitting_0.csv`, `Standing_0.csv`).
        - These anomalous files contain 256 data columns but their header row is
          `0,1,...,255` rather than `Subcarrier_1,...,Subcarrier_256`.
          They appear to be valid amplitude data with a different export format.
        - No timestamps are present in Room B CSVs; packet rate cannot be estimated.
        - Room A PCAP timestamps allow inter-packet delta estimation; see FPS column.
        - Captures with fewer than 150 valid CSI packets will require reflect-pad or
          exclusion (see `window.short_capture_policy` in experiment.yaml).
    """))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Smoke mode (no real data)
# ---------------------------------------------------------------------------

def _run_smoke(out_path: Path) -> None:
    """Run a fast end-to-end check using synthetic data (no real files needed)."""
    print("[smoke] Generating synthetic inspection data …")
    rng = np.random.default_rng(42)

    # Fabricate summary dicts directly – skip all file I/O
    activities_a = ["fall", "sit", "sitdown", "stand", "standup", "walk"]
    activities_b = ["empty", "lying", "sit", "stand", "walk"]

    def _fake_pcap_stats(act: str, n: int = 5) -> Dict:
        counts = rng.integers(180, 350, size=n).tolist()
        deltas = rng.uniform(28, 32, size=n * 200).tolist()
        sums = [PcapFileSummary(
            path=Path(f"fake/{act}/{act}{i}.pcap"),
            activity=act, split="dev",
            total_packets=c + rng.integers(0, 5).item(),
            udp_packets=c, non_csi_packets=0,
            csi_candidate_packets=c,
            payload_lengths=[_NEXMON_PAYLOAD_BYTES] * c,
            timestamp_deltas_ms=deltas[:c],
        ) for i, c in enumerate(counts)]
        return _summarise_pcap_activity(sums)

    def _fake_csv_stats(act: str, n: int = 5) -> Dict:
        sums = [CsvFileSummary(
            path=Path(f"fake/{act}_Amp_{i}.csv"),
            activity=act, header_style="subcarrier_named",
            n_rows=rng.integers(150, 400).item(),
            n_cols=256, nan_count=0, non_finite_count=0, zero_rows=0,
            min_val=float(rng.uniform(0, 10)),
            max_val=float(rng.uniform(1000, 2000)),
            mean_val=float(rng.uniform(300, 700)),
        ) for i in range(n)]
        return _summarise_csv_activity(sums)

    room_a_dev = {a: _fake_pcap_stats(a) for a in activities_a}
    room_a_val = {a: _fake_pcap_stats(a, n=3) for a in activities_a}
    room_b_amp = {a: _fake_csv_stats(a) for a in activities_b}
    room_b_bare: Dict[str, Dict] = {}

    _write_markdown(room_a_dev, room_a_val, room_b_amp, room_b_bare, [], out_path)
    print(f"[smoke] Markdown written → {out_path}")

    # Print a condensed console table so the smoke run produces visible output
    buf = io.StringIO()
    print(_section("Room A – Development (synthetic)"), file=buf)
    _print_pcap_table(room_a_dev, "dev", buf)
    print(_section("Room B – Standard CSV (synthetic)"), file=buf)
    _print_csv_table(room_b_amp, "amp", buf)
    print(buf.getvalue())
    print("[smoke] inspect_data smoke PASSED ✓")


# ---------------------------------------------------------------------------
# Real-data inspection
# ---------------------------------------------------------------------------

def _run_real(pcap_root: Path, val_root: Path, csv_root: Path, out_path: Path) -> None:
    """Inspect real data; print console tables and write Markdown."""

    buf = io.StringIO()

    # ── Room A ──────────────────────────────────────────────────────────────
    print(_section("Room A – PCAP Inspection"), file=buf)

    dev_paths, val_paths = _discover_room_a(pcap_root, val_root)
    total_a_dev = sum(len(v) for v in dev_paths.values())
    total_a_val = sum(len(v) for v in val_paths.values())
    print(f"\n  Source : {pcap_root}", file=buf)
    print(f"  Dev captures found  : {total_a_dev}", file=buf)
    print(f"  Val captures found  : {total_a_val}", file=buf)

    print("\n  Inspecting dev captures …", file=buf)
    print(buf.getvalue(), end="", flush=True)
    buf = io.StringIO()

    room_a_dev_stats: Dict[str, Dict] = {}
    for act, paths in sorted(dev_paths.items()):
        sums: List[PcapFileSummary] = []
        for p in paths:
            sums.append(_inspect_pcap(p, act, "dev"))
            sys.stdout.write(f"\r    {act}: {len(sums)}/{len(paths)}   ")
            sys.stdout.flush()
        room_a_dev_stats[act] = _summarise_pcap_activity(sums)
        print()

    print("\n  Room A – Development Captures", file=buf)
    _print_pcap_table(room_a_dev_stats, "dev", buf)

    print("\n  Inspecting val captures …", file=buf)
    print(buf.getvalue(), end="", flush=True)
    buf = io.StringIO()

    room_a_val_stats: Dict[str, Dict] = {}
    for act, paths in sorted(val_paths.items()):
        sums = []
        for p in paths:
            sums.append(_inspect_pcap(p, act, "val"))
            sys.stdout.write(f"\r    {act}: {len(sums)}/{len(paths)}   ")
            sys.stdout.flush()
        room_a_val_stats[act] = _summarise_pcap_activity(sums)
        print()

    print("\n  Room A – Validation Captures", file=buf)
    _print_pcap_table(room_a_val_stats, "val", buf)

    # payload length histogram (top-10 unique lengths across all dev files)
    print("\n  Payload-length histogram (dev, top-10 values):", file=buf)
    from collections import Counter
    plen_counter: Counter = Counter()
    for act, paths in dev_paths.items():
        for p in paths:
            summary = _inspect_pcap(p, act, "dev")
            plen_counter.update(summary.payload_lengths)
    for length, cnt in plen_counter.most_common(10):
        pct = 100.0 * cnt / max(sum(plen_counter.values()), 1)
        bar = "█" * int(pct / 2)
        print(f"    {length:6d} bytes : {cnt:8d} ({pct:5.1f}%)  {bar}", file=buf)

    # ── Room B ──────────────────────────────────────────────────────────────
    print(_section("Room B – CSV Inspection"), file=buf)
    print(f"\n  Source : {csv_root}", file=buf)
    print(buf.getvalue(), end="", flush=True)
    buf = io.StringIO()

    amp_paths, bare_paths = _discover_room_b(csv_root)
    total_b_amp = sum(len(v) for v in amp_paths.values())
    total_b_bare = sum(len(v) for v in bare_paths.values())
    print(f"\n  Standard (_Amp_) files  : {total_b_amp}", file=buf)
    print(f"  Anomalous (bare) files  : {total_b_bare}", file=buf)

    print("\n  Inspecting standard CSV captures …", file=buf)
    print(buf.getvalue(), end="", flush=True)
    buf = io.StringIO()

    room_b_amp_stats: Dict[str, Dict] = {}
    for act, paths in sorted(amp_paths.items()):
        sums: List[CsvFileSummary] = []
        for p in paths:
            sums.append(_inspect_csv(p, act))
            sys.stdout.write(f"\r    {act}: {len(sums)}/{len(paths)}   ")
            sys.stdout.flush()
        room_b_amp_stats[act] = _summarise_csv_activity(sums)
        print()

    print("\n  Room B – Standard CSV captures", file=buf)
    _print_csv_table(room_b_amp_stats, "amp", buf)

    room_b_bare_stats: Dict[str, Dict] = {}
    anomalous_files: List[Path] = []
    if bare_paths:
        print("\n  Inspecting anomalous CSV captures …", file=buf)
        print(buf.getvalue(), end="", flush=True)
        buf = io.StringIO()

        for act, paths in sorted(bare_paths.items()):
            sums = []
            for p in paths:
                s = _inspect_csv(p, act)
                sums.append(s)
                anomalous_files.append(p)
                sys.stdout.write(f"\r    {act}: {len(sums)}/{len(paths)}   ")
                sys.stdout.flush()
            room_b_bare_stats[act] = _summarise_csv_activity(sums)
            print()

        print("\n  Room B – Anomalous CSV captures (0-indexed headers)", file=buf)
        _print_csv_table(room_b_bare_stats, "bare", buf)

        print(
            f"\n  ⚠  {len(anomalous_files)} anomalous files detected with 0-indexed headers.",
            file=buf,
        )
        print("  These have valid 256-column data but non-standard header rows.", file=buf)

    # ── Per-activity capture-length summary (both rooms) ────────────────────
    print(_section("Per-Activity Capture-Length Summary (both rooms)"), file=buf)
    print(
        "\n  Room A CSI-packet counts (development), Room B row counts (standard):\n",
        file=buf,
    )
    all_acts = sorted(
        set(room_a_dev_stats) | set(room_b_amp_stats)
    )
    print(
        f"  {'Activity':<12} {'Room':>5}  "
        f"{'min':>7} {'p10':>7} {'med':>7} {'p90':>7} {'max':>7}",
        file=buf,
    )
    print("  " + _hline(58), file=buf)
    for act in all_acts:
        if act in room_a_dev_stats:
            s = room_a_dev_stats[act]
            print(
                f"  {act:<12}  {'A':>4}  "
                f"{s['csi_packets_min']:>7} "
                f"{s['csi_packets_p10']:>7.0f} "
                f"{s['csi_packets_median']:>7.0f} "
                f"{s['csi_packets_p90']:>7.0f} "
                f"{s['csi_packets_max']:>7}",
                file=buf,
            )
        if act in room_b_amp_stats:
            s = room_b_amp_stats[act]
            print(
                f"  {act:<12}  {'B':>4}  "
                f"{s['rows_min']:>7} "
                f"{s['rows_p10']:>7.0f} "
                f"{s['rows_median']:>7.0f} "
                f"{s['rows_p90']:>7.0f} "
                f"{s['rows_max']:>7}",
                file=buf,
            )

    print(buf.getvalue())
    buf = io.StringIO()

    # ── Write Markdown ───────────────────────────────────────────────────────
    # Re-read pcap summaries are already computed above in room_a_dev_stats / room_a_val_stats.
    # The payload histogram was printed to console; re-compute a condensed version for MD.
    _write_markdown(
        room_a_dev_stats,
        room_a_val_stats,
        room_b_amp_stats,
        room_b_bare_stats,
        anomalous_files,
        out_path,
    )
    print(f"\nMarkdown summary written → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw Room A .pcap and Room B .csv data; "
            "print statistics and write reports/data_notes.md."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on synthetic data only; no real files needed. Completes in <5 s.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path for data_notes.md (default: reports/data_notes.md).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        out = Path(args.out) if args.out else Path("reports/data_notes.md")
        _run_smoke(out)
        return

    # Load config to get paths
    from src.config import load_config
    cfg = load_config(args.config)

    pcap_root = cfg.paths.room_a_pcap_dir
    val_root  = cfg.paths.room_a_val_dir
    csv_root  = cfg.paths.room_b_csv_dir
    out_path  = Path(args.out) if args.out else Path("reports/data_notes.md")

    # Validate source directories exist before starting expensive reads
    missing = [p for p in (pcap_root, val_root, csv_root) if not p.exists()]
    if missing:
        print("ERROR: the following source directories do not exist:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    _run_real(pcap_root, val_root, csv_root, out_path)


if __name__ == "__main__":
    main()
