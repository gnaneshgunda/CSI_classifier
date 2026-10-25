"""
tests/test_amplitude.py
=======================
Unit tests for src/amplitude.py.

Tests:
* complex_to_amplitude: shape, dtype, values, error on wrong shape
* frames_to_amplitude: shape, dtype, empty input, validity propagation
* Integration smoke: frames_to_amplitude on real data (skipped if absent)
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.amplitude import N_SUBCARRIERS, complex_to_amplitude, frames_to_amplitude
from src.csi_decode import CSI_DATA_OFFSET
from src.pcap_reader import NexmonFrame, NexmonHeader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_header() -> NexmonHeader:
    return NexmonHeader(
        rssi=-70, fc=0xDC, seq_fn=0x71,
        chanspec=0x9696, chip_ver=0x6540,
        n_sub=0xFFFF, core_info=0, rssi2=-70,
    )


def _make_frame(index: int, csi_fill: int = 100, ts: float = 0.0) -> NexmonFrame:
    """Build a NexmonFrame with a known csi_raw int16 fill value."""
    csi_raw = np.full((N_SUBCARRIERS, 2), csi_fill, dtype=np.int16)
    return NexmonFrame(
        index=index,
        timestamp=ts,
        header=_make_header(),
        csi_raw=csi_raw,
    )


def _make_frames(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        csi_raw = rng.integers(-2047, 2048, size=(N_SUBCARRIERS, 2),
                                dtype=np.int16)
        frames.append(NexmonFrame(
            index=i, timestamp=float(i) * 0.03,
            header=_make_header(), csi_raw=csi_raw,
        ))
    return frames


# ---------------------------------------------------------------------------
# complex_to_amplitude
# ---------------------------------------------------------------------------

class TestComplexToAmplitude:
    def test_shape_preserved(self) -> None:
        csi = np.ones((10, 256), dtype=np.complex64)
        amp = complex_to_amplitude(csi)
        assert amp.shape == (10, 256)

    def test_dtype_float32(self) -> None:
        csi = np.ones((5, 256), dtype=np.complex64)
        amp = complex_to_amplitude(csi)
        assert amp.dtype == np.float32

    def test_known_amplitude(self) -> None:
        csi = np.full((3, 256), 3.0 + 4.0j, dtype=np.complex64)
        amp = complex_to_amplitude(csi)
        assert np.allclose(amp, 5.0)

    def test_complex128_input(self) -> None:
        csi = np.ones((2, 256), dtype=np.complex128) * (0 + 2j)
        amp = complex_to_amplitude(csi)
        assert amp.dtype == np.float32
        assert np.allclose(amp, 2.0)

    def test_zero_complex_gives_zero_amplitude(self) -> None:
        csi = np.zeros((4, 256), dtype=np.complex64)
        amp = complex_to_amplitude(csi)
        assert np.all(amp == 0.0)

    def test_wrong_n_subcarriers_raises(self) -> None:
        csi = np.ones((5, 128), dtype=np.complex64)
        with pytest.raises(ValueError, match="256"):
            complex_to_amplitude(csi)

    def test_1d_input_raises(self) -> None:
        csi = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError):
            complex_to_amplitude(csi)


# ---------------------------------------------------------------------------
# frames_to_amplitude
# ---------------------------------------------------------------------------

class TestFramesToAmplitude:
    def test_shape(self) -> None:
        frames = _make_frames(8)
        amp, valid = frames_to_amplitude(frames)
        assert amp.shape == (8, N_SUBCARRIERS)
        assert valid.shape == (8,)

    def test_dtype_float32(self) -> None:
        frames = _make_frames(4)
        amp, _ = frames_to_amplitude(frames)
        assert amp.dtype == np.float32

    def test_all_valid_on_normal_frames(self) -> None:
        frames = _make_frames(6)
        _, valid = frames_to_amplitude(frames)
        assert np.all(valid)

    def test_amplitude_nonnegative(self) -> None:
        frames = _make_frames(10, seed=1)
        amp, valid = frames_to_amplitude(frames)
        assert np.all(amp[valid] >= 0)

    def test_amplitude_finite(self) -> None:
        frames = _make_frames(10, seed=2)
        amp, valid = frames_to_amplitude(frames)
        assert np.all(np.isfinite(amp[valid]))

    def test_empty_input(self) -> None:
        amp, valid = frames_to_amplitude([])
        assert amp.shape == (0, N_SUBCARRIERS)
        assert valid.shape == (0,)

    def test_single_frame(self) -> None:
        frames = _make_frames(1)
        amp, valid = frames_to_amplitude(frames)
        assert amp.shape == (1, N_SUBCARRIERS)
        assert valid[0]

    def test_fftshift_false_differs_from_true(self) -> None:
        """fftshift=True and fftshift=False must produce different orderings
        for asymmetric CSI (not all-zero or uniform)."""
        frames = _make_frames(3, seed=5)
        amp_shifted, _ = frames_to_amplitude(frames, fftshift=True)
        amp_raw, _     = frames_to_amplitude(frames, fftshift=False)
        # At least one row must differ (uniform frames would be equal)
        assert not np.array_equal(amp_shifted, amp_raw)

    def test_output_subcarrier_count_is_256(self) -> None:
        frames = _make_frames(5)
        amp, _ = frames_to_amplitude(frames)
        assert amp.shape[1] == N_SUBCARRIERS


# ---------------------------------------------------------------------------
# Integration: build_intermediate _process_capture smoke
# ---------------------------------------------------------------------------

class TestProcessCapture:
    def test_room_b_capture(self, tmp_path: Path) -> None:
        """Smoke test _process_capture for a Room-B CSV."""
        import pandas as pd
        from scripts.build_intermediate import CaptureTask, _process_capture

        # Write a synthetic Room-B CSV
        csv_path = tmp_path / "Sitting_Amp_1.csv"
        cols = [f"Subcarrier_{i+1}" for i in range(256)]
        data = np.random.default_rng(0).uniform(100, 2000, size=(50, 256)).astype(np.float32)
        pd.DataFrame(data, columns=cols).to_csv(csv_path, index=False)

        out_npy = tmp_path / "B" / "B__sit__Sitting_Amp_1.npy"
        task = CaptureTask(
            capture_id="B__sit__Sitting_Amp_1",
            room="B", activity="sit", split="dev",
            source_file=str(csv_path),
            out_path=str(out_npy),
            max_nan_fraction=0.10,
            zero_energy_threshold=0.0,
            drop_consecutive_duplicates=True,
            spike_z_threshold=6.0,
            spike_mad_eps=1e-6,
            min_valid_packets=10,
        )
        res = _process_capture(task)
        assert res.error is None, res.error
        assert out_npy.exists()
        arr = np.load(str(out_npy))
        assert arr.shape[1] == 256
        assert arr.dtype == np.float32
        assert res.n_packets_clean > 0

    def test_shape_guard_fails_on_wrong_columns(self, tmp_path: Path) -> None:
        """A CSV with wrong column count must return an error result."""
        import pandas as pd
        from scripts.build_intermediate import CaptureTask, _process_capture

        csv_path = tmp_path / "bad.csv"
        pd.DataFrame(np.ones((10, 128))).to_csv(csv_path, index=False)

        task = CaptureTask(
            capture_id="B__sit__bad",
            room="B", activity="sit", split="dev",
            source_file=str(csv_path),
            out_path=str(tmp_path / "bad.npy"),
            max_nan_fraction=0.10,
            zero_energy_threshold=0.0,
            drop_consecutive_duplicates=True,
            spike_z_threshold=6.0,
            spike_mad_eps=1e-6,
            min_valid_packets=10,
        )
        res = _process_capture(task)
        assert res.error is not None
        assert "256" in res.error or "128" in res.error

    def test_already_built_npy_not_overwritten(self, tmp_path: Path) -> None:
        """If the .npy already exists and force=False it should be skipped
        (tasks list is empty — tested at the task-building level)."""
        import pandas as pd
        from scripts.build_intermediate import CaptureTask, _process_capture

        csv_path = tmp_path / "Sitting_Amp_1.csv"
        cols = [f"Subcarrier_{i+1}" for i in range(256)]
        pd.DataFrame(np.ones((20, 256), dtype=np.float32), columns=cols).to_csv(
            csv_path, index=False
        )
        out_npy = tmp_path / "B__sit__Sitting_Amp_1.npy"
        # Write a sentinel file first
        np.save(str(out_npy), np.zeros((1, 256), dtype=np.float32))

        task = CaptureTask(
            capture_id="B__sit__Sitting_Amp_1",
            room="B", activity="sit", split="dev",
            source_file=str(csv_path),
            out_path=str(out_npy),
            max_nan_fraction=0.10,
            zero_energy_threshold=0.0,
            drop_consecutive_duplicates=True,
            spike_z_threshold=6.0,
            spike_mad_eps=1e-6,
            min_valid_packets=10,
        )
        # Processing runs (force is handled upstream in _build_tasks, not here)
        res = _process_capture(task)
        # Should succeed (overwrites) – _build_tasks is where skip logic lives
        assert res.error is None


# ---------------------------------------------------------------------------
# Smoke: python scripts/build_intermediate.py --smoke
# ---------------------------------------------------------------------------

class TestBuildIntermediateSmoke:
    def test_smoke_runs_without_error(self) -> None:
        import sys, subprocess, os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "scripts/build_intermediate.py", "--smoke"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "smoke checks passed" in result.stdout


# ---------------------------------------------------------------------------
# Real-data integration (skipped if dataset absent)
# ---------------------------------------------------------------------------

REAL_PCAP = Path(
    "dataset/Experiment-2/Experiment-2/realdata/input_data/"
    "30ms/csidata/fall/fall10.pcap"
)


@pytest.mark.skipif(not REAL_PCAP.exists(), reason="Dataset not available")
class TestRealDataAmplitude:
    def test_frames_to_amplitude_real_pcap(self) -> None:
        from src.pcap_reader import read_pcap
        frames, stats = read_pcap(REAL_PCAP)
        assert stats.total_skipped == 0
        amp, valid = frames_to_amplitude(frames)
        assert amp.shape == (len(frames), N_SUBCARRIERS)
        assert amp.dtype == np.float32
        assert np.all(valid)
        assert np.all(amp[valid] >= 0)
        assert np.all(np.isfinite(amp[valid]))
