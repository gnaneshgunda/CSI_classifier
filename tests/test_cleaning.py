"""
tests/test_cleaning.py
======================
Unit tests for src/cleaning.py.

Each cleaning rule has its own test class with at least:
- one test that the rule fires when it should
- one test that it does NOT fire on clean data
- boundary / edge conditions

Additional classes cover the report, CSV writer, and the combined
``clean()`` function.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.cleaning import (
    CleaningConfig,
    CleaningReport,
    _consecutive_duplicate_rows,
    _gain_spike_rows,
    _near_zero_rows,
    _non_finite_rows,
    _to_amplitude,
    clean,
    write_report_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_amp(n_rows: int = 10, fill: float = 100.0,
              seed: int = 0) -> np.ndarray:
    """Return a well-behaved (n_rows, 256) float32 amplitude array."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(50.0, 200.0, size=(n_rows, 256)).astype(np.float32)
    return base


def _alive_all(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# _to_amplitude
# ---------------------------------------------------------------------------

class TestToAmplitude:
    def test_float32_passthrough(self) -> None:
        arr = np.ones((5, 256), dtype=np.float32) * 3.0
        out = _to_amplitude(arr)
        assert out.dtype == np.float32
        assert np.allclose(out, 3.0)

    def test_float64_cast_to_float32(self) -> None:
        arr = np.ones((5, 256), dtype=np.float64)
        out = _to_amplitude(arr)
        assert out.dtype == np.float32

    def test_complex_gives_abs(self) -> None:
        arr = np.array([[3.0 + 4.0j] * 256], dtype=np.complex64)
        out = _to_amplitude(arr)
        assert out.dtype == np.float32
        assert np.allclose(out, 5.0)

    def test_complex128_gives_abs(self) -> None:
        arr = np.ones((3, 256), dtype=np.complex128) * (0 + 2j)
        out = _to_amplitude(arr)
        assert np.allclose(out, 2.0)

    def test_negative_real_abs(self) -> None:
        """Real-valued input is passed through as-is (already amplitude).
        Negative values are preserved; abs() is only applied to complex input."""
        arr = np.full((2, 256), -7.0, dtype=np.float32)
        out = _to_amplitude(arr)
        # Real-valued input: cast to float32, no abs applied
        assert np.all(out == -7.0)


# ---------------------------------------------------------------------------
# Rule 1 – non_finite
# ---------------------------------------------------------------------------

class TestNonFiniteRule:
    def test_clean_row_not_flagged(self) -> None:
        amp = _make_amp(5)
        mask = _non_finite_rows(amp, 0.0)
        assert not mask.any()

    def test_single_nan_flagged_at_zero_tolerance(self) -> None:
        amp = _make_amp(5)
        amp[2, 10] = np.nan
        mask = _non_finite_rows(amp, 0.0)
        assert mask[2] and not mask[[0,1,3,4]].any()

    def test_inf_flagged(self) -> None:
        amp = _make_amp(5)
        amp[1, 0] = np.inf
        mask = _non_finite_rows(amp, 0.0)
        assert mask[1]

    def test_neg_inf_flagged(self) -> None:
        amp = _make_amp(5)
        amp[3, 5] = -np.inf
        mask = _non_finite_rows(amp, 0.0)
        assert mask[3]

    def test_fraction_tolerance_allows_few_nans(self) -> None:
        amp = _make_amp(3)
        # 10 NaNs out of 256 = 3.9% < 5% tolerance → should NOT be flagged
        amp[1, :10] = np.nan
        mask = _non_finite_rows(amp, max_nan_fraction=0.05)
        assert not mask[1]

    def test_fraction_tolerance_drops_heavy_nan_row(self) -> None:
        amp = _make_amp(3)
        # 50 NaNs out of 256 = 19.5% > 10% → flagged
        amp[1, :50] = np.nan
        mask = _non_finite_rows(amp, max_nan_fraction=0.10)
        assert mask[1]

    def test_all_nan_row_flagged(self) -> None:
        amp = _make_amp(3)
        amp[0, :] = np.nan
        mask = _non_finite_rows(amp, 0.0)
        assert mask[0]


# ---------------------------------------------------------------------------
# Rule 2 – near_zero_energy
# ---------------------------------------------------------------------------

class TestNearZeroRule:
    def test_normal_rows_not_flagged(self) -> None:
        amp = _make_amp(5)
        alive = _alive_all(5)
        mask = _near_zero_rows(amp, threshold=0.0, alive=alive)
        assert not mask.any()

    def test_all_zero_row_flagged(self) -> None:
        amp = _make_amp(5)
        amp[2, :] = 0.0
        alive = _alive_all(5)
        mask = _near_zero_rows(amp, threshold=0.0, alive=alive)
        assert mask[2]
        assert not mask[[0,1,3,4]].any()

    def test_low_energy_row_flagged_with_threshold(self) -> None:
        amp = np.zeros((3, 256), dtype=np.float32)
        amp[1, :] = 0.01          # energy = 256 * 0.0001 = 0.0256
        alive = _alive_all(3)
        mask = _near_zero_rows(amp, threshold=1.0, alive=alive)
        assert mask[1]

    def test_already_dead_row_not_double_counted(self) -> None:
        amp = _make_amp(5)
        amp[0, :] = 0.0           # zero energy
        alive = _alive_all(5)
        alive[0] = False          # already dropped
        mask = _near_zero_rows(amp, threshold=0.0, alive=alive)
        assert not mask[0]        # not counted again


# ---------------------------------------------------------------------------
# Rule 3 – consecutive_duplicate
# ---------------------------------------------------------------------------

class TestConsecutiveDuplicateRule:
    def test_unique_rows_not_flagged(self) -> None:
        amp = _make_amp(5, seed=99)
        alive = _alive_all(5)
        mask = _consecutive_duplicate_rows(amp, alive)
        assert not mask.any()

    def test_exact_duplicate_consecutive_flagged(self) -> None:
        amp = _make_amp(5)
        amp[3] = amp[2].copy()    # row 3 == row 2
        alive = _alive_all(5)
        mask = _consecutive_duplicate_rows(amp, alive)
        assert mask[3]
        assert not mask[[0,1,2,4]].any()

    def test_non_consecutive_duplicate_not_flagged(self) -> None:
        amp = _make_amp(5)
        amp[4] = amp[0].copy()    # rows 0 and 4 are equal but not adjacent
        alive = _alive_all(5)
        mask = _consecutive_duplicate_rows(amp, alive)
        assert not mask.any()

    def test_three_consecutive_identical_rows(self) -> None:
        amp = _make_amp(5)
        amp[1] = amp[0].copy()
        amp[2] = amp[0].copy()
        alive = _alive_all(5)
        mask = _consecutive_duplicate_rows(amp, alive)
        # row 0 kept, rows 1 and 2 flagged
        assert not mask[0]
        assert mask[1]
        assert mask[2]

    def test_dead_rows_skipped_in_duplicate_check(self) -> None:
        amp = _make_amp(5)
        amp[2] = amp[0].copy()    # row 2 == row 0
        alive = _alive_all(5)
        alive[1] = False          # row 1 is dead → row 2's predecessor is row 0
        mask = _consecutive_duplicate_rows(amp, alive)
        # row 2 should be flagged because nearest preceding alive row is row 0
        assert mask[2]

    def test_near_duplicate_not_flagged(self) -> None:
        """Float perturbation large enough to survive float32 precision must NOT
        be treated as duplicate."""
        amp = _make_amp(3)
        amp[1] = amp[0].copy()
        amp[1, 0] += 1.0          # 1.0 is well above float32 epsilon
        alive = _alive_all(3)
        mask = _consecutive_duplicate_rows(amp, alive)
        assert not mask[1]

    def test_empty_array(self) -> None:
        amp = np.zeros((0, 256), dtype=np.float32)
        alive = np.zeros(0, dtype=bool)
        mask = _consecutive_duplicate_rows(amp, alive)
        assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# Rule 4 – gain_spike
# ---------------------------------------------------------------------------

class TestGainSpikeRule:
    def test_uniform_energy_not_flagged(self) -> None:
        amp = np.ones((20, 256), dtype=np.float32) * 10.0
        alive = _alive_all(20)
        mask = _gain_spike_rows(amp, alive, z_threshold=6.0, eps=1e-6)
        assert not mask.any()

    def test_large_spike_flagged(self) -> None:
        amp = np.ones((20, 256), dtype=np.float32) * 10.0
        # Row 5 has 1000× the normal amplitude → huge energy spike
        amp[5, :] = 1000.0
        alive = _alive_all(20)
        mask = _gain_spike_rows(amp, alive, z_threshold=6.0, eps=1e-6)
        assert mask[5]
        assert not mask[[i for i in range(20) if i != 5]].any()

    def test_dead_rows_excluded_from_stats(self) -> None:
        amp = np.ones((10, 256), dtype=np.float32) * 10.0
        amp[0, :] = 1000.0        # row 0 is the spike
        alive = _alive_all(10)
        alive[0] = False          # already dead; must not appear in spike mask
        mask = _gain_spike_rows(amp, alive, z_threshold=6.0, eps=1e-6)
        assert not mask[0]

    def test_high_threshold_allows_moderate_spike(self) -> None:
        rng = np.random.default_rng(7)
        amp = rng.uniform(8.0, 12.0, size=(20, 256)).astype(np.float32)
        # Row 3 is 20% above mean — not a spike at threshold=100
        amp[3, :] = amp[3, :] * 1.2
        alive = _alive_all(20)
        mask = _gain_spike_rows(amp, alive, z_threshold=100.0, eps=1e-6)
        assert not mask.any()

    def test_single_alive_row_no_crash(self) -> None:
        amp = _make_amp(5)
        alive = np.zeros(5, dtype=bool)
        alive[2] = True           # only one alive row
        mask = _gain_spike_rows(amp, alive, z_threshold=6.0, eps=1e-6)
        assert not mask.any()    # rule skipped, not crashed

    def test_all_constant_no_false_positive(self) -> None:
        """Constant-energy capture: MAD=0, eps prevents div-by-zero, no spikes."""
        amp = np.full((10, 256), 42.0, dtype=np.float32)
        alive = _alive_all(10)
        mask = _gain_spike_rows(amp, alive, z_threshold=6.0, eps=1e-6)
        assert not mask.any()


# ---------------------------------------------------------------------------
# clean() – combined function
# ---------------------------------------------------------------------------

class TestCleanFunction:
    def test_clean_data_unchanged(self) -> None:
        amp = _make_amp(10)
        cleaned, report = clean(amp)
        assert cleaned.shape == (10, 256)
        assert report.n_clean == 10
        assert report.total_dropped == 0
        assert report.valid_mask.all()

    def test_output_dtype_is_float32(self) -> None:
        amp = _make_amp(5).astype(np.float64)
        cleaned, _ = clean(amp)
        assert cleaned.dtype == np.float32

    def test_complex_input_accepted(self) -> None:
        rng = np.random.default_rng(5)
        re = rng.uniform(1.0, 5.0, size=(8, 256)).astype(np.float32)
        im = rng.uniform(1.0, 5.0, size=(8, 256)).astype(np.float32)
        csi = (re + 1j * im).astype(np.complex64)
        cleaned, report = clean(csi)
        assert cleaned.dtype == np.float32
        assert report.n_clean == 8
        assert np.all(cleaned >= 0)

    def test_rule1_drops_nan_rows(self) -> None:
        amp = _make_amp(5)
        amp[2, 0] = np.nan
        cleaned, report = clean(amp, CleaningConfig(max_nan_fraction_per_packet=0.0))
        assert report.dropped_non_finite == 1
        assert cleaned.shape[0] == 4

    def test_rule2_drops_zero_rows(self) -> None:
        amp = _make_amp(5)
        amp[4, :] = 0.0           # last row: zero energy, no dup chain effect
        cleaned, report = clean(amp)
        assert report.dropped_near_zero == 1
        assert cleaned.shape[0] == 4

    def test_rule3_drops_consecutive_dup(self) -> None:
        amp = _make_amp(5)
        amp[3] = amp[2].copy()
        cleaned, report = clean(amp)
        assert report.dropped_consecutive_dup == 1
        assert cleaned.shape[0] == 4

    def test_rule4_drops_spike(self) -> None:
        rng = np.random.default_rng(3)
        amp = rng.uniform(8.0, 12.0, size=(20, 256)).astype(np.float32)
        amp[5, :] = 9999.0
        cleaned, report = clean(amp, CleaningConfig(spike_z_threshold=6.0))
        assert report.dropped_gain_spike == 1
        assert cleaned.shape[0] == 19

    def test_time_order_preserved(self) -> None:
        """After dropping rows, the remaining rows must be in original order."""
        amp = _make_amp(10, seed=7)
        amp[3, :] = 0.0    # will be dropped
        cleaned, report = clean(amp)
        # Indices 0,1,2,4,5,6,7,8,9 survive → check values match original
        expected = amp[report.valid_mask]
        assert np.array_equal(cleaned, expected)

    def test_valid_mask_aligned_to_original_index(self) -> None:
        amp = _make_amp(6)
        amp[2, :] = 0.0   # row 2 dropped
        _, report = clean(amp)
        assert report.valid_mask.shape == (6,)
        assert not report.valid_mask[2]
        assert report.valid_mask[[0,1,3,4,5]].all()

    def test_empty_input(self) -> None:
        amp = np.zeros((0, 256), dtype=np.float32)
        cleaned, report = clean(amp)
        assert cleaned.shape == (0, 256)
        assert report.n_clean == 0
        assert report.total_dropped == 0

    def test_all_rows_dropped(self) -> None:
        amp = np.zeros((5, 256), dtype=np.float32)   # all zero energy
        cleaned, report = clean(amp)
        assert cleaned.shape[0] == 0
        assert report.n_clean == 0

    def test_capture_id_in_report(self) -> None:
        amp = _make_amp(3)
        _, report = clean(amp, capture_id="A__fall__fall10")
        assert report.capture_id == "A__fall__fall10"

    def test_rules_applied_in_order_no_double_count(self) -> None:
        """A NaN row that is also zero-energy should only be counted as non_finite."""
        amp = _make_amp(5)
        amp[1, :] = 0.0
        amp[1, 5] = np.nan        # both zero-energy AND non-finite
        _, report = clean(amp, CleaningConfig(max_nan_fraction_per_packet=0.0))
        # Rule 1 fires first; rule 2 must not also count it
        assert report.dropped_non_finite == 1
        assert report.dropped_near_zero == 0

    def test_cfg_disable_dup_rule(self) -> None:
        amp = _make_amp(5)
        amp[2] = amp[1].copy()
        cfg = CleaningConfig(drop_consecutive_duplicates=False)
        cleaned, report = clean(amp, cfg)
        assert report.dropped_consecutive_dup == 0
        assert cleaned.shape[0] == 5

    def test_cfg_disable_spike_rule(self) -> None:
        amp = np.ones((10, 256), dtype=np.float32) * 10.0
        amp[3, :] = 9999.0
        cfg = CleaningConfig(spike_z_threshold=float("inf"))
        _, report = clean(amp, cfg)
        assert report.dropped_gain_spike == 0


# ---------------------------------------------------------------------------
# CleaningConfig validation
# ---------------------------------------------------------------------------

class TestCleaningConfig:
    def test_defaults_valid(self) -> None:
        CleaningConfig()   # must not raise

    def test_nan_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            CleaningConfig(max_nan_fraction_per_packet=1.5)

    def test_negative_threshold(self) -> None:
        with pytest.raises(ValueError):
            CleaningConfig(zero_energy_threshold=-1.0)

    def test_zero_spike_z(self) -> None:
        with pytest.raises(ValueError):
            CleaningConfig(spike_z_threshold=0.0)


# ---------------------------------------------------------------------------
# CleaningReport helpers
# ---------------------------------------------------------------------------

class TestCleaningReport:
    def test_total_dropped_sum(self) -> None:
        r = CleaningReport(
            dropped_non_finite=2,
            dropped_near_zero=1,
            dropped_consecutive_dup=3,
            dropped_gain_spike=0,
        )
        assert r.total_dropped == 6

    def test_to_dict_keys(self) -> None:
        r = CleaningReport(capture_id="test", n_raw=10, n_clean=8)
        d = r.to_dict()
        for key in ("capture_id","n_raw","n_clean","dropped_non_finite",
                    "dropped_near_zero","dropped_consecutive_dup",
                    "dropped_gain_spike","total_dropped"):
            assert key in d

    def test_to_dict_excludes_valid_mask(self) -> None:
        r = CleaningReport()
        r.valid_mask = np.array([True, False])
        d = r.to_dict()
        assert "valid_mask" not in d


# ---------------------------------------------------------------------------
# write_report_csv
# ---------------------------------------------------------------------------

class TestWriteReportCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cleaning_report.csv"
        reports = [
            CleaningReport(capture_id="A__fall__fall1", n_raw=200, n_clean=190,
                           dropped_non_finite=5, dropped_near_zero=3,
                           dropped_consecutive_dup=2, dropped_gain_spike=0),
            CleaningReport(capture_id="A__sit__sit1",  n_raw=150, n_clean=148,
                           dropped_non_finite=0, dropped_near_zero=1,
                           dropped_consecutive_dup=1, dropped_gain_spike=0),
        ]
        write_report_csv(reports, p)
        assert p.exists()

    def test_correct_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "report.csv"
        reports = [CleaningReport(capture_id=str(i)) for i in range(5)]
        write_report_csv(reports, p)
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 5

    def test_header_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "report.csv"
        write_report_csv([CleaningReport()], p)
        with p.open(newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
        for col in ("capture_id","n_raw","n_clean","total_dropped"):
            assert col in header

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "report.csv"
        write_report_csv([CleaningReport()], p)
        assert p.exists()

    def test_values_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "report.csv"
        r = CleaningReport(capture_id="X", n_raw=100, n_clean=95,
                           dropped_non_finite=2, dropped_near_zero=1,
                           dropped_consecutive_dup=1, dropped_gain_spike=1)
        write_report_csv([r], p)
        with p.open(newline="") as fh:
            row = next(csv.DictReader(fh))
        assert row["capture_id"] == "X"
        assert int(row["n_raw"]) == 100
        assert int(row["total_dropped"]) == 5

    def test_empty_reports_writes_header_only(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        write_report_csv([], p)
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []
