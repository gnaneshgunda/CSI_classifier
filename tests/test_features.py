"""
tests/test_features.py
======================
Unit tests for src/features.py.

Verifies:
1. Feature dimensions:
   - Constant FEATURE_DIM == 524
   - Layout: 256 subcarrier means, 256 subcarrier stds, 4 spatial bands,
     5 Doppler frequency bands, 3 temporal statistics
   - extract_features returns shape (524,) float32
   - extract_features_batch returns shape (N, 524) float32
   - feature_names() returns 524 distinct strings with expected nomenclature
2. Numerical calculations:
   - Known constant windows yield exact known means, zero std, exact spatial energies,
     zero temporal variance/range, and exact Shannon entropy ln(T)
   - Subcarrier statistics match numpy mean and population std (ddof=0)
   - Spatial band partitioning separates subcarriers across 4 equal-width bands
   - Pure determinism across repeated executions
3. STFT frequency bands:
   - Doppler bands match the 5 configured frequency ranges
   - Synthetic pure sine signals produce dominant spectral energy in matching Doppler bands
   - Static DC signal concentrates maximum energy in static Doppler band (0 - 0.5 Hz)
   - packet_rate parameter scales physical frequency binning
   - Clamping behaviour when T < n_fft (e.g. T=16 < 32)
4. Temporal statistics:
   - All-zero window produces zero variance, zero entropy, and zero range without NaNs
   - Single-pulse (impulse) power envelope produces expected range and entropy
   - Non-negative variance and range on varying power profiles
5. Invalid inputs:
   - Wrong dimensions (1D, 3D for single window; 2D, 4D for batch) raise ValueError
   - Wrong subcarrier count (not 256) raises ValueError
   - Zero-length time dimension (T=0) raises ValueError
   - Extreme input values (near 0, large) maintain numerical stability without NaNs or Infs
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from src.features import (
    FEATURE_DIM,
    _DOPPLER_EDGES_HZ,
    _N_DOPPLER_BANDS,
    _N_SPATIAL_BANDS,
    _SUBCARRIER_STAT_DIM,
    _spatial_band_energies,
    _stft_doppler_energies,
    _subcarrier_stats,
    _temporal_features,
    extract_features,
    extract_features_batch,
    feature_names,
)


# ===========================================================================
# 1. Feature Dimensions and Metadata
# ===========================================================================

class TestFeatureDimensionsAndLayout:
    """Tests for feature vector dimensionality, layout, and naming."""

    def test_feature_dim_constant(self) -> None:
        """FEATURE_DIM must equal 524 (512 stats + 4 spatial + 5 Doppler + 3 temporal)."""
        expected = 256 * 2 + 4 + 5 + 3
        assert FEATURE_DIM == 524
        assert FEATURE_DIM == expected
        assert _SUBCARRIER_STAT_DIM == 512
        assert _N_SPATIAL_BANDS == 4
        assert _N_DOPPLER_BANDS == 5

    def test_feature_names_length_and_uniqueness(self) -> None:
        """feature_names() must return exactly FEATURE_DIM unique strings."""
        names = feature_names()
        assert isinstance(names, list)
        assert len(names) == FEATURE_DIM
        assert len(set(names)) == FEATURE_DIM

    def test_feature_names_layout_and_prefixes(self) -> None:
        """Verify the exact order and structure of feature names."""
        names = feature_names()

        # 0..255: sub000_mean .. sub255_mean
        for i in range(256):
            assert names[i] == f"sub{i:03d}_mean"

        # 256..511: sub000_std .. sub255_std
        for i in range(256):
            assert names[256 + i] == f"sub{i:03d}_std"

        # 512..515: spatial band energies
        assert names[512] == "band_energy_sub0-63"
        assert names[513] == "band_energy_sub64-127"
        assert names[514] == "band_energy_sub128-191"
        assert names[515] == "band_energy_sub192-255"

        # 516..520: Doppler frequency bands
        expected_doppler = [
            "doppler_static_0-0.5Hz",
            "doppler_veryslow_0.5-1Hz",
            "doppler_slow_1-3Hz",
            "doppler_body_3-8Hz",
            "doppler_fast_8Hz+",
        ]
        assert names[516:521] == expected_doppler

        # 521..523: temporal statistics
        expected_temporal = ["temporal_variance", "temporal_entropy", "temporal_range"]
        assert names[521:524] == expected_temporal

    def test_extract_features_shape_and_dtype(self) -> None:
        """extract_features on (T, 256) returns float32 array of shape (524,)."""
        rng = np.random.default_rng(123)
        win = rng.uniform(5.0, 50.0, size=(150, 256)).astype(np.float32)

        feat = extract_features(win)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (FEATURE_DIM,)
        assert feat.dtype == np.float32
        assert np.isfinite(feat).all()

    def test_extract_features_batch_shape_and_dtype(self) -> None:
        """extract_features_batch on (N, T, 256) returns float32 array of shape (N, 524)."""
        rng = np.random.default_rng(456)
        batch = rng.uniform(1.0, 30.0, size=(5, 100, 256)).astype(np.float32)

        feats = extract_features_batch(batch)
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (5, FEATURE_DIM)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all()

        # Batch output must match single extraction row-by-row
        for i in range(5):
            single = extract_features(batch[i])
            np.testing.assert_allclose(feats[i], single, rtol=1e-6, atol=1e-6)

    def test_varying_window_length_t(self) -> None:
        """Feature extraction succeeds and preserves dimension across varying T."""
        rng = np.random.default_rng(789)
        for t_len in [25, 50, 100, 150, 200]:
            win = rng.uniform(0.0, 10.0, size=(t_len, 256)).astype(np.float32)
            feat = extract_features(win)
            assert feat.shape == (FEATURE_DIM,)
            assert feat.dtype == np.float32
            assert np.isfinite(feat).all()


# ===========================================================================
# 2. Numerical Calculations & Subcarrier Statistics
# ===========================================================================

class TestNumericalCalculations:
    """Tests for exact numerical values in subcarrier, spatial, and temporal features."""

    def test_constant_window_exact_values(self) -> None:
        """For a constant window c, verify subcarrier mean, std, spatial energy, and temporal stats."""
        c = 3.0
        T = 100
        win = np.full((T, 256), c, dtype=np.float32)

        feat = extract_features(win, packet_rate=30.0)

        # 1. Means: all 256 subcarriers should be exactly c
        means = feat[:256]
        np.testing.assert_allclose(means, c, rtol=1e-6)

        # 2. Stds: all 256 subcarriers should be exactly 0.0
        stds = feat[256:512]
        np.testing.assert_allclose(stds, 0.0, atol=1e-7)

        # 3. Spatial bands: mean squared amplitude is c^2 = 9.0 for all 4 bands
        spatial = feat[512:516]
        np.testing.assert_allclose(spatial, c ** 2, rtol=1e-6)

        # 4. Temporal statistics:
        # Each packet L2 norm = sqrt(256 * c^2) = 16 * c = 48.0
        # Power envelope is completely constant across all T packets
        var_idx = 521
        ent_idx = 522
        rng_idx = 523
        assert feat[var_idx] == pytest.approx(0.0, abs=1e-5)
        assert feat[rng_idx] == pytest.approx(0.0, abs=1e-5)

        # Shannon entropy of uniform PMF p_t = 1/T is -sum(1/T * ln(1/T)) = ln(T)
        expected_entropy = math.log(T)
        assert feat[ent_idx] == pytest.approx(expected_entropy, rel=1e-4)

    def test_subcarrier_stats_helper(self) -> None:
        """_subcarrier_stats returns concatenated mean and population std."""
        T = 40
        rng = np.random.default_rng(42)
        win = rng.normal(loc=15.0, scale=4.0, size=(T, 256)).astype(np.float32)

        stats = _subcarrier_stats(win)
        assert stats.shape == (512,)
        assert stats.dtype == np.float32

        expected_mean = win.mean(axis=0)
        expected_std = win.std(axis=0, ddof=0)
        np.testing.assert_allclose(stats[:256], expected_mean, rtol=1e-6)
        np.testing.assert_allclose(stats[256:], expected_std, rtol=1e-6)

    def test_spatial_band_energies_isolation(self) -> None:
        """Setting energy only in one spatial band isolates that band's energy."""
        win = np.zeros((50, 256), dtype=np.float32)
        # Band 0: 0..63
        # Band 1: 64..127
        # Band 2: 128..191
        # Band 3: 192..255
        win[:, 64:128] = 4.0  # only band 1 has non-zero amplitude

        energies = _spatial_band_energies(win, n_bands=4)
        assert energies.shape == (4,)
        assert energies[0] == pytest.approx(0.0, abs=1e-6)
        assert energies[1] == pytest.approx(16.0, rel=1e-5)  # 4^2 = 16
        assert energies[2] == pytest.approx(0.0, abs=1e-6)
        assert energies[3] == pytest.approx(0.0, abs=1e-6)

    def test_extract_features_determinism(self) -> None:
        """Repeated feature extractions on copy or same array yield identical outputs."""
        rng = np.random.default_rng(999)
        win = rng.uniform(0.0, 100.0, size=(150, 256)).astype(np.float32)

        feat1 = extract_features(win, packet_rate=33.33)
        feat2 = extract_features(win.copy(), packet_rate=33.33)
        assert np.array_equal(feat1, feat2)


# ===========================================================================
# 3. STFT Doppler Frequency Bands
# ===========================================================================

class TestSTFTFrequencyBands:
    """Tests for STFT computation, Doppler frequency binning, and sampling rate scaling."""

    def test_doppler_edges_and_band_count(self) -> None:
        """Verify the module-level Doppler band configuration."""
        assert _DOPPLER_EDGES_HZ == (0.0, 0.5, 1.0, 3.0, 8.0, float("inf"))
        assert _N_DOPPLER_BANDS == 5

    def test_dc_signal_concentrates_in_static_band(self) -> None:
        """A pure DC signal (constant in time) must concentrate energy in band 0 (0-0.5 Hz)."""
        T = 150
        packet_rate = 30.0  # Hz
        win = np.full((T, 256), 10.0, dtype=np.float32)

        doppler = _stft_doppler_energies(win, packet_rate=packet_rate, n_fft=32, hop=8)
        assert doppler.shape == (5,)
        # Band 0 (static) is strictly the dominant band (argmax = 0)
        assert int(np.argmax(doppler)) == 0
        assert doppler[0] > 0.0
        # High-frequency Doppler bands (1-3Hz, 3-8Hz, 8Hz+) have negligible energy compared to DC
        assert doppler[2] < 0.02 * doppler[0]
        assert doppler[3] < 0.01 * doppler[0]
        assert doppler[4] < 0.001 * doppler[0]

    def test_pure_tone_frequency_routing(self) -> None:
        """Synthetic sine oscillations at specific Doppler frequencies peak in corresponding bands.

        Bands:
          0: 0.0 - 0.5 Hz
          1: 0.5 - 1.0 Hz
          2: 1.0 - 3.0 Hz
          3: 3.0 - 8.0 Hz
          4: 8.0 - Nyquist
        """
        T = 150

        # Case A: rate = 16.0 Hz -> df = 0.5 Hz, covering bands 1, 2, 3
        rate_a = 16.0
        t_a = np.arange(T) / rate_a
        cases_a = [
            (0.5, 1),  # 0.5 Hz is in [0.5, 1.0) -> band 1
            (1.5, 2),  # 1.5 Hz is in [1.0, 3.0) -> band 2
            (5.0, 3),  # 5.0 Hz is in [3.0, 8.0) -> band 3
        ]
        for freq_hz, expected_band in cases_a:
            sig = (10.0 * np.sin(2.0 * np.pi * freq_hz * t_a)).astype(np.float32)
            win = np.repeat(sig[:, None], 256, axis=1)
            doppler = _stft_doppler_energies(win, packet_rate=rate_a, n_fft=32, hop=8)
            dominant_band = int(np.argmax(doppler))
            assert dominant_band == expected_band, (
                f"For {freq_hz} Hz at {rate_a} Hz rate, expected band {expected_band}, got {dominant_band}"
            )

        # Case B: rate = 32.0 Hz -> Nyquist = 16.0 Hz, 10.0 Hz falls in [8.0, inf) -> band 4
        rate_b = 32.0
        t_b = np.arange(T) / rate_b
        sig_b = (10.0 * np.sin(2.0 * np.pi * 10.0 * t_b)).astype(np.float32)
        win_b = np.repeat(sig_b[:, None], 256, axis=1)
        doppler_b = _stft_doppler_energies(win_b, packet_rate=rate_b, n_fft=32, hop=8)
        assert int(np.argmax(doppler_b)) == 4

    def test_packet_rate_affects_physical_frequency_binning(self) -> None:
        """Varying packet_rate changes physical frequency binning of the STFT."""
        T = 150
        rng = np.random.default_rng(77)
        win = rng.normal(0.0, 1.0, size=(T, 256)).astype(np.float32)

        doppler_30 = _stft_doppler_energies(win, packet_rate=30.0)
        doppler_100 = _stft_doppler_energies(win, packet_rate=100.0)

        assert not np.allclose(doppler_30, doppler_100)

    def test_short_window_stft_clamping(self) -> None:
        """When T < n_fft (e.g. T=16 < 32), n_fft is clamped to T and executes cleanly."""
        T = 16
        win = np.ones((T, 256), dtype=np.float32)

        doppler = _stft_doppler_energies(win, packet_rate=30.0, n_fft=32, hop=8)
        assert doppler.shape == (5,)
        assert np.isfinite(doppler).all()


# ===========================================================================
# 4. Temporal Statistics
# ===========================================================================

class TestTemporalStatistics:
    """Tests for power envelope calculation, variance, entropy, and range."""

    def test_all_zeros_window(self) -> None:
        """All-zero window has zero power sum; entropy, variance, range must be 0 without NaNs."""
        win = np.zeros((100, 256), dtype=np.float32)

        temp = _temporal_features(win)
        assert temp.shape == (3,)
        assert temp[0] == 0.0  # variance
        assert temp[1] == 0.0  # entropy
        assert temp[2] == 0.0  # range
        assert np.isfinite(temp).all()

    def test_single_pulse_power_envelope(self) -> None:
        """Single-pulse impulse has power concentrated in one packet."""
        T = 50
        win = np.zeros((T, 256), dtype=np.float32)
        # Put energy in row 10 only: norm across 256 subcarriers with value 1.0 is sqrt(256) = 16.0
        win[10, :] = 1.0

        temp = _temporal_features(win)
        variance, entropy, range_ = temp

        # Range should be max (16.0) - min (0.0) = 16.0
        assert range_ == pytest.approx(16.0, rel=1e-5)
        # Variance should be > 0
        assert variance > 0.0

        # In a 1-hot distribution p = [0, ..., 1, ..., 0], Shannon entropy is 0.
        # Implementation uses np.clip(power / s, 1e-12, None) so clipping 0 to 1e-12
        # gives a tiny epsilon: -(T-1) * 1e-12 * ln(1e-12) ~ 0
        assert entropy == pytest.approx(0.0, abs=1e-4)

    def test_monotonically_increasing_power_range_and_variance(self) -> None:
        """Power envelope with known min and max yields exact range and positive variance."""
        T = 80
        # Let packet norm increase linearly from 1 to 80
        # Subcarrier values at time t = (t + 1) / sqrt(256) = (t + 1) / 16
        t_indices = np.arange(1, T + 1, dtype=np.float32)[:, None]
        win = np.repeat(t_indices / 16.0, 256, axis=1)

        temp = _temporal_features(win)
        variance, entropy, range_ = temp

        expected_powers = np.arange(1, T + 1, dtype=np.float64)
        expected_range = float(expected_powers.max() - expected_powers.min())
        expected_variance = float(np.var(expected_powers, ddof=0))

        assert range_ == pytest.approx(expected_range, rel=1e-5)
        assert variance == pytest.approx(expected_variance, rel=1e-5)
        assert entropy > 0.0


# ===========================================================================
# 5. Invalid Inputs and Edge Cases
# ===========================================================================

class TestInvalidInputs:
    """Tests verifying input validation and error raising."""

    def test_wrong_dimensions_single_window(self) -> None:
        """extract_features raises ValueError if window is not 2D."""
        with pytest.raises(ValueError, match="Expected window shape \\(T, 256\\)"):
            extract_features(np.zeros(256, dtype=np.float32))

        with pytest.raises(ValueError, match="Expected window shape \\(T, 256\\)"):
            extract_features(np.zeros((10, 150, 256), dtype=np.float32))

    def test_wrong_subcarrier_count_single_window(self) -> None:
        """extract_features raises ValueError if second dimension != 256."""
        with pytest.raises(ValueError, match="Expected window shape \\(T, 256\\)"):
            extract_features(np.zeros((150, 128), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected window shape \\(T, 256\\)"):
            extract_features(np.zeros((150, 512), dtype=np.float32))

    def test_zero_time_steps_raises(self) -> None:
        """extract_features raises ValueError if T == 0."""
        with pytest.raises(ValueError, match="at least one time step"):
            extract_features(np.zeros((0, 256), dtype=np.float32))

    def test_wrong_dimensions_batch(self) -> None:
        """extract_features_batch raises ValueError if windows is not 3D or subcarriers != 256."""
        with pytest.raises(ValueError, match="Expected \\(N, T, 256\\)"):
            extract_features_batch(np.zeros((150, 256), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected \\(N, T, 256\\)"):
            extract_features_batch(np.zeros((2, 10, 150, 256), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected \\(N, T, 256\\)"):
            extract_features_batch(np.zeros((5, 150, 64), dtype=np.float32))

    def test_extreme_numerical_stability(self) -> None:
        """Tiny values (~1e-12) and large values (~1e4) don't produce NaN or Inf."""
        # Tiny positive numbers
        win_tiny = np.full((150, 256), 1e-10, dtype=np.float32)
        feat_tiny = extract_features(win_tiny)
        assert np.isfinite(feat_tiny).all()

        # Large numbers
        win_large = np.full((150, 256), 1e4, dtype=np.float32)
        feat_large = extract_features(win_large)
        assert np.isfinite(feat_large).all()
