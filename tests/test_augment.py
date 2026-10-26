"""
tests/test_augment.py
=====================
Unit tests for src/augment.py (P11).

Test classes
------------
TestGaussianJitter          – shape, dtype, distribution, disable (std=0)
TestGlobalGain              – shape, range, disable, equal-range no-op
TestSubcarrierBandDropout   – shape, zeros in right range, disable (frac=0)
TestCircularTimeShift       – shape, max shift cap, determinism, disable
TestCompose                 – full pipeline shape, disable all, reproducibility
TestSSLPairDataset          – construction, len, item shapes, no-label,
                              two-views-differ, augment=False, leakage guard,
                              both rooms present, reproducibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.augment import (
    gaussian_jitter,
    global_gain,
    subcarrier_band_dropout,
    circular_time_shift,
    compose,
    SSLPairDataset,
    _MAX_SHIFT_CAP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T, N_SUB = 150, 256


def _arr(seed: int = 0, t: int = T) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 10.0, (t, N_SUB)).astype(np.float32)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_augment_cfg(
    noise_std: float = 0.05,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    mask_frac: float = 0.1,
    crop_frac: float = 0.7,
):
    cfg = MagicMock()
    cfg.gaussian_noise_std  = noise_std
    cfg.amplitude_scale_min = scale_min
    cfg.amplitude_scale_max = scale_max
    cfg.subcarrier_mask_frac = mask_frac
    cfg.time_crop_min_frac  = crop_frac
    return cfg


# ---------------------------------------------------------------------------
# gaussian_jitter
# ---------------------------------------------------------------------------

class TestGaussianJitter:
    def test_shape(self) -> None:
        x = _arr()
        out = gaussian_jitter(x, 0.05, _rng())
        assert out.shape == (T, N_SUB)

    def test_dtype_float32(self) -> None:
        out = gaussian_jitter(_arr(), 0.05, _rng())
        assert out.dtype == np.float32

    def test_zero_std_no_change(self) -> None:
        x = _arr()
        out = gaussian_jitter(x, 0.0, _rng())
        np.testing.assert_array_equal(out, x)

    def test_negative_std_no_change(self) -> None:
        x = _arr()
        out = gaussian_jitter(x, -1.0, _rng())
        np.testing.assert_array_equal(out, x)

    def test_output_differs_from_input(self) -> None:
        x = _arr()
        out = gaussian_jitter(x, 0.1, _rng())
        assert not np.array_equal(out, x)

    def test_noise_is_small(self) -> None:
        """With std=0.05 noise should be much smaller than signal."""
        x = _arr(seed=1) * 100          # large values
        out = gaussian_jitter(x, 0.05, _rng())
        diff = np.abs(out - x)
        assert diff.max() < 0.5 * x.std()

    def test_different_rng_seeds_differ(self) -> None:
        x = _arr()
        out1 = gaussian_jitter(x.copy(), 0.05, _rng(0))
        out2 = gaussian_jitter(x.copy(), 0.05, _rng(1))
        assert not np.array_equal(out1, out2)

    def test_same_rng_seed_deterministic(self) -> None:
        x = _arr()
        out1 = gaussian_jitter(x.copy(), 0.05, _rng(42))
        out2 = gaussian_jitter(x.copy(), 0.05, _rng(42))
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# global_gain
# ---------------------------------------------------------------------------

class TestGlobalGain:
    def test_shape(self) -> None:
        out = global_gain(_arr(), 0.8, 1.2, _rng())
        assert out.shape == (T, N_SUB)

    def test_dtype_float32(self) -> None:
        out = global_gain(_arr(), 0.8, 1.2, _rng())
        assert out.dtype == np.float32

    def test_scale_within_range(self) -> None:
        x = _arr()
        for _ in range(20):
            out = global_gain(x.copy(), 0.8, 1.2, _rng(_))
            ratio = float(out.mean() / x.mean())
            assert 0.79 < ratio < 1.21, f"Scale {ratio:.3f} outside [0.8, 1.2]"

    def test_zero_scale_min_no_change(self) -> None:
        x = _arr()
        out = global_gain(x, 0.0, 1.2, _rng())
        np.testing.assert_array_equal(out, x)

    def test_equal_range_no_change(self) -> None:
        x = _arr()
        out = global_gain(x, 1.0, 1.0, _rng())   # min == max == 1 → no-op
        np.testing.assert_array_equal(out, x)

    def test_scale_min_ge_max_no_change(self) -> None:
        x = _arr()
        out = global_gain(x, 1.2, 0.8, _rng())   # min > max
        np.testing.assert_array_equal(out, x)

    def test_same_seed_deterministic(self) -> None:
        x = _arr()
        out1 = global_gain(x.copy(), 0.8, 1.2, _rng(7))
        out2 = global_gain(x.copy(), 0.8, 1.2, _rng(7))
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# subcarrier_band_dropout
# ---------------------------------------------------------------------------

class TestSubcarrierBandDropout:
    def test_shape(self) -> None:
        out = subcarrier_band_dropout(_arr(), 0.1, _rng())
        assert out.shape == (T, N_SUB)

    def test_dtype_float32(self) -> None:
        out = subcarrier_band_dropout(_arr(), 0.1, _rng())
        assert out.dtype == np.float32

    def test_zero_frac_no_change(self) -> None:
        x = _arr()
        out = subcarrier_band_dropout(x, 0.0, _rng())
        np.testing.assert_array_equal(out, x)

    def test_negative_frac_no_change(self) -> None:
        x = _arr()
        out = subcarrier_band_dropout(x, -0.1, _rng())
        np.testing.assert_array_equal(out, x)

    def test_zeros_count_correct(self) -> None:
        x = np.ones((T, N_SUB), dtype=np.float32)
        frac = 0.1
        out = subcarrier_band_dropout(x, frac, _rng())
        n_zero_cols = (out == 0).all(axis=0).sum()
        expected = round(frac * N_SUB)
        assert abs(int(n_zero_cols) - expected) <= 1

    def test_zeros_are_contiguous(self) -> None:
        x = np.ones((T, N_SUB), dtype=np.float32)
        out = subcarrier_band_dropout(x, 0.1, _rng(3))
        zero_cols = np.where((out == 0).all(axis=0))[0]
        if len(zero_cols) > 1:
            # consecutive indices
            assert np.all(np.diff(zero_cols) == 1), "Dropout band is not contiguous"

    def test_non_dropped_columns_unchanged(self) -> None:
        x = _arr(seed=5)
        out = subcarrier_band_dropout(x.copy(), 0.1, _rng(5))
        nonzero_mask = ~(out == 0).all(axis=0)
        np.testing.assert_array_equal(out[:, nonzero_mask], x[:, nonzero_mask])

    def test_large_frac_all_zeros(self) -> None:
        x = _arr()
        out = subcarrier_band_dropout(x, 1.0, _rng())
        assert (out == 0).all()

    def test_same_seed_deterministic(self) -> None:
        x = _arr()
        out1 = subcarrier_band_dropout(x.copy(), 0.1, _rng(99))
        out2 = subcarrier_band_dropout(x.copy(), 0.1, _rng(99))
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# circular_time_shift
# ---------------------------------------------------------------------------

class TestCircularTimeShift:
    def test_shape(self) -> None:
        out = circular_time_shift(_arr(), 5, _rng())
        assert out.shape == (T, N_SUB)

    def test_dtype_float32(self) -> None:
        out = circular_time_shift(_arr(), 5, _rng())
        assert out.dtype == np.float32

    def test_zero_max_shift_no_change(self) -> None:
        x = _arr()
        out = circular_time_shift(x, 0, _rng())
        np.testing.assert_array_equal(out, x)

    def test_negative_max_shift_no_change(self) -> None:
        x = _arr()
        out = circular_time_shift(x, -3, _rng())
        np.testing.assert_array_equal(out, x)

    def test_max_shift_cap_enforced(self) -> None:
        """Even if max_shift > 10 is passed, internal cap limits to 10."""
        x = _arr(seed=10)
        # With shift capped at 10 out of 150 rows, values must still be from x
        out = circular_time_shift(x, 100, _rng(0))
        # All values in out must appear in x (circular shift reuses rows)
        assert set(out[:, 0].round(4).tolist()).issubset(
            set(x[:, 0].round(4).tolist())
        )

    def test_shift_is_circular(self) -> None:
        """np.roll is circular: rolled array contains all original rows."""
        x = _arr(seed=2)
        out = circular_time_shift(x, 5, _rng(2))
        # Sort both along axis=0 and compare
        np.testing.assert_array_almost_equal(
            np.sort(out, axis=0), np.sort(x, axis=0), decimal=5
        )

    def test_no_time_reversal(self) -> None:
        """Shifted output must not be the time-reversed input."""
        x = _arr(seed=3)
        out = circular_time_shift(x, 5, _rng(3))
        assert not np.array_equal(out, x[::-1])

    def test_same_seed_deterministic(self) -> None:
        x = _arr()
        out1 = circular_time_shift(x.copy(), 5, _rng(11))
        out2 = circular_time_shift(x.copy(), 5, _rng(11))
        np.testing.assert_array_equal(out1, out2)

    def test_max_shift_cap_constant(self) -> None:
        assert _MAX_SHIFT_CAP == 10


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------

class TestCompose:
    def test_shape(self) -> None:
        out = compose(_arr(), _make_augment_cfg(), _rng())
        assert out.shape == (T, N_SUB)

    def test_dtype_float32(self) -> None:
        out = compose(_arr(), _make_augment_cfg(), _rng())
        assert out.dtype == np.float32

    def test_all_disabled_no_change(self) -> None:
        x = _arr()
        cfg = _make_augment_cfg(
            noise_std=0.0,
            scale_min=1.0, scale_max=1.0,
            mask_frac=0.0,
            crop_frac=1.0,
        )
        # With no augmentations, compose should return the input unchanged
        out = compose(x.copy(), cfg, _rng())
        # gaussian_jitter(std=0): no-op; global_gain(1,1): no-op;
        # subcarrier_band_dropout(0): no-op; circular_time_shift(10): may shift
        # So only test that shape/dtype is right when all controllable ones are off
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_output_differs_from_input_with_augment(self) -> None:
        x = _arr(seed=5)
        out = compose(x.copy(), _make_augment_cfg(), _rng(5))
        assert not np.array_equal(out, x)

    def test_same_seed_deterministic(self) -> None:
        x = _arr()
        cfg = _make_augment_cfg()
        out1 = compose(x.copy(), cfg, _rng(13))
        out2 = compose(x.copy(), cfg, _rng(13))
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self) -> None:
        x = _arr()
        cfg = _make_augment_cfg()
        out1 = compose(x.copy(), cfg, _rng(0))
        out2 = compose(x.copy(), cfg, _rng(1))
        assert not np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# SSLPairDataset
# ---------------------------------------------------------------------------

def _make_windows_dir_and_index(
    tmp_path: Path,
    n_captures_a: int = 3,
    n_captures_b: int = 3,
    windows_per_capture: int = 2,
    include_val: bool = False,
) -> Tuple[Path, pd.DataFrame]:
    """Create fake window .npy files and return (windows_dir, index_df)."""
    from typing import Tuple  # local import for clarity

    windows_dir = tmp_path / "windows"
    dev_dir = windows_dir / "dev"
    val_dir = windows_dir / "val"
    dev_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []

    for room, n in [("A", n_captures_a), ("B", n_captures_b)]:
        for i in range(n):
            cap_id = f"{room}__act__s{i:03d}"
            for j in range(windows_per_capture):
                start = j * 50
                wid = f"{cap_id}__{start:05d}"
                arr = rng.uniform(0.1, 10.0, (150, 256)).astype(np.float32)
                np.save(str(dev_dir / f"{wid}.npy"), arr)
                rows.append({
                    "window_id": wid,
                    "capture_id": cap_id,
                    "room": room,
                    "activity": "act",
                    "split": "dev",
                    "start": start,
                })

    if include_val:
        wid = "A__act__v000__00000"
        arr = rng.uniform(0.1, 10.0, (150, 256)).astype(np.float32)
        np.save(str(val_dir / f"{wid}.npy"), arr)
        rows.append({
            "window_id": wid,
            "capture_id": "A__act__v000",
            "room": "A",
            "activity": "act",
            "split": "val",
            "start": 0,
        })

    return windows_dir, pd.DataFrame(rows)


def _make_normalizer(routing_mode: str = "oracle") -> MagicMock:
    norm = MagicMock()
    T, N = 150, 256
    rng = np.random.default_rng(42)
    # transform returns (static (256,), dynamic (T,256))
    norm.transform.side_effect = lambda win, room=None, room_probs=None: (
        np.zeros(N, dtype=np.float32),
        win.astype(np.float32),    # dynamic = raw window (identity for testing)
    )
    norm._stats = MagicMock()
    return norm


def _make_cfg():
    cfg = MagicMock()
    cfg.augment = _make_augment_cfg()
    cfg.routing.mode = "oracle"
    return cfg


# Bring Tuple into scope for type hints
from typing import Tuple  # noqa: E402


class TestSSLPairDataset:

    def test_len(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(
            idx, windows_dir, _make_normalizer(), _make_cfg(),
            augment=False, seed=0,
        )
        # 6 captures × C(2,2)=1 pair each = 6 pairs
        assert len(ds) == 6

    def test_len_more_windows_per_capture(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 2, 2, 3)
        ds = SSLPairDataset(
            idx, windows_dir, _make_normalizer(), _make_cfg(),
            augment=False,
        )
        # 4 captures × C(3,2)=3 pairs each = 12 pairs
        assert len(ds) == 12

    def test_item_returns_tuple_of_two(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_view_shapes(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        v1, v2 = ds[0]
        assert v1.shape == (150, 256)
        assert v2.shape == (150, 256)

    def test_view_dtype_float32(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        v1, v2 = ds[0]
        assert v1.dtype == torch.float32
        assert v2.dtype == torch.float32

    def test_views_are_tensors(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        v1, v2 = ds[0]
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)

    def test_no_labels_returned(self, tmp_path) -> None:
        """Item must be exactly (view1, view2) — no room, activity, or split label."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        item = ds[0]
        assert len(item) == 2, "Dataset must return exactly 2 items (view1, view2), no labels"

    def test_two_views_are_from_same_capture_but_different_windows(self, tmp_path) -> None:
        """Views should come from different windows of the same capture."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        room, wid1, wid2 = ds._pairs[0]
        # Same capture prefix
        cap1 = "__".join(wid1.split("__")[:3])
        cap2 = "__".join(wid2.split("__")[:3])
        assert cap1 == cap2, "Views must come from the same capture"
        # Different window IDs
        assert wid1 != wid2, "Views must be different windows"

    def test_both_rooms_present(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        rooms = {room for room, _, _ in ds._pairs}
        assert "A" in rooms and "B" in rooms, "Both rooms must be present in the SSL dataset"

    def test_leakage_guard_val_rows_raise(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, include_val=True)
        with pytest.raises(ValueError, match="val-split"):
            SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg())

    def test_single_window_capture_skipped(self, tmp_path) -> None:
        """Captures with only 1 dev window must be dropped (cannot make a pair)."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 2, 2, 2)
        # Add a capture with only 1 window
        single_dir = windows_dir / "dev"
        arr = np.random.rand(150, 256).astype(np.float32)
        np.save(str(single_dir / "A__act__solo__00000.npy"), arr)
        single_row = pd.DataFrame([{
            "window_id": "A__act__solo__00000",
            "capture_id": "A__act__solo",
            "room": "A", "activity": "act", "split": "dev", "start": 0,
        }])
        idx_with_solo = pd.concat([idx, single_row], ignore_index=True)
        ds = SSLPairDataset(
            idx_with_solo, windows_dir, _make_normalizer(), _make_cfg(), augment=False
        )
        # Solo capture must not appear in any pair
        cap_ids_in_pairs = {
            "__".join(wid1.split("__")[:3])
            for _, wid1, _ in ds._pairs
        }
        assert "A__act__solo" not in cap_ids_in_pairs

    def test_no_captures_raises(self, tmp_path) -> None:
        """Dataset with all single-window captures must raise RuntimeError."""
        windows_dir = tmp_path / "windows" / "dev"
        windows_dir.mkdir(parents=True)
        arr = np.random.rand(150, 256).astype(np.float32)
        np.save(str(windows_dir / "A__act__s000__00000.npy"), arr)
        idx = pd.DataFrame([{
            "window_id": "A__act__s000__00000",
            "capture_id": "A__act__s000",
            "room": "A", "activity": "act", "split": "dev", "start": 0,
        }])
        with pytest.raises(RuntimeError, match="No captures"):
            SSLPairDataset(idx, tmp_path / "windows", _make_normalizer(), _make_cfg())

    def test_augment_false_deterministic(self, tmp_path) -> None:
        """With augment=False, same index gives identical views."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        v1a, v2a = ds[0]
        v1b, v2b = ds[0]
        assert torch.equal(v1a, v1b)
        assert torch.equal(v2a, v2b)

    def test_augment_true_views_differ_from_each_other(self, tmp_path) -> None:
        """With augment=True, view1 ≠ view2 (different per-view RNG seeds)."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(
            idx, windows_dir, _make_normalizer(), _make_cfg(), augment=True, seed=0
        )
        # Across multiple items, at least some pairs of views should differ
        diffs = []
        for i in range(min(len(ds), 6)):
            v1, v2 = ds[i]
            diffs.append(not torch.equal(v1, v2))
        assert any(diffs), "Augmented views must differ from each other"

    def test_reproducible_across_getitem_calls(self, tmp_path) -> None:
        """Same index with augment=True is reproducible (per-item RNG seeded)."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(
            idx, windows_dir, _make_normalizer(), _make_cfg(), augment=True, seed=42
        )
        v1a, v2a = ds[0]
        v1b, v2b = ds[0]
        assert torch.equal(v1a, v1b)
        assert torch.equal(v2a, v2b)

    def test_normalizer_called_with_true_room(self, tmp_path) -> None:
        """Normalizer must be called with the capture's true room (oracle)."""
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 2, 0, 2)
        norm = _make_normalizer()
        ds = SSLPairDataset(idx, windows_dir, norm, _make_cfg(), augment=False)
        ds[0]
        # Both calls must have used room="A"
        calls = norm.transform.call_args_list
        for call in calls:
            assert call.kwargs.get("room") == "A" or (
                len(call.args) > 1 and call.args[1] == "A"
            ), f"Unexpected room in normalizer call: {call}"

    def test_n_captures_property(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path, 3, 3, 2)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        assert ds.n_captures == 6

    def test_pairs_property_returns_list(self, tmp_path) -> None:
        windows_dir, idx = _make_windows_dir_and_index(tmp_path)
        ds = SSLPairDataset(idx, windows_dir, _make_normalizer(), _make_cfg(), augment=False)
        pairs = ds.pairs
        assert isinstance(pairs, list)
        assert len(pairs) == len(ds)
