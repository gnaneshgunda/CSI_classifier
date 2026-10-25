"""
tests/test_normalization.py
===========================
Unit tests for src/normalization.py (P09).

Test classes
------------
TestRobustScale          – _robust_scale() helper
TestDecompose            – Normalizer._decompose() internal
TestFitLeakageGuard      – fit() raises on val-split rows (critical)
TestFitBasic             – fit() happy path: shapes, dtypes, save/load round-trip
TestFitPerRoom           – per-room stats differ from global when rooms differ
TestFitMissingRoom       – warning + fallback when one room has no dev windows
TestTransformShapes      – transform() output shapes & dtypes
TestTransformOracleMode  – oracle routing uses correct room stats
TestTransformHardMode    – predicted_hard routing uses argmax room
TestTransformSoftMode    – predicted_soft routing produces a mixture
TestTransformNoneMode    – none routing uses global stats
TestTransformNoStats     – RuntimeError before fit/load
TestLog1pToggle          – log1p=False produces different outputs
TestSaveLoad             – round-trip through npz file
TestTransformMissingArgs – ValueError for missing required args
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.normalization import Normalizer, NormStats, _robust_scale

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

N_SUB = 256
T = 150
ROOMS = ("A", "B")


def _make_window(seed: int = 0, t: int = T, n_sub: int = N_SUB) -> np.ndarray:
    """Return a deterministic random (T, 256) float32 window."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 100.0, size=(t, n_sub)).astype(np.float32)


def _make_df(
    n_per_room: int = 10,
    split: str = "dev",
    rooms: tuple = ROOMS,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Build a small windows DataFrame with pre-loaded arrays."""
    rows = []
    for i, room in enumerate(rooms):
        for j in range(n_per_room):
            rows.append(
                {
                    "split": split,
                    "room": room,
                    "array": _make_window(seed=i * 100 + j + seed_offset),
                }
            )
    return pd.DataFrame(rows)


def _make_cfg(routing_mode: str = "oracle", tmp_path: Path = Path("/tmp")):
    """Return a minimal ExperimentConfig-like object."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.routing.mode = routing_mode
    cfg.paths.derived_dir = str(tmp_path)
    return cfg


# ---------------------------------------------------------------------------
# _robust_scale
# ---------------------------------------------------------------------------

class TestRobustScale:
    def test_shape(self) -> None:
        arr = np.random.randn(500, N_SUB).astype(np.float32)
        scale = _robust_scale(arr)
        assert scale.shape == (N_SUB,)

    def test_dtype(self) -> None:
        arr = np.random.randn(500, N_SUB).astype(np.float64)
        scale = _robust_scale(arr)
        assert scale.dtype == np.float32

    def test_positive(self) -> None:
        arr = np.random.randn(500, N_SUB).astype(np.float32)
        scale = _robust_scale(arr)
        assert np.all(scale > 0)

    def test_constant_column_clamped(self) -> None:
        arr = np.zeros((100, N_SUB), dtype=np.float32)
        scale = _robust_scale(arr)
        # Constant column has IQR=0 → clamped to _MIN_SCALE
        assert np.all(scale >= 1e-6)

    def test_known_normal(self) -> None:
        """Standard-normal data → scale ≈ 1.0 (within 10% for 10k samples)."""
        rng = np.random.RandomState(0)
        arr = rng.randn(10_000, 1).astype(np.float32)
        scale = _robust_scale(arr)
        assert abs(float(scale[0]) - 1.0) < 0.10


# ---------------------------------------------------------------------------
# Normalizer._decompose
# ---------------------------------------------------------------------------

class TestDecompose:
    def test_shapes(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        win = _make_window()
        static, dynamic = norm._decompose(win)
        assert static.shape  == (N_SUB,)
        assert dynamic.shape == (T, N_SUB)

    def test_dynamic_zero_mean(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        win = _make_window(seed=3)
        _, dynamic = norm._decompose(win)
        # Column means of residuals must be (approximately) zero
        assert np.allclose(dynamic.mean(axis=0), 0.0, atol=1e-5)

    def test_log1p_false(self, tmp_path) -> None:
        norm_with = Normalizer(_make_cfg(tmp_path=tmp_path), log1p=True)
        norm_without = Normalizer(_make_cfg(tmp_path=tmp_path), log1p=False)
        win = _make_window(seed=7)
        s1, d1 = norm_with._decompose(win)
        s2, d2 = norm_without._decompose(win)
        assert not np.allclose(s1, s2), "log1p=True/False must differ"


# ---------------------------------------------------------------------------
# fit() – leakage guard (most critical test)
# ---------------------------------------------------------------------------

class TestFitLeakageGuard:
    def test_raises_on_val_rows(self, tmp_path) -> None:
        """fit() must raise ValueError if any row is split='val'."""
        df = _make_df()  # all dev
        df_with_val = pd.concat(
            [df, _make_df(n_per_room=2, split="val")], ignore_index=True
        )
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(ValueError, match="validation-split"):
            norm.fit(df_with_val)

    def test_raises_on_all_val(self, tmp_path) -> None:
        """fit() with 100% val rows still raises."""
        df = _make_df(split="val")
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(ValueError, match="validation-split"):
            norm.fit(df)

    def test_raises_on_single_val_row(self, tmp_path) -> None:
        """Even a single val row should trigger the guard."""
        df = _make_df()
        df.loc[0, "split"] = "val"
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(ValueError, match="validation-split"):
            norm.fit(df)

    def test_no_split_column_raises(self, tmp_path) -> None:
        df = _make_df().drop(columns=["split"])
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(ValueError, match="'split' column"):
            norm.fit(df)

    def test_empty_df_raises(self, tmp_path) -> None:
        df = _make_df().iloc[0:0]          # empty, but has 'split' column
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(ValueError, match="empty"):
            norm.fit(df)

    def test_dev_only_does_not_raise(self, tmp_path) -> None:
        """Sanity: fit on dev-only data succeeds."""
        df = _make_df()
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(df)  # must not raise


# ---------------------------------------------------------------------------
# fit() – basic shapes and dtypes
# ---------------------------------------------------------------------------

class TestFitBasic:
    def test_stats_not_none_after_fit(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert norm._stats is not None

    def test_global_mean_shape(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert norm._stats.global_mean.shape == (N_SUB,)

    def test_global_std_shape(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert norm._stats.global_std.shape == (N_SUB,)

    def test_global_scale_shape(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert norm._stats.global_scale.shape == (N_SUB,)

    def test_room_stats_shapes(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        for room in ROOMS:
            assert norm._stats.room_mean[room].shape  == (N_SUB,)
            assert norm._stats.room_std[room].shape   == (N_SUB,)
            assert norm._stats.room_scale[room].shape == (N_SUB,)

    def test_stats_dtype_float32(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        for arr in [
            norm._stats.global_mean,
            norm._stats.global_std,
            norm._stats.global_scale,
        ]:
            assert arr.dtype == np.float32

    def test_npz_file_created(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert (tmp_path / "norm_stats.npz").exists()

    def test_npz_keys(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        data = np.load(str(tmp_path / "norm_stats.npz"))
        expected_keys = {
            "global_mean", "global_std", "global_scale",
            "mean_A", "std_A", "scale_A",
            "mean_B", "std_B", "scale_B",
        }
        assert set(data.files) == expected_keys

    def test_fit_returns_self(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        result = norm.fit(_make_df())
        assert result is norm

    def test_scales_positive(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        assert np.all(norm._stats.global_scale > 0)
        for room in ROOMS:
            assert np.all(norm._stats.room_scale[room] > 0)


# ---------------------------------------------------------------------------
# fit() – per-room statistics differ when rooms have different distributions
# ---------------------------------------------------------------------------

class TestFitPerRoom:
    def test_room_means_differ_from_each_other(self, tmp_path) -> None:
        """Rooms with different signal levels should yield different means."""
        rng = np.random.RandomState(0)
        rows = []
        # Room A: low amplitude
        for i in range(20):
            rows.append({
                "split": "dev", "room": "A",
                "array": rng.uniform(0.1, 10.0, (T, N_SUB)).astype(np.float32),
            })
        # Room B: high amplitude
        for i in range(20):
            rows.append({
                "split": "dev", "room": "B",
                "array": rng.uniform(100.0, 500.0, (T, N_SUB)).astype(np.float32),
            })
        df = pd.DataFrame(rows)
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(df)
        # Room A mean < Room B mean on average
        assert norm._stats.room_mean["A"].mean() < norm._stats.room_mean["B"].mean()

    def test_global_mean_between_room_means(self, tmp_path) -> None:
        rng = np.random.RandomState(1)
        rows = []
        for i in range(20):
            rows.append({
                "split": "dev", "room": "A",
                "array": rng.uniform(0.1, 10.0, (T, N_SUB)).astype(np.float32),
            })
        for i in range(20):
            rows.append({
                "split": "dev", "room": "B",
                "array": rng.uniform(100.0, 500.0, (T, N_SUB)).astype(np.float32),
            })
        df = pd.DataFrame(rows)
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(df)
        ga = norm._stats.room_mean["A"].mean()
        gb = norm._stats.room_mean["B"].mean()
        gg = norm._stats.global_mean.mean()
        assert ga < gg < gb


# ---------------------------------------------------------------------------
# fit() – missing room triggers warning + fallback
# ---------------------------------------------------------------------------

class TestFitMissingRoom:
    def test_missing_room_warns(self, tmp_path) -> None:
        # Only room A data
        rows = [
            {"split": "dev", "room": "A", "array": _make_window(i)}
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.warns(UserWarning, match="No development windows found for room 'B'"):
            norm.fit(df)

    def test_missing_room_uses_global_fallback(self, tmp_path) -> None:
        rows = [
            {"split": "dev", "room": "A", "array": _make_window(i)}
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.warns(UserWarning):
            norm.fit(df)
        # Room B stats should equal global stats (fallback)
        np.testing.assert_array_equal(
            norm._stats.room_mean["B"], norm._stats.global_mean
        )


# ---------------------------------------------------------------------------
# transform() – shapes and dtypes
# ---------------------------------------------------------------------------

class TestTransformShapes:
    def _fitted_norm(self, tmp_path, mode="oracle"):
        norm = Normalizer(_make_cfg(routing_mode=mode, tmp_path=tmp_path))
        norm.fit(_make_df())
        return norm

    def test_static_shape(self, tmp_path) -> None:
        norm = self._fitted_norm(tmp_path)
        win = _make_window()
        static, _ = norm.transform(win, room="A")
        assert static.shape == (N_SUB,)

    def test_dynamic_shape(self, tmp_path) -> None:
        norm = self._fitted_norm(tmp_path)
        win = _make_window()
        _, dynamic = norm.transform(win, room="A")
        assert dynamic.shape == (T, N_SUB)

    def test_static_dtype(self, tmp_path) -> None:
        norm = self._fitted_norm(tmp_path)
        static, _ = norm.transform(_make_window(), room="A")
        assert static.dtype == np.float32

    def test_dynamic_dtype(self, tmp_path) -> None:
        norm = self._fitted_norm(tmp_path)
        _, dynamic = norm.transform(_make_window(), room="A")
        assert dynamic.dtype == np.float32

    def test_dynamic_zero_mean(self, tmp_path) -> None:
        """Dynamic residuals must sum to zero along time axis."""
        norm = self._fitted_norm(tmp_path)
        win = _make_window(seed=42)
        _, dynamic = norm.transform(win, room="A")
        # Column means of the *raw* residuals ≈ 0 before scale normalization
        # After dividing by scale, mean should still be ~0
        assert np.allclose(dynamic.mean(axis=0), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# transform() – oracle routing
# ---------------------------------------------------------------------------

class TestTransformOracleMode:
    def test_oracle_a_uses_room_a_stats(self, tmp_path) -> None:
        """Oracle routing for room A should use room-A mean in z-scoring."""
        rng = np.random.RandomState(0)
        rows_a = [
            {"split": "dev", "room": "A",
             "array": rng.uniform(0.1, 5.0, (T, N_SUB)).astype(np.float32)}
            for _ in range(15)
        ]
        rows_b = [
            {"split": "dev", "room": "B",
             "array": rng.uniform(200.0, 500.0, (T, N_SUB)).astype(np.float32)}
            for _ in range(15)
        ]
        df = pd.DataFrame(rows_a + rows_b)

        norm_a = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm_a.fit(df)

        win = rng.uniform(0.1, 5.0, (T, N_SUB)).astype(np.float32)
        s_a, _ = norm_a.transform(win, room="A")
        s_b, _ = norm_a.transform(win, room="B")
        # Room A stats are more "centred" on this window → smaller |z|
        assert np.abs(s_a).mean() < np.abs(s_b).mean()

    def test_oracle_missing_room_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="'room' argument"):
            norm.transform(_make_window())    # room=None

    def test_oracle_bad_room_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="Unknown room"):
            norm.transform(_make_window(), room="C")


# ---------------------------------------------------------------------------
# transform() – predicted_hard routing
# ---------------------------------------------------------------------------

class TestTransformHardMode:
    def test_hard_picks_argmax_room(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_hard", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window()

        # Confidently predict room A
        probs_a = np.array([0.95, 0.05], dtype=np.float32)
        s_a, _ = norm.transform(win, room_probs=probs_a)

        # Confidently predict room B
        probs_b = np.array([0.05, 0.95], dtype=np.float32)
        s_b, _ = norm.transform(win, room_probs=probs_b)

        # Results differ because different room stats are used
        assert not np.allclose(s_a, s_b)

    def test_hard_selects_a_on_50_50_tie(self, tmp_path) -> None:
        """50/50 tie → argmax picks index 0 → room A."""
        norm = Normalizer(_make_cfg(routing_mode="predicted_hard", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window()
        probs = np.array([0.5, 0.5], dtype=np.float32)
        s_hard, _ = norm.transform(win, room_probs=probs)
        # Must equal oracle result for room A
        norm2 = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm2._stats = norm._stats
        s_oracle, _ = norm2.transform(win, room="A")
        np.testing.assert_array_equal(s_hard, s_oracle)

    def test_hard_no_probs_fallback_to_oracle(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_hard", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window()
        # room_probs=None but room supplied → falls back to oracle
        s_hard, _ = norm.transform(win, room="B")
        norm2 = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm2._stats = norm._stats
        s_oracle, _ = norm2.transform(win, room="B")
        np.testing.assert_array_equal(s_hard, s_oracle)

    def test_hard_no_probs_no_room_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_hard", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="room_probs"):
            norm.transform(_make_window())

    def test_hard_wrong_probs_shape_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_hard", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="2 elements"):
            norm.transform(_make_window(), room_probs=np.array([0.3, 0.3, 0.4]))


# ---------------------------------------------------------------------------
# transform() – predicted_soft routing
# ---------------------------------------------------------------------------

class TestTransformSoftMode:
    def test_soft_mixes_stats(self, tmp_path) -> None:
        """Soft mixture with equal weights must produce result between room-A and room-B."""
        rng = np.random.RandomState(0)
        rows_a = [
            {"split": "dev", "room": "A",
             "array": rng.uniform(0.1, 5.0, (T, N_SUB)).astype(np.float32)}
            for _ in range(15)
        ]
        rows_b = [
            {"split": "dev", "room": "B",
             "array": rng.uniform(200.0, 500.0, (T, N_SUB)).astype(np.float32)}
            for _ in range(15)
        ]
        df = pd.DataFrame(rows_a + rows_b)
        norm = Normalizer(_make_cfg(routing_mode="predicted_soft", tmp_path=tmp_path))
        norm.fit(df)

        win = _make_window(seed=99)
        # Pure A vs pure B
        norm_a = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm_a._stats = norm._stats
        s_a, _ = norm_a.transform(win, room="A")

        norm_b = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm_b._stats = norm._stats
        s_b, _ = norm_b.transform(win, room="B")

        # 50/50 soft
        s_soft, _ = norm.transform(win, room_probs=np.array([0.5, 0.5]))

        # Soft must differ from both pure rooms
        assert not np.allclose(s_soft, s_a)
        assert not np.allclose(s_soft, s_b)

    def test_soft_no_probs_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_soft", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="room_probs"):
            norm.transform(_make_window())

    def test_soft_pure_a_equals_oracle_a(self, tmp_path) -> None:
        """Soft with probs=[1,0] must equal oracle room A."""
        norm = Normalizer(_make_cfg(routing_mode="predicted_soft", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window()
        s_soft, d_soft = norm.transform(win, room_probs=np.array([1.0, 0.0]))

        norm_o = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm_o._stats = norm._stats
        s_oracle, d_oracle = norm_o.transform(win, room="A")
        np.testing.assert_allclose(s_soft, s_oracle, rtol=1e-5)


# ---------------------------------------------------------------------------
# transform() – none routing (global stats)
# ---------------------------------------------------------------------------

class TestTransformNoneMode:
    def test_none_uses_global_stats(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="none", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window()
        s, d = norm.transform(win)   # room=None, room_probs=None -> global
        assert s.shape == (N_SUB,)
        assert d.shape == (T, N_SUB)

    def test_none_ignores_room_arg(self, tmp_path) -> None:
        """room arg is ignored in 'none' mode."""
        norm = Normalizer(_make_cfg(routing_mode="none", tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window(seed=11)
        s1, d1 = norm.transform(win, room="A")
        s2, d2 = norm.transform(win, room="B")
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# transform() – RuntimeError before fit/load
# ---------------------------------------------------------------------------

class TestTransformNoStats:
    def test_raises_before_fit(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(RuntimeError, match="fit\\(\\) or load\\(\\)"):
            norm.transform(_make_window(), room="A")


# ---------------------------------------------------------------------------
# log1p toggle
# ---------------------------------------------------------------------------

class TestLog1pToggle:
    def test_log1p_true_false_differ(self, tmp_path) -> None:
        df = _make_df()
        norm_on  = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path / "on"),  log1p=True)
        norm_off = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path / "off"), log1p=False)
        norm_on.fit(df)
        norm_off.fit(df)
        win = _make_window(seed=5)
        s_on,  _ = norm_on.transform(win,  room="A")
        s_off, _ = norm_off.transform(win, room="A")
        assert not np.allclose(s_on, s_off), "log1p toggle must affect output"

    def test_log1p_consistent_within_run(self, tmp_path) -> None:
        """Same window twice must give identical output."""
        df = _make_df()
        norm = Normalizer(_make_cfg(routing_mode="oracle", tmp_path=tmp_path))
        norm.fit(df)
        win = _make_window(seed=17)
        s1, d1 = norm.transform(win.copy(), room="B")
        s2, d2 = norm.transform(win.copy(), room="B")
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_load_restores_global_mean(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        mean_before = norm._stats.global_mean.copy()

        norm2 = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm2.load()
        np.testing.assert_array_equal(norm2._stats.global_mean, mean_before)

    def test_load_restores_room_stats(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())

        norm2 = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm2.load()
        for room in ROOMS:
            np.testing.assert_array_equal(
                norm2._stats.room_mean[room], norm._stats.room_mean[room]
            )

    def test_load_before_fit_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        with pytest.raises(FileNotFoundError, match="norm_stats.npz"):
            norm.load()

    def test_transform_after_load_matches_fit(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm.fit(_make_df())
        win = _make_window(seed=99)
        s1, d1 = norm.transform(win, room="A")

        norm2 = Normalizer(_make_cfg(tmp_path=tmp_path))
        norm2.load()
        s2, d2 = norm2.transform(win, room="A")
        np.testing.assert_allclose(s1, s2, rtol=1e-6)
        np.testing.assert_allclose(d1, d2, rtol=1e-6)


# ---------------------------------------------------------------------------
# Missing-arg edge cases
# ---------------------------------------------------------------------------

class TestTransformMissingArgs:
    def test_soft_wrong_probs_shape_raises(self, tmp_path) -> None:
        norm = Normalizer(_make_cfg(routing_mode="predicted_soft", tmp_path=tmp_path))
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="2 elements"):
            norm.transform(_make_window(), room_probs=np.array([0.5]))

    def test_unknown_routing_mode_raises(self, tmp_path) -> None:
        cfg = _make_cfg(routing_mode="unicorn", tmp_path=tmp_path)
        # Bypass config validation for this test
        norm = Normalizer(cfg)
        norm.fit(_make_df())
        with pytest.raises(ValueError, match="Unknown routing mode"):
            norm.transform(_make_window(), room="A")
