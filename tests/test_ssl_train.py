"""
tests/test_ssl_train.py
=======================
Unit tests for src/ssl_train.py (P12).

Test classes
------------
TestVICRegLoss         – correct term shapes/values, weight scaling, collapse detection
TestEmbeddingDiag      – std stats and effective rank computation
TestMakeBlockMask      – mask shape, ratio, block structure
TestProjector          – shapes, dtype, no NaN
TestMaskedDecoder      – shapes, dtype, gradient flow
TestVICRegTrain        – full 2-epoch smoke (CPU), encoder-only final save, CSV log
TestMaskedTrain        – full 2-epoch smoke (CPU), encoder-only final save
TestCLISmoke           – python -m src.ssl_train --smoke exits 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make sure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ssl_train import (
    _Projector,
    _MaskedDecoder,
    vicreg_loss,
    make_block_mask,
    embedding_diagnostics,
    _train_vicreg,
    _train_masked,
)
from src.encoder import ResNet1DEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, T, C = 16, 150, 256
EMB = 128
PROJ = 64


def _rand_z(b: int = B, d: int = EMB) -> torch.Tensor:
    return torch.randn(b, d)


def _smoke_loader(b: int = B, n: int = 32, t: int = T) -> DataLoader:
    v = torch.randn(n, t, C)
    ds = TensorDataset(v, v.clone())
    return DataLoader(ds, batch_size=b, drop_last=True)


def _smoke_cfg(emb: int = EMB, proj: int = PROJ, epochs: int = 2):
    cfg = MagicMock()
    cfg.ssl.embedding_dim  = emb
    cfg.ssl.projection_dim = proj
    cfg.ssl.epochs         = epochs
    cfg.ssl.batch_size     = B
    cfg.ssl.lr             = 1e-3
    cfg.ssl.weight_decay   = 1e-4
    cfg.ssl.lr_min         = 1e-5
    return cfg


def _vicreg_args(run_name="test_vicreg", ckpt_every=100) -> argparse.Namespace:
    return argparse.Namespace(
        run_name=run_name, mode="vicreg",
        inv_weight=25., var_weight=25., cov_weight=1.,
        mask_ratio=0.5, ckpt_every=ckpt_every,
    )


def _masked_args(run_name="test_masked", ckpt_every=100) -> argparse.Namespace:
    return argparse.Namespace(
        run_name=run_name, mode="masked",
        inv_weight=25., var_weight=25., cov_weight=1.,
        mask_ratio=0.5, ckpt_every=ckpt_every,
    )


def _enc(emb: int = EMB) -> ResNet1DEncoder:
    return ResNet1DEncoder(embedding_dim=emb, verbose=False)


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

class TestVICRegLoss:
    def test_returns_tensor_and_dict(self) -> None:
        loss, terms = vicreg_loss(_rand_z(), _rand_z())
        assert isinstance(loss, torch.Tensor)
        assert isinstance(terms, dict)

    def test_dict_has_required_keys(self) -> None:
        _, terms = vicreg_loss(_rand_z(), _rand_z())
        assert {"inv", "var", "cov", "total"} == set(terms.keys())

    def test_loss_is_positive(self) -> None:
        loss, _ = vicreg_loss(_rand_z(), _rand_z())
        assert float(loss) >= 0

    def test_loss_is_scalar(self) -> None:
        loss, _ = vicreg_loss(_rand_z(), _rand_z())
        assert loss.shape == ()

    def test_total_matches_weighted_sum(self) -> None:
        z1, z2 = _rand_z(), _rand_z()
        lw, vw, cw = 25., 25., 1.
        loss, terms = vicreg_loss(z1, z2, inv_weight=lw, var_weight=vw, cov_weight=cw)
        expected = lw * terms["inv"] + vw * terms["var"] + cw * terms["cov"]
        assert abs(float(loss) - expected) < 1e-4

    def test_identical_views_zero_invariance(self) -> None:
        z = _rand_z()
        _, terms = vicreg_loss(z, z.clone())
        assert terms["inv"] < 1e-10

    def test_variance_term_nonneg(self) -> None:
        _, terms = vicreg_loss(_rand_z(), _rand_z())
        assert terms["var"] >= 0

    def test_covariance_term_nonneg(self) -> None:
        _, terms = vicreg_loss(_rand_z(), _rand_z())
        assert terms["cov"] >= 0

    def test_collapsed_embeddings_high_variance_loss(self) -> None:
        """Collapsed (constant) embeddings should have high variance loss."""
        z_const = torch.ones(B, EMB)
        _, terms = vicreg_loss(z_const, z_const)
        assert terms["var"] > 0.9, "Variance loss should be high for collapsed embeddings"

    def test_weight_scaling(self) -> None:
        z1, z2 = _rand_z(), _rand_z()
        _, t1 = vicreg_loss(z1, z2, inv_weight=1., var_weight=1., cov_weight=1.)
        _, t2 = vicreg_loss(z1, z2, inv_weight=50., var_weight=1., cov_weight=1.)
        assert t2["total"] > t1["total"] or abs(t1["inv"]) < 1e-8

    def test_gradient_flows(self) -> None:
        z1 = _rand_z().requires_grad_(True)
        z2 = _rand_z().requires_grad_(True)
        loss, _ = vicreg_loss(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None


# ---------------------------------------------------------------------------
# Embedding diagnostics
# ---------------------------------------------------------------------------

class TestEmbeddingDiag:
    def test_keys(self) -> None:
        d = embedding_diagnostics(_rand_z())
        assert {"emb_std_mean", "emb_std_min", "eff_rank"} == set(d.keys())

    def test_std_mean_positive(self) -> None:
        d = embedding_diagnostics(_rand_z())
        assert d["emb_std_mean"] > 0

    def test_std_min_le_mean(self) -> None:
        d = embedding_diagnostics(_rand_z())
        assert d["emb_std_min"] <= d["emb_std_mean"] + 1e-6

    def test_eff_rank_at_least_1(self) -> None:
        d = embedding_diagnostics(_rand_z())
        assert d["eff_rank"] >= 1.0

    def test_eff_rank_at_most_dim(self) -> None:
        d = embedding_diagnostics(_rand_z())
        assert d["eff_rank"] <= EMB + 1.0

    def test_constant_embedding_low_std(self) -> None:
        z = torch.ones(B, EMB)
        d = embedding_diagnostics(z)
        assert d["emb_std_mean"] < 0.02

    def test_no_nan_or_inf(self) -> None:
        d = embedding_diagnostics(_rand_z())
        for v in d.values():
            assert not (v != v or abs(v) == float("inf"))


# ---------------------------------------------------------------------------
# make_block_mask
# ---------------------------------------------------------------------------

class TestMakeBlockMask:
    def test_shape(self) -> None:
        m = make_block_mask(B, T, C, 0.5, torch.device("cpu"))
        assert m.shape == (B, T, C)

    def test_dtype_bool(self) -> None:
        m = make_block_mask(B, T, C, 0.5, torch.device("cpu"))
        assert m.dtype == torch.bool

    def test_mask_ratio_approximate(self) -> None:
        m = make_block_mask(4, T, C, 0.5, torch.device("cpu"))
        ratio = float(m.float().mean())
        assert 0.25 <= ratio <= 0.75, f"Mask ratio {ratio:.2f} far from 0.5"

    def test_zero_ratio_no_mask(self) -> None:
        m = make_block_mask(B, T, C, 0.0, torch.device("cpu"))
        assert not m.any()

    def test_full_ratio_all_masked(self) -> None:
        m = make_block_mask(B, T, C, 1.0, torch.device("cpu"))
        assert m.all()

    def test_independent_per_batch(self) -> None:
        m = make_block_mask(8, T, C, 0.5, torch.device("cpu"))
        # Different batch items should (with high prob) have different masks
        diffs = [(m[i] != m[i + 1]).any() for i in range(7)]
        assert any(diffs)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class TestProjector:
    def test_output_shape(self) -> None:
        proj = _Projector(EMB, 512, PROJ)
        out = proj(_rand_z())
        assert out.shape == (B, PROJ)

    def test_output_dtype(self) -> None:
        proj = _Projector(EMB, 512, PROJ)
        out = proj(_rand_z())
        assert out.dtype == torch.float32

    def test_no_nan(self) -> None:
        proj = _Projector(EMB, 512, PROJ)
        out = proj(_rand_z())
        assert not torch.isnan(out).any()

    def test_gradient_flows(self) -> None:
        proj = _Projector(EMB, 512, PROJ)
        z = _rand_z().requires_grad_(True)
        proj(z).sum().backward()
        assert z.grad is not None

    def test_batch_size_1_raises_batchnorm(self) -> None:
        """BatchNorm with B=1 raises in train mode — expected behaviour."""
        proj = _Projector(EMB, 512, PROJ)
        proj.train()
        with pytest.raises(ValueError):
            proj(torch.randn(1, EMB))

    def test_eval_mode_batch_size_1_ok(self) -> None:
        proj = _Projector(EMB, 512, PROJ)
        proj.eval()
        out = proj(torch.randn(1, EMB))
        assert out.shape == (1, PROJ)


# ---------------------------------------------------------------------------
# Masked decoder
# ---------------------------------------------------------------------------

class TestMaskedDecoder:
    def test_output_shape(self) -> None:
        dec = _MaskedDecoder(EMB, out_channels=C, t_out=T)
        z = _rand_z()
        out = dec(z)
        assert out.shape == (B, T, C)

    def test_output_dtype(self) -> None:
        dec = _MaskedDecoder(EMB, out_channels=C, t_out=T)
        out = dec(_rand_z())
        assert out.dtype == torch.float32

    def test_no_nan(self) -> None:
        dec = _MaskedDecoder(EMB, out_channels=C, t_out=T)
        out = dec(_rand_z())
        assert not torch.isnan(out).any()

    def test_gradient_flows(self) -> None:
        dec = _MaskedDecoder(EMB, out_channels=C, t_out=T)
        z = _rand_z().requires_grad_(True)
        dec(z).sum().backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# VICReg full training loop (2 epochs, CPU)
# ---------------------------------------------------------------------------

class TestVICRegTrain:
    def test_encoder_only_final_save(self, tmp_path) -> None:
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        final = tmp_path / "tr.pt"
        assert final.exists(), "Final encoder checkpoint not saved"
        sd = torch.load(str(final), map_location="cpu")
        # Must be a flat state_dict (dict of tensors), not a nested checkpoint
        assert isinstance(sd, dict)
        assert all(isinstance(v, torch.Tensor) for v in sd.values())

    def test_final_checkpoint_loadable(self, tmp_path) -> None:
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr2"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        sd = torch.load(str(tmp_path / "tr2.pt"), map_location="cpu")
        enc2 = _enc()
        enc2.load_state_dict(sd)   # must not raise

    def test_csv_log_created(self, tmp_path) -> None:
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr3"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        log_csv = tmp_path / "tr3_log.csv"
        assert log_csv.exists()

    def test_csv_has_correct_columns(self, tmp_path) -> None:
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr4"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        rows = list(csv.DictReader((tmp_path / "tr4_log.csv").open()))
        assert len(rows) == 2   # 2 epochs
        assert "inv" in rows[0] and "var" in rows[0] and "cov" in rows[0]
        assert "emb_std_mean" in rows[0] and "eff_rank" in rows[0]

    def test_intermediate_checkpoint_saved(self, tmp_path) -> None:
        enc = _enc()
        args = _vicreg_args("tr5", ckpt_every=1)
        _train_vicreg(enc, _smoke_loader(), args, _smoke_cfg(), torch.device("cpu"), tmp_path)
        # ckpt_every=1 → checkpoint at ep 1 and ep 2
        assert (tmp_path / "tr5_ep0001.pt").exists()
        assert (tmp_path / "tr5_ep0002.pt").exists()

    def test_intermediate_checkpoint_has_encoder_and_opt(self, tmp_path) -> None:
        enc = _enc()
        args = _vicreg_args("tr6", ckpt_every=1)
        _train_vicreg(enc, _smoke_loader(), args, _smoke_cfg(), torch.device("cpu"), tmp_path)
        ckpt = torch.load(str(tmp_path / "tr6_ep0001.pt"), map_location="cpu")
        assert "encoder"   in ckpt
        assert "projector" in ckpt
        assert "opt"       in ckpt

    def test_loss_decreases_or_bounded(self, tmp_path) -> None:
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr7"), _smoke_cfg(epochs=3), torch.device("cpu"), tmp_path)
        rows = list(csv.DictReader((tmp_path / "tr7_log.csv").open()))
        totals = [float(r["total"]) for r in rows]
        # loss should stay finite
        assert all(t < 1e6 for t in totals), "Loss exploded"

    def test_no_labels_in_saved_encoder(self, tmp_path) -> None:
        """Final .pt must contain encoder weights only — no labels/projector."""
        enc = _enc()
        _train_vicreg(enc, _smoke_loader(), _vicreg_args("tr8"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        sd = torch.load(str(tmp_path / "tr8.pt"), map_location="cpu")
        # Projector keys start with "net."; encoder keys start with "stem.", "blocks.", etc.
        assert not any("net." in k for k in sd.keys()), "Projector weights leaked into final save"


# ---------------------------------------------------------------------------
# Masked training loop (2 epochs, CPU)
# ---------------------------------------------------------------------------

class TestMaskedTrain:
    def test_encoder_only_final_save(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        final = tmp_path / "mt.pt"
        assert final.exists()
        sd = torch.load(str(final), map_location="cpu")
        assert isinstance(sd, dict)
        assert all(isinstance(v, torch.Tensor) for v in sd.values())

    def test_final_checkpoint_loadable(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt2"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        sd = torch.load(str(tmp_path / "mt2.pt"), map_location="cpu")
        enc2 = _enc()
        enc2.load_state_dict(sd)

    def test_csv_log_created(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt3"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        assert (tmp_path / "mt3_log.csv").exists()

    def test_csv_has_recon_column(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt4"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        rows = list(csv.DictReader((tmp_path / "mt4_log.csv").open()))
        assert "recon" in rows[0]
        assert "emb_std_mean" in rows[0]

    def test_recon_loss_finite(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt5"), _smoke_cfg(epochs=3), torch.device("cpu"), tmp_path)
        rows = list(csv.DictReader((tmp_path / "mt5_log.csv").open()))
        for r in rows:
            assert float(r["recon"]) < 1e6

    def test_no_decoder_in_final_save(self, tmp_path) -> None:
        enc = _enc()
        _train_masked(enc, _smoke_loader(), _masked_args("mt6"), _smoke_cfg(), torch.device("cpu"), tmp_path)
        sd = torch.load(str(tmp_path / "mt6.pt"), map_location="cpu")
        # Decoder keys contain "expand." or "dec."
        assert not any(("expand." in k or k.startswith("dec.")) for k in sd.keys()), \
            "Decoder weights leaked into final encoder save"


# ---------------------------------------------------------------------------
# CLI smoke (subprocess)
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_smoke_exits_zero(self) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "src.ssl_train", "--smoke"],
            capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        assert result.returncode == 0, f"Smoke failed:\n{result.stderr}"
        assert "PASSED" in result.stdout
