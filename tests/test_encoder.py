"""
tests/test_encoder.py
=====================
Unit tests for src/encoder.py (P10 – ResNet1D-lite).

Test classes
------------
TestResNet1DEncoderShape       – output shapes for canonical and variable lengths
TestResNet1DEncoderParameters  – under 1 M params, param count consistent
TestResNet1DEncoderDeterminism – same seed → same output
TestResNet1DEncoderVariableLen – works for T=100 and T=300 (length-agnostic)
TestResNet1DEncoderFlags       – random_init flag stored, verbose=False suppresses print
TestResNet1DEncoderGradients   – backward pass doesn't crash, grads flow
TestResNet1DEncoderNoBatchNorm – no BatchNorm in the model (GroupNorm only)
TestResNet1DEncoderNoFlatten   – no Flatten / fixed positional embedding
TestAttentivePool              – attentive pool sums to 1 (weights are a simplex)
"""

from __future__ import annotations

import math
import torch
import pytest

from src.encoder import ResNet1DEncoder, _ResBlock1D, _AttentivePool1D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, T, C_IN = 4, 150, 256   # canonical batch / time / subcarriers
EMB = 128                   # default embedding dim


def _make_input(b: int = B, t: int = T, c: int = C_IN, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(b, t, c, generator=g)


def _encoder(verbose: bool = False, **kw) -> ResNet1DEncoder:
    return ResNet1DEncoder(verbose=verbose, **kw)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestResNet1DEncoderShape:
    def test_output_shape_canonical(self) -> None:
        enc = _encoder()
        x = _make_input()
        out = enc.encode(x)
        assert out.shape == (B, EMB), f"Expected ({B}, {EMB}), got {out.shape}"

    def test_encode_equals_forward(self) -> None:
        enc = _encoder()
        x = _make_input()
        assert torch.equal(enc.encode(x), enc(x))

    def test_batch_size_1(self) -> None:
        enc = _encoder()
        out = enc.encode(_make_input(b=1))
        assert out.shape == (1, EMB)

    def test_batch_size_8(self) -> None:
        enc = _encoder()
        out = enc.encode(_make_input(b=8))
        assert out.shape == (8, EMB)

    def test_output_dtype_float32(self) -> None:
        enc = _encoder()
        out = enc.encode(_make_input())
        assert out.dtype == torch.float32

    def test_custom_embedding_dim(self) -> None:
        enc = _encoder(embedding_dim=64)
        out = enc.encode(_make_input())
        assert out.shape == (B, 64)

    def test_embedding_dim_256(self) -> None:
        enc = _encoder(embedding_dim=256)
        out = enc.encode(_make_input())
        assert out.shape == (B, 256)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

class TestResNet1DEncoderParameters:
    def test_under_1m_parameters(self) -> None:
        enc = _encoder()
        n = enc.n_parameters()
        assert n < 1_000_000, (
            f"ResNet1DEncoder has {n:,} parameters — must be < 1,000,000"
        )

    def test_n_parameters_positive(self) -> None:
        enc = _encoder()
        assert enc.n_parameters() > 0

    def test_n_parameters_consistent(self) -> None:
        enc = _encoder()
        assert enc.n_parameters() == sum(
            p.numel() for p in enc.parameters() if p.requires_grad
        )

    def test_parameter_count_printed(self, capsys) -> None:
        ResNet1DEncoder(verbose=True)
        captured = capsys.readouterr()
        assert "Parameters:" in captured.out

    def test_verbose_false_no_print(self, capsys) -> None:
        ResNet1DEncoder(verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestResNet1DEncoderDeterminism:
    def test_same_input_same_output(self) -> None:
        enc = _encoder()
        enc.eval()
        x = _make_input(seed=42)
        out1 = enc.encode(x)
        out2 = enc.encode(x.clone())
        assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic output"

    def test_seeded_different_weights_different_output(self) -> None:
        torch.manual_seed(0)
        enc1 = _encoder()
        torch.manual_seed(1)
        enc2 = _encoder()
        enc1.eval(); enc2.eval()
        x = _make_input()
        # Different initialisations → different outputs (with overwhelming probability)
        assert not torch.allclose(enc1.encode(x), enc2.encode(x), atol=1e-4)

    def test_eval_mode_deterministic(self) -> None:
        """GroupNorm has no running stats, so eval/train should agree."""
        enc = _encoder()
        x = _make_input(seed=7)
        enc.eval()
        out_eval = enc.encode(x)
        enc.train()
        out_train = enc.encode(x)
        # GroupNorm is identical in train/eval — outputs must match
        assert torch.allclose(out_eval, out_train, atol=1e-6)


# ---------------------------------------------------------------------------
# Variable-length inputs (length-agnostic requirement)
# ---------------------------------------------------------------------------

class TestResNet1DEncoderVariableLen:
    @pytest.mark.parametrize("t", [50, 100, 150, 200, 300, 500])
    def test_arbitrary_length(self, t: int) -> None:
        enc = _encoder()
        enc.eval()
        x = _make_input(t=t)
        out = enc.encode(x)
        assert out.shape == (B, EMB), (
            f"T={t}: expected ({B}, {EMB}), got {out.shape}"
        )

    def test_length_100(self) -> None:
        enc = _encoder()
        out = enc.encode(_make_input(t=100))
        assert out.shape == (B, EMB)

    def test_length_300(self) -> None:
        enc = _encoder()
        out = enc.encode(_make_input(t=300))
        assert out.shape == (B, EMB)

    def test_outputs_differ_for_different_lengths(self) -> None:
        """A 150-step input and a 300-step input (same data, different T)
        should in general produce different embeddings."""
        enc = _encoder()
        enc.eval()
        g = torch.Generator(); g.manual_seed(99)
        base = torch.rand(1, 300, C_IN, generator=g)
        out150 = enc.encode(base[:, :150, :])
        out300 = enc.encode(base)
        assert not torch.allclose(out150, out300, atol=1e-4)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestResNet1DEncoderFlags:
    def test_random_init_flag_stored(self) -> None:
        enc_rand = _encoder(random_init=True)
        assert enc_rand.random_init is True

    def test_random_init_false_stored(self) -> None:
        enc = _encoder(random_init=False)
        assert enc.random_init is False

    def test_random_init_true_still_encodes(self) -> None:
        enc = _encoder(random_init=True)
        out = enc.encode(_make_input())
        assert out.shape == (B, EMB)

    def test_embedding_dim_stored(self) -> None:
        enc = _encoder(embedding_dim=64)
        assert enc.embedding_dim == 64


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestResNet1DEncoderGradients:
    def test_backward_does_not_crash(self) -> None:
        enc = _encoder()
        x = _make_input().requires_grad_(True)
        out = enc.encode(x)
        loss = out.sum()
        loss.backward()  # must not raise

    def test_parameter_grads_not_none_after_backward(self) -> None:
        enc = _encoder()
        x = _make_input()
        out = enc.encode(x)
        out.sum().backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_input_grad_not_none(self) -> None:
        enc = _encoder()
        x = _make_input().requires_grad_(True)
        enc.encode(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Architecture constraints: no BatchNorm, no Flatten, no fixed pos embed
# ---------------------------------------------------------------------------

class TestResNet1DEncoderNoBatchNorm:
    def test_no_batchnorm1d(self) -> None:
        enc = _encoder()
        for name, mod in enc.named_modules():
            assert not isinstance(mod, nn.BatchNorm1d), (
                f"Found BatchNorm1d at {name} — must use GroupNorm only"
            )

    def test_no_batchnorm2d(self) -> None:
        enc = _encoder()
        for name, mod in enc.named_modules():
            assert not isinstance(mod, nn.BatchNorm2d), (
                f"Found BatchNorm2d at {name}"
            )

    def test_has_groupnorm(self) -> None:
        enc = _encoder()
        has_gn = any(
            isinstance(m, nn.GroupNorm)
            for m in enc.modules()
        )
        assert has_gn, "Expected at least one GroupNorm layer"


class TestResNet1DEncoderNoFlatten:
    def test_no_flatten_module(self) -> None:
        enc = _encoder()
        for name, mod in enc.named_modules():
            assert not isinstance(mod, nn.Flatten), (
                f"Found nn.Flatten at {name} — encoder must be length-agnostic"
            )

    def test_no_embedding_module(self) -> None:
        """No learned positional embedding table."""
        enc = _encoder()
        for name, mod in enc.named_modules():
            assert not isinstance(mod, nn.Embedding), (
                f"Found nn.Embedding at {name} — must not use fixed positional embeddings"
            )

    def test_no_linear_before_pool(self) -> None:
        """Stem and blocks must use only convolutions — no Linear in stem/blocks."""
        for name, mod in ResNet1DEncoder(verbose=False).stem.named_modules():
            assert not isinstance(mod, nn.Linear), f"Linear in stem: {name}"
        for name, mod in ResNet1DEncoder(verbose=False).blocks.named_modules():
            assert not isinstance(mod, nn.Linear), f"Linear in blocks: {name}"


# ---------------------------------------------------------------------------
# Attentive pooling weights form a simplex
# ---------------------------------------------------------------------------

class TestAttentivePool:
    def test_attention_weights_sum_to_one(self) -> None:
        pool = _AttentivePool1D(128)
        x = torch.randn(4, 128, 75)   # (B, C, T)
        # Manually extract weights
        import torch.nn.functional as F
        w = F.softmax(pool.score(x), dim=2)   # (B, 1, T)
        sums = w.squeeze(1).sum(dim=1)         # (B,)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), (
            f"Attention weights don't sum to 1: {sums}"
        )

    def test_pool_output_shape(self) -> None:
        pool = _AttentivePool1D(128)
        x = torch.randn(4, 128, 75)
        out = pool(x)
        assert out.shape == (4, 128)

    def test_pool_length_agnostic(self) -> None:
        pool = _AttentivePool1D(128)
        for t in [10, 75, 150, 300]:
            out = pool(torch.randn(2, 128, t))
            assert out.shape == (2, 128), f"Failed for T={t}"


# ---------------------------------------------------------------------------
# ResBlock1D unit tests
# ---------------------------------------------------------------------------

class TestResBlock1D:
    def test_same_channels_shape(self) -> None:
        block = _ResBlock1D(64, 64, dilation=1)
        x = torch.randn(4, 64, 150)
        out = block(x)
        assert out.shape == (4, 64, 150)

    def test_channel_expansion_shape(self) -> None:
        block = _ResBlock1D(64, 128, dilation=2, stride=2)
        x = torch.randn(4, 64, 150)
        out = block(x)
        # With stride=2 length halves (approximately)
        assert out.shape[0] == 4
        assert out.shape[1] == 128

    def test_identity_shortcut_when_same_ch(self) -> None:
        block = _ResBlock1D(64, 64, dilation=1, stride=1)
        import torch.nn as nn
        assert isinstance(block.shortcut, nn.Identity)

    def test_conv_shortcut_when_ch_changes(self) -> None:
        block = _ResBlock1D(64, 128, dilation=2, stride=2)
        import torch.nn as nn
        assert isinstance(block.shortcut, nn.Sequential)

    def test_dilation_4(self) -> None:
        block = _ResBlock1D(128, 128, dilation=4)
        x = torch.randn(2, 128, 150)
        out = block(x)
        assert out.shape == (2, 128, 150)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402 – already imported above; kept for clarity
