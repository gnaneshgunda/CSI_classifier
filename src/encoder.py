"""
src/encoder.py
==============
ResNet-1D lite encoder for the CSI-HAR pipeline.

Architecture
------------
Input: ``(B, T, 256)`` float32 — B windows, T time steps, 256 subcarriers.

The first thing the model does is **permute** so subcarriers become channels::

    (B, T, 256)  →  (B, 256, T)     # channels-first for Conv1d

Stem
~~~~
* Conv1d(256 → 64, kernel=7, stride=1, padding=3, bias=False)
* GroupNorm(8 groups, 64 channels)
* GELU

Residual blocks (3 × ResBlock1D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Block 1: 64  → 64,  dilation 1
Block 2: 64  → 128, dilation 2  (stride-2 downsampling)
Block 3: 128 → 128, dilation 4

Each ResBlock1D:
  - Conv1d(in → out, k=3, dilation=d, padding=d, bias=False)
  - GroupNorm → GELU
  - Conv1d(out → out, k=3, dilation=d, padding=d, bias=False)
  - GroupNorm
  - Shortcut: Conv1d(in → out, k=1) + GroupNorm if in≠out, else identity
  - Downsampling: when stride=2, first conv uses stride=2

Attentive pooling
~~~~~~~~~~~~~~~~~
A learned 1×1 convolution produces per-step scalar scores, softmax over T,
then the weighted sum over T gives the ``(B, 128)`` embedding.

Output
~~~~~~
``encode(x) → (B, 128)`` regardless of input length (any T > 0).

Parameter budget
~~~~~~~~~~~~~~~~
Under 1 M parameters. Printed at construction with ``verbose=True``.

Flags
~~~~~
``random_init=True``  – model is constructed but weights are *not* trained
                       (standard ``__init__`` initialisation is kept; this flag
                        is a marker so downstream code can skip loading a
                        checkpoint).

Public API
----------
``ResNet1DEncoder(embedding_dim=128, random_init=False, verbose=True)``
    ``.encode(x)``        – forward pass, returns ``(B, embedding_dim)``
    ``.n_parameters()``   – total trainable parameter count
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: choose number of GroupNorm groups that divides channel count
# ---------------------------------------------------------------------------

def _gn_groups(n_channels: int, preferred: int = 8) -> int:
    """Return the largest divisor of *n_channels* that is ≤ *preferred*."""
    for g in range(preferred, 0, -1):
        if n_channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    """Pre-activation residual block for 1-D sequences (channels-first).

    Parameters
    ----------
    in_ch:
        Number of input channels.
    out_ch:
        Number of output channels.  If ``in_ch != out_ch`` a 1×1 shortcut
        is added.
    dilation:
        Dilation factor for both internal Conv1d layers.
    stride:
        Stride applied to the *first* conv only (for downsampling).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        pad = dilation  # keep same length when stride=1

        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=3, stride=stride, dilation=dilation,
            padding=pad, bias=False,
        )
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch,
            kernel_size=3, stride=1, dilation=dilation,
            padding=pad, bias=False,
        )

        # Shortcut path
        if in_ch != out_ch or stride != 1:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(_gn_groups(out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation on the residual path
        residual = self.shortcut(x)
        x = F.gelu(self.norm1(x))
        x = self.conv1(x)
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


# ---------------------------------------------------------------------------
# Attentive pooling
# ---------------------------------------------------------------------------

class _AttentivePool1D(nn.Module):
    """Weighted sum over the time axis using learned per-step scores.

    The score network is a single Conv1d(C→1, k=1) followed by softmax
    over the time dimension.  This makes the operation length-agnostic.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(in_ch, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = F.softmax(self.score(x), dim=2)   # (B, 1, T)
        pooled = (x * w).sum(dim=2)            # (B, C)
        return pooled


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class ResNet1DEncoder(nn.Module):
    """ResNet-1D lite encoder.  Input ``(B, T, 256)`` → embedding ``(B, 128)``.

    Parameters
    ----------
    embedding_dim:
        Dimension of the output embedding (default 128).
    random_init:
        If ``True``, the encoder is used as-is without loading a pre-trained
        checkpoint.  Weights are still randomly initialised by PyTorch's
        default scheme.
    verbose:
        If ``True`` (default), print the parameter count at construction time.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        random_init: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.random_init    = random_init

        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(_gn_groups(64), 64),
            nn.GELU(),
        )

        # ---- residual blocks ----
        # Block 1: 64 → 64, dilation 1, no downsampling
        # Block 2: 64 → 128, dilation 2, stride-2 downsampling
        # Block 3: 128 → 128, dilation 4, no downsampling
        self.blocks = nn.Sequential(
            _ResBlock1D(64,  64,  dilation=1, stride=1),
            _ResBlock1D(64,  128, dilation=2, stride=2),
            _ResBlock1D(128, 128, dilation=4, stride=1),
        )

        # ---- attentive pooling → embedding ----
        self.pool = _AttentivePool1D(128)

        # ---- projection to embedding_dim (identity if already 128) ----
        if embedding_dim != 128:
            self.proj: nn.Module = nn.Linear(128, embedding_dim, bias=False)
        else:
            self.proj = nn.Identity()

        if verbose:
            n = self.n_parameters()
            print(f"[ResNet1DEncoder] Parameters: {n:,}  ({n / 1e6:.3f} M)")
            if n >= 1_000_000:
                import warnings
                warnings.warn(
                    f"ResNet1DEncoder has {n:,} parameters (≥ 1 M). "
                    "Consider reducing channel widths.",
                    UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Parameters
        ----------
        x:
            Float tensor of shape ``(B, T, 256)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, embedding_dim)``.
        """
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of windows to embeddings.

        Parameters
        ----------
        x:
            Float tensor ``(B, T, 256)``.  T can be any positive integer.

        Returns
        -------
        torch.Tensor
            Shape ``(B, embedding_dim)``, dtype same as *x*.
        """
        # (B, T, 256) → (B, 256, T)  — subcarriers become channels
        x = x.permute(0, 2, 1)

        x = self.stem(x)    # (B, 64, T)
        x = self.blocks(x)  # (B, 128, T')
        x = self.pool(x)    # (B, 128)
        x = self.proj(x)    # (B, embedding_dim)
        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
