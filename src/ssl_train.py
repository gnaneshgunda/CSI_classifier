"""
src/ssl_train.py
================
Self-supervised pre-training for the CSI-HAR encoder.

Two modes
---------
``vicreg``
    VICReg (Bardes et al., 2022).

    * Encoder  → 2-layer projector (dim 256 → 512 → projection_dim)
    * Loss = λ·invariance + μ·variance + ν·covariance
      - invariance: MSE between the two projected views
      - variance:   hinge loss std(z) ≥ γ=1 per dimension (averaged over dims)
      - covariance: squared off-diagonal entries of the normalised covariance matrix
    * Default weights λ=25, μ=25, ν=1 (configurable via CLI)

``masked``
    Masked auto-encoding (MAE-style).

    * A single view is masked: 50 % of (time × subcarrier) blocks are zeroed.
    * A small 2-layer 1-D CNN decoder reconstructs the original at masked positions.
    * Loss: MSE only on masked positions.

Common
------
* AdamW optimiser with cosine annealing (T_max = epochs, η_min = cfg.ssl.lr_min).
* Mixed precision (``torch.amp``) when CUDA is available.
* Every epoch: log invariance, variance, covariance (or recon) losses, mean and
  min per-dimension embedding std, and effective rank of the embedding covariance.
* Checkpoint every ``--ckpt_every`` epochs → ``checkpoints/{run_name}_ep{e}.pt``.
* At the end save **encoder weights only** → ``checkpoints/{run_name}.pt``.

CLI
---
    python -m src.ssl_train --config configs/experiment.yaml \\
        --mode vicreg --run_name vicreg_aug
    python -m src.ssl_train --config configs/experiment.yaml \\
        --mode vicreg --run_name vicreg_noaug --no_aug
    python -m src.ssl_train --config configs/experiment.yaml \\
        --mode masked --run_name masked_aug
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import ExperimentConfig, load_config
from src.encoder import ResNet1DEncoder
from src.augment import SSLPairDataset
from src.normalization import Normalizer

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class _Projector(nn.Module):
    """2-layer MLP projector: embedding_dim → 512 → projection_dim.

    Uses BatchNorm (appropriate for the projector head; encoder uses GroupNorm).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),   # no affine on last BN (VICReg)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Masked decoder (for masked mode)
# ---------------------------------------------------------------------------

class _MaskedDecoder(nn.Module):
    """Lightweight 2-layer 1-D CNN decoder.

    Takes the encoder embedding ``(B, emb_dim)`` and reconstructs the masked
    window ``(B, T, 256)``.  We broadcast the embedding over T via a linear
    expansion, then apply two Conv1d layers.
    """

    def __init__(self, emb_dim: int, out_channels: int = 256, t_out: int = 150) -> None:
        super().__init__()
        self._t_out = t_out
        self.expand = nn.Linear(emb_dim, 64 * t_out, bias=True)
        self.dec = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, emb_dim)
        B = z.shape[0]
        x = self.expand(z)                    # (B, 64*T)
        x = x.view(B, 64, self._t_out)        # (B, 64, T)
        x = self.dec(x)                       # (B, 256, T)
        return x.permute(0, 2, 1)             # (B, T, 256)


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute VICReg loss.

    Parameters
    ----------
    z1, z2:
        Projected embeddings ``(B, D)``.
    inv_weight, var_weight, cov_weight:
        Loss term weights.
    gamma:
        Variance hinge target (default 1.0).
    eps:
        Small constant for numerical stability.

    Returns
    -------
    (total_loss, terms_dict)
        ``terms_dict`` has keys ``inv``, ``var``, ``cov``.
    """
    B, D = z1.shape

    # ---- invariance: MSE between the two views ----
    inv = F.mse_loss(z1, z2)

    # ---- variance: hinge on std ≥ gamma ----
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var = (
        F.relu(gamma - std1).mean() +
        F.relu(gamma - std2).mean()
    ) / 2.0

    # ---- covariance: off-diagonal squared entries ----
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov1 = (z1_c.T @ z1_c) / (B - 1)
    cov2 = (z2_c.T @ z2_c) / (B - 1)

    def _off_diag_sq(cov: torch.Tensor) -> torch.Tensor:
        mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
        return cov[mask].pow(2).sum() / D

    cov = (_off_diag_sq(cov1) + _off_diag_sq(cov2)) / 2.0

    total = inv_weight * inv + var_weight * var + cov_weight * cov
    return total, {
        "inv":  float(inv),
        "var":  float(var),
        "cov":  float(cov),
        "total": float(total),
    }


# ---------------------------------------------------------------------------
# Masked-MAE loss helper
# ---------------------------------------------------------------------------

def make_block_mask(
    B: int,
    T: int,
    n_sub: int,
    mask_ratio: float,
    device: torch.device,
    block_t: int = 15,
    block_s: int = 16,
) -> torch.Tensor:
    """Create a boolean mask for block-masking.

    Approximately ``mask_ratio`` of ``(T, n_sub)`` positions are True (masked).
    Blocks of size ``(block_t, block_s)`` are randomly selected.

    Returns ``(B, T, n_sub)`` bool tensor (True = masked).
    """
    n_blocks_t = max(1, T // block_t)
    n_blocks_s = max(1, n_sub // block_s)
    total_blocks = n_blocks_t * n_blocks_s
    n_mask = round(mask_ratio * total_blocks)
    if n_mask <= 0:
        return torch.zeros(B, T, n_sub, dtype=torch.bool, device=device)  # ratio=0 → no masking

    mask = torch.zeros(B, T, n_sub, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(total_blocks, device=device)[:n_mask]
        for i in idx:
            bt = int(i) // n_blocks_s
            bs = int(i) % n_blocks_s
            t0 = bt * block_t
            s0 = bs * block_s
            mask[b, t0: t0 + block_t, s0: s0 + block_s] = True
    return mask


# ---------------------------------------------------------------------------
# Embedding diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def embedding_diagnostics(
    z: torch.Tensor,
    eps: float = 1e-4,
) -> Dict[str, float]:
    """Compute per-dimension std stats and effective rank.

    Parameters
    ----------
    z:
        Embeddings ``(B, D)``.

    Returns
    -------
    dict with keys ``emb_std_mean``, ``emb_std_min``, ``eff_rank``.
    """
    stds = torch.sqrt(z.var(dim=0) + eps)              # (D,)
    z_c  = z - z.mean(dim=0)
    cov  = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)    # (D, D)
    # effective rank via normalised entropy of eigenvalues
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    total   = eigvals.sum()
    if total > 0:
        p = eigvals / total
        # filter near-zero to avoid log(0)
        p = p[p > 1e-10]
        eff_rank = float(torch.exp(-( p * torch.log(p)).sum()))
    else:
        eff_rank = 1.0

    return {
        "emb_std_mean": float(stds.mean()),
        "emb_std_min":  float(stds.min()),
        "eff_rank":     eff_rank,
    }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_vicreg(
    encoder: ResNet1DEncoder,
    loader: DataLoader,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    device: torch.device,
    ckpt_dir: Path,
) -> None:
    projector = _Projector(
        in_dim=cfg.ssl.embedding_dim,
        hidden_dim=512,
        out_dim=cfg.ssl.projection_dim,
    ).to(device)

    params = list(encoder.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.ssl.lr, weight_decay=cfg.ssl.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.ssl.epochs, eta_min=cfg.ssl.lr_min
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    csv_path = ckpt_dir / f"{args.run_name}_log.csv"
    fieldnames = ["epoch", "inv", "var", "cov", "total",
                  "emb_std_mean", "emb_std_min", "eff_rank", "lr", "time_s"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    log.info(
        "VICReg | λ=%.0f μ=%.0f ν=%.2f | epochs=%d batch=%d lr=%.2e",
        args.inv_weight, args.var_weight, args.cov_weight,
        cfg.ssl.epochs, cfg.ssl.batch_size, cfg.ssl.lr,
    )

    for epoch in range(1, cfg.ssl.epochs + 1):
        encoder.train()
        projector.train()
        t0 = time.time()

        running: Dict[str, float] = {"inv": 0., "var": 0., "cov": 0., "total": 0.}
        all_z: List[torch.Tensor] = []
        n_batches = 0

        for view1, view2 in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z1 = projector(encoder.encode(view1))
                z2 = projector(encoder.encode(view2))
                loss, terms = vicreg_loss(
                    z1, z2,
                    inv_weight=args.inv_weight,
                    var_weight=args.var_weight,
                    cov_weight=args.cov_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            for k, v in terms.items():
                running[k] += v
            all_z.append(encoder.encode(view1).detach().cpu())
            n_batches += 1

        sched.step()

        # ---- diagnostics ----
        z_all  = torch.cat(all_z, dim=0)
        diag   = embedding_diagnostics(z_all)
        elapsed = time.time() - t0
        row = {
            "epoch":        epoch,
            "inv":          running["inv"]   / n_batches,
            "var":          running["var"]   / n_batches,
            "cov":          running["cov"]   / n_batches,
            "total":        running["total"] / n_batches,
            "emb_std_mean": diag["emb_std_mean"],
            "emb_std_min":  diag["emb_std_min"],
            "eff_rank":     diag["eff_rank"],
            "lr":           sched.get_last_lr()[0],
            "time_s":       round(elapsed, 2),
        }
        log.info(
            "Ep %3d/%d | inv=%.4f var=%.4f cov=%.4f total=%.4f | "
            "std_mean=%.3f std_min=%.3f eff_rank=%.1f | lr=%.2e | %.1fs",
            epoch, cfg.ssl.epochs,
            row["inv"], row["var"], row["cov"], row["total"],
            row["emb_std_mean"], row["emb_std_min"], row["eff_rank"],
            row["lr"], elapsed,
        )
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

        # ---- checkpoint ----
        if epoch % args.ckpt_every == 0:
            ckpt = ckpt_dir / f"{args.run_name}_ep{epoch:04d}.pt"
            torch.save({
                "epoch":          epoch,
                "encoder":        encoder.state_dict(),
                "projector":      projector.state_dict(),
                "opt":            opt.state_dict(),
                "sched":          sched.state_dict(),
                "args":           vars(args),
            }, ckpt)
            log.info("Checkpoint saved: %s", ckpt)

    # ---- final: encoder weights only ----
    final = ckpt_dir / f"{args.run_name}.pt"
    torch.save(encoder.state_dict(), final)
    log.info("Encoder weights saved: %s", final)


def _train_masked(
    encoder: ResNet1DEncoder,
    loader: DataLoader,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    device: torch.device,
    ckpt_dir: Path,
) -> None:
    T      = 150
    n_sub  = 256
    decoder = _MaskedDecoder(
        emb_dim=cfg.ssl.embedding_dim,
        out_channels=n_sub,
        t_out=T,
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt   = torch.optim.AdamW(params, lr=cfg.ssl.lr, weight_decay=cfg.ssl.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.ssl.epochs, eta_min=cfg.ssl.lr_min
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    csv_path = ckpt_dir / f"{args.run_name}_log.csv"
    fieldnames = ["epoch", "recon", "emb_std_mean", "emb_std_min", "eff_rank", "lr", "time_s"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    log.info(
        "Masked MAE | mask_ratio=%.2f | epochs=%d batch=%d lr=%.2e",
        args.mask_ratio, cfg.ssl.epochs, cfg.ssl.batch_size, cfg.ssl.lr,
    )

    for epoch in range(1, cfg.ssl.epochs + 1):
        encoder.train()
        decoder.train()
        t0 = time.time()

        running_recon = 0.0
        all_z: List[torch.Tensor] = []
        n_batches = 0

        for view1, _view2 in loader:
            view1 = view1.to(device, non_blocking=True)      # (B, T, 256)
            B, Tv, Sv = view1.shape

            mask = make_block_mask(B, Tv, Sv, args.mask_ratio, device)
            masked_input = view1.clone()
            masked_input[mask] = 0.0

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z    = encoder.encode(masked_input)          # (B, emb)
                recon = decoder(z)                           # (B, T, 256)
                # MSE only on masked positions
                loss  = F.mse_loss(recon[mask], view1[mask])

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            running_recon += float(loss)
            all_z.append(z.detach().cpu())
            n_batches += 1

        sched.step()

        z_all   = torch.cat(all_z, dim=0)
        diag    = embedding_diagnostics(z_all)
        elapsed = time.time() - t0
        row = {
            "epoch":        epoch,
            "recon":        running_recon / n_batches,
            "emb_std_mean": diag["emb_std_mean"],
            "emb_std_min":  diag["emb_std_min"],
            "eff_rank":     diag["eff_rank"],
            "lr":           sched.get_last_lr()[0],
            "time_s":       round(elapsed, 2),
        }
        log.info(
            "Ep %3d/%d | recon=%.6f | std_mean=%.3f std_min=%.3f eff_rank=%.1f | lr=%.2e | %.1fs",
            epoch, cfg.ssl.epochs,
            row["recon"], row["emb_std_mean"], row["emb_std_min"], row["eff_rank"],
            row["lr"], elapsed,
        )
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

        if epoch % args.ckpt_every == 0:
            ckpt = ckpt_dir / f"{args.run_name}_ep{epoch:04d}.pt"
            torch.save({
                "epoch":   epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "opt":     opt.state_dict(),
                "sched":   sched.state_dict(),
                "args":    vars(args),
            }, ckpt)
            log.info("Checkpoint saved: %s", ckpt)

    final = ckpt_dir / f"{args.run_name}.pt"
    torch.save(encoder.state_dict(), final)
    log.info("Encoder weights saved: %s", final)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke() -> None:
    """End-to-end smoke: 2 epochs on tiny synthetic data, both modes."""
    import tempfile
    from unittest.mock import MagicMock
    from src.augment import compose

    print("[smoke] ssl_train smoke ...")

    T, N, B_SMOKE, EMB = 150, 256, 8, 128

    # ---- synthetic dataset ----
    class _SmokeDataset(torch.utils.data.Dataset):
        def __len__(self): return 32
        def __getitem__(self, i):
            rng = np.random.default_rng(i)
            v = torch.from_numpy(rng.uniform(0.1, 5.0, (T, N)).astype(np.float32))
            return v, v.clone()

    loader = DataLoader(_SmokeDataset(), batch_size=B_SMOKE, drop_last=True)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)

        cfg = MagicMock()
        cfg.ssl.embedding_dim  = EMB
        cfg.ssl.projection_dim = 64
        cfg.ssl.epochs         = 2
        cfg.ssl.batch_size     = B_SMOKE
        cfg.ssl.lr             = 1e-3
        cfg.ssl.weight_decay   = 1e-4
        cfg.ssl.lr_min         = 1e-5

        args_vicreg = argparse.Namespace(
            run_name="smoke_vicreg", mode="vicreg",
            inv_weight=25., var_weight=25., cov_weight=1.,
            mask_ratio=0.5, ckpt_every=10,
        )
        args_masked = argparse.Namespace(
            run_name="smoke_masked", mode="masked",
            inv_weight=25., var_weight=25., cov_weight=1.,
            mask_ratio=0.5, ckpt_every=10,
        )

        device = torch.device("cpu")

        # VICReg
        enc = ResNet1DEncoder(embedding_dim=EMB, verbose=False)
        enc.to(device)
        _train_vicreg(enc, loader, args_vicreg, cfg, device, ckpt_dir)
        vicreg_out = ckpt_dir / "smoke_vicreg.pt"
        assert vicreg_out.exists(), "VICReg encoder weights not saved"
        sd = torch.load(str(vicreg_out), map_location="cpu")
        assert isinstance(sd, dict), "Encoder save must be a state_dict"
        # Make sure it loads back
        enc2 = ResNet1DEncoder(embedding_dim=EMB, verbose=False)
        enc2.load_state_dict(sd)
        print("[smoke] VICReg: encoder weights saved and loadable  ✓")

        # Masked
        enc3 = ResNet1DEncoder(embedding_dim=EMB, verbose=False)
        enc3.to(device)
        _train_masked(enc3, loader, args_masked, cfg, device, ckpt_dir)
        masked_out = ckpt_dir / "smoke_masked.pt"
        assert masked_out.exists(), "Masked encoder weights not saved"
        print("[smoke] Masked MAE: encoder weights saved  ✓")

        # Check CSV logs exist
        assert (ckpt_dir / "smoke_vicreg_log.csv").exists()
        assert (ckpt_dir / "smoke_masked_log.csv").exists()
        print("[smoke] CSV logs written  ✓")

    print("[smoke] ssl_train smoke PASSED ✓")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSI-HAR self-supervised pre-training (VICReg / Masked MAE)"
    )
    p.add_argument("--config",   default="configs/experiment.yaml",
                   help="Path to experiment YAML")
    p.add_argument("--mode",     choices=["vicreg", "masked"], default="vicreg",
                   help="SSL mode (default: vicreg)")
    p.add_argument("--run_name", default="ssl_run",
                   help="Name for checkpoints and logs")
    p.add_argument("--no_aug",   action="store_true",
                   help="Disable augmentation (use raw windows)")

    # VICReg loss weights
    p.add_argument("--inv_weight", type=float, default=25.0,
                   help="VICReg invariance weight λ (default: 25)")
    p.add_argument("--var_weight", type=float, default=25.0,
                   help="VICReg variance weight μ (default: 25)")
    p.add_argument("--cov_weight", type=float, default=1.0,
                   help="VICReg covariance weight ν (default: 1)")

    # Masked mode
    p.add_argument("--mask_ratio", type=float, default=0.5,
                   help="Fraction of blocks to mask (default: 0.5)")

    # Checkpointing
    p.add_argument("--ckpt_every", type=int, default=10,
                   help="Save a checkpoint every N epochs (default: 10)")

    p.add_argument("--smoke", action="store_true",
                   help="Run smoke test on synthetic data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.smoke:
        _run_smoke()
        return

    cfg = load_config(args.config)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ---- normalization ----
    norm = Normalizer(cfg, log1p=True)
    norm.load()
    log.info("Normalizer loaded from %s", Path(cfg.paths.derived_dir) / "norm_stats.npz")

    # ---- windows index ----
    index_csv = Path(cfg.paths.windows_index_csv)
    if not index_csv.exists():
        log.error("windows_index.csv not found: %s  (run build_windows.py first)", index_csv)
        sys.exit(1)
    index_df = pd.read_csv(str(index_csv))
    dev_df   = index_df[index_df["split"] == "dev"].copy()
    log.info("Dev windows: %d", len(dev_df))

    # ---- dataset & loader ----
    dataset = SSLPairDataset(
        dev_df,
        windows_dir=Path(cfg.paths.windows_dir),
        normalizer=norm,
        cfg=cfg,
        augment=not args.no_aug,
        seed=cfg.seeds.global_seed,
    )
    log.info("SSL pairs: %d  (augment=%s)", len(dataset), not args.no_aug)

    # Try to fit a large batch; fall back gracefully
    batch_size = max(128, cfg.ssl.batch_size)
    if len(dataset) < batch_size:
        batch_size = max(1, len(dataset) // 2)
        log.warning("Dataset smaller than batch_size; using batch_size=%d", batch_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # ---- encoder ----
    encoder = ResNet1DEncoder(
        embedding_dim=cfg.ssl.embedding_dim,
        random_init=True,
        verbose=True,
    ).to(device)

    # ---- checkpoint dir ----
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- train ----
    if args.mode == "vicreg":
        _train_vicreg(encoder, loader, args, cfg, device, ckpt_dir)
    else:
        _train_masked(encoder, loader, args, cfg, device, ckpt_dir)


if __name__ == "__main__":
    main()
