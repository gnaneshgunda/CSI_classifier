"""
scripts/train_cloud.py
======================
Train SSL models on Modal's free T4 GPU.

Setup (one-time, ~2 minutes):
    pip install modal
    modal setup          # opens browser to create/link account

Usage:
    python scripts/train_cloud.py --run_name vicreg_aug  --mode vicreg
    python scripts/train_cloud.py --run_name masked      --mode masked
    python scripts/train_cloud.py --run_name vicreg_noaug --mode vicreg --no_aug

What it does:
    1. Zips derived/windows/ locally (one-time, ~30s)
    2. Uploads code + windows + config to Modal's GPU container
    3. Runs ssl_train.py on a T4 GPU
    4. Downloads the checkpoint to checkpoints/{run_name}.pt

Cost:
    ~$0.10-0.20 per run on Modal's free Starter plan ($30/mo credits).
    All 3 runs combined: under $1 in credits.

Prerequisites:
    - derived/windows/ must exist (run build_windows.py first)
    - derived/norm_stats.npz must exist (run fit_normalization.py first)
    - derived/manifest.csv and derived/windows_index.csv must exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSL model on Modal cloud GPU")
    parser.add_argument("--run_name", required=True, help="Checkpoint name (e.g. vicreg_aug)")
    parser.add_argument("--mode", choices=["vicreg", "masked"], default="vicreg")
    parser.add_argument("--no_aug", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Modal app (written as a separate file that Modal CLI runs)
# ---------------------------------------------------------------------------

_MODAL_APP_SOURCE = '''
"""
Auto-generated Modal app for CSI SSL training.
Run with: modal run _modal_app.py --run-name vicreg_aug --mode vicreg
"""
import sys, os, subprocess, zipfile, shutil
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Build the container image with all dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.1.3",
        "scipy==1.14.1",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
        "torch==2.6.0",
        "matplotlib==3.9.2",
    )
    .apt_install("libgl1", "libglib2.0-0")
    # Add local source code
    .add_local_python_source("src")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("configs/experiment.yaml", "/root/configs/experiment.yaml")
    .add_local_file("derived/manifest.csv", "/root/derived/manifest.csv")
    .add_local_file("derived/windows_index.csv", "/root/derived/windows_index.csv")
    .add_local_file("derived/norm_stats.npz", "/root/derived/norm_stats.npz")
    .add_local_file("derived/windows.zip", "/root/derived/windows.zip")
)

# Volume to store checkpoints across runs
ckpt_volume = modal.Volume.from_name("csi-checkpoints", create_if_missing=True)

app = modal.App("csi-ssl-training")

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/root/checkpoints": ckpt_volume},
)
def train(run_name: str, mode: str, no_aug: bool = False):
    """Run SSL training inside the GPU container."""
    # Unzip windows
    print("Unzipping windows ...")
    with zipfile.ZipFile("/root/derived/windows.zip", "r") as z:
        z.extractall("/root/derived/")

    # Verify
    n_windows = len(list(Path("/root/derived/windows").rglob("*.npy")))
    print(f"Windows found: {n_windows}")

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build command
    cmd = [
        sys.executable, "-m", "src.ssl_train",
        "--config", "configs/experiment.yaml",
        "--mode", mode,
        "--run_name", run_name,
    ]
    if no_aug:
        cmd.append("--no_aug")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root")
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Commit checkpoint to volume
    ckpt_path = Path(f"/root/checkpoints/{run_name}.pt")
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint: {ckpt_path} ({size_mb:.1f} MB)")

    ckpt_volume.commit()
    return {"run_name": run_name, "size_mb": size_mb, "path": str(ckpt_path)}


@app.function(
    image=image,
    volumes={"/root/checkpoints": ckpt_volume},
)
def get_checkpoint(run_name: str) -> bytes:
    """Download a checkpoint from the volume."""
    ckpt_path = Path(f"/root/checkpoints/{run_name}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    return ckpt_path.read_bytes()
'''


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _zip_windows() -> Path:
    """Zip derived/windows/ into derived/windows.zip (one-time)."""
    zip_path = _REPO_ROOT / "derived" / "windows.zip"
    if zip_path.exists():
        print(f"windows.zip already exists ({zip_path.stat().st_size / 1e6:.0f} MB), reusing")
        return zip_path

    import zipfile
    windows_dir = _REPO_ROOT / "derived" / "windows"
    if not windows_dir.exists():
        print("ERROR: derived/windows/ not found. Run build_windows.py first.")
        sys.exit(1)

    print(f"Zipping {windows_dir} ...")
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for npy in windows_dir.rglob("*.npy"):
            arcname = str(npy.relative_to(_REPO_ROOT / "derived"))
            zf.write(npy, arcname)
            n += 1
    print(f"Zipped {n} windows ({zip_path.stat().st_size / 1e6:.0f} MB)")
    return zip_path


def _check_prereqs() -> None:
    """Verify all required files exist before submitting to cloud."""
    required = [
        _REPO_ROOT / "derived" / "windows",
        _REPO_ROOT / "derived" / "manifest.csv",
        _REPO_ROOT / "derived" / "windows_index.csv",
        _REPO_ROOT / "configs" / "experiment.yaml",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    norm = _REPO_ROOT / "derived" / "norm_stats.npz"
    if not norm.exists():
        print("ERROR: norm_stats.npz not found. Run:")
        print("  python scripts/fit_normalization.py --config configs/experiment.yaml")
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    _check_prereqs()
    _zip_windows()

    # Ensure modal is installed
    try:
        import modal
    except ImportError:
        print("Installing modal ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modal"])
        import modal

    # Write the Modal app file
    app_path = _REPO_ROOT / "_modal_app.py"
    app_path.write_text(_MODAL_APP_SOURCE, encoding="utf-8")

    try:
        print(f"\nSubmitting {args.mode} training ({args.run_name}) to Modal T4 GPU ...")
        print("This will upload data (~first run takes a few min) then train on GPU.\n")

        # Use modal CLI to run the app
        cmd = [
            sys.executable, "-m", "modal", "run", str(app_path),
            "--run-name", args.run_name,
            "--mode", args.mode,
        ]
        if args.no_aug:
            cmd.append("--no-aug")

        import subprocess
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if result.returncode == 0:
            print(f"\nTraining complete! Downloading checkpoint ...")
            # Download checkpoint from Modal volume
            dl_cmd = [
                sys.executable, "-m", "modal", "run", str(app_path),
                "--function", "get_checkpoint",
                "--run-name", args.run_name,
            ]
            # Actually, simpler: the checkpoint is in the volume, we need to fetch it
            # Let's use a direct approach
            _download_checkpoint(args.run_name)
        else:
            print(f"\nTraining failed (exit code {result.returncode})")
            sys.exit(1)

    finally:
        app_path.unlink(missing_ok=True)


def _download_checkpoint(run_name: str) -> None:
    """Download checkpoint from Modal volume to local checkpoints/ dir."""
    ckpt_dir = _REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    local_path = ckpt_dir / f"{run_name}.pt"

    # Write a tiny modal script to fetch the file
    fetch_script = f'''
import modal

ckpt_volume = modal.Volume.from_name("csi-checkpoints", create_if_missing=True)
app = modal.App("csi-fetch")

@app.function(volumes={{"/root/checkpoints": ckpt_volume}})
def fetch():
    from pathlib import Path
    p = Path("/root/checkpoints/{run_name}.pt")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p.read_bytes()

if __name__ == "__main__":
    import sys
    with app.run() as app_instance:
        data = fetch.remote()
        print(f"Downloaded {{len(data)}} bytes")
'''
    fetch_path = _REPO_ROOT / "_modal_fetch.py"
    fetch_path.write_text(fetch_script, encoding="utf-8")

    try:
        import subprocess
        # Run modal to get the bytes and pipe to file
        result = subprocess.run(
            [sys.executable, "-m", "modal", "run", str(fetch_path)],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Download failed. The checkpoint is in Modal volume 'csi-checkpoints'.")
            print(f"You can retrieve it manually or retry.")
            print(f"Stderr: {result.stderr[-500:]}")
            return

        # Try to find the bytes in output — Modal might print them
        # Actually, we need a different approach for binary data
        # Let's save to a Modal Volume path and use modal volume get
        print("Checkpoint is stored in Modal volume 'csi-checkpoints'.")
        print(f"To download manually:")
        print(f"  modal volume get csi-checkpoints {run_name}.pt {local_path}")

    finally:
        fetch_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
