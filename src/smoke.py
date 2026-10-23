"""Smoke test: end-to-end pipeline on tiny synthetic data.

Run with:
    python -m src.smoke

Expected output (< 30 s):
    [smoke] Generating synthetic captures...
    [smoke] Building manifest...
    [smoke] Manifest OK: 20 captures, splits={'train': 14, 'val': 3, 'test': 3}
    [smoke] Loading windows...
    [smoke] Windows shape: (N, 150, 256) dtype=float32
    [smoke] Normalization stats computed on train only.
    [smoke] DONE
"""
from __future__ import annotations

import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ALLOWED_ACTIVITIES, Config, load_config
from src.manifest import MANIFEST_COLUMNS, _assign_splits, _validate


N_CAPTURES_PER_CLASS = 4
PACKETS_PER_CAPTURE = 200
N_SUBCARRIERS = 256
WINDOW_LEN = 150


def _make_synthetic_dataset(root: Path) -> None:
    """Write tiny synthetic CSVs mimicking real data format."""
    header = ",".join(str(i) for i in range(N_SUBCARRIERS))
    rng = np.random.default_rng(0)
    activities = list(ALLOWED_ACTIVITIES)
    for act in activities:
        for idx in range(N_CAPTURES_PER_CLASS):
            data = rng.integers(0, 1500, size=(PACKETS_PER_CAPTURE, N_SUBCARRIERS))
            rows = [header] + [",".join(map(str, row)) for row in data]
            (root / f"{act}_{idx}.csv").write_text("\n".join(rows))


def _load_windows(csv_path: Path, window_len: int) -> np.ndarray:
    """Load a CSV and return (n_windows, window_len, 256) float32 array."""
    df = pd.read_csv(csv_path, header=0)
    arr = df.values.astype(np.float32)  # (n_packets, 256)
    n = len(arr)
    if n < window_len:
        # reflect pad
        repeats = (window_len // n) + 1
        arr = np.tile(arr, (repeats, 1))[:window_len]
    windows = []
    for start in range(0, n - window_len + 1, window_len):
        windows.append(arr[start : start + window_len])
    return np.stack(windows, axis=0) if windows else arr[np.newaxis, :window_len]


def run_smoke() -> None:
    print("[smoke] Generating synthetic captures...")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_synthetic_dataset(root)

        print("[smoke] Building manifest...")
        records = []
        for csv_path in sorted(root.glob("*.csv")):
            stem = csv_path.stem
            activity = stem.split("_")[0]
            records.append({
                "capture_id": stem,
                "activity": activity,
                "room": "room1",
                "csv_path": str(csv_path),
                "n_packets": PACKETS_PER_CAPTURE,
            })

        records = _assign_splits(records, train_frac=0.7, val_frac=0.15, seed=42)
        df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
        _validate(df)

        split_counts = Counter(df["split"])
        print(f"[smoke] Manifest OK: {len(df)} captures, splits={dict(split_counts)}")

        print("[smoke] Loading windows...")
        all_windows, all_splits = [], []
        for _, row in df.iterrows():
            wins = _load_windows(Path(row["csv_path"]), WINDOW_LEN)
            all_windows.append(wins)
            all_splits.extend([row["split"]] * len(wins))

        X = np.concatenate(all_windows, axis=0)
        assert X.shape[1:] == (WINDOW_LEN, N_SUBCARRIERS), f"Unexpected shape {X.shape}"
        assert X.dtype == np.float32
        print(f"[smoke] Windows shape: {X.shape} dtype={X.dtype}")

        # Normalization: compute stats on train only
        train_mask = np.array([s == "train" for s in all_splits])
        X_train = X[train_mask]
        mu = X_train.mean(axis=(0, 1), keepdims=True)
        sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm = (X - mu) / sigma
        print("[smoke] Normalization stats computed on train only.")

    print("[smoke] DONE")


if __name__ == "__main__":
    run_smoke()
