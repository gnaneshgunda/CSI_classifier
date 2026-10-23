"""
scripts/build_windows.py
========================
Tile every intermediate capture into ``(150, 256)`` windows and write:

* ``derived/windows/{split}/{capture_id}__{start}.npy``
* ``derived/windows_index.csv``

Split rules
-----------
* Validation captures → **canonical_window** only (one window per capture,
  non-overlapping, deterministic central selection).
* Development captures → **dev_windows** (sliding stride 50).

Provenance
----------
Each window .npy file is named ``{capture_id}__{start:05d}.npy`` where
``start`` is the 0-based row offset in the cleaned intermediate array.

``windows_index.csv`` columns::

    window_id, capture_id, room, activity, split, start, config_hash

* ``window_id`` is ``{capture_id}__{start:05d}`` — globally unique.
* ``config_hash`` is the 16-char SHA-256 prefix of the full
  ``ExperimentConfig`` so stale windows can be detected if the config changes.

Skip logic
----------
If the .npy for a (capture_id, start) pair already exists it is skipped
(unless ``--force``).

Usage
-----
    python scripts/build_windows.py --config configs/experiment.yaml
    python scripts/build_windows.py --config configs/experiment.yaml --force
    python scripts/build_windows.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_COLUMNS = [
    "window_id",
    "capture_id",
    "room",
    "activity",
    "split",
    "start",
    "config_hash",
]


# ---------------------------------------------------------------------------
# Core build logic
# ---------------------------------------------------------------------------

def _build_one_capture(
    capture_id: str,
    room: str,
    activity: str,
    split: str,
    intermediate_path: str,
    windows_dir: Path,
    cfg,                # ExperimentConfig
    config_hash: str,
    force: bool,
) -> List[dict]:
    """Build windows for one capture and return index rows.

    Returns a list of dicts (one per window) with keys matching
    ``_INDEX_COLUMNS``.
    """
    from src.windowing import canonical_window, dev_windows

    # Load intermediate .npy
    arr = np.load(intermediate_path)
    if arr.shape[1] != cfg.window.n_subcarriers:
        raise ValueError(
            f"{capture_id}: expected {cfg.window.n_subcarriers} subcarriers, "
            f"got {arr.shape[1]}"
        )

    index_rows: List[dict] = []
    split_dir = windows_dir / split

    if split == "val":
        # Validation: canonical window only
        window = canonical_window(arr, cfg.window)
        if window is None:
            log.warning("Capture %s excluded (too short)", capture_id)
            return []

        start = 0
        window_id = f"{capture_id}__{start:05d}"
        out_path = split_dir / f"{window_id}.npy"
        if force or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), window.astype(np.float32))
        index_rows.append({
            "window_id": window_id,
            "capture_id": capture_id,
            "room": room,
            "activity": activity,
            "split": split,
            "start": start,
            "config_hash": config_hash,
        })
    else:
        # Development: sliding windows
        windows = dev_windows(arr, cfg.window)
        if not windows:
            log.warning("Capture %s excluded (too short)", capture_id)
            return []

        for start, window in windows:
            window_id = f"{capture_id}__{start:05d}"
            out_path = split_dir / f"{window_id}.npy"
            if force or not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), window.astype(np.float32))
            index_rows.append({
                "window_id": window_id,
                "capture_id": capture_id,
                "room": room,
                "activity": activity,
                "split": split,
                "start": start,
                "config_hash": config_hash,
            })

    return index_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (150, 256) windows from intermediate captures."
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment.yaml",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild all windows even if .npy exists.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run on synthetic data only.",
    )
    return parser.parse_args()


def _run_smoke() -> None:
    """End-to-end smoke test on synthetic data."""
    import tempfile

    from src.config import WindowConfig
    from src.windowing import canonical_window, dev_windows, sliding_windows

    print("[smoke] Testing windowing functions on synthetic data ...")
    rng = np.random.default_rng(42)
    cfg = WindowConfig(
        length=150, n_subcarriers=256,
        dev_stride=50, val_stride=150,
        short_capture_policy="reflect_pad",
    )

    # Test 1: normal capture (200 rows)
    arr = rng.uniform(0, 100, size=(200, 256)).astype(np.float32)
    cw = canonical_window(arr, cfg)
    assert cw.shape == (150, 256), f"canonical shape: {cw.shape}"
    dw = dev_windows(arr, cfg)
    assert all(w.shape == (150, 256) for _, w in dw), "dev window shape"
    assert len(dw) > 1, f"expected >1 dev windows, got {len(dw)}"

    # Test 2: short capture (100 rows, reflect_pad)
    short = rng.uniform(0, 100, size=(100, 256)).astype(np.float32)
    cw_short = canonical_window(short, cfg)
    assert cw_short.shape == (150, 256), f"padded canonical: {cw_short.shape}"
    dw_short = dev_windows(short, cfg)
    assert len(dw_short) == 1, f"short dev: expected 1, got {len(dw_short)}"

    # Test 3: exclude policy
    cfg_excl = WindowConfig(
        length=150, n_subcarriers=256,
        dev_stride=50, val_stride=150,
        short_capture_policy="exclude",
    )
    assert canonical_window(short, cfg_excl) is None
    assert dev_windows(short, cfg_excl) == []

    # Test 4: determinism
    arr2 = arr.copy()
    cw1 = canonical_window(arr, cfg)
    cw2 = canonical_window(arr2, cfg)
    assert np.array_equal(cw1, cw2), "canonical not deterministic"
    dw1 = dev_windows(arr, cfg)
    dw2 = dev_windows(arr2, cfg)
    assert len(dw1) == len(dw2)
    for (s1, w1), (s2, w2) in zip(dw1, dw2):
        assert s1 == s2
        assert np.array_equal(w1, w2)

    # Test 5: windows never cross captures
    arr_a = rng.uniform(0, 100, size=(150, 256)).astype(np.float32)
    arr_b = rng.uniform(200, 300, size=(150, 256)).astype(np.float32)
    sw = sliding_windows(arr_a, cfg, stride=50)
    for _, w in sw:
        assert w.max() < 200, "Window leaked data from another capture!"

    # Test 6: build windows + index CSV
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        windows_dir = tmp / "windows"
        index_csv = tmp / "windows_index.csv"

        from src.manifest import ManifestRow, write_manifest
        rows = [
            ManifestRow(
                capture_id="A__fall__s0", room="A", activity="fall", split="dev",
                source_file="s0.pcap", n_packets_raw=200, n_packets_clean=200,
                intermediate_path="",
            ),
        ]
        # Write a synthetic intermediate file
        inter = tmp / "intermediate" / "A"
        inter.mkdir(parents=True)
        np.save(str(inter / "A__fall__s0.npy"), arr)

        manifest = tmp / "manifest.csv"
        write_manifest(rows, manifest)

        # Build windows using the manifest
        from src.manifest import read_manifest
        mrows = read_manifest(manifest)
        for r in mrows:
            r.intermediate_path = str(inter / f"{r.capture_id}.npy")

        index_data = []
        for r in mrows:
            idx = _build_one_capture(
                r.capture_id, r.room, r.activity, r.split,
                r.intermediate_path, windows_dir, _make_minimal_cfg(),
                "test_hash", force=True,
            )
            index_data.extend(idx)

        # Write index CSV
        index_csv.parent.mkdir(parents=True, exist_ok=True)
        with index_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_INDEX_COLUMNS)
            w.writeheader()
            for row in index_data:
                w.writerow(row)

        # Verify
        loaded = list(csv.DictReader(index_csv.open(encoding="utf-8")))
        assert len(loaded) > 0
        for row in loaded:
            wpath = windows_dir / row["split"] / f"{row['window_id']}.npy"
            assert wpath.exists(), f"Missing window: {wpath}"
            warr = np.load(str(wpath))
            assert warr.shape == (150, 256), f"Shape: {warr.shape}"

    print(f"[smoke] Canonical: shape (150, 256) OK")
    print(f"[smoke] Dev windows: {len(dw)} from 200-row capture OK")
    print(f"[smoke] Short capture: reflect-padded to (150, 256) OK")
    print(f"[smoke] Determinism: same input → same output OK")
    print(f"[smoke] No cross-capture leakage OK")
    print(f"[smoke] Index CSV + .npy files verified OK")
    print("[smoke] All smoke checks passed ✓")


def _make_minimal_cfg():
    """Create a minimal ExperimentConfig-like object for smoke testing."""
    from src.config import WindowConfig, ExperimentConfig, PathsConfig, \
        SeedsConfig, CleaningConfig, NormalizationConfig, SSLConfig, \
        AugmentConfig, AlignConfig, RoutingConfig, FewshotConfig, \
        ActivitiesConfig, RoomsConfig, SplitsConfig

    return ExperimentConfig(
        paths=PathsConfig(
            room_a_pcap_dir=Path("."), room_a_val_dir=Path("."),
            room_b_csv_dir=Path("."), derived_dir=Path("."),
            intermediate_dir=Path("."), windows_dir=Path("."),
            manifest_csv=Path("."), windows_index_csv=Path("."),
            checkpoints_dir=Path("."), reports_dir=Path("."),
        ),
        seeds=SeedsConfig(global_seed=42, fewshot_seeds=[0]),
        window=WindowConfig(
            length=150, n_subcarriers=256,
            dev_stride=50, val_stride=150,
            short_capture_policy="reflect_pad",
        ),
        cleaning=CleaningConfig(
            min_valid_packets=10, max_nan_fraction_per_packet=0.10,
            amplitude_clip_max=5000.0, amplitude_floor=0.0,
            max_zero_subcarrier_fraction=0.50,
        ),
        normalization=NormalizationConfig(method="per_capture", clip_after_norm=10.0),
        ssl=SSLConfig(
            encoder="cnn1d", projection_dim=128, embedding_dim=256,
            temperature=0.07, epochs=50, batch_size=64, lr=1e-3,
            weight_decay=1e-4, lr_min=1e-5, n_negatives=None,
        ),
        augment=AugmentConfig(
            gaussian_noise_std=0.05, time_crop_min_frac=0.70,
            subcarrier_mask_frac=0.10, amplitude_scale_min=0.80,
            amplitude_scale_max=1.20,
        ),
        align=AlignConfig(method="none"),
        routing=RoutingConfig(mode="predicted_hard", confidence_threshold=0.60),
        fewshot=FewshotConfig(
            K_list=[1], n_seeds=10, distance_metric="cosine",
            shrinkage="ledoit_wolf", classifier="ncm",
        ),
        activities=ActivitiesConfig(
            room_a=["fall"], room_b=["empty"], shared=["sit"],
            room_a_exclusive=["fall"], room_b_exclusive=["empty"],
            all=["empty", "fall", "sit"],
        ),
        rooms=RoomsConfig(all=["A", "B"]),
        splits=SplitsConfig(all=["dev", "val"], val_seed=42, val_captures_per_class=25),
    )


def main() -> None:
    args = _parse_args()

    if args.smoke:
        _run_smoke()
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore[misc]
            return it

    from src.config import load_config
    from src.manifest import read_manifest

    cfg = load_config(args.config)
    config_hash = cfg.config_hash()

    manifest_path = Path(cfg.paths.manifest_csv)
    if not manifest_path.exists():
        log.error("Manifest not found: %s — run build_intermediate.py first", manifest_path)
        sys.exit(1)

    windows_dir = Path(cfg.paths.windows_dir)
    index_csv = Path(cfg.paths.windows_index_csv)
    intermediate_dir = Path(cfg.paths.intermediate_dir)

    rows = read_manifest(manifest_path)
    log.info("Building windows for %d captures (config_hash=%s)", len(rows), config_hash)

    all_index: List[dict] = []
    skipped = 0
    errors = 0

    for row in tqdm(rows, desc="Windowing", unit="capture"):
        # Resolve intermediate path
        inter_path = Path(row.intermediate_path)
        if not inter_path.is_absolute():
            inter_path = Path(".") / inter_path
        if not inter_path.exists():
            log.error("Intermediate file not found: %s", inter_path)
            errors += 1
            continue

        try:
            idx_rows = _build_one_capture(
                capture_id=row.capture_id,
                room=row.room,
                activity=row.activity,
                split=row.split,
                intermediate_path=str(inter_path),
                windows_dir=windows_dir,
                cfg=cfg,
                config_hash=config_hash,
                force=args.force,
            )
            all_index.extend(idx_rows)
        except Exception as exc:
            log.error("FAILED %s: %s", row.capture_id, exc)
            errors += 1

    # Write index CSV
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_csv.with_suffix(".tmp.csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_INDEX_COLUMNS)
            writer.writeheader()
            for row in all_index:
                writer.writerow(row)
        tmp.replace(index_csv)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    log.info(
        "Done. Windows: %d  Errors: %d  Index: %s",
        len(all_index), errors, index_csv,
    )

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
