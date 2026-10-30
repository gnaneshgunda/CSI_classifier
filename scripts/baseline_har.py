"""
scripts/baseline_har.py
=======================
Baseline Human Activity Recognition (HAR) classifiers using hand-crafted CSI
features extracted by ``src/features.py``.

Three classifiers are swept:
  * Logistic Regression  (LR)
  * Support Vector Machine (SVM, RBF kernel)
  * Random Forest         (RF)

Training data
-------------
All **development-split** windows from ``derived/windows_index.csv``.
No validation data is used during training or hyperparameter fitting.

Scoring
-------
Evaluated on the **fixed validation-split** windows only.
Metrics reported:
  * ``macro_f1``         – macro-averaged F1 across activity classes
  * ``balanced_accuracy``– balanced accuracy (accounts for class imbalance)

Output
------
``reports/baseline_har_results.csv`` with columns::

    experiment, target_room, K, metric, value

Where:
  * ``experiment``   – ``"LR"`` / ``"SVM"`` / ``"RF"``
  * ``target_room``  – ``"A"``, ``"B"``, or ``"joint"``
  * ``K``            – always ``-1`` (no few-shot; sentinel for joint-table compatibility)
  * ``metric``       – ``"macro_f1"`` or ``"balanced_accuracy"``
  * ``value``        – float score in [0, 1]

Usage
-----
    python scripts/baseline_har.py --config configs/experiment.yaml
    python scripts/baseline_har.py --smoke   # synthetic data, no real files needed

Options
-------
``--config PATH``      Path to experiment.yaml (default: configs/experiment.yaml)
``--smoke``            Run on tiny synthetic data (no real files needed)
``--workers N``        Parallelism for RandomForest (default: 1)
``--packet-rate HZ``   Override packet rate in Hz (default: infer from config)
``--out PATH``         Override output CSV path

Design notes
------------
* Feature extraction uses ``src.features.extract_features`` with
  ``packet_rate = cfg_packet_rate(cfg)`` (derived from the 30 ms path convention or explicit cfg).
* Classifiers use sklearn defaults except:
  - LR:  ``max_iter=2000``, ``C=1.0``, ``solver='lbfgs'``, multi-class ``'multinomial'``
  - SVM: ``kernel='rbf'``, ``C=10.0``, ``gamma='scale'``
  - RF:  ``n_estimators=300``, ``max_depth=None``, ``n_jobs=workers``
* Feature scaling: ``StandardScaler`` fitted on dev set only (no leakage).
* Rooms evaluated independently (per-room) AND jointly (all rooms together).
* If a room has no validation windows, that room's rows are omitted with a
  WARNING rather than raising an error (graceful degradation on partial data).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
log = logging.getLogger("baseline_har")

# ---------------------------------------------------------------------------
# Packet-rate helper
# ---------------------------------------------------------------------------

# Default packet rate for Room A (path says 30ms inter-packet → 33.3 Hz).
# Room B nominal rate varies; we use the same default so features are
# comparable across rooms.
_DEFAULT_PACKET_RATE_HZ: float = 100.0 / 3.0   # 33.33... Hz


def cfg_packet_rate(cfg) -> float:
    """Infer packet rate in Hz from configuration (best-effort).

    Checks for explicit ``packet_rate`` / ``fps`` configuration attributes,
    then inspects ``paths.room_a_pcap_dir`` for timing tokens
    ('30ms' -> ~33.3 Hz, '10ms' -> 100 Hz, '50ms' -> 20 Hz), and falls back
    to ``_DEFAULT_PACKET_RATE_HZ`` (100/3 Hz).
    """
    if cfg is not None:
        for attr in ("packet_rate", "fps"):
            val = getattr(cfg, attr, None)
            if val is not None:
                try:
                    fval = float(val)
                    if fval > 0:
                        return fval
                except (ValueError, TypeError):
                    pass
        for sec in ("window", "features", "paths"):
            sec_obj = getattr(cfg, sec, None)
            if sec_obj is not None:
                val = getattr(sec_obj, "packet_rate", None)
                if val is not None:
                    try:
                        fval = float(val)
                        if fval > 0:
                            return fval
                    except (ValueError, TypeError):
                        pass
    try:
        pcap_dir = str(cfg.paths.room_a_pcap_dir)
        if "30ms" in pcap_dir:
            return 1000.0 / 30.0    # 33.333... Hz
        if "10ms" in pcap_dir:
            return 100.0
        if "50ms" in pcap_dir:
            return 20.0
    except Exception:
        pass
    return _DEFAULT_PACKET_RATE_HZ


# ---------------------------------------------------------------------------
# Feature extraction over a split
# ---------------------------------------------------------------------------

def _load_windows_for_split(
    index_df,          # pandas DataFrame
    split: str,
    windows_dir: Path,
    packet_rate: float,
    room: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load windows from disk, extract features, return (X, y_str, rooms).

    Parameters
    ----------
    index_df:
        Full windows index DataFrame.
    split:
        ``"dev"`` or ``"val"``.
    windows_dir:
        Root directory containing ``dev/`` and ``val/`` subdirectories.
    packet_rate:
        CSI packet rate in Hz.
    room:
        If supplied, filter to this room only.

    Returns
    -------
    X : np.ndarray shape ``(N, FEATURE_DIM)``
    y : np.ndarray shape ``(N,)`` dtype object (activity strings)
    rooms : list of str length N
    """
    from src.features import extract_features, FEATURE_DIM

    df = index_df[index_df["split"] == split].copy()
    if room is not None:
        df = df[df["room"] == room]

    if df.empty:
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,),             dtype=object),
            [],
        )

    X_list: List[np.ndarray] = []
    y_list: List[str]        = []
    rooms_list: List[str]    = []
    skipped = 0

    for _, row in df.iterrows():
        wid   = row["window_id"]
        wpath = windows_dir / split / f"{wid}.npy"
        if not wpath.exists():
            skipped += 1
            continue
        try:
            win = np.load(str(wpath)).astype(np.float32)
            feat = extract_features(win, packet_rate=packet_rate)
            X_list.append(feat)
            y_list.append(str(row["activity"]))
            rooms_list.append(str(row["room"]))
        except Exception as exc:
            log.warning("Skipping window %s: %s", wid, exc)
            skipped += 1

    if skipped:
        log.warning(
            "Skipped %d windows (split=%s, room=%s)",
            skipped, split, room or "all",
        )

    if not X_list:
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,),             dtype=object),
            [],
        )

    return (
        np.vstack(X_list).astype(np.float32),
        np.array(y_list,     dtype=object),
        rooms_list,
    )


# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------

def _make_classifiers(n_jobs: int = 1) -> Dict[str, object]:
    """Return a dict of {name: sklearn_estimator} to sweep."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm          import SVC
    from sklearn.ensemble      import RandomForestClassifier

    return {
        "LR": LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=n_jobs,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return macro_f1 and balanced_accuracy."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _run_one_room(
    name: str,
    clf,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_room: str,
) -> List[dict]:
    """Fit on dev, score on val for a single room (or joint).

    Returns list of result dicts ready for CSV writing.
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if X_dev.shape[0] == 0:
        log.warning("[%s] No dev windows for room=%s — skipping.", name, target_room)
        return []
    if X_val.shape[0] == 0:
        log.warning("[%s] No val windows for room=%s — skipping.", name, target_room)
        return []

    # Encode labels
    le   = LabelEncoder()
    # Fit on dev labels; val labels must be a subset
    le.fit(y_dev)
    y_dev_enc = le.transform(y_dev)

    # Val labels might include unseen activities (in joint mode); remap safely
    known = set(le.classes_)
    mask  = np.array([lbl in known for lbl in y_val], dtype=bool)
    if not mask.all():
        n_unknown = (~mask).sum()
        log.warning(
            "[%s] room=%s: %d val samples have unseen activity labels — excluded from scoring.",
            name, target_room, n_unknown,
        )
    X_val_filt = X_val[mask]
    y_val_filt = le.transform(y_val[mask])

    if X_val_filt.shape[0] == 0:
        log.warning("[%s] No valid val samples after label filter (room=%s).", name, target_room)
        return []

    # Scale features: fit on dev, apply to both
    scaler   = StandardScaler()
    X_dev_sc = scaler.fit_transform(X_dev)
    X_val_sc = scaler.transform(X_val_filt)

    # Fit
    import time
    t0 = time.perf_counter()
    clf.fit(X_dev_sc, y_dev_enc)
    t_fit = time.perf_counter() - t0

    # Predict
    y_pred = clf.predict(X_val_sc)

    # Score
    scores = _score(y_val_filt, y_pred)
    log.info(
        "[%s] room=%-5s  macro_f1=%.4f  bal_acc=%.4f  dev=%d  val=%d  fit=%.1fs",
        name, target_room,
        scores["macro_f1"], scores["balanced_accuracy"],
        X_dev.shape[0], X_val_filt.shape[0],
        t_fit,
    )

    rows = []
    for metric, value in scores.items():
        rows.append({
            "experiment":   name,
            "target_room":  target_room,
            "K":            -1,
            "metric":       metric,
            "value":        f"{value:.6f}",
        })
    return rows


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(
    index_csv: Path,
    windows_dir: Path,
    packet_rate: float,
    reports_dir: Path,
    out_csv: Optional[Path] = None,
    n_jobs: int = 1,
) -> Path:
    """Run the full baseline HAR sweep and write results CSV.

    Parameters
    ----------
    index_csv:
        Path to ``derived/windows_index.csv``.
    windows_dir:
        Path to ``derived/windows/`` directory.
    packet_rate:
        CSI packet rate in Hz.
    reports_dir:
        Path to ``reports/`` directory (created if absent).
    out_csv:
        Override output path.  Defaults to ``reports_dir/baseline_har_results.csv``.
    n_jobs:
        Parallelism for RF.

    Returns
    -------
    Path to the written CSV.
    """
    import pandas as pd

    if out_csv is None:
        out_csv = reports_dir / "baseline_har_results.csv"

    log.info("Loading windows index: %s", index_csv)
    index_df = pd.read_csv(str(index_csv))
    log.info("Total windows: %d (dev=%d, val=%d)",
             len(index_df),
             (index_df["split"] == "dev").sum(),
             (index_df["split"] == "val").sum())

    rooms = sorted(index_df["room"].unique())
    log.info("Rooms in dataset: %s", rooms)

    # --- Load features for all rooms, both splits ---
    log.info("Extracting features from dev windows ...")
    X_dev_all, y_dev_all, _ = _load_windows_for_split(
        index_df, "dev", windows_dir, packet_rate, room=None
    )

    log.info("Extracting features from val windows ...")
    X_val_all, y_val_all, rooms_val_all = _load_windows_for_split(
        index_df, "val", windows_dir, packet_rate, room=None
    )

    # Per-room dev features (re-load to avoid alignment issues)
    dev_by_room: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    val_by_room: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for room in rooms:
        log.info("Loading room-%s dev features ...", room)
        X_r, y_r, _ = _load_windows_for_split(
            index_df, "dev", windows_dir, packet_rate, room=room
        )
        dev_by_room[room] = (X_r, y_r)

        log.info("Loading room-%s val features ...", room)
        X_rv, y_rv, _ = _load_windows_for_split(
            index_df, "val", windows_dir, packet_rate, room=room
        )
        val_by_room[room] = (X_rv, y_rv)

    # --- Sweep classifiers ---
    classifiers = _make_classifiers(n_jobs=n_jobs)
    all_results: List[dict] = []

    for clf_name, clf_proto in classifiers.items():
        import copy

        # ---- Per-room ----
        for room in rooms:
            X_dev_r, y_dev_r = dev_by_room[room]
            X_val_r, y_val_r = val_by_room[room]
            log.info("--- %s  room=%s ---", clf_name, room)
            rows = _run_one_room(
                name=clf_name,
                clf=copy.deepcopy(clf_proto),
                X_dev=X_dev_r,
                y_dev=y_dev_r,
                X_val=X_val_r,
                y_val=y_val_r,
                target_room=room,
            )
            all_results.extend(rows)

        # ---- Joint (all rooms together) ----
        log.info("--- %s  room=joint ---", clf_name)
        rows = _run_one_room(
            name=clf_name,
            clf=copy.deepcopy(clf_proto),
            X_dev=X_dev_all,
            y_dev=y_dev_all,
            X_val=X_val_all,
            y_val=y_val_all,
            target_room="joint",
        )
        all_results.extend(rows)

    # --- Write CSV ---
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _CSV_COLS = ["experiment", "target_room", "K", "metric", "value"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    log.info("Results written -> %s  (%d rows)", out_csv, len(all_results))
    _print_summary(all_results)
    return out_csv


def _print_summary(results: List[dict]) -> None:
    """Pretty-print a summary table to stdout."""
    print("\n" + "=" * 72)
    print("Baseline HAR results")
    print("=" * 72)
    print(f"{'Experiment':<12} {'Room':<8} {'K':>4}  {'Metric':<22} {'Value':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['experiment']:<12} {r['target_room']:<8} {r['K']:>4}  "
            f"{r['metric']:<22} {float(r['value']):>8.4f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke(args: argparse.Namespace) -> None:
    """End-to-end smoke test on tiny synthetic data (no real files needed)."""
    import tempfile
    import numpy as np

    print("[smoke] baseline_har smoke test ...")

    rng      = np.random.default_rng(42)
    T, N_SUB = 150, 256

    # Activities per room
    activities_A = ["fall", "sit", "stand", "walk", "sitdown", "standup"]
    activities_B = ["empty", "lying", "sit", "stand", "walk"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp        = Path(tmp)
        windows_dir = tmp / "windows"
        reports_dir = tmp / "reports"
        index_rows: List[dict] = []

        n_dev_per_class = 8
        n_val_per_class = 4

        def _write_window(split: str, window_id: str, room: str,
                           activity: str, arr: np.ndarray) -> dict:
            out_dir = windows_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / f"{window_id}.npy"), arr)
            return {
                "window_id":   window_id,
                "capture_id":  f"cap_{window_id}",
                "room":        room,
                "activity":    activity,
                "split":       split,
                "start":       0,
                "config_hash": "smoke00",
            }

        wi = 0
        for room, activities in [("A", activities_A), ("B", activities_B)]:
            for act in activities:
                for _ in range(n_dev_per_class):
                    arr = rng.uniform(0, 100, (T, N_SUB)).astype(np.float32)
                    index_rows.append(
                        _write_window("dev", f"w{wi:06d}", room, act, arr)
                    )
                    wi += 1
                for _ in range(n_val_per_class):
                    arr = rng.uniform(0, 100, (T, N_SUB)).astype(np.float32)
                    index_rows.append(
                        _write_window("val", f"w{wi:06d}", room, act, arr)
                    )
                    wi += 1

        # Write index CSV
        index_csv = tmp / "windows_index.csv"
        _CSV_COLS_IDX = ["window_id", "capture_id", "room", "activity", "split", "start", "config_hash"]
        with index_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=_CSV_COLS_IDX)
            wr.writeheader()
            for r in index_rows:
                wr.writerow(r)

        out_csv = run_evaluation(
            index_csv=index_csv,
            windows_dir=windows_dir,
            packet_rate=_DEFAULT_PACKET_RATE_HZ,
            reports_dir=reports_dir,
            out_csv=tmp / "smoke_results.csv",
            n_jobs=1,
        )

        # Validate output
        import csv as csv_mod
        with out_csv.open(encoding="utf-8") as fh:
            rows = list(csv_mod.DictReader(fh))

        assert len(rows) > 0, "No results written"
        expected_cols = {"experiment", "target_room", "K", "metric", "value"}
        assert set(rows[0].keys()) >= expected_cols, f"Missing columns: {set(rows[0].keys())}"

        experiments = {r["experiment"] for r in rows}
        rooms       = {r["target_room"] for r in rows}
        metrics     = {r["metric"]       for r in rows}
        assert experiments == {"LR", "SVM", "RF"}, f"Unexpected experiments: {experiments}"
        assert "A" in rooms and "B" in rooms and "joint" in rooms, f"Rooms: {rooms}"
        assert "macro_f1" in metrics and "balanced_accuracy" in metrics, f"Metrics: {metrics}"

        for r in rows:
            v = float(r["value"])
            assert 0.0 <= v <= 1.0, f"Value out of range: {v}"

        print(f"[smoke] {len(rows)} result rows written  OK")
        print(f"[smoke] experiments: {sorted(experiments)}  OK")
        print(f"[smoke] rooms: {sorted(rooms)}  OK")
        print(f"[smoke] metrics: {sorted(metrics)}  OK")
        print("[smoke] baseline_har PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Baseline HAR classifiers (LR, SVM, RF) trained on dev windows, "
            "evaluated on val windows."
        )
    )
    p.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment.yaml (default: configs/experiment.yaml)",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Run smoke test on synthetic data (no real files needed).",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for RandomForest (default: 1).",
    )
    p.add_argument(
        "--packet-rate", type=float, default=None,
        help="Override packet rate in Hz (default: infer from config).",
    )
    p.add_argument(
        "--out", default=None,
        help="Override output CSV path (default: reports/baseline_har_results.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        _run_smoke(args)
        return

    from src.config import load_config

    cfg = load_config(args.config)

    index_csv   = Path(cfg.paths.windows_index_csv)
    windows_dir = Path(cfg.paths.windows_dir)
    reports_dir = Path(cfg.paths.reports_dir)

    if not index_csv.exists():
        log.error(
            "windows_index.csv not found: %s\n"
            "Run scripts/build_windows.py first.",
            index_csv,
        )
        sys.exit(1)

    if not windows_dir.exists():
        log.error(
            "Windows directory not found: %s\n"
            "Run scripts/build_windows.py first.",
            windows_dir,
        )
        sys.exit(1)

    out_csv = Path(args.out) if args.out else None
    rate    = args.packet_rate if args.packet_rate is not None else cfg_packet_rate(cfg)
    log.info("Packet rate: %.4f Hz", rate)

    run_evaluation(
        index_csv=index_csv,
        windows_dir=windows_dir,
        packet_rate=rate,
        reports_dir=reports_dir,
        out_csv=out_csv,
        n_jobs=args.workers,
    )


if __name__ == "__main__":
    main()
