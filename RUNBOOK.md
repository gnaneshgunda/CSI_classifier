# RUNBOOK – CSI-HAR Pipeline

Quick reference for every runnable command in the repository.  
All commands are run from the **repository root** unless noted.

---

## 0. Environment setup

```bash
# Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 1. Verify config loads cleanly

```bash
python -c "from src.config import load_config; cfg = load_config('configs/experiment.yaml'); print('config_hash:', cfg.config_hash())"
```

**Expected output**
```
config_hash: <16 hex characters>
```

---

## 2. Run the full test suite

```bash
pytest tests/ -v
```

**Expected output (abridged)**
```
tests/test_config.py::TestLoadConfigValid::test_returns_experiment_config PASSED
tests/test_config.py::TestLoadConfigValid::test_paths_are_path_objects PASSED
...
tests/test_manifest.py::TestWriteReadRoundTrip::test_write_then_read_returns_same_data PASSED
...
============================= N passed in X.XXs ==============================
```

All tests must pass with **0 failures** before any PR is merged.

---

## 3. Run tests with coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## 4. Smoke test (end-to-end on tiny synthetic data, < 30 s)

The smoke flag bypasses real data ingestion and runs every pipeline stage on
randomly generated arrays that satisfy the `(150, 256) float32` contract.

```bash
python run_episode.py --smoke
```

**Expected output**
```
[smoke] Generating synthetic data ...
[smoke] Room A: 10 captures  Room B: 10 captures
[smoke] Manifest written  -> derived/manifest.csv
[smoke] Windows written   -> derived/windows/
[smoke] Task 1 room accuracy  : 1.00  (oracle, synthetic labels)
[smoke] Task 2 linear probe   : done  (random encoder baseline)
[smoke] Task 3 few-shot K=1   : done
[smoke] All smoke checks passed in <30 s
```

> **Note:** The smoke run writes to `derived/` (gitignored).  
> Delete `derived/` before running a real pipeline to avoid stale cache.

---

## 5. Inspect raw source files

```bash
# Full inspection (reads every pcap + csv; takes several minutes on ~1000 pcaps)
python scripts/inspect_data.py --config configs/experiment.yaml

# Quick smoke (synthetic data, no real files needed, <5 s)
python scripts/inspect_data.py --smoke
```

**Expected output (smoke)**
```
[smoke] Generating synthetic inspection data …
[smoke] Markdown written -> reports\data_notes.md
════════════════ Room A – Development (synthetic) ════════════════
  Activity         N   CSI med  CSI min  CSI max  <150  FPS est  Δt med ms
  ──────────────────────────────────────────────────────────────────────
  fall             5       254      195      311     0     33.4     29.92
  ...
════════════════ Room B – Standard CSV (synthetic) ════════════════
  ...
[smoke] inspect_data smoke PASSED ✓
```

**What the real run produces:**
- Console tables with per-activity statistics for both rooms
- `reports/data_notes.md` – Markdown summary with:
  - PCAP: CSI-candidate packet counts, inter-packet timing, payload-length histogram
  - CSV: row/column counts, NaN totals, header style classification, amplitude range
  - Per-activity capture-length summary (both rooms combined)
  - Flagged anomalous files (0-indexed headers, unexpected naming)

**Key findings from real data (Room B):**
- All standard files: 500 rows × 256 cols, `Subcarrier_1..256` headers
- `Walking_{N}.csv` files (209 total): `Subcarrier_1..256` headers except `Walking_0.csv` (0-indexed)
- 5 anomalous files with 0-indexed headers: `Empty_11.csv`, `Lying_0.csv`, `Sitting_0.csv`,
  `Standing_0.csv`, `Walking_0.csv` — all 500 rows × 256 cols, data appears valid

---

## 6. Build intermediate representations (real data)

```bash
# Build all captures (skips already-built .npy files)
python scripts/build_intermediate.py --config configs/experiment.yaml

# Force rebuild everything
python scripts/build_intermediate.py --config configs/experiment.yaml --force

# Process one room only
python scripts/build_intermediate.py --config configs/experiment.yaml --room A

# Use multiple workers (may fail on Windows due to scapy fork issues)
python scripts/build_intermediate.py --config configs/experiment.yaml --workers 4

# Quick smoke (synthetic data, no real files needed)
python scripts/build_intermediate.py --smoke
```

**Expected output (smoke)**
```
[smoke] Building synthetic intermediate data ...
[smoke] 3 captures processed
[smoke] Manifest verified: 3 rows
[smoke] Shape guard (256 subcarriers): OK
[smoke] All smoke checks passed ✓
```

**What the real run produces:**
- `derived/intermediate/A/{capture_id}.npy` — float32 (n_clean, 256) per Room-A capture
- `derived/intermediate/B/{capture_id}.npy` — float32 (n_clean, 256) per Room-B capture
- `derived/manifest.csv` — capture_id, room, activity, split, source_file, n_packets_raw, n_packets_clean, intermediate_path
- `derived/cleaning_report.csv` — per-capture rejection counts (non_finite, near_zero, consecutive_dup, gain_spike)

Pipeline per capture: pcap read → packed-float decode → amplitude → 4-rule cleaning → save .npy.
Room B: CSV read → cleaning → save .npy.
Shape `[1] != 256` causes immediate hard failure. Captures with fewer than `min_valid_packets` after cleaning print a WARNING.

---

## 7. Build windows

```bash
# Build all windows from intermediate captures
python scripts/build_windows.py --config configs/experiment.yaml

# Force rebuild
python scripts/build_windows.py --config configs/experiment.yaml --force

# Quick smoke (synthetic data, no real files needed)
python scripts/build_windows.py --smoke
```

**Expected output (smoke)**
```
[smoke] Testing windowing functions on synthetic data ...
[smoke] Canonical: shape (150, 256) OK
[smoke] Dev windows: 2 from 200-row capture OK
[smoke] Short capture: reflect-padded to (150, 256) OK
[smoke] Determinism: same input -> same output OK
[smoke] No cross-capture leakage OK
[smoke] Index CSV + .npy files verified OK
[smoke] All smoke checks passed ✓
```

**What the real run produces:**
- `derived/windows/dev/{capture_id}__{start:05d}.npy` — float32 (150, 256), sliding stride 50
- `derived/windows/val/{capture_id}__00000.npy` — float32 (150, 256), canonical central window only
- `derived/windows_index.csv` — window_id, capture_id, room, activity, split, start, config_hash

Validation captures get **one canonical window** (central 150 rows, reflect-padded if short).
Development captures get **sliding windows** (stride 50, tail-anchored to avoid data loss).
Windows never cross capture boundaries. Each window is traceable via `start` index + `capture_id`.

---

## 7. Task 1 – Room identification

```bash
python scripts/baseline_har.py --config configs/experiment.yaml --task room
```

---

## 8. Task 2 – Self-supervised encoder + linear probe

```bash
# Pre-train the encoder
python scripts/task2_eval.py --config configs/experiment.yaml --stage pretrain

# Evaluate frozen encoder with linear probe on validation set
python scripts/task2_eval.py --config configs/experiment.yaml --stage eval
```

---

## 9. Task 3 – Cross-room few-shot sweep

```bash
# Sweep K ∈ {1,5,10,20} for both A→B and B→A
python scripts/sweep.py --config configs/experiment.yaml
```

Results are written to `reports/task3_sweep.csv`.

---

## 10. Generate figures

```bash
python scripts/make_figures.py --config configs/experiment.yaml
```

Writes PNG files to `reports/figures/`:
- `label_efficiency_curve.png`
- `encoder_umap_by_room.png`
- `encoder_umap_by_activity.png`
- `room_accuracy_vs_window_length.png`
- `routing_contribution.png`
- `confusion_matrix_roomA_K5.png`
- `confusion_matrix_roomB_K5.png`

---

## 11. Run a single challenge episode (teaching-team reproducibility check)

```bash
python run_episode.py \
    --episode_dir <path/to/episode_captures> \
    --config configs/experiment.yaml \
    --output reports/episode_predictions.csv
```

**Output format** (`reports/episode_predictions.csv`)
```
window_id,room,label,confidence
ep_0000,A,fall,0.923
ep_0001,B,empty,0.871
...
```

---

## 12. Inspect raw source files

```bash
python scripts/inspect_data.py --config configs/experiment.yaml
```

Prints per-room packet/row counts, subcarrier dimensionality, and a sample
amplitude histogram without writing any derived files.

---

## Reproducibility checklist

| Step | Command | Deterministic? |
|------|---------|---------------|
| Intermediate build | `build_intermediate.py` | Yes (no randomness) |
| Window build | `build_windows.py` | Yes (deterministic stride) |
| Val split (Room B) | inside `build_intermediate.py` | Yes (seed 42) |
| SSL pre-training | `task2_eval.py --stage pretrain` | Yes (global seed 42) |
| Few-shot sweep | `sweep.py` | Yes (n_seeds=10 fixed seeds) |

All seeds are set in `configs/experiment.yaml` and propagated via
`src/config.py`.  Never hard-code seeds in scripts.
