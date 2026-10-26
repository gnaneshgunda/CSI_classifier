"""Build a tiny synthetic derived/ tree for gate-checking the pipeline."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import ManifestRow, write_manifest

derived = Path("derived")
inter_a = derived / "intermediate" / "A"
inter_b = derived / "intermediate" / "B"
for d in [inter_a, inter_b, derived / "windows" / "dev", derived / "windows" / "val"]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(42)
T, N = 200, 256

activities_a = ["fall", "sit", "sitdown", "stand", "standup", "walk"]
activities_b = ["empty", "lying", "sit", "stand", "walk"]

rows = []

for act in activities_a:
    for i, split in enumerate(["dev", "dev", "dev", "val"]):
        cid = f"A__{act}__s{i:03d}"
        arr = rng.uniform(0.1, 500.0, (T, N)).astype("float32")
        path = inter_a / f"{cid}.npy"
        np.save(str(path), arr)
        rows.append(ManifestRow(
            capture_id=cid, room="A", activity=act, split=split,
            source_file=str(path), n_packets_raw=T, n_packets_clean=T,
            intermediate_path=str(path),
        ))

for act in activities_b:
    for i, split in enumerate(["dev", "dev", "dev", "val"]):
        cid = f"B__{act}__s{i:03d}"
        arr = rng.uniform(0.1, 500.0, (T, N)).astype("float32")
        path = inter_b / f"{cid}.npy"
        np.save(str(path), arr)
        rows.append(ManifestRow(
            capture_id=cid, room="B", activity=act, split=split,
            source_file=str(path), n_packets_raw=T, n_packets_clean=T,
            intermediate_path=str(path),
        ))

write_manifest(rows, derived / "manifest.csv")
print(f"Manifest: {len(rows)} rows written to derived/manifest.csv")
print(f"  Room A: {sum(1 for r in rows if r.room=='A')} captures")
print(f"  Room B: {sum(1 for r in rows if r.room=='B')} captures")
print(f"  dev: {sum(1 for r in rows if r.split=='dev')}  val: {sum(1 for r in rows if r.split=='val')}")
