"""
scripts/make_colab_zip.py
Build colab_code.zip containing ONLY source code + config + small derived files.
Windows are large and should be zipped separately or reused from existing zip.

Usage:
    python scripts/make_colab_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "colab_code.zip"

if OUT.exists():
    OUT.unlink()

n_files = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:

    for p in sorted((ROOT / "src").rglob("*.py")):
        zf.write(p, p.relative_to(ROOT)); n_files += 1

    for p in sorted((ROOT / "scripts").rglob("*.py")):
        zf.write(p, p.relative_to(ROOT)); n_files += 1

    cfg = ROOT / "configs" / "experiment.yaml"
    if cfg.exists():
        zf.write(cfg, cfg.relative_to(ROOT)); n_files += 1

    for name in ["manifest.csv", "windows_index.csv", "norm_stats.npz"]:
        p = ROOT / "derived" / name
        if p.exists():
            zf.write(p, p.relative_to(ROOT)); n_files += 1

print(f"Created: {OUT}  ({OUT.stat().st_size/1e3:.0f} KB, {n_files} files)")
print("Upload this to Google Drive alongside your existing derived_for_colab.zip")
