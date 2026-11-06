# CS60055 Ubiquitous Computing — Hackathon Challenge 2
## Environment-Aware, Label-Efficient Activity Recognition from WiFi CSI

Cross-room few-shot human activity recognition using WiFi Channel State Information (CSI).
Built for the [CSI Human Activity dataset](https://ieee-dataport.org/open-access/csi-human-activity)
(Schäfer et al., *Applied Sciences* 2021).

---

## Quick Start — Reproduce Task 3 Results

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd CSI_classifier
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt

# 2. Place data at the paths in configs/experiment.yaml
#    dataset/Experiment-2/...  (Room A .pcap files)
#    dataset/Experiment-3/...  (Room B .csv files)

# 3. Run the full pipeline (see RUNBOOK.md for details)
python scripts/build_intermediate.py --config configs/experiment.yaml
python scripts/build_windows.py --config configs/experiment.yaml
python scripts/fit_normalization.py --config configs/experiment.yaml
python src/room_model.py --config configs/experiment.yaml
python -m src.ssl_train --mode vicreg --run_name vicreg_aug_v4 \
    --cross_room_frac 0.5 --config configs/experiment.yaml
python scripts/sweep.py --config configs/experiment.yaml \
    --condition vicreg_aug_v4 --shared_only_align \
    --out reports/task3_v4_coral_shared_vicreg_aug.csv

# 4. Run a single demo episode (as used in live demo)
python run_episode.py --target_room B --K 5 --seed 42 \
    --condition vicreg_aug_v4 \
    --out reports/episode.csv \
    --predictions_out reports/predictions.csv
```

---

## Repository Structure

```
CSI_classifier/
├── src/                         # All library modules
│   ├── pcap_reader.py           # Nexmon PCAP parsing
│   ├── csi_decode.py            # Packed-CSI reconstruction
│   ├── amplitude.py             # Complex → amplitude
│   ├── cleaning.py              # Quality control
│   ├── splits.py                # Dev/val split construction
│   ├── windowing.py             # (150, 256) window extraction
│   ├── normalization.py         # log1p + per-room z-score
│   ├── augment.py               # CSI augmentations + SSLPairDataset
│   ├── encoder.py               # ResNet1DEncoder
│   ├── ssl_train.py             # VICReg + Masked MAE training
│   ├── room_model.py            # Task 1 room classifier
│   ├── probe.py                 # Task 2 linear probe
│   ├── fewshot.py               # Task 3 support sampling + prototypes
│   ├── align.py                 # CORAL domain alignment
│   ├── routing.py               # RoomRouter
│   └── metrics.py               # Evaluation metrics
├── scripts/
│   ├── build_intermediate.py    # Stage 1: PCAP/CSV → intermediate .npy
│   ├── build_windows.py         # Stage 2: intermediate → (150,256) windows
│   ├── fit_normalization.py     # Stage 3: fit normalizer on dev data
│   ├── task2_eval.py            # Task 2 linear probe evaluation
│   ├── sweep.py                 # Task 3 K-shot sweep
│   └── make_figures.py          # Regenerate all publication figures
├── configs/
│   └── experiment.yaml          # Single source of truth for all config
├── reports/
│   ├── technical_report.md      # Full technical report
│   └── figures/                 # All publication figures (PNG + PDF)
├── run_episode.py               # Entry point for single episode evaluation
├── requirements.txt             # Pinned Python dependencies
├── RUNBOOK.md                   # Full command reference
└── README.md                    # This file
```

---

## Environment

Python 3.10+. All dependencies pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.6.0`, `scikit-learn==1.6.1`, `scapy==2.6.1`,
`umap-learn==0.5.7`, `numpy==2.1.3`, `pandas==2.2.3`.

---

## Entry Point for Demo / Evaluation

The grader-facing entry point is `run_episode.py`:

```bash
python run_episode.py \
    --target_room B \        # A or B
    --K 5 \                  # shots per exclusive activity
    --seed 42 \
    --condition vicreg_aug_v4 \
    --config configs/experiment.yaml \
    --out reports/episode.csv \
    --predictions_out reports/predictions.csv
```

Output format (`predictions.csv`):

| window_id | room | label | confidence |
|-----------|------|-------|------------|
| B__sit__cap001__00000 | B | sit | 0.823 |
| ... | | | |

---

## Task Summary

| Task | Method | Key Result |
|------|--------|-----------|
| **Task 1** Room ID | Logistic Regression on per-subcarrier mean/std | **100% accuracy** |
| **Task 2** SSL Encoder | VICReg + cross-room pairs (`cross_room_frac=0.5`) | Joint F1: 0.87 (random), 0.70 (v1 aug) |
| **Task 3** Few-shot | NCM + shared-only CORAL alignment | A→B macro F1: **0.350** at K=20 |

---

## Reproducibility Notes

- All random seeds fixed: `global_seed: 42`, `fewshot_seeds: [0..9]`
- Validation split for Room B: 25 captures/class, `seed=42`, alphabetically sorted filenames
- Derived artifacts (`derived/`, `checkpoints/`) are gitignored but fully regenerable
- Run `python run_episode.py --smoke` to verify the pipeline without real data

---

## References

1. Schäfer et al. "Human Activity Recognition Using CSI Information with Nexmon." *Applied Sciences* 2021.
2. Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR 2022*.
3. Sun & Saenko. "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." *ECCV 2016*.
4. Snell et al. "Prototypical Networks for Few-shot Learning." *NeurIPS 2017*.
