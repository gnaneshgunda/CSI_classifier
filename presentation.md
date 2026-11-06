# Cross-Room Few-Shot Adaptation
## CS60055 Ubiquitous Computing | Hackathon Challenge 2
### Environment-Aware, Label-Efficient Activity Recognition from WiFi CSI

---

## What We Built: Complete Python Pipeline

- **1. Data Ingestion & CSI Reconstruction**
  — PCAP decode → complex CSI → amplitude → common 256-column format
  — No MATLAB, no source authors' code

- **2. Recorded-Environment Fingerprint (Task 1)**
  — Predict Room A vs Room B from a (150 × 256) CSI window
  — Routing component for normalization and few-shot adaptation

- **3. Self-Supervised CSI Encoder (Task 2)**
  — Learn representations **without** activity or room labels
  — VICReg (contrastive) and Masked Auto-Encoding

- **4. Cross-Room Few-Shot Adaptation (Task 3)**
  — Logistic regression head: source all labels + K target exclusive shots
  — Shared-class-conditioned CORAL domain alignment
  — Shared activities (sit/stand/walk) transferred with **zero** target labels

---

## Step 1: PCAP Decoding & Data Ingestion

**Room A — raw PCAP (Nexmon BCM4366c0 @ 80 MHz)**
- Parse UDP payloads with `scapy`; filter by port 5500, magic bytes `0x1111`
- Decode packed 32-bit little-endian integers → 256 complex subcarriers
- Compute amplitude: `|z| = sqrt(re² + im²)` → `(N_packets, 256)` float32

**Room B — pre-extracted amplitude CSV**
- Direct `pandas` read; rename columns to `subcarrier_001…256`
- Handle two header formats (5 anomalous files with 0-indexed headers)

**Output:** one `.npy` per capture in `derived/intermediate/{A,B}/`

**Code:** `src/pcap_reader.py`, `src/csi_decode.py`, `src/room_b_loader.py`

---

## Step 2: Quality Control & Cleaning

Four sequential rules — each row that fails is dropped, never interpolated:

| Rule | Condition | Reason |
|------|-----------|--------|
| Non-finite | >10% NaN/Inf per packet | Hardware corruption |
| Near-zero energy | sum(amp²) ≤ 0 | Dead / silent packet |
| Consecutive duplicate | Bit-for-bit identical to previous | Firmware stutter |
| Gain spike | Robust z-score > 6σ | AGC transient |

Entire captures with < 10 clean packets are rejected.
Amplitude values > 5000 are **clipped** (preserves temporal structure).

**Output:** cleaned `.npy` + `derived/cleaning_report.csv`
**Code:** `src/cleaning.py`

---

## Step 3: Windowing Strategy

**Official model input:** `(150, 256)` float32 — 150 consecutive packets × 256 subcarriers

| Split | Strategy | Windows per capture |
|-------|----------|-------------------|
| Development | Sliding windows, stride = 50 | Multiple (maximises training data) |
| Validation | Single canonical window (central crop; reflect-pad if short) | Exactly 1 |

**Critical rule:** original capture = atomic unit. All windows from one capture stay in the same split. No cross-capture windows. No leakage.

**Dataset totals:** 10,937 windows — 10,662 dev + 275 val (150 Room A + 125 Room B)

**Code:** `src/windowing.py`, `scripts/build_windows.py`

---

## Step 4: Normalization

Per-window log1p z-score — no fitting, no leakage:

```python
w = np.log1p(np.clip(window, 0, None))    # log-compress amplitude
norm = (w - w.mean()) / (w.std() + 1e-8)  # per-window z-score
```

**Why per-window:**
- Different gain states across captures require local normalization
- No room statistics needed → no val leakage risk
- Exactly matches the normalization used inside the encoder inference path

**Code:** `src/normalization.py`

---

## Step 5: Room Classifier — Task 1

**512-d handcrafted features:**
```
per-subcarrier mean over time → (256,)
per-subcarrier std  over time → (256,)
→ log1p → z-score → (512,)
```

**Two classifiers trained on dev windows only (GroupKFold, no leakage):**
- Logistic Regression — `C=1.0`, `lbfgs`
- MLP — `[512 → 256 → 128 → 2]`, ReLU, early stopping

| Model | Val Accuracy | Bal. Accuracy |
|-------|-------------|--------------|
| LR  | **100.0%** | **100.0%** |
| MLP | **100.0%** | **100.0%** |

Perfect separation at **all** window lengths T ∈ {10, 25, 50, 100, 150}.

Saved as `checkpoints/room_model.pkl` — used for routing in Tasks 2 & 3.

**Code:** `src/room_model.py`

---

## Step 6: SSL Encoder Architecture

**ResNet1D-Lite** — 321K parameters, input `(B, 150, 256)` → output `(B, 128)`

```
Stem:    Conv1d(256→64, k=7) + GroupNorm + GELU
Block 1: ResBlock1D(64→64,   dilation=1, stride=1)
Block 2: ResBlock1D(64→128,  dilation=2, stride=2)  ← downsample
Block 3: ResBlock1D(128→128, dilation=4, stride=1)
Pool:    Attentive pooling (learned per-timestep weights)
Proj:    Linear(128→128)
```

**Design rationale:**
- 1D convolutions: subcarriers as channels, time as signal axis
- Dilated convolutions: multi-scale temporal pattern capture
- GroupNorm: stable without large batches
- Attentive pooling: length-agnostic, weighted temporal fusion

**Code:** `src/encoder.py`

---

## Step 7: SSL Training — VICReg

**VICReg loss (Bardes et al., ICLR 2022):**
```
L = λ·Invariance + μ·Variance + ν·Covariance
    MSE between    Hinge: std ≥ γ   Off-diagonal
    two views      per dimension    covariance → 0
```

**Key insight — the γ problem:**
- γ=1.0 (default): PyTorch init already gives std≈1.5 → `relu(1.0-1.5)=0`
- **Variance term was ZERO from epoch 1 across all v1–v4 runs**
- Encoder only trained via covariance decorrelation → directional bias

**v5 fix: γ=2.5**
- `relu(2.5-1.5)=1.0` → variance loss fires from epoch 1
- `var=1.498` throughout training vs `var=0.000` before
- `eff_rank` improved: 22 (v4) → 37 (v5)

**Code:** `src/ssl_train.py`

---

## Step 8: SSL Training — Cross-Room Pairs (v4)

**Original problem:** all SSL pairs were same-capture windows.
The invariance loss had a trivial solution — two similar windows from the same recording are easy to match. The encoder learned **capture-specific features**, not activity features.

**Fix (v4):** 50% of SSL pairs are cross-room — one Room A window + one Room B window, **no labels used**.

```
inv at ep50:  v1 = 0.0005  →  v4 = 0.009  (18× higher)
```

The invariance loss now has to work across rooms. The only consistent strategy for the encoder is to suppress room-specific features (which vary between all A/B pairs) and preserve whatever is consistent — activity-correlated patterns.

**This is spec-compliant:** no activity labels are used in pairing. Room membership (from `windows_index.csv`) is structural metadata, not a label.

**Code:** `src/augment.py` — `SSLPairDataset(cross_room_frac=0.5)`

---

## Step 9: Linear Probe Evaluation — Task 2

Freeze encoder → train Logistic Regression on dev embeddings → evaluate on val.

| Encoder | Joint F1 | Room A F1 | Room B F1 |
|---------|----------|-----------|-----------|
| Handcrafted features | 0.642 | 0.953 | 1.000 |
| Random encoder | 0.873 | 0.893 | 0.893 |
| VICReg + aug (v1) | 0.700 | 0.651 | 0.912 |
| VICReg no-aug | 0.857 | 0.814 | 0.943 |
| **Masked MAE** | **0.853** | **0.819** | **0.968** |

**Key findings:**
- Random encoder (0.873) > VICReg+aug (0.700): architecture has strong inductive bias; strong augmentation hurt
- Masked MAE best per-room: reconstruction forces structural CSI learning
- All SSL encoders beat handcrafted on **joint** evaluation — cross-room, SSL wins

**Code:** `scripts/task2_eval.py`

---

## Step 10: CORAL Domain Alignment

Without alignment: Room A and Room B embeddings live in **non-overlapping regions** → Shared F1 = 0.000.

**CORAL (Sun & Saenko, 2016):** align second-order statistics
```
z_B' = (z_B − μ_B) @ Σ_B^{-½} @ Σ_A^{½} − μ_global
```
Whitens Room B then re-colours to match Room A covariance.

**Key improvement — shared-class-conditioned CORAL:**
Fit covariance matrices using **only shared-activity windows** (sit/stand/walk).
Targets the subspace that matters for zero-shot transfer, not the full distribution.

| Alignment | A→B shared_f1 | B→A shared_f1 |
|-----------|-------------|-------------|
| None | 0.067 | 0.021 |
| Global CORAL | 0.161 | 0.097 |
| **Shared-only CORAL** | **0.279** | **0.106** |

Implemented with eigen-decomposition + regularisation (ε=1e-5) for numerical stability.

**Code:** `src/align.py`

---

## Step 11: Few-Shot Classification Head — Task 3

### Why not Nearest-Class Mean (NCM)?

Debug revealed: all cosine similarities between any embedding and any prototype ≈ **0.97**.
Softmax over near-equal values → uniform 0.20 probabilities for all 5 classes.

**Root cause:** VICReg with inactive variance term (γ=1.0) produces directionally biased embeddings — all vectors point roughly the same direction even though their exact 128-d coordinates differ.

### Logistic Regression head

```
X_train = [all source dev embeddings] + [K-shot target exclusive embeddings]
y_train = [source labels]             + [target exclusive labels]
→ StandardScaler → LogisticRegression(C=1.0, max_iter=1000)
→ Predict target validation windows (masked to valid target room classes)
```

**Why LR works when NCM fails:**
- NCM uses cosine similarity (direction only) → all ~0.97 → uniform output
- LR uses full 128-d coordinates → learns linear boundary in the 128-d space
- Proven feasible by Task 2: LR on same-room embeddings = 0.91 F1

**Code:** `run_episode.py`, `scripts/sweep.py`

---

## Task 3 Results — A→B (Room B as target)

Source: Room A (all dev labels) | Target: Room B (K-shot empty + lying only)

| K | Macro F1 | Bal. Acc | Excl. F1 | Shared F1 |
|---|----------|---------|---------|---------|
| 1 | 0.175 ± 0.053 | 0.174 ± 0.035 | 0.204 ± 0.105 | 0.155 ± 0.039 |
| 5 | 0.312 ± 0.049 | 0.306 ± 0.050 | 0.468 ± 0.081 | 0.208 ± 0.037 |
| 10 | 0.408 ± 0.053 | 0.413 ± 0.047 | 0.644 ± 0.071 | 0.251 ± 0.043 |
| **20** | **0.422 ± 0.034** | **0.432 ± 0.029** | **0.693 ± 0.040** | **0.241 ± 0.033** |

**Exclusive_f1 at K=20 = 0.693** — LR successfully learns empty/lying from K shots.
**Shared_f1 plateaus ~0.21–0.25** — zero-shot CORAL transfer is the ceiling.

---

## Task 3 Results — B→A (Room A as target)

Source: Room B (all dev labels) | Target: Room A (K-shot fall + sitdown + standup only)

| K | Macro F1 | Bal. Acc | Excl. F1 | Shared F1 |
|---|----------|---------|---------|---------|
| 1 | 0.083 ± 0.002 | 0.128 ± 0.003 | 0.000 ± 0.000 | 0.167 ± 0.004 |
| 5 | 0.112 ± 0.021 | 0.134 ± 0.016 | 0.073 ± 0.040 | 0.151 ± 0.013 |
| 10 | 0.167 ± 0.035 | 0.170 ± 0.031 | 0.208 ± 0.057 | 0.125 ± 0.024 |
| **20** | **0.208 ± 0.029** | **0.235 ± 0.033** | **0.330 ± 0.057** | 0.086 ± 0.009 |

**B→A harder than A→B:**
- Room A has 3× fewer shared-activity dev windows (1,219 vs 3,955) → noisier CORAL
- 3 exclusive classes vs 2 — LR needs more shots per class to learn boundaries
- K=1 excl_f1=0.000 — LR cannot learn 3 classes from 1 example each (NCM at least has a centroid here)

---

## Task 3: What Each Fix Contributed

| Change | A→B macro K=20 | A→B excl K=20 | A→B shared K=5 |
|--------|--------------|-------------|--------------|
| No alignment (baseline) | 0.256 | 0.499 | 0.067 |
| + Global CORAL | 0.176 | 0.193 | 0.161 |
| + Shared-only CORAL | 0.296 | 0.350 | **0.279** |
| + v4 encoder (cross-room pairs) | 0.350 | 0.458 | 0.279 |
| **+ LR head (final)** | **0.422** | **0.693** | 0.208 |

Each change targeted a diagnosed failure:
1. **Global CORAL** → brought shared_f1 from 0 to 0.161
2. **Shared-only CORAL** → fit on the relevant subspace (+73% shared_f1)
3. **Cross-room pairs** → encoder learns room-invariant features (inv 18× higher)
4. **LR head** → bypasses cosine collapse in NCM, uses full 128-d geometry

---

## Encoder Version Comparison

| Encoder | Key change | eff_rank | A→B macro K=20 | A→B excl K=20 |
|---------|-----------|---------|--------------|-------------|
| v1 (vicreg_aug) | Baseline VICReg, same-capture | 28 | 0.350 | 0.458 |
| v2 | Stronger augmentation | 31 | 0.123 | 0.281 |
| v3 | Per-capture norm | 25 | 0.155 | 0.352 |
| **v4** | **Cross-room pairs** | **22** | **0.422** | **0.693** |
| v5 | γ=2.5, cross-room | 37 | 0.398 | 0.739 |

v2 and v3 made things **worse** despite higher eff_rank — higher eff_rank ≠ better cross-room transfer.
v4 is the best overall. v5 improves exclusive_f1 slightly but hurts shared_f1 and B→A.

---

## Routing Contribution (Task 2 — Environment-Aware Normalisation)

| Routing Mode | Room A F1 | Room B F1 | Overall |
|--------------|-----------|-----------|---------|
| None (joint model) | 0.853 | 0.853 | 0.853 |
| Oracle (true label) | 0.819 | **0.968** | 0.893 |
| Predicted (classifier) | 0.819 | **0.968** | 0.893 |

100% room accuracy → predicted routing = oracle routing exactly.
Room B benefits +11.5% from room-specific normalisation.

---

## Struggles We Faced

### 1. PCAP Decoding Complexity (~2 days)
No Python library for Nexmon CSI. Reverse-engineered BCM4366c0 binary format.
Debugged byte-order, endianness, packet validity filtering from scratch.

### 2. VICReg γ=1.0 — Variance Term Never Fired
`var=0.000` every epoch across v1–v4. Encoder converged with all embeddings pointing the same direction.
NCM output uniform 0.20 probabilities. Diagnosed late; fixed with γ=2.5 in v5.

### 3. Encoder Retraining Regressions (v2, v3)
- v2 (stronger augmentation): higher eff_rank but **worse** cross-room F1
- v3 (per-capture norm): train/inference normalization mismatch → shared_f1=0.000
- Lesson: eff_rank is not a proxy for room-invariance

### 4. Few-Shot Support Set — Window vs Capture Counting
Initially counted windows as shots (one capture = 10+ windows = inflated K).
Redesigned: capture is the atomic unit. Results dropped but became honest.

### 5. NCM Cosine Collapse
Cosine similarity ~0.97 between all class pairs → uniform softmax output.
Switching to LR: A→B excl_f1 0.193 (NCM) → **0.693** (LR).

### 6. CORAL Numerical Instability
Singular matrix errors during covariance inversion.
Fixed: ε=1e-5 regularisation + eigen-decomposition for matrix square root.

---

## What We Learned

**About Room Fingerprinting:**
- Per-subcarrier mean+std is a near-perfect environment descriptor
- 100% accuracy even at T=10 — multipath patterns are highly distinctive

**About SSL for CSI:**
- Architecture inductive bias matters: random encoder (0.873 F1) > trained VICReg+aug (0.700)
- Same-capture pairs = trivial SSL task → cross-room pairs create real invariance pressure
- Masked reconstruction > contrastive for within-room activity recognition
- γ matters: VICReg variance term must actually fire to have any effect on directional spread

**About Cross-Room Transfer:**
- CORAL is necessary but not sufficient for shared activity transfer
- Shared-only CORAL significantly better than global (+73% shared_f1)
- LR uses full embedding geometry; NCM only uses direction — critical difference at high cosine similarity
- The gap: within-room LR = 0.91 F1 vs cross-room LR = 0.42 — this gap is the open research problem

---

## Known Limitations

| Limitation | Impact |
|-----------|--------|
| Rooms recorded in separate sessions | 100% room accuracy may exploit hardware state, not just environment |
| Room A: activity-specific spatial zones | Classifier may recognise location, not activity |
| Only two rooms | Cannot claim generalisation to unseen environments |
| No subject-disjoint guarantee | Results reflect capture-level disjointness only |
| CORAL needs target dev embeddings to fit | Not truly deployment-ready without setup data |
| Room A has 3× fewer shared-activity windows | B→A CORAL noisier → consistently weaker than A→B |
| v5 γ fix came late | Variance term firing but std_min still below γ=2.5 after 50 epochs |

---

## Summary of All Results

| Task | Metric | Value |
|------|--------|-------|
| **Task 1** Room ID | Validation accuracy | **1.000** |
| **Task 2** masked MAE probe | Joint macro-F1 | **0.853** |
| **Task 2** masked MAE probe | Room B F1 | **0.968** |
| **Task 3** A→B K=20 | Macro F1 | **0.422** |
| **Task 3** A→B K=20 | Exclusive F1 (empty, lying) | **0.693** |
| **Task 3** A→B K=5 | Shared F1 (sit, stand, walk) | **0.208** |
| **Task 3** B→A K=20 | Macro F1 | **0.208** |
| **Task 3** B→A K=20 | Exclusive F1 (fall, sitdown, standup) | **0.330** |

All results: **v4 encoder + shared-only CORAL + LR head**, averaged over 10 seeds.

---

## Reproducibility & Code Structure

```
CSI_classifier/
├── src/                     # Core modules
│   ├── pcap_reader.py       # Nexmon PCAP parsing
│   ├── csi_decode.py        # Packed-CSI reconstruction
│   ├── cleaning.py          # Quality control (4 rules)
│   ├── windowing.py         # (150, 256) window extraction
│   ├── normalization.py     # log1p + per-window z-score
│   ├── encoder.py           # ResNet1DEncoder
│   ├── ssl_train.py         # VICReg + Masked MAE
│   ├── room_model.py        # Task 1 room classifier
│   ├── align.py             # CORAL alignment
│   └── fewshot.py           # Support sampling
├── scripts/
│   ├── build_intermediate.py
│   ├── build_windows.py
│   ├── task2_eval.py
│   └── sweep.py             # Task 3 K-shot sweep
├── configs/experiment.yaml  # All hyperparameters — single source of truth
├── run_episode.py           # Entry point for single episode
└── app.py                   # Streamlit demo UI
```

**Quick demo:**
```bash
python run_episode.py --target_room B --K 5 --seed 42 \
    --condition vicreg_aug_v4 \
    --out reports/episode.csv \
    --predictions_out reports/predictions.csv
```

All seeds fixed in `configs/experiment.yaml`. Fully reproducible.

---

## AI Use Disclosure

**Tools used:** Kiro IDE (Claude) for development assistance

**AI-assisted:**
- Code generation and debugging across all pipeline components
- Root cause diagnosis (var=0 issue, NCM cosine collapse, CORAL stale stats)
- Architecture suggestions (shared-only CORAL, cross-room pairs, γ fix, LR head)
- Report and presentation writing

**Human validation:**
- All code reviewed, understood, and tested by team members
- Every team member can explain every preprocessing and modelling stage
- Results verified through smoke tests, gate checks, and cross-validation
- All technical claims verified against actual logs and implementation

---

## Conclusion

**Complete Python pipeline** from raw Nexmon PCAP to cross-room activity predictions.

**What works:**
- Perfect room fingerprinting (100%)
- Strong within-room activity recognition (85–97% F1)
- Exclusive activity adaptation scales reliably with K (excl_f1 = 0.693 at K=20)

**The open problem:**
- Shared activity zero-shot transfer plateaus at ~0.21–0.25 shared F1
- Within-room: 0.91 F1. Cross-room: 0.42 F1. That 0.49 gap is the research frontier.
- Encoder learns room-correlated features despite cross-room SSL training

**Critical insight:**
> Closing the within-room to cross-room gap requires learning activity features
> that are **invariant to the physical environment** —
> not yet solved in WiFi-based HAR.

---

## Questions?

**CS60055 Ubiquitous Computing | Hackathon Challenge 2**

*Cross-Room Few-Shot Adaptation —
Environment-Aware, Label-Efficient Activity Recognition from WiFi CSI*
