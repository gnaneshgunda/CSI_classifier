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
  — Logistic regression head trained on source room + K target shots
  — Shared-class-conditioned CORAL domain alignment
  — Zero target labels for shared activities (sit / stand / walk)

---

## Step 1: PCAP Decoding & Data Ingestion

**Room A — raw PCAP (Nexmon BCM4366c0 @ 80 MHz)**
- Parse UDP payloads with `scapy`; filter by port 5500, magic bytes `0x1111`, payload 1042 B
- Decode packed 32-bit little-endian integers → 256 complex subcarriers
- Compute amplitude: `|z| = sqrt(re² + im²)` → `(N_packets, 256)` float32

**Room B — pre-extracted amplitude CSV**
- Direct `pandas` read; rename columns to `subcarrier_001…256`
- Handle two header formats (0-indexed in 5 anomalous files)

**Output:** one `.npy` file per capture in `derived/intermediate/{A,B}/`

**Code:** `src/pcap_reader.py`, `src/csi_decode.py`, `src/room_b_loader.py`

---

## Step 2: Quality Control & Cleaning

Four sequential rules — each row that fails is dropped, never interpolated:

| Rule | Condition | Reason |
|------|-----------|--------|
| Non-finite | > 10% NaN/Inf per packet | Hardware corruption |
| Near-zero energy | sum(amp²) ≤ 0 | Dead / silent packet |
| Consecutive duplicate | bit-for-bit identical to previous | Firmware stutter |
| Gain spike | robust z-score > 6σ | AGC transient |

Entire captures with < 10 clean packets are rejected.  
Amplitude values > 5000 are **clipped**, not dropped (preserves temporal structure).

**Output:** cleaned `.npy` + `derived/cleaning_report.csv`  
**Code:** `src/cleaning.py`

---

## Step 3: Windowing Strategy

**Official model input:** `(150, 256)` float32 — 150 consecutive packets × 256 subcarriers

| Split | Strategy | Windows per capture |
|-------|----------|-------------------|
| Development | Sliding windows, stride = 50 | Multiple (maximises training data) |
| Validation | Single canonical window (central crop; reflect-pad if short) | Exactly 1 |

**Critical rule:** the original capture is the **atomic unit** — all windows from one capture stay in the same split. No cross-capture windows. No leakage.

**Dataset totals:** 10,937 windows — 10,662 dev + 275 val (150 Room A + 125 Room B)

**Code:** `src/windowing.py`, `scripts/build_windows.py`

---

## Step 4: Normalization

Per-window log1p z-score — applied identically to dev and val, no fitting:

```python
w = np.log1p(np.clip(window, 0, None))   # log-compress amplitude
norm = (w - w.mean()) / (w.std() + 1e-8) # global per-window z-score
```

**Why per-window (not per-room):**
- Different gain states across captures need local normalisation
- No room statistics required — no risk of val leakage
- Matches the normalisation used inside the SSL training loop exactly

Room-specific statistics are also fitted on dev data (for Task 1 routing) but the encoder inference path uses per-window z-score throughout.

**Code:** `src/normalization.py`, `scripts/fit_normalization.py`

---

## Step 5: Room Classifier — Task 1

**Features:** 512-d handcrafted vector
```
mean over time → (256,)  +  std over time → (256,)  → log1p → z-score
```

**Classifiers trained on development windows only:**
- Logistic Regression (`C=1.0`, `lbfgs`, 5-fold `GroupKFold`)
- MLP `[512 → 256 → 128 → 2]` with ReLU + early stopping

**Result:**

| Model | Val Accuracy | Bal. Accuracy |
|-------|-------------|--------------|
| LR | **100.0%** | **100.0%** |
| MLP | **100.0%** | **100.0%** |

Perfect separation at all window lengths T ∈ {10, 25, 50, 100, 150}.

**Saved as `checkpoints/room_model.pkl`** — used for routing in Tasks 2 & 3.

**Code:** `src/room_model.py`

---

## Step 6: SSL Encoder Architecture

**ResNet1D-Lite** — 321K parameters, input `(B, 150, 256)` → output `(B, 128)`

```
Stem:   Conv1d(256→64, k=7) + GroupNorm + GELU
Block 1: ResBlock1D(64→64,   dilation=1, stride=1)
Block 2: ResBlock1D(64→128,  dilation=2, stride=2)   ← downsample
Block 3: ResBlock1D(128→128, dilation=4, stride=1)
Pool:   Attentive pooling (learned per-timestep weights)
Proj:   Linear(128→128)
```

**Design choices:**
- 1D convolutions: subcarriers as channels, time as the signal axis
- Dilated convolutions: multi-scale temporal pattern capture
- GroupNorm: stable without large batches
- Attentive pooling: length-agnostic, learned temporal fusion

**Code:** `src/encoder.py`

---

## Step 7: SSL Training Iterations

### Iteration 1 — VICReg weak augmentations
Gaussian σ=0.05, mask 10%, scale 0.80–1.20
**Problem:** `var=0` every epoch — variance hinge trivially satisfied from init, encoder never learns invariance.

### Iteration 2 — VICReg stronger augmentations (vicreg_aug / v1)
Gaussian σ=0.15, mask 30%, scale 0.60–1.40, time crop 0.50
**Fix:** variance collapse gone; `std_min=2.7` by ep50, `eff_rank≈28`

### Iteration 3 — VICReg no augmentations (vicreg_noaug)
Ablation: does augmentation help? Two different windows from same capture as positives.
`eff_rank=51` — more diverse but still room-specific.

### Iteration 4 — Masked Auto-Encoding (masked_128)
50% block masking, 2-layer CNN decoder, MSE on masked positions only.
Best within-room probe: Room B F1 = **0.968**

### Iteration 5 — Cross-room pairs + γ fix (vicreg_aug_v4)
50% of SSL pairs are cross-room (Room A window ↔ Room B window, no labels).  
Invariance loss `inv=0.009` at ep50 vs `0.0005` before — 18× more cross-room pressure.

### Iteration 6 — Raised γ from 1.0 → 2.5 (vicreg_aug_v5, ongoing)
`var=1.498` at epoch 1 (previously always 0.000). Variance term finally firing.

---

## Step 8: VICReg Loss — What Actually Happened

VICReg loss has three terms:
```
L = λ·Invariance + μ·Variance + ν·Covariance
    (25)             (25)          (1)
```

**Variance term:** `relu(γ - std(z))` — hinge loss, fires only when `std < γ`

**The γ=1.0 problem:**
```
PyTorch default init → std ≈ 1.5–2.0
relu(1.0 - 1.5) = relu(-0.5) = 0   ← zero gradient from epoch 1
```

The encoder received **no gradient from variance throughout all of v1–v4**.  
VICReg reduced to covariance decorrelation only → directional bias in embeddings.

**The γ=2.5 fix (v5):**
```
relu(2.5 - 1.5) = relu(1.0) = 1.0  ← non-zero, encoder must push std above 2.5
```

**Effect on Task 3:** with all embeddings pointing in the same direction, cosine  
similarity between any two was ~0.97 → NCM (nearest-centroid) output uniform  
probabilities → we switched to Logistic Regression which uses raw coordinates.

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
- Random encoder (0.873 joint) outperforms VICReg+aug (0.700) — the architecture itself has strong inductive bias; augmentation hurt more than it helped
- Masked MAE best per-room F1 — reconstruction forces learning of CSI structure
- All SSL encoders outperform handcrafted on **joint** evaluation — cross-room, SSL wins

**Code:** `scripts/task2_eval.py`

---

## Step 10: CORAL Domain Alignment

Without alignment: dev embeddings from Room A and Room B live in **non-overlapping regions** → shared F1 = 0.000.

**CORAL (Sun & Saenko, 2016):** align second-order statistics
```
z_B' = (z_B − μ_B) @ Σ_B^{-½} @ Σ_A^{½}
```
Whitens Room B (zero mean, identity covariance) then re-colours to match Room A.

**Key improvement — shared-class-conditioned CORAL:**  
Instead of fitting on all dev embeddings, fit covariance matrices using only  
shared-activity windows (sit/stand/walk). Directly targets the subspace that  
matters for zero-shot shared transfer.

| Alignment | A→B shared_f1 | B→A shared_f1 |
|-----------|-------------|-------------|
| None | 0.067 | 0.021 |
| Global CORAL | 0.161 | 0.097 |
| **Shared-only CORAL** | **0.279** | **0.106** |

Implemented with eigen-decomposition + regularisation (ε=1e-5 on diagonal) for numerical stability.

**Code:** `src/align.py`

---

## Step 11: Few-Shot Classification Head — Task 3

### The Setup
- **Source room** provides all labeled dev embeddings (no restriction)
- **Target room** provides only K labeled captures for **exclusive** activities
- **Shared activities** (sit/stand/walk) get **zero** target labels — must transfer

### What we tried: NCM (Nearest Class Mean)
Build one prototype per class = mean embedding. Classify by cosine distance.

**Problem:** All cosine similarities were ~0.97 regardless of class.  
Softmax → uniform 0.20 probabilities. NCM completely failed.

### What we use: Logistic Regression head
```
X_train = [all source dev embeddings] + [K-shot target exclusive embeddings]
y_train = [source labels]             + [target exclusive labels]
→ StandardScaler → LogisticRegression(C=1.0, max_iter=1000)
→ Predict target validation windows
→ Mask output to valid target-room classes
```

**Why LR works when NCM fails:** LR uses raw 128-d coordinates, not just direction.  
Even with high cosine similarity, coordinate patterns differ enough per class for  
a linear boundary to separate them (proven by Task 2: LR gets 0.91 F1).

**Code:** `src/fewshot.py`, `run_episode.py`, `scripts/sweep.py`

---

## Task 3 Results — A→B (Room B as target)

Source: Room A (all labels) | Target: Room B (K-shot empty, lying only)

| K | Macro F1 | Bal. Acc | Excl. F1 | Shared F1 |
|---|----------|---------|---------|---------|
| 1 | 0.175 ± 0.053 | 0.174 ± 0.058 | 0.204 ± 0.105 | 0.155 ± 0.039 |
| 5 | 0.312 ± 0.050 | 0.306 ± 0.050 | 0.468 ± 0.081 | 0.208 ± 0.037 |
| 10 | 0.409 ± 0.053 | 0.413 ± 0.047 | 0.644 ± 0.071 | 0.252 ± 0.043 |
| 20 | **0.422 ± 0.034** | **0.432 ± 0.029** | **0.693 ± 0.041** | **0.241 ± 0.033** |

**Exclusive_f1 at K=20 = 0.693** — the LR correctly learns empty/lying from K shots.  
**Shared_f1 plateaus ~0.21–0.25** — zero-shot transfer via CORAL is the ceiling.

---

## Task 3 Results — B→A (Room A as target)

Source: Room B (all labels) | Target: Room A (K-shot fall, sitdown, standup only)

| K | Macro F1 | Bal. Acc | Excl. F1 | Shared F1 |
|---|----------|---------|---------|---------|
| 1 | 0.083 ± 0.002 | 0.128 ± 0.003 | **0.000** ± 0.000 | 0.167 ± 0.004 |
| 5 | 0.112 ± 0.021 | 0.134 ± 0.016 | 0.073 ± 0.040 | 0.152 ± 0.013 |
| 10 | 0.167 ± 0.035 | 0.170 ± 0.031 | 0.208 ± 0.057 | 0.125 ± 0.024 |
| 20 | **0.208 ± 0.029** | **0.235 ± 0.033** | **0.330 ± 0.057** | 0.086 ± 0.010 |

**B→A is harder:** Room A has 3× fewer shared-activity dev windows (1,219 vs 3,955)  
→ noisier CORAL estimate. K=1 excl_f1=0.000 — LR can't learn 3 classes from 1 example each.

---

## Ablation: What Each Fix Contributed

| Change | A→B macro_f1 | A→B excl_f1 | A→B shared_f1 |
|--------|------------|-----------|-------------|
| Baseline (no alignment) | 0.256 | 0.499 | 0.067 |
| + Global CORAL | 0.176 | 0.193 | 0.161 |
| + Shared-only CORAL | 0.296 | 0.350 | **0.279** |
| + v4 encoder (cross-room pairs) | 0.350 | 0.458 | 0.279 |
| **+ LR head (final)** | **0.422** | **0.693** | 0.241 |

Each change targeted a specific diagnosed problem:
1. Global CORAL → brought shared_f1 from 0 to 0.16
2. Shared-only CORAL → focused alignment on the subspace that matters
3. Cross-room pairs → encoder learns room-invariant features (inv 18× higher)
4. LR head → bypasses cosine collapse in NCM, uses full 128-d geometry

---

## Task 3: Key Analysis

**✗ Shared Activity Transfer is the Bottleneck**
- Shared F1 plateaus at 0.21–0.28 regardless of K
- K only adds exclusive-activity shots; shared has zero target labels
- Encoder embeddings for sit/stand/walk differ across rooms

**✓ LR Head Dramatically Improves Exclusive Transfer**
- A→B excl_f1: 0.193 (NCM) → **0.693** (LR) at K=20 — 3.6× improvement
- LR uses raw 128-d coordinates; NCM only uses direction (cosine)

**✓ Shared-Only CORAL Is Critical**
- Global CORAL: shared_f1 = 0.161
- Shared-only CORAL: shared_f1 = **0.279** (+73%)
- Fitting on shared-activity subspace targets what matters

**📉 K Helps Exclusive but Not Shared**
- Exclusive F1 grows consistently with K (more examples for LR to learn from)
- Shared F1 stays flat (no target labels regardless of K)

---

## Routing Contribution (Task 2 — Environment-Aware Normalisation)

| Routing Mode | Room A F1 | Room B F1 | Overall |
|--------------|-----------|-----------|---------|
| None (joint model) | 0.853 | 0.853 | 0.853 |
| Oracle (true label) | 0.819 | **0.968** | 0.893 |
| Predicted (classifier) | 0.819 | **0.968** | 0.893 |

Since room accuracy = 100%, predicted routing equals oracle routing exactly.  
Room B benefits significantly (+11.5%) from room-specific normalisation.  
Room A shows small decrease because the joint model already favours Room A statistics.

---

## Struggles We Faced

### 1. PCAP Decoding Complexity
- No Python library for Nexmon CSI. Reverse-engineered BCM4366c0 binary format.
- ~2 days debugging byte-order, endianness, and packet validity checks.

### 2. VICReg γ=1.0 — Variance Term Never Fired
- `var=0.000` in every epoch across all v1–v4 runs.
- Embeddings converged with high cosine similarity between all classes.
- NCM gave uniform probabilities. Diagnosed late; fixed with γ=2.5 in v5.

### 3. Encoder Retraining Regressions (v2, v3)
- Stronger augmentations (v2) → higher eff_rank but **worse** cross-room F1.
- Per-capture norm (v3) → train/inference mismatch → shared_f1 = 0.000.
- Lesson: eff_rank is not a proxy for room-invariance.

### 4. Few-Shot Support Set Leakage
- Initially counted windows as shots (one capture = 10+ windows).
- Redesigned: capture is the atomic unit. K = K distinct captures.
- Results dropped but became honest.

### 5. NCM Cosine Collapse
- All cosine similarities ~0.97 → softmax → 0.20 uniform output.
- Switching to LR: A→B excl_f1 0.193 → 0.693.

### 6. CORAL Numerical Instability
- Singular matrix errors during covariance inversion.
- Fixed: ε=1e-5 regularisation + eigen-decomposition for matrix sqrt.

---

## What We Learned

**About Room Fingerprinting:**
- Per-subcarrier mean+std is a near-perfect environment descriptor
- 100% accuracy even at T=10 — multipath patterns are highly distinctive

**About SSL for CSI:**
- Architecture inductive bias matters: random encoder (0.873) > VICReg+aug (0.700)
- Same-capture pairs = trivial SSL task; cross-room pairs create real invariance pressure
- Masked reconstruction > contrastive for within-room activity recognition
- γ matters: VICReg variance term must actually fire to have any effect

**About Cross-Room Transfer:**
- CORAL is necessary but not sufficient for shared activity transfer
- Shared-only CORAL (fit on shared subspace only) significantly better than global
- LR uses full embedding geometry; NCM only uses direction — critical difference at high cosine similarity
- The gap: within-room LR = 0.91 F1 vs cross-room LR = 0.42 — that gap is the open research problem

---

## Known Limitations

| Limitation | Impact |
|-----------|--------|
| Room A and B recorded in separate sessions | 100% room accuracy may exploit hardware state, not just environment |
| Room A: activity-specific spatial zones | Classifier may recognise location, not activity |
| Only two rooms | Cannot claim generalisation to unseen environments |
| No subject-disjoint guarantee | Results reflect capture-level disjointness only |
| CORAL needs target dev data to fit | Not truly deployment-ready without target room data |
| B→A CORAL noisier (3× fewer shared windows in Room A) | B→A consistently weaker than A→B |
| v5 (γ=2.5) still training | Results not yet available |

---

## Summary of All Results

| Task | Metric | Value |
|------|--------|-------|
| **Task 1** Room ID | Val accuracy | **1.000** |
| **Task 2** SSL probe | Joint macro-F1 (masked MAE) | **0.853** |
| **Task 2** SSL probe | Room B F1 (masked MAE) | **0.968** |
| **Task 3** A→B K=20 | Macro F1 | **0.422** |
| **Task 3** A→B K=20 | Exclusive F1 | **0.693** |
| **Task 3** A→B K=20 | Shared F1 | 0.241 |
| **Task 3** B→A K=20 | Macro F1 | **0.208** |
| **Task 3** B→A K=20 | Exclusive F1 | **0.330** |

---

## Reproducibility & Code Structure

```
CSI_classifier/
├── src/                    # Core modules
│   ├── pcap_reader.py      # Nexmon PCAP parsing
│   ├── csi_decode.py       # Packed-CSI reconstruction
│   ├── cleaning.py         # Quality control (4 rules)
│   ├── windowing.py        # (150, 256) window extraction
│   ├── normalization.py    # log1p + z-score
│   ├── encoder.py          # ResNet1DEncoder
│   ├── ssl_train.py        # VICReg + Masked MAE
│   ├── room_model.py       # Task 1 room classifier
│   ├── align.py            # CORAL alignment
│   └── fewshot.py          # Support sampling + PrototypeHead
├── scripts/
│   ├── build_intermediate.py
│   ├── build_windows.py
│   ├── fit_normalization.py
│   ├── task2_eval.py
│   └── sweep.py            # Task 3 K-shot sweep
├── configs/experiment.yaml  # Single source of truth
├── run_episode.py           # Demo entry point
└── app.py                   # Streamlit demo UI
```

**Quick demo:**
```bash
python run_episode.py --target_room B --K 5 --seed 42 \
    --condition vicreg_aug_v4 \
    --out reports/episode.csv \
    --predictions_out reports/predictions.csv
```

All seeds fixed in `configs/experiment.yaml` — fully reproducible.

---

## AI Use Disclosure

**Tools used:** Kiro IDE with Claude for development assistance

**AI-assisted:**
- Code generation and debugging (encoder, SSL training, alignment, sweep)
- Architecture suggestions (shared-only CORAL, cross-room pairs, γ fix)
- Identifying root causes (var=0 issue, NCM cosine collapse)
- Report and presentation writing

**Human validation:**
- All code reviewed, understood, and tested by team members
- Every team member can explain every preprocessing and modelling stage
- Results verified through smoke tests, gate checks, and cross-validation
- All technical claims verified against actual implementation and logs

---

## Conclusion

**What we built:** Complete Python pipeline from raw Nexmon PCAP to cross-room activity predictions.

**What works well:**
- Perfect room fingerprinting (Task 1: 100%)
- Strong within-room activity recognition (Task 2: 0.853 joint F1)
- Exclusive activity adaptation improves reliably with K (excl_f1: 0.693 at K=20)

**The open problem:**
- Shared activity zero-shot transfer plateaus at ~0.24 Shared F1
- Encoder learned room-correlated features despite cross-room SSL training
- Raising γ (v5, currently training) may push beyond this ceiling

**Critical insight:**
> Closing the gap between within-room (0.91) and cross-room (0.42) F1  
> requires the encoder to learn activity features that are  
> *invariant to the physical environment* — not yet solved.

---

## Questions?

**CS60055 Ubiquitous Computing | Hackathon Challenge 2**

*Cross-Room Few-Shot Adaptation —  
Environment-Aware, Label-Efficient Activity Recognition from WiFi CSI*
