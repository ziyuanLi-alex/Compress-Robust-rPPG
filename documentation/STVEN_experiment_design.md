# STVEN Pretraining Experiment Design

**Date:** 2026-04-11
**Purpose:** Rigorous experimental protocol for STVEN pretraining with ablation studies, designed for final paper submission.

---

## 1. Experimental Overview

### 1.1 What We Are Testing

The core claim: **a task-specific pretraining loss (rPPG signal supervision through frozen PhysFormer) produces better STVEN initialization than pixel-level losses (reconstruction + cycle consistency).**

To support this claim rigorously, we ablate along three dimensions:

| Dimension | Variable | Values to Compare |
|-----------|----------|-------------------|
| **A. Pretraining loss** | L_cyc+L_rec vs L_rPPG vs no pretraining | 3 conditions |
| **B. CRF encoding** | One-hot vs continuous scalar vs no conditioning | 3 conditions |
| **C. Training CRF range** | Narrow [0,14] vs Broad [0,14,18,20,24] vs Full [all 7 levels] | 3 conditions |

### 1.2 Fixed Factors (Controlled Variables)

These must be **identical** across all experiments to enable fair comparison:

| Factor | Value | Rationale |
|--------|-------|-----------|
| Target dataset | UBFC-rPPG | 42 subjects, standard benchmark, most data |
| PhysFormer checkpoint | PURE_PhysFormer_DiffNormalized.pth | Strongest cross-dataset transfer (established) |
| Train/val/test split | See Section 1.4 | Subject-level leave-out, consistent across all stages |
| Preprocessing | DiffNormalized, chunk_length=160, 128x128, Y5F | Matches PhysFormer requirements |
| STVEN architecture | base_channels=16, 6 ST blocks, global residual | Current design |
| Joint training loss | NegPearson (frozen PhysFormer) | Same fine-tuning for all ablations |
| Joint training epochs | 8, LR=1e-3, batch=2, Adam | Same schedule for all |
| Test CRF levels | 0, 14, 16, 18, 20, 22, 24 | Full evaluation sweep |
| Metrics | MAE, RMSE, MAPE, Pearson, SNR | Standard rPPG metrics |
| Random seed | Fixed (e.g., 42) | Reproducibility |

### 1.3 Why UBFC-rPPG (Not UBFC-Phys) for Primary Experiments

| Criterion | UBFC-rPPG | UBFC-Phys |
|-----------|-----------|-----------|
| Usable subjects | 42 | 12 (s8/s9 have no data) |
| Video clips | 42 (1 per subject) | 24 (1-3 per subject) |
| Resolution | 640x480 | 1024x1024 (downsampled to 128x128 anyway) |
| CRF variants | 7 levels (0,14,16,18,20,22,24) | 7 levels (same) |
| Per-CRF chunks | ~483 total | ~885 total |
| Avg frames per clip | ~2000 (~11 chunks) | ~6325 (~37 chunks) |
| Standard benchmark | Yes | Yes |
| Test subjects for stats | 5 (sufficient for paired t-test) | 3 (marginal) |

UBFC-rPPG has more subjects (42 vs 12), giving better statistical power for ablation comparisons (5 test subjects vs 3). UBFC-Phys has ~1.8x more chunks overall due to longer videos, but fewer independent subjects limits statistical tests. UBFC-Phys is reserved for **cross-dataset generalization** testing (Section 4).

**Paper narrative:** Primary ablations on UBFC-rPPG (rigorous statistics), cross-dataset generalization to UBFC-Phys (transferability), optionally train on UBFC-Phys (scalability).

### 1.4 Data Splitting Strategy

#### Core Principle

**Subject-level splitting:** all CRF levels, all chunks, all videos for the same subject must stay in the same split. This is already enforced by `STVENLoader.split_raw_data()`. **Pretraining and joint training use the exact same split** — the test set is never touched at any stage.

#### UBFC-rPPG Split (42 subjects)

```
Train:  subjects 1,3,4,5,8,9,10,11,12,13,14,15,16,17,18,20,22,
        23,24,25,26,27,30,31,32,33,34,35,36,37,38,39,40
        → 33 subjects, ~380 chunks × 4 CRF = ~1520 training samples

Valid:  subjects 41,42,43,44
        →  4 subjects, ~48 chunks × 4 CRF = ~192 validation samples

Test:   subjects 45,46,47,48,49
        →  5 subjects, ~60 chunks × 7 CRF = ~420 test samples
```

**Config values:**
```yaml
TRAIN:
  DATA:
    BEGIN: 0.0
    END: 0.79    # int(0.79 × 42) = 33 → subjects 1-40
VALID:
  DATA:
    BEGIN: 0.79
    END: 0.90    # int(0.90 × 42) = 37 → subjects 41-44
TEST:
  DATA:
    BEGIN: 0.90
    END: 1.0     # → subjects 45-49
```

**Why these boundaries:** `split_raw_data()` computes `start_idx = int(begin × num_subjects)` and `end_idx = int(end × num_subjects)` on the numerically sorted subject list. With 42 subjects:
- `int(0.79 × 42) = 33` → subjects[0:33] = subjects 1-40
- `int(0.90 × 42) = 37` → subjects[33:37] = subjects 41-44
- `int(1.0 × 42) = 42` → subjects[37:42] = subjects 45-49

**Why this split is adequate for statistics:** 5 test subjects × 7 CRF levels = 35 data points per model. For paired t-test comparing two models, n=5 per CRF level is low but standard in the rPPG community. We supplement with Win/Tie/Loss counts and Cohen's d effect sizes.

#### UBFC-Phys Split (12 usable subjects)

```
Train:  s1, s2, s3, s4, s5, s6, s7, s10
        → 8 subjects, ~600 chunks

Valid:  s11, s12
        → 2 subjects, ~150 chunks

Test:   s13, s14
        → 2 subjects, ~135 chunks
```

**Config values:**
```yaml
TRAIN:
  DATA:
    BEGIN: 0.0
    END: 0.67    # int(0.67 × 12) = 8 → s1-s10 (excl. s8/s9 which have no data)
VALID:
  DATA:
    BEGIN: 0.67
    END: 0.83    # → s11-s12
TEST:
  DATA:
    BEGIN: 0.83
    END: 1.0     # → s13-s14
```

Note: s8 and s9 have no video files and are automatically skipped during data loading. The effective subject count is 12, so percentages are computed over the 12 usable subjects.

**Why this split is limited:** Only 2 test subjects — insufficient for statistical tests. UBFC-Phys is used only for cross-dataset generalization (qualitative comparison), not for primary ablation statistics.

#### Split Consistency Rules

1. **Same split across all stages.** Pretraining, joint training, and evaluation all use the same train/valid/test subjects. The pretraining config and joint training config must have identical `BEGIN`/`END` values.

2. **Same split across all ablation variants.** A1, A2, A3, B1, B2, B3, C1, C2, C3 all share the same test subjects. This enables paired comparisons.

3. **Test set is never used for model selection.** Only validation loss selects the best epoch. Test results are computed once, at the very end.

4. **Previous configs are inconsistent.** The current `STVEN_pretrain.yaml` uses BEGIN=0.0/END=0.3 for training while `joint_st_phys.yaml` uses BEGIN=0.0/END=0.8. The pretrained STVEN was only exposed to subjects 1-12, then joint training introduced subjects 13-34 as "new" data. The new protocol fixes this by using the same 33 subjects in both stages.

5. **Previous CRF levels don't exist.** The current configs reference CRF 5 and CRF 10, which are not present on disk. Only CRF 0, 14, 16, 18, 20, 22, 24 exist.

---

## 2. Ablation Experiments

### 2.1 Experiment A: Pretraining Loss Ablation

**Question:** Does pretraining help, and if so, which loss function produces the best initialization?

**Protocol:**

| ID | Pretraining | Loss | Epochs | Joint Training |
|----|-------------|------|--------|----------------|
| A1 | None (random init) | N/A | 0 | NegPearson, 8 epochs |
| A2 | Pixel-level | L_cyc + L_rec | 8 | NegPearson, 8 epochs |
| A3 | Task-specific | NegPearson through frozen PhysFormer | 8 | NegPearson, 8 epochs |

**Fixed for all A experiments:**
- CRF encoding: one-hot (current implementation, no code changes needed)
- Training CRF range: [0, 14, 20, 24] (representative coverage: light/medium/heavy)
- CRF levels for model conditioning: 4 classes

**Expected outcome:**
- A1 vs A2 tests whether pretraining helps at all
- A2 vs A3 tests whether task-specific loss is better than pixel-level loss
- If A1 >= A3, pretraining is not useful and the pipeline simplifies to single-stage

**Evaluation:** After joint training, evaluate all A models on CRF 0, 14, 16, 18, 20, 22, 24.

### 2.2 Experiment B: CRF Encoding Ablation

**Question:** Does continuous CRF encoding improve generalization to unseen CRF levels compared to one-hot?

**Protocol:** Take the best pretraining strategy from Experiment A and vary encoding:

| ID | Encoding | Implementation | num_bitrate_levels |
|----|----------|---------------|-------------------|
| B1 | No conditioning | No label injected, conv1 input = 3 channels | N/A |
| B2 | One-hot | Current implementation | = number of training CRF levels |
| B3 | Continuous scalar | crf/51.0 broadcast to [B,1,T,H,W], conv1 input = 4 channels | 1 (single channel) |

**Fixed for all B experiments:**
- Pretraining: best from Experiment A
- Training CRF range: [0, 14, 20, 24]
- Joint training: same protocol

**Key test:** Evaluate on CRF 16, 18, 22 (NOT in training set). These are the **unseen** levels that test generalization.

**Expected outcome:**
- B1 vs B2 tests whether CRF conditioning helps
- B2 vs B3 tests whether continuous encoding generalizes better than categorical
- B3 should produce smoother MAE-vs-CRF curve (no anomaly at CRF 16)

### 2.3 Experiment C: Training CRF Range Ablation

**Question:** How does the breadth of training CRF levels affect downstream performance?

**Protocol:** Take the best settings from A and B, vary training data:

| ID | Training CRF levels | Samples per level | Total compression conditions |
|----|--------------------|-------------------|------------------------------|
| C1 | [0, 14] | 2 levels | Light compression only |
| C2 | [0, 14, 20, 24] | 4 levels | Moderate coverage |
| C3 | [0, 14, 16, 18, 20, 22, 24] | 7 levels | Full coverage |

**Fixed for all C experiments:**
- Pretraining: best from A
- CRF encoding: best from B

**Key test:** All models evaluated on the full CRF sweep (0, 14, 16, 18, 20, 22, 24).

**Expected outcome:**
- C1 should perform well on low CRF but fail at high CRF (distribution shift)
- C3 should have the most robust performance across all CRF levels
- C2 vs C3 reveals whether moderate coverage is sufficient (training efficiency tradeoff)

---

## 3. Detailed Walkthrough: Experiment A

This section traces the full data flow for Experiment A (pretraining loss ablation) to show exactly what happens at each step. Experiments B and C follow the same structure, varying only their respective dimension.

### 3.1 Training CRF Levels for Experiment A

All A experiments train on **CRF [0, 14, 20, 24]** (4 levels). This provides coverage across light, moderate, and heavy compression while leaving CRF 16, 18, 22 as **unseen** test levels for generalization analysis later.

Each training sample is a triplet: `(compressed_chunk, uncompressed_chunk, crf_label)` from the STVENLoader. With 33 train subjects × ~380 chunks × 4 CRF levels, each epoch sees ~1520 samples (760 iterations at batch_size=2).

### 3.2 Experiment A1: No Pretraining

```
┌─────────────────────────────────────────────────┐
│  A1: Skip pretraining entirely                  │
│                                                  │
│  STVEN weights: random initialization           │
│                                                  │
│  Joint Training (8 epochs):                     │
│    Train: subjects 1-40, CRF [0,14,20,24]       │
│    Loss: NegPearson through frozen PhysFormer   │
│    Valid: subjects 41-44, CRF [0,14,20,24]      │
│    → Save all 8 epoch checkpoints               │
│    → Select best epoch by validation loss        │
│                                                  │
│  Config: joint_A1.yaml                          │
│    MODEL.STVEN.PRETRAINED_PATH: "" (empty)      │
└─────────────────────────────────────────────────┘
```

This is the simplest baseline. If A1 matches or beats A2/A3, pretraining is unnecessary.

### 3.3 Experiment A2: Pixel-Level Pretraining

```
┌─────────────────────────────────────────────────┐
│  A2: Pretrain with L_cyc + L_rec                │
│                                                  │
│  ┌─── Step 1: STVEN Pretraining (8 epochs) ───┐ │
│  │  Train: subjects 1-40, CRF [0,14,20,24]    │ │
│  │  Loss: L_rec (MSE+L1) + L_cyc (L1)         │ │
│  │        enhanced vs uncompressed (L_rec)      │ │
│  │        cycle: enhanced→recompress vs orig   │ │
│  │  Valid: subjects 41-44, CRF [0,14,20,24]    │ │
│  │  → Save all 8 checkpoints                   │ │
│  │  → Select best epoch by validation loss      │ │
│  │  → Output: stven_pretrained_A2.pth          │ │
│  └─────────────────────────────────────────────┘ │
│                        ↓                         │
│  ┌─── Step 2: Joint Training (8 epochs) ──────┐ │
│  │  Load STVEN from stven_pretrained_A2.pth    │ │
│  │  Load PhysFormer from PURE pretrained       │ │
│  │  Freeze PhysFormer                          │ │
│  │  Train: subjects 1-40, CRF [0,14,20,24]    │ │
│  │  Loss: NegPearson(STVEN→PhysFormer, GT PPG) │ │
│  │  Valid: subjects 41-44, CRF [0,14,20,24]    │ │
│  │  → Save all 8 checkpoints                   │ │
│  │  → Select best epoch by validation loss      │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  Configs: stven_pretrain_A2.yaml + joint_A2.yaml │
│    Both must have identical BEGIN/END splits      │
│    joint_A2.yaml MODEL.STVEN.PRETRAINED_PATH     │
│      points to stven_pretrained_A2.pth            │
└─────────────────────────────────────────────────┘
```

This is the current approach (L_cyc + L_rec), but with corrected data splits and CRF levels.

### 3.4 Experiment A3: Task-Specific Pretraining (Proposed)

```
┌─────────────────────────────────────────────────┐
│  A3: Pretrain with L_rPPG (NegPearson)          │
│                                                  │
│  ┌─── Step 1: STVEN Pretraining (8 epochs) ───┐ │
│  │  Load PhysFormer from PURE pretrained       │ │
│  │  Freeze PhysFormer                          │ │
│  │  Train: subjects 1-40, CRF [0,14,20,24]    │ │
│  │  Loss: NegPearson(STVEN→PhysFormer, GT PPG) │ │
│  │  Valid: subjects 41-44, CRF [0,14,20,24]    │ │
│  │  → Save all 8 checkpoints                   │ │
│  │  → Select best epoch by validation loss      │ │
│  │  → Output: stven_pretrained_A3.pth          │ │
│  └─────────────────────────────────────────────┘ │
│                        ↓                         │
│  ┌─── Step 2: Joint Training (8 epochs) ──────┐ │
│  │  Load STVEN from stven_pretrained_A3.pth    │ │
│  │  Load PhysFormer from PURE pretrained       │ │
│  │  Freeze PhysFormer                          │ │
│  │  Train: subjects 1-40, CRF [0,14,20,24]    │ │
│  │  Loss: NegPearson(STVEN→PhysFormer, GT PPG) │ │
│  │  Valid: subjects 41-44, CRF [0,14,20,24]    │ │
│  │  → Save all 8 checkpoints                   │ │
│  │  → Select best epoch by validation loss      │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  Configs: stven_pretrain_A3.yaml + joint_A3.yaml │
│    Step 1 and Step 2 use the SAME loss           │
│    (NegPearson through frozen PhysFormer)         │
│    The difference: Step 1 starts from random     │
│    init, Step 2 starts from Step 1's weights     │
└─────────────────────────────────────────────────┘
```

A3 is effectively a two-stage curriculum: Step 1 teaches STVEN basic enhancement through the task-specific loss, Step 2 continues from there. Both steps use NegPearson — the only difference is the initialization.

**Key comparison:** A2 Step 1 optimizes pixel-level losses (no PhysFormer involved), while A3 Step 1 already optimizes through PhysFormer. If A3 Step 1 produces good STVEN weights, then A3 Step 2 has a much better starting point than A2 Step 2.

### 3.5 Unified Evaluation (After All A Models Are Trained)

```
For each model (A1_best, A2_best, A3_best):
  For each CRF level (0, 14, 16, 18, 20, 22, 24):
    Load model checkpoint (best epoch by validation)
    Evaluate on test subjects (45-49) at this CRF
    Record per-subject metrics: MAE, RMSE, MAPE, Pearson, SNR

Total: 3 models × 7 CRF levels = 21 inference runs
```

**Output per run:** 5 test subjects × metrics → used for statistical comparison.

**Evaluation configs:** Generate 21 inference configs (3 models × 7 CRFs). Each config points to:
- The appropriate model checkpoint (A1/A2/A3 best epoch)
- The appropriate CRF dataset path
- Test split: BEGIN=0.90, END=1.0

### 3.6 Statistical Comparison for Experiment A

For each CRF level, we have 5 MAE values per model:

```
CRF 20 example:
  A1: [2.1, 3.5, 1.8, 2.9, 2.4]   ← per-subject MAE
  A2: [1.9, 3.2, 2.0, 2.7, 2.2]
  A3: [1.5, 2.8, 1.6, 2.3, 1.8]

Comparisons:
  A1 vs A2: paired t-test, Cohen's d  → does pretraining help?
  A2 vs A3: paired t-test, Cohen's d  → does task-specific loss help?
  A1 vs A3: paired t-test, Cohen's d  → overall improvement
```

Plus Win/Tie/Loss counts across the 5 test subjects.

### 3.7 Experiment A Config Files Summary

All configs are in `configs/train_configs/A/`:

```
configs/train_configs/A/
├── A1/
│   └── joint_A1.yaml              # No pretraining, joint training only
├── A2/
│   ├── stven_pretrain_A2.yaml     # Step 1: Pixel-level pretraining (L_cyc + L_rec)
│   └── joint_A2.yaml              # Step 2: Joint training with pretrained STVEN
└── A3/
    ├── stven_pretrain_A3.yaml     # Step 1: Task-specific pretraining (NegPearson)
    └── joint_A3.yaml              # Step 2: Joint training with pretrained STVEN
```

| Config File | Path | MODEL.NAME | Loss | STVEN Init |
|-------------|------|------------|------|------------|
| `joint_A1.yaml` | `A/A1/` | JointSTPhys | NegPearson | Random |
| `stven_pretrain_A2.yaml` | `A/A2/` | STVEN | L_cyc + L_rec | Random |
| `joint_A2.yaml` | `A/A2/` | JointSTPhys | NegPearson | From stven_pretrain_A2 |
| `stven_pretrain_A3.yaml` | `A/A3/` | JointSTPhys | NegPearson | Random |
| `joint_A3.yaml` | `A/A3/` | JointSTPhys | NegPearson | From stven_pretrain_A3 |

**Note on A3:** `stven_pretrain_A3.yaml` uses `MODEL.NAME: JointSTPhys` (not `STVEN`) because the pretraining loss goes through PhysFormer. This requires the `STVENTrainer` to support a "task-specific pretraining" mode that loads and freezes PhysFormer. If the trainer does not support this yet, use `JointSTVENPhysFormerTrainer` with empty `PRETRAINED_PATH` for both STVEN and PhysFormer init.

**Before running:** Update `joint_A2.yaml` and `joint_A3.yaml` `PRETRAINED_PATH` fields after Steps 1 complete. The current placeholder paths point to Epoch 0 — replace with the best validation epoch.

**Shared settings across all configs:**
- Train split: `BEGIN: 0.0, END: 0.79` → subjects 1-40
- Valid split: `BEGIN: 0.79, END: 0.90` → subjects 41-44
- Test split: `BEGIN: 0.90, END: 1.0` → subjects 45-49
- Training CRF levels: [0, 14, 20, 24]
- `num_bitrate_levels: 4`
- PhysFormer checkpoint: `PURE_PhysFormer_DiffNormalized.pth`
- Seed: 42

### 3.8 Execution Commands

```bash
# A1: No pretraining (single stage)
python main.py --config_file ./configs/train_configs/A/A1/joint_A1.yaml

# A2: Pixel-level pretraining → joint training (two stages)
python main.py --config_file ./configs/train_configs/A/A2/stven_pretrain_A2.yaml
# After Step 1: update PRETRAINED_PATH in joint_A2.yaml to best epoch
python main.py --config_file ./configs/train_configs/A/A2/joint_A2.yaml

# A3: Task-specific pretraining → joint training (two stages)
python main.py --config_file ./configs/train_configs/A/A3/stven_pretrain_A3.yaml
# After Step 1: update PRETRAINED_PATH in joint_A3.yaml to best epoch
python main.py --config_file ./configs/train_configs/A/A3/joint_A3.yaml
```

**Evaluation:** After all training completes, generate inference configs for the best checkpoints and run evaluation across all 7 CRF levels (separate step, not covered by `train_and_test` mode which only tests on one CRF at a time).

---

## 4. Experiment Execution Plan

### 4.1 Prerequisites (Before Any Training)

- [ ] **Generate CRF 0 for UBFC-rPPG.** The current CRF0 dataset exists but verify it is truly lossless (x264 with `-crf 0`).
- [ ] **Verify all 7 CRF datasets** have 42 subjects each and consistent video lengths.
- [ ] **Preprocess all CRF levels** with DiffNormalized, chunk_length=160, Y5F, 128x128. Clear caches and regenerate.
- [ ] **Fix random seed** in all training scripts (add `torch.manual_seed(42)`, `np.random.seed(42)`).
- [ ] **Implement continuous CRF encoding** in STVEN.py (behind a config flag).
- [ ] **Implement task-specific pretraining** (STVENTrainer with frozen PhysFormer, NegPearson loss).

### 4.2 Execution Order

The ablations are sequential (each depends on the previous):

```
Step 1: Experiment A (3 training runs + evaluation)
        │
        ▼ select best pretraining strategy
Step 2: Experiment B (3 training runs + evaluation)
        │
        ▼ select best CRF encoding
Step 3: Experiment C (3 training runs + evaluation)
        │
        ▼ final model selected
Step 4: Cross-dataset evaluation (Section 4)
```

**Total training runs: 9** (3 + 3 + 3)
**Total evaluation runs: 9 models × 7 CRF levels = 63 inference runs**

Each training run: ~8 epochs pretraining + ~8 epochs joint = ~16 epochs total. With batch=2, 42 subjects, ~40 chunks each ≈ 840 iterations per epoch. Total per experiment: ~13,440 iterations.

### 4.3 Per-Experiment Checklist

For each training run, record:

1. **Config file** (committed to git)
2. **Training loss curves** (per-epoch average)
3. **Validation loss curve** (per-epoch)
4. **Best epoch** (selected by validation loss)
5. **All epoch checkpoints** (saved for later analysis)
6. **Final model checkpoint** (best epoch)

For each evaluation run, record:

1. **Config file**
2. **Per-subject metrics** (not just averages — needed for statistical tests)
3. **Aggregated metrics** (MAE, RMSE, MAPE, Pearson, SNR ± std)

---

## 5. Cross-Dataset Generalization

### 5.1 Purpose

After selecting the best STVEN configuration from ablations on UBFC-rPPG, test whether the enhancement transfers to a different dataset.

### 5.2 Protocol

**Train on UBFC-rPPG, test on UBFC-Phys:**

```
Training: UBFC-rPPG subjects 1-42, CRF [best range from C]
          Standard split: 0-0.8 / 0.8-0.9 / 0.9-1.0

Testing:  UBFC-Phys subjects s1-s14 (excluding s8/s9)
          All available CRF levels (0, 14, 16, 18, 20, 22, 24)
```

**Compare against:**
- Standalone PhysFormer (no STVEN) on UBFC-Phys compressed video
- Current STVEN-PhysFormer (one-hot, L_cyc+L_rec pretraining) on UBFC-Phys

This tests whether the improvements from our ablations transfer across datasets.

### 5.3 Additional: Train on UBFC-Phys, Test on UBFC-rPPG

This is a harder test (only 12 training subjects, ~24 clips). If time permits:

```
Training: UBFC-Phys subjects s1-s14, CRF [best range]
          Split: s1-s10 train, s11-s12 valid, s13-s14 test

Testing:  UBFC-rPPG subjects 1-42, all CRF levels
```

This tests whether STVEN can learn compression enhancement from limited data.

---

## 6. Statistical Rigor

### 6.1 Required Statistical Tests

**1. Paired t-test across subjects:**
For each CRF level, compare MAE distributions across models using paired t-test (same test subjects, different models). Report p-values.

```
Model A MAE per subject: [2.1, 3.5, 1.8, ...]  (42 values)
Model B MAE per subject: [1.9, 3.2, 2.0, ...]  (42 values)
→ Paired t-test, report t-statistic and p-value
```

**2. Win/Tie/Loss counting:**
For each CRF level, count how many subjects improve/degrade/wstay same for each ablation comparison.

**3. Effect size (Cohen's d):**
Not just p-values — report effect sizes to show practical significance, not just statistical significance.

### 6.2 Multiple Comparisons Correction

With 9 models × 7 CRF levels = 63 comparisons, apply Bonferroni correction:
- Significance threshold: α = 0.05 / 63 ≈ 0.0008
- Or use Holm-Bonferroni for less conservative correction

### 6.3 Reporting Format

All results tables should include:

| Model | CRF | MAE | MAE_std | RMSE | Pearson | Pearson_std | p-value | Cohen's d |
|-------|-----|-----|---------|------|---------|-------------|---------|-----------|

Where p-value and Cohen's d are relative to the baseline (no STVEN).

---

## 7. Baselines Required

Before running any STVEN experiments, establish these baselines on the **same data split** and **same preprocessing**:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| B_physformer_clean | PhysFormer on CRF 0 (uncompressed) | Upper bound — best achievable |
| B_physformer_crf{14,16,18,20,22,24} | PhysFormer on each CRF level | Lower bounds — no enhancement |
| B_physmamba_crf{0,14,16,18,20,22,24} | PhysMamba on each CRF level | External baseline (different arch) |

The STVEN improvement is measured as the reduction in MAE/RMSE relative to B_physformer at each CRF level.

**Critical:** These baselines must use the exact same train/val/test split as the STVEN experiments. Do not compare against previously reported numbers that used different splits.

---

## 8. Paper Tables and Figures

### 8.1 Required Tables

**Table 1: Ablation — Pretraining Loss (Experiment A)**

| Pretraining | CRF 0 | CRF 14 | CRF 16 | CRF 18 | CRF 20 | CRF 22 | CRF 24 | Avg(CRF>0) | Drop(CRF0→24) |
|-------------|-------|--------|--------|--------|--------|--------|--------|-------------|----------------|
| None | — | — | — | — | — | — | — | — | — |
| L_cyc+L_rec | — | — | — | — | — | — | — | — | — |
| L_rPPG | — | — | — | — | — | — | — | — | — |

Report MAE (top) and Pearson (bottom) for each cell.

**Table 2: Ablation — CRF Encoding (Experiment B)**

| Encoding | CRF 0 | CRF 14 | **CRF 16*** | **CRF 18*** | CRF 20 | **CRF 22*** | CRF 24 |
|----------|-------|--------|------------|------------|--------|------------|--------|
| None | — | — | — | — | — | — | — |
| One-hot | — | — | — | — | — | — | — |
| Continuous | — | — | — | — | — | — | — |

*Italic CRF levels are unseen during training — test generalization.

**Table 3: Ablation — Training CRF Range (Experiment C)**

| Training CRFs | CRF 0 | CRF 14 | CRF 16 | CRF 18 | CRF 20 | CRF 22 | CRF 24 |
|---------------|-------|--------|--------|--------|--------|--------|--------|
| [0,14] | — | — | — | — | — | — | — |
| [0,14,20,24] | — | — | — | — | — | — | — |
| All 7 levels | — | — | — | — | — | — | — |

**Table 4: Cross-Dataset Generalization**

| Training Data | Test Data | CRF 0 | CRF 14 | CRF 20 | CRF 24 |
|---------------|-----------|-------|--------|--------|--------|
| UBFC-rPPG | UBFC-rPPG | — | — | — | — |
| UBFC-rPPG | UBFC-Phys | — | — | — | — |
| UBFC-Phys | UBFC-rPPG | — | — | — | — |

### 8.2 Required Figures

**Figure 1:** MAE vs CRF curve — all ablation variants overlaid. X-axis: CRF level (0-24). Y-axis: MAE (BPM). One line per model, with std as shaded region.

**Figure 2:** Pearson vs CRF curve — same format.

**Figure 3:** Enhancement visualization — select 2-3 subjects, show: compressed frame → enhanced frame → difference (what STVEN learned to change). Include CRF 14 and CRF 24 examples.

**Figure 4:** rPPG signal comparison — for one subject at CRF 20: ground truth BVP, PhysFormer-only rPPG, STVEN-PhysFormer rPPG. Show how enhancement improves signal recovery.

**Figure 5:** Loss curve comparison (Experiment A) — training/validation loss curves for L_cyc+L_rec vs L_rPPG pretraining. Demonstrate that L_rPPG converges faster and to a better minimum.

---

## 9. Implementation Roadmap

### Phase 0: Code Changes (Before Any Training)

| Task | File(s) | Description |
|------|---------|-------------|
| Continuous CRF encoding | `STVEN.py`, `STVENLoader.py`, `config.py` | Add `CONTINUOUS_CRF` flag; implement scalar broadcast |
| Task-specific pretraining | `STVENTrainer.py` | Add option to use frozen PhysFormer + NegPearson instead of L_cyc+L_rec |
| Seed fixing | `STVENTrainer.py`, `JointSTVENPhysFormerTrainer.py` | Add `torch.manual_seed`, `np.random.seed`, `random.seed` |
| Per-subject metrics | `evaluation/metrics.py` or trainer test methods | Ensure metrics are saved per-subject for statistical tests |

### Phase 1: Baselines (1-2 days)

- [ ] Run PhysFormer on UBFC-rPPG at all 7 CRF levels with the standard split
- [ ] Run PhysMamba on UBFC-rPPG at all 7 CRF levels
- [ ] Save per-subject metrics for all baselines

### Phase 2: Experiment A — Loss Ablation (3-4 days)

- [ ] A1: Random init → joint training (8 epochs)
- [ ] A2: L_cyc+L_rec pretraining (8 epochs) → joint training (8 epochs)
- [ ] A3: L_rPPG pretraining through frozen PhysFormer (8 epochs) → joint training (8 epochs)
- [ ] Evaluate all 3 on 7 CRF levels (21 inference runs)
- [ ] Analyze: select best pretraining strategy

### Phase 3: Experiment B — Encoding Ablation (3-4 days)

- [ ] B1: No conditioning (best pretraining from A)
- [ ] B2: One-hot (best pretraining from A)
- [ ] B3: Continuous scalar (best pretraining from A)
- [ ] Evaluate all 3 on 7 CRF levels, focus on unseen levels (21 inference runs)
- [ ] Analyze: select best encoding

### Phase 4: Experiment C — CRF Range Ablation (3-4 days)

- [ ] C1: Train on CRF [0, 14]
- [ ] C2: Train on CRF [0, 14, 20, 24]
- [ ] C3: Train on CRF [0, 14, 16, 18, 20, 22, 24]
- [ ] Evaluate all 3 on 7 CRF levels (21 inference runs)
- [ ] Analyze: select best CRF range → final model

### Phase 5: Cross-Dataset + Paper Figures (2-3 days)

- [ ] Evaluate final model on UBFC-Phys
- [ ] Generate all paper figures
- [ ] Compute statistical tests (paired t-test, Cohen's d)
- [ ] Write results section

**Estimated total: ~2 weeks**

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| L_rPPG pretraining is unstable (PhysFormer gradients vanish) | Medium | High | Start with L_rec as warmup for 2 epochs, then switch to L_rPPG |
| Continuous encoding shows no improvement over one-hot | Low-Medium | Medium | One-hot is still a valid conditioning method; report as negative result |
| UBFC-Phys cross-dataset shows poor generalization | Medium | Medium | Expected — different resolution and acquisition. Report honestly with analysis |
| CRF 14 performs differently from CRF 0 (not truly lossless) | Low | Low | Verify CRF 0 is actually lossless x264; if not, note in paper |
| Training runs take too long | Medium | Medium | Reduce pretraining epochs based on early loss curves; batch=2 is already small |
| Overfitting with only 42 subjects | Medium | High | Monitor train/val gap; use early stopping; report overfitting if observed |

---

## 11. Summary of Deliverables

For the paper, the minimum set of experiments that must be completed:

1. **Baselines:** PhysFormer on 7 CRF levels (7 inference runs with pretrained checkpoint)
2. **Experiment A:** 3 training runs + 21 inference runs
3. **Experiment B:** 3 training runs + 21 inference runs
4. **Experiment C:** 3 training runs + 21 inference runs
5. **Cross-dataset:** 1 evaluation on UBFC-Phys (7 inference runs)
6. **Statistical analysis:** Paired t-tests and effect sizes for all comparisons

**Total: 9 training runs, 77 inference runs, 2 weeks estimated.**

If time is constrained, the minimum viable ablation is **Experiment A only** (pretraining loss). The CRF encoding and range ablations strengthen the paper but the loss function is the most impactful claim.
