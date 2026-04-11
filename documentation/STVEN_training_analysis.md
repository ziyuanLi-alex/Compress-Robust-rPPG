# STVEN Training Analysis and Proposed Improvements

**Date:** 2026-04-10
**Reference:** Yu et al., 2019 — "Remote Heart Rate Measurement from Highly Compressed Facial Videos: An End-to-End Deep Learning Solution with Video Enhancement" (IEEE JBHI)
**Context:** Analysis of the original STVEN training methodology against our current implementation, with proposed improvements for compression-robust rPPG extraction.

---

## 1. Original STVEN Training Method (Yu et al., 2019)

### 1.1 Loss Functions

The original paper uses a 4-component loss:

| Loss | Formula | Description |
|------|---------|-------------|
| L_np | Negative Pearson(P_c, P_d) | Differentiable heart rate loss; Pearson correlation between rPPG signals from compressed-enhanced and uncompressed paths |
| L_cyc | \|\|E_c(E_d(I_d, 0), k) - I_d\|\|_1 (subsampled) | Cycle consistency; enhanced→re-compressed should reconstruct original compressed frames |
| L_rec | \|\|P_c(E_d(I_d, k)) - P_c(I_d)\|\|_1 (subsampled) | Reconstruction at rPPGNet output level; rPPG features from enhanced should match uncompressed |
| L_p | Σ_i \|\|φ_c^(i) - φ_d^(i)\|\|_1 | Perceptual loss; skin-attention feature maps from teacher rPPGNet should match between compressed-enhanced and uncompressed paths |

**Total loss (Stage 3):**

```
L_net = L_np + ε·L_p + ρ·L_STVEN
```

where ε = 1, ρ = 1e-4.

### 1.2 Training Pipeline

```
Stage 1: Pre-train rPPGNet on HIGH-QUALITY (uncompressed) videos — 20k iterations
         ↓
Stage 2: Pre-train STVEN on COMPRESSED videos with L_cyc + L_rec — 20k iterations
         ↓
Stage 3: Joint Fine-tuning (Teacher-Student) — 20k iterations
         Teacher path: uncompressed → rPPGNet (frozen)
         Student path: compressed → STVEN → rPPGNet
         Loss: L_np + ε·L_p + ρ·L_STVEN
```

**Optimizer:** Adam, lr = 1e-4, weight decay = 1e-5.
**Batch:** 64 patches per iteration.
**Downsampling:** L_cyc and L_rec computed on 1/4 spatial, 1/4 temporal resolution for speed.

### 1.3 Compression Conditioning

CRF/bitrate level injected as one-hot encoding concatenated at input:

```
Input: [B, 3+num_levels, T, H, W]
```

Original uses 3 bitrate levels (500/1000/1500 kb/s H.264/H.265).

---

## 2. Current Implementation

### 2.1 What We Kept

| Component | Original | Current | Status |
|-----------|----------|---------|--------|
| R(2+1)D factorization | Yes | Yes | Preserved |
| Encoder-Bottleneck-Decoder | Yes | Yes | Preserved |
| One-hot compression conditioning | Yes | Yes | Preserved |
| Instance Normalization | Yes | Yes | Preserved |
| 3D convolutions | Yes | Yes | Preserved |
| Global residual learning | No | Yes | Added |

### 2.2 What We Changed

| Component | Original | Current | Rationale (from docs) |
|-----------|----------|---------|----------------------|
| Backend | rPPGNet (CNN + skin attention) | PhysFormer (ViT + temporal difference) | SOTA performance on uncompressed video |
| Cycle loss (L_cyc) | Included in Stage 2 | Included in pretraining (limited to 3-5 epochs) | "Extremely slow convergence on H.264; hard to observe convergence signals" |
| Teacher-student joint training | Stage 3 with perceptual loss | Direct rPPG waveform loss, frozen PhysFormer | "Very bad results — performance degradation when aligning to uncompressed teacher features" |
| Base channels | 64 | 16 | Reduced memory for end-to-end training with large ViT |
| Global residual | Not used | Output = Network(Input) + Input | Faster convergence, artifact learning |
| Frame length | 64 | 160 | Matches PhysFormer's temporal context requirement |

### 2.4 Current Training Configuration

**STVEN Pretraining (`STVEN_pretrain.yaml`):**
- CRF levels: [0, 5, 10]
- Loss: L_rec (MSE + L1) + L_cyc (L1)
- Epochs: 8, LR: 1e-3, Batch: 2
- Data split: 0-30% train, 30-40% valid, 40-50% test
- Selected checkpoint: Epoch 1 (DiffNormalized, Y5F backend)

**Joint Training (`joint_st_phys.yaml`):**
- CRF levels: [0, 5, 10] (train), [22, 24] (test)
- Loss: NegPearson (rPPG waveform)
- PhysFormer: Frozen
- Epochs: 8, LR: 1e-3, Batch: 2
- Data split: 0-80% train, 80-90% valid, 90-100% test

---

## 3. Root Cause Analysis

### 3.1 Why L_cyc Converges Slowly (Primary Issue)

The cycle consistency loss assumes a smooth, approximately invertible mapping:

```
enhanced → re-compressed ≈ original compressed
```

This assumption holds for low-bitrate H.263/H.264 used in the original paper (500-1500 kb/s), where compression is a gradual quality degradation with smooth gradients.

It does **not** hold for CRF-based x264 compression, which uses hard quantization:
- CRF 0 = lossless (no quantization)
- CRF 14 = perceptually lossless
- CRF 20+ = aggressive quantization with visible block boundaries and chroma subsampling
- CRF 24+ = severe artifacts, near-zero spatial coherence

The mapping from CRF 24 → enhanced → re-compressed at CRF 24 is not well-defined because x264's rate-distortion optimization makes content-dependent decisions. The network cannot learn a meaningful inverse, so L_cyc provides near-zero gradient signal.

### 3.2 Why L_rec is Misaligned with PhysFormer

L_rec compares pixel-level output of STVEN against uncompressed frames:

```
L_rec = MSE(enhanced, uncompressed) + L1(enhanced, uncompressed)
```

In the original paper, this makes sense because rPPGNet's skin attention module produces spatially meaningful features — pixel-level fidelity directly affects the downstream skin ROI extraction.

PhysFormer operates differently:
- Input is divided into non-overlapping patches
- Self-attention mixes spatial information globally
- Temporal difference attention operates on patch-level feature differences

A pixel-perfect reconstruction is unnecessary — and potentially harmful, because PhysFormer needs compression-invariant temporal features, not pixel-accurate spatial features. L_rec optimizes for the wrong objective.

### 3.3 Why One-Hot CRF Encoding Limits Generalization

One-hot encoding treats CRF levels as independent categories:

```
CRF 14 → [0, 1, 0, 0, 0]
CRF 20 → [0, 0, 0, 1, 0]
```

The network learns separate enhancement strategies per CRF level with no awareness that CRF 20 is "more compressed" than CRF 14. This explains:
- **CRF 16 anomaly:** CRF 16 was not in the training set [0, 5, 10]; the network maps it to the nearest learned category, which may be inappropriate
- **No interpolation:** The network cannot generalize to unseen CRF values
- **Discontinuous behavior:** Small CRF changes (e.g., 18→20) may cause disproportionate output changes

### 3.4 Why Pretraining Epoch 1 Was Selected

The joint training config loads `STVEN_pretrain_STVEN_Epoch1.pth`. The documentation states that cycle/reconstruction loss convergence is "slow and hard to observe," motivating early stopping. However, selecting Epoch 1 (the second checkpoint) likely means:
- The losses had barely decreased by then
- The pretrained STVEN weights are near-initialization
- The joint training phase effectively trains STVEN from scratch with the rPPG loss anyway

This raises the question of whether pretraining with L_cyc + L_rec provides any benefit at all.

### 3.5 Data Split Mismatch

Pretraining uses subjects 0-30%, joint training uses 0-80%. The pretrained STVEN was never exposed to subjects 30-80% during pretraining, but then must enhance them during joint training. This is not a critical issue (the joint training adapts the weights), but it means the pretraining provides no transfer benefit for those subjects.

---

## 4. Proposed Improvements

### 4.1 Replace L_cyc + L_rec with Task-Specific Loss (Priority 1)

**Change:** Drop pixel-level losses entirely. Use rPPG signal loss through frozen PhysFormer during pretraining.

```
L_pretrain = NegPearson(STVEN→PhysFormer(compressed), ground_truth_PPG)
```

**Why this works:**
- Gradients flow through PhysFormer, so STVEN learns to produce output that PhysFormer can extract rPPG from — not output that looks like uncompressed video
- The paper itself (Section 3.4) states the rPPG signals "satisfy the same temporal characteristics regardless of the original compression levels and lighting variations" — this is exactly the invariance we want
- Removes the invertibility assumption that fails for x264
- Eliminates the slow convergence problem entirely

**Implementation:**
- During pretraining: freeze PhysFormer, train STVEN with NegPearson loss (same as current joint training)
- This effectively makes pretraining and joint training identical, simplifying the pipeline to a single stage

### 4.2 Continuous CRF Encoding (Priority 2)

**Change:** Replace one-hot categorical encoding with a continuous scalar.

```python
# Option A: Normalized scalar broadcast to spatial dimensions
crf_scalar = crf_value / 51.0  # x264 CRF range: 0-51
label_map = crf_scalar.view(B, 1, 1, 1, 1).expand(-1, -1, T, H, W)
x = torch.cat([x, label_map], dim=1)  # [B, 4, T, H, W] instead of [B, 6, T, H, W]

# Option B: Learned embedding + sinusoidal positional encoding
crf_embedding = nn.Embedding(1, embed_dim)  # Project scalar to vector
crf_spatial = sinusoidal_encode(crf_scalar)  # [B, embed_dim, T, H, W]
```

**Why this works:**
- The network learns a continuous function of compression severity, not per-level memorization
- Generalizes to unseen CRF levels (e.g., train on [0, 10, 20], evaluate on 15)
- Reduces model parameters (1 extra input channel vs N extra channels)
- CRF 18 and CRF 20 produce similar inputs, so the network produces similar outputs

**Impact on model:**
- `conv1` input channels change from `3 + num_bitrate_levels` to `4` (or `3 + embed_dim`)
- `num_bitrate_levels` config parameter becomes unnecessary
- `use_bitrate_labels` remains as a boolean toggle

### 4.3 Train on Full CRF Range (Priority 3)

**Change:** Use all available CRF datasets during training.

```yaml
CRF_DATASETS:
  "0": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF0"
  "5": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF5"
  "10": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF10"
  "14": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF14"
  "16": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF16"
  "18": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF18"
  "20": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF20"
  "22": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF22"
  "24": "/home/zyuanli/dev/lib/data/UBFC-rPPG-CRF24"
CRF_LEVELS: [0, 5, 10, 14, 16, 18, 20, 22, 24]
```

**Why this works:**
- Eliminates the CRF 16 distribution gap (currently not in training, causing anomalous results)
- With continuous encoding (4.2), the network learns the full degradation spectrum
- Training on hard cases (CRF 22-24) improves robustness at moderate levels (CRF 16-20)

### 4.4 Multi-Scale Perceptual Loss as Optional Regularizer (Priority 4)

**Change:** If spatial losses are desired for pretraining stabilization, replace pixel-level L_rec with feature-level perceptual loss.

```python
# Use frozen VGG or similar pretrained network
# Compare multi-scale features between enhanced and uncompressed
L_perceptual = Σ_s ||φ_s(enhanced) - φ_s(uncompressed)||_1
```

**Why this is lower priority:**
- If 4.1 (task-specific loss) works well, perceptual loss may be unnecessary
- Adds computational overhead (extra forward pass through feature extractor)
- Main benefit is providing a denser gradient signal during early training when PhysFormer gradients are weak

### 4.5 Adaptive Training Schedule with Partial Unfreezing (Priority 5)

**Change:** Replace fixed frozen PhysFormer with gradual unfreezing.

```
Phase 1 (2-3 epochs): Freeze PhysFormer entirely, train STVEN only
Phase 2 (3-5 epochs): Unfreeze PhysFormer's last 2-3 transformer layers
                       Use 10x smaller LR for PhysFormer than STVEN
Phase 3 (1-2 epochs): Full joint fine-tuning with very small LR
```

**Why this works:**
- PhysFormer was trained on uncompressed video; its feature distribution shifts when processing enhanced (not uncompressed) input
- Partial unfreezing lets PhysFormer adapt to the enhanced input distribution without catastrophic forgetting
- The original paper's teacher-student approach failed because it forced feature alignment; direct joint training avoids this

**Risks:**
- Overfitting on small dataset if unfreezing too aggressively
- Increased memory usage (more parameters in computation graph)

---

## 5. Simplified Training Pipeline (After Changes)

The proposed improvements reduce the training from 3 stages to 1-2 stages:

```
Proposed Pipeline:
==================

Stage 1: Joint STVEN + PhysFormer Training
         Input: Compressed videos at CRF [0, 5, 10, 14, 16, 18, 20, 22, 24]
         Encoding: Continuous CRF scalar (normalized to [0, 1])
         Loss: NegPearson(rPPG_STVEN_PhysFormer, ground_truth_PPG)
         PhysFormer: Frozen (Phase 1) → Partially unfrozen (Phase 2)
         Stop: Validation Pearson correlation plateaus
```

### Compared to Original

| Aspect | Original (3 stages) | Current (2 stages) | Proposed (1 stage) |
|--------|---------------------|--------------------|--------------------|
| Pretraining loss | L_cyc + L_rec | L_cyc + L_rec | NegPearson (same as joint) |
| Joint loss | L_np + L_p + L_cyc + L_rec | NegPearson | NegPearson |
| CRF encoding | One-hot | One-hot | Continuous scalar |
| CRF training range | 3 levels | 3 levels | All available levels |
| Convergence | Slow (20k iters) | Slow (8 epochs) | Fast (task-specific signal) |
| PhysFormer | Frozen (teacher) | Frozen | Frozen → partially unfrozen |

---

## 6. Expected Impact

| Metric Area | Current | Expected After Changes | Basis |
|-------------|---------|----------------------|-------|
| CRF 16 anomaly | RMSE 8.31 | Comparable to CRF 18-20 (~3-4) | Continuous encoding + full CRF training eliminates distribution gap |
| Training convergence | 8+ epochs, unclear stopping | 5-8 epochs with clear validation signal | Task-specific loss provides direct optimization signal |
| CRF 22-24 performance | MAE 7.59/14.90 | Moderate improvement | Training on full CRF range improves generalization |
| Code complexity | 2 trainers (STVENTrainer + JointTrainer) | 1 trainer | Pretraining uses same loss as joint training |
| CRF 0 baseline | MAE 0.28 | Maintained or improved | Continuous encoding provides cleaner signal than one-hot |

---

## 7. Implementation Notes

### Files to Modify

| File | Change |
|------|--------|
| `neural_methods/model/STVEN.py` | Add continuous CRF encoding option; remove `num_bitrate_levels` dependency |
| `neural_methods/trainer/STVENTrainer.py` | Replace L_cyc + L_rec with NegPearson loss through PhysFormer |
| `neural_methods/data_loader/STVENLoader.py` | Support continuous CRF scalar output instead of one-hot |
| `config.py` | Add `CONTINUOUS_CRF_ENCODING` flag; update defaults |
| `configs/train_configs/STVEN_pretrain.yaml` | Add full CRF range; remove L_cyc/L_rec references |
| `configs/train_configs/joint_st_phys.yaml` | Consolidate into single-stage config |

### Backward Compatibility

- Existing pretrained checkpoints (one-hot encoding) can be loaded with `strict=False` — the `conv1` layer will have different input channels and need retraining anyway
- Keep the one-hot encoding path behind a config flag for A/B comparison

### UBFC-Phys Integration

UBFC-Phys has 14 subjects with higher resolution. The continuous CRF encoding naturally handles resolution differences (the CRF scalar is resolution-independent). The full CRF training range should include UBFC-Phys CRF variants once generated.
