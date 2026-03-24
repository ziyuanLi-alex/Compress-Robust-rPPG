# STVEN: Spatio-Temporal Video Enhancement Network

**Architecture Report**
**Author:** Implementation for rPPG-Toolbox
**Purpose:** Compression-robust remote photoplethysmography via video enhancement

---

## Overview

STVEN is a 3D-CNN encoder-decoder network designed to enhance compressed video frames before they are processed by PhysFormer for rPPG signal extraction. The network learns to remove compression artifacts while preserving physiological signal information.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STVEN Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: [B, 3, T, 128, 128]          Output: [B, 3, T, 128, 128]       │
│       (compressed video)                  (enhanced video)              │
│                              │                                          │
│  Encoder                       ▼                       Decoder          │
│  ┌──────────────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │  Conv_1          │    │          │    │  DConv_1                 │  │
│  │  3→64 ch         │───▶│  Conv_2  │───▶│  512→128 ch              │  │
│  │  128→128         │    │  64→128  │    │  T/2→T, 32→64            │  │
│  └──────────────────┘    │  128→64  │    └──────────────────────────┘  │
│                          └──────────┘               │                  │
│                              │                      ▼                  │
│                          ┌──────────┐    ┌──────────────────────────┐  │
│                          │  Conv_3  │    │  DConv_2                 │  │
│                          │  128→512 │    │  128→64 ch               │  │
│                          │  64→32   │    │  64→128                  │  │
│                          │  T→T/2   │    └──────────────────────────┘  │
│                          └──────────┘               │                  │
│                              │                      ▼                  │
│                          ┌──────────┐    ┌──────────────────────────┐  │
│  ┌─────────────────┐     │  6× ST   │    │  DConv_3                 │  │
│  │  (Optional)     │────▶│  Blocks  │───▶│  64→3 ch                 │  │
│  │  Bitrate Label  │     │  512 ch  │    │  128→128                 │  │
│  └─────────────────┘     └──────────┘    └──────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │ Global Residual │                                  │
│                    │  x + input      │                                  │
│                    └─────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. STBlock (Spatio-Temporal Block)

**Purpose:** R(2+1)D residual block for efficient spatio-temporal feature extraction.

**Design Principle:** Decomposes a 3D convolution into separate spatial (2D) and temporal (1D) convolutions, reducing parameters while maintaining representational capacity.

```python
# R(2+1)D Decomposition
# Standard 3D Conv: C_in × C_out × (T×H×W) parameters
# R(2+1)D: C_in × M × (H×W) + M × C_out × T parameters
# where M = floor((T × d² × C_in × C_out) / (d² × C_in + T × C_out))
```

**Structure:**
```
Input ──┬──────────────────────────────────────┬── (+) ── ReLU ── Output
        │                                      │
        ▼                                      │
    [Conv2D] 1×H×W, stride (1,h,w)             │
        │                                      │
        ▼                                      │
    [InstanceNorm]                             │
        │                                      │
        ▼                                      │
    [ReLU]                                     │
        │                                      │
        ▼                                      │
    [Conv1D] T×1×1, stride (t,1,1)             │
        │                                      │
        ▼                                      │
    [InstanceNorm]                             │
        │                                      │
        └──────────────────────────────────────┘
                    (Residual Connection)
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| kernel_size | (3, 3, 3) | Temporal × Height × Width |
| stride | (1, 1, 1) | Stride per dimension |
| padding | (1, 1, 1) | Padding per dimension |

---

### 2. ConvBlock (Encoder Block)

**Purpose:** Downsampling encoder convolutional block.

**Structure:**
```
Input ──▶ [Conv3D] ──▶ [InstanceNorm3D] ──▶ [ReLU] ──▶ Output
```

**Usage in Encoder:**
| Block | Input | Output | Kernel | Stride | Purpose |
|-------|-------|--------|--------|--------|---------|
| Conv_1 | 3×T×128×128 | 64×T×128×128 | (3,7,7) | (1,1,1) | Feature extraction |
| Conv_2 | 64×T×128×128 | 128×T×64×64 | (3,4,4) | (1,2,2) | Spatial downsampling |
| Conv_3 | 128×T×64×64 | 512×T/2×32×32 | (4,4,4) | (2,2,2) | Spatio-temporal downsampling |

---

### 3. DeconvBlock (Decoder Block)

**Purpose:** Upsampling transposed convolutional block for spatial/temporal reconstruction.

**Structure:**
```
Input ──▶ [ConvTranspose3D] ──▶ [InstanceNorm3D] ──▶ [ReLU] ──▶ Output
```

**Usage in Decoder:**
| Block | Input | Output | Kernel | Stride | Purpose |
|-------|-------|--------|--------|--------|---------|
| DConv_1 | 512×T/2×32×32 | 128×T×64×64 | (4,4,4) | (2,2,2) | Spatio-temporal upsampling |
| DConv_2 | 128×T×64×64 | 64×T×128×128 | (1,4,4) | (1,2,2) | Spatial upsampling |
| DConv_3 | 64×T×128×128 | 3×T×128×128 | (1,7,7) | (1,1,1) | Output reconstruction |

**Note:** DConv_3 omits activation and normalization (`with_act=False`) for linear output.

---

### 4. Full STVEN Network

**Encoder Path:**
```
Input: [B, 3, T, 128, 128]
           │
           ▼
    ┌──────────────┐
    │   Conv_1     │  (3,7,7), same padding
    │  3 → 64 ch   │
    └──────────────┘
           │
           ▼  [B, 64, T, 128, 128]
    ┌──────────────┐
    │   Conv_2     │  (3,4,4), stride (1,2,2)
    │  64 → 128 ch │
    └──────────────┘
           │
           ▼  [B, 128, T, 64, 64]
    ┌──────────────┐
    │   Conv_3     │  (4,4,4), stride (2,2,2)
    │  128 → 512 ch│
    └──────────────┘
           │
           ▼  [B, 512, T/2, 32, 32]
```

**Bottleneck:**
```
[B, 512, T/2, 32, 32]
           │
           ▼
    ┌──────────────┐
    │  ST_Block 1  │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  ST_Block 2  │
    └──────────────┘
           │
           ...
           ▼
    ┌──────────────┐
    │  ST_Block 6  │
    └──────────────┘
           │
           ▼  [B, 512, T/2, 32, 32]
```

**Decoder Path:**
```
[B, 512, T/2, 32, 32]
           │
           ▼
    ┌──────────────┐
    │   DConv_1    │  (4,4,4), stride (2,2,2)
    │ 512 → 128 ch │
    └──────────────┘
           │
           ▼  [B, 128, T, 64, 64]
    ┌──────────────┐
    │   DConv_2    │  (1,4,4), stride (1,2,2)
    │ 128 → 64 ch  │
    └──────────────┘
           │
           ▼  [B, 64, T, 128, 128]
    ┌──────────────┐
    │   DConv_3    │  (1,7,7), no activation
    │  64 → 3 ch   │
    └──────────────┘
           │
           ▼  [B, 3, T, 128, 128]
```

**Global Residual Learning:**
```
Output = Decoder_Output + Input
```

The network learns the *residual* (compression artifacts/noise) to subtract from the input, rather than learning the full enhancement mapping directly.

---

## Bitrate Label Injection

**Purpose:** Condition the enhancement on the compression level (CRF value) for adaptive processing.

**Mechanism:**
```
Input: [B, 3, T, H, W]         Label: [B, num_levels]
            │                          │
            │                    [One-Hot Embedding]
            │                          │
            │              [B, num_levels, 1, 1, 1]
            │                          │
            │              [Expand to video dimensions]
            │                          │
            │              [B, num_levels, T, H, W]
            │                          │
            └──────────[Concat]────────┘
                        │
                        ▼
              [B, 3+num_levels, T, H, W]
                        │
                        ▼
                  [Conv_1]
```

**Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| use_bitrate_labels | False | Enable label injection |
| num_bitrate_levels | 3 | Number of CRF classes (e.g., CRF 0, 5, 10) |

---

## Integration with PhysFormer

**Class:** `PhysFormerWithSTVEN`

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Video   │────▶│      STVEN      │────▶│    PhysFormer   │
│  [B, 3, T, H, W]│     │  (Enhancement)  │     │   (rPPG Net)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                     │
                                                     ▼
                                    ┌────────────────────────────────┐
                                    │ rPPG: [B, T]                   │
                                    │ score1, score2, score3: Attn   │
                                    └────────────────────────────────┘
```

**Forward Pass:**
```python
enhanced_video = self.stven(x, bitrate_label)
rPPG, score1, score2, score3 = self.physformer(enhanced_video, gra_sharp)
```

---

## Training Configuration

**From `joint_st_phys.yaml`:**
```yaml
MODEL:
  NAME: JointSTPhys
  STVEN:
    in_channels: 3
    out_channels: 3
    base_channels: 16       # Note: smaller than default 64
    num_st_blocks: 6
    frame_length: 160
    use_bitrate_labels: True
    num_bitrate_levels: 3   # CRF 0, 5, 10
    PRETRAINED_PATH: "runs/exp/.../STVEN_pretrain_STVEN_Epoch1.pth"

  PHYSFORMER:
    PRETRAINED_PATH: "final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth"
    PATCH_SIZE: 4
    DIM: 96
    FF_DIM: 144
    NUM_HEADS: 4
    NUM_LAYERS: 12
    THETA: 0.7
```

---

## Design Decisions

### 1. R(2+1)D Factorization

**Rationale:** Standard 3D convolutions couple spatial and temporal learning. R(2+1)D decomposition:
- Reduces parameters while maintaining capacity
- Allows independent learning of spatial appearance and temporal motion
- Proven effective in video understanding tasks (Tran et al., 2018)

### 2. Instance Normalization

**Choice:** InstanceNorm3D instead of BatchNorm3D

**Rationale:**
- Better suited for video style transfer tasks
- Normalizes per-sample rather than per-batch
- More stable with small batch sizes (common in video tasks)

### 3. Global Residual Learning

**Formula:** `Output = Enhancement_Network(Input) + Input`

**Rationale:**
- Easier to learn compression artifacts (residual) than full clean frames
- Provides direct gradient path for training stability
- Standard approach in image restoration

### 4. Symmetric Encoder-Decoder

**Design:** Mirror downsampling with upsampling

**Rationale:**
- Preserves spatial resolution from input to output
- Enables skip connections (not used here, but common in U-Net variants)
- Standard architecture for image-to-image tasks

---

## Comparison to Default STVEN

The implementation here uses `base_channels=16` (vs. original `64`), creating a smaller model:

| Configuration | Base Channels | Approx. Params | Use Case |
|---------------|---------------|----------------|----------|
| Original | 64 | ~10M | Full-capacity enhancement |
| Joint Training | 16 | ~0.6M | Lightweight, joint optimization |

**Rationale for smaller model in joint training:**
- Reduced memory footprint for end-to-end training
- PhysFormer backend provides strong gradient signal
- Prevents overfitting on limited compressed video data

---

## Key Functions Summary

| Function | Purpose |
|----------|---------|
| `STBlock.__init__` | Initialize R(2+1)D block with mid-channel calculation |
| `STBlock.forward` | Apply spatial conv → temporal conv with residual |
| `ConvBlock.__init__` | Initialize Conv3D + Norm + ReLU |
| `ConvBlock.forward` | Standard encoder forward pass |
| `DeconvBlock.__init__` | Initialize ConvTranspose3D + optional Norm + ReLU |
| `DeconvBlock.forward` | Decoder upsample forward pass |
| `STVEN.__init__` | Build full encoder-bottleneck-decoder |
| `STVEN._initialize_weights` | Kaiming initialization for all layers |
| `STVEN.forward` | Full forward pass with optional label injection |
| `PhysFormerWithSTVEN.__init__` | Combine STVEN and PhysFormer |
| `PhysFormerWithSTVEN.forward` | Cascade: enhance → predict rPPG |

---

## Usage Examples

### Standalone STVEN Inference
```python
from neural_methods.model.STVEN import STVEN

model = STVEN(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    num_st_blocks=6
)

# Load pretrained weights
checkpoint = torch.load("STVEN_pretrain.pth")
model.load_state_dict(checkpoint)

# Forward pass
input_video = torch.randn(1, 3, 160, 128, 128)  # [B, C, T, H, W]
enhanced = model(input_video)
```

### Joint STVEN + PhysFormer
```python
from neural_methods.model.STVEN import PhysFormerWithSTVEN

stven_config = {
    'in_channels': 3,
    'out_channels': 3,
    'base_channels': 16,
    'num_st_blocks': 6,
    'frame_length': 160,
    'use_bitrate_labels': True,
    'num_bitrate_levels': 3
}

physformer_config = {
    'IMAGE_SIZE': [160, 128, 128],
    'PATCHES': [4, 4, 4],
    'DIM': 96,
    'FF_DIM': 144,
    'NUM_HEADS': 4,
    'NUM_LAYERS': 12,
    'DROPOUT_RATE': 0.2,
    'THETA': 0.7
}

model = PhysFormerWithSTVEN(stven_config, physformer_config)

# Forward with bitrate conditioning
input_video = torch.randn(1, 3, 160, 128, 128)
bitrate_label = torch.tensor([[1, 0, 0]])  # One-hot: CRF 0
rppg, s1, s2, s3 = model(input_video, bitrate_label, gra_sharp=2.0)
```

---

## References

1. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR.
2. Yu, Z., et al. (2022). "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer." CVPR.
3. Luo, Z., et al. (2024). "PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba."
