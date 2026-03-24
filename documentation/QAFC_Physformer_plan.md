# QAFC-PhysFormer: Quality-Aware Feature Conditioning for rPPG-Toolbox

**Status:** Implementation Plan (Localized for rPPG-Toolbox)
**Date:** 2026-03-24
**Author:** Compression-Robust rPPG Project
**Related:** `documentation/rppg_toolbox_pipeline.md` (integration reference)

---

## Overview

This document describes the QAFC-PhysFormer architecture for compression-robust rPPG extraction, adapted for integration into the rPPG-Toolbox framework. QAFC replaces explicit pixel-level video enhancement (STVEN) with feature-wise quality conditioning using FiLM layers.

**Key Innovation:** Instead of enhancing compressed video pixels before feeding to PhysFormer, QAFC learns to modulate PhysFormer's internal features based on estimated compression quality.

---

## Integration Summary (vs. Original Plan)

| Aspect | Original QAFC Plan | rPPG-Toolbox Localization |
|--------|-------------------|--------------------------|
| PhysFormer Base | Custom ViT wrapper | `ViT_ST_ST_Compact3_TDC_gra_sharp` from `neural_methods/model/PhysFormer.py` |
| Data Loader | Custom `MultiCRFRPPGDataset` | Extend `STVENLoader` pattern with quality pair sampling |
| Trainer | Custom `QAFCTrainer` class | Inherit from `BaseTrainer`, follow `JointSTVENPhysFormerTrainer` pattern |
| Config | Standalone Python dict | YACS `CfgNode` in `config.py` |
| Loss | Custom `CombinedLoss` | Use existing `Neg_Pearson` + quality ranking losses |
| Three-Phase Training | Full three-phase | Simplified two-phase (joint + fine-tune) |

---

## Problem Statement

### Current Issue: CRF Relationship Loss

The current STVEN implementation in rPPG-Toolbox uses naive one-hot encoding for different CRF (Constant Rate Factor) levels:

```python
# Current approach in STVEN (neural_methods/model/STVEN.py)
# Label: [B, num_bitrate_levels] with one-hot encoding
# e.g., CRF 0 → [1, 0, 0], CRF 5 → [0, 1, 0], CRF 10 → [0, 0, 1]
bitrate_label = torch.tensor([[1, 0, 0]])  # One-hot for CRF 0
```

**Problems:**
1. **No ordinal relationship:** One-hot encoding treats CRF 0, 5, 10 as independent categories, not ordered compression levels
2. **No interpolation:** Cannot generalize to unseen CRF values (e.g., CRF 7)
3. **No physical meaning:** The model doesn't learn what compression actually does to video
4. **Poor extrapolation:** Fails on CRF values outside training range

### Evidence from Current Results

From `results/PhysFormer_batch.csv`:
| CRF | MAE | Pearson |
|-----|-----|---------|
| 0 | 1.84 | 0.976 |
| 16 | 8.16 | 0.551 |
| 18 | 11.43 | 0.466 |
| 20 | 22.91 | 0.122 |
| 22 | 26.68 | 0.158 |
| 24 | 28.94 | 0.156 |

The sharp degradation suggests the model hasn't learned the *continuous relationship* between compression strength and signal quality.

---

## Proposed Solution: QAFC + PhysFormer

### Core Insight

**STVEN's Fundamental Contradiction:**

```
STVEN's Goal: minimize ||enhanced_frames - original_frames|| (pixel fidelity)
rPPG's Need:  preserve subtle color/brightness changes (~0.1% signal)
```

These goals are **conflicting**:

| What STVEN Does | Impact on rPPG |
|-----------------|----------------|
| Removes blocking artifacts | Good - reduces noise |
| Smooths texture regions | **Bad** - rPPG signal is in subtle skin texture variations |
| Reconstructs high-frequency details | **Dangerous** - may hallucinate non-existent rPPG signals |
| Pixel-level reconstruction loss | **Irrelevant** - rPPG cares about temporal color changes, not pixel accuracy |

**Key Insight:** STVEN operates in pixel space, but rPPG needs feature space quality conditioning. Enhanced pixels ≠ enhanced rPPG signal.

### QAFC Architecture

**Core Approach: Feature-wise Modulation, Not Pixel Enhancement**

```
                                    ┌─────────────────────┐
                                    │   Quality Branch    │
Compressed Video ──────────────────►│   (learn quality    │──── quality_emb
       │                            │    from pixels)     │        [B, C_q, T, H', W']
       │                            └─────────────────────┘              │
       │                                                                 │
       │         ┌───────────────────────────────────────────────────────┘
       │         │  FiLM conditioning (scale + shift)
       │         │
       ▼         ▼
   PhysFormer Backbone (frozen or fine-tuned)
       │
       ▼
   rPPG Signal [B, T]
```

**Key Differences:**

| STVEN | QAFC |
|-------|------|
| Pixel space reconstruction → loses rPPG signal | Feature space modulation → preserves rPPG signal |
| Heavy Encoder-Decoder, many parameters | Lightweight CNN + GRU, few parameters |
| Needs pixel-level GT (original video) | Only needs ranking pairs (self-supervised) |
| Enhanced video is separate intermediate product | Quality embedding and rPPG jointly optimized end-to-end |
| Treats all frames equally | Per-frame quality awareness (I-frame vs P-frame differences) |

---

## Architecture Components (rPPG-Toolbox Adaptation)

### 3.1 QualitySpatialEncoder

Adapted for rPPG-Toolbox's NCDHW data format (batch-first, channel-second):

```python
class QualitySpatialEncoder(nn.Module):
    """
    Extracts quality-related spatial features from compressed frames.

    Adapted for rPPG-Toolbox:
    - Input: [B, C, T, H, W] (NCDHW format, consistent with STVENLoader)
    - Output: [B, C_q, T, H', W']
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        use_block_pattern: bool = True
    ):
        super().__init__()
        self.use_block_pattern = use_block_pattern

        # Path A: Block-level artifacts (stride=8, matching H.264 8x8 blocks)
        if self.use_block_pattern:
            self.block_path = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=(1, 8, 8), stride=(1, 8, 8), padding=0),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
            )
            block_out = 32
        else:
            block_out = 0

        # Path B: Texture/blur analysis (larger receptive field)
        self.texture_path = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 15, 15), stride=(1, 4, 4), padding=(0, 7, 7)),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Merge paths
        combined = block_out + 32
        self.fusion = nn.Sequential(
            nn.Conv3d(combined, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video [B, C, T, H, W]
        Returns:
            Spatial quality features [B, out_channels, T, H', W']
        """
        B, C, T, H, W = x.shape
        features = []

        if self.use_block_pattern:
            block_feat = self.block_path(x)
            block_feat = F.interpolate(block_feat, size=(T, H, W), mode='trilinear', align_corners=False)
            features.append(block_feat)

        texture_feat = self.texture_path(x)
        texture_feat = F.interpolate(texture_feat, size=(T, H, W), mode='trilinear', align_corners=False)
        features.append(texture_feat)

        combined = torch.cat(features, dim=1)
        return self.fusion(combined)
```

### 3.2 QualityTemporalEncoder

```python
class QualityTemporalEncoder(nn.Module):
    """
    Temporal quality modeling using Bidirectional GRU.

    Captures H.264 quality patterns:
    - I-frame: highest quality (intra-coded, no reference error accumulation)
    - P-frame: quality decays progressively (depends on previous frame reconstruction)
    - Scene change: quality resets (new GOP starts)
    - High motion regions: worse quality (large motion compensation residual)
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Global spatial pooling per frame

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        gru_output = hidden_size * 2 if bidirectional else hidden_size
        self.spatial_project = nn.Sequential(
            nn.Conv1d(gru_output, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial quality features [B, C_q, T, H, W]
        Returns:
            Temporal quality features [B, C_q, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Spatial pooling per frame: [B, C, T, H, W] -> [B, T, C]
        x_pooled = self.spatial_pool(x).view(B, C, T).permute(0, 2, 1)  # [B, T, C]

        # GRU temporal modeling
        gru_out, _ = self.gru(x_pooled)  # [B, T, hidden*2]

        # Project back and broadcast to spatial dimensions
        projected = self.spatial_project(gru_out.view(B*T, -1)).view(B, T, -1, 1, 1)
        projected = projected.permute(0, 2, 1, 3, 4)  # [B, C, T, 1, 1]

        return F.interpolate(projected, size=(T, H, W), mode='trilinear', align_corners=False)
```

### 3.3 QualityScalarHead

```python
class QualityScalarHead(nn.Module):
    """
    Projects quality features to scalar quality score.
    Used for quality ranking loss.
    """

    def __init__(self, in_channels: int = 64, hidden_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B, C, 1, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)  # Scalar quality score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Quality features [B, C_q, T, H, W]
        Returns:
            Quality score [B, 1]
        """
        return self.head(x)
```

### 3.4 FiLMLayer

```python
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for rPPG-Toolbox PhysFormer.

    Applies quality-aware affine transformation:
        FiLM(x, q) = gamma(q) * x + beta(q)

    Physical meaning:
    - scale (gamma): "How unreliable is this feature channel under low quality?"
      → Low quality: scale approaches 0,屏蔽 unreliable features
      → High quality: scale approaches 1, preserve full features
    - shift (beta): "Systematic bias from low quality"
      → Compensates for H.264 quantization-induced color bias
    """

    def __init__(self, feature_channels: int, quality_channels: int = 64, reduction_ratio: int = 4):
        super().__init__()
        self.quality_pool = nn.AdaptiveAvgPool3d(1)  # [B, C_q, 1, 1, 1]
        hidden_channels = max(quality_channels // reduction_ratio, 16)

        # Gamma (scale) projection - outputs in [0, 2], centered around 1.0
        self.gamma_net = nn.Sequential(
            nn.Conv3d(quality_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, feature_channels, kernel_size=1),
            nn.Sigmoid()  # Scale factor in [0, 1], will be *2
        )

        # Beta (shift) projection
        self.beta_net = nn.Sequential(
            nn.Conv3d(quality_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, feature_channels, kernel_size=1)
        )

    def forward(self, features: torch.Tensor, quality_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features [B, C_f, T, H, W]
            quality_features: Quality conditioning [B, C_q, T', H', W']
        Returns:
            Modulated features [B, C_f, T, H, W]
        """
        # Interpolate quality features to match temporal resolution
        if quality_features.shape[2] != features.shape[2]:
            quality_features = F.interpolate(
                quality_features, size=(features.shape[2], features.shape[3], features.shape[4]),
                mode='trilinear', align_corners=False
            )

        quality_pooled = self.quality_pool(quality_features)  # [B, C_q, 1, 1, 1]
        gamma = self.gamma_net(quality_pooled)  # [B, C_f, 1, 1, 1]
        beta = self.beta_net(quality_pooled)  # [B, C_f, 1, 1, 1]

        return gamma * features + beta
```

### 3.5 QualityAwarePhysFormer (Wrapper for rPPG-Toolbox PhysFormer)

```python
class QualityAwarePhysFormer(nn.Module):
    """
    Wraps rPPG-Toolbox's ViT_ST_ST_Compact3_TDC_gra_sharp with FiLM layers.

    Inserts FiLM after:
    1. transformer1 (spatial transformer)
    2. transformer2 (temporal transformer)
    """

    def __init__(
        self,
        physformer_config: dict,
        quality_channels: int = 64,
        film_after_spatial: bool = True,
        film_after_temporal: bool = True
    ):
        super().__init__()
        self.film_after_spatial = film_after_spatial
        self.film_after_temporal = film_after_temporal

        # Load base PhysFormer from rPPG-Toolbox
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        self.physformer = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=tuple(physformer_config['IMAGE_SIZE']),
            patches=tuple(physformer_config['PATCHES']),
            dim=physformer_config['DIM'],
            ff_dim=physformer_config['FF_DIM'],
            num_heads=physformer_config['NUM_HEADS'],
            num_layers=physformer_config['NUM_LAYERS'],
            dropout_rate=physformer_config['DROPOUT_RATE'],
            theta=physformer_config['THETA']
        )

        dim = physformer_config['DIM']

        # FiLM layers
        if self.film_after_spatial:
            self.film_spatial = FiLMLayer(feature_channels=dim, quality_channels=quality_channels)
        if self.film_after_temporal:
            self.film_temporal = FiLMLayer(feature_channels=dim, quality_channels=quality_channels)

    def forward(self, x: torch.Tensor, quality_features: torch.Tensor, gra_sharp: float = 2.0):
        """
        Args:
            x: Input video [B, C, T, H, W]
            quality_features: Quality features [B, C_q, T', H', W']
            gra_sharp: Gradient sharpening parameter for attention
        Returns:
            rPPG, Score1, Score2, Score3
        """
        b, c, t, fh, fw = x.shape

        # PhysFormer stem
        x = self.physformer.Stem0(x)
        x = self.physformer.Stem1(x)
        x = self.physformer.Stem2(x)  # [B, 64, 160, 64, 64]

        # Patch embedding
        x = self.physformer.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 640, 64]

        # Transformer 1 (Spatial) + FiLM
        trans_features, score1 = self.physformer.transformer1(x, gra_sharp)
        if self.film_after_spatial:
            trans_features = trans_features.view(b, 64, t//4, 4, 4)
            trans_features = self.film_spatial(trans_features, quality_features)
            trans_features = trans_features.flatten(2).transpose(1, 2)

        # Transformer 2 (Temporal) + FiLM
        trans_features2, score2 = self.physformer.transformer2(trans_features, gra_sharp)
        if self.film_after_temporal:
            trans_features2 = trans_features2.view(b, 64, t//4, 4, 4)
            trans_features2 = self.film_temporal(trans_features2, quality_features)
            trans_features2 = trans_features2.flatten(2).transpose(1, 2)

        # Transformer 3 (No FiLM)
        trans_features3, score3 = self.physformer.transformer3(trans_features2, gra_sharp)

        # Heads
        features_last = trans_features3.transpose(1, 2).view(b, self.physformer.dim, t//4, 4, 4)
        features_last = self.physformer.upsample(features_last)
        features_last = self.physformer.upsample2(features_last)
        features_last = torch.mean(features_last, dim=3)
        features_last = torch.mean(features_last, dim=3)
        rPPG = self.physformer.ConvBlockLast(features_last)
        rPPG = rPPG.squeeze(1)

        return rPPG, score1, score2, score3
```

### 3.6 QAFCPhysFormer (Main Model)

```python
class QAFCPhysFormer(nn.Module):
    """
    Quality-Aware Feature Conditioning PhysFormer for rPPG-Toolbox.

    Combines:
    1. Quality Branch: Spatial Encoder → Temporal Encoder → Quality Features
    2. Quality-Aware Backbone: PhysFormer with FiLM conditioning
    3. Quality Head: Scalar quality score for ranking loss
    """

    def __init__(
        self,
        physformer_config: dict,
        quality_spatial_channels: int = 64,
        quality_temporal_hidden: int = 128,
        use_block_pattern: bool = True,
        film_after_spatial: bool = True,
        film_after_temporal: bool = True
    ):
        super().__init__()

        # Quality Branch
        self.quality_spatial_encoder = QualitySpatialEncoder(
            in_channels=3,
            out_channels=quality_spatial_channels,
            use_block_pattern=use_block_pattern
        )
        self.quality_temporal_encoder = QualityTemporalEncoder(
            in_channels=quality_spatial_channels,
            hidden_size=quality_temporal_hidden,
            bidirectional=True
        )
        self.quality_head = QualityScalarHead(
            in_channels=quality_spatial_channels,
            hidden_channels=quality_temporal_hidden
        )

        # Quality-Aware Backbone
        self.quality_aware_backbone = QualityAwarePhysFormer(
            physformer_config=physformer_config,
            quality_channels=quality_spatial_channels,
            film_after_spatial=film_after_spatial,
            film_after_temporal=film_after_temporal
        )

    def forward(self, x: torch.Tensor, gra_sharp: float = 2.0):
        """
        Args:
            x: Input video [B, C, T, H, W]
            gra_sharp: Gradient sharpening parameter
        Returns:
            rPPG [B, T], quality_score [B, 1], Score1, Score2, Score3
        """
        # Quality branch
        quality_spatial = self.quality_spatial_encoder(x)
        quality_temporal = self.quality_temporal_encoder(quality_spatial)
        quality_score = self.quality_head(quality_temporal)

        # Quality-aware backbone
        rPPG, score1, score2, score3 = self.quality_aware_backbone(
            x, quality_temporal, gra_sharp
        )

        return rPPG, quality_score, score1, score2, score3
```

---

## Loss Functions (rPPG-Toolbox Integration)

### 4.1 Base Loss: NegPearson (Already in rPPG-Toolbox)

Use existing `neural_methods.loss.PhysNetNegPearsonLoss.Neg_Pearson`:

```python
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson

criterion_Pearson = Neg_Pearson()
loss_rppg = criterion_Pearson(pred_rPPG, gt_rPPG)
```

### 4.2 Quality Ranking Loss

```python
class QualityRankingLoss(nn.Module):
    """
    Self-supervised quality ranking loss.

    For two compressed versions of the same video (different CRF),
    the higher quality version should have higher quality score.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, score_high: torch.Tensor, score_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_high: Quality score for higher quality video [B, 1]
            score_low: Quality score for lower quality video [B, 1]
        Returns:
            Ranking loss (scalar)
        """
        # score_high should be > score_low
        loss = F.relu(score_low - score_high + self.margin)
        return loss.mean()
```

### 4.3 Combined Loss for QAFC

```python
class QAFCLoss(nn.Module):
    """
    Combined loss for QAFC training.

    Uses uncertainty weighting (Kendall et al., 2018) to balance tasks.
    """

    def __init__(self, ranking_margin: float = 0.1, diversity_weight: float = 0.01):
        super().__init__()
        self.rppg_loss = Neg_Pearson()
        self.ranking_loss = QualityRankingLoss(margin=ranking_margin)
        self.diversity_weight = diversity_weight

        # Learnable uncertainty weights
        self.log_var_rppg = nn.Parameter(torch.tensor(0.0))
        self.log_var_ranking = nn.Parameter(torch.tensor(0.0))

    def _weighted(self, loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return torch.exp(-log_var) * loss + log_var

    def forward(
        self,
        pred_rppg: torch.Tensor,
        gt_rppg: torch.Tensor,
        quality_score_high: torch.Tensor,
        quality_score_low: torch.Tensor
    ) -> dict:
        l_rppg = self.rppg_loss(pred_rppg, gt_rppg)
        l_rank = self.ranking_loss(quality_score_high, quality_score_low)

        total = (
            self._weighted(l_rppg, self.log_var_rppg) +
            self._weighted(l_rank, self.log_var_ranking)
        )

        return {
            'total': total,
            'rppg': l_rppg.detach(),
            'ranking': l_rank.detach(),
        }
```

---

## Data Pipeline (rPPG-Toolbox STVENLoader Extension)

### 5.1 Extending STVENLoader for QAFC

The existing `STVENLoader` already handles multi-CRF data. We extend it to support quality pair sampling:

```python
# dataset/data_loader/QAFCPhysFormerLoader.py
from dataset.data_loader.STVENLoader import STVENLoader
import random

class QAFCPhysFormerLoader(STVENLoader):
    """
    Data loader for QAFC-PhysFormer training.

    Extends STVENLoader to support:
    - Quality pair sampling (two different CRF levels for same video)
    - Ranking loss data preparation
    """

    def __getitem__(self, index):
        # Load base data (compressed, uncompressed, bitrate_label, bvp_label)
        compressed_data, uncompressed_data, bitrate_label, bvp_label = super().__getitem__(index)

        # For QAFC: also sample a paired CRF level for ranking loss
        # Parse current CRF from filename
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        match = re.search(r'_crf(\d+)_', item_path_filename)
        current_crf = int(match.group(1)) if match else 0

        # Sample a different CRF level for ranking pair
        available_crfs = self.config_data.CRF_LEVELS
        other_crfs = [c for c in available_crfs if c != current_crf]
        paired_crf = random.choice(other_crfs)

        # Load paired video
        paired_filename_base = item_path_filename.replace(f"_crf{current_crf}_", f"_crf{paired_crf}_")
        paired_path = os.path.join(self.cached_path, paired_filename_base)

        if os.path.exists(paired_path):
            paired_data = np.load(paired_path)
            # Process same format as compressed_data
            if self.data_format == 'NDCHW':
                paired_data = np.transpose(paired_data, (0, 3, 1, 2))
            paired_data = np.float32(paired_data)
        else:
            paired_data = compressed_data.copy()  # Fallback

        # Determine which is higher quality (lower CRF = higher quality)
        if current_crf < paired_crf:
            video_high, video_low = compressed_data, paired_data
        else:
            video_high, video_low = paired_data, compressed_data

        return {
            'video_high': torch.from_numpy(video_high),
            'video_low': torch.from_numpy(video_low),
            'bvp_label': torch.from_numpy(bvp_label),
            'crf_current': current_crf,
            'crf_paired': paired_crf,
        }
```

---

## Training Strategy (rPPG-Toolbox Adaptation)

### 6.1 Two-Phase Training (Simplified from Original Three-Phase)

| Phase | Epochs | Trainable Components | Learning Rate | Purpose |
|-------|--------|---------------------|---------------|---------|
| Phase 1: Joint Training | 0-70 | Quality Branch + FiLM + Backbone | 1e-4 (quality), 1e-5 (backbone) | Learn quality conditioning while preserving rPPG |
| Phase 2: Fine-tuning | 70-100 | FiLM + Backbone (Quality frozen) | 5e-5 (FiLM), 1e-5 (backbone) | Refine rPPG under stable quality conditioning |

### 6.2 QAFCPhysFormerTrainer Pattern

Following `JointSTVENPhysFormerTrainer` pattern from rPPG-Toolbox:

```python
# neural_methods/trainer/QAFCPhysFormerTrainer.py
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson

class QAFCPhysFormerTrainer(BaseTrainer):
    """
    Trainer for QAFC-PhysFormer.

    Follows rPPG-Toolbox BaseTrainer pattern with:
    - train(): Training loop with two-phase support
    - valid(): Validation loop
    - test(): Testing/evaluation loop
    """

    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME

        # Initialize model
        from neural_methods.model.QAFCPhysFormer import QAFCPhysFormer
        self.model = QAFCPhysFormer(
            physformer_config=config.MODEL.QAFC_PHYSFORMER,
            quality_spatial_channels=config.MODEL.QAFC_PHYSFORMER.QUALITY_CHANNELS,
            quality_temporal_hidden=config.MODEL.QAFC_PHYSFORMER.HIDDEN_SIZE,
        ).to(self.device)

        # Load pretrained PhysFormer weights if specified
        self._load_pretrained_physformer(config)

        # Phase tracking
        self.current_phase = 1
        self.phase_boundary = config.TRAIN.QAFC.PHASE_BOUNDARY  # e.g., 70

        # Optimizer with parameter groups
        self.optimizer = self._create_optimizer(config)

        # Loss functions
        self.criterion_Pearson = Neg_Pearson()
        self.criterion_Ranking = QualityRankingLoss(margin=config.TRAIN.QAFC.RANKING_MARGIN)

    def _load_pretrained_physformer(self, config):
        """Load pretrained PhysFormer backbone weights."""
        if config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH:
            phys_state = torch.load(config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH, map_location=self.device)
            if 'state_dict' in phys_state:
                phys_state = phys_state['state_dict']
            # Load into backbone only
            backbone_keys = {k.replace('quality_aware_backbone.physformer.', ''): v
                           for k, v in phys_state.items()
                           if k.startswith('quality_aware_backbone.physformer.')}
            self.model.quality_aware_backbone.physformer.load_state_dict(backbone_keys, strict=False)
            print(f"Loaded pretrained PhysFormer weights from {config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH}")

    def _create_optimizer(self, config):
        """Create optimizer with phase-aware parameter groups."""
        quality_params = list(self.model.quality_spatial_encoder.parameters()) + \
                        list(self.model.quality_temporal_encoder.parameters()) + \
                        list(self.model.quality_head.parameters())
        film_params = list(self.model.quality_aware_backbone.film_spatial.parameters()) + \
                     list(self.model.quality_aware_backbone.film_temporal.parameters())
        backbone_params = [p for n, p in self.model.quality_aware_backbone.physformer.named_parameters()]

        return optim.AdamW([
            {'params': quality_params, 'lr': config.TRAIN.LR},
            {'params': film_params, 'lr': config.TRAIN.LR},
            {'params': backbone_params, 'lr': config.TRAIN.LR * 0.1},  # Backbone learns slower
        ], weight_decay=1e-4)

    def train(self, data_loader):
        """Training loop with two-phase support."""
        for epoch in range(self.max_epoch_num):
            # Phase switching
            if epoch >= self.phase_boundary and self.current_phase == 1:
                self.current_phase = 2
                # Freeze quality branch in Phase 2
                for param in self.model.quality_spatial_encoder.parameters():
                    param.requires_grad = False
                for param in self.model.quality_temporal_encoder.parameters():
                    param.requires_grad = False
                for param in self.model.quality_head.parameters():
                    param.requires_grad = False
                # Adjust learning rates
                self.optimizer = self._create_optimizer(self.config)

            self.model.train()
            # Training batches...
            # (similar to JointSTVENPhysFormerTrainer.train())
```

---

## Configuration (YACS for rPPG-Toolbox)

### 7.1 config.py Additions

```python
# In config.py, add MODEL.QAFC_PHYSFORMER section

_C.MODEL.QAFC_PHYSFORMER = CfgNode()
_C.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH = ""
_C.MODEL.QAFC_PHYSFORMER.IMAGE_SIZE = [160, 128, 128]  # [T, H, W]
_C.MODEL.QAFC_PHYSFORMER.PATCHES = [4, 16, 16]
_C.MODEL.QAFC_PHYSFORMER.DIM = 64
_C.MODEL.QAFC_PHYSFORMER.FF_DIM = 256
_C.MODEL.QAFC_PHYSFORMER.NUM_HEADS = 4
_C.MODEL.QAFC_PHYSFORMER.NUM_LAYERS = 12
_C.MODEL.QAFC_PHYSFORMER.DROPOUT_RATE = 0.2
_C.MODEL.QAFC_PHYSFORMER.THETA = 0.2
_C.MODEL.QAFC_PHYSFORMER.QUALITY_CHANNELS = 64
_C.MODEL.QAFC_PHYSFORMER.HIDDEN_SIZE = 128
_C.MODEL.QAFC_PHYSFORMER.USE_BLOCK_PATTERN = True

# Training-specific config
_C.TRAIN.QAFC = CfgNode()
_C.TRAIN.QAFC.PHASE_BOUNDARY = 70  # Epoch to switch from Phase 1 to Phase 2
_C.TRAIN.QAFC.RANKING_MARGIN = 0.1
```

### 7.2 Sample Training Config

```yaml
# configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml

TOOLBOX_MODE: "train_and_test"

TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 100
  LR: 1e-4
  MODEL_FILE_NAME: "QAFCPhysFormer"
  QAFC:
    PHASE_BOUNDARY: 70
    RANKING_MARGIN: 0.1

VALID:
  BATCH_SIZE: 4

TEST:
  BATCH_SIZE: 1
  DATA:
    DATASET: "PURE"
    DO_PREPROCESS: False
  BEGIN: 0.0
  END: 1.0

MODEL:
  NAME: "QAFCPhysFormer"
  QAFC_PHYSFORMER:
    PRETRAINED_PATH: "/path/to/pretrained/PhysFormer.pth"
    IMAGE_SIZE: [160, 128, 128]
    PATCHES: [4, 16, 16]
    DIM: 64
    FF_DIM: 256
    NUM_HEADS: 4
    NUM_LAYERS: 12
    DROPOUT_RATE: 0.2
    THETA: 0.2
    QUALITY_CHANNELS: 64
    HIDDEN_SIZE: 128
    USE_BLOCK_PATTERN: True

TRAIN.DATA:
  DATASET: "UBFC-rPPG"
  DO_PREPROCESS: True
  DATA_PATH: "/path/to/UBFC-rPPG"
  BEGIN: 0.0
  END: 0.7
  FS: 30

VALID.DATA:
  DATASET: "UBFC-rPPG"
  DO_PREPROCESS: False
  DATA_PATH: "/path/to/UBFC-rPPG"
  BEGIN: 0.7
  END: 1.0

TEST.DATA:
  DATASET: "PURE"
  DO_PREPROCESS: False
  DATA_PATH: "/path/to/PURE"
  BEGIN: 0.0
  END: 1.0

# CRF datasets for compression-aware training
CRF_DATASETS:
  0: "/path/to/UBFC-rPPG-crf0"
  10: "/path/to/UBFC-rPPG-crf10"
  20: "/path/to/UBFC-rPPG-crf20"
  30: "/path/to/UBFC-rPPG-crf30"

CRF_LEVELS: [0, 10, 20, 30]
```

---

## Integration Checklist (Step-by-Step)

### Phase 1: Model Implementation

- [ ] **Create `neural_methods/model/QAFCPhysFormer.py`**
  - Implement `QualitySpatialEncoder`
  - Implement `QualityTemporalEncoder`
  - Implement `QualityScalarHead`
  - Implement `FiLMLayer`
  - Implement `QualityAwarePhysFormer`
  - Implement `QAFCPhysFormer`

- [ ] **Test model forward pass**
  ```python
  model = QAFCPhysFormer(physformer_config)
  x = torch.randn(2, 3, 160, 128, 128)
  rPPG, quality_score, s1, s2, s3 = model(x)
  assert rPPG.shape == (2, 160)
  assert quality_score.shape == (2, 1)
  ```

### Phase 2: Trainer Implementation

- [ ] **Create `neural_methods/trainer/QAFCPhysFormerTrainer.py`**
  - Inherit from `BaseTrainer`
  - Implement `_load_pretrained_physformer()`
  - Implement `_create_optimizer()`
  - Implement `train()` with two-phase support
  - Implement `valid()`
  - Implement `test()`

- [ ] **Create loss module `neural_methods/loss/QAFCPhysFormerLoss.py`**
  - Implement `QualityRankingLoss`
  - Implement `QAFCLoss` (combined loss)

### Phase 3: Configuration

- [ ] **Update `config.py`**
  - Add `MODEL.QAFC_PHYSFORMER` section
  - Add `TRAIN.QAFC` section

- [ ] **Create sample config `configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml`**

### Phase 4: Main.py Integration

- [ ] **Register QAFCPhysFormer in `main.py`**
  ```python
  elif config.MODEL.NAME == "QAFCPhysFormer":
      model_trainer = trainer.QAFCPhysFormerTrainer.QAFCPhysFormerTrainer(config, data_loader_dict)
  ```

### Phase 5: Testing

- [ ] **Run forward pass test**
  ```bash
  python main.py --config_file configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_QAFCPHYSFORMER.yaml
  ```

- [ ] **Compare with baseline**
  - Raw PhysFormer (no compression adaptation)
  - STVEN + PhysFormer (pixel enhancement)
  - QAFC + PhysFormer (feature conditioning) ← Expected best

---

## Expected Results

Based on the architecture design, QAFC-PhysFormer should show:

1. **Better compression robustness** than raw PhysFormer (especially CRF 18+)
2. **Comparable or better performance** than STVEN+PhysFormer (more efficient, feature-level conditioning)
3. **Monotonic quality scores** - predicted quality should correlate with CRF level
4. **Faster inference** than STVEN (no pixel-level decoder)

| Model | CRF 0 (Pearson) | CRF 20 (Pearson) | CRF 28 (Pearson) | Params |
|-------|-----------------|------------------|------------------|--------|
| PhysFormer (raw) | 0.97 | 0.12 | 0.08 | 5.2M |
| STVEN+PhysFormer | 0.98 | 0.65 | 0.45 | 12.8M |
| **QAFC+PhysFormer** | **0.97** | **0.70** | **0.50** | **6.1M** |

---

## References

- rPPG-Toolbox Pipeline: `documentation/rppg_toolbox_pipeline.md`
- Original PhysFormer: https://github.com/ZitongYu/PhysFormer
- STVEN Implementation: `neural_methods/model/STVEN.py`
- Joint Training Pattern: `neural_methods/trainer/JointSTVENPhysFormerTrainer.py`
- FiLM: Perez et al., "Film: Visual reasoning with a general conditioning layer", 2018
