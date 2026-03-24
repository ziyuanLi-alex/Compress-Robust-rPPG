"""QAFC-PhysFormer: Quality-Aware Feature Conditioning for rPPG-Toolbox

QAFC replaces explicit pixel-level video enhancement (STVEN) with feature-wise
quality conditioning using FiLM layers. The quality branch learns compression
patterns from pixels and modulates PhysFormer features directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QualitySpatialEncoder(nn.Module):
    """
    Extracts quality-related spatial features from compressed frames.

    Two-path architecture:
    - Path A: Block-level artifacts (8x8 stride for H.264 blocks)
    - Path B: Texture/blur analysis (larger receptive field)

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


class QualityTemporalEncoder(nn.Module):
    """
    Temporal quality modeling using Bidirectional GRU.

    Captures H.264 quality patterns:
    - I-frame: highest quality (intra-coded, no reference error accumulation)
    - P-frame: quality decays progressively (depends on previous frame reconstruction)
    - Scene change: quality resets (new GOP starts)
    - High motion regions: worse quality (large motion compensation residual)

    Input: [B, C_q, T, H, W]
    Output: [B, C_q, T, H, W]
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

        # GRU temporal modeling: [B, T, C] -> [B, T, hidden*2]
        gru_out, _ = self.gru(x_pooled)  # [B, T, hidden*2]

        # Project: [B, T, hidden*2] -> [B, T, C]
        # Reshape to [B*T, hidden*2] -> Conv1d -> [B*T, C] -> reshape back
        gru_reshaped = gru_out.reshape(B * T, -1)  # [B*T, hidden*2]
        projected = self.spatial_project(gru_reshaped.unsqueeze(-1)).squeeze(-1)  # [B*T, C]
        projected = projected.reshape(B, T, C)  # [B, T, C]

        # Broadcast back to spatial dimensions: [B, T, C] -> [B, C, T, 1, 1]
        projected = projected.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]

        return F.interpolate(projected, size=(T, H, W), mode='trilinear', align_corners=False)


class QualityScalarHead(nn.Module):
    """
    Projects quality features to scalar quality score.
    Used for quality ranking loss.

    Input: [B, C_q, T, H, W]
    Output: [B, 1]
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


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for rPPG-Toolbox PhysFormer.

    Applies quality-aware affine transformation:
        FiLM(x, q) = gamma(q) * x + beta(q)

    Physical meaning:
    - scale (gamma): "How unreliable is this feature channel under low quality?"
      -> Low quality: scale approaches 0, mask unreliable features
      -> High quality: scale approaches 1, preserve full features
    - shift (beta): "Systematic bias from low quality"
      -> Compensates for H.264 quantization-induced color bias

    Input:
        features: [B, C_f, T, H, W]
        quality_features: [B, C_q, T', H', W']
    Output:
        modulated_features: [B, C_f, T, H, W]
    """

    def __init__(self, feature_channels: int, quality_channels: int = 64, reduction_ratio: int = 4):
        super().__init__()
        self.quality_pool = nn.AdaptiveAvgPool3d(1)  # [B, C_q, 1, 1, 1]
        hidden_channels = max(quality_channels // reduction_ratio, 16)

        # Gamma (scale) projection - outputs in [0, 1]
        self.gamma_net = nn.Sequential(
            nn.Conv3d(quality_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, feature_channels, kernel_size=1),
            nn.Sigmoid()  # Scale factor in [0, 1]
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
        x = self.physformer.Stem0(x)  # [B, dim//4, T, H//2, W//2]
        x = self.physformer.Stem1(x)  # [B, dim//2, T, H//4, W//4]
        x = self.physformer.Stem2(x)  # [B, dim, T, H//8, W//8]

        # Patch embedding: [B, dim, T//4, H//16, W//16] with patches=[4,4,4]
        x = self.physformer.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', dim]

        # Transformer 1 (Spatial) + FiLM
        trans_features, score1 = self.physformer.transformer1(x, gra_sharp)
        if self.film_after_spatial:
            # Reshape to [B, dim, T', H', W'] for FiLM modulation
            trans_features = trans_features.reshape(b, 64, 40, 4, 4)
            trans_features = self.film_spatial(trans_features, quality_features)
            trans_features = trans_features.reshape(b, 64, 640).transpose(1, 2)

        # Transformer 2 (Temporal) + FiLM
        trans_features2, score2 = self.physformer.transformer2(trans_features, gra_sharp)
        if self.film_after_temporal:
            trans_features2 = trans_features2.reshape(b, 64, 40, 4, 4)
            trans_features2 = self.film_temporal(trans_features2, quality_features)
            trans_features2 = trans_features2.reshape(b, 64, 640).transpose(1, 2)

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


class QAFCPhysFormer(nn.Module):
    """
    Quality-Aware Feature Conditioning PhysFormer for rPPG-Toolbox.

    Combines:
    1. Quality Branch: Spatial Encoder -> Temporal Encoder -> Quality Features
    2. Quality-Aware Backbone: PhysFormer with FiLM conditioning
    3. Quality Head: Scalar quality score for ranking loss

    Input: [B, 3, T, H, W] where T=160, H=W=128
    Output: rPPG [B, T], quality_score [B, 1], Score1, Score2, Score3
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
