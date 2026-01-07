"""STVEN: Spatio-Temporal Video Enhancement Network
Based on the 3D-CNN architecture for enhancing compressed video quality
before feeding to PhysFormer for rPPG extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class STBlock(nn.Module):
    """
    Spatio-Temporal Block (R(2+1)D Block)
    
    Implementation of the (2+1)D Residual Block described in:
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (Tran et al., 2018)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3), # (t, h, w)
        stride: tuple = (1, 1, 1),      # (t, h, w)
        padding: tuple = (1, 1, 1)      # (t, h, w)
    ):
        """
            Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Overall 3D kernel size (t, h, w)
            stride: Overall 3D stride (t, h, w)
            padding: Overall 3D padding (t, h, w)
        """
        super(STBlock, self).__init__()

        # Formula: Mi = floor( (t * d^2 * Ni-1 * Ni) / (d^2 * Ni-1 + t * Ni) )
        t, d, _ = kernel_size
        numerator = t * d**2 * in_channels * out_channels
        denominator = d**2 * in_channels + t * out_channels
        mid_channels = math.floor(numerator / denominator)

        self.conv2D = nn.Conv3d(
            in_channels,
            mid_channels,
            (1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False
        )
        self.norm1 = nn.InstanceNorm3d(mid_channels)

        self.conv1D = nn.Conv3d(
            mid_channels,
            out_channels,
            (kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            Output tensor [B, C, T, H, W]
        """
        identity = x

        x = self.conv2D(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv1D(x)
        x = self.norm2(x)

        x += identity
        x = self.relu(x)
        
        return x


class ConvBlock(nn.Module):
    """Encoder Convolution Block
    Used in Conv_1, Conv_2, Conv_3
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),
        stride: tuple = (1, 1, 1),
        padding: tuple | str = (1, 1, 1)
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: 3D convolution kernel size
            stride: 3D convolution stride
            padding: 3D convolution padding
        """
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv(x)
        x = self.norm(x) # Instance normalization
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    """Decoder Deconvolution Block
    Used in DConv_1, DConv_2, DConv_3
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),
        stride: tuple = (1, 1, 1),
        padding: tuple | str = (1, 1, 1),
        output_padding: tuple = (0, 0, 0)
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: 3D transposed convolution kernel size
            stride: 3D transposed convolution stride
            padding: 3D transposed convolution padding
            output_padding: 3D transposed convolution output padding
        """
        super(DeconvBlock, self).__init__()

        self.deconv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size, stride, padding, output_padding
        ) # in our case we don't need output padding. Very usual 2x upsample
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            Output tensor [B, C, T, H*2, W*2] (upsampled)
        """
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x



class STVEN(nn.Module):
    """Spatio-Temporal Video Enhancement Network (STVEN)

    3D-CNN encoder-decoder architecture with:
    - Encoder: Conv_1 -> Conv_2 -> Conv_3
    - Bottleneck: 6 ST_Blocks
    - Decoder: DConv_1 -> DConv_2 -> DConv_3
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_st_blocks: int = 6,
        frame_length: int = 160,
        use_bitrate_labels: bool = False,
        num_bitrate_levels: int = 1
    ):
        """
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            out_channels: Number of output channels (default: 3 for RGB)
            base_channels: Number of base channels (default: 64)
            num_st_blocks: Number of ST blocks in bottleneck (default: 6)
            frame_length: Length of video frames (default: 160)
            use_bitrate_labels: Whether to use bitrate/compression labels
            num_bitrate_levels: Number of bitrate levels (for CRF values)
        """
        super(STVEN, self).__init__()

        self.use_bitrate_labels = use_bitrate_labels
        self.num_bitrate_levels = num_bitrate_levels

        # ============ ENCODER ============
        # Conv_1: Downsample spatial dimensions 
        # Input: 3xTx128x128 or (3+num_bitrate_levels)xTx128x128
        conv1_in_channels = in_channels
        if self.use_bitrate_labels:
            conv1_in_channels += num_bitrate_levels

        self.conv1 = ConvBlock(
            in_channels=conv1_in_channels,
            out_channels=base_channels,
            kernel_size=(3, 7, 7),
            padding="same" # resolution not affected, only dimensional change
        ) # Output: 64xTx128x128

        # Conv_2: Further downsampling
        self.conv2 = ConvBlock( 
            in_channels=base_channels, # 64
            out_channels=base_channels * 2, # 64 * 2 = 128
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2), # to lessen the resolution
            padding=(1, 2, 2) # time dimension not affected
        ) # Output: 128xTx64x64

        # Conv_3: Final encoder block
        self.conv3 = ConvBlock(
            in_channels=base_channels * 2, # 64 * 2 = 128
            out_channels=base_channels * 8, # 64 * 8 = 512
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2), # both time and space resolution are reduced
            padding=(1, 1, 1) # checked. padding = 1 works for stride = 2 both dimensions
        ) # Output: 512 x T/2 x 32 x 32

        # ============ BOTTLENECK ============
        # 6 Spatio-Temporal Blocks, no change in resolution, channel etc.
        self.st_blocks = nn.ModuleList([
            STBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8
            ) for i in range(num_st_blocks) # 6 blocks, 
        ])

        # ============ DECODER ============

        self.dconv1 = DeconvBlock(
            in_channels=base_channels * 8, 
            out_channels=base_channels * 2,
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1)
        )


        self.dconv2 = DeconvBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )


        self.dconv3 = DeconvBlock(
            in_channels=base_channels, 
            out_channels=3,
            kernel_size=(1, 7, 7),
            stride=(1, 1, 1),
            padding=(0, 3, 3)
        ) # Output: 3 x T x 128 x 128

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.InstanceNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, bitrate_label: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through STVEN

        Args:
            x: Input video tensor [B, C, T, H, W]
            bitrate_label: Optional bitrate label [B, num_bitrate_levels] (one-hot for CRF)

        Returns:
            Enhanced video tensor [B, C, T, H, W]
        """

        # ============ ENCODER PATH ============
        # Label Injection
        if self.use_bitrate_labels and bitrate_label is not None:
             B, C, T, H, W = x.shape
             # 1. Unsqueeze: [B, num_classes] -> [B, num_classes, 1, 1, 1]
             label_map = bitrate_label.view(B, -1, 1, 1, 1)
             # 2. Expand: [B, num_classes, 1, 1, 1] -> [B, num_classes, T, H, W]
             label_map = label_map.expand(-1, -1, T, H, W)
             # 3. Concatenate along channel dim (dim=1)
             x = torch.cat([x, label_map], dim=1)

        # Conv_1 -> Conv_2 -> Conv_3
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Bottleneck: Spatio-Temporal Blocks
        for st_block in self.st_blocks:
            x = st_block(x)

        # ============ DECODER PATH ============
        # DConv_1 -> DConv_2 -> DConv_3
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)

        return x


class PhysFormerWithSTVEN(nn.Module):
    """PhysFormer enhanced with STVEN preprocessing

    Combines STVEN (for video enhancement) with PhysFormer (for rPPG prediction).
    STVEN enhances compressed video before PhysFormer processes it.
    """

    def __init__(
        self,
        stven_config: dict,
        physformer_config: dict
    ):
        """
        Args:
            stven_config: Dictionary containing STVEN parameters:
                - in_channels: Input channels (default: 3)
                - out_channels: Output channels (default: 3)
                - base_channels: Base channels (default: 64)
                - num_st_blocks: Number of ST blocks (default: 6)
                - frame_length: Frame length (default: 160)
                - use_bitrate_labels: Use CRF labels (default: False)
                - num_bitrate_levels: Number of CRF levels (default: 1)
            physformer_config: Dictionary containing PhysFormer parameters
        """
        super(PhysFormerWithSTVEN, self).__init__()

        # Import PhysFormer here to avoid circular imports
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp

        # STVEN for video enhancement
        self.stven = STVEN(
            in_channels=stven_config.get('in_channels', 3),
            out_channels=stven_config.get('out_channels', 3),
            base_channels=stven_config.get('base_channels', 64),
            num_st_blocks=stven_config.get('num_st_blocks', 6),
            frame_length=stven_config.get('frame_length', 160),
            use_bitrate_labels=stven_config.get('use_bitrate_labels', False),
            num_bitrate_levels=stven_config.get('num_bitrate_levels', 1)
        )

        # PhysFormer for rPPG prediction
        self.physformer = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=physformer_config['image_size'],
            patches=physformer_config['patches'],
            dim=physformer_config['dim'],
            ff_dim=physformer_config['ff_dim'],
            num_heads=physformer_config['num_heads'],
            num_layers=physformer_config['num_layers'],
            dropout_rate=physformer_config['dropout_rate'],
            theta=physformer_config['theta']
        )

    def forward(self, x: torch.Tensor, bitrate_label: torch.Tensor = None, gra_sharp: float = 2.0):
        """
        Forward pass

        Args:
            x: Input video tensor [B, C, T, H, W]
            bitrate_label: Optional bitrate label [B, num_bitrate_levels]
            gra_sharp: Sharpening parameter for PhysFormer

        Returns:
            rPPG: rPPG signal
            score1, score2, score3: Attention scores from PhysFormer
        """

        # Pass through STVEN for video enhancement
        enhanced_video = self.stven(x, bitrate_label)

        # Pass through PhysFormer for rPPG prediction
        rPPG, score1, score2, score3 = self.physformer(enhanced_video, gra_sharp)

        return rPPG, score1, score2, score3
