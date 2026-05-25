"""Compare MACs: current STVEN (mixed 3D + (2+1)D) vs full (2+1)D encoder/decoder.

Only modifies ConvBlock and DeconvBlock to use (2+1)D decomposition.
STBlock already uses (2+1)D — unchanged.

Usage:
    conda run -n rppg-toolbox python tools/profile_2plus1d.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from thop import profile, clever_format


# ==================== Original (current) ====================

def resolve_padding(kernel_size, padding):
    """Convert 'same' to explicit padding tuple."""
    if padding == "same":
        t, h, w = kernel_size
        return ((t-1)//2, (h-1)//2, (w-1)//2)
    return padding


class ConvBlock_Original(nn.Module):
    """Full 3D Conv (current)"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        pad = resolve_padding(kernel_size, padding)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, pad)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DeconvBlock_Original(nn.Module):
    """Full 3D ConvTranspose (current)"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1,1,1), padding=(1,1,1),
                 output_padding=(0,0,0), with_act=True):
        super().__init__()
        self.with_act = with_act
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        if with_act:
            self.norm = nn.InstanceNorm3d(out_ch)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        if self.with_act:
            x = self.relu(self.norm(x))
        return x


# ==================== (2+1)D version ====================

def calc_mid_channels(in_ch, out_ch, t, d):
    """(2+1)D mid channels formula from Tran et al. 2018"""
    numerator = t * d**2 * in_ch * out_ch
    denominator = d**2 * in_ch + t * out_ch
    return max(1, math.floor(numerator / denominator))


class ConvBlock_2plus1D(nn.Module):
    """(2+1)D Conv: spatial conv + temporal conv"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        t, h, w = kernel_size
        st, sh, sw = stride
        pt, ph, pw = resolve_padding(kernel_size, padding)

        mid = calc_mid_channels(in_ch, out_ch, t, h)

        self.conv_spatial = nn.Conv3d(
            in_ch, mid, (1, h, w),
            stride=(1, sh, sw), padding=(0, ph, pw), bias=False
        )
        self.norm1 = nn.InstanceNorm3d(mid)

        self.conv_temporal = nn.Conv3d(
            mid, out_ch, (t, 1, 1),
            stride=(st, 1, 1), padding=(pt, 0, 0), bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv_spatial(x)))
        x = self.relu(self.norm2(self.conv_temporal(x)))
        return x


class DeconvBlock_2plus1D(nn.Module):
    """(2+1)D Deconv: temporal deconv + spatial deconv"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1,1,1), padding=(1,1,1),
                 output_padding=(0,0,0), with_act=True):
        super().__init__()
        self.with_act = with_act
        t, h, w = kernel_size
        st, sh, sw = stride
        pt, ph, pw = padding if isinstance(padding, tuple) else (padding, padding, padding)

        mid = calc_mid_channels(in_ch, out_ch, t, h)

        self.deconv_temporal = nn.ConvTranspose3d(
            in_ch, mid, (t, 1, 1),
            stride=(st, 1, 1), padding=(pt, 0, 0), bias=False
        )
        self.norm1 = nn.InstanceNorm3d(mid)

        self.deconv_spatial = nn.ConvTranspose3d(
            mid, out_ch, (1, h, w),
            stride=(1, sh, sw), padding=(0, ph, pw), bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.deconv_temporal(x)))
        x = self.relu(self.norm2(self.deconv_spatial(x)))
        return x


# ==================== STBlock (shared, already (2+1)D) ====================

class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3,3)):
        super().__init__()
        t, d, _ = kernel_size
        mid = calc_mid_channels(in_ch, out_ch, t, d)
        self.conv2D = nn.Conv3d(in_ch, mid, (1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.norm1 = nn.InstanceNorm3d(mid)
        self.conv1D = nn.Conv3d(mid, out_ch, (3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv2D(x)))
        x = self.norm2(self.conv1D(x))
        return self.relu(x + identity)


# ==================== Full STVEN models ====================

class STVEN_Original(nn.Module):
    """Current STVEN: Full 3D encoder/decoder + (2+1)D bottleneck"""
    def __init__(self, base_channels=16, num_st_blocks=6, num_bitrate_levels=4):
        super().__init__()
        bc = base_channels
        in_ch = 3 + num_bitrate_levels  # RGB + CRF one-hot

        self.conv1 = ConvBlock_Original(in_ch, bc, (3,7,7), padding="same")
        self.conv2 = ConvBlock_Original(bc, bc*2, (3,4,4), stride=(1,2,2), padding=(1,2,2))
        self.conv3 = ConvBlock_Original(bc*2, bc*8, (4,4,4), stride=(2,2,2), padding=(1,1,1))

        self.st_blocks = nn.ModuleList([
            STBlock(bc*8, bc*8) for _ in range(num_st_blocks)
        ])

        self.dconv1 = DeconvBlock_Original(bc*8, bc*2, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        self.dconv2 = DeconvBlock_Original(bc*2, bc, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dconv3 = DeconvBlock_Original(bc, 3, (1,7,7), stride=(1,1,1), padding=(0,3,3), with_act=False)

    def forward(self, x, bitrate_label=None):
        residual = x
        if bitrate_label is not None:
            B, C, T, H, W = x.shape
            label_map = bitrate_label.view(B, -1, 1, 1, 1).expand(-1, -1, T, H, W)
            x = torch.cat([x, label_map], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for blk in self.st_blocks:
            x = blk(x)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        return x + residual


class STVEN_2plus1D(nn.Module):
    """Modified STVEN: Full (2+1)D encoder/decoder + (2+1)D bottleneck"""
    def __init__(self, base_channels=16, num_st_blocks=6, num_bitrate_levels=4):
        super().__init__()
        bc = base_channels
        in_ch = 3 + num_bitrate_levels

        self.conv1 = ConvBlock_2plus1D(in_ch, bc, (3,7,7), padding="same")
        self.conv2 = ConvBlock_2plus1D(bc, bc*2, (3,4,4), stride=(1,2,2), padding=(1,2,2))
        self.conv3 = ConvBlock_2plus1D(bc*2, bc*8, (4,4,4), stride=(2,2,2), padding=(1,1,1))

        self.st_blocks = nn.ModuleList([
            STBlock(bc*8, bc*8) for _ in range(num_st_blocks)
        ])

        self.dconv1 = DeconvBlock_2plus1D(bc*8, bc*2, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        self.dconv2 = DeconvBlock_2plus1D(bc*2, bc, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dconv3 = DeconvBlock_2plus1D(bc, 3, (1,7,7), stride=(1,1,1), padding=(0,3,3), with_act=False)

    def forward(self, x, bitrate_label=None):
        residual = x
        if bitrate_label is not None:
            B, C, T, H, W = x.shape
            label_map = bitrate_label.view(B, -1, 1, 1, 1).expand(-1, -1, T, H, W)
            x = torch.cat([x, label_map], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for blk in self.st_blocks:
            x = blk(x)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        return x + residual


def main():
    B = 1
    dummy_input = torch.randn(B, 3, 160, 128, 128)
    dummy_bitrate = torch.zeros(B, 4)
    dummy_bitrate[:, 0] = 1.0

    print("=" * 70)
    print("STVEN: Full 3D vs (2+1)D Encoder/Decoder Comparison")
    print("=" * 70)
    print(f"Config: base_channels=16, num_st_blocks=6, frame_length=160")
    print(f"Input:  [{B}, 7, 160, 128, 128] (3 RGB + 4 CRF one-hot)")
    print()

    # Original
    model_orig = STVEN_Original(base_channels=16, num_st_blocks=6, num_bitrate_levels=4)
    params_orig = sum(p.numel() for p in model_orig.parameters())
    macs_orig, params_fmt_orig = profile(
        model_orig, inputs=(dummy_input, dummy_bitrate), verbose=False
    )
    macs_orig_fmt, _ = clever_format([macs_orig, params_fmt_orig], "%.2f")

    # (2+1)D
    model_new = STVEN_2plus1D(base_channels=16, num_st_blocks=6, num_bitrate_levels=4)
    params_new = sum(p.numel() for p in model_new.parameters())
    macs_new, params_fmt_new = profile(
        model_new, inputs=(dummy_input, dummy_bitrate), verbose=False
    )
    macs_new_fmt, _ = clever_format([macs_new, params_fmt_new], "%.2f")

    # Per-layer breakdown
    print("Layer-by-layer comparison (encoder + decoder only, bottleneck shared):")
    print("-" * 70)

    bc = 16
    layers = [
        ("Conv1  (3+4)→16,  k(3,7,7)", 7, bc, (3,7,7)),
        ("Conv2  16→32,       k(3,4,4)", bc, bc*2, (3,4,4)),
        ("Conv3  32→128,      k(4,4,4)", bc*2, bc*8, (4,4,4)),
        ("DConv1 128→32,     k(4,4,4)", bc*8, bc*2, (4,4,4)),
        ("DConv2 32→16,      k(1,4,4)", bc*2, bc, (1,4,4)),
        ("DConv3 16→3,       k(1,7,7)", bc, 3, (1,7,7)),
    ]

    for name, cin, cout, k in layers:
        t, h, w = k
        # Full 3D MACs per element: cin * t * h * w
        macs_per_elem_3d = cin * t * h * w
        # (2+1)D
        mid = calc_mid_channels(cin, cout, t, h)
        macs_per_elem_2p1 = cin * h * w + mid * t
        ratio = macs_per_elem_2p1 / macs_per_elem_3d * 100
        print(f"  {name:<30} 3D: {macs_per_elem_3d:>8,}  (2+1)D: {macs_per_elem_2p1:>8,}  ({ratio:.0f}%)  mid_ch={mid}")

    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'':<30} {'Params':>12} {'MACs':>12}")
    print("-" * 56)
    print(f"{'Original (mixed 3D+(2+1)D)':<30} {params_orig:>12,} {macs_orig_fmt:>12}")
    print(f"{'Full (2+1)D':<30} {params_new:>12,} {macs_new_fmt:>12}")
    print()

    macs_reduction = (1 - macs_new / macs_orig) * 100
    params_change = (params_new - params_orig) / params_orig * 100
    print(f"MACs reduction:   {macs_reduction:.1f}%")
    print(f"Params change:    {params_change:+.1f}%")
    print()

    # With PhysFormer context
    physformer_macs = 50.61e9  # from earlier profiling
    joint_orig = macs_orig + physformer_macs
    joint_new = macs_new + physformer_macs
    joint_orig_fmt = f"{joint_orig/1e9:.2f}G"
    joint_new_fmt = f"{joint_new/1e9:.2f}G"
    joint_reduction = (1 - joint_new / joint_orig) * 100

    print("JointSTPhys (STVEN + PhysFormer):")
    print(f"  Original:  {joint_orig_fmt} MACs")
    print(f"  (2+1)D:    {joint_new_fmt} MACs")
    print(f"  Reduction: {joint_reduction:.1f}%")


if __name__ == "__main__":
    main()
