"""Profile STVEN, PhysFormer, and JointSTPhys models: parameter counts and MACs.

Usage:
    python tools/profile_model.py
    python tools/profile_model.py --config configs/train_configs/A/A1/joint_A1.yaml

Uses thop (already in requirements.txt via torch ecosystem) for MACs calculation.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from thop import profile, clever_format
from neural_methods.model.STVEN import STVEN, PhysFormerWithSTVEN
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_model(model, input_tensor, bitrate_label=None, gra_sharp=None):
    """Profile a model using thop."""
    inputs = (input_tensor,)
    if bitrate_label is not None and gra_sharp is not None:
        inputs = (input_tensor, bitrate_label, gra_sharp)
    elif bitrate_label is not None:
        inputs = (input_tensor, bitrate_label)
    elif gra_sharp is not None:
        inputs = (input_tensor, gra_sharp)
    macs, params = profile(model, inputs=inputs, verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    return macs, params


def main():
    parser = argparse.ArgumentParser(description="Profile STVEN+PhysFormer models")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # A1 config values (hardcoded defaults matching joint_A1.yaml)
    stven_config = {
        'in_channels': 3,
        'out_channels': 3,
        'base_channels': 16,
        'num_st_blocks': 6,
        'frame_length': 160,
        'use_bitrate_labels': True,
        'num_bitrate_levels': 4,
    }

    physformer_config = {
        'PATCH_SIZE': 4,
        'DIM': 96,
        'FF_DIM': 144,
        'NUM_HEADS': 4,
        'NUM_LAYERS': 12,
        'THETA': 0.7,
        'IMAGE_SIZE': [160, 128, 128],
        'PATCHES': [4, 4, 4],
        'DROPOUT_RATE': 0.2,
    }

    B = args.batch_size
    # Input: [B, C, T, H, W] = [B, 3, 160, 128, 128]
    dummy_input = torch.randn(B, 3, 160, 128, 128)
    # Bitrate label: one-hot with 4 levels
    dummy_bitrate = torch.zeros(B, 4)
    dummy_bitrate[:, 0] = 1.0  # CRF=0

    device = args.device

    print("=" * 70)
    print("STVEN + PhysFormer Computational Efficiency Analysis")
    print("=" * 70)
    print(f"Input shape:  [{B}, 3, 160, 128, 128]")
    print(f"Batch size:   {B}")
    print()

    # --- STVEN ---
    stven = STVEN(
        in_channels=stven_config['in_channels'],
        out_channels=stven_config['out_channels'],
        base_channels=stven_config['base_channels'],
        num_st_blocks=stven_config['num_st_blocks'],
        frame_length=stven_config['frame_length'],
        use_bitrate_labels=stven_config['use_bitrate_labels'],
        num_bitrate_levels=stven_config['num_bitrate_levels'],
    ).to(device)

    stven_total, stven_trainable = count_parameters(stven)
    stven_macs, stven_params_fmt = profile_model(
        stven, dummy_input.to(device), dummy_bitrate.to(device)
    )

    print("STVEN (Video Enhancement Frontend)")
    print("-" * 40)
    print(f"  Total params:     {stven_total:>12,}")
    print(f"  Trainable params: {stven_trainable:>12,}")
    print(f"  MACs:             {stven_macs}")
    print(f"  Params (thop):    {stven_params_fmt}")
    print()

    # --- PhysFormer ---
    physformer = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=tuple(physformer_config['IMAGE_SIZE']),
        patches=tuple(physformer_config['PATCHES']),
        dim=physformer_config['DIM'],
        ff_dim=physformer_config['FF_DIM'],
        num_heads=physformer_config['NUM_HEADS'],
        num_layers=physformer_config['NUM_LAYERS'],
        dropout_rate=physformer_config['DROPOUT_RATE'],
        theta=physformer_config['THETA'],
    ).to(device)

    pf_total, pf_trainable = count_parameters(physformer)
    pf_macs, pf_params_fmt = profile_model(
        physformer, dummy_input.to(device), gra_sharp=2.0
    )

    print("PhysFormer (rPPG Estimation Backend)")
    print("-" * 40)
    print(f"  Total params:     {pf_total:>12,}")
    print(f"  Trainable params: {pf_trainable:>12,}")
    print(f"  MACs:             {pf_macs}")
    print(f"  Params (thop):    {pf_params_fmt}")
    print()

    # --- JointSTPhys ---
    joint = PhysFormerWithSTVEN(stven_config, physformer_config).to(device)

    joint_total, joint_trainable = count_parameters(joint)
    joint_macs, joint_params_fmt = profile_model(
        joint, dummy_input.to(device), dummy_bitrate.to(device), gra_sharp=2.0
    )

    print("JointSTPhys (Combined STVEN + PhysFormer)")
    print("-" * 40)
    print(f"  Total params:     {joint_total:>12,}")
    print(f"  Trainable params: {joint_trainable:>12,}")
    print(f"  MACs:             {joint_macs}")
    print(f"  Params (thop):    {joint_params_fmt}")
    print()

    # --- Summary table ---
    print("=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Model':<30} {'Params':>12} {'MACs':>12}")
    print("-" * 56)
    print(f"{'STVEN':<30} {stven_total:>12,} {stven_macs:>12}")
    print(f"{'PhysFormer':<30} {pf_total:>12,} {pf_macs:>12}")
    print(f"{'JointSTPhys (STVEN+Phys)':<30} {joint_total:>12,} {joint_macs:>12}")
    print()

    # Trainable breakdown for JointSTPhys (PhysFormer frozen in training)
    print("Trainable Parameters (during joint training, PhysFormer frozen):")
    print(f"  STVEN trainable:       {stven_trainable:>12,}")
    print(f"  PhysFormer trainable:  {0:>12,}  (frozen)")
    print(f"  Total trainable:       {stven_trainable:>12,}")
    print()


if __name__ == "__main__":
    main()
