#!/usr/bin/env python3
"""
Extract STVEN-enhanced video frames for paper illustration.

Loads the STVEN checkpoint, reads compressed and original video frames,
runs STVEN enhancement, reconstructs pixel values from DiffNormalized space,
and saves side-by-side comparison frames.

Usage:
    python scripts/visualization/extract_stven_enhanced_frames.py \
        --compressed /home/zyuanli/dev/lib/data/UBFC-rPPG-CRF24/subject1/vid.mp4 \
        --original /home/zyuanli/dev/lib/data/UBFC-rPPG-CRF0/subject1/vid.mp4 \
        --checkpoint results/checkpoints/PURE_STVEN.pth \
        --output results/figures/stven_enhanced_frames/
"""

import argparse
import os
import sys
import warnings
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning)


def load_yolo_face_detector(device="cuda:0"):
    """Load YOLO5Face detector from the toolbox."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    return YOLO5Face("Y5F", device)


def read_video_frames(video_path, max_frames=None):
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames)  # [T, H, W, C]


def detect_and_crop_face(frames, detector):
    """Detect face in every frame and crop, then resize to 128x128."""
    processed = []
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox = detector.detect_face(bgr)
        if bbox is None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = bbox

        # Expand box by 50%
        w_box, h_box = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = int(w_box * 1.5), int(h_box * 1.5)
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(frame.shape[1], x1 + new_w)
        y2 = min(frame.shape[0], y1 + new_h)

        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
        processed.append(resized)

    return np.array(processed, dtype=np.float32)


def diff_normalized(frames):
    """Convert frames to DiffNormalized representation.

    Returns:
        dn: DiffNormalized data, shape [T, H, W, C], last frame is zero padding
        std: standard deviation used for normalization
        ref_frame: the first raw frame (needed for reconstruction)
    """
    T, H, W, C = frames.shape
    dn = np.zeros((T, H, W, C), dtype=np.float32)
    for j in range(T - 1):
        dn[j] = (frames[j + 1] - frames[j]) / (frames[j + 1] + frames[j] + 1e-7)
    dn_std = np.std(dn[:-1])  # std of meaningful data, excluding padding
    dn = dn / (dn_std + 1e-7)
    dn[np.isnan(dn)] = 0
    # last frame is already zero from initialization
    return dn, dn_std, frames[0].copy()


def reconstruct_frames(dn_enhanced, std, ref_frame):
    """Reconstruct pixel values from DiffNormalized data.

    Given dn[i]*std = (frame[i+1] - frame[i]) / (frame[i+1] + frame[i]):
        frame[i+1] = frame[i] * (1 + dn[i]*std) / (1 - dn[i]*std)

    Args:
        dn_enhanced: enhanced DiffNormalized data [T, H, W, C]
        std: standard deviation used in DiffNormalized normalization
        ref_frame: first frame [H, W, C] to start reconstruction

    Returns:
        reconstructed frames [T, H, W, C]
    """
    T = dn_enhanced.shape[0]
    H, W, C = ref_frame.shape
    frames = np.zeros((T, H, W, C), dtype=np.float32)
    frames[0] = ref_frame

    for i in range(T - 1):
        d = dn_enhanced[i] * std
        # Clamp to avoid division by zero or negative values
        d = np.clip(d, -0.999, 0.999)
        frames[i + 1] = frames[i] * (1.0 + d) / (1.0 - d + 1e-7)
        frames[i + 1] = np.clip(frames[i + 1], 0.0, 255.0)

    return frames


def load_stven(checkpoint_path, device="cuda:0"):
    """Load STVEN model from checkpoint."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from neural_methods.model.STVEN import STVEN

    model = STVEN(
        in_channels=3,
        out_channels=3,
        base_channels=16,
        num_st_blocks=6,
        frame_length=160,
        use_bitrate_labels=True,
        num_bitrate_levels=3,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    # Handle key mismatches (DataParallel prefix, STVEN-only vs full model)
    filtered = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("stven."):
            k = k[6:]  # remove 'stven.' prefix
        filtered[k] = v

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def create_comparison_grid(compressed_frames, enhanced_frames, original_frames,
                           sample_indices, save_path):
    """Create a side-by-side comparison figure of selected frames.

    Three rows: Compressed (CRF=24) | STVEN Enhanced | Original (CRF=0)
    """
    n_frames = len(sample_indices)
    fig, axes = plt.subplots(3, n_frames, figsize=(2.5 * n_frames, 7))

    titles = ["Compressed\n(CRF=24)", "STVEN\nEnhanced", "Original\n(CRF=0)"]
    frame_sets = [compressed_frames, enhanced_frames, original_frames]

    for row, (ax_row, title, frame_set) in enumerate(zip(axes, titles, frame_sets)):
        ax_row[0].set_ylabel(title, fontsize=10, fontweight="bold", rotation=0,
                             labelpad=30, va="center")
        for col, idx in enumerate(sample_indices):
            ax = axes[row, col]
            frame = frame_set[idx].astype(np.uint8)
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {idx / 30:.1f}s", fontsize=9)

    # Add zoom-in region highlight on compressed frames (bottom row would show
    # the original, but for paper-style we mark a region of interest on the first
    # compressed frame)
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid to {save_path}")


def create_single_zoomed_comparison(compressed_frames, enhanced_frames, original_frames,
                                    frame_idx, zoom_region, save_path):
    """Create a single-frame detailed comparison with zoom-in regions.

    Shows three frames (compressed, enhanced, original) side by side, each with
    a zoom-in callout of the specified region.
    """
    x1, y1, x2, y2 = zoom_region  # crop region in 128x128 coordinates

    fig, axes = plt.subplots(2, 3, figsize=(9, 6),
                             gridspec_kw={"height_ratios": [2, 1]})

    titles = ["Compressed (CRF=24)", "STVEN Enhanced", "Original (CRF=0)"]
    frame_sets = [compressed_frames, enhanced_frames, original_frames]

    for col, (title, frame_set) in enumerate(zip(titles, frame_sets)):
        frame = frame_set[frame_idx].astype(np.uint8)

        # Full frame (top row)
        ax_full = axes[0, col]
        ax_full.imshow(frame)
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=1.5, edgecolor="red", facecolor="none")
        ax_full.add_patch(rect)
        ax_full.set_title(title, fontsize=10, fontweight="bold")
        ax_full.set_xticks([])
        ax_full.set_yticks([])

        # Zoom region (bottom row)
        ax_zoom = axes[1, col]
        zoomed = frame[y1:y2, x1:x2]
        ax_zoom.imshow(zoomed)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.set_title(f"Zoom (detail)", fontsize=8, color="red")

    fig.suptitle(f"STVEN Enhancement — Frame {frame_idx} (t = {frame_idx / 30:.1f}s)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved zoomed comparison to {save_path}")


def save_individual_frames(compressed_frames, enhanced_frames, original_frames,
                           indices, out_dir):
    """Save individual PNG frames for each sample index."""
    os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        for name, frame_set in [("compressed", compressed_frames),
                                 ("enhanced", enhanced_frames),
                                 ("original", original_frames)]:
            fname = os.path.join(out_dir, f"frame_{idx:04d}_{name}.png")
            img = frame_set[idx].astype(np.uint8)
            cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved {len(indices) * 3} individual frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract STVEN-enhanced video frames")
    parser.add_argument("--compressed", required=True,
                        help="Path to compressed video (e.g., CRF=24)")
    parser.add_argument("--original", required=True,
                        help="Path to original uncompressed video (CRF=0)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to STVEN checkpoint (.pth)")
    parser.add_argument("--output", default="results/figures/stven_enhanced_frames",
                        help="Output directory for saved figures")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index")
    parser.add_argument("--num-frames", type=int, default=160,
                        help="Number of raw frames to process (STVEN requires T=160)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load face detector and STVEN model
    print("Loading YOLO5Face detector...")
    detector = load_yolo_face_detector(device)

    print("Loading STVEN model...")
    model = load_stven(args.checkpoint, device)

    # 2. Read video frames
    total_frames = args.start_frame + args.num_frames
    print(f"Reading frames {args.start_frame}..{total_frames - 1} from compressed video...")
    compressed_raw = read_video_frames(args.compressed, max_frames=total_frames)
    compressed_raw = compressed_raw[args.start_frame:total_frames]

    print(f"Reading same frames from original video...")
    original_raw = read_video_frames(args.original, max_frames=total_frames)
    original_raw = original_raw[args.start_frame:total_frames]

    print(f"Loaded {len(compressed_raw)} frames, shape {compressed_raw.shape}")

    # 3. Face detection and cropping
    print("Detecting and cropping faces...")
    compressed_cropped = detect_and_crop_face(compressed_raw, detector)
    original_cropped = detect_and_crop_face(original_raw, detector)
    print(f"Cropped shape: {compressed_cropped.shape}")

    # 4. DiffNormalized for STVEN input
    print("Computing DiffNormalized...")
    dn_input, dn_std, ref_frame = diff_normalized(compressed_cropped)

    # 5. Run STVEN
    print("Running STVEN enhancement...")
    # Convert to [B, C, T, H, W] NCDHW format
    x = torch.from_numpy(dn_input).float().to(device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, 128, 128]

    # Prepare bitrate label: target high quality (CRF=0 level)
    bitrate_label = torch.zeros(1, 3).to(device)  # 3 CRF levels
    bitrate_label[:, 0] = 1.0  # target CRF=0 / highest quality

    with torch.no_grad():
        enhanced_dn = model(x, bitrate_label)  # [1, 3, T, 128, 128]

    enhanced_dn = enhanced_dn.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # [T, 128, 128, 3]

    # 6. Reconstruct pixel values from enhanced DiffNormalized
    print("Reconstructing enhanced frames...")
    enhanced_frames = reconstruct_frames(enhanced_dn, dn_std, ref_frame)

    # 7. Save comparison visualizations
    # Sample frames to display
    sample_indices = list(range(0, min(160, len(enhanced_frames)), 20))

    # Grid comparison
    grid_path = os.path.join(args.output, "stven_comparison_grid.png")
    create_comparison_grid(compressed_cropped, enhanced_frames, original_cropped,
                           sample_indices, grid_path)

    # Single-frame zoomed comparison at the middle frame
    mid_idx = 80
    zoom_region = (32, 32, 96, 96)  # center 64x64 region of 128x128
    zoom_path = os.path.join(args.output, f"stven_zoomed_frame_{mid_idx}.png")
    create_single_zoomed_comparison(compressed_cropped, enhanced_frames,
                                    original_cropped, mid_idx, zoom_region, zoom_path)

    # Individual frames
    save_individual_frames(compressed_cropped, enhanced_frames, original_cropped,
                           sample_indices, os.path.join(args.output, "frames"))

    print("\nDone. Output files in:", args.output)


if __name__ == "__main__":
    main()
