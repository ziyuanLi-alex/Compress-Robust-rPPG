#!/usr/bin/env python3
"""
Plot rPPG waveform from STVEN+PhysFormer inference output.

Loads a pickle file produced by JointSTVENPhysFormerTrainer.test() and plots
the predicted rPPG signal against ground truth BVP for selected subjects.

Usage:
    python scripts/visualization/plot_rppg_waveform.py \
        --pickle runs/exp/<path>/saved_test_outputs/<name>_outputs.pickle \
        --output results/figures/rppg_waveform.pdf

    # With custom options:
    python scripts/visualization/plot_rppg_waveform.py \
        --pickle <path> \
        --output <path> \
        --subjects subject23 subject38 \
        --duration 15 \
        --fs 30
"""

import argparse
import pickle
import numpy as np
import torch as _torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def _to_cpu(obj):
    """Recursively move tensors in nested dicts to CPU."""
    if isinstance(obj, _torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return _to_cpu(data)


def reconstruct_signal(chunks_dict):
    """Reconstruct full signal from chunk dict {chunk_id: tensor[160]}."""
    sorted_ids = sorted(chunks_dict.keys())
    segments = [chunks_dict[cid].numpy().flatten() for cid in sorted_ids]
    return np.concatenate(segments)


def plot_waveform(pred, label, fs, duration, subject, out_path):
    """Plot predicted rPPG vs ground truth BVP."""
    n_samples = int(duration * fs)
    pred = pred[:n_samples]
    label = label[:n_samples]
    t = np.arange(len(pred)) / fs

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(t, label, linewidth=1.0, color="#2c3e50", alpha=0.85, label="Ground Truth BVP")
    ax.plot(t, pred, linewidth=1.0, color="#e74c3c", alpha=0.85, label="STVEN+PhysFormer rPPG")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Normalized Amplitude", fontsize=11)
    ax.set_title(f"rPPG Waveform — {subject}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved waveform plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot rPPG waveform from inference pickle")
    parser.add_argument("--pickle", required=True, help="Path to the pickle output file")
    parser.add_argument("--output", default="results/figures/rppg_waveform.pdf", help="Output path")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to plot")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds to plot")
    parser.add_argument("--fs", type=int, default=30, help="Sampling rate (Hz)")
    args = parser.parse_args()

    data = load_pickle(args.pickle)
    predictions = data["predictions"]
    labels = data["labels"]
    fs = data.get("fs", args.fs)

    subjects = args.subjects or sorted(predictions.keys())
    print(f"Plotting {len(subjects)} subject(s): {subjects}")

    for subj in subjects:
        pred = reconstruct_signal(predictions[subj])
        label_sig = reconstruct_signal(labels[subj])
        min_len = min(len(pred), len(label_sig))
        pred, label_sig = pred[:min_len], label_sig[:min_len]

        # Normalize both signals for visual comparison
        pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-7)
        label_sig = (label_sig - np.mean(label_sig)) / (np.std(label_sig) + 1e-7)

        out_name = args.output.replace(".pdf", f"_{subj}.pdf").replace(".png", f"_{subj}.png")
        plot_waveform(pred, label_sig, fs, args.duration, subj, out_name)


if __name__ == "__main__":
    main()
