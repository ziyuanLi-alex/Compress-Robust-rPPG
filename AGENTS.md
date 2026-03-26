# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

rPPG-Toolbox is an open-source platform for camera-based physiological sensing (remote photoplethysmography). The project supports multiple neural networks (PhysFormer, PhysMamba, TSCAN, DeepPhys, EfficientPhys, etc.) and unsupervised methods for extracting heart rate signals from facial videos.

## Setup & Environment

```bash
# Using conda
bash setup.sh conda
conda activate rppg-toolbox

# Using uv
bash setup.sh uv
source .venv/bin/activate
```

Dependencies are in `requirements.txt` (PyTorch 2.1.2+cu121, mamba-ssm, etc.)

## Common Commands

```bash
# Train and test with a config
python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml

# Test only (inference)
python main.py --config_file ./configs/infer_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml

# Unsupervised methods
python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml

# Clear preprocessing cache
rm ./<dataset>-cache/*
```

## Architecture

```
rPPG-Toolbox/
├── main.py                    # Entry point - routes to train/test/unsupervised pipelines
├── config.py                  # YACS config system - defines all hyperparameters
├── configs/
│   ├── train_configs/         # Training configs: [TRAIN]_[VALID]_[TEST]_[MODEL].yaml
│   └── infer_configs/         # Inference configs: [TRAIN]_[TEST]_[MODEL].yaml
├── dataset/
│   └── data_loader/           # Dataset loaders (UBFC-rPPG, PURE, SCAMPS, MMPD, BP4D+, iBVP, etc.)
├── neural_methods/
│   ├── model/                 # Network architectures
│   ├── trainer/               # Training loops (one Trainer class per model)
│   └── loss/                  # Loss functions (NegPearson, etc.)
├── unsupervised_methods/      # Traditional signal processing (POS, CHROM, ICA, GREEN, LGI, PBV, OMIT)
├── evaluation/                # Metrics (MAE, RMSE, MAPE, Pearson, SNR, Bland-Altman)
└── tools/                     # Visualization, Mamba module source
```

## Config System

All configs use YACS (`config.py`). Key sections:
- `TOOLBOX_MODE`: "train_and_test", "only_test", or "unsupervised_method"
- `TRAIN/VALID/TEST.DATA`: Dataset paths, preprocessing options, data splits (BEGIN/END)
- `MODEL`: Architecture choice and hyperparameters
- `INFERENCE`: Batch size, model path, evaluation settings

Custom models add their own config sections (e.g., `MODEL.PHYSFORMER`, `MODEL.STVEN`).

## Adding New Components

**New Dataset**: Create loader in `dataset/data_loader/`, implement `preprocess_dataset()`, `read_video()`, `read_wave()`, add to `main.py` loader selection.

**New Neural Model**: Create model in `neural_methods/model/`, trainer in `neural_methods/trainer/`, add to `main.py` train/test dispatch, add config section.

**New Unsupervised Method**: Create in `unsupervised_methods/methods/`, add to `main.py` unsupervised_method_inference().

## Supported Models

- DeepPhys, TSCAN, PhysNet, EfficientPhys
- BigSmall (multi-task: PPG, respiration, AU classification)
- PhysFormer, RhythmFormer, PhysMamba (Mamba-based)
- FactorizePhys (with FSAM attention)
- STVEN, JointSTPhys (joint STVEN+PhysFormer training)

## Supported Datasets

UBFC-rPPG, PURE, SCAMPS, MMPD, BP4D+, UBFC-Phys, iBVP, PhysDrive, SUMS, LADH

## Preprocessing

Data is preprocessed into chunks (`.npy` files) stored in `CACHED_PATH`. First run requires `DO_PREPROCESS: True`. Face detection backends: Haar Cascade (HC) or YOLO5Face (Y5F).
