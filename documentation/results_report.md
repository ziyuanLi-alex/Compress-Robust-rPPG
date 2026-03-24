# Results Directory Report

**Generated:** 2026-03-23

## Overview

The `results/` directory contains experimental results from rPPG (remote photoplethysmography) model evaluations, including CSV files with performance metrics, comparison figures, and pretrained model checkpoints.

## Directory Structure

```
results/
├── figures/
│   ├── Joint_comparison.png      # Joint STVEN+PhysFormer model comparison
│   └── mae_drop_plot.png         # MAE degradation vs compression plot
├── checkpoints/
│   └── STVEN_pretrain.pth        # Pretrained STVEN frontend model (12.3 MB)
├── PhysFormer_CRF1624_incl0.csv  # PhysFormer CRF results
├── PhysMamba_CRF1624_incl0.csv   # PhysMamba CRF results (duplicate/variant)
├── PhysMamba_batch.csv           # PhysMamba batch inference results
├── Physformer_batch.csv          # PhysFormer batch inference results
├── STVEN_Joint_batch.csv         # Joint STVEN+PhysFormer batch results
├── batch_inference_results.csv   # Comprehensive multi-model comparison
└── joint.csv                     # Joint training results summary
```

## CSV Results Files

### 1. `batch_inference_results.csv` - Multi-Model Benchmark

Comprehensive comparison of 17 different model configurations trained on PURE dataset and tested on UBFC-rPPG (and SCAMPS variants).

**Models included:**
| Model | MAE | MAPE | Pearson | SNR |
|-------|-----|------|---------|-----|
| PhysFormer_BASIC | 0.73 | 1.02 | 0.997 | - |
| RhythmFormer_BASIC | 1.03 | 0.95 | 0.974 | 1.98 |
| FactorizePhys_FSAM_Res (PURE) | 1.43 | 1.88 | 0.992 | 1.36 |
| TSCAN_BASIC | 1.35 | 1.56 | 0.989 | - |
| PhysMamba_BASIC | 1.60 | 1.81 | 0.986 | 0.31 |
| PhysNet_BASIC | 1.61 | 1.66 | 0.984 | - |
| DeepPhys_BASIC | 1.82 | 2.11 | 0.984 | - |
| EfficientPhys | 3.13 | 3.12 | 0.808 | - |
| iBVPNet_BASIC | 3.16 | 3.06 | 0.906 | - |
| SCAMPS_PhysFormer_BASIC | 5.11 | 6.16 | 0.744 | - |
| SCAMPS_DeepPhys_BASIC | 6.04 | 6.02 | 0.719 | - |
| SCAMPS_PhysNet_BASIC | 6.25 | 6.02 | 0.749 | - |
| SCAMPS_TSCAN_BASIC | 6.96 | 6.48 | 0.715 | - |
| Unsupervised (POS/CHROM/ICA/etc.) | 14.36 | 13.47 | 0.425 | - |
| SCAMPS_EfficientPhys | 75.88 | 79.18 | - | - |

**Key findings:**
- PhysFormer achieves best performance (MAE: 0.73)
- Traditional unsupervised methods show significantly higher error
- SCAMPS-trained models show degraded cross-dataset performance on UBFC-rPPG

### 2. `PhysFormer_CRF1624_incl0.csv` - Compression Robustness (PhysFormer)

Evaluation of PhysFormer under video compression (CRF levels 0, 16, 18, 20, 22, 24).

| CRF Level | MAE | RMSE | MAPE | Pearson | SNR |
|-----------|-----|------|------|---------|-----|
| CRF 0 (lossless) | 1.84 | 3.93 | 1.99 | 0.976 | 0.26 |
| CRF 16 | 8.16 | 18.68 | 7.19 | 0.551 | N/A |
| CRF 18 | 11.43 | 20.98 | 10.15 | 0.466 | N/A |
| CRF 20 | 22.91 | 29.88 | 21.82 | 0.122 | N/A |
| CRF 22 | 26.68 | 33.67 | 24.71 | 0.158 | N/A |
| CRF 24 | 28.94 | 34.80 | 27.21 | 0.156 | N/A |

**Observation:** Significant performance degradation beyond CRF 16, with Pearson correlation dropping from 0.976 to ~0.15.

### 3. `PhysMamba_batch.csv` - PhysMamba Compression Robustness

| CRF Level | MAE | RMSE | MAPE | Pearson | SNR |
|-----------|-----|------|------|---------|-----|
| CRF 0 | 1.39 | 3.80 | 1.68 | 0.979 | 2.15 |
| CRF 16 | 2.96 | 7.50 | 3.65 | 0.917 | N/A |
| CRF 18 | 12.58 | 22.92 | 11.68 | 0.510 | N/A |
| CRF 20 | 15.08 | 24.16 | 13.92 | 0.531 | N/A |
| CRF 22 | 26.64 | 34.56 | 25.88 | 0.220 | N/A |
| CRF 24 | 24.33 | 28.81 | 24.79 | 0.197 | N/A |

### 4. `STVEN_Joint_batch.csv` - Joint STVEN+PhysFormer

Joint training results with STVEN frontend and PhysFormer backend:

| CRF Level | MAE | RMSE | MAPE | Pearson | SNR |
|-----------|-----|------|------|---------|-----|
| CRF 0 | 1.85 | 4.31 | 1.71 | 0.977 | 1.63 |
| CRF 16 | 8.79 | 19.03 | 8.07 | 0.646 | N/A |
| CRF 18 | 12.77 | 21.49 | 11.79 | 0.529 | N/A |
| CRF 20 | 13.78 | 22.73 | 12.76 | 0.537 | N/A |
| CRF 22 | 20.35 | 25.40 | 20.73 | 0.277 | N/A |
| CRF 24 | 27.94 | 34.03 | 29.92 | N/A | N/A |

### 5. `joint.csv` - Alternative Joint Training Results

| CRF Level | MAE | RMSE | MAPE | Pearson |
|-----------|-----|------|------|---------|
| CRF 0 | 1.59 | 3.90 | 1.73 | 0.977 |
| CRF 16 | 9.21 | 17.16 | 8.89 | 0.634 |
| CRF 18 | 11.78 | 20.87 | 11.26 | 0.580 |
| CRF 20 | 19.52 | 27.74 | 18.47 | 0.381 |
| CRF 22 | 20.51 | 27.82 | 19.93 | 0.412 |
| CRF 24 | 27.43 | 33.60 | 26.19 | 0.030 |

### 6. `PhysFormer_CRF1624_incl0.csv` - Additional PhysFormer Results

Contains duplicate/variant runs with similar CRF evaluation.

## Figures

### `Joint_comparison.png` (322 KB)
Visual comparison of Joint STVEN+PhysFormer model performance, likely showing:
- MAE degradation curves across compression levels
- Comparison between different model variants

### `mae_drop_plot.png` (198 KB)
Plot showing Mean Absolute Error degradation as video compression increases (CRF levels).

## Checkpoints

### `STVEN_pretrain.pth` (12.3 MB)
Pretrained weights for the STVEN (Spatio-Temporal Video Enhancement Network) frontend model. Used as initialization for joint training with PhysFormer backend.

**Configured path in `joint_st_phys.yaml`:**
```
PRETRAINED_PATH: "runs/exp/.../PreTrainedModels/STVEN_pretrain_STVEN_Epoch1.pth"
```

## Metrics Explanation

All CSV files use consistent metrics:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| MAE | Mean Absolute Error (BPM) | 0 |
| MAE_Std | Standard deviation of MAE | 0 |
| RMSE | Root Mean Square Error (BPM) | 0 |
| RMSE_Std | Standard deviation of RMSE | 0 |
| MAPE | Mean Absolute Percentage Error (%) | 0 |
| MAPE_Std | Standard deviation of MAPE | 0 |
| Pearson | Pearson correlation coefficient | 1.0 |
| Pearson_Std | Standard deviation of Pearson | 0 |
| SNR | Signal-to-Noise Ratio (dB) | Higher is better |
| SNR_Std | Standard deviation of SNR | 0 |

## Key Research Findings

1. **Compression Robustness Gap:** All models show significant degradation beyond CRF 16, indicating a need for compression-robust training strategies.

2. **Best Baseline Performance:** PhysFormer achieves the lowest MAE (0.73 BPM) on uncompressed video.

3. **Joint Training:** STVEN+PhysFormer joint training shows comparable performance to standalone PhysFormer at CRF 0, but similar degradation patterns at higher CRF levels.

4. **Cross-Dataset Generalization:** Models trained on SCAMPS (synthetic data) show poor generalization to UBFC-rPPG (real data), with MAE increasing 5-10x.

5. **Unsupervised Methods:** Traditional signal processing methods (POS, CHROM, ICA, GREEN, LGI, PBV) achieve MAE of 14.36 BPM, significantly worse than supervised neural methods.
