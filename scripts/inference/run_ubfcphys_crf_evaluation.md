# UBFC-Phys Compression Robustness Evaluation (CRF)

Evaluates model performance on H.264-compressed UBFC-Phys videos at various CRF levels.

## Models

| Model | Trained On | Pretrained Weights | CHUNK_LENGTH |
|---|---|---|---|
| PhysFormer | PURE | `PURE_PhysFormer_DiffNormalized.pth` | 160 |
| PhysMamba | PURE | `PURE_PhysMamba_DiffNormalized.pth` | 128 |

## Test Data

- Dataset: UBFC-PHYS (subjects s1–s14, tasks T1/T2/T3)
- Sampling rate: 35 fps
- Backend: Y5F (all batches unified)
- Data type: DiffNormalized, 30s evaluation window
- CRF levels: 0, 14, 16, 18, 20, 22, 24

### Exclusion List (s1–s14 only)

| Task | Excluded Subjects | Count |
|---|---|---|
| T1 | s3, s8, s9 | 3 |
| T2 | s1, s4, s6, s8, s9, s11, s12, s13, s14 | 9 |
| T3 | s5, s8, s9, s10, s13, s14 | 6 |

18 excluded, **24 usable** of 42 total videos.

## Step 1: Remove Excluded Videos and Compress

Raw UBFC-Phys videos are MJPEG AVI at 1024×1024, ~4.7GB each, stored at `/mnt/k/RawData`.

First, delete the 18 excluded videos to save disk space (~83GB) and avoid preprocessing them:

```bash
bash scripts/inference/ubfcphys_remove_excluded.sh /mnt/k/RawData
```

Then compress with H.264 at each CRF level. Output goes to `/mnt/k/UBFC-Phys-CRF{N}/`:

```bash
# Compress at all CRF levels (source: /mnt/k/RawData, dest: /mnt/k/UBFC-Phys-CRF{N})
for crf in 0 14 16 18 20 22 24; do
    bash scripts/inference/compress_ubfcphys_crf.sh $crf
done

# Or compress a single CRF level
bash scripts/inference/compress_ubfcphys_crf.sh 14
```

### Re-compress from CRF0 (Space-Saving)

If you already have CRF0 videos, you can re-compress from them instead of the raw AVI files. This saves disk space by avoiding repeated reads of the large raw files:

```bash
# Re-compress from CRF0 at all CRF levels (source: /mnt/h/lib/UBFC-Phys-CRF0)
for crf in 14 16 18 20 22 24; do
    bash scripts/inference/compress_ubfcphys_from_crf0.sh $crf
done

# Or re-compress a single CRF level
bash scripts/inference/compress_ubfcphys_from_crf0.sh 14

# Custom paths
bash scripts/inference/compress_ubfcphys_from_crf0.sh 14 ~/dev/lib/data/UBFC-Phys-CRF0 ~/dev/lib/data/UBFC-Phys-CRF14

    for crf in 14 16 18 20 22 24; do
        bash scripts/inference/compress_ubfcphys_from_crf0.sh "$crf" ~/dev/lib/UBFC-Phys/UBFC-Phys-CRF0 ~/dev/lib/UBFC-Phys/UBFC-Phys-CRF"$crf"
    done
```

Output structure:
```
/mnt/k/
  RawData/                          # Originals (after exclusion deletion)
  UBFC-Phys-CRF0/                   # Lossless H.264
  UBFC-Phys-CRF14/                  # CRF 14
  UBFC-Phys-CRF16/                  # CRF 16
  ...                               # CRF 18, 20, 22, 24
  UBFC-Phys-cache/                  # Preprocessed cache (generated at runtime)

Each UBFC-Phys-CRF{N}/ contains:
  s1/
    vid_s1_T1.mp4
    bvp_s1_T1.csv
    vid_s1_T2.mp4
    bvp_s1_T2.csv
    ...
```

## Step 2: Generate Batch Configs

CRF configs are generated from `*_BASIC.yaml` templates:

```bash
# PhysFormer (PURE-trained)
bash scripts/inference/generate_batch_configs_ubfcphys.sh \
    --config configs/infer_configs/PURE_UBFC-Phys_PHYSFORMER_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/PURE_UBFC-Phys_PhysFormer_CRF

# PhysMamba (PURE-trained)
bash scripts/inference/generate_batch_configs_ubfcphys.sh \
    --config configs/infer_configs/PURE_UBFC-Phys_PHYSMAMBA_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/PURE_UBFC-Phys_PhysMamba_CRF
```

## DO_PREPROCESS Convention

Only ONE batch per test dataset should run preprocessing. All others reuse the cached data.

| Batch | DO_PREPROCESS | Notes |
|---|---|---|
| `PURE_UBFC-Phys_PhysFormer_CRF/` | `True` | Run first |
| `PURE_UBFC-Phys_PhysMamba_CRF/` | `True` | Run first (separate cache, CHUNK_LENGTH=128) |

Set `DO_PREPROCESS: False` for subsequent CRF batches that share the same CHUNK_LENGTH.

## Running Inference

### Per-Model Batches

```bash
# PhysFormer - PURE
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/PURE_UBFC-Phys_PhysFormer_CRF \
    results/inference_logs/PURE_PhysFormer_UBFCPhys

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/PURE_PhysFormer_UBFCPhys \
    --output_csv results/PURE_PhysFormer_UBFCPhys_CRF.csv

# PhysMamba - PURE
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/PURE_UBFC-Phys_PhysMamba_CRF \
    results/inference_logs/PURE_PhysMamba_UBFCPhys

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/PURE_PhysMamba_UBFCPhys \
    --output_csv results/PURE_PhysMamba_UBFCPhys_CRF.csv
```

### All Batches (Single-Step)

```bash
python scripts/inference/batch_inference.py \
    --config_dir configs/infer_configs \
    --output_csv results/ubfcphys_batch_results.csv
```

## Output

CSV files in `results/`:
- `PURE_PhysFormer_UBFCPhys_CRF.csv`
- `PURE_PhysMamba_UBFCPhys_CRF.csv`

Each CSV: Config, MAE, MAE_Std, RMSE, RMSE_Std, MAPE, MAPE_Std, Pearson, Pearson_Std, SNR, SNR_Std

Raw logs: `results/inference_logs/<model>/<config_name>.log`

## Key Differences from UBFC-rPPG CRF Evaluation

| Aspect | UBFC-rPPG | UBFC-Phys |
|---|---|---|
| Sampling rate | 30 fps | 35 fps |
| Videos per entry | 1 per subject | 3 per subject (T1/T2/T3) |
| Face detection | Y5F | Y5F |
| Exclusion list | None | 18 of 42 videos (s1–s14) |
| Config generation script | `generate_batch_configs_ubfcrppg.sh` | `generate_batch_configs_ubfcphys.sh` |
| Loader (compressed) | `UBFCrPPGh264Loader` | `UBFCPHYSh264Loader` |
| Dataset name (compressed) | `UBFC-rPPG-h264` | `UBFC-PHYS-h264` |

## File Inventory

| File | Purpose |
|---|---|
| `scripts/inference/ubfcphys_remove_excluded.sh` | Delete excluded videos from raw data |
| `scripts/inference/compress_ubfcphys_crf.sh` | ffmpeg compression (from raw AVI) |
| `scripts/inference/compress_ubfcphys_from_crf0.sh` | ffmpeg re-compression (from CRF0 MP4) |
| `scripts/inference/generate_batch_configs_ubfcphys.sh` | CRF config generator for UBFC-PHYS |
| `dataset/data_loader/UBFCPHYSh264Loader.py` | Compressed UBFC-PHYS dataloader |
| `configs/infer_configs/PURE_UBFC-Phys_PHYSFORMER_BASIC.yaml` | PhysFormer base config |
| `configs/infer_configs/PURE_UBFC-Phys_PHYSMAMBA_BASIC.yaml` | PhysMamba base config |
