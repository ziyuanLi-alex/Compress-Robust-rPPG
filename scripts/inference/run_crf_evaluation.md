# Compression Robustness Evaluation (CRF)

Evaluates model performance on H.264-compressed UBFC-rPPG videos at various CRF levels.

## Models

| Model | Trained On | Pretrained Weights | CHUNK_LENGTH |
|---|---|---|---|
| PhysFormer | PURE | `PURE_PhysFormer_DiffNormalized.pth` | 160 |
| PhysFormer | SCAMPS | `SCAMPS_PhysFormer_DiffNormalized.pth` | 160 |
| PhysFormer | UBFC | `UBFC-rPPG_PhysFormer_DiffNormalized.pth` | 160 |
| PhysMamba | PURE | `PURE_PhysMamba_DiffNormalized.pth` | 128 |
| PhysMamba | UBFC | `UBFC-rPPG_PhysMamba_DiffNormalized.pth` | 128 |

The 128 chunk length is used for PhysMamba, which is only compatible with a factor of 512.

## Test Data

- Dataset: UBFC-rPPG-h264 at CRFs 0, 14, 16, 18, 20, 22, 24
- Backend: Y5F (all batches unified)
- Data type: DiffNormalized, window size varies by model

## Generating Batch Configs

CRF configs are generated from `*_BASIC.yaml` templates:

```bash
# PhysFormer (PURE-trained)
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/PURE_PhysFormer_CRF

# PhysFormer (SCAMPS-trained)
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/SCAMPS_UBFC-rPPG_PHYSFORMER_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/SCAMPS_PhysFormer_CRF

# PhysFormer (UBFC-trained)
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/UBFC_UBFC-rPPG_PHYSFORMER_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/UBFC_PhysFormer_CRF

# PhysMamba (PURE-trained)
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/PURE_UBFC-rPPG_PHYSMAMBA_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/PURE_PhysMamba_CRF

# PhysMamba (UBFC-trained)
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/UBFC_UBFC-rPPG_PHYSMAMBA_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/UBFC_PhysMamba_CRF
```

## DO_PREPROCESS Convention

Only ONE batch per test dataset should run preprocessing. All others reuse the cached data.

| Batch | DO_PREPROCESS | Notes |
|---|---|---|
| `PURE_PhysFormer_CRF/` | `True` | Run first |
| `SCAMPS_PhysFormer_CRF/` | `False` | Reuses PURE cache |
| `UBFC_PhysFormer_CRF/` | `False` | Reuses PURE cache |
| `PURE_PhysMamba_CRF/` | `True` | Run first |
| `UBFC_PhysMamba_CRF/` | `False` | Reuses PURE cache |

## Running Inference

### Per-Model Batches

```bash
# PhysFormer - PURE
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/PURE_PhysFormer_CRF \
    results/inference_logs/PURE_PhysFormer

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/PURE_PhysFormer \
    --output_csv results/PURE_PhysFormer_CRF.csv

# PhysFormer - SCAMPS
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/SCAMPS_PhysFormer_CRF \
    results/inference_logs/SCAMPS_PhysFormer

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/SCAMPS_PhysFormer \
    --output_csv results/SCAMPS_PhysFormer_CRF.csv

# PhysFormer - UBFC
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/UBFC_PhysFormer_CRF \
    results/inference_logs/UBFC_PhysFormer

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/UBFC_PhysFormer \
    --output_csv results/UBFC_PhysFormer_CRF.csv

# PhysMamba - PURE
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/PURE_PhysMamba_CRF \
    results/inference_logs/PURE_PhysMamba

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/PURE_PhysMamba \
    --output_csv results/PURE_PhysMamba_CRF.csv

# PhysMamba - UBFC
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/UBFC_PhysMamba_CRF \
    results/inference_logs/UBFC_PhysMamba

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/UBFC_PhysMamba \
    --output_csv results/UBFC_PhysMamba_CRF.csv
```

### All Batches (Single-Step)

```bash
python scripts/inference/batch_inference.py \
    --config_dir configs/infer_configs \
    --output_csv results/batch_results.csv
```

## Output

CSV files in `results/`:
- `PURE_PhysFormer_CRF.csv`
- `SCAMPS_PhysFormer_CRF.csv`
- `UBFC_PhysFormer_CRF.csv`
- `PURE_PhysMamba_CRF.csv`
- `UBFC_PhysMamba_CRF.csv`

Each CSV: Config, MAE, MAE_Std, RMSE, RMSE_Std, MAPE, MAPE_Std, Pearson, Pearson_Std, SNR, SNR_Std

Raw logs: `results/inference_logs/<model>/<config_name>.log`

## STVEN-PhysFormer (Joint Model)

Joint model using STVEN for preprocessing and PhysFormer as the backend. Both components were trained on UBFC-rPPG, so the test split must match the UBFC-rPPG PhysFormer convention (`BEGIN: 0.8, END: 1.0`) to prevent data leakage.

### Model

| Model | Trained On | STVEN Weights | PhysFormer Weights | CHUNK_LENGTH |
|---|---|---|---|---|
| JointSTPhys | UBFC-rPPG | `results/checkpoints/PURE_STVEN.pth` | `UBFC-rPPG_PhysFormer_DiffNormalized.pth` | 160 |

### Base Config Alignment

The base config `PURE_UBFC-rPPG_STVEN_PHYSFORMER_BASIC.yaml` was aligned with `UBFC_UBFC-rPPG_PHYSFORMER_BASIC.yaml`:

- `TEST.DATA.BEGIN`: 0.8
- `TEST.DATA.END`: 1.0
- `TEST.DATA.DATASET`: UBFC-rPPG (base, no CRF)
- `TEST.DATA.DATA_PATH` / `CACHED_PATH`: raw UBFC-rPPG paths (no CRF suffix)
- `MODEL.DROP_RATE`: 0.2
- `INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE`: 30

### Generating Batch Configs

```bash
bash scripts/inference/generate_batch_configs_ubfcrppg.sh \
    --config configs/infer_configs/PURE_UBFC-rPPG_STVEN_PHYSFORMER_BASIC.yaml \
    --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
    --output_dir configs/infer_configs/generated_configs
```

### Running Inference

```bash
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/generated_configs \
    results/inference_logs/STVEN_PhysFormer

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/STVEN_PhysFormer \
    --output_csv results/STVEN_PhysFormer_CRF.csv
```

### DO_PREPROCESS

Set `DO_PREPROCESS: False` if reusing cache from any other UBFC-rPPG batch.
