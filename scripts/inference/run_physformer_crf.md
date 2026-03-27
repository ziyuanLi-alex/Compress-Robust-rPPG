# PhysFormer Compression Robness Evaluation

## Pretrained Models
- PURE: `PURE_PhysFormer_DiffNormalized.pth`
- SCAMPS: `SCAMPS_PhysFormer_DiffNormalized.pth`
- UBFC: `UBFC-rPPG_PhysFormer_DiffNormalized.pth`

## Test Data
- Dataset: UBFC-rPPG at CRFs 0, 14, 16, 18, 20, 22, 24
- Window: 30s, DiffNormalized
- PhysFormer: unchanged architecture

## Run All (recommended)

### Option 1: Shell Script (Two-Step Process)
```bash
# Step 1: Run batch inference and store raw logs
bash scripts/inference/batch_inference.sh \
    configs/infer_configs \
    results/inference_logs

# Step 2: Parse logs into CSV
python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs \
    --output_csv results/batch_results.csv
```

### Option 2: Python Script (Single-Step)
```bash
python scripts/inference/batch_inference.py \
    --config_dir configs/infer_configs \
    --output_csv results/batch_results.csv
```

## Run for Specific Config Folders

### Shell Script Approach
```bash
# PURE configs
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/PURE_PhysFormer_CRF \
    results/inference_logs/PURE

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/PURE \
    --output_csv results/PURE_PhysFormer_CRF.csv

# SCAMPS configs
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/SCAMPS_PhysFormer_CRF \
    results/inference_logs/SCAMPS

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/SCAMPS \
    --output_csv results/SCAMPS_PhysFormer_CRF.csv

# UBFC configs
bash scripts/inference/batch_inference.sh \
    configs/infer_configs/UBFC_PhysFormer_CRF \
    results/inference_logs/UBFC

python scripts/inference/parse_inference_logs.py \
    --log_dir results/inference_logs/UBFC \
    --output_csv results/UBFC_PhysFormer_CRF.csv
```



## Output
- `results/PURE_PhysFormer_CRF.csv`
- `results/SCAMPS_PhysFormer_CRF.csv`
- `results/UBFC_PhysFormer_CRF.csv`

Each CSV contains: Config, MAE, MAE_Std, RMSE, RMSE_Std, MAPE, MAPE_Std, Pearson, Pearson_Std, SNR, SNR_Std

Raw logs (shell script approach):
- `results/inference_logs/<config_name>.log`
