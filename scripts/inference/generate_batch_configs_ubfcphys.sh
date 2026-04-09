#!/bin/bash

# Generate CRF-specific inference configs for UBFC-PHYS evaluation.
# Replaces UBFC-PHYS paths/names with UBFC-PHYS-h264 CRF variants.
#
# Usage:
#   bash scripts/inference/generate_batch_configs_ubfcphys.sh \
#       --config configs/infer_configs/PURE_UBFC-Phys_PHYSFORMER_BASIC.yaml \
#       --start_crf 14 --end_crf 24 --step_crf 2 --include_crf0 \
#       --output_dir configs/infer_configs/PURE_UBFC-Phys_PhysFormer_CRF

set -euo pipefail

usage() {
    echo "Usage: $0 --config <path_to_config> --start_crf <start> --end_crf <end> --step_crf <step> [--output_dir <dir>] [--include_crf0]"
    exit 1
}

output_dir=""
include_crf0=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --start_crf) start_crf="$2"; shift ;;
        --end_crf) end_crf="$2"; shift ;;
        --step_crf) step_crf="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --include_crf0) include_crf0=true ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [ -z "$config" ] || [ -z "$start_crf" ] || [ -z "$end_crf" ] || [ -z "$step_crf" ]; then
    usage
fi

if [ ! -f "$config" ]; then
    echo "Error: Config file not found: $config"
    exit 1
fi

if [ -z "$output_dir" ]; then
    output_dir="$(dirname "$config")/generated_configs"
fi

mkdir -p "$output_dir"
echo "Output directory: $output_dir"

# Build CRF list
crf_values=()
if [ "$include_crf0" = true ]; then
    crf_values+=(0)
fi
for (( crf=start_crf; crf<=end_crf; crf+=step_crf )); do
    crf_values+=($crf)
done

for crf in "${crf_values[@]}"; do
    echo "Generating config for CRF: $crf"

    base_name=$(basename "$config")
    filename="${base_name%.yaml}"

    if [[ "$filename" == *"_BASIC"* ]]; then
        new_filename="${filename/_BASIC/_CRF$crf}.yaml"
    else
        new_filename="${filename}_CRF$crf.yaml"
    fi

    output_config="$output_dir/$new_filename"
    cp "$config" "$output_config"

    # UBFC-PHYS -> UBFC-PHYS-h264 dataset name
    sed -i 's/DATASET: UBFC-PHYS\s*$/DATASET: UBFC-PHYS-h264/g' "$output_config"

    # Replace DATA_PATH: /home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys -> /home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-CRF$crf
    sed -i "s|DATA_PATH: \"/home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys\"|DATA_PATH: \"/home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-CRF$crf\"|g" "$output_config"

    # Replace CACHED_PATH: /home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-cache -> /home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-CRF$crf-cache
    sed -i "s|CACHED_PATH: \"/home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-cache\"|CACHED_PATH: \"/home/zyuanli/dev/lib/UBFC-Phys/UBFC-Phys-CRF$crf-cache\"|g" "$output_config"

    echo "Created: $output_config"
done
