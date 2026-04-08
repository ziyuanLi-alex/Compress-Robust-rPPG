#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --config <path_to_config> --start_crf <start_val> --end_crf <end_val> --step_crf <step_val> [--output_dir <output_directory>] [--include_crf0]"
    echo "Example: $0 --config configs/infer_configs/PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml --start_crf 24 --end_crf 24 --step_crf 1 --include_crf0"
    exit 1
}

output_dir=""
include_crf0=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --start_crf) start_crf="$2"; shift ;;
        --end_crf) end_crf="$2"; shift ;;
        --step_crf) step_crf="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --include_crf0) include_crf0=true ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all parameters are provided
if [ -z "$config" ] || [ -z "$start_crf" ] || [ -z "$end_crf" ] || [ -z "$step_crf" ]; then
    usage
fi

if [ ! -f "$config" ]; then
    echo "Error: Config file not found: $config"
    exit 1
fi

# Set default output directory if not provided
if [ -z "$output_dir" ]; then
    output_dir="$(dirname "$config")/generated_configs"
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"
echo "Output directory: $output_dir"

# Build list of CRFs to process
crf_values=()
if [ "$include_crf0" = true ]; then
    crf_values+=(0)
fi

for (( crf=start_crf; crf<=end_crf; crf+=step_crf )); do
    crf_values+=($crf)
done

# Loop through CRF values
for crf in "${crf_values[@]}"; do
    echo "Generating config for CRF: $crf"
    
    # Determine new filename
    base_name=$(basename "$config")
    
    # Extract the part before .yaml
    filename="${base_name%.yaml}"
    
    # Check if _BASIC is in the filename and replace it, otherwise append _CRF
    if [[ "$filename" == *"_BASIC"* ]]; then
        new_filename="${filename/_BASIC/_CRF$crf}.yaml"
    else
        new_filename="${filename}_CRF$crf.yaml"
    fi
    
    output_config="$output_dir/$new_filename"
    
    cp "$config" "$output_config"
    
    # UBFC-rPPG to UBFC-rPPG-h264 w/ corresponding crfs
    sed -i 's/DATASET: UBFC-rPPG\s*$/DATASET: UBFC-rPPG-h264/g' "$output_config"
    
    sed -i "s|/UBFC-rPPG\(-CRF[0-9]*\)\?\"|/UBFC-rPPG-CRF$crf\"|g" "$output_config"
    sed -i "s|/UBFC-rPPG\(-CRF[0-9]*\)\?-cache\"|/UBFC-rPPG-CRF$crf-cache\"|g" "$output_config"
    
    echo "Created: $output_config"
done
