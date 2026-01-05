#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --config <path_to_config> --start_crf <start_val> --end_crf <end_val> --step_crf <step_val> [--output_dir <output_directory>]"
    echo "Example: $0 --config configs/infer_configs/PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml --start_crf 24 --end_crf 24 --step_crf 1 --output_dir configs/infer_configs/generated"
    exit 1
}

output_dir=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --start_crf) start_crf="$2"; shift ;;
        --end_crf) end_crf="$2"; shift ;;
        --step_crf) step_crf="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
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

# Loop through CRF values
for (( crf=start_crf; crf<=end_crf; crf+=step_crf )); do
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
    
    # Update DATASET: UBFC-rPPG -> UBFC-rPPG-h264
    sed -i 's/DATASET: UBFC-rPPG/DATASET: UBFC-rPPG-h264/g' "$output_config"
    
    # Update DATA_PATH: UBFC-rPPG -> UBFC-rPPG-CRF{val}
    # Using a flexible regex to handle potential trailing partial matches if user didn't quote paths exactly as expected, 
    # but based on the example files, direct replacement of the folder name seems safest.
    # We replace "UBFC-rPPG" with "UBFC-rPPG-CRF$crf" ONLY in the DATA_PATH line roughly.
    # Actually, the requirement is to update paths.
    # Let's be specific to the lines we saw:
    # DATA_PATH: "/home/zyuanli/dev/lib/data/UBFC-rPPG"
    # CACHED_PATH: "/home/zyuanli/dev/lib/data/UBFC-rPPG-cache"
    
    # Note: If the input config already has -CRF config in it (e.g. copying from another CRF config), 
    # we might need to be careful. But the task implies starting from a base config (BASIC).
    # Assuming BASIC config has "UBFC-rPPG" and "UBFC-rPPG-cache".
    
    sed -i "s|/UBFC-rPPG\"|/UBFC-rPPG-CRF$crf\"|g" "$output_config"
    sed -i "s|/UBFC-rPPG-cache\"|/UBFC-rPPG-CRF$crf-cache\"|g" "$output_config"
    
    echo "Created: $output_config"
done
