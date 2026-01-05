#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --config_dir <directory_with_configs> [--output_csv <output_csv_path>]"
    echo "Example: $0 --config_dir configs/infer_configs/generated_configs --output_csv results/my_results.csv"
    exit 1
}

output_csv="results/batch_results.csv"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_dir) config_dir="$2"; shift ;;
        --output_csv) output_csv="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$config_dir" ]; then
    echo "Error: --config_dir is required."
    usage
fi

if [ ! -d "$config_dir" ]; then
    echo "Error: Config directory not found: $config_dir"
    exit 1
fi

# Create directory for output CSV if it doesn't exist
mkdir -p "$(dirname "$output_csv")"

# Initialize CSV file if it doesn't exist, or overwrite? 
# Usually for batch runs we might want to start fresh or append. 
# Let's overwrite to ensure clean results for the run, or check if user wants append. 
# For simplicity and typical batch usage: overwrite or create new.
echo "Config,MAE,MAE_Std,RMSE,RMSE_Std,MAPE,MAPE_Std,Pearson,Pearson_Std,SNR,SNR_Std" > "$output_csv"
echo "Results will be saved to: $output_csv"

# Iterate over yaml files
count=0
total=$(find "$config_dir" -name "*.yaml" | wc -l)

if [ "$total" -eq 0 ]; then
    echo "No .yaml files found in $config_dir"
    exit 0
fi

for config_file in "$config_dir"/*.yaml; do
    config_name=$(basename "$config_file")
    echo "Processing ($((++count))/$total): $config_name"
    
    # Run inference and capture output
    # Source conda to enable activate command
    source /home/zyuanli/miniconda3/etc/profile.d/conda.sh
    conda activate rppg-toolbox
    
    # We use 2>&1 to combine stderr and stdout just in case, though usually print is stdout
    output=$(python main.py --config_file "$config_file" 2>&1)
    
    # Extract metrics using grep and sed/awk
    # Pattern: FFT MAE (FFT Label): 8.642578125 +/- 2.388339277616328
    
    # Helper to extract value
    extract_metric() {
        local content="$1"
        local metric_name="$2"
        # Find line with metric name, get part after colon, then remove everything starting from " +/-"
        echo "$content" | grep "$metric_name" | awk -F': ' '{print $2}' | sed 's/ +\/-\ .*//'
    }

    # Helper to extract std deviation
    extract_std() {
        local content="$1"
        local metric_name="$2"
        # Find line with metric name, get part after "+/- ", then remove any trailing text like (dB)
        # Using sed to remove everything up to "+/- "
        # echo "DEBUG STD INPUT: $(echo "$content" | grep "$metric_name")" >&2
        echo "$content" | grep "$metric_name" | awk -v FS='\\+/- ' '{print $2}' | cut -d' ' -f1 
    }
    
    mae=$(extract_metric "$output" "FFT MAE")
    mae_std=$(extract_std "$output" "FFT MAE")
    
    rmse=$(extract_metric "$output" "FFT RMSE")
    rmse_std=$(extract_std "$output" "FFT RMSE")
    
    mape=$(extract_metric "$output" "FFT MAPE")
    mape_std=$(extract_std "$output" "FFT MAPE")
    
    pearson=$(extract_metric "$output" "FFT Pearson")
    pearson_std=$(extract_std "$output" "FFT Pearson")
    
    snr=$(extract_metric "$output" "FFT SNR")
    snr_std=$(extract_std "$output" "FFT SNR")
    
    # Handle cases where metrics might be missing (failed run)
    if [ -z "$mae" ]; then
        echo "  WARNING: Failed to Extract metrics for $config_name"
        # define empty values or error
        mae="N/A"
        mae_std="N/A"
        rmse="N/A"
        rmse_std="N/A"
        mape="N/A"
        mape_std="N/A"
        pearson="N/A"
        pearson_std="N/A"
        snr="N/A"
        snr_std="N/A"
    else
        echo "  Extracted: MAE=$mae +/- $mae_std, RMSE=$rmse +/- $rmse_std, MAPE=$mape +/- $mape_std, Pearson=$pearson +/- $pearson_std, SNR=$snr +/- $snr_std"
    fi
    
    echo "$config_name,$mae,$mae_std,$rmse,$rmse_std,$mape,$mape_std,$pearson,$pearson_std,$snr,$snr_std" >> "$output_csv"

done

echo "Batch inference finished. results saved in $output_csv"
