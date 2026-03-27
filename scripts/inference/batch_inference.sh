#!/bin/bash

# Batch inference script for rPPG-Toolbox
# Runs all .yaml configs in a directory and stores raw output to log files

set -e

# Configuration
CONFIG_DIR="${1:-configs/infer_configs}"
OUTPUT_DIR="${2:-results/inference_logs}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration directory: $CONFIG_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Find all yaml config files
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" -type f | sort)

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "No configuration files found in $CONFIG_DIR"
    exit 1
fi

echo "Found ${#CONFIG_FILES[@]} configuration files."
echo ""

# Process each config file
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    LOG_FILE="$OUTPUT_DIR/${CONFIG_NAME}.log"

    echo "=========================================="
    echo "Processing: $CONFIG_NAME"
    echo "Log file: $LOG_FILE"
    echo "=========================================="

    # Run main.py and capture all output to log file
    # Also tee to stdout for live monitoring
    python "$PROJECT_ROOT/main.py" --config_file "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

    echo ""
    echo "Completed: $CONFIG_NAME"
    echo ""
done

echo "=========================================="
echo "Batch inference completed."
echo "Logs saved to: $OUTPUT_DIR"
echo "=========================================="
