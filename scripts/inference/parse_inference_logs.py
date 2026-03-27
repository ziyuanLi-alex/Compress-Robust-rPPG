#!/usr/bin/env python3
"""Parse inference logs and generate CSV summary."""

import os
import re
import glob
import pandas as pd
import argparse


def parse_log_file(log_path):
    """Extract metrics from a log file."""
    metric_pattern = re.compile(r"(.+?)\s\((.+?)\):\s(\S+)\s\+/-\s(\S+)")

    metrics = {
        "Config": os.path.basename(log_path).replace(".log", ""),
        "MAE": "N/A",
        "MAE_Std": "N/A",
        "RMSE": "N/A",
        "RMSE_Std": "N/A",
        "MAPE": "N/A",
        "MAPE_Std": "N/A",
        "Pearson": "N/A",
        "Pearson_Std": "N/A",
        "SNR": "N/A",
        "SNR_Std": "N/A",
    }

    try:
        with open(log_path, "r") as f:
            for line in f:
                match = metric_pattern.search(line)
                if match:
                    metric_name = match.group(1).strip()
                    value = match.group(3)
                    std = match.group(4)

                    if "MAE" in metric_name:
                        metrics["MAE"] = value
                        metrics["MAE_Std"] = std
                    elif "RMSE" in metric_name:
                        metrics["RMSE"] = value
                        metrics["RMSE_Std"] = std
                    elif "MAPE" in metric_name:
                        metrics["MAPE"] = value
                        metrics["MAPE_Std"] = std
                    elif "Pearson" in metric_name:
                        metrics["Pearson"] = value
                        metrics["Pearson_Std"] = std
                    elif "SNR" in metric_name:
                        metrics["SNR"] = value
                        metrics["SNR_Std"] = std
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parse inference logs to CSV")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/inference_logs",
        help="Directory containing .log files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/batch_results.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))

    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return

    results = [parse_log_file(log) for log in log_files]
    df = pd.DataFrame(results)

    # Ensure column order
    columns = [
        "Config", "MAE", "MAE_Std", "RMSE", "RMSE_Std",
        "MAPE", "MAPE_Std", "Pearson", "Pearson_Std", "SNR", "SNR_Std"
    ]
    df = df[columns]
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(results)} results to {args.output_csv}")


if __name__ == "__main__":
    main()
