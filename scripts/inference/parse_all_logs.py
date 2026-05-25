#!/usr/bin/env python3
"""Parse all inference logs and generate per-group CSV summaries."""

import os
import re
import glob
import pandas as pd


METRIC_RE = re.compile(r"(.+?)\s\((.+?)\):\s(\S+)\s\+/-\s(\S+)")
METHOD_RE = re.compile(r"Used Unsupervised Method:\s+(\w+)")


def parse_metrics_from_lines(lines):
    """Extract metric dict from a list of text lines."""
    metrics = {}
    for line in lines:
        m = METRIC_RE.search(line)
        if m:
            name = m.group(1).strip()
            # Keep only the metric short name (e.g. "FFT MAE" -> "MAE")
            short = name.replace("FFT ", "")
            metrics[f"{short}"] = m.group(3)
            metrics[f"{short}_Std"] = m.group(4)
    return metrics


def parse_neural_log(log_path):
    """Parse a single neural-method log file. Returns one dict of metrics."""
    with open(log_path) as f:
        lines = f.readlines()
    metrics = parse_metrics_from_lines(lines)
    crf_match = re.search(r"CRF(\d+)", os.path.basename(log_path))
    metrics["CRF"] = int(crf_match.group(1)) if crf_match else -1
    metrics["Config"] = os.path.basename(log_path).replace(".log", "")
    return metrics


def parse_unsupervised_log(log_path):
    """Parse an unsupervised log with multiple methods. Returns list of dicts."""
    with open(log_path) as f:
        lines = f.readlines()

    crf_match = re.search(r"CRF(\d+)", os.path.basename(log_path))
    crf = int(crf_match.group(1)) if crf_match else -1

    # Split into segments by method header
    results = []
    current_method = None
    current_lines = []

    for line in lines:
        m = METHOD_RE.search(line)
        if m:
            # Save previous method's metrics
            if current_method and current_lines:
                metrics = parse_metrics_from_lines(current_lines)
                metrics["CRF"] = crf
                metrics["Method"] = current_method
                results.append(metrics)
            current_method = m.group(1)
            current_lines = []
        elif current_method:
            current_lines.append(line)

    # Save last method
    if current_method and current_lines:
        metrics = parse_metrics_from_lines(current_lines)
        metrics["CRF"] = crf
        metrics["Method"] = current_method
        results.append(metrics)

    return results


COLUMNS_NEURAL = [
    "Config", "CRF", "MAE", "MAE_Std", "RMSE", "RMSE_Std",
    "MAPE", "MAPE_Std", "Pearson", "Pearson_Std", "SNR", "SNR_Std",
]

COLUMNS_UNSUPERVISED = [
    "Method", "CRF", "MAE", "MAE_Std", "RMSE", "RMSE_Std",
    "MAPE", "MAPE_Std", "Pearson", "Pearson_Std", "SNR", "SNR_Std",
]


def save_csv(rows, columns, csv_path):
    """Save rows as CSV, auto-filling missing columns."""
    if not rows:
        print(f"  No data for {csv_path}")
        return
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = "N/A"
    df = df[[c for c in columns if c in df.columns]]
    df = df.sort_values("CRF").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows to {csv_path}")


def is_unsupervised_log(log_path):
    """Detect unsupervised log by reading the full file for method markers."""
    with open(log_path) as f:
        for line in f:
            if METHOD_RE.search(line):
                return True
    return False


def process_log_directory(log_dir, csv_path):
    """Process a directory of .log files into one CSV."""
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        return

    if is_unsupervised_log(log_files[0]):
        rows = []
        for lf in log_files:
            rows.extend(parse_unsupervised_log(lf))
        save_csv(rows, COLUMNS_UNSUPERVISED, csv_path)
    else:
        rows = [parse_neural_log(lf) for lf in log_files]
        save_csv(rows, COLUMNS_NEURAL, csv_path)


def main():
    base_dir = os.path.join("results", "inference_logs")

    # Walk all directories that contain .log files
    for root, dirs, files in os.walk(base_dir):
        log_files = [f for f in files if f.endswith(".log")]
        if not log_files:
            continue

        # Generate CSV right next to the log files
        csv_name = os.path.basename(root) + "_results.csv"
        csv_path = os.path.join(root, csv_name)
        print(f"Processing {root} ({len(log_files)} logs)...")
        process_log_directory(root, csv_path)

    # Generate combined CSVs for nested groups (A/, C/)
    for group in sorted(os.listdir(base_dir)):
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path):
            continue
        subdirs = [d for d in os.listdir(group_path)
                   if os.path.isdir(os.path.join(group_path, d))]
        # Only combine if there are subdirectories that have CSVs
        sub_csvs = []
        for sd in sorted(subdirs):
            csv = os.path.join(group_path, sd, f"{sd}_results.csv")
            if os.path.exists(csv):
                sub_csvs.append(csv)
        if len(sub_csvs) > 1:
            dfs = []
            for csv in sub_csvs:
                df = pd.read_csv(csv)
                df.insert(0, "SubExperiment", os.path.basename(os.path.dirname(csv)))
                dfs.append(df)
            combined = pd.concat(dfs, ignore_index=True)
            combined_csv = os.path.join(group_path, f"{group}_combined_results.csv")
            combined.to_csv(combined_csv, index=False)
            print(f"Saved combined {group} CSV ({len(combined)} rows) to {combined_csv}")


if __name__ == "__main__":
    main()
