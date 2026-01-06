import os
import glob
import subprocess
import re
import pandas as pd
from tqdm import tqdm
import sys
import argparse

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description="Run batch inference on rPPG-Toolbox configs.")
    parser.add_argument("--config_dir", type=str, default='configs/infer_configs', 
                        help="Directory containing .yaml configuration files.")
    parser.add_argument("--output_csv", type=str, default='results/batch_results.csv',
                        help="Path to the output CSV file.")
    args = parser.parse_args()

    # Convert relative paths to absolute if needed, or join with project_root
    if not os.path.isabs(args.config_dir):
        configs_dir = os.path.join(project_root, args.config_dir)
    else:
        configs_dir = args.config_dir
        
    if not os.path.isabs(args.output_csv):
        output_csv = os.path.join(project_root, args.output_csv)
    else:
        output_csv = args.output_csv

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Find all yaml config files
    config_files = sorted(glob.glob(os.path.join(configs_dir, '*.yaml')))
    
    if not config_files:
        print(f"No configuration files found in {configs_dir}")
        return

    print(f"Found {len(config_files)} configuration files.")
    print(f"Results will be saved to: {output_csv}")
    
    results = []
    
    # Regex pattern to capture metrics
    # Example: FFT MAE (FFT Label): 1.5153556034482758 +/- 0.5764673263538488
    # Also handles (dB) at the end for SNR
    metric_pattern = re.compile(r'(.+?)\s\((.+?)\):\s([\d\.]+)\s\+/-\s([\d\.]+)')

    for config_file in tqdm(config_files, desc="Running Batch Inference"):
        config_name = os.path.basename(config_file)
        print(f"\nProcessing: {config_name}")
        
        try:
            # Run main.py with the config file
            # We want to capture output but also show it to the user.
            # Using subprocess.Popen to stream output
            process = subprocess.Popen(
                ['python', 'main.py', '--config_file', config_file],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            
            # Stream output
            for line in process.stdout:
                print(line, end='') # Print to console
                output_lines.append(line)
                
            process.wait()
            
            if process.returncode != 0:
                print(f"Error running {config_name}. Return code: {process.returncode}")
                continue
                
            # Parse metrics from captured output
            # We want to match the shell script's metrics: MAE, RMSE, MAPE, Pearson, SNR
            # and their +/- (standard deviation) values.
            
            # Initialize with N/A
            file_metrics = {
                'Config': config_name,
                'MAE': 'N/A', 'MAE_Std': 'N/A',
                'RMSE': 'N/A', 'RMSE_Std': 'N/A',
                'MAPE': 'N/A', 'MAPE_Std': 'N/A',
                'Pearson': 'N/A', 'Pearson_Std': 'N/A',
                'SNR': 'N/A', 'SNR_Std': 'N/A'
            }
            
            extracted_any = False
            for line in output_lines:
                match = metric_pattern.search(line)
                if match:
                    metric_full_name = match.group(1).strip() # e.g., "FFT MAE"
                    value = match.group(3)
                    std_dev = match.group(4)
                    
                    if "MAE" in metric_full_name:
                        file_metrics['MAE'] = value
                        file_metrics['MAE_Std'] = std_dev
                        extracted_any = True
                    elif "RMSE" in metric_full_name:
                        file_metrics['RMSE'] = value
                        file_metrics['RMSE_Std'] = std_dev
                        extracted_any = True
                    elif "MAPE" in metric_full_name:
                        file_metrics['MAPE'] = value
                        file_metrics['MAPE_Std'] = std_dev
                        extracted_any = True
                    elif "Pearson" in metric_full_name:
                        file_metrics['Pearson'] = value
                        file_metrics['Pearson_Std'] = std_dev
                        extracted_any = True
                    elif "SNR" in metric_full_name:
                        file_metrics['SNR'] = value
                        file_metrics['SNR_Std'] = std_dev
                        extracted_any = True
            
            if not extracted_any:
                print(f"  WARNING: No metrics found for {config_name}")
            else:
                print(f"  Extracted: MAE={file_metrics['MAE']} +/- {file_metrics['MAE_Std']}, RMSE={file_metrics['RMSE']} +/- {file_metrics['RMSE_Std']}, MAPE={file_metrics['MAPE']} +/- {file_metrics['MAPE_Std']}, Pearson={file_metrics['Pearson']} +/- {file_metrics['Pearson_Std']}, SNR={file_metrics['SNR']} +/- {file_metrics['SNR_Std']}")
            
            results.append(file_metrics)

        except Exception as e:
            print(f"An error occurred while processing {config_name}: {e}")
            
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        # Reorder columns to match the shell script's header
        columns = ['Config', 'MAE', 'MAE_Std', 'RMSE', 'RMSE_Std', 'MAPE', 'MAPE_Std', 'Pearson', 'Pearson_Std', 'SNR', 'SNR_Std']
        df = df[columns]
        df.to_csv(output_csv, index=False)
        print(f"\nBatch inference completed. Results saved to {output_csv}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
