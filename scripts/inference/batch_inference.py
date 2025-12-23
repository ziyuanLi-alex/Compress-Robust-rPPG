import os
import glob
import subprocess
import re
import csv
import pandas as pd
from tqdm import tqdm
import sys

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configs_dir = os.path.join(project_root, 'configs', 'infer_configs')
    results_dir = os.path.join(project_root, 'results')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Find all yaml config files
    config_files = glob.glob(os.path.join(configs_dir, '*.yaml'))
    
    if not config_files:
        print(f"No configuration files found in {configs_dir}")
        return

    print(f"Found {len(config_files)} configuration files.")
    
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
            file_metrics = {'Config': config_name}
            
            for line in output_lines:
                match = metric_pattern.search(line)
                if match:
                    metric_name = match.group(1).strip()
                    label_type = match.group(2).strip()
                    value = float(match.group(3))
                    std_dev = float(match.group(4))
                    
                    # Store as Metric Name, Metric Name_Std
                    file_metrics[metric_name] = value
                    file_metrics[f"{metric_name}_Std"] = std_dev
            
            if len(file_metrics) > 1: # Config name is always there
                results.append(file_metrics)
            else:
                print(f"No metrics found for {config_name}")

        except Exception as e:
            print(f"An error occurred while processing {config_name}: {e}")
            
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_csv = os.path.join(results_dir, 'batch_inference_results.csv')
        df.to_csv(output_csv, index=False)
        print(f"\nBatch inference completed. Results saved to {output_csv}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
