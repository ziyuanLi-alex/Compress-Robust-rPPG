import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Compare Unsupervised Methods: Raw (CRF=-1) vs CRF=0')
    parser.add_argument('raw_csv', type=str, help='Path to raw (CRF=-1) results CSV')
    parser.add_argument('crf_csv', type=str, help='Path to CRF compressed results CSV')
    parser.add_argument('--output', type=str, default='results/figures/unsupervised_baseline_comparison.png',
                        help='Path to save the output plot')
    parser.add_argument('--metric', type=str, default='MAE', help='Metric to plot')
    
    args = parser.parse_args()

    # Load data
    df_raw = pd.read_csv(args.raw_csv)
    df_crf = pd.read_csv(args.crf_csv)
    
    # Filter to only CRF=-1 and CRF=0
    df_raw_filtered = df_raw[df_raw['CRF'] == -1].copy()
    df_crf_filtered = df_crf[df_crf['CRF'] == 0].copy()
    
    # Set seaborn theme
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.set_context("paper", font_scale=2.5)
    sns.set_palette("tab10")

    # Define methods order
    methods = ['ICA', 'POS', 'CHROM', 'GREEN', 'LGI', 'PBV', 'OMIT']
    method_colors = dict(zip(methods, sns.color_palette("tab10", len(methods))))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for plotting
    x = np.arange(len(methods))
    width = 0.35
    
    # Extract data
    raw_data = []
    crf0_data = []
    raw_std = []
    crf0_std = []
    
    for method in methods:
        raw_row = df_raw_filtered[df_raw_filtered['Method'] == method]
        crf0_row = df_crf_filtered[df_crf_filtered['Method'] == method]
        
        if not raw_row.empty:
            raw_data.append(raw_row[args.metric].values[0])
            raw_std.append(raw_row[f'{args.metric}_Std'].values[0])
        else:
            raw_data.append(0)
            raw_std.append(0)
            
        if not crf0_row.empty:
            crf0_data.append(crf0_row[args.metric].values[0])
            crf0_std.append(crf0_row[f'{args.metric}_Std'].values[0])
        else:
            crf0_data.append(0)
            crf0_std.append(0)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, raw_data, width, yerr=raw_std, 
                   label='Raw (CRF=-1)', color='steelblue', capsize=5, linewidth=1.5)
    bars2 = ax.bar(x + width/2, crf0_data, width, yerr=crf0_std,
                   label='CRF=0', color='coral', capsize=5, linewidth=1.5)
    
    # Labels and title
    ax.set_xlabel('Method', fontsize=20, fontweight='bold')
    ax.set_ylabel(f'{args.metric}', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=18, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    plt.tight_layout()
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path.resolve()}")


if __name__ == '__main__':
    main()
