import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec


def main():
    parser = argparse.ArgumentParser(description='Visualize Unsupervised Method Degradation vs CRF')
    parser.add_argument('raw_csv', type=str, help='Path to raw (CRF=-1) results CSV')
    parser.add_argument('crf_csv', type=str, help='Path to CRF compressed results CSV')
    parser.add_argument('--output_dir', type=str, default='results/figures/', 
                        help='Directory to save the output plots')
    parser.add_argument('--metrics', nargs='+', type=str, default=['MAE', 'RMSE', 'Pearson', 'MAPE'],
                        help='Metrics to plot')
    
    args = parser.parse_args()

    # Load data - only use CRF CSV (contains CRF=0 and CRF>=14)
    df_crf = pd.read_csv(args.crf_csv)
    
    # Filter: keep CRF=0 and >= 14
    combined_df = df_crf[df_crf['CRF'].isin([0]) | (df_crf['CRF'] >= 14)].copy()
    
    # Set seaborn theme
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.set_context("paper", font_scale=2.5)
    sns.set_palette("tab10")

    # Define methods order for consistent coloring
    methods = ['ICA', 'POS', 'CHROM', 'GREEN', 'LGI', 'PBV', 'OMIT']
    method_colors = dict(zip(methods, sns.color_palette("tab10", len(methods))))
    
    # Metrics configuration with y-axis limits
    metrics_config = {
        'MAE': {'ylabel': 'MAE (bpm)', 'ylim': (0, 40)},
        'RMSE': {'ylabel': 'RMSE (bpm)', 'ylim': (0, 50)},
        'Pearson': {'ylabel': 'Pearson r', 'ylim': (-0.1, 1.0)},
        'MAPE': {'ylabel': 'MAPE (%)', 'ylim': (0, 40)}
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual metric plots
    for metric_idx, metric in enumerate(args.metrics):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 5], wspace=0.18)
        
        config = metrics_config[metric]
        
        # Left axis (CRF=0)
        ax_left = fig.add_subplot(gs[0])
        # Right axis (CRF 14 to 24)
        ax_right = fig.add_subplot(gs[1], sharey=ax_left)
        
        # Set limits
        ax_left.set_xlim(-0.5, 0.5)
        ax_right.set_xlim(13.5, 24.5)
        ax_left.set_ylim(config['ylim'])
        ax_right.set_ylim(config['ylim'])
        
        # Set x-axis ticks to integers only
        ax_left.set_xticks([0])
        ax_right.set_xticks([14, 16, 18, 20, 22, 24])
        
        # Hide spines for broken axis effect
        ax_left.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        
        # Add diagonal break marks
        d = 0.015
        # Left axis break marks (right side)
        ax_left.plot((1-d, 1+d), (-d, +d), transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax_left.plot((1-d, 1+d), (1-d, 1+d), transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1.5)
        # Right axis break marks (left side)
        ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax_right.plot((-d, +d), (1-d, 1+d), transform=ax_right.transAxes, color='k', clip_on=False, linewidth=1.5)
        
        # Hide ticks on inner sides
        ax_left.tick_params(right=False)
        ax_right.tick_params(left=False, labelleft=False)
        
        # Set ylabel
        ax_left.set_ylabel(config['ylabel'], fontsize=20, fontweight='bold')
        ax_left.yaxis.set_label_coords(-0.25, 0.5)
        
        # Plot each method
        for method_idx, method in enumerate(methods):
            method_data = combined_df[combined_df['Method'] == method]
            color = method_colors[method]
            
            # Main line plot for CRF >= 14 (right axis)
            data_main = method_data[method_data['CRF'] >= 14]
            if not data_main.empty:
                ax_right.plot(data_main['CRF'], data_main[metric],
                             marker='o', markersize=10, linewidth=2.5,
                             color=color, label=method)
                
                # Error bars
                ax_right.errorbar(data_main['CRF'], data_main[metric],
                                 yerr=data_main[f'{metric}_Std'],
                                 fmt='none', color=color, ecolor=color,
                                 capsize=5, alpha=0.7, linewidth=2)
            
            # Plot CRF=0 (left axis)
            data_crf0 = method_data[method_data['CRF'] == 0]
            if not data_crf0.empty:
                ax_left.errorbar(data_crf0['CRF'], data_crf0[metric],
                                yerr=data_crf0[f'{metric}_Std'],
                                fmt='o', color=color, ecolor=color,
                                markersize=12, capsize=5, alpha=0.7, linewidth=2)
        
        # Grid
        ax_left.grid(True, linestyle=':', alpha=0.6)
        ax_right.grid(True, linestyle=':', alpha=0.6)
        
        # Tick labels
        ax_left.tick_params(labelsize=18)
        ax_right.tick_params(labelsize=18)
        
        # Title
        ax_right.set_title(metric, fontsize=24, fontweight='bold', pad=10)
        
        # Remove legend from plot
        legend = ax_right.get_legend()
        if legend:
            legend.remove()
        legend = ax_left.get_legend()
        if legend:
            legend.remove()
        
        # Common x-label
        fig.text(0.5, 0.02, 'CRF Value', ha='center', fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual metric plot
        output_path = output_dir / f'unsupervised_degradation_{metric}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path.resolve()}")
        plt.close()
    
    # Create separate legend figure
    fig_legend = plt.figure(figsize=(14, 2))
    legend_handles = []
    legend_labels = []
    
    for method_idx, method in enumerate(methods):
        color = method_colors[method]
        line = plt.Line2D([0], [0], marker='o', color=color, markersize=10, linewidth=2.5, label=method)
        legend_handles.append(line)
        legend_labels.append(method)
    
    fig_legend.legend(legend_handles, legend_labels, loc='center', ncol=len(methods), 
                      fontsize=20, frameon=False)
    plt.axis('off')
    
    # Save legend
    legend_path = output_dir / 'unsupervised_degradation_legend.png'
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    print(f"Legend saved to {legend_path.resolve()}")
    plt.close()


if __name__ == '__main__':
    main()
