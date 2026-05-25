import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec


def main():
    parser = argparse.ArgumentParser(description='Visualize PhysFormer vs PhysMamba Degradation vs CRF')
    parser.add_argument('physformer_csv', type=str, help='Path to PhysFormer results CSV')
    parser.add_argument('physmamba_csv', type=str, help='Path to PhysMamba results CSV')
    parser.add_argument('--output_dir', type=str, default='results/figures/', 
                        help='Directory to save the output plots')
    parser.add_argument('--metrics', nargs='+', type=str, default=['MAE', 'RMSE', 'Pearson', 'MAPE'],
                        help='Metrics to plot')
    
    args = parser.parse_args()

    # Load data
    df_physformer = pd.read_csv(args.physformer_csv)
    df_physmamba = pd.read_csv(args.physmamba_csv)
    
    # Add Model column
    df_physformer['Model'] = 'PhysFormer'
    df_physmamba['Model'] = 'PhysMamba'
    
    # Combine data
    combined_df = pd.concat([df_physformer, df_physmamba], ignore_index=True)
    
    # Set seaborn theme
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.set_context("paper", font_scale=2.5)

    # Model colors
    model_colors = {
        'PhysFormer': '#1f77b4',  # blue
        'PhysMamba': '#ff7f0e'    # orange
    }
    
    # Metrics configuration with y-axis limits
    metrics_config = {
        'MAE': {'ylabel': 'MAE (bpm)', 'ylim': (0, 35)},
        'RMSE': {'ylabel': 'RMSE (bpm)', 'ylim': (0, 45)},
        'Pearson': {'ylabel': 'Pearson r', 'ylim': (-0.2, 1.0)},
        'MAPE': {'ylabel': 'MAPE (%)', 'ylim': (0, 35)}
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
        
        # Plot each model
        for model_idx, model in enumerate(['PhysFormer', 'PhysMamba']):
            model_data = combined_df[combined_df['Model'] == model]
            color = model_colors[model]
            
            # Main line plot for CRF >= 14 (right axis)
            data_main = model_data[model_data['CRF'] >= 14]
            if not data_main.empty:
                ax_right.plot(data_main['CRF'], data_main[metric],
                             marker='o', markersize=10, linewidth=2.5,
                             color=color, label=model)
                
                # Error bars
                ax_right.errorbar(data_main['CRF'], data_main[metric],
                                 yerr=data_main[f'{metric}_Std'],
                                 fmt='none', color=color, ecolor=color,
                                 capsize=5, alpha=0.7, linewidth=2)
            
            # Plot CRF=0 (left axis)
            data_crf0 = model_data[model_data['CRF'] == 0]
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
        
        # Add legend inside the plot (top right)
        ax_right.legend(loc='upper left', fontsize=18, frameon=False)
        
        # Common x-label
        fig.text(0.5, 0.02, 'CRF Value', ha='center', fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual metric plot
        output_path = output_dir / f'PURE_PhysFormer_PhysMamba_{metric}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path.resolve()}")
        plt.close()


if __name__ == '__main__':
    main()
