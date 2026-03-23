import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path

def parse_crf(config_name):
    # Search for CRF followed by digits
    match = re.search(r'CRF(\d+)', config_name)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description='Visualize MAE drop vs CRF with Standard Deviation')
    parser.add_argument('input_csvs', nargs='+', type=str, help='Path(s) to input CSV files')
    parser.add_argument('--output', type=str, default='scripts/visualization/mae_drop_plot.png', help='Path to save the output plot')
    parser.add_argument('--omit', nargs='+', type=str, help='Range(s) to omit from X-axis. Expects pairs of numbers: start end [start end ...]')
    
    args = parser.parse_args()

    # Determine X-axis ranges
    # Data Loading
    all_data = []
    
    # Set seaborn theme
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.set_context("paper", font_scale=2.8)
    # sns.color_palette("tab10") # set_theme/set_context doesn't return palette to variable usually, but user snippet had it as a standalone line which effectively sets the active palette? No, sns.color_palette just returns it. sns.set_palette sets it.
    sns.set_palette("tab10")

    for file_path in args.input_csvs:
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            df = pd.read_csv(file_path)
            
            # Extract CRF values
            df['CRF'] = df['Config'].apply(parse_crf)
            
            # Check if any CRFs were found
            if df['CRF'].isnull().all():
                 print(f"Warning: No 'CRF' values could be parsed from 'Config' column in {file_path}. Skipping.")
                 continue

            # Drop rows where CRF could not be parsed
            df_clean = df.dropna(subset=['CRF']).copy()
            df_clean['CRF'] = df_clean['CRF'].astype(int)
            
            # Add Source column for identifying different CSVs/Models
            df_clean['Source'] = path_obj.stem
            
            # Sort by CRF
            df_clean = df_clean.sort_values('CRF')
            
            all_data.append(df_clean)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_data:
        print("No valid data found to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get unique sources and assign colors
    sources = combined_df['Source'].unique()
    palette = sns.color_palette("tab10", len(sources))

    # Determine X-axis ranges
    all_crf = combined_df['CRF'].values
    min_crf, max_crf = all_crf.min(), all_crf.max()
    pad = (max_crf - min_crf) * 0.05 if max_crf != min_crf else 1.0 # 5% padding
    
    # Parse omit ranges
    omit_ranges = []
    if args.omit:
        try:
            # Expect pairs of numbers
            if len(args.omit) % 2 != 0:
                print("Error: --omit requires pairs of numbers (start end).")
                return
            for i in range(0, len(args.omit), 2):
                omit_ranges.append((float(args.omit[i]), float(args.omit[i+1])))
            omit_ranges.sort()
        except ValueError:
            print("Error: --omit values must be numbers.")
            return

    # Calculate visible segments
    # Start with full range including padding
    current_start = min_crf - pad
    final_end = max_crf + pad
    
    segments = []
    if not omit_ranges:
        segments.append((current_start, final_end))
    else:
        # Create segments based on omit ranges
        # Logic: [current_start, omit_1_start], [omit_1_end, omit_2_start], ...
        
        # Adjust first segment start if omit cuts into it?
        # Typically omit is within the range.
        
        last_pos = current_start
        for o_start, o_end in omit_ranges:
            if o_start > last_pos:
                segments.append((last_pos, o_start))
            last_pos = max(last_pos, o_end)
        
        if last_pos < final_end:
            segments.append((last_pos, final_end))

    # Calculate width ratios
    ratios = [s[1] - s[0] for s in segments]
    
    # Create subplots
    fig, axes = plt.subplots(1, len(segments), figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': ratios})
    if len(segments) == 1:
        axes = [axes]

    # Plot on each subplot
    for ax_idx, ax in enumerate(axes):
        segment = segments[ax_idx]
        ax.set_xlim(segment)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Hide spines for broken axis look
        if len(segments) > 1:
            if ax_idx < len(segments) - 1:
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelright=False)  # don't put tick labels at the top
                
                # Add break marks
                d = .015  # how big to make the diagonal lines in axes coordinates
                kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            if ax_idx > 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft=False)
                ax.tick_params(left=False) # Hide ticks
                
                d = .015
                kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                ax.plot((-d, +d), (-d, +d), **kwargs)
                ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        # Plot data on this axis
        for i, source in enumerate(sources):
            data = combined_df[combined_df['Source'] == source]
            color = palette[i]
            
            data_main = data[data['CRF'] != 0]
            data_crf0 = data[data['CRF'] == 0]
            
            if not data_main.empty:
                sns.lineplot(
                    data=data_main, 
                    x='CRF', 
                    y='MAE', 
                    label=source if ax_idx == 0 else None, # Only label once
                    color=color, 
                    marker='o',
                    markersize=10, # Increased from 8
                    linewidth=3,   # Increased from 2
                    ax=ax,
                    legend=False
                )
                
                ax.errorbar(
                    x=data_main['CRF'], 
                    y=data_main['MAE'], 
                    yerr=data_main['MAE_Std'], 
                    fmt='none', 
                    color=color, 
                    ecolor=color, 
                    capsize=5, 
                    alpha=0.6,
                    linewidth=2 # Increased from 1.5
                )

            if not data_crf0.empty:
                # Logic for legend: only add to first subplot if available
                do_label = (ax_idx == 0) and (data_main.empty)
                
                ax.errorbar(
                    x=data_crf0['CRF'], 
                    y=data_crf0['MAE'], 
                    yerr=data_crf0['MAE_Std'], 
                    fmt='o', 
                    color=color, 
                    ecolor=color, 
                    capsize=5, 
                    markersize=10, # Increased from 8
                    label=source if do_label else None
                )

        # Remove automatic seaborn labels
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Common labels
    fig.text(0.5, 0.04, 'CRF Value', ha='center', fontsize=26, fontweight='bold')
    # y-label is tricky with subplots. Put it on the first axis or figure?
    fig.text(0.04, 0.5, 'MAE', va='center', rotation='vertical', fontsize=26, fontweight='bold')
    
    # Title - Optional, maybe remove if not in snippet? I'll keep it but larger.
    # fig.suptitle('MAE Drop vs CRF Value', fontsize=16) 
    
    # Update title handling: User snippet didn't used title, commented out. 
    # I will comment it out too to match "shown as this section".
    # fig.suptitle('MAE Drop vs CRF Value', fontsize=30, fontweight='bold')

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Move legend to be cleaner, maybe inside if room, or outside.
    # Snippet doesn't show legend code (except hue='Type'), but usually seaborn puts it automatically.
    # I'll keep my manual legend but font scaled.
    # User requested top left.
    fig.legend(by_label.values(), by_label.keys(), loc='upper left', title='Source', fontsize=20, title_fontsize=22, bbox_to_anchor=(0.12, 0.93))
    
    results_path = Path(args.output)
    
    # Adjust layout
    # plt.subplots_adjust(right=0.85) # Make room for legend if outside
    plt.tight_layout(rect=[0.05, 0.05, 1, 1]) # Adjust for fig text
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path.resolve()}")

if __name__ == '__main__':
    main()
