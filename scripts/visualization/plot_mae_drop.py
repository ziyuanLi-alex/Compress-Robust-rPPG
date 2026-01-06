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
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")

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
    palette = sns.color_palette("deep", len(sources))

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
            
            # Draw lines manually using plot to ensure they respect the axis limits (clip)
            # sns.lineplot might try to auto-scale or be weird on multiple axes?
            # Actually sns.lineplot is fine, just pass ax=ax
            
            # We must be careful: sns.lineplot will connect points.
            # If a segment is [0, 2], and we have points at 0 and 16.
            # The line 0->16 exists in data.
            # In axis [0, 2], it will draw 0->(offscreen). Correct.
            # In axis [14, 24], it will draw (offscreen)->16. Correct.
            
            if not data_main.empty:
                sns.lineplot(
                    data=data_main, 
                    x='CRF', 
                    y='MAE', 
                    label=source if ax_idx == 0 else None, # Only label once
                    color=color, 
                    marker='o',
                    markersize=8,
                    linewidth=2,
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
                    linewidth=1.5
                )

            if not data_crf0.empty:
                # Logic for legend: only add to first subplot if available
                # If data_main was empty, we need to label this.
                # Simplification: Add label to plot only if it hasn't been added.
                # But we handled label in lineplot above for ax_idx==0.
                
                do_label = (ax_idx == 0) and (data_main.empty)
                
                ax.errorbar(
                    x=data_crf0['CRF'], 
                    y=data_crf0['MAE'], 
                    yerr=data_crf0['MAE_Std'], 
                    fmt='o', 
                    color=color, 
                    ecolor=color, 
                    capsize=5, 
                    markersize=8,
                    label=source if do_label else None
                )

    # Common labels
    fig.text(0.5, 0.04, 'CRF Value', ha='center', fontsize=14)
    # y-label is tricky with subplots. Put it on the first axis or figure?
    # Figure text is easiest for centering.
    fig.text(0.04, 0.5, 'MAE', va='center', rotation='vertical', fontsize=14)
    
    # Title
    fig.suptitle('MAE Drop vs CRF Value', fontsize=16)

    # Legend
    # Since we suppressed legend in loop, we need to create a common legend.
    # But we added labels to lines in ax[0].
    # So we can just ask ax[0] for handles/labels.
    handles, labels = axes[0].get_legend_handles_labels()
    # If some sources only appear in other segments (unlikely), we might miss them.
    # Assuming all sources cover similar ranges or at least appear in first segment or are handled.
    # Actually, if a source is ONLY in segment 2, ax[0] won't have it.
    # Better to gather handles from all axes?
    # Or just rely on the fact that we iterate sources in outer loop and add to ax[0] if possible.
    # Simplest: Manually build legend from handles.
    
    # Deduplicate labels
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center right', title='Source')
    results_path = Path(args.output)
    plt.subplots_adjust(right=0.85) # Make room for legend
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path.resolve()}")

if __name__ == '__main__':
    main()
