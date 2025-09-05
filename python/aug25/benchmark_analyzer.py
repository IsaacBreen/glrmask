import argparse
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(result_files: list[Path], output_dir: Path):
    """
    Loads benchmark results from JSON files, computes statistics, and generates plots.
    """
    all_data = []
    commit_timings_raw = None
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            competitor_name = Path(data["competitor_script"]).stem
            timings = data["results"]["get_mask_timings_seconds"]
            if commit_timings_raw is None:
                commit_timings_raw = data["results"].get("commit_timings_seconds")

            mismatch_indices = set(
                data["results"]
                .get("mask_correctness_check", {})
                .get("mismatch_indices", [])
            )
            for i, timing in enumerate(timings):
                all_data.append({
                    "competitor": competitor_name,
                    "token_index": i,
                    "time_sec": timing,
                    "equivalence_passed": data["results"]["equivalence_check"]["passed"],
                    "mask_mismatch": i in mismatch_indices
                })

    if not all_data:
        print("No data to analyze.")
        return

    df = pd.DataFrame(all_data)

    # Create commit DataFrame
    df_commit = pd.DataFrame()
    if commit_timings_raw:
        df_commit = pd.DataFrame({
            "token_index": range(len(commit_timings_raw)),
            "time_sec": commit_timings_raw
        })

    # --- Print Summary Statistics ---
    print("--- Benchmark Summary ---")
    summary = df.groupby('competitor')['time_sec'].agg(
        ['mean', 'std', 'min', 'median', 'max', 'count']
    ).rename(columns={'median': 'p50'})

    # Add percentiles
    percentiles = df.groupby('competitor')['time_sec'].quantile([0.90, 0.99]).unstack(level=1)
    percentiles.columns = [f'p{int(c*100)}' for c in percentiles.columns]
    summary = summary.join(percentiles)

    # Add equivalence check info
    equiv_status = df.groupby('competitor')['equivalence_passed'].first()
    summary['equivalent'] = equiv_status.map({True: '✅', False: '❌'})

    # Reorder columns for display
    summary = summary[['equivalent', 'count', 'mean', 'std', 'min', 'p50', 'p90', 'p99', 'max']]
    
    # Format for printing
    summary[['mean', 'std', 'min', 'p50', 'p90', 'p99', 'max']] *= 1000 # convert to ms
    summary = summary.rename(columns=lambda c: c + ' (ms)' if c not in ['equivalent', 'count'] else c)

    print(summary.to_string(float_format="%.4f"))
    print("\n" + "✅ = Equivalent to reference, ❌ = Not equivalent")

    # --- Generate Plots ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}...")

    # 1. Line plot of timings per token
    plt.figure(figsize=(15, 8))
    ax = sns.lineplot(data=df, x='token_index', y='time_sec', hue='competitor', alpha=0.8, linewidth=1.2)

    mismatch_df = df[df['mask_mismatch']]
    plot_title = 'get_mask() Performance per Token'

    if not mismatch_df.empty:
        plot_title += ' (X marks mask mismatches)'
        
        # Get hue order from the lineplot's legend to ensure colors match
        handles, labels = ax.get_legend_handles_labels()
        # The first entry is the title of the legend, so we skip it.
        competitor_labels = labels[1:1+len(df['competitor'].unique())]

        sns.scatterplot(
            data=mismatch_df,
            x='token_index',
            y='time_sec',
            hue='competitor',
            hue_order=competitor_labels,
            style='mask_mismatch',
            markers=['X'],
            s=150,
            edgecolor='black',
            linewidth=1,
            legend=False,
            ax=ax,
            zorder=5 # Ensure markers are on top of lines
        )

    ax.set_xlabel('Token Index in Sequence')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Linear scale
    ax.set_yscale('linear')
    ax.set_title(plot_title)
    linear_path = output_dir / "timings_per_token_linear.png"
    plt.savefig(linear_path, dpi=300, bbox_inches='tight')
    print(f"Saved linear scale plot to {linear_path}")

    # Log scale
    ax.set_yscale('log')
    ax.set_title(plot_title + ' (Log Scale)')
    log_path = output_dir / "timings_per_token_log.png"
    plt.savefig(log_path, dpi=300, bbox_inches='tight')
    print(f"Saved log scale plot to {log_path}")
    plt.close()

    # 2. Box plot of timing distributions
    plt.figure(figsize=(12, 7))
    # Convert to ms for better readability on the plot
    df_ms = df.copy()
    df_ms['time_ms'] = df_ms['time_sec'] * 1000
    sns.boxplot(data=df_ms, x='competitor', y='time_ms')
    plt.title('Distribution of get_mask() Timings')
    plt.xlabel('Competitor')
    plt.ylabel('Time (milliseconds)')
    plt.xticks(rotation=15)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    box_path = output_dir / "timings_distribution_boxplot.png"
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {box_path}")
    plt.close()

    # 3. Line plot of commit timings per token
    if not df_commit.empty:
        plt.figure(figsize=(15, 8))
        ax_commit = sns.lineplot(data=df_commit, x='token_index', y='time_sec', color='purple', label='commit() time', linewidth=1.2)

        ax_commit.set_xlabel('Token Index in Sequence')
        ax_commit.set_ylabel('Time (seconds)')
        ax_commit.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Linear scale
        ax_commit.set_yscale('linear')
        ax_commit.set_title('commit() Performance per Token')
        commit_linear_path = output_dir / "commit_timings_per_token_linear.png"
        plt.savefig(commit_linear_path, dpi=300, bbox_inches='tight')
        print(f"Saved commit linear scale plot to {commit_linear_path}")

        # Log scale
        ax_commit.set_yscale('log')
        ax_commit.set_title('commit() Performance per Token (Log Scale)')
        commit_log_path = output_dir / "commit_timings_per_token_log.png"
        plt.savefig(commit_log_path, dpi=300, bbox_inches='tight')
        print(f"Saved commit log scale plot to {commit_log_path}")
        plt.close()




def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results for grammar constraint models.")
    parser.add_argument(
        "result_paths",
        nargs='+',
        help="Paths to benchmark JSON files or directories containing them."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="benchmark_plots",
        help="Directory to save the generated plots."
    )
    args = parser.parse_args()

    result_files = []
    for path_str in args.result_paths:
        path = Path(path_str)
        if path.is_dir():
            result_files.extend(path.glob("*.json"))
        elif path.is_file() and path.suffix == '.json':
            result_files.append(path)

    if not result_files:
        print(f"Error: No .json files found in the specified paths.")
        return

    analyze_results(sorted(list(set(result_files))), Path(args.output_dir))

if __name__ == "__main__":
    main()
