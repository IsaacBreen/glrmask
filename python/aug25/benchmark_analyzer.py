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
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            competitor_name = Path(data["competitor_script"]).stem
            timings = data["results"]["get_mask_timings_seconds"]
            for i, timing in enumerate(timings):
                all_data.append({
                    "competitor": competitor_name,
                    "token_index": i,
                    "time_sec": timing,
                    "equivalence_passed": data["results"]["equivalence_check"]["passed"]
                })

    if not all_data:
        print("No data to analyze.")
        return

    df = pd.DataFrame(all_data)

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
    sns.lineplot(data=df, x='token_index', y='time_sec', hue='competitor', alpha=0.8)
    plt.title('get_mask() Performance per Token')
    plt.xlabel('Token Index in Sequence')
    plt.ylabel('Time (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Competitor')
    
    # Linear scale
    plt.yscale('linear')
    linear_path = output_dir / "timings_per_token_linear.png"
    plt.savefig(linear_path, dpi=300, bbox_inches='tight')
    print(f"Saved linear scale plot to {linear_path}")

    # Log scale
    plt.yscale('log')
    plt.title('get_mask() Performance per Token (Log Scale)')
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
