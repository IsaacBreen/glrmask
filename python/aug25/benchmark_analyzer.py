import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _normalize_intervals(ranges: Optional[List[List[int]]]) -> Tuple[Tuple[int, int], ...]:
    """
    Normalize a list of [start, end] intervals into a sorted, merged, disjoint tuple of pairs.
    """
    if not ranges:
        return tuple()
    items = sorted((int(s), int(e)) for s, e in ranges)
    merged: List[Tuple[int, int]] = []
    cs, ce = items[0]
    for ns, ne in items[1:]:
        if ns <= ce + 1:
            ce = max(ce, ne)
        else:
            merged.append((cs, ce))
            cs, ce = ns, ne
    merged.append((cs, ce))
    return tuple(merged)


def analyze_results(result_files: List[Path], output_dir: Path, baseline_key: Optional[str] = None):
    """
    Loads benchmark results from JSON files, computes statistics, compares masks against a chosen baseline,
    and generates plots.
    """
    all_data_rows = []
    commit_timings_by_model: Dict[str, List[float]] = {}
    masks_by_model: Dict[str, List[Tuple[Tuple[int, int], ...]]] = {}
    get_mask_timings_by_model: Dict[str, List[float]] = {}

    model_order: List[str] = []

    # Load all results
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        model_script = data.get("model_script") or data.get("competitor_script")  # legacy fallback
        model_name = Path(model_script).stem if model_script else Path(file_path).stem

        if model_name not in model_order:
            model_order.append(model_name)

        timings = data["results"].get("get_mask_timings_seconds", [])
        get_mask_timings_by_model[model_name] = timings

        commit_timings = data["results"].get("commit_timings_seconds", [])
        commit_timings_by_model[model_name] = commit_timings

        masks_raw = data["results"].get("masks_ranges") or data["results"].get("masks_intervals")
        if masks_raw is None:
            print(f"Warning: No masks present in {file_path}. Mask comparisons will be skipped for {model_name}.")
            masks_by_model[model_name] = []
        else:
            masks_by_model[model_name] = [_normalize_intervals(r) for r in masks_raw]

    if not get_mask_timings_by_model:
        print("No data to analyze.")
        return

    # Determine baseline
    if baseline_key:
        # Allow either a model name (stem) or a path to a results file
        candidate = baseline_key
        path_candidate = Path(candidate)
        if path_candidate.exists():
            try:
                with open(path_candidate, 'r') as f:
                    d = json.load(f)
                candidate_name = Path(d.get("model_script", path_candidate)).stem
            except Exception:
                candidate_name = path_candidate.stem
        else:
            candidate_name = candidate
        if candidate_name not in masks_by_model:
            print(f"Warning: Baseline '{baseline_key}' not found among models: {list(masks_by_model.keys())}. Using first available model.")
            baseline_name = model_order[0]
        else:
            baseline_name = candidate_name
    else:
        baseline_name = model_order[0]

    baseline_masks = masks_by_model.get(baseline_name, [])
    baseline_timings = get_mask_timings_by_model.get(baseline_name, [])
    print(f"Selected baseline: {baseline_name}")

    # Compute per-model mismatch indices against the baseline
    mismatch_indices_by_model: Dict[str, List[int]] = {}
    equivalent_by_model: Dict[str, bool] = {}

    have_masks = all(len(v) > 0 for v in masks_by_model.values())

    for model_name, masks in masks_by_model.items():
        if not have_masks or not baseline_masks or not masks:
            mismatch_indices_by_model[model_name] = []
            equivalent_by_model[model_name] = True if model_name == baseline_name else False
            continue
        length = min(len(baseline_masks), len(masks))
        mismatches: List[int] = []
        for i in range(length):
            if baseline_masks[i] != masks[i]:
                mismatches.append(i)
        # If lengths differ, count extra indices as mismatches (conservative)
        if len(baseline_masks) != len(masks):
            extra_mismatches = list(range(length, max(len(baseline_masks), len(masks))))
            mismatches.extend(extra_mismatches)
        mismatch_indices_by_model[model_name] = mismatches
        equivalent_by_model[model_name] = (len(mismatches) == 0)

    # Build a unified dataframe for timings and mismatch flags
    for model_name, timings in get_mask_timings_by_model.items():
        mismatches_set = set(mismatch_indices_by_model.get(model_name, []))
        for i, t in enumerate(timings):
            all_data_rows.append({
                "model": model_name,
                "token_index": i,
                "time_sec": t,
                "mask_mismatch": (i in mismatches_set)
            })

    if not all_data_rows:
        print("No timing rows to analyze.")
        return

    df = pd.DataFrame(all_data_rows)

    # Create commit DataFrame (use baseline's commit timings if available, otherwise the first present)
    commit_timings_raw: List[float] = []
    if baseline_name in commit_timings_by_model and commit_timings_by_model[baseline_name]:
        commit_timings_raw = commit_timings_by_model[baseline_name]
    else:
        # fallback to any available
        for v in commit_timings_by_model.values():
            if v:
                commit_timings_raw = v
                break

    df_commit = pd.DataFrame()
    if commit_timings_raw:
        df_commit = pd.DataFrame({
            "token_index": range(len(commit_timings_raw)),
            "time_sec": commit_timings_raw
        })

    # --- Print Summary Statistics ---
    print("--- Benchmark Summary ---")
    summary = df.groupby('model')['time_sec'].agg(
        ['mean', 'std', 'min', 'median', 'max', 'count']
    ).rename(columns={'median': 'p50'})

    # Add percentiles
    percentiles = df.groupby('model')['time_sec'].quantile([0.90, 0.99]).unstack(level=1)
    percentiles.columns = [f'p{int(c*100)}' for c in percentiles.columns]
    summary = summary.join(percentiles)

    # Add equivalence info vs baseline
    eq_series = pd.Series({k: ('✅' if v else '❌') for k, v in equivalent_by_model.items()}, name='equivalent')
    # Add mismatch counts
    mm_series = pd.Series({k: len(v) for k, v in mismatch_indices_by_model.items()}, name='mask_mismatch_count')

    summary = summary.join(eq_series).join(mm_series)

    # Reorder columns for display
    summary = summary[['equivalent', 'mask_mismatch_count', 'count', 'mean', 'std', 'min', 'p50', 'p90', 'p99', 'max']]

    # Format for printing
    summary[['mean', 'std', 'min', 'p50', 'p90', 'p99', 'max']] *= 1000  # convert to ms
    summary = summary.rename(columns=lambda c: c + ' (ms)' if c not in ['equivalent', 'mask_mismatch_count', 'count'] else c)

    print(summary.to_string(float_format="%.4f"))
    print(f"\nBaseline: {baseline_name}")
    print("✅ = Masks identical to baseline across all steps, ❌ = At least one mask mismatch")

    # --- Generate Plots ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}...")

    # 1. Line plot of timings per token
    plt.figure(figsize=(15, 8))
    ax = sns.lineplot(
        data=df, x='token_index', y='time_sec', hue='model', style='model',
        markers=True, dashes=True, alpha=0.7, linewidth=1.0
    )

    mismatch_df = df[df['mask_mismatch']]
    plot_title = f'get_mask() Performance per Token (Baseline: {baseline_name})'

    if not mismatch_df.empty:
        plot_title += ' (X marks mask mismatches)'

        # Get hue order from the lineplot's legend to ensure colors match
        handles, labels = ax.get_legend_handles_labels()
        # The first entry is the title of the legend, so we skip it.
        model_labels = labels[1:1+len(df['model'].unique())]

        sns.scatterplot(
            data=mismatch_df,
            x='token_index',
            y='time_sec',
            hue='model',
            hue_order=model_labels,
            style='mask_mismatch',
            markers=['X'],
            s=150,
            edgecolor='black',
            linewidth=1,
            legend=False,
            ax=ax,
            zorder=5  # Ensure markers are on top of lines
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
    sns.boxplot(data=df_ms, x='model', y='time_ms')
    plt.title('Distribution of get_mask() Timings')
    plt.xlabel('Model')
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
    parser.add_argument(
        "-b", "--baseline",
        default=None,
        help="Baseline model name (stem) or a path to a results JSON file. Defaults to the first model found."
    )
    args = parser.parse_args()

    result_files: List[Path] = []
    for path_str in args.result_paths:
        path = Path(path_str)
        if path.is_dir():
            result_files.extend(path.glob("*.json"))
        elif path.is_file() and path.suffix == '.json':
            result_files.append(path)

    if not result_files:
        print(f"Error: No .json files found in the specified paths.")
        return

    analyze_results(sorted(list(set(result_files))), Path(args.output_dir), baseline_key=args.baseline)


if __name__ == "__main__":
    main()
