import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import warnings

import pandas as pd


# Suppress noisy FutureWarning from seaborn/pandas
warnings.filterwarnings('ignore', category=FutureWarning)

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


def analyze_results(result_files: List[Path], output_dir: Path, baseline_key: Optional[str] = None, agg_method: Optional[str] = None):
    """
    Loads benchmark results from JSON files, computes statistics, compares masks against a chosen baseline,
    and generates plots.
    """
    all_data_rows = []
    commit_timings_by_model: Dict[str, List[List[float]]] = {}
    masks_by_model: Dict[str, List[List[Tuple[Tuple[int, int], ...]]]] = {}
    get_mask_timings_by_model: Dict[str, List[List[float]]] = {}

    model_order: List[str] = []

    # Load all results
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        model_script = data.get("model_script") or data.get("competitor_script")  # legacy fallback
        model_name = Path(model_script).stem if model_script else Path(file_path).stem

        if model_name not in model_order:
            model_order.append(model_name)

        # Initialize if first time seeing model
        if model_name not in get_mask_timings_by_model:
            get_mask_timings_by_model[model_name] = []
            commit_timings_by_model[model_name] = []
            masks_by_model[model_name] = []

        timings = data["results"].get("get_mask_timings_seconds", [])
        get_mask_timings_by_model[model_name].append(timings)

        commit_timings = data["results"].get("commit_timings_seconds", [])
        commit_timings_by_model[model_name].append(commit_timings)

        masks_raw = data["results"].get("masks_ranges") or data["results"].get("masks_intervals")
        if masks_raw is None:
            print(f"Warning: No masks present in {file_path}. Mask comparisons will be skipped for {model_name}.")
            masks_by_model[model_name].append([])
        else:
            masks_by_model[model_name].append([_normalize_intervals(r) for r in masks_raw])

    if not get_mask_timings_by_model:
        print("No data to analyze.")
        return

    # --- Process repeated runs ---
    final_get_mask_timings: Dict[str, List[float]] = {}
    final_commit_timings: Dict[str, List[float]] = {}
    final_masks_by_model: Dict[str, List[Tuple[Tuple[int, int], ...]]] = {}
    final_model_order: List[str] = []

    if agg_method:
        print(f"--- Aggregating results from multiple runs using '{agg_method}' ---")
        import numpy as np
        for model_name in model_order:
            # Aggregate get_mask timings
            runs = get_mask_timings_by_model.get(model_name, [])
            if runs and any(r for r in runs):
                max_len = max(len(r) for r in runs if r)
                padded_runs = [r + ([np.nan] * (max_len - len(r))) for r in runs]
                df_runs = pd.DataFrame(padded_runs).T
                final_get_mask_timings[model_name] = df_runs.agg(agg_method, axis=1).dropna().tolist()
            else:
                final_get_mask_timings[model_name] = []

            # Aggregate commit timings
            runs_commit = commit_timings_by_model.get(model_name, [])
            if runs_commit and any(r for r in runs_commit):
                max_len_commit = max(len(r) for r in runs_commit if r)
                padded_runs_commit = [r + ([np.nan] * (max_len_commit - len(r))) for r in runs_commit]
                df_runs_commit = pd.DataFrame(padded_runs_commit).T
                final_commit_timings[model_name] = df_runs_commit.agg(agg_method, axis=1).dropna().tolist()
            else:
                final_commit_timings[model_name] = []

            # For masks, use the first run
            mask_runs = masks_by_model.get(model_name, [])
            if mask_runs:
                final_masks_by_model[model_name] = mask_runs[0]
                if len(mask_runs) > 1:
                    print(f"Info: For model '{model_name}', using masks from the first of {len(mask_runs)} runs for equivalence checks.")
            else:
                final_masks_by_model[model_name] = []
            final_model_order.append(model_name)
    else:  # No aggregation, unpack runs
        for model_name in model_order:
            num_runs = len(get_mask_timings_by_model.get(model_name, []))
            if num_runs > 1:
                for i in range(num_runs):
                    run_name = f"{model_name}_run{i+1}"
                    final_get_mask_timings[run_name] = get_mask_timings_by_model[model_name][i]
                    final_commit_timings[run_name] = commit_timings_by_model[model_name][i]
                    final_masks_by_model[run_name] = masks_by_model[model_name][i]
                    final_model_order.append(run_name)
            elif num_runs == 1:
                final_get_mask_timings[model_name] = get_mask_timings_by_model[model_name][0]
                final_commit_timings[model_name] = commit_timings_by_model[model_name][0]
                final_masks_by_model[model_name] = masks_by_model[model_name][0]
                final_model_order.append(model_name)

    # Replace original data structures with the processed ones
    get_mask_timings_by_model = final_get_mask_timings
    commit_timings_by_model = final_commit_timings
    masks_by_model = final_masks_by_model
    model_order = final_model_order

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
                print(f"Mask mismatch at token index {i} for model {model_name}")
                print(f"Baseline: {baseline_masks[i]}")
                print(f"Current:  {masks[i]}")
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

    # Create commit DataFrame
    all_commit_rows = []
    for model_name, timings in commit_timings_by_model.items():
        for i, t in enumerate(timings):
            all_commit_rows.append({
                "model": model_name,
                "token_index": i,
                "time_sec": t,
            })
    df_commit = pd.DataFrame(all_commit_rows)

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
    sys.stdout.flush()

    # --- Defer slow imports until after summary is printed ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import to_rgba
    import colorsys

    # --- Generate Plots ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}...")

    # 1. Line plot of timings per token
    plt.figure(figsize=(15, 8))
    ax = sns.lineplot(data=df, x='token_index', y='time_sec', hue='model', alpha=0.7, linewidth=0.5)

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
        ax_commit = sns.lineplot(data=df_commit, x='token_index', y='time_sec', hue='model', alpha=0.7, linewidth=0.5)

        ax_commit.set_xlabel('Token Index in Sequence')
        ax_commit.set_ylabel('Time (seconds)')
        ax_commit.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Linear scale
        ax_commit.set_yscale('linear')
        ax_commit.set_title('commit() Performance per Token by Model')
        commit_linear_path = output_dir / "commit_timings_per_token_linear.png"
        plt.savefig(commit_linear_path, dpi=300, bbox_inches='tight')
        print(f"Saved commit linear scale plot to {commit_linear_path}")

        # Log scale
        ax_commit.set_yscale('log')
        ax_commit.set_title('commit() Performance per Token by Model (Log Scale)')
        commit_log_path = output_dir / "commit_timings_per_token_log.png"
        plt.savefig(commit_log_path, dpi=300, bbox_inches='tight')
        print(f"Saved commit log scale plot to {commit_log_path}")
        plt.close()

    # 4. Combined get_mask and commit timings per token
    if not df.empty and not df_commit.empty:
        df_get_mask = df.copy()
        df_get_mask['operation'] = 'get_mask'
        df_commit_copy = df_commit.copy()
        df_commit_copy['operation'] = 'commit'

        df_combined = pd.concat([
            df_get_mask[['model', 'token_index', 'time_sec', 'operation']],
            df_commit_copy[['model', 'token_index', 'time_sec', 'operation']]
        ], ignore_index=True)

        def adjust_lightness(color, amount=0.7):
            try:
                c = colorsys.rgb_to_hls(*to_rgba(color)[:3])
                return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
            except Exception:
                return color

        df_combined['hue_key'] = df_combined['model'] + " | " + df_combined['operation']
        unique_models = model_order
        base_colors = sns.color_palette(n_colors=len(unique_models))

        custom_palette = {}
        hue_order_list = []
        for i, model in enumerate(unique_models):
            base_color = base_colors[i]
            get_mask_key = f"{model} | get_mask"
            commit_key = f"{model} | commit"
            hue_order_list.extend([get_mask_key, commit_key])
            custom_palette[get_mask_key] = base_color
            custom_palette[commit_key] = adjust_lightness(base_color)

        plt.figure(figsize=(15, 8))
        ax_combined = sns.lineplot(
            data=df_combined,
            x='token_index',
            y='time_sec',
            hue='hue_key',
            hue_order=hue_order_list,
            style='operation',
            style_order=['get_mask', 'commit'],
            palette=custom_palette,
            linewidth=0.5,
            alpha=0.7
        )
        ax_combined.set_xlabel('Token Index in Sequence')
        ax_combined.set_ylabel('Time (seconds)')
        ax_combined.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Linear scale
        ax_combined.set_yscale('linear')
        ax_combined.set_title('get_mask() (solid) vs commit() (dashed) Performance')
        combined_linear_path = output_dir / "combined_timings_per_token_linear.png"
        plt.savefig(combined_linear_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined linear scale plot to {combined_linear_path}")

        # Log scale
        ax_combined.set_yscale('log')
        ax_combined.set_title('get_mask() (solid) vs commit() (dashed) Performance (Log Scale)')
        combined_log_path = output_dir / "combined_timings_per_token_log.png"
        plt.savefig(combined_log_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined log scale plot to {combined_log_path}")
        plt.close()

    # 5. Stacked area plot of timings per model
    if not df.empty and not df_commit.empty:
        models = model_order # Use the determined model order for consistency
        num_models = len(models)
        if num_models > 0:
            cols = 2 if num_models > 1 else 1
            rows = (num_models + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), sharex=True, sharey=True, squeeze=False)
            axes = axes.flatten()

            for i, model_name in enumerate(models):
                ax = axes[i]
                model_df_get_mask = df[df['model'] == model_name].sort_values('token_index')
                model_df_commit = df_commit[df_commit['model'] == model_name].sort_values('token_index')

                # Align indices and fill missing values with 0, which is safe for plotting time
                merged = pd.merge(
                    model_df_get_mask[['token_index', 'time_sec']],
                    model_df_commit[['token_index', 'time_sec']],
                    on='token_index',
                    how='outer',
                    suffixes=('_get_mask', '_commit')
                ).sort_values('token_index').fillna(0)

                x = merged['token_index']
                y_get_mask = merged['time_sec_get_mask']
                y_commit = merged['time_sec_commit']

                ax.stackplot(x, y_get_mask, y_commit, labels=['get_mask', 'commit'], alpha=0.8)
                ax.set_title(f'Stacked Timings for {model_name}')
                ax.set_ylabel('Time (seconds)')
                ax.grid(True, linestyle='--', linewidth=0.5)
                ax.set_yscale('linear') # Explicitly linear as requested

                # Only show legend on the first plot to avoid clutter
                if i == 0:
                    ax.legend(loc='upper left')

            # Hide any unused subplots
            for j in range(num_models, len(axes)):
                fig.delaxes(axes[j])

            # Add a common X-axis label
            fig.text(0.5, 0.02, 'Token Index in Sequence', ha='center', va='center')
            fig.suptitle('Stacked get_mask() and commit() Performance per Token (Linear Scale)', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            stacked_area_path = output_dir / "timings_stacked_area.png"
            plt.savefig(stacked_area_path, dpi=300, bbox_inches='tight')
            print(f"Saved stacked area plot to {stacked_area_path}")
            plt.close()

    # 6. Total time (get_mask + commit) per token
    if not df.empty and not df_commit.empty:
        df_total = pd.merge(
            df[['model', 'token_index', 'time_sec']],
            df_commit[['model', 'token_index', 'time_sec']],
            on=['model', 'token_index'],
            suffixes=('_get_mask', '_commit')
        )
        df_total['time_sec'] = df_total['time_sec_get_mask'] + df_total['time_sec_commit']

        plt.figure(figsize=(15, 8))
        ax_total = sns.lineplot(data=df_total, x='token_index', y='time_sec', hue='model', alpha=0.7, linewidth=0.5)
        ax_total.set_xlabel('Token Index in Sequence')
        ax_total.set_ylabel('Total Time (seconds)')
        ax_total.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Linear scale
        ax_total.set_yscale('linear')
        ax_total.set_title('Total (get_mask + commit) Performance per Token')
        total_linear_path = output_dir / "total_timings_per_token_linear.png"
        plt.savefig(total_linear_path, dpi=300, bbox_inches='tight')
        print(f"Saved total time linear scale plot to {total_linear_path}")

        # Log scale
        ax_total.set_yscale('log')
        ax_total.set_title('Total (get_mask + commit) Performance per Token (Log Scale)')
        total_log_path = output_dir / "total_timings_per_token_log.png"
        plt.savefig(total_log_path, dpi=300, bbox_inches='tight')
        print(f"Saved total time log scale plot to {total_log_path}")
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
    parser.add_argument(
        "--agg-method",
        choices=['mean', 'median', 'min', 'max'],
        default=None,
        help="Aggregation method for repeated runs. If not set, runs are plotted individually."
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

    analyze_results(sorted(list(set(result_files))), Path(args.output_dir), baseline_key=args.baseline, agg_method=args.agg_method)


if __name__ == "__main__":
    main()
