#!/usr/bin/env python
import sys
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def parse_log_data(log_content):
    """Parses the benchmark log content to extract all necessary data."""
    data = {}
    try:
        # Find the main output directory for the benchmark run
        output_dir_match = re.search(r"Benchmark results will be saved in: (benchmark_results/\S+)", log_content)
        if not output_dir_match:
            print("Error: Could not find the benchmark output directory in the log.", file=sys.stderr)
            return None
        data['output_dir'] = output_dir_match.group(1)

        # Isolate the detailed log for the Python model
        python_model_log = re.search(r">>> Running benchmark for: precompute3_model_pure_python_get_mask_only\.py(.*?)>>> Finished benchmark for:", log_content, re.DOTALL).group(1)

        # --- Extract GSS Stats ---
        initial_stats_blocks = re.findall(r"Initial GSS stats:\n(.*?)(?=Stats after seeding:)", python_model_log, re.DOTALL)

        data['total_stacks'] = [int(m) for m in re.findall(r"- stacks: total=(\d+)", "\n".join(initial_stats_blocks))]
        data['initial_upper_branch_nodes'] = [int(m) for m in re.findall(r"nodes: UpperBranch=(\d+)", "\n".join(initial_stats_blocks))]
        data['initial_interface_nodes'] = [int(m) for m in re.findall(r"Interface=(\d+)", "\n".join(initial_stats_blocks))]
        data['initial_lower_nodes'] = [int(m) for m in re.findall(r"Lower=(\d+)", "\n".join(initial_stats_blocks))]
        data['total_accumulator_instances'] = [int(m) for m in re.findall(r"total_accumulator_instances=(\d+)", "\n".join(initial_stats_blocks))]
        data['unique_accumulators'] = [int(m) for m in re.findall(r"unique_accumulators_count=(\d+)", "\n".join(initial_stats_blocks))]
        data['structural_sharing_factor'] = [float(m) for m in re.findall(r"structural_sharing_factor=([\d.]+)", "\n".join(initial_stats_blocks))]

        # --- Extract get_mask() Profiling Stats ---
        profiling_blocks = re.findall(r"--- get_mask\(\) profiling stats for call #\d+ ---(.*?)(?=commit \(ms\))", python_model_log, re.DOTALL)
        profiling_text = "\n".join(profiling_blocks)

        data['get_mask_total_time'] = [float(m) for m in re.findall(r"Total time:\s+([\d.]+) ms", profiling_text)]
        data['init_time'] = [float(m) for m in re.findall(r"Initialization time:\s+([\d.]+) ms", profiling_text)]
        data['main_loop_time'] = [float(m) for m in re.findall(r"Main loop time:\s+([\d.]+) ms", profiling_text)]
        data['final_conversion_time'] = [float(m) for m in re.findall(r"Final conversion:\s+([\d.]+) ms", profiling_text)]
        data['main_loop_apply_calls'] = [int(m) for m in re.findall(r"Main loop GSS.apply calls: (\d+)", profiling_text)]
        data['main_loop_intersection_calls'] = [int(m) for m in re.findall(r"Main loop Bitset.intersection calls: (\d+)", profiling_text)]
        data['main_loop_union_calls'] = [int(m) for m in re.findall(r"Main loop Bitset.union calls: (\d+)", profiling_text)]
        data['main_loop_merge_calls'] = [int(m) for m in re.findall(r"Main loop GSS.merge calls: (\d+)", profiling_text)]

        # --- Extract Detailed Bitset Operation Counts ---
        data['bitset_union_calls'] = [int(m) for m in re.findall(r"bitset_union calls: (\d+)", profiling_text)]
        data['bitset_intersection_calls'] = [int(m) for m in re.findall(r"bitset_intersection calls: (\d+)", profiling_text)]
        data['bitset_difference_calls'] = [int(m) for m in re.findall(r"bitset_difference calls: (\d+)", profiling_text)]
        data['hybrid_complement_calls'] = [int(m) for m in re.findall(r"hybrid_complement calls: (\d+)", profiling_text)]
        data['acc_merge_calls'] = [int(m) for m in re.findall(r"acc_merge calls: (\d+)", profiling_text)]

        # --- Extract Commit Times ---
        data['commit_time'] = [float(m) for m in re.findall(r"commit \(ms\): ([\d.]+)", python_model_log)]

        # --- Final Check and Data Validation ---
        num_steps = len(data['get_mask_total_time'])
        if num_steps == 0:
            print("Error: No profiling steps found for the Python model.", file=sys.stderr)
            return None
        data['steps'] = list(range(1, num_steps + 1))

        # Handle optional metrics that might not be in older logs
        if len(data['acc_merge_calls']) == 0 and num_steps > 0:
            print("Info: 'acc_merge calls' not found in log. Assuming zero for all steps.", file=sys.stderr)
            data['acc_merge_calls'] = [0] * num_steps

        print(f"Successfully parsed {num_steps} steps.", file=sys.stderr)
        return data

    except Exception as e:
        print(f"An error occurred during parsing: {e}", file=sys.stderr)
        return None

def generate_plots(data):
    """Generates and saves all the plots."""
    plot_dir = os.path.join(data['output_dir'], "custom_plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving custom plots to: {plot_dir}", file=sys.stderr)

    steps = data['steps']

    # Plot 1: GSS Node and Accumulator Counts
    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['initial_interface_nodes'], 'o-', label='Interface Nodes')
    plt.plot(steps, data['initial_lower_nodes'], 'x-', label='Lower Nodes')
    plt.plot(steps, data['total_accumulator_instances'], 's-', label='Total Accumulator Instances')
    plt.plot(steps, data['unique_accumulators'], '^-', label='Unique Accumulators')
    plt.xlabel('Benchmark Step'); plt.ylabel('Count'); plt.title('GSS Node and Accumulator Counts (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gss_node_counts.png"))
    plt.close()

    # Plot 2: UpperBranch Node Counts
    plt.figure(figsize=(12, 8))
    plt.plot(steps, np.array(data['initial_upper_branch_nodes']) + 0.1, 'o-', label='UpperBranch Nodes')
    plt.xlabel('Benchmark Step'); plt.ylabel('Count (0 plotted as 0.1)'); plt.title('UpperBranch Node Counts (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.yticks([0.1, 1.1], ['0', '1']); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "gss_upper_branch_counts.png"))
    plt.close()

    # Plot 3: Main Loop Apply and Intersection Calls
    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['main_loop_apply_calls'], 'o-', label='Main Loop GSS.apply Calls')
    plt.plot(steps, data['main_loop_intersection_calls'], 'x-', label='Main Loop Bitset.intersection Calls')
    plt.plot(steps, data['bitset_intersection_calls'], 's--', alpha=0.7, label='Total bitset_intersection Calls')
    plt.xlabel('Benchmark Step'); plt.ylabel('Number of Calls'); plt.title('Main Loop Apply & Intersection Calls (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "main_loop_apply_intersection.png"))
    plt.close()

    # Plot 4: Main Loop Union and Merge Calls (UPDATED)
    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['main_loop_union_calls'], 'o-', label='Main Loop Bitset.union Calls')
    plt.plot(steps, data['main_loop_merge_calls'], 'x-', label='Main Loop GSS.merge Calls')
    plt.plot(steps, data['bitset_union_calls'], 's--', alpha=0.7, label='Total bitset_union Calls')
    plt.plot(steps, data['acc_merge_calls'], 'd-.', alpha=0.8, label='acc_merge Calls')
    plt.xlabel('Benchmark Step'); plt.ylabel('Number of Calls'); plt.title('Main Loop Union & Merge Calls (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "main_loop_union_merge.png"))
    plt.close()

    # Plot 5: New plot for other bitset operations
    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['bitset_difference_calls'], 'o-', label='bitset_difference Calls')
    plt.plot(steps, data['hybrid_complement_calls'], 'x-', label='hybrid_complement Calls')
    plt.xlabel('Benchmark Step'); plt.ylabel('Number of Calls'); plt.title('Bitset Difference and Hybrid Complement Calls')
    plt.xticks(steps); plt.ylim(-0.1, 1); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "other_bitset_calls.png"))
    plt.close()

    # Plot 6: Exponential Growth Plots
    plt.figure(figsize=(10, 6))
    plt.plot(steps, data['get_mask_total_time'], 'o-')
    plt.xlabel('Benchmark Step'); plt.ylabel('Time (ms)'); plt.title('Exponential Growth in get_mask() Total Time (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp_growth_get_mask_time.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, data['total_stacks'], 's-', color='green')
    plt.xlabel('Benchmark Step'); plt.ylabel('Count'); plt.title('Exponential Growth in Total GSS Stacks (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp_growth_total_stacks.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, data['bitset_union_calls'], '^-', color='red')
    plt.xlabel('Benchmark Step'); plt.ylabel('Number of Calls'); plt.title('Exponential Growth in bitset_union Calls (Log Scale)')
    plt.yscale('log'); plt.xticks(steps); plt.grid(True, which="both", ls="--"); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp_growth_bitset_unions.png"))
    plt.close()

    # Plot 7: Time Decomposition
    plt.figure(figsize=(12, 8))
    plt.stackplot(steps, data['init_time'], data['main_loop_time'], data['final_conversion_time'],
                  labels=['Initialization', 'Main Loop', 'Final Conversion'], colors=['#4c72b0', '#dd8452', '#55a868'])
    plt.xlabel('Benchmark Step'); plt.ylabel('Time (ms)'); plt.title('Decomposition of get_mask() Execution Time')
    plt.legend(loc='upper left'); plt.xticks(steps); plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "time_decomposition.png"))
    plt.close()

    # Plot 8: Performance vs. Complexity
    plt.figure(figsize=(10, 6))
    total_stacks_np = np.array(data['total_stacks'])
    get_mask_time_np = np.array(data['get_mask_total_time'])
    plt.scatter(total_stacks_np, get_mask_time_np, label='Data Points')
    log_x = np.log(total_stacks_np).reshape(-1, 1); log_y = np.log(get_mask_time_np)
    model = LinearRegression().fit(log_x, log_y)
    y_pred = np.exp(model.predict(log_x))
    plt.plot(total_stacks_np, y_pred, color='red', linestyle='--', label=f'Trend Line (R²={model.score(log_x, log_y):.3f})')
    plt.xlabel('Total Stacks in GSS'); plt.ylabel('get_mask() Total Time (ms)'); plt.title('Performance vs. GSS Complexity (Log-Log Scale)')
    plt.xscale('log'); plt.yscale('log'); plt.grid(True, which="both", ls="--"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "performance_vs_complexity.png"))
    plt.close()

    # Plot 9: Efficiency and Sharing
    fig, ax1 = plt.subplots(figsize=(12, 8))
    unions_per_stack = np.array(data['bitset_union_calls']) / total_stacks_np
    ax1.plot(steps, unions_per_stack, 'o-', color='tab:blue', label='Bitset Unions per Stack')
    ax1.set_xlabel('Benchmark Step'); ax1.set_ylabel('Bitset Union Calls per Stack', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(steps, data['structural_sharing_factor'], 's--', color='tab:green', label='Structural Sharing Factor')
    ax2.set_ylabel('Structural Sharing Factor', color='tab:green'); ax2.tick_params(axis='y', labelcolor='tab:green'); ax2.set_ylim(bottom=0)
    plt.title('Algorithmic Efficiency and GSS Sharing Factor'); plt.xticks(steps); fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left'); plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.savefig(os.path.join(plot_dir, "efficiency_and_sharing.png"))
    plt.close()

    print("All plots generated successfully.", file=sys.stderr)

if __name__ == "__main__":
    log_content = sys.stdin.read()
    parsed_data = parse_log_data(log_content)
    if parsed_data:
        generate_plots(parsed_data)
    else:
        print("Failed to parse log data. No plots were generated.", file=sys.stderr)
        sys.exit(1)