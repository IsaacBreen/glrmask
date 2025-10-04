import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when running as a script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import gzip
import copy
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from python.aug25.benchmark_runner import load_model_class, greedy_tokenizer
from python.aug25.stats import Stats


def find_expensive_steps(model, token_ids, args):
    """
    Run the model through the token sequence multiple times to identify the
    most expensive get_mask calls based on the model's cost metric.
    """
    print(f"--- Phase 1: Finding up to {args.num_steps_to_find} most expensive step(s) ---")
    print(f"Running {args.find_steps_repeat} repetitions to gather costs...")

    costs_per_step = defaultdict(list)
    initial_model_state = model.state

    for i in range(args.find_steps_repeat):
        model.state = copy.deepcopy(initial_model_state) # Reset state for each run
        progress_bar = tqdm(
            enumerate(token_ids),
            total=len(token_ids),
            desc=f"Finding steps (run {i+1}/{args.find_steps_repeat})",
            leave=False
        )
        for step_idx, token_id in progress_bar:
            Stats.get().reset()
            model.get_mask()
            costs_per_step[step_idx].append(model.last_get_mask_cost)
            model.commit(token_id)

    # Aggregate costs
    agg_costs = {}
    agg_func = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'median': np.median,
    }.get(args.agg_method)

    if not agg_func:
        raise ValueError(f"Invalid aggregation method: {args.agg_method}")

    for step, costs in costs_per_step.items():
        if costs:
            agg_costs[step] = agg_func(costs)

    # Find top N steps
    sorted_steps = sorted(agg_costs.items(), key=lambda item: item[1], reverse=True)
    top_steps = [
        {"step_index": step, "cost": cost}
        for step, cost in sorted_steps[:args.num_steps_to_find]
    ]

    print("\nMost expensive steps found:")
    for item in top_steps:
        print(f"  - Step {item['step_index']:<5}: Aggregated cost = {item['cost']:,.2f} (using '{args.agg_method}')")
    print("-" * 20)

    return top_steps


def benchmark_variations_at_step(initial_model, token_ids, step_info, args):
    """
    For a given expensive step, advance the model to that state and then
    benchmark all defined variations.
    """
    step_index = step_info['step_index']
    print(f"\n--- Phase 2: Benchmarking variations for step {step_index} ---")

    # 1. Get model to the state *before* the expensive get_mask call
    print(f"Advancing model to step {step_index}...")
    model_at_step = copy.deepcopy(initial_model)
    for i in tqdm(range(step_index), desc="Committing tokens"):
        model_at_step.commit(token_ids[i])

    # 2. Get variations and benchmark configuration from the model
    variations = model_at_step.get_optimization_variations()
    config = model_at_step.get_benchmark_config()
    stats_to_collect = config['stats_to_collect']
    print_report = config['print_report']

    print(f"Found {len(variations)} variations to test.")

    # 3. Run benchmarks for each variation
    for var in variations:
        all_run_stats = defaultdict(list)

        # Apply variation to a deep copy to ensure isolation
        model_variant = copy.deepcopy(model_at_step)
        var.apply(model_variant) # This should print variation details

        # Run the benchmark multiple times
        for i in range(args.benchmark_repeat):
            Stats.get().reset()
            model_variant.get_mask()

            # Collect specified stats
            stats = Stats.get()
            for key in stats_to_collect:
                if key in stats.times:
                    all_run_stats[key].append(stats.times[key])
                elif key in stats.counts:
                    all_run_stats[key].append(float(stats.counts[key]))

        # 4. Report aggregated results for this variation
        print_report(str(var), all_run_stats)


def main():
    parser = argparse.ArgumentParser(description="Find expensive steps in a model and benchmark optimization variations.")
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to the model .py file.")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to the code file to use as input.")
    parser.add_argument("-f", "--constraint-file", type=Path, required=True, help="Path to the pre-compiled .json.gz grammar constraint file.")
    parser.add_argument('--find-steps-repeat', type=int, default=2, help="Number of repetitions to find expensive steps.")
    parser.add_argument('--benchmark-repeat', type=int, default=5, help="Number of repetitions for benchmarking each variation.")
    parser.add_argument('--num-steps-to-find', type=int, default=1, help="Number of top expensive steps to analyze.")
    parser.add_argument('--agg-method', choices=['min', 'max', 'mean', 'median'], default='min', help="Aggregation method for finding expensive steps.")

    args = parser.parse_args()

    for p in [args.model, args.code, args.constraint_file]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    # --- Setup ---
    print("--- Setting up analysis environment ---")

    # 1. Load constraint file
    print(f"Loading grammar constraint from: {args.constraint_file}")
    p_str = str(args.constraint_file)
    if p_str.endswith('.gz'):
        with gzip.open(p_str, 'rt', encoding='utf-8') as f:
            constraint_json_str = f.read()
    else:
        constraint_json_str = args.constraint_file.read_text(encoding='utf-8')

    # 2. Load model
    print(f"Loading model from: {args.model}")
    ModelClass = load_model_class(args.model)
    initial_model = ModelClass.from_json_string(constraint_json_str)
    print("Model loaded.")

    # Check for required methods on the model
    required_methods = [
        'get_optimization_variations',
        'get_benchmark_config',
        'last_get_mask_cost'
    ]
    for method in required_methods:
        if not hasattr(initial_model, method):
            parser.error(f"Model class in {args.model} must have a '{method}' attribute/method for optimization analysis.")

    # 3. Tokenize input code
    print(f"Loading and tokenizing code from: {args.code}")
    constraint_json = json.loads(constraint_json_str)
    llm_token_map = constraint_json['llm_token_map']
    id_to_token = {token_id: bytes(token_bytes) for token_bytes, token_id in llm_token_map}
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Tokenized into {len(token_ids)} tokens.")
    print("-" * 20)

    # --- Run Analysis ---

    # Phase 1: Find the most expensive steps
    expensive_steps = find_expensive_steps(initial_model, token_ids, args)

    if not expensive_steps:
        print("No expensive steps found or no costs were recorded. Exiting.")
        return

    # Phase 2: Benchmark variations for each expensive step
    for step_info in expensive_steps:
        benchmark_variations_at_step(initial_model, token_ids, step_info, args)

    print("\n--- Optimization analysis finished ---")


if __name__ == "__main__":
    main()
