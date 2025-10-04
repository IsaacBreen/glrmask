import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when running as a script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import gzip
import json
import os
import time
from datetime import datetime, timezone
from statistics import mean, median

import numpy as np

from python.aug25.stats import Stats
from python.aug25.models.precompute3_model_pure_python_opt3 import Model
from python.aug25.models.precompute3_model_pure_python_opt3 import VariationBase
from python.aug25.models.precompute3_model_pure_python_opt3 import ArenaOptimizeParams, TraversalBudgetVariation, ArenaReorderVariation

# Reuse tokenizer and loader helpers from benchmark_runner
from python.aug25.benchmark_runner import greedy_tokenizer, load_model_class


def load_constraint_text(path: Path) -> str:
    p = str(path)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return f.read()
    return path.read_text(encoding='utf-8')


def build_id_to_token(constraint_json_str: str) -> dict[int, bytes]:
    constraint_json = json.loads(constraint_json_str)
    llm_token_map = constraint_json['llm_token_map']
    id_to_token: dict[int, bytes] = {}
    for token_bytes, token_id in llm_token_map:
        id_to_token[token_id] = bytes(token_bytes)
    return id_to_token


def aggregate(values, method: str):
    if not values:
        return 0.0
    if method == "min":
        return float(min(values))
    if method == "max":
        return float(max(values))
    if method == "mean":
        return float(mean(values))
    if method == "median":
        return float(median(values))
    # Fallback: mean
    return float(mean(values))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def format_human(n: float) -> str:
    return f"{n:,.3f}"


def coordinator_main():
    parser = argparse.ArgumentParser(description="Optimization coordinator: find hot get_mask steps and evaluate variations.")
    parser.add_argument("-f", "--constraint-file", type=Path, required=True, help="Path to the pre-compiled .json.gz grammar constraint file.")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to the code file to use as input.")
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to the model .py file.")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for JSON results.")
    parser.add_argument("--detect-repeat", type=int, default=3, help="Number of full input passes for hot-step detection (commit+get_mask).")
    parser.add_argument("--eval-repeat", type=int, default=10, help="Number of repeated get_mask measurements at the selected hot step.")
    parser.add_argument("--agg-method", type=str, default="max", choices=["min", "mean", "median", "max"], help="Aggregation method for detection and evaluation series.")
    parser.add_argument("--metric", type=str, default="edges_traversed", choices=["edges_traversed", "main_loop_ms"], help="Metric to optimize (and to aggregate).")
    parser.add_argument("--hot-steps", type=int, default=1, help="Number of hot steps to select (currently evaluation uses the first).")
    args = parser.parse_args()

    for p in [args.constraint_file, args.code, args.model]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    # Output directory
    if args.output:
        out_dir = args.output
    else:
        out_dir = Path(f"variations_results/{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}")
    ensure_dir(out_dir)

    # Load constraint and model
    constraint_json_str = load_constraint_text(args.constraint_file)
    id_to_token = build_id_to_token(constraint_json_str)
    ModelClass = load_model_class(args.model)
    model: Model = ModelClass.from_json_string(constraint_json_str)
    model.suppress_stats_report = True

    # Tokenize input code
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Input tokens: {len(token_ids)}")
    print(f"Detection repeats: {args.detect_repeat}, Aggregation: {args.agg_method}, Metric: {args.metric}")

    # Hot step detection (initialize model once; only reset state between repeats)
    values_per_step = [[] for _ in range(len(token_ids))]
    for r in range(args.detect_repeat):
        print(f"--- Detection pass {r+1}/{args.detect_repeat} ---")
        model.reset_state()
        for i, tok in enumerate(token_ids):
            # Measure and store metric for this step
            model.get_mask()
            m = model.get_last_get_mask_metrics()
            values_per_step[i].append(float(m.get(args.metric, 0.0)))
            # Advance state
            model.commit(tok)

    aggregated = [aggregate(vals, args.agg_method) for vals in values_per_step]
    # Let the model decide which steps are hot (default: top-k by value)
    hot_steps = model.select_hot_steps(aggregated, k=max(1, args.hot_steps))

    print("\n=== Hot step detection summary ===")
    for i in range(min(5, len(aggregated))):
        print(f"  Step {i:>4} agg={format_human(aggregated[i])}")
    print("  ...")
    print("  Top hot steps (by aggregated metric):", hot_steps)

    if not hot_steps:
        print("No hot steps detected; exiting.")
        return
    chosen_step = hot_steps[0]
    print(f"\n>>> Chosen hot step: {chosen_step} (aggregated {args.metric}={format_human(aggregated[chosen_step])})")

    # Prepare variants (baseline + model-provided variations)
    variations: list[VariationBase] = []
    # Baseline: keep current traversal + arena
    baseline_variation = TraversalBudgetVariation(name="baseline", max_edges=model.gm_max_edges, max_dests=model.gm_max_dests)
    variations.append(baseline_variation)
    # Model suggests additional variations
    try:
        variations.extend(model.default_variations())
    except Exception as e:
        print(f"Warning: model.default_variations() failed: {e}")

    print("\n=== Evaluating variations at chosen step ===")
    results = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "grammar_file": str(args.constraint_file),
            "code_file": str(args.code),
            "model_script": str(args.model),
            "detect_repeat": args.detect_repeat,
            "eval_repeat": args.eval_repeat,
            "agg_method": args.agg_method,
            "metric": args.metric,
            "chosen_step": chosen_step,
        },
        "variations": [],
        "hot_steps_agg": aggregated,
        "selected_hot_steps": hot_steps,
    }

    def run_eval_for_variation(var: VariationBase):
        print("\n----------------------------------------")
        print(f"[Coordinator] Variation: {var.name}")
        # New instance sharing structures
        vm: Model = model.clone_sharing_structure()
        # Apply variation once
        var(vm)
        # Reach the chosen state by committing up to the chosen step
        vm.reset_state()
        for i in range(chosen_step):
            vm.commit(token_ids[i])
        # Evaluate get_mask multiple times in that same state
        eval_values = []
        for r in range(args.eval_repeat):
            vm.get_mask()
            metric_value = float(vm.get_last_get_mask_metrics().get(args.metric, 0.0))
            eval_values.append(metric_value)
        agg_value = aggregate(eval_values, args.agg_method)
        print(f"[Coordinator] Variation '{var.name}' aggregated {args.metric}={format_human(agg_value)} over {args.eval_repeat} runs")
        return {
            "name": var.name,
            "metric": args.metric,
            "agg_method": args.agg_method,
            "eval_values": eval_values,
            "aggregated": agg_value,
        }

    for var in variations:
        try:
            res = run_eval_for_variation(var)
            results["variations"].append(res)
        except Exception as e:
            print(f"[Coordinator] Variation '{var.name}' failed with error: {e}")
            results["variations"].append({
                "name": var.name,
                "error": str(e),
            })

    # Save results
    out_path = out_dir / "variations_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Optimization evaluation complete ===")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    coordinator_main()
