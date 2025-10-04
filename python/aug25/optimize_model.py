#!/usr/bin/env python3
"""
Model Optimization Framework

This script runs a model through input text to identify expensive get_mask() calls,
then tries different optimization variations to improve performance on those steps.

Usage:
  python -m python.aug25.optimize_model \\
    --model python/aug25/models/precompute3_model_pure_python_opt3.py \\
    --code ./src/example_code7.js \\
    --constraint-file ./.cache/test_vocabs/js_grammar_constraint.json.gz \\
    --warmup-reps 2 \\
    --test-reps 10 \\
    --num-steps 1 \\
    --agg-method min
"""

import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import gzip
import time
import importlib.util
from typing import Dict, List


def load_model_class(model_path: Path):
    """Dynamically loads the 'Model' class from a Python file."""
    module_path_for_name = model_path
    if module_path_for_name.is_absolute():
        try:
            module_path_for_name = module_path_for_name.relative_to(_project_root)
        except ValueError:
            pass

    module_name = ".".join(module_path_for_name.with_suffix('').parts)
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'Model'):
        raise AttributeError(f"Model script {model_path} must define a 'Model' class.")
    
    return getattr(module, 'Model')


def greedy_tokenizer(text_bytes, id_to_token):
    """Greedy tokenizer using a trie."""
    trie_root = {}
    for token_id, token_bytes in id_to_token.items():
        node = trie_root
        for byte_val in token_bytes:
            node = node.setdefault(byte_val, {})
        node['<ID>'] = token_id

    token_ids = []
    pos = 0
    while pos < len(text_bytes):
        node = trie_root
        longest_match_id = -1
        longest_match_len = 0
        
        for i in range(len(text_bytes) - pos):
            current_byte = text_bytes[pos + i]
            if current_byte in node:
                node = node[current_byte]
                if '<ID>' in node:
                    longest_match_id = node['<ID>']
                    longest_match_len = i + 1
            else:
                break
        
        if longest_match_len > 0:
            token_ids.append(longest_match_id)
            pos += longest_match_len
        else:
            raise ValueError(f"Failed to tokenize at position {pos}")
    return token_ids


def main():
    parser = argparse.ArgumentParser(description="Optimize model performance on expensive get_mask calls.")
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to model .py file")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to input code file")
    parser.add_argument("-f", "--constraint-file", type=Path, required=True, help="Path to constraint .json.gz file")
    parser.add_argument("--warmup-reps", type=int, default=2, help="Repetitions for warmup phase")
    parser.add_argument("--test-reps", type=int, default=10, help="Repetitions for testing each variation")
    parser.add_argument("--num-steps", type=int, default=1, help="Number of expensive steps to select")
    parser.add_argument("--agg-method", default="mean", choices=['mean', 'median', 'min', 'max'],
                        help="Aggregation method for selecting expensive steps")
    
    args = parser.parse_args()

    # Validate files
    for p in [args.model, args.code, args.constraint_file]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    print("=" * 80)
    print("MODEL OPTIMIZATION FRAMEWORK")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Code: {args.code}")
    print(f"Constraint: {args.constraint_file}")
    print(f"Warmup repetitions: {args.warmup_reps}")
    print(f"Test repetitions: {args.test_reps}")
    print(f"Number of steps to select: {args.num_steps}")
    print(f"Aggregation method: {args.agg_method}")
    print("=" * 80)

    # Load constraint and extract vocabulary
    print("\n[1/6] Loading constraint and vocabulary...")
    if str(args.constraint_file).endswith('.gz'):
        with gzip.open(args.constraint_file, 'rt', encoding='utf-8') as f:
            constraint_json_str = f.read()
    else:
        constraint_json_str = args.constraint_file.read_text(encoding='utf-8')
    
    constraint_json = json.loads(constraint_json_str)
    llm_token_map = constraint_json['llm_token_map']
    id_to_token: Dict[int, bytes] = {token_id: bytes(token_bytes) for token_bytes, token_id in llm_token_map}

    # Tokenize input
    print("[2/6] Tokenizing input...")
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"  Tokenized into {len(token_ids)} tokens.")

    # Load model class
    print("[3/6] Loading model class...")
    ModelClass = load_model_class(args.model)

    # Warmup phase: identify expensive steps
    print(f"\n[4/6] Warmup phase: running {args.warmup_reps} repetitions to identify expensive steps...")
    all_costs: List[List[float]] = []
    state_checkpoints: List[Dict] = []
    
    for rep in range(args.warmup_reps):
        print(f"  Repetition {rep + 1}/{args.warmup_reps}...")
        model = ModelClass.from_json_string(constraint_json_str)
        costs = []
        checkpoints = []
        
        for i, token_id in enumerate(token_ids):
            # Checkpoint state before get_mask
            if rep == 0:  # Only need to save checkpoints once
                checkpoints.append(model.checkpoint())
            
            # Run get_mask and track cost
            model.get_mask()
            costs.append(model.last_get_mask_cost)
            
            # Commit token
            model.commit(token_id)
        
        all_costs.append(costs)
        if rep == 0:
            state_checkpoints = checkpoints
        
        print(f"    Total cost across all steps: {sum(costs):.0f}")
    
    # Select expensive steps
    expensive_steps = ModelClass.select_expensive_steps(all_costs, args.num_steps, args.agg_method)
    print(f"\n  Selected {len(expensive_steps)} expensive step(s):")
    for step_idx in expensive_steps:
        step_costs = [costs[step_idx] for costs in all_costs]
        print(f"    Step {step_idx}: costs={step_costs}, {args.agg_method}={sum(step_costs)/len(step_costs) if args.agg_method=='mean' else min(step_costs):.0f}")

    # Get variations from model
    print("\n[5/6] Loading variations from model...")
    temp_model = ModelClass.from_json_string(constraint_json_str)
    variations = temp_model.get_variations()
    print(f"  Found {len(variations)} variation(s) to test.")
    del temp_model

    # Test each variation
    print(f"\n[6/6] Testing variations (running get_mask {args.test_reps} times per step)...")
    print("=" * 80)
    
    for var_idx, variation in enumerate(variations):
        print(f"\nVARIATION {var_idx + 1}/{len(variations)}")
        print("-" * 80)
        
        # Load fresh model and apply variation
        model = ModelClass.from_json_string(constraint_json_str)
        variation(model)
        
        # Test on each expensive step
        all_step_results = []
        for step_idx in expensive_steps:
            print(f"\n  Testing on step {step_idx}...")
            
            # Restore to state before this step
            model.restore(state_checkpoints[step_idx])
            
            # Run get_mask multiple times
            results = []
            for rep in range(args.test_reps):
                start_time = time.perf_counter()
                model.get_mask()
                elapsed = time.perf_counter() - start_time
                cost = model.last_get_mask_cost
                results.append({'time': elapsed, 'cost': cost})
            
            # Aggregate results for this step
            agg = ModelClass.aggregate_results(results)
            all_step_results.append(agg)
            
            print(f"    Time:  min={agg['time_min']*1000:.3f}ms, mean={agg['time_mean']*1000:.3f}ms, "
                  f"median={agg['time_median']*1000:.3f}ms, max={agg['time_max']*1000:.3f}ms")
            print(f"    Cost:  min={agg['cost_min']:.0f}, mean={agg['cost_mean']:.0f}, "
                  f"median={agg['cost_median']:.0f}, max={agg['cost_max']:.0f}")
        
        # Summary across all steps for this variation
        if len(all_step_results) > 1:
            print(f"\n  Summary across {len(expensive_steps)} steps:")
            avg_time_mean = sum(r['time_mean'] for r in all_step_results) / len(all_step_results)
            avg_cost_mean = sum(r['cost_mean'] for r in all_step_results) / len(all_step_results)
            print(f"    Average time (mean): {avg_time_mean*1000:.3f}ms")
            print(f"    Average cost (mean): {avg_cost_mean:.0f}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
