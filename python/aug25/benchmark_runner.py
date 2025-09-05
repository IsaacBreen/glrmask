import argparse
import json
import gzip
import os
import requests
import time
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from collections import Counter
from tqdm import tqdm

import _sep1
from aug25.equality import are_equivalent_for_state
from aug25.common_interface import RangeSet

# --- Helper Functions (from former example_js.py) ---

def load_or_download_gpt2_vocab(cache_dir, file_name, url):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / file_name
    if not cache_path.exists():
        print(f"Downloading GPT-2 vocab from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    vocab_map = json.loads(cache_path.read_text(encoding='utf-8'))

    # Exclude tokens longer than 5
    vocab_map = {k: v for k, v in vocab_map.items() if len(k.encode('utf-8')) <= 5}

    return vocab_map

def greedy_tokenizer(text_bytes, id_to_token):
    # Build a Trie for fast prefix matching.
    # The key '<ID>' stores the token ID for a complete token.
    trie_root = {}
    for token_id, token_bytes in id_to_token.items():
        node = trie_root
        for byte_val in token_bytes:
            node = node.setdefault(byte_val, {})
        node['<ID>'] = token_id

    token_ids = []
    pos = 0
    while pos < len(text_bytes):
        # Find the longest possible token match starting at the current position.
        node = trie_root
        longest_match_id = -1
        longest_match_len = 0
        
        # Traverse the Trie with bytes from the current position.
        for i in range(len(text_bytes) - pos):
            current_byte = text_bytes[pos + i]
            if current_byte in node:
                node = node[current_byte]
                if '<ID>' in node:
                    # Found a valid token, record it and keep searching for a longer one.
                    longest_match_id = node['<ID>']
                    longest_match_len = i + 1
            else:
                # No further matches possible from this prefix.
                break
        
        if longest_match_len > 0:
            token_ids.append(longest_match_id)
            pos += longest_match_len
        else:
            raise ValueError(f"Failed to tokenize. No token found for prefix: {text_bytes[pos:pos+20]!r}")
    return token_ids

def load_competitor_model(competitor_path: Path):
    """Dynamically loads the 'Model' class from a Python file."""
    # Convert path like 'aug25/precompute3_model.py' to 'aug25.precompute3_model' for correct package context
    module_name = ".".join(competitor_path.with_suffix('').parts)
    spec = importlib.util.spec_from_file_location(module_name, competitor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {competitor_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'Model'):
        raise AttributeError(f"Competitor script {competitor_path} must define a 'Model' class.")
    
    return getattr(module, 'Model')

def print_model_statistics(model, model_name: str):
    """Analyzes and prints statistics about the loaded graph model."""
    print(f"\n--- Statistics for {model_name} ---")

    if not hasattr(model, 'arena') or not model.arena:
        print("  - Model has no arena or is empty. Cannot compute stats.")
        return

    num_nodes = len(model.arena)
    num_roots = len(model.roots_map)
    num_end_nodes = sum(1 for i in model.arena if model.is_end(i))

    print(f"  - Roots: {num_roots}")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - End nodes: {num_end_nodes} ({num_end_nodes/num_nodes:.2%} of total)")

    # Detect model type by inspecting the first child edge of the first node with children
    is_precompute3 = False
    for node_data in model.arena.values():
        children = node_data.get("children")
        if children:
            edge_key, _ = children[0]
            # precompute3 has (pop, RangeSet) as key, precompute2 has (pop, int | None)
            if isinstance(edge_key[1], RangeSet):
                 is_precompute3 = True
            break
    
    model_type = "Precompute3" if is_precompute3 else "Precompute2"
    print(f"  - Detected Model Type: {model_type}")

    pops = []
    fan_outs = []
    llm_rs_sizes = []
    state_rs_sizes = [] # precompute3 only

    if is_precompute3:
        total_edge_groups = 0
        total_dest_edges = 0
        for node_data in model.arena.values():
            children = node_data.get("children", [])
            fan_outs.append(len(children))
            total_edge_groups += len(children)
            for (pop, llm_rs), dests in children:
                pops.append(pop)
                if llm_rs.intervals:
                    llm_rs_sizes.append(sum(e - s + 1 for s, e in llm_rs.intervals))
                total_dest_edges += len(dests)
                for _, state_bv in dests:
                    if state_bv:
                        state_rs_sizes.append(sum(e - s + 1 for s, e in state_bv))
        print(f"  - Edge Groups (pop, llm_rs): {total_edge_groups}")
        print(f"  - Destination Edges (dest, state_bv): {total_dest_edges}")

    else: # Precompute2
        total_edge_groups = 0
        total_dest_edges = 0
        for node_data in model.arena.values():
            children = node_data.get("children", [])
            fan_outs.append(len(children))
            total_edge_groups += len(children)
            for (pop, _), dests in children:
                pops.append(pop)
                total_dest_edges += len(dests)
                for _, llm_rs in dests:
                    if llm_rs.intervals:
                        llm_rs_sizes.append(sum(e - s + 1 for s, e in llm_rs.intervals))
        print(f"  - Edge Groups (pop, sid): {total_edge_groups}")
        print(f"  - Destination Edges (dest, llm_rs): {total_dest_edges}")

    def print_dist(name, data):
        if not data:
            print(f"  - {name} Distribution: N/A (no data)")
            return
        arr = np.array(data)
        p = np.percentile(arr, [0, 25, 50, 75, 90, 99, 100])
        print(f"  - {name} Distribution:\n    - Mean: {arr.mean():.2f}, Std: {arr.std():.2f}\n    - Min: {p[0]:.0f}, 25%: {p[1]:.0f}, Med: {p[2]:.0f}, 75%: {p[3]:.0f}, 90%: {p[4]:.0f}, 99%: {p[5]:.0f}, Max: {p[6]:.0f}")

    print_dist("Pop counts", pops)
    print_dist("Node fan-out", fan_outs)
    print_dist("LLM RangeSet sizes", llm_rs_sizes)
    if is_precompute3:
        print_dist("State RangeSet sizes", state_rs_sizes)
    print("-" * 20)

def run_benchmark(args):
    """Main benchmark logic."""
    print("--- Setting up benchmark environment ---")
    is_builtin_ref = (args.reference == 'builtin')

    # 1. Load pre-compiled GrammarConstraint
    print(f"Loading pre-compiled grammar constraint from: {args.constraint_file}")
    # Read the gzipped JSON into a string and construct the GrammarConstraint using the
    # available from_json_string(...) API.
    p = str(args.constraint_file)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            constraint_json_str = f.read()
    else:
        # Fallback: if the file is plain JSON, read it directly.
        constraint_json_str = args.constraint_file.read_text(encoding='utf-8')
    grammar_constraint = _sep1.GrammarConstraint.from_json_string(constraint_json_str)

    # 2. Extract vocabulary for tokenizer
    id_to_token = grammar_constraint.get_id_to_token_map()

    # 4. Load competitor model
    print(f"Loading competitor model from: {args.competitor}")
    CompetitorModel = load_competitor_model(args.competitor)
    
    t_start_load = time.perf_counter()
    full_constraint_json_str = grammar_constraint.to_json_string()
    competitor_model = CompetitorModel.from_json_string(full_constraint_json_str)
    load_time = time.perf_counter() - t_start_load
    print(f"Competitor model loaded in {load_time:.4f} seconds.")

    if args.print_stats:
        print_model_statistics(competitor_model, args.competitor.name)

    # 5. Equivalence Check / Reference Model Setup
    reference_model = None
    if is_builtin_ref:
        print("Using 'builtin' reference (GrammarConstraintState) for mask verification.")
        # Equivalence check is not possible against builtin.
        equivalence_passed = True
        equivalence_details = "N/A (reference is 'builtin')"
    else:
        ref_path = Path(args.reference)
        print(f"Loading reference model from: {ref_path}")
        ReferenceModel = load_competitor_model(ref_path)
        # All models are loaded from precompute3 JSON for this benchmark suite.
        reference_model = ReferenceModel.from_json_string(full_constraint_json_str)

        if args.print_stats:
            print_model_statistics(reference_model, ref_path.name)

        print(f"Running equivalence check against reference model: {ref_path.name}")

        equivalence_passed = True
        equivalence_details = "All tested states are equivalent."

        if args.run_equivalence_check:
            # Check that root sets are the same before proceeding
            ref_roots = set(reference_model.roots_map.keys())
            comp_roots = set(competitor_model.roots_map.keys())
            if ref_roots != comp_roots:
                equivalence_passed = False
                equivalence_details = f"Root sets differ. Ref: {len(ref_roots)} roots, Comp: {len(comp_roots)} roots."
            else:
                sorted_roots = sorted(list(ref_roots))
                total_roots = len(sorted_roots)
                print(f"Checking equivalence across {total_roots} tokenizer states...")
                for i, sid in enumerate(sorted_roots):
                    # Progress indicator
                    if (i > 0 and i % 25 == 0) or i == total_roots - 1 or i == 0:
                        print(f"  ... verified {i+1}/{total_roots} states", end='\r')

                    passed, details = are_equivalent_for_state(
                        reference_model, reference_model.get_root(sid),
                        competitor_model, competitor_model.get_root(sid),
                        verbose=False
                    )
                    if not passed:
                        print() # Newline to clear the progress indicator
                        equivalence_passed = False
                        equivalence_details = f"Equivalence failed for tokenizer state {sid}. Details: {details}"
                        break
                if equivalence_passed:
                    print() # Final newline after progress indicator

            if equivalence_passed:
                print("✅ Equivalence check passed.")
            else:
                print(f"❌ Equivalence check failed: {equivalence_details}")
        else:
            equivalence_details = "Equivalence check was skipped."

    # 6. Prepare for benchmarking loop
    print(f"Loading and tokenizing code from: {args.code}")
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Tokenized into {len(token_ids)} tokens.")

    constraint_state = _sep1.GrammarConstraintState(grammar_constraint)
    get_mask_timings = []
    commit_timings = []
    mask_correctness_passed = True
    mask_correctness_details = "All masks matched the reference implementation."
    mask_mismatch_indices = []

    # 7. Run benchmark loop
    print(f"\n--- Running benchmark ({len(token_ids)} steps) ---")
    progress_bar = tqdm(enumerate(token_ids), total=len(token_ids), desc="Benchmarking steps")
    for i, token_id in progress_bar:
        # Get the state map for the competitor. This is needed for all mask calculations.
        progress_bar.set_postfix_str("filtered_state_gss_map")
        state_to_gss = constraint_state.filtered_state_gss_map()

        # Get the reference mask to check correctness (if enabled)
        if args.verify_masks:
            progress_bar.set_postfix_str("get_mask (ref)")
            if is_builtin_ref:
                reference_mask_np = constraint_state.get_mask()
                if not reference_mask_np[token_id]:
                    print(f"\nFATAL: Builtin reference mask forbids token {token_id} at step {i}. Aborting.")
                    sys.exit(1)
            else:
                reference_mask_rs = reference_model.get_mask(state_to_gss)
                if not reference_mask_rs.contains(token_id):
                    print(f"\nFATAL: Reference model mask forbids token {token_id} at step {i}. Aborting.")
                    sys.exit(1)

        # --- TIMED SECTION ---
        progress_bar.set_postfix_str("get_mask (competitor)")
        t_start_mask = time.perf_counter()
        competitor_mask = competitor_model.get_mask(state_to_gss)
        t_end_mask = time.perf_counter()
        get_mask_timings.append(t_end_mask - t_start_mask)
        # --- END TIMED SECTION ---

        # Verify the competitor's mask (if enabled)
        if args.verify_masks:
            progress_bar.set_postfix_str("Verifying mask")
            # This check is expensive but crucial for validation
            if is_builtin_ref:
                ref_indices = {idx for idx, v in enumerate(reference_mask_np) if v}
            else:
                ref_indices = set(reference_mask_rs.to_indices())

            comp_indices = set(competitor_mask.to_indices())
            if ref_indices != comp_indices:
                mask_correctness_passed = False
                mask_mismatch_indices.append(i)

        # Advance the state
        progress_bar.set_postfix_str("commit")
        t_start_commit = time.perf_counter()
        constraint_state.commit(token_id)
        t_end_commit = time.perf_counter()
        commit_timings.append(t_end_commit - t_start_commit)

    if not mask_correctness_passed:
        mask_correctness_details = f"Found {len(mask_mismatch_indices)} mask mismatches across {len(token_ids)} steps."
        print(f"\n❌ {mask_correctness_details}")

    print("--- Benchmark finished ---")

    # 8. Compile and save results
    if get_mask_timings:
        p = np.percentile(get_mask_timings, [50, 90, 99])
        summary_stats = {
            "count": len(get_mask_timings),
            "mean": float(np.mean(get_mask_timings)),
            "stddev": float(np.std(get_mask_timings)),
            "min": float(np.min(get_mask_timings)),
            "max": float(np.max(get_mask_timings)),
            "p50": float(p[0]),
            "p90": float(p[1]),
            "p99": float(p[2]),
        }
    else:
        summary_stats = {}

    output_data = {
        "competitor_script": str(args.competitor),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "grammar_file": str(args.constraint_file),
            "code_file": str(args.code),
        },
        "results": {
            "load_time_seconds": load_time,
            "equivalence_check": {
                "passed": equivalence_passed,
                "details": equivalence_details,
            },
            "mask_correctness_check": {
                "enabled": args.verify_masks,
                "passed": mask_correctness_passed,
                "details": mask_correctness_details,
                "mismatch_indices": mask_mismatch_indices,
            },
            "get_mask_timings_seconds": get_mask_timings,
            "commit_timings_seconds": commit_timings,
            "summary_stats": summary_stats,
        }
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"{args.competitor.stem}_results.json"
    else:
        output_path = Path(f"{args.competitor.stem}_results.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a grammar constraint model implementation.")
    parser.add_argument("-f", "--constraint-file", type=Path, required=True, help="Path to the pre-compiled .json.gz grammar constraint file.")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to the code file to use as input.")
    parser.add_argument("-m", "--competitor", type=Path, required=True, help="Path to the competitor's model .py file.")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Path to the reference model .py file for equivalence checking, or 'builtin' to use the slow C++ implementation for verification.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file or directory.")
    
    parser.add_argument('--no-verify-masks', dest='verify_masks', action='store_false',
                        help="Disable correctness verification of masks at each step (improves benchmark purity).")
    parser.add_argument('--print-stats', action='store_true',
                        help="Print detailed statistics about the loaded models before running the benchmark.")
    parser.add_argument('--skip-equivalence-check', dest='run_equivalence_check', action='store_false',
                        help="Disable the slow graph equivalence check against the reference model.")
    parser.set_defaults(verify_masks=True, run_equivalence_check=True)

    args = parser.parse_args()

    paths_to_check = [args.constraint_file, args.code, args.competitor]
    if args.reference != 'builtin':
        # Convert to Path for existence check
        paths_to_check.append(Path(args.reference))

    for p in paths_to_check:
        if not p.exists():
            parser.error(f"File not found: {p}")

    run_benchmark(args)

if __name__ == "__main__":
    main()
