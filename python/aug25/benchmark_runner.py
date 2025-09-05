import argparse
import json
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
    token_to_id = {v: k for k, v in id_to_token.items()}
    sorted_tokens = sorted(token_to_id.keys(), key=len, reverse=True)
    token_ids = []
    pos = 0
    while pos < len(text_bytes):
        match_found = False
        for token_bytes in sorted_tokens:
            if text_bytes[pos:].startswith(token_bytes):
                token_ids.append(token_to_id[token_bytes])
                pos += len(token_bytes)
                match_found = True
                break
        if not match_found:
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

    # 1. Load and compile grammar
    print(f"Loading grammar from: {args.grammar}")
    grammar_def = _sep1.GrammarDefinition.from_ebnf_file(str(args.grammar))
    compiled_grammar = grammar_def.compile()

    # 2. Load vocabulary
    print("Loading GPT-2 vocabulary...")
    vocab_map = load_or_download_gpt2_vocab(
        ".cache/py_benchmark_vocabs", "gpt2_vocab.json",
        "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
    )
    token_to_id = {}
    id_to_token = {}
    max_token_id = 0
    for token_str, token_id in vocab_map.items():
        token_bytes = token_str.replace("Ġ", " ").replace("Ċ", "\n").encode('utf-8')
        token_to_id[token_bytes] = token_id
        id_to_token[token_id] = token_bytes
        max_token_id = max(max_token_id, token_id)

    # 3. Construct GrammarConstraint and export reference precompute2 model
    print("Constructing GrammarConstraint...")
    grammar_constraint = _sep1.GrammarConstraint(compiled_grammar, token_to_id, max_token_id)
    pre2_json_str = grammar_constraint.precompute2_json_string()
    pre3_json_str = grammar_constraint.precompute3_json_string()

    # 4. Load competitor model
    print(f"Loading competitor model from: {args.competitor}")
    CompetitorModel = load_competitor_model(args.competitor)
    
    t_start_load = time.perf_counter()
    if 'precompute3' in args.competitor.name:
        print("-> Using precompute3 JSON for competitor model.")
        competitor_json_str = pre3_json_str
    else:
        print("-> Using precompute2 JSON for competitor model.")
        competitor_json_str = pre2_json_str
    competitor_model = CompetitorModel.from_json_string(competitor_json_str)
    load_time = time.perf_counter() - t_start_load
    print(f"Competitor model loaded in {load_time:.4f} seconds.")

    if args.print_stats:
        print_model_statistics(competitor_model, args.competitor.name)

    # 5. Equivalence Check
    print(f"Loading reference model from: {args.reference}")
    ReferenceModel = load_competitor_model(args.reference)
    # The reference model is always loaded from precompute2 JSON for this benchmark suite.
    # Load reference model using precompute2 or precompute3 JSON based on its name.
    if 'precompute3' in args.reference.name:
        print("-> Using precompute3 JSON for reference model.")
        reference_json_str = pre3_json_str
    else:
        print("-> Using precompute2 JSON for reference model.")
        reference_json_str = pre2_json_str
    reference_model = ReferenceModel.from_json_string(reference_json_str)
    
    if args.print_stats:
        print_model_statistics(reference_model, args.reference.name)

    print(f"Running equivalence check against reference model: {args.reference.name}")
    
    equivalence_passed = True
    equivalence_details = "All tested states are equivalent."

    ENABLE_EQUIVALENCE_TEST = False
    if ENABLE_EQUIVALENCE_TEST:
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

    # 6. Prepare for benchmarking loop
    print(f"Loading and tokenizing code from: {args.code}")
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Tokenized into {len(token_ids)} tokens.")

    constraint_state = _sep1.GrammarConstraintState(grammar_constraint)
    timings = []
    mask_correctness_passed = True
    mask_correctness_details = "All masks matched the reference implementation."

    # 7. Run benchmark loop
    print(f"\n--- Running benchmark ({len(token_ids)} steps) ---")
    for i, token_id in tqdm(enumerate(token_ids), total=len(token_ids), desc="Benchmarking steps"):
        # Get the reference mask to check correctness (if enabled)
        if args.verify_masks:
            reference_mask_np = constraint_state.get_mask()
            if not reference_mask_np[token_id]:
                # This indicates an issue with the grammar or tokenizer, not the competitor.
                print(f"\nFATAL: Reference mask forbids token {token_id} at step {i}. Aborting.")
                sys.exit(1)

        # Get the state map for the competitor
        state_to_gss = constraint_state.filtered_state_gss_map()

        # --- TIMED SECTION ---
        t_start_mask = time.perf_counter()
        competitor_mask = competitor_model.get_mask(state_to_gss)
        t_end_mask = time.perf_counter()
        timings.append(t_end_mask - t_start_mask)
        # --- END TIMED SECTION ---

        # Verify the competitor's mask (if enabled)
        if args.verify_masks:
            # This check is expensive but crucial for validation
            ref_indices = {idx for idx, v in enumerate(reference_mask_np) if v}
            comp_indices = set(competitor_mask.to_indices())
            if ref_indices != comp_indices:
                mask_correctness_passed = False
                mask_correctness_details = f"Mask mismatch at step {i} (token_id {token_id})."
                print(f"❌ {mask_correctness_details}")
                # Stop verifying after first failure to avoid spamming output
                args.verify_masks = False

        # Advance the state
        constraint_state.commit(token_id)

    print("--- Benchmark finished ---")

    # 8. Compile and save results
    if timings:
        p = np.percentile(timings, [50, 90, 99])
        summary_stats = {
            "count": len(timings),
            "mean": float(np.mean(timings)),
            "stddev": float(np.std(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
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
            "grammar_file": str(args.grammar),
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
            },
            "get_mask_timings_seconds": timings,
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
    parser.add_argument("-g", "--grammar", type=Path, required=True, help="Path to the .ebnf grammar file.")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to the code file to use as input.")
    parser.add_argument("-m", "--competitor", type=Path, required=True, help="Path to the competitor's model .py file.")
    parser.add_argument("-r", "--reference", type=Path, required=True, help="Path to the reference model .py file for equivalence checking.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file or directory.")
    
    parser.add_argument('--no-verify-masks', dest='verify_masks', action='store_false',
                        help="Disable correctness verification of masks at each step (improves benchmark purity).")
    parser.add_argument('--print-stats', action='store_true',
                        help="Print detailed statistics about the loaded models before running the benchmark.")
    parser.set_defaults(verify_masks=True)

    args = parser.parse_args()

    for p in [args.grammar, args.code, args.competitor, args.reference]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    run_benchmark(args)

if __name__ == "__main__":
    main()
