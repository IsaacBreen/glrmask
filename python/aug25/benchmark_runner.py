import argparse
import json
import gzip
import os
import time
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm

# --- Helper Functions (from former example_js.py) ---

def load_or_download_gpt2_vocab(cache_dir, file_name, url):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / file_name
    if not cache_path.exists():
        import requests
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

def load_model_class(model_path: Path):
    """Dynamically loads the 'Model' class from a Python file."""
    # Convert path like 'aug25/precompute3_model.py' to 'aug25.precompute3_model' for correct package context
    module_name = ".".join(model_path.with_suffix('').parts)
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'Model'):
        raise AttributeError(f"Model script {model_path} must define a 'Model' class.")
    
    return getattr(module, 'Model')

def print_model_statistics(model, model_name: str):
    """Analyzes and prints statistics about the loaded graph model."""
    from aug25.common_interface import RangeSet  # local import to avoid cycles
    import _sep1 as ffi
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
            if isinstance(edge_key[1], ffi.Bitset):
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
            for (pop, llm_bv), dests in children:
                pops.append(pop)
                if not llm_bv.is_empty():
                    llm_rs_sizes.append(llm_bv.len())
                total_dest_edges += len(dests)
                for _, state_bv in dests:
                    if not state_bv.is_empty():
                        state_rs_sizes.append(state_bv.len())
        print(f"  - Edge Groups (pop, llm_bv): {total_edge_groups}")
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
        import numpy as np
        arr = np.array(data)
        p = np.percentile(arr, [0, 25, 50, 75, 90, 99, 100])
        print(f"  - {name} Distribution:\n    - Mean: {arr.mean():.2f}, Std: {arr.std():.2f}\n    - Min: {p[0]:.0f}, 25%: {p[1]:.0f}, Med: {p[2]:.0f}, 75%: {p[3]:.0f}, 90%: {p[4]:.0f}, 99%: {p[5]:.0f}, Max: {p[6]:.0f}")

    print_dist("Pop counts", pops)
    print_dist("Node fan-out", fan_outs)
    print_dist("LLM Bitset sizes", llm_rs_sizes)
    if is_precompute3:
        print_dist("State Bitset sizes", state_rs_sizes)
    print("-" * 20)

def run_benchmark(args):
    """Main benchmark logic (single model; no baseline/reference in-process)."""
    print("--- Setting up benchmark environment ---")

    # 1. Load pre-compiled GrammarConstraint
    print(f"Loading pre-compiled grammar constraint from: {args.constraint_file}")
    p = str(args.constraint_file)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            constraint_json_str = f.read()
    else:
        constraint_json_str = args.constraint_file.read_text(encoding='utf-8')

    # 2. Extract vocabulary for tokenizer
    constraint_json = json.loads(constraint_json_str)
    # The vocabulary maps token strings to integer IDs. We need ID -> token bytes.
    llm_token_map: tuple[list[int], int] = constraint_json['llm_token_map']
    id_to_token: dict[int, bytes] = {}
    for token_bytes, token_id in llm_token_map:
        id_to_token[token_id] = bytes(token_bytes)

    # 3. Load model
    print(f"Loading model from: {args.model}")
    ModelClass = load_model_class(args.model)
    
    t_start_load = time.perf_counter()
    model = ModelClass.from_json_string(constraint_json_str)
    load_time = time.perf_counter() - t_start_load
    print(f"Model loaded in {load_time:.4f} seconds.")

    if args.print_stats:
        print_model_statistics(model, args.model.name)

    # 4. Tokenize input code
    print(f"Loading and tokenizing code from: {args.code}")
    code_bytes = args.code.read_bytes()
    token_ids = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Tokenized into {len(token_ids)} tokens.")

    get_mask_timings: list[float] = []
    commit_timings: list[float] = []
    masks_ranges: list[list[list[int]]] = []  # list of [[s,e], ...] per step

    # 5. Run benchmark loop

    print(f"\n--- Running benchmark ({len(token_ids)} steps) ---")
    progress_bar = tqdm(enumerate(token_ids), total=len(token_ids), desc="Benchmarking steps", disable=os.environ.get("DISABLE_TQDM") == "1")
    for i, token_id in progress_bar:
        t_start_mask = time.perf_counter()
        progress_bar.set_postfix_str("get_mask")
        mask_rs = model.get_mask()
        t_end_mask = time.perf_counter()
        get_mask_timings.append(t_end_mask - t_start_mask)
        # Export the mask for later cross-model comparison during analysis
        masks_ranges.append(mask_rs.to_ranges())

        # Advance the state
        progress_bar.set_postfix_str("commit")
        t_start_commit = time.perf_counter()
        model.commit(token_id)
        t_end_commit = time.perf_counter()
        commit_timings.append(t_end_commit - t_start_commit)

    print("--- Benchmark finished ---")

    # 6. Compile and save results
    if get_mask_timings:
        pcts = np.percentile(get_mask_timings, [50, 90, 99])
        summary_stats = {
            "count": len(get_mask_timings),
            "mean": float(np.mean(get_mask_timings)),
            "stddev": float(np.std(get_mask_timings)),
            "min": float(np.min(get_mask_timings)),
            "max": float(np.max(get_mask_timings)),
            "p50": float(pcts[0]),
            "p90": float(pcts[1]),
            "p99": float(pcts[2]),
        }
    else:
        summary_stats = {}

    output_data = {
        "model_script": str(args.model),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "grammar_file": str(args.constraint_file),
            "code_file": str(args.code),
        },
        "results": {
            "load_time_seconds": load_time,
            "get_mask_timings_seconds": get_mask_timings,
            "commit_timings_seconds": commit_timings,
            "masks_ranges": masks_ranges,
            "summary_stats": summary_stats,
        }
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"{args.model.stem}_results.json"
    else:
        output_path = Path(f"{args.model.stem}_results.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a grammar constraint model implementation (single model, masks exported).")
    parser.add_argument("-f", "--constraint-file", type=Path, required=True, help="Path to the pre-compiled .json.gz grammar constraint file.")
    parser.add_argument("-c", "--code", type=Path, required=True, help="Path to the code file to use as input.")
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to the model .py file.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file or directory.")
    parser.add_argument('--print-stats', action='store_true', help="Print detailed statistics about the loaded models before running the benchmark.")

    args = parser.parse_args()

    for p in [args.constraint_file, args.code, args.model]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    run_benchmark(args)


if __name__ == "__main__":
    main()
