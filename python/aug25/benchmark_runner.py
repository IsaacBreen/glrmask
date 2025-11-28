import sys
from pathlib import Path

# Add project root to sys.path to resolve imports when running as a script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import gzip
import gc
import os
import time
import importlib.util
import tempfile
import requests
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm

from python.aug25.constraint_utils import extract_id_to_token_map
from python.aug25.stats import Stats

# --- Helper Functions (from former example_js.py) ---


def get_vocab(url: str | None, path: Path | None, cache_dir: Path, force_download: bool) -> dict[str, int]:
    """
    Loads a vocabulary from a local path or a URL.
    The vocabulary can be a JSON dictionary (token -> id) or a JSON list of strings.
    """
    if not url and not path:
        raise ValueError("Either --vocab-url or --vocab-path must be provided.")
    if url and path:
        raise ValueError("Provide either --vocab-url or --vocab-path, not both.")

    def _load_and_process_vocab(vocab_path: Path) -> dict[str, int]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            print(f"Loaded vocabulary is a list of strings, converting to a dictionary ({len(data)} tokens).")
            return {token: i for i, token in enumerate(data)}
        raise TypeError(f"Unsupported vocabulary format in {vocab_path}. Expected a dict[str, int] or a list[str], but got {type(data)}.")

    if path:
        print(f"Loading vocabulary from local path: {path}")
        return _load_and_process_vocab(path)

    # Handle URL download and caching
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.split("/")[-1]
    cache_path = cache_dir / file_name

    if not cache_path.exists() or force_download:
        print(f"Downloading vocabulary from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved vocabulary to cache: {cache_path}")
        except requests.RequestException as e:
            print(f"Error downloading vocabulary: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Loading vocabulary from cache: {cache_path}")

    return _load_and_process_vocab(cache_path)


def parse_len_ranges(ranges: list[str] | None) -> tuple[set[int], int | None]:
    """
    Parses a list of string representations of integer ranges.
    e.g., ["1", "3-5", "8-"] -> ({1, 3, 4, 5}, 8)
    """
    if not ranges:
        return set(), None

    allowed_lengths = set()
    min_len_unbounded = None

    for r in ranges:
        if '-' in r:
            parts = r.split('-', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: {r}")
            start_str, end_str = parts

            if not start_str:
                raise ValueError(f"Invalid range format: {r}. Start must be specified.")

            try:
                start = int(start_str)
            except ValueError:
                raise ValueError(f"Invalid start of range in '{r}'")

            if not end_str: # e.g. "8-"
                if min_len_unbounded is not None:
                    min_len_unbounded = min(min_len_unbounded, start)
                else:
                    min_len_unbounded = start
            else: # e.g. "3-5"
                try:
                    end = int(end_str)
                except ValueError:
                    raise ValueError(f"Invalid end of range in '{r}'")
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) > end ({end}) in '{r}'")
                allowed_lengths.update(range(start, end + 1))
        else:
            try:
                allowed_lengths.add(int(r))
            except ValueError:
                raise ValueError(f"Invalid length value: {r}")

    return allowed_lengths, min_len_unbounded


def filter_vocab(vocab: dict[str, int], allowed_lengths: set[int], min_len_unbounded: int | None) -> dict[str, int]:
    """
    Applies filters to the vocabulary based on token byte length.
    """
    if not allowed_lengths and min_len_unbounded is None:
        return vocab

    print(f"Filtering vocabulary by token byte length...")

    filtered = {}
    for token_str, token_id in vocab.items():
        # Convert GPT-2 byte-level BPE Unicode characters to actual bytes
        # See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
        # Ġ (U+0120) -> space, Ċ (U+010A) -> newline, ĉ (U+0109) -> tab, č (U+010D) -> CR
        # Note: ą appears to be a legacy mapping that also represents newline in some contexts
        processed_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n").replace("ĉ", "\t").replace("č", "\r")
        token_len = len(processed_str.encode('utf-8'))

        keep = False
        if token_len in allowed_lengths:
            keep = True
        elif min_len_unbounded is not None and token_len >= min_len_unbounded:
            keep = True

        if keep:
            filtered[token_str] = token_id

    print(f"  -> Filtered vocabulary from {len(vocab)} to {len(filtered)} tokens.")
    return filtered


def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping from byte values to unicode strings for GPT-2 byte-level BPE.
    See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# Build the inverse mapping: unicode char -> byte value
_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def gpt2_token_str_to_bytes(token_str: str) -> bytes:
    """Convert a GPT-2 byte-level BPE token string to actual bytes."""
    return bytes([_UNICODE_TO_BYTE[c] for c in token_str])

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

    tokens_with_pos = []
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
            tokens_with_pos.append((longest_match_id, pos, pos + longest_match_len))
            pos += longest_match_len
        else:
            raise ValueError(f"Failed to tokenize. No token found for prefix: {text_bytes[pos:pos+20]!r}")
    return tokens_with_pos

def load_model_class(model_path: Path):
    """Dynamically loads the 'Model' class from a Python file."""
    # To support relative imports within the model file, we must construct
    # a module name that reflects its package structure relative to the project root.
    module_path_for_name = model_path
    if module_path_for_name.is_absolute():
        try:
            # Try to make it relative to the project root.
            module_path_for_name = module_path_for_name.relative_to(_project_root)
        except ValueError:
            # Path is not within the project root. The original absolute path's parts
            # will be used, which is unlikely to work for relative imports but is the
            # best we can do.
            pass

    module_name = ".".join(module_path_for_name.with_suffix('').parts)

    # The file location for importlib must be the original path.
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
    print("Decrecated")
    return
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
            if isinstance(edge_key[1], ffi.HybridBitset):
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

def run_benchmark(args, model, tokens_with_pos, load_time, run_index: int = 0):
    """Main benchmark logic (single model; no baseline/reference in-process)."""
    token_ids = [t[0] for t in tokens_with_pos]
    token_positions = [(t[1], t[2]) for t in tokens_with_pos]

    get_mask_timings: list[float] = []
    commit_timings: list[float] = []
    masks_ranges: list[list[list[int]]] = []  # list of [[s,e], ...] per step
    early_termination_index: int | None = None

    # 5. Run benchmark loop

    print(f"\n--- Running benchmark ({len(token_ids)} tokens) ---")
    progress_bar = tqdm(enumerate(token_ids), total=len(token_ids), desc="Benchmarking steps", disable=os.environ.get("DISABLE_TQDM") == "1")
    for i, token_id in progress_bar:
        if not os.environ.get("NO_GET_MASK") == '1':
            Stats.get().reset()
            gc.disable()
            t_start_mask = time.perf_counter()
            result = model.get_mask()
            t_end_mask = time.perf_counter()
            gc.enable()

            timing = t_end_mask - t_start_mask
            mask_rs = result

            if isinstance(result, dict) and result.get("type") == "timed_output":
                if "output" in result and "time_sec" in result:
                    mask_rs = result["output"]
                    timing = float(result["time_sec"])
                else:
                    raise ValueError("Model returned timed_output dict for get_mask without 'output' or 'time_sec'.")

            # Check if the mask is empty, indicating the model has rejected valid input
            ranges = mask_rs.to_ranges()
            if not ranges:
                print(f"\n[WARNING] Model returned empty mask at step {i} (token_id={token_id}). Stopping benchmark for this model.")
                early_termination_index = i
                break

            get_mask_timings.append(timing)
            # Export the mask for later cross-model comparison during analysis
            masks_ranges.append(ranges)

        # Advance the state
        gc.disable()
        # Note: if get_mask() wasn't called or wasn't empty, we proceed to commit.
        t_start_commit = time.perf_counter()
        result = model.commit(token_id)
        t_end_commit = time.perf_counter()
        gc.enable()
        timing = t_end_commit - t_start_commit
        if isinstance(result, dict) and result.get("type") == "timed_output":
            if "time_sec" in result:
                timing = float(result["time_sec"])
            else:
                raise ValueError("Model returned timed_output dict for commit without 'time_sec'.")
        commit_timings.append(timing)

    print("--- Benchmark finished ---")

    # Finalize model if it has a finalize method (e.g., for printing stats)
    if hasattr(model, 'finalize'):
        print("--- Finalizing model ---")
        model.finalize()

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
            "total_input_tokens": len(token_ids),
            "early_termination_index": early_termination_index,
            "load_time_seconds": load_time,
            "get_mask_timings_seconds": get_mask_timings,
            "commit_timings_seconds": commit_timings,
            "masks_ranges": masks_ranges,
            "token_positions": token_positions,
            "summary_stats": summary_stats,
        }
    }

    # Determine output path
    constraint_stem = Path(args.constraint_file).name.replace('.json.gz', '').replace('.json', '')
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            base_name = f"{args.model.stem}__{constraint_stem}_results.json"
            if args.repeat > 1:
                base_name = f"{args.model.stem}__{constraint_stem}_run{run_index + 1}_results.json"
            output_path = output_path / base_name
        else:  # It's a file path
            if args.repeat > 1:
                output_path = output_path.with_name(f"{output_path.stem}_run{run_index + 1}{output_path.suffix}")
    else:
        base_name = f"{args.model.stem}__{constraint_stem}_results.json"
        if args.repeat > 1:
            base_name = f"{args.model.stem}__{constraint_stem}_run{run_index + 1}_results.json"
        output_path = Path(base_name)
    
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
    parser.add_argument('--repeat', type=int, default=1, help="Number of times to repeat the benchmark run.")

    vocab_source_group = parser.add_mutually_exclusive_group()
    vocab_source_group.add_argument('--vocab-url', type=str, default="https://huggingface.co/openai-community/gpt2/raw/main/vocab.json",
                        help="URL to the full vocabulary JSON for tokenization (default: GPT-2 from HuggingFace).")
    vocab_source_group.add_argument("--vocab-path", type=Path, help="Path to a local JSON vocabulary file.")
    parser.add_argument("--vocab-list", type=str, nargs='+', help="A list of strings to use as the vocabulary. Can be combined with --vocab-url/--vocab-path.")

    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vocabs"), help="Directory to cache downloaded vocabularies.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the vocabulary even if it exists in the cache.")
    parser.add_argument("--token-len", type=str, nargs='+', help="Filter vocabulary to include tokens with specific byte lengths or ranges. E.g., '1' '3-5' '8-'.")

    args = parser.parse_args()

    for p in [args.constraint_file, args.code, args.model]:
        if not p.exists():
            parser.error(f"File not found: {p}")

    # --- Setup: Load model and data once ---
    print("--- Setting up benchmark environment ---")
    print(f"Loading model class from: {args.model}")
    ModelClass = load_model_class(args.model)

    print(f"Loading pre-compiled grammar constraint from: {args.constraint_file}")
    p = str(args.constraint_file)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            constraint_json_str = f.read()
    else:
        constraint_json_str = args.constraint_file.read_text(encoding='utf-8')

    # Initial model load
    print("Performing initial model load...")
    t_start_load = time.perf_counter()
    model = ModelClass.from_json_string(constraint_json_str)
    load_time = time.perf_counter() - t_start_load
    print(f"Model loaded in {load_time:.4f} seconds.")

    if args.print_stats:
        print_model_statistics(model, args.model.name)

    # Tokenize input code once
    print(f"Loading and tokenizing code from: {args.code}")
    
    # Load the vocabulary for tokenization, applying filters if specified
    print(f"Loading vocabulary...")
    vocab_cache_dir = args.cache_dir
    vocab_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get the base vocabulary from a file/URL if provided
    base_vocab = {}
    if args.vocab_url or args.vocab_path:
        if args.vocab_path:
            vocab_url = None
        else:
            vocab_url = args.vocab_url
        base_vocab = get_vocab(vocab_url, args.vocab_path, vocab_cache_dir, args.force_download)

    # 2. Apply filters to the base vocabulary
    try:
        allowed_lengths, min_len_unbounded = parse_len_ranges(args.token_len)
    except ValueError as e:
        parser.error(f"Invalid --token-len value: {e}")

    modified_vocab = filter_vocab(base_vocab, allowed_lengths, min_len_unbounded)

    # 3. Add tokens from vocab-list (filters do not apply to these)
    if args.vocab_list:
        print(f"Adding {len(args.vocab_list)} tokens from --vocab-list.")
        max_id = -1
        if modified_vocab:
            max_id = max(modified_vocab.values())
        for token in args.vocab_list:
            if token not in modified_vocab:
                max_id += 1
                modified_vocab[token] = max_id

    # Use the modified vocabulary for tokenization
    raw_vocab = modified_vocab if modified_vocab else base_vocab

    # If no vocab was loaded, use the original default URL
    if not raw_vocab:
        print(f"No vocabulary specified, using default from: {args.vocab_url}")
        vocab_file_name = args.vocab_url.split("/")[-1]
        raw_vocab = load_or_download_gpt2_vocab(vocab_cache_dir, vocab_file_name, args.vocab_url)

    # Convert from {token_str: id} to {id: token_bytes} for greedy_tokenizer
    # Use proper GPT-2 byte-level BPE decoding
    id_to_token = {}
    for token_str, token_id in raw_vocab.items():
        try:
            id_to_token[token_id] = gpt2_token_str_to_bytes(token_str)
        except KeyError:
            # Skip tokens with characters not in the GPT-2 byte mapping
            pass
    print(f"Vocabulary loaded with {len(id_to_token)} tokens.")
    
    code_bytes = args.code.read_bytes()
    tokens_with_pos = greedy_tokenizer(code_bytes, id_to_token)
    print(f"Tokenized into {len(tokens_with_pos)} tokens.")

    # --- Run benchmark loop ---
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Running benchmark: Run {i + 1}/{args.repeat} for {args.model.name} ---")

        # For runs after the first, reset or reload the model
        if i > 0:
            if hasattr(model, 'reset'):
                print("Resetting model state for new run.")
                model.reset()
            else:
                print("Model has no reset method, reloading for new run.")
                t_start_reload = time.perf_counter()
                model = ModelClass.from_json_string(constraint_json_str)
                reload_time = time.perf_counter() - t_start_reload
                print(f"Model reloaded in {reload_time:.4f} seconds.")

        run_benchmark(args, model, tokens_with_pos, load_time, run_index=i)


if __name__ == "__main__":
    main()
