#!/usr/bin/env python3
"""
Simple benchmark for sep1 grammar constraint system.
Measures pure constraint computation time (load, get_mask, commit).
"""
import gzip
import json
import time
import sys
import statistics
from pathlib import Path

def load_vocab(vocab_path: str = ".cache/vocabs/vocab.json") -> dict:
    """Load GPT-2 vocabulary. Returns {token_str: token_id}"""
    with open(vocab_path) as f:
        return json.load(f)

def load_vocab_reverse(vocab_path: str = ".cache/vocabs/vocab.json") -> dict:
    """Load GPT-2 vocabulary as {token_id: token_bytes}"""
    with open(vocab_path) as f:
        vocab = json.load(f)
    # Create reverse mapping
    id_to_bytes = {}
    for token_str, token_id in vocab.items():
        # token_str is the repr of bytes, need to decode
        try:
            # Handle various token formats
            token_bytes = token_str.encode('utf-8', errors='surrogateescape')
        except:
            token_bytes = token_str.encode('utf-8')
        id_to_bytes[token_id] = token_bytes
    return id_to_bytes

def tokenize(text: str, vocab: dict = None) -> list[int]:
    """Tokenize using tiktoken (GPT-2 tokenizer)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode(text)
    except ImportError:
        print("Warning: tiktoken not available, using transformers")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return tokenizer.encode(text)

def get_token_bytes_map():
    """Get a mapping from token_id to bytes using tiktoken."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return {i: enc.decode_single_token_bytes(i) for i in range(enc.n_vocab)}

def benchmark_rust_native(constraint_path: str, tokens: list[int], token_bytes_map: dict, warmup: int = 3, runs: int = 20):
    """Benchmark using Rust implementation directly."""
    try:
        import _sep1
    except ImportError:
        print("Error: _sep1 module not found. Run 'cd python && maturin develop -r' first.")
        return None
    
    # Load constraint
    print(f"Loading constraint from {constraint_path}...")
    load_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with gzip.open(constraint_path, 'rt') as f:
            constraint_json_str = f.read()
        load_time = time.perf_counter() - t0
        load_times.append(load_time)
    
    print(f"  Load time: {min(load_times)*1000:.1f}ms (min of 3)")
    
    # Create GrammarConstraint from JSON string
    print("Creating GrammarConstraint...")
    t0 = time.perf_counter()
    constraint = _sep1.GrammarConstraint.from_json_string(constraint_json_str)
    constraint_time = time.perf_counter() - t0
    print(f"  Constraint creation time: {constraint_time*1000:.2f}ms")
    
    # Initialize state
    print("Initializing constraint state...")
    t0 = time.perf_counter()
    state = _sep1.GrammarConstraintState(constraint)
    init_time = time.perf_counter() - t0
    print(f"  Init time: {init_time*1000:.2f}ms")
    
    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        test_state = _sep1.GrammarConstraintState(constraint)
        for tok in tokens[:min(len(tokens), 10)]:
            mask = test_state.get_mask()
            # Mask is numpy array where mask[tok] indicates if token is valid
            if hasattr(mask, '__getitem__'):
                if tok < len(mask) and mask[tok]:
                    test_state.commit_bytes(token_bytes_map[tok])
                else:
                    break
            else:
                # Assume it's valid
                test_state.commit_bytes(token_bytes_map[tok])
    
    # Benchmark
    print(f"Running {runs} benchmark iterations...")
    all_get_mask_times = []
    all_commit_times = []
    valid_token_count = 0
    
    for run in range(runs):
        state = _sep1.GrammarConstraintState(constraint)
        get_mask_times = []
        commit_times = []
        
        for tok in tokens:
            t0 = time.perf_counter()
            mask = state.get_mask()
            t1 = time.perf_counter()
            get_mask_times.append(t1 - t0)
            
            # Check if token is valid
            is_valid = tok < len(mask) and mask[tok] if hasattr(mask, '__getitem__') else True
            
            if is_valid:
                state.commit_bytes(token_bytes_map[tok])
                t2 = time.perf_counter()
                commit_times.append(t2 - t1)
                valid_token_count += 1
            else:
                # Token not valid according to grammar, skip
                if run == 0:
                    print(f"  Warning: Token {tok} not valid at position {len(get_mask_times)}")
                break
        
        all_get_mask_times.extend(get_mask_times)
        all_commit_times.extend(commit_times)
    
    return {
        'load_time_ms': min(load_times) * 1000,
        'init_time_ms': init_time * 1000,
        'get_mask': {
            'mean_us': statistics.mean(all_get_mask_times) * 1e6,
            'median_us': statistics.median(all_get_mask_times) * 1e6,
            'min_us': min(all_get_mask_times) * 1e6,
            'max_us': max(all_get_mask_times) * 1e6,
            'p75_us': sorted(all_get_mask_times)[int(len(all_get_mask_times)*0.75)] * 1e6,
            'p95_us': sorted(all_get_mask_times)[int(len(all_get_mask_times)*0.95)] * 1e6,
            'p99_us': sorted(all_get_mask_times)[int(len(all_get_mask_times)*0.99)] * 1e6,
        },
        'commit': {
            'mean_us': statistics.mean(all_commit_times) * 1e6,
            'median_us': statistics.median(all_commit_times) * 1e6,
            'min_us': min(all_commit_times) * 1e6,
            'max_us': max(all_commit_times) * 1e6,
        },
        'total_tokens': len(tokens) * runs,
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_sep1.py <constraint.json.gz> <input_file>")
        print("Example: python benchmark_sep1.py .cache/test_vocabs/json_constraint.json.gz src/example.json")
        sys.exit(1)
    
    constraint_path = sys.argv[1]
    input_path = sys.argv[2]
    
    # Load token bytes mapping for commit
    print("Loading token bytes mapping...")
    token_bytes_map = get_token_bytes_map()
    print(f"  Loaded {len(token_bytes_map)} token mappings")
    
    print(f"Tokenizing input file: {input_path}")
    with open(input_path) as f:
        text = f.read()
    tokens = tokenize(text)
    print(f"  Input tokens: {len(tokens)}")
    
    # Run benchmark
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    results = benchmark_rust_native(constraint_path, tokens, token_bytes_map)
    
    if results:
        print(f"\n{'Metric':<30} {'Value':>15}")
        print("-"*45)
        print(f"{'Load time (ms)':<30} {results['load_time_ms']:>15.2f}")
        print(f"{'Init time (ms)':<30} {results['init_time_ms']:>15.2f}")
        print(f"{'Total tokens processed':<30} {results['total_tokens']:>15}")
        print()
        print("get_mask() timing (microseconds):")
        gm = results['get_mask']
        print(f"  {'Mean':<26} {gm['mean_us']:>15.2f}")
        print(f"  {'Median':<26} {gm['median_us']:>15.2f}")
        print(f"  {'Min':<26} {gm['min_us']:>15.2f}")
        print(f"  {'Max':<26} {gm['max_us']:>15.2f}")
        print(f"  {'p75':<26} {gm['p75_us']:>15.2f}")
        print(f"  {'p95':<26} {gm['p95_us']:>15.2f}")
        print(f"  {'p99':<26} {gm['p99_us']:>15.2f}")
        print()
        print("commit() timing (microseconds):")
        cm = results['commit']
        print(f"  {'Mean':<26} {cm['mean_us']:>15.2f}")
        print(f"  {'Median':<26} {cm['median_us']:>15.2f}")
        
        # Summary comparison with llguidance
        print()
        print("="*60)
        print("COMPARISON WITH STATE-OF-THE-ART")
        print("="*60)
        our_median = gm['median_us']
        llg_avg = 64  # LLGuidance average TBM from MaskBench
        xgr_avg = 728  # XGrammar average TBM from MaskBench
        print(f"sep1 (this work):     {our_median:>8.1f} μs (median)")
        print(f"LLGuidance (MaskBench): {llg_avg:>6} μs (average)")
        print(f"XGrammar (MaskBench):   {xgr_avg:>6} μs (average)")
        print()
        if our_median < llg_avg:
            print(f"✓ sep1 is {llg_avg/our_median:.1f}x faster than LLGuidance!")
        else:
            print(f"  sep1 is {our_median/llg_avg:.1f}x slower than LLGuidance")

if __name__ == "__main__":
    main()
