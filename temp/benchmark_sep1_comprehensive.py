#!/usr/bin/env python3
"""
Comprehensive benchmark for sep1 grammar constraint system.
Measures TTFM (Time To First Mask) and TBM (Time Between Masks).
"""
import _sep1
import gzip
import time
import tiktoken
import statistics
from pathlib import Path

def benchmark_grammar(constraint_path: str, name: str, test_code: str = None):
    """Benchmark a grammar constraint."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Constraint file: {constraint_path}")
    print('='*60)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # TTFM: Time To First Mask (load + create + first get_mask)
    print("\n--- TTFM (Time To First Mask) ---")
    
    ttfm_times = []
    for i in range(5):
        # Load from disk
        t_start = time.perf_counter()
        with gzip.open(constraint_path, "rt") as f:
            json_str = f.read()
        
        # Create constraint
        constraint = _sep1.GrammarConstraint.from_json_string(json_str)
        
        # Create state and get first mask
        state = _sep1.GrammarConstraintState(constraint)
        mask = state.get_mask_bv()
        
        ttfm = (time.perf_counter() - t_start) * 1000
        ttfm_times.append(ttfm)
    
    print(f"  TTFM (load+init+first_mask): {min(ttfm_times):.2f}ms (min of 5)")
    print(f"  TTFM median: {statistics.median(ttfm_times):.2f}ms")
    
    # Breakdown
    with gzip.open(constraint_path, "rt") as f:
        json_str = f.read()
    
    load_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with gzip.open(constraint_path, "rt") as f:
            _ = f.read()
        load_times.append((time.perf_counter() - t0) * 1000)
    
    constraint = _sep1.GrammarConstraint.from_json_string(json_str)
    
    create_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        state = _sep1.GrammarConstraintState(constraint)
        create_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"  - Load from disk: {min(load_times):.2f}ms")
    print(f"  - Create state: {min(create_times):.3f}ms")
    
    # TBM: Time Between Masks (pure runtime performance)
    print("\n--- TBM (Time Between Masks) ---")
    
    # Initial state get_mask
    initial_mask_times = []
    for _ in range(100):
        state = _sep1.GrammarConstraintState(constraint)
        t0 = time.perf_counter()
        mask = state.get_mask_bv()
        initial_mask_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"  Initial state get_mask:")
    print(f"    median: {statistics.median(initial_mask_times)*1000:.1f}us")
    print(f"    p75: {sorted(initial_mask_times)[75]*1000:.1f}us")
    print(f"    p99: {sorted(initial_mask_times)[99]*1000:.1f}us")
    
    # Simulate generation if test code provided
    if test_code:
        tokens = enc.encode(test_code)
        print(f"\n  Simulating generation ({len(tokens)} tokens):")
        print(f"    Code: {test_code[:50]}...")
        
        # Run multiple trials
        all_mask_times = []
        all_commit_times = []
        
        for trial in range(5):
            state = _sep1.GrammarConstraintState(constraint)
            mask_times = []
            commit_times = []
            
            for tok in tokens:
                # Get mask
                t0 = time.perf_counter()
                mask = state.get_mask_bv()
                mask_times.append((time.perf_counter() - t0) * 1000)
                
                # Commit token
                tok_bytes = enc.decode_single_token_bytes(tok)
                t0 = time.perf_counter()
                state.commit_bytes(tok_bytes)
                commit_times.append((time.perf_counter() - t0) * 1000)
            
            all_mask_times.extend(mask_times)
            all_commit_times.extend(commit_times)
        
        all_tbm = [m + c for m, c in zip(all_mask_times, all_commit_times)]
        
        print(f"    get_mask median: {statistics.median(all_mask_times)*1000:.1f}us")
        print(f"    commit median: {statistics.median(all_commit_times)*1000:.1f}us")
        print(f"    Total TBM median: {statistics.median(all_tbm)*1000:.1f}us")
        print(f"    Total TBM p75: {sorted(all_tbm)[int(len(all_tbm)*0.75)]*1000:.1f}us")
        print(f"    Total TBM p99: {sorted(all_tbm)[int(len(all_tbm)*0.99)]*1000:.1f}us")
        print(f"    Total TBM average: {statistics.mean(all_tbm)*1000:.1f}us")
    
    return {
        'ttfm_min_ms': min(ttfm_times),
        'ttfm_median_ms': statistics.median(ttfm_times),
        'initial_mask_median_us': statistics.median(initial_mask_times) * 1000,
    }


def main():
    print("sep1 Grammar Constraint Benchmark")
    print("=" * 60)
    
    grammars = [
        (".cache/test_vocabs/constraint_json.json.gz", "JSON (RFC 8259)", '{"name": "test", "value": 42}'),
        (".cache/test_vocabs/constraint_js.json.gz", "JavaScript", "function test(x) { return x + 1; }"),
    ]
    
    results = []
    for path, name, test_code in grammars:
        if Path(path).exists():
            result = benchmark_grammar(path, name, test_code)
            result['name'] = name
            results.append(result)
        else:
            print(f"\nSkipping {name}: {path} not found")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Grammar':<25} {'TTFM (ms)':<15} {'TBM (us)':<15}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25} {r['ttfm_min_ms']:<15.2f} {r['initial_mask_median_us']:<15.1f}")


if __name__ == "__main__":
    main()
