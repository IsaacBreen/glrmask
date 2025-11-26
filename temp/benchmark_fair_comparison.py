#!/usr/bin/env python3
"""
Fair comparison benchmark: sep1 vs llguidance on the same machine.

This script runs both systems on the same JavaScript grammar and measures
TTFM and TBM to enable a fair comparison.
"""
import time
import statistics
import gzip
import numpy as np
from pathlib import Path

# Check if both libraries are available
try:
    import _sep1
    HAS_SEP1 = True
except ImportError:
    HAS_SEP1 = False
    print("Warning: _sep1 not available")

try:
    import llguidance
    from llguidance import LarkCompiler, LLInterpreter, LLTokenizer
    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False
    print("Warning: llguidance not available")

import tiktoken


def benchmark_sep1(constraint_path: str, test_tokens: list, n_trials: int = 100):
    """Benchmark sep1 system."""
    print("\n=== Sep1 Benchmark ===")
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Load constraint
    with gzip.open(constraint_path, "rt") as f:
        json_str = f.read()
    
    # TTFM: Time to First Mask (load + create + first mask)
    ttfm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        constraint = _sep1.GrammarConstraint.from_json_string(json_str)
        state = _sep1.GrammarConstraintState(constraint)
        mask = state.get_mask_bv()
        ttfm_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"TTFM: {min(ttfm_times):.2f}ms (min of 5)")
    
    # TBM: Time Between Masks - initial state
    constraint = _sep1.GrammarConstraint.from_json_string(json_str)
    
    initial_mask_times = []
    for _ in range(n_trials):
        state = _sep1.GrammarConstraintState(constraint)
        t0 = time.perf_counter()
        mask = state.get_mask_bv()
        initial_mask_times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    
    print(f"TBM (initial state):")
    print(f"  median: {statistics.median(initial_mask_times):.1f}μs")
    print(f"  p75: {sorted(initial_mask_times)[int(n_trials*0.75)]:.1f}μs")
    print(f"  p99: {sorted(initial_mask_times)[int(n_trials*0.99)]:.1f}μs")
    
    # TBM during generation
    if test_tokens:
        all_mask_times = []
        for _ in range(5):  # 5 trials
            state = _sep1.GrammarConstraintState(constraint)
            for tok in test_tokens:
                t0 = time.perf_counter()
                mask = state.get_mask_bv()
                all_mask_times.append((time.perf_counter() - t0) * 1e6)
                
                # Commit token
                tok_bytes = enc.decode_single_token_bytes(tok)
                state.commit_bytes(tok_bytes)
        
        print(f"TBM (during generation, {len(test_tokens)} tokens):")
        print(f"  median: {statistics.median(all_mask_times):.1f}μs")
        print(f"  average: {statistics.mean(all_mask_times):.1f}μs")
    
    return {
        'name': 'sep1',
        'ttfm_ms': min(ttfm_times),
        'tbm_initial_us': statistics.median(initial_mask_times),
        'tbm_gen_us': statistics.median(all_mask_times) if test_tokens else None
    }


def benchmark_llguidance_json(n_trials: int = 100):
    """Benchmark llguidance on JSON grammar."""
    print("\n=== LLGuidance Benchmark (JSON) ===")
    
    import json
    
    # Create a simple JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        },
        "required": ["name", "value"]
    }
    
    # Create tokenizer (GPT-2)
    enc = tiktoken.get_encoding("gpt2")
    
    # Get vocab info for llguidance tokenizer
    # llguidance needs a tokenizer wrapper
    try:
        from llguidance.tiktoken import lltokenizer_from_encoding
        tok = lltokenizer_from_encoding(enc)
    except Exception as e:
        print(f"Error creating tokenizer: {e}")
        return None
    
    # Compile JSON schema
    compiler = llguidance.JsonCompiler()
    
    ttfm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        grammar = compiler.compile(json.dumps(schema))
        interp = LLInterpreter(tok, grammar)
        interp.start_without_prompt()
        mask = interp.compute_mask()
        ttfm_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"TTFM (compile + first mask): {min(ttfm_times):.2f}ms (min of 5)")
    
    # TBM: Time Between Masks
    grammar = compiler.compile(json.dumps(schema))
    
    initial_mask_times = []
    for _ in range(n_trials):
        interp = LLInterpreter(tok, grammar)
        interp.start_without_prompt()
        
        t0 = time.perf_counter()
        mask = interp.compute_mask()
        initial_mask_times.append((time.perf_counter() - t0) * 1e6)
    
    print(f"TBM (initial state):")
    print(f"  median: {statistics.median(initial_mask_times):.1f}μs")
    print(f"  p75: {sorted(initial_mask_times)[int(n_trials*0.75)]:.1f}μs")
    print(f"  p99: {sorted(initial_mask_times)[int(n_trials*0.99)]:.1f}μs")
    print(f"  average: {statistics.mean(initial_mask_times):.1f}μs")
    
    return {
        'name': 'llguidance (JSON)',
        'ttfm_ms': min(ttfm_times),
        'tbm_initial_us': statistics.median(initial_mask_times),
        'tbm_avg_us': statistics.mean(initial_mask_times),
        'tbm_gen_us': None
    }


def main():
    print("=" * 60)
    print("Fair Comparison Benchmark: sep1 vs llguidance")
    print("=" * 60)
    print(f"Hardware: Same machine, same conditions")
    
    # Test code
    test_code = "function test(x) { return x + 1; }"
    test_json = '{"name": "test", "value": 42}'
    enc = tiktoken.get_encoding("gpt2")
    test_tokens_js = enc.encode(test_code)
    test_tokens_json = enc.encode(test_json)
    
    print(f"\nJS Test code: {test_code}")
    print(f"JS Tokens: {len(test_tokens_js)}")
    print(f"JSON Test: {test_json}")
    print(f"JSON Tokens: {len(test_tokens_json)}")
    
    results = []
    
    # Sep1 JavaScript benchmark
    if HAS_SEP1:
        constraint_path = ".cache/test_vocabs/constraint_js.json.gz"
        if Path(constraint_path).exists():
            result = benchmark_sep1(constraint_path, test_tokens_js)
            if result:
                result['name'] = 'sep1 (JS)'
                results.append(result)
        else:
            print(f"Sep1 JS constraint file not found: {constraint_path}")
    
    # Sep1 JSON benchmark
    if HAS_SEP1:
        constraint_path = ".cache/test_vocabs/constraint_json.json.gz"
        if Path(constraint_path).exists():
            result = benchmark_sep1(constraint_path, test_tokens_json)
            if result:
                result['name'] = 'sep1 (JSON)'
                results.append(result)
        else:
            print(f"Sep1 JSON constraint file not found: {constraint_path}")
    
    # LLGuidance benchmark (JSON since that's their primary use case)
    if HAS_LLGUIDANCE:
        result = benchmark_llguidance_json()
        if result:
            results.append(result)
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'System':<20} {'TTFM (ms)':<12} {'TBM Initial (μs)':<18} {'TBM Avg (μs)'}")
        print("-" * 65)
        for r in results:
            ttfm = f"{r['ttfm_ms']:.2f}"
            tbm_init = f"{r['tbm_initial_us']:.1f}"
            tbm_avg = f"{r.get('tbm_avg_us', r.get('tbm_gen_us', 'N/A'))}"
            if isinstance(tbm_avg, float):
                tbm_avg = f"{tbm_avg:.1f}"
            print(f"{r['name']:<20} {ttfm:<12} {tbm_init:<18} {tbm_avg}")


if __name__ == "__main__":
    main()
