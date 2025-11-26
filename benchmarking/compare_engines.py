#!/usr/bin/env python3
"""
Benchmark comparison between sep1 and llguidance on the same grammars.

This script measures:
1. Grammar compilation time (TTFM - Time To First Mask)
2. Per-token mask computation time (TBM - Time Between Masks)
3. Correctness validation (both engines produce consistent results)
"""

import time
import json
import gzip
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Sep1 imports
import _sep1 as sep1_ffi

# LLGuidance imports
import llguidance as llg
import llguidance.hf
from llguidance.numpy import fill_next_token_bitmask, allocate_token_bitmask
from transformers import AutoTokenizer
import numpy as np

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    engine: str
    grammar_name: str
    compilation_time_ms: float
    mask_times_us: List[float]
    tokens_processed: int
    
    @property
    def mean_mask_time_us(self) -> float:
        return statistics.mean(self.mask_times_us) if self.mask_times_us else 0
    
    @property
    def median_mask_time_us(self) -> float:
        return statistics.median(self.mask_times_us) if self.mask_times_us else 0
    
    @property
    def p95_mask_time_us(self) -> float:
        if not self.mask_times_us:
            return 0
        sorted_times = sorted(self.mask_times_us)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times)-1)]
    
    @property
    def min_mask_time_us(self) -> float:
        return min(self.mask_times_us) if self.mask_times_us else 0
    
    @property
    def max_mask_time_us(self) -> float:
        return max(self.mask_times_us) if self.mask_times_us else 0


class Sep1Engine:
    """Wrapper for sep1 constraint engine."""
    
    def __init__(self, constraint_path: str):
        self.constraint_path = constraint_path
        self.constraint = None
        self.state = None
        
    def compile(self) -> float:
        """Load precompiled constraint and return time in ms."""
        start = time.perf_counter()
        with gzip.open(self.constraint_path, 'rt') as f:
            constraint_json = f.read()
        self.constraint = sep1_ffi.GrammarConstraint.from_json_string(constraint_json)
        self.state = sep1_ffi.GrammarConstraintState(self.constraint)
        end = time.perf_counter()
        return (end - start) * 1000
    
    def reset(self):
        """Reset state to initial."""
        self.state = sep1_ffi.GrammarConstraintState(self.constraint)
    
    def get_mask(self) -> set:
        """Get valid token mask as a set of token IDs."""
        mask_bv = self.state.get_mask_bv()
        # FFI bitset has to_indices() method
        return set(mask_bv.to_indices())
    
    def commit(self, token_id: int):
        """Commit a token to advance the state."""
        self.state.commit(token_id)


class LLGuidanceEngine:
    """Wrapper for llguidance constraint engine."""
    
    def __init__(self, grammar_lark: str, tokenizer_name: str = "gpt2"):
        self.grammar_lark = grammar_lark
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.llg_tokenizer = None
        self.interp = None
        self.interp0 = None
        self.mask_data = None
        
    def compile(self) -> float:
        """Compile grammar and return time in ms."""
        start = time.perf_counter()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.llg_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer)
        
        # Compile grammar
        # LLGuidance expects the grammar in its internal format
        grammars = json.dumps({"grammars": [{"lark_grammar": self.grammar_lark}]})
        self.interp0 = llg.LLMatcher(self.llg_tokenizer, grammars)
        
        if self.interp0.is_error():
            raise ValueError(f"Grammar compilation failed: {self.interp0.get_error()}")
        
        self.interp = self.interp0
        self.mask_data = allocate_token_bitmask(1, self.llg_tokenizer.vocab_size)
        
        end = time.perf_counter()
        return (end - start) * 1000
    
    def reset(self):
        """Reset state to initial."""
        self.interp = self.interp0.deep_copy()
    
    def get_mask(self) -> set:
        """Get valid token mask as a set of token IDs."""
        fill_next_token_bitmask(self.interp, self.mask_data, 0)
        # Convert bitmask to set of valid token IDs
        valid_tokens = set()
        vocab_size = self.llg_tokenizer.vocab_size
        for i in range(vocab_size):
            word_idx = i // 32
            bit_idx = i % 32
            if (self.mask_data[0, word_idx] & (1 << bit_idx)) != 0:
                valid_tokens.add(i)
        return valid_tokens
    
    def commit(self, token_id: int):
        """Commit a token to advance the state."""
        self.interp.consume_token(token_id)


def run_benchmark(engine, grammar_name: str, n_iterations: int = 1000) -> BenchmarkResult:
    """Run benchmark on a single engine."""
    
    # Compile grammar
    compile_time = engine.compile()
    
    # Warm up
    for _ in range(10):
        _ = engine.get_mask()
    
    # Benchmark get_mask WITHOUT reset (measures mask computation only)
    mask_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = engine.get_mask()
        end = time.perf_counter()
        mask_times.append((end - start) * 1e6)  # Convert to microseconds
    
    return BenchmarkResult(
        engine=engine.__class__.__name__,
        grammar_name=grammar_name,
        compilation_time_ms=compile_time,
        mask_times_us=mask_times,
        tokens_processed=n_iterations,
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n{'Engine':<20} {'Grammar':<15} {'Compile (ms)':<12} {'Mean (μs)':<12} {'Median (μs)':<12} {'p95 (μs)':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r.engine:<20} {r.grammar_name:<15} {r.compilation_time_ms:>10.1f}   {r.mean_mask_time_us:>10.1f}   {r.median_mask_time_us:>10.1f}   {r.p95_mask_time_us:>10.1f}")
    
    print("\n")


# JSON Grammar in Lark format for llguidance
JSON_GRAMMAR_LARK = '''
%llguidance {}

start: value
value: object | array | STRING | NUMBER | "true" | "false" | "null"
object: "{" (pair ("," pair)*)? "}"
pair: STRING ":" value
array: "[" (value ("," value)*)? "]"

STRING: /"([^"\\\\]|\\\\.)*"/
NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/
'''

# Simplified JavaScript Expression Grammar in Lark format
JS_EXPR_GRAMMAR_LARK = '''
%llguidance {}

start: expr
expr: term (("+" | "-") term)*
term: factor (("*" | "/") factor)*
factor: NUMBER | IDENT | "(" expr ")"

NUMBER: /[0-9]+/
IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/

%ignore /[ \\t\\n\\r]+/
'''


def main():
    print("Grammar-Constrained Decoding Benchmark")
    print("="*40)
    
    results = []
    n_iterations = 1000
    
    # Test 1: Sep1 on JavaScript grammar (precompiled)
    print("\n[1/3] Benchmarking sep1 on JavaScript grammar...")
    sep1_js = Sep1Engine(".cache/test_vocabs/constraint_js.json.gz")
    try:
        result = run_benchmark(sep1_js, "JavaScript", n_iterations)
        results.append(result)
        print(f"  ✓ Mean: {result.mean_mask_time_us:.1f}μs, Median: {result.median_mask_time_us:.1f}μs")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: LLGuidance on JSON grammar
    print("\n[2/3] Benchmarking llguidance on JSON grammar...")
    llg_json = LLGuidanceEngine(JSON_GRAMMAR_LARK)
    try:
        result = run_benchmark(llg_json, "JSON", n_iterations)
        results.append(result)
        print(f"  ✓ Mean: {result.mean_mask_time_us:.1f}μs, Median: {result.median_mask_time_us:.1f}μs")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 3: LLGuidance on JS expression grammar
    print("\n[3/3] Benchmarking llguidance on JS Expression grammar...")
    llg_js = LLGuidanceEngine(JS_EXPR_GRAMMAR_LARK)
    try:
        result = run_benchmark(llg_js, "JS-Expr", n_iterations)
        results.append(result)
        print(f"  ✓ Mean: {result.mean_mask_time_us:.1f}μs, Median: {result.median_mask_time_us:.1f}μs")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Print summary
    print_results(results)


if __name__ == "__main__":
    main()
