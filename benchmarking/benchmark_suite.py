#!/usr/bin/env python3
"""
Comprehensive benchmark runner for grammar constraint systems.

This script provides fair comparisons by:
1. Using the same grammars across all systems (where possible)
2. Measuring TBM at comparable states
3. Reporting compilation + runtime separately
"""

import sys
import time
import json
import gzip
import statistics
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# Test Grammars in Different Formats
# ==============================================================================

# Simple JSON Grammar (for fair comparison)
JSON_EBNF_GRAMMAR = r'''
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
array ::= "[" ws (value ("," ws value)*)? ws "]"
pair ::= string ws ":" ws value
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ("0" | [1-9] [0-9]*)
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
'''

JSON_SCHEMA_SIMPLE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["name", "value"]
}

JSON_SCHEMA_COMPLEX = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "email": {"type": "string", "format": "email"},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "country": {"type": "string"},
                "zipcode": {"type": "string", "pattern": "^[0-9]{5}$"}
            },
            "required": ["street", "city"]
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10
        },
        "active": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 100}
    },
    "required": ["id", "name"]
}

# GBNF format for llama.cpp
JSON_GBNF_GRAMMAR = r'''
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
array ::= "[" ws (value ("," ws value)*)? ws "]"
pair ::= string ws ":" ws value
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ("0" | [1-9] digit*)
digit ::= [0-9]
frac ::= "." digit+
exp ::= [eE] [+-]? digit+
ws ::= [ \t\n\r]*
'''


# ==============================================================================
# Benchmark Result
# ==============================================================================

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result."""
    system: str
    grammar: str
    grammar_type: str
    
    # Compilation
    compile_time_ms: float = 0.0
    
    # TTFM (Time to First Mask)
    ttfm_ms: float = 0.0
    
    # TBM (Time Between Masks) statistics
    tbm_median_us: float = 0.0
    tbm_mean_us: float = 0.0
    tbm_min_us: float = 0.0
    tbm_max_us: float = 0.0
    tbm_p75_us: float = 0.0
    tbm_p95_us: float = 0.0
    tbm_p99_us: float = 0.0
    tbm_stddev_us: float = 0.0
    
    # Validity
    num_valid_tokens: int = 0
    
    # Error (if any)
    error: str = ""
    
    # Raw samples for detailed analysis
    tbm_samples: List[float] = field(default_factory=list)


# ==============================================================================
# System Wrappers
# ==============================================================================

class XGrammarBenchmark:
    """XGrammar benchmark wrapper."""
    
    def __init__(self):
        import xgrammar as xgr
        import torch
        from transformers import AutoTokenizer
        
        self.xgr = xgr
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = len(self.tokenizer)
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.vocab_size
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        
    @property
    def name(self):
        return "xgrammar"
    
    def benchmark_json_schema(self, schema: dict, n_iter: int = 100) -> BenchmarkResult:
        result = BenchmarkResult(
            system=self.name,
            grammar="json_schema",
            grammar_type="json_schema"
        )
        
        try:
            schema_str = json.dumps(schema)
            
            # Compile
            compile_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = self.compiler.compile_json_schema(schema)
                compile_times.append((time.perf_counter() - t0) * 1000)
            result.compile_time_ms = min(compile_times)
            
            # TTFM
            ttfm_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = self.compiler.compile_json_schema(schema)
                matcher = self.xgr.GrammarMatcher(compiled)
                bitmask = self.xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
                matcher.fill_next_token_bitmask(bitmask)
                ttfm_times.append((time.perf_counter() - t0) * 1000)
            result.ttfm_ms = min(ttfm_times)
            
            # TBM measurement
            compiled = self.compiler.compile_json_schema(schema)
            tbm_samples = []
            valid_tokens = None
            
            for _ in range(n_iter):
                matcher = self.xgr.GrammarMatcher(compiled)
                bitmask = self.xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
                
                t0 = time.perf_counter()
                matcher.fill_next_token_bitmask(bitmask)
                tbm_samples.append((time.perf_counter() - t0) * 1e6)
                
                if valid_tokens is None:
                    logits = self.torch.zeros(self.tokenizer_info.vocab_size)
                    self.xgr.apply_token_bitmask_inplace(logits, bitmask)
                    valid_tokens = (logits > -float('inf')).nonzero(as_tuple=True)[0].tolist()
            
            result.tbm_samples = tbm_samples
            self._compute_stats(result, tbm_samples)
            result.num_valid_tokens = len(valid_tokens) if valid_tokens else 0
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def benchmark_ebnf(self, grammar: str, n_iter: int = 100) -> BenchmarkResult:
        result = BenchmarkResult(
            system=self.name,
            grammar="ebnf",
            grammar_type="ebnf"
        )
        
        try:
            # Compile
            compile_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = self.compiler.compile_grammar(grammar)
                compile_times.append((time.perf_counter() - t0) * 1000)
            result.compile_time_ms = min(compile_times)
            
            # TTFM
            ttfm_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = self.compiler.compile_grammar(grammar)
                matcher = self.xgr.GrammarMatcher(compiled)
                bitmask = self.xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
                matcher.fill_next_token_bitmask(bitmask)
                ttfm_times.append((time.perf_counter() - t0) * 1000)
            result.ttfm_ms = min(ttfm_times)
            
            # TBM
            compiled = self.compiler.compile_grammar(grammar)
            tbm_samples = []
            valid_tokens = None
            
            for _ in range(n_iter):
                matcher = self.xgr.GrammarMatcher(compiled)
                bitmask = self.xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
                
                t0 = time.perf_counter()
                matcher.fill_next_token_bitmask(bitmask)
                tbm_samples.append((time.perf_counter() - t0) * 1e6)
                
                if valid_tokens is None:
                    logits = self.torch.zeros(self.tokenizer_info.vocab_size)
                    self.xgr.apply_token_bitmask_inplace(logits, bitmask)
                    valid_tokens = (logits > -float('inf')).nonzero(as_tuple=True)[0].tolist()
            
            result.tbm_samples = tbm_samples
            self._compute_stats(result, tbm_samples)
            result.num_valid_tokens = len(valid_tokens) if valid_tokens else 0
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def _compute_stats(self, result: BenchmarkResult, samples: List[float]):
        if not samples:
            return
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        result.tbm_median_us = statistics.median(samples)
        result.tbm_mean_us = statistics.mean(samples)
        result.tbm_min_us = min(samples)
        result.tbm_max_us = max(samples)
        result.tbm_stddev_us = statistics.stdev(samples) if len(samples) > 1 else 0
        result.tbm_p75_us = sorted_samples[int(n * 0.75)]
        result.tbm_p95_us = sorted_samples[int(n * 0.95)]
        result.tbm_p99_us = sorted_samples[min(int(n * 0.99), n - 1)]


class LLGuidanceBenchmark:
    """LLGuidance benchmark wrapper."""
    
    def __init__(self):
        import llguidance
        from llguidance import JsonCompiler, LLInterpreter
        import tiktoken
        from llguidance.tiktoken import lltokenizer_from_encoding
        
        self.llguidance = llguidance
        self.JsonCompiler = JsonCompiler
        self.LLInterpreter = LLInterpreter
        
        self.enc = tiktoken.get_encoding("gpt2")
        self.ll_tokenizer = lltokenizer_from_encoding(self.enc)
        
    @property
    def name(self):
        return "llguidance"
    
    def benchmark_json_schema(self, schema: dict, n_iter: int = 100) -> BenchmarkResult:
        result = BenchmarkResult(
            system=self.name,
            grammar="json_schema",
            grammar_type="json_schema"
        )
        
        try:
            schema_str = json.dumps(schema)
            compiler = self.JsonCompiler()
            
            # Compile
            compile_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = compiler.compile(schema_str)
                compile_times.append((time.perf_counter() - t0) * 1000)
            result.compile_time_ms = min(compile_times)
            
            # TTFM
            ttfm_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compiled = compiler.compile(schema_str)
                interp = self.LLInterpreter(self.ll_tokenizer, compiled)
                interp.start_without_prompt()
                mask = interp.compute_mask()
                ttfm_times.append((time.perf_counter() - t0) * 1000)
            result.ttfm_ms = min(ttfm_times)
            
            # TBM
            compiled = compiler.compile(schema_str)
            tbm_samples = []
            valid_tokens = None
            
            for _ in range(n_iter):
                interp = self.LLInterpreter(self.ll_tokenizer, compiled)
                interp.start_without_prompt()
                
                t0 = time.perf_counter()
                mask = interp.compute_mask()
                tbm_samples.append((time.perf_counter() - t0) * 1e6)
                
                if valid_tokens is None:
                    valid_tokens = self._parse_mask(mask[0])
            
            result.tbm_samples = tbm_samples
            self._compute_stats(result, tbm_samples)
            result.num_valid_tokens = len(valid_tokens) if valid_tokens else 0
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def _parse_mask(self, mask_bytes):
        """Parse mask bytes to list of valid token IDs."""
        valid_tokens = []
        if isinstance(mask_bytes, bytes):
            for byte_idx, byte_val in enumerate(mask_bytes):
                if byte_val == 0:
                    continue
                for bit_idx in range(8):
                    if (byte_val >> bit_idx) & 1:
                        token_id = byte_idx * 8 + bit_idx
                        valid_tokens.append(token_id)
        return valid_tokens
    
    def _compute_stats(self, result: BenchmarkResult, samples: List[float]):
        if not samples:
            return
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        result.tbm_median_us = statistics.median(samples)
        result.tbm_mean_us = statistics.mean(samples)
        result.tbm_min_us = min(samples)
        result.tbm_max_us = max(samples)
        result.tbm_stddev_us = statistics.stdev(samples) if len(samples) > 1 else 0
        result.tbm_p75_us = sorted_samples[int(n * 0.75)]
        result.tbm_p95_us = sorted_samples[int(n * 0.95)]
        result.tbm_p99_us = sorted_samples[min(int(n * 0.99), n - 1)]


class Sep1Benchmark:
    """Sep1 benchmark wrapper."""
    
    def __init__(self):
        import _sep1
        import tiktoken
        
        self._sep1 = _sep1
        self.enc = tiktoken.get_encoding("gpt2")
        
    @property
    def name(self):
        return "sep1"
    
    def benchmark_precompiled(self, constraint_path: Path, n_iter: int = 100) -> BenchmarkResult:
        result = BenchmarkResult(
            system=self.name,
            grammar=constraint_path.stem,
            grammar_type="precompiled"
        )
        
        try:
            # Load constraint
            if str(constraint_path).endswith('.gz'):
                with gzip.open(constraint_path, 'rt') as f:
                    json_str = f.read()
            else:
                json_str = constraint_path.read_text()
            
            # "Compile" time = loading time
            compile_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                constraint = self._sep1.GrammarConstraint.from_json_string(json_str)
                compile_times.append((time.perf_counter() - t0) * 1000)
            result.compile_time_ms = min(compile_times)
            
            # TTFM
            ttfm_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                constraint = self._sep1.GrammarConstraint.from_json_string(json_str)
                state = self._sep1.GrammarConstraintState(constraint)
                mask = state.get_mask_bv()
                ttfm_times.append((time.perf_counter() - t0) * 1000)
            result.ttfm_ms = min(ttfm_times)
            
            # TBM
            constraint = self._sep1.GrammarConstraint.from_json_string(json_str)
            tbm_samples = []
            valid_tokens = None
            
            for _ in range(n_iter):
                state = self._sep1.GrammarConstraintState(constraint)
                
                t0 = time.perf_counter()
                mask = state.get_mask_bv()
                tbm_samples.append((time.perf_counter() - t0) * 1e6)
                
                if valid_tokens is None:
                    valid_tokens = list(mask.to_indices())
            
            result.tbm_samples = tbm_samples
            self._compute_stats(result, tbm_samples)
            result.num_valid_tokens = len(valid_tokens) if valid_tokens else 0
            
        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def _compute_stats(self, result: BenchmarkResult, samples: List[float]):
        if not samples:
            return
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        result.tbm_median_us = statistics.median(samples)
        result.tbm_mean_us = statistics.mean(samples)
        result.tbm_min_us = min(samples)
        result.tbm_max_us = max(samples)
        result.tbm_stddev_us = statistics.stdev(samples) if len(samples) > 1 else 0
        result.tbm_p75_us = sorted_samples[int(n * 0.75)]
        result.tbm_p95_us = sorted_samples[int(n * 0.95)]
        result.tbm_p99_us = sorted_samples[min(int(n * 0.99), n - 1)]


class OutlinesBenchmark:
    """Outlines benchmark wrapper using CFGGuide."""
    
    def __init__(self):
        from outlines.fsm.guide import CFGGuide
        from outlines.models.transformers import TransformerTokenizer
        from transformers import AutoTokenizer
        
        self.CFGGuide = CFGGuide
        self.hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer = TransformerTokenizer(self.hf_tokenizer)
        
    @property
    def name(self):
        return "outlines"
    
    def benchmark_lark_grammar(self, grammar: str, grammar_name: str, n_iter: int = 100) -> BenchmarkResult:
        """Benchmark Outlines on a Lark grammar."""
        result = BenchmarkResult(
            system=self.name,
            grammar=grammar_name,
            grammar_type="lark"
        )
        
        try:
            import warnings
            
            # Compile (this is the expensive part in Outlines)
            compile_times = []
            for _ in range(3):  # Fewer iterations due to slow compilation
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    guide = self.CFGGuide(grammar, self.tokenizer)
                compile_times.append((time.perf_counter() - t0) * 1000)
            result.compile_time_ms = min(compile_times)
            
            # TTFM
            ttfm_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    guide = self.CFGGuide(grammar, self.tokenizer)
                instruction = guide.get_next_instruction(guide.initial_state)
                ttfm_times.append((time.perf_counter() - t0) * 1000)
            result.ttfm_ms = min(ttfm_times)
            
            # TBM - REUSE the compiled guide
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                guide = self.CFGGuide(grammar, self.tokenizer)
            
            tbm_samples = []
            valid_tokens = None
            initial_state = guide.initial_state
            
            for _ in range(n_iter):
                t0 = time.perf_counter()
                instruction = guide.get_next_instruction(initial_state)
                tbm_samples.append((time.perf_counter() - t0) * 1e6)
                
                if valid_tokens is None:
                    if hasattr(instruction, 'tokens'):
                        valid_tokens = instruction.tokens.tolist()
                    else:
                        valid_tokens = []
            
            result.tbm_samples = tbm_samples
            self._compute_stats(result, tbm_samples)
            result.num_valid_tokens = len(valid_tokens) if valid_tokens else 0
            
        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def _compute_stats(self, result: BenchmarkResult, samples: List[float]):
        if not samples:
            return
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        result.tbm_median_us = statistics.median(samples)
        result.tbm_mean_us = statistics.mean(samples)
        result.tbm_min_us = min(samples)
        result.tbm_max_us = max(samples)
        result.tbm_stddev_us = statistics.stdev(samples) if len(samples) > 1 else 0
        result.tbm_p75_us = sorted_samples[int(n * 0.75)]
        result.tbm_p95_us = sorted_samples[int(n * 0.95)]
        result.tbm_p99_us = sorted_samples[min(int(n * 0.99), n - 1)]


# ==============================================================================
# Main Runner
# ==============================================================================

def print_results_table(results: List[BenchmarkResult]):
    """Print results in formatted table."""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    
    header = (f"{'System':<12} | {'Grammar':<20} | {'Compile(ms)':<11} | {'TTFM(ms)':<9} | "
              f"{'TBM-p50(μs)':<11} | {'TBM-p99(μs)':<11} | {'TBM-max(μs)':<11} | "
              f"{'Valid Tokens':<12}")
    print(header)
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: (x.grammar, x.tbm_median_us)):
        if r.error:
            print(f"{r.system:<12} | {r.grammar:<20} | ERROR: {r.error[:70]}")
        else:
            print(f"{r.system:<12} | {r.grammar:<20} | {r.compile_time_ms:<11.2f} | "
                  f"{r.ttfm_ms:<9.2f} | {r.tbm_median_us:<11.1f} | {r.tbm_p99_us:<11.1f} | "
                  f"{r.tbm_max_us:<11.1f} | {r.num_valid_tokens:<12}")
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Grammar Constraint Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=100, help="TBM iterations")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--sep1-js", type=Path, 
                        default=Path(".cache/test_vocabs/constraint_js.json.gz"),
                        help="Sep1 JS constraint file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Grammar Constraint Benchmark Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print("=" * 80)
    
    results = []
    
    # Initialize systems
    print("\n--- Initializing Systems ---")
    
    try:
        xgr = XGrammarBenchmark()
        print("✓ XGrammar initialized")
    except Exception as e:
        print(f"✗ XGrammar failed: {e}")
        xgr = None
    
    try:
        llg = LLGuidanceBenchmark()
        print("✓ LLGuidance initialized")
    except Exception as e:
        print(f"✗ LLGuidance failed: {e}")
        llg = None
    
    try:
        sep1 = Sep1Benchmark()
        print("✓ Sep1 initialized")
    except Exception as e:
        print(f"✗ Sep1 failed: {e}")
        sep1 = None
    
    try:
        outlines = OutlinesBenchmark()
        print("✓ Outlines initialized")
    except Exception as e:
        print(f"✗ Outlines failed: {e}")
        outlines = None
    
    # Run benchmarks
    print("\n--- Running JSON Schema Benchmarks ---")
    
    for schema, name in [(JSON_SCHEMA_SIMPLE, "simple_json"), 
                          (JSON_SCHEMA_COMPLEX, "complex_json")]:
        if xgr:
            print(f"  XGrammar on {name}...")
            r = xgr.benchmark_json_schema(schema, args.iterations)
            r.grammar = name
            results.append(r)
        
        if llg:
            print(f"  LLGuidance on {name}...")
            r = llg.benchmark_json_schema(schema, args.iterations)
            r.grammar = name
            results.append(r)
    
    print("\n--- Running EBNF Benchmarks ---")
    
    if xgr:
        print("  XGrammar on json_ebnf...")
        r = xgr.benchmark_ebnf(JSON_EBNF_GRAMMAR, args.iterations)
        r.grammar = "json_ebnf"
        results.append(r)
    
    print("\n--- Running Sep1 Benchmarks ---")
    
    if sep1 and args.sep1_js.exists():
        print(f"  Sep1 on {args.sep1_js.name}...")
        r = sep1.benchmark_precompiled(args.sep1_js, args.iterations)
        results.append(r)
    
    print("\n--- Running Outlines Benchmarks ---")
    
    # Outlines uses Lark grammar format
    LARK_JSON_GRAMMAR = r'''
?start: value
value: object | array | string | number | "true" | "false" | "null"
object: "{" [pair ("," pair)*] "}"
array: "[" [value ("," value)*] "]"
pair: string ":" value
string: /"[^"]*"/
number: /-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/
'''
    
    if outlines:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("  Outlines on json_lark...")
            r = outlines.benchmark_lark_grammar(LARK_JSON_GRAMMAR, "json_lark", args.iterations)
            results.append(r)
    
    # Print results
    print_results_table(results)
    
    # Compute comparison metrics
    print("\n--- Performance Comparison ---")
    
    # Group by grammar type
    json_results = [r for r in results if "json" in r.grammar.lower() and not r.error]
    
    if json_results:
        print("\nJSON/JSON-Schema Performance:")
        fastest = min(json_results, key=lambda x: x.tbm_median_us)
        print(f"  Fastest: {fastest.system} at {fastest.tbm_median_us:.1f}μs median")
        
        for r in json_results:
            if r.system != fastest.system:
                ratio = r.tbm_median_us / fastest.tbm_median_us
                print(f"  {r.system}: {r.tbm_median_us:.1f}μs ({ratio:.1f}× slower)")
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "iterations": args.iterations
            },
            "results": [asdict(r) for r in results]
        }
        # Remove tbm_samples from output to keep file small
        for r in output_data["results"]:
            r.pop("tbm_samples", None)
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
