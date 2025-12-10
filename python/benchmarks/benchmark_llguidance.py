#!/usr/bin/env python3
"""
llguidance Benchmark Script

Measures Grammar Compilation Time (GCT) and Time Between Masks (TBM) for llguidance.

GCT is measured as the FULL end-to-end time:
- Initialize tokenizer
- Create grammar compiler
- Compile grammar/schema

TBM is measured per-token after the initial state is created.

Usage:
    python -m python.benchmarks.benchmark_llguidance \\
        --grammar grammar.lark \\
        --input code.txt \\
        --output results/llguidance.json \\
        --gct-runs 5 \\
        --tbm-runs 3
        
    # For JSON schemas:
    python -m python.benchmarks.benchmark_llguidance \\
        --schema schema.json \\
        --output results/llguidance_schema.json

Requirements:
    pip install llguidance tiktoken
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.benchmarks.common import (
    BenchmarkResult,
    load_gpt2_vocab,
    build_id_to_token_bytes,
    greedy_tokenize,
)

# Try to import llguidance
try:
    import llguidance
    from llguidance import JsonCompiler, LarkCompiler, LLInterpreter, LLTokenizer
    import tiktoken
    from llguidance.tiktoken import lltokenizer_from_encoding
    LLGUIDANCE_AVAILABLE = True
except ImportError as e:
    LLGUIDANCE_AVAILABLE = False
    LLGUIDANCE_ERROR = str(e)


def measure_gct(
    grammar_str: Optional[str],
    schema: Optional[dict],
    grammar_type: str,
) -> tuple[float, any, any]:
    """
    Measure FULL Grammar Compilation Time (GCT) for llguidance.
    
    This includes:
    - Creating tokenizer
    - Creating grammar compiler  
    - Compiling the grammar/schema
    
    Returns: (gct_seconds, compiled_grammar, ll_tokenizer)
    """
    start = time.perf_counter()
    
    # Initialize tokenizer (this is part of the setup llguidance needs)
    enc = tiktoken.get_encoding("gpt2")
    ll_tokenizer = lltokenizer_from_encoding(enc)
    
    # Compile grammar
    if grammar_type == "json_schema" and schema is not None:
        compiler = JsonCompiler()
        schema_str = json.dumps(schema)
        compiled = compiler.compile(schema_str)
    elif grammar_type == "lark" and grammar_str is not None:
        compiler = LarkCompiler()
        compiled = compiler.compile(grammar_str)
    else:
        raise ValueError(f"Unsupported grammar type for llguidance: {grammar_type}")
    
    gct = time.perf_counter() - start
    
    return gct, compiled, ll_tokenizer


def measure_tbm(
    compiled,
    ll_tokenizer,
    token_ids: List[int],
) -> tuple[float, List[float]]:
    """
    Measure Time Between Masks (TBM) for llguidance.
    
    Returns: (initial_mask_time_us, list of per-token mask times in microseconds)
    """
    # Create interpreter
    interpreter = LLInterpreter(ll_tokenizer, compiled)
    interpreter.start_without_prompt()
    
    # Measure initial mask
    t0 = time.perf_counter()
    mask_result = interpreter.compute_mask()
    initial_mask_us = (time.perf_counter() - t0) * 1e6
    
    # Process tokens and measure TBM
    tbm_samples = []
    for token_id in token_ids:
        # Commit the token
        try:
            interpreter.commit_token(token_id)
        except Exception as e:
            # Token rejected
            break
        
        # Measure get_mask time
        t0 = time.perf_counter()
        mask_result = interpreter.compute_mask()
        tbm_us = (time.perf_counter() - t0) * 1e6
        tbm_samples.append(tbm_us)
    
    return initial_mask_us, tbm_samples


def parse_mask_bytes(mask_bytes: bytes) -> List[int]:
    """Parse llguidance mask bytes to list of valid token IDs."""
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark llguidance grammar-constrained decoding")
    parser.add_argument("--grammar", type=Path, help="Path to grammar file (.lark)")
    parser.add_argument("--schema", type=Path, help="Path to JSON schema file")
    parser.add_argument("--input", type=Path, help="Path to input code file to tokenize and process")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--gct-runs", type=int, default=3, help="Number of GCT measurement runs")
    parser.add_argument("--tbm-runs", type=int, default=1, help="Number of TBM measurement runs")
    parser.add_argument("--vocab-cache", type=Path, default=Path(".cache/vocab_cache"), help="Vocab cache directory")
    
    args = parser.parse_args()
    
    if not LLGUIDANCE_AVAILABLE:
        print(f"ERROR: llguidance not available: {LLGUIDANCE_ERROR}")
        print("Install with: pip install llguidance tiktoken")
        sys.exit(1)
    
    # Validate arguments
    if not args.grammar and not args.schema:
        parser.error("Either --grammar or --schema must be provided")
    
    # Determine grammar type
    if args.schema:
        grammar_type = "json_schema"
        grammar_name = args.schema.name
        with open(args.schema) as f:
            schema = json.load(f)
        grammar_str = None
    else:
        grammar_type = "lark"
        grammar_name = args.grammar.name
        grammar_str = args.grammar.read_text()
        schema = None
    
    # Build result object
    result = BenchmarkResult(
        system_name="llguidance",
        grammar_name=grammar_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    # Measure GCT
    print(f"Measuring GCT ({args.gct_runs} runs)...")
    compiled = None
    ll_tokenizer = None
    
    for i in range(args.gct_runs):
        gct, compiled, ll_tokenizer = measure_gct(
            grammar_str=grammar_str,
            schema=schema,
            grammar_type=grammar_type,
        )
        result.gct_samples_sec.append(gct)
        print(f"  Run {i+1}: {gct*1000:.1f} ms")
    
    # Measure TBM if input file provided
    if args.input:
        print(f"\nTokenizing input: {args.input}")
        
        # Load vocab for tokenization
        vocab = load_gpt2_vocab(args.vocab_cache)
        id_to_token = build_id_to_token_bytes(vocab)
        
        input_bytes = args.input.read_bytes()
        tokens = greedy_tokenize(input_bytes, id_to_token)
        token_ids = [t[0] for t in tokens]
        print(f"  Tokenized into {len(token_ids)} tokens")
        
        result.input_file = str(args.input)
        result.num_tokens_processed = len(token_ids)
        
        print(f"\nMeasuring TBM ({args.tbm_runs} runs)...")
        all_tbm_samples = []
        all_initial_masks = []
        
        for i in range(args.tbm_runs):
            initial_mask_us, tbm_samples = measure_tbm(compiled, ll_tokenizer, token_ids)
            all_tbm_samples.extend(tbm_samples)
            all_initial_masks.append(initial_mask_us)
            
            median_tbm = sorted(tbm_samples)[len(tbm_samples)//2] if tbm_samples else 0
            print(f"  Run {i+1}: initial={initial_mask_us:.1f}μs, median TBM={median_tbm:.1f}μs")
        
        result.tbm_samples_us = all_tbm_samples
        result.initial_mask_us = sum(all_initial_masks) / len(all_initial_masks) if all_initial_masks else 0
    
    # Compute statistics
    result.compute_statistics()
    
    # Print results
    print("\n" + str(result))
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
