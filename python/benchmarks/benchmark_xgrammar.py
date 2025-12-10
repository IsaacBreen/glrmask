#!/usr/bin/env python3
"""
XGrammar Benchmark Script

Measures Grammar Compilation Time (GCT) and Time Between Masks (TBM) for XGrammar.

GCT is measured as the FULL end-to-end time:
- Initialize tokenizer info (from HuggingFace tokenizer)
- Create grammar compiler
- Compile grammar/schema

TBM is measured per-token after the initial state is created.

Usage:
    python -m python.benchmarks.benchmark_xgrammar \\
        --grammar src/js.ebnf \\
        --input src/example_code11.js \\
        --output results/xgrammar_js.json \\
        --gct-runs 5 \\
        --tbm-runs 3
        
    # For JSON schemas:
    python -m python.benchmarks.benchmark_xgrammar \\
        --schema schema.json \\
        --output results/xgrammar_schema.json

Requirements:
    pip install xgrammar transformers
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

# Try to import xgrammar
try:
    import xgrammar as xgr
    from transformers import AutoTokenizer
    import torch
    XGRAMMAR_AVAILABLE = True
except ImportError as e:
    XGRAMMAR_AVAILABLE = False
    XGRAMMAR_ERROR = str(e)


def measure_gct(
    grammar_str: Optional[str],
    schema: Optional[dict],
    grammar_type: str,
    tokenizer_name: str = "gpt2",
) -> tuple[float, any, any]:
    """
    Measure FULL Grammar Compilation Time (GCT) for XGrammar.
    
    This includes:
    - Loading/creating tokenizer info
    - Creating grammar compiler  
    - Compiling the grammar/schema
    
    Returns: (gct_seconds, compiled_grammar, tokenizer_info)
    """
    start = time.perf_counter()
    
    # Initialize tokenizer (this is part of the setup XGrammar needs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    
    # Create compiler
    compiler = xgr.GrammarCompiler(tokenizer_info)
    
    # Compile grammar
    if grammar_type == "json_schema" and schema is not None:
        compiled = compiler.compile_json_schema(schema)
    elif grammar_type == "ebnf" and grammar_str is not None:
        compiled = compiler.compile_grammar(grammar_str)
    else:
        raise ValueError(f"Unsupported grammar type: {grammar_type}")
    
    gct = time.perf_counter() - start
    
    return gct, compiled, tokenizer_info


def measure_tbm(
    compiled,
    tokenizer_info,
    token_ids: List[int],
) -> tuple[float, List[float]]:
    """
    Measure Time Between Masks (TBM) for XGrammar.
    
    Returns: (initial_mask_time_us, list of per-token mask times in microseconds)
    """
    # Create matcher
    matcher = xgr.GrammarMatcher(compiled)
    
    # Allocate bitmask once
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    # Measure initial mask
    t0 = time.perf_counter()
    matcher.fill_next_token_bitmask(bitmask)
    initial_mask_us = (time.perf_counter() - t0) * 1e6
    
    # Count initial valid tokens
    logits = torch.zeros(tokenizer_info.vocab_size)
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    
    # Process tokens and measure TBM
    tbm_samples = []
    for token_id in token_ids:
        # Accept the token
        accepted = matcher.accept_token(token_id)
        if not accepted:
            break
        
        # Measure get_mask time
        t0 = time.perf_counter()
        matcher.fill_next_token_bitmask(bitmask)
        tbm_us = (time.perf_counter() - t0) * 1e6
        tbm_samples.append(tbm_us)
    
    return initial_mask_us, tbm_samples


def main():
    parser = argparse.ArgumentParser(description="Benchmark XGrammar grammar-constrained decoding")
    parser.add_argument("--grammar", type=Path, help="Path to grammar file (.ebnf)")
    parser.add_argument("--schema", type=Path, help="Path to JSON schema file")
    parser.add_argument("--input", type=Path, help="Path to input code file to tokenize and process")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--gct-runs", type=int, default=3, help="Number of GCT measurement runs")
    parser.add_argument("--tbm-runs", type=int, default=1, help="Number of TBM measurement runs")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--vocab-cache", type=Path, default=Path(".cache/vocab_cache"), help="Vocab cache directory")
    
    args = parser.parse_args()
    
    if not XGRAMMAR_AVAILABLE:
        print(f"ERROR: XGrammar not available: {XGRAMMAR_ERROR}")
        print("Install with: pip install xgrammar transformers torch")
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
        grammar_type = "ebnf"
        grammar_name = args.grammar.name
        grammar_str = args.grammar.read_text()
        schema = None
    
    # Build result object
    result = BenchmarkResult(
        system_name="xgrammar",
        grammar_name=grammar_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    # Measure GCT
    print(f"Measuring GCT ({args.gct_runs} runs)...")
    compiled = None
    tokenizer_info = None
    
    for i in range(args.gct_runs):
        gct, compiled, tokenizer_info = measure_gct(
            grammar_str=grammar_str,
            schema=schema,
            grammar_type=grammar_type,
            tokenizer_name=args.tokenizer,
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
            initial_mask_us, tbm_samples = measure_tbm(compiled, tokenizer_info, token_ids)
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
