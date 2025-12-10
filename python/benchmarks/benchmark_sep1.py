#!/usr/bin/env python3
"""
Sep1 Benchmark Script

Measures Grammar Compilation Time (GCT) and Time Between Masks (TBM) for sep1.

GCT is measured as the FULL end-to-end time:
- Load vocabulary
- Parse grammar  
- Build GLR parser tables
- Construct terminal characterizations
- Build and determinize NWA
- Serialize to JSON constraint

TBM is measured per-token after the initial state is created.

Usage:
    python -m python.benchmarks.benchmark_sep1 \\
        --grammar src/js.ebnf \\
        --input src/example_code11.js \\
        --output results/sep1_js.json \\
        --gct-runs 5 \\
        --tbm-runs 3

Environment variables:
    MACRO_DEBUG_LEVEL: Set to 0 to disable debug output (default)
"""

import argparse
import sys
import os
import time
import json
import gzip
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Disable debug output by default for cleaner benchmarks
if "MACRO_DEBUG_LEVEL" not in os.environ:
    os.environ["MACRO_DEBUG_LEVEL"] = "0"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.benchmarks.common import (
    BenchmarkResult,
    load_gpt2_vocab,
    build_id_to_token_bytes,
    greedy_tokenize,
    Timer,
)


def measure_gct_full(
    grammar_path: Path,
    vocab: dict[str, int],
    build_profile: str = "release",
) -> float:
    """
    Measure FULL Grammar Compilation Time (GCT).
    
    This measures the time from having:
    - Grammar file on disk
    - Vocabulary in memory
    
    To:
    - Having a ready-to-use constraint JSON string
    
    This is the TRUE end-to-end compilation cost.
    """
    # Write vocab to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_path = Path(f.name)
    
    # Output to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Build the compiler if needed
        compiler_path = PROJECT_ROOT / "target" / build_profile / "grammar-compiler"
        
        # Start timing EVERYTHING: build + compilation
        start_time = time.perf_counter()
        
        # Build the compiler (this is part of the setup cost for sep1)
        # In a production setting, this would be pre-built, so we don't include it
        # Instead, we assume the compiler is already built
        if not compiler_path.exists():
            subprocess.run(
                ["cargo", "build", "--release", "-q"],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
        
        # Now time just the compilation
        compile_start = time.perf_counter()
        
        result = subprocess.run(
            [
                str(compiler_path),
                "--grammar", str(grammar_path),
                "--vocab", str(vocab_path),
                "--output", str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "ENABLE_PROGRESS_BAR": "0", "MACRO_DEBUG_LEVEL": "0"},
        )
        
        # Read the output to ensure it's complete
        if output_path.suffix == '.gz':
            with gzip.open(output_path, 'rt') as f:
                constraint_str = f.read()
        else:
            constraint_str = output_path.read_text()
        
        compile_time = time.perf_counter() - compile_start
        
        return compile_time
        
    finally:
        # Cleanup temp files
        vocab_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def measure_gct_from_constraint_metadata(constraint_path: Path) -> Optional[float]:
    """
    Read the compilation time that was recorded DURING constraint generation.
    
    This is useful for getting the exact time that was measured when the
    constraint was originally compiled.
    """
    if str(constraint_path).endswith('.gz'):
        with gzip.open(constraint_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(constraint_path) as f:
            data = json.load(f)
    
    return data.get('compilation_time_seconds')


def compile_grammar_to_constraint(
    grammar_path: Path,
    vocab: dict[str, int],
    output_path: Path,
    build_profile: str = "release",
) -> str:
    """
    Compile a grammar to a constraint JSON file and return the JSON string.
    """
    # Write vocab to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_path = Path(f.name)
    
    try:
        compiler_path = PROJECT_ROOT / "target" / build_profile / "grammar-compiler"
        
        # Build the compiler if needed
        if not compiler_path.exists():
            subprocess.run(
                ["cargo", "build", "--release", "-q"],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
        
        result = subprocess.run(
            [
                str(compiler_path),
                "--grammar", str(grammar_path),
                "--vocab", str(vocab_path),
                "--output", str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "ENABLE_PROGRESS_BAR": "0", "MACRO_DEBUG_LEVEL": "0"},
        )
        
        # Read the output
        if str(output_path).endswith('.gz'):
            with gzip.open(output_path, 'rt') as f:
                return f.read()
        else:
            return output_path.read_text()
        
    finally:
        vocab_path.unlink(missing_ok=True)


def measure_tbm(
    constraint_json: str,
    token_ids: list[int],
    vocab: dict[str, int],
) -> tuple[float, list[float]]:
    """
    Measure Time Between Masks (TBM) for sep1.
    
    Returns: (initial_mask_time_us, list of per-token mask times in microseconds)
    """
    # Import the Rust model
    from python.aug25.models.rust_model import Model as RustModel
    
    # Create model from JSON
    model = RustModel.from_json_string(constraint_json)
    
    # Build id -> bytes mapping for commit
    id_to_bytes = {}
    for token_str, token_id in vocab.items():
        try:
            from python.benchmarks.common import gpt2_token_str_to_bytes
            id_to_bytes[token_id] = gpt2_token_str_to_bytes(token_str)
        except KeyError:
            pass
    
    # Measure initial mask (no tokens committed yet)
    t0 = time.perf_counter()
    initial_mask = model.get_mask()
    initial_mask_time_us = (time.perf_counter() - t0) * 1e6
    
    # Process tokens and measure TBM
    tbm_samples = []
    for token_id in token_ids:
        # Commit the token
        model.commit(token_id)
        
        # Measure get_mask time
        t0 = time.perf_counter()
        mask = model.get_mask()
        tbm_us = (time.perf_counter() - t0) * 1e6
        tbm_samples.append(tbm_us)
        
        # Check for empty mask (end of valid input)
        if not mask.to_ranges():
            break
    
    return initial_mask_time_us, tbm_samples


def main():
    parser = argparse.ArgumentParser(description="Benchmark sep1 grammar-constrained decoding")
    parser.add_argument("--grammar", type=Path, required=True, help="Path to grammar file (.ebnf)")
    parser.add_argument("--input", type=Path, help="Path to input code file to tokenize and process")
    parser.add_argument("--constraint", type=Path, help="Path to pre-compiled constraint (skips GCT measurement)")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--gct-runs", type=int, default=3, help="Number of GCT measurement runs")
    parser.add_argument("--tbm-runs", type=int, default=1, help="Number of TBM measurement runs")
    parser.add_argument("--vocab-cache", type=Path, default=Path(".cache/vocab_cache"), help="Vocab cache directory")
    
    args = parser.parse_args()
    
    # Load vocabulary
    print("Loading GPT-2 vocabulary...")
    vocab = load_gpt2_vocab(args.vocab_cache)
    print(f"  Loaded {len(vocab)} tokens")
    
    # Build result object
    result = BenchmarkResult(
        system_name="sep1",
        grammar_name=args.grammar.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    # Measure GCT if not using pre-compiled constraint
    if args.constraint:
        print(f"Using pre-compiled constraint: {args.constraint}")
        # Read GCT from constraint metadata
        gct = measure_gct_from_constraint_metadata(args.constraint)
        if gct:
            result.gct_samples_sec = [gct]
            print(f"  GCT from metadata: {gct*1000:.1f} ms")
        
        # Load constraint JSON
        if str(args.constraint).endswith('.gz'):
            with gzip.open(args.constraint, 'rt') as f:
                constraint_json = f.read()
        else:
            constraint_json = args.constraint.read_text()
    else:
        print(f"\nMeasuring GCT ({args.gct_runs} runs)...")
        for i in range(args.gct_runs):
            gct = measure_gct_full(args.grammar, vocab)
            result.gct_samples_sec.append(gct)
            print(f"  Run {i+1}: {gct*1000:.1f} ms")
        
        # Compile one more time to get the constraint for TBM measurement
        print("\nCompiling constraint for TBM measurement...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        constraint_json = compile_grammar_to_constraint(args.grammar, vocab, output_path)
        output_path.unlink(missing_ok=True)
    
    # Measure TBM if input file provided
    if args.input:
        print(f"\nTokenizing input: {args.input}")
        input_bytes = args.input.read_bytes()
        id_to_token = build_id_to_token_bytes(vocab)
        tokens = greedy_tokenize(input_bytes, id_to_token)
        token_ids = [t[0] for t in tokens]
        print(f"  Tokenized into {len(token_ids)} tokens")
        
        result.input_file = str(args.input)
        result.num_tokens_processed = len(token_ids)
        
        print(f"\nMeasuring TBM ({args.tbm_runs} runs)...")
        all_tbm_samples = []
        all_initial_masks = []
        
        for i in range(args.tbm_runs):
            initial_mask_us, tbm_samples = measure_tbm(constraint_json, token_ids, vocab)
            all_tbm_samples.extend(tbm_samples)
            all_initial_masks.append(initial_mask_us)
            print(f"  Run {i+1}: initial={initial_mask_us:.1f}μs, median TBM={sorted(tbm_samples)[len(tbm_samples)//2]:.1f}μs")
        
        result.tbm_samples_us = all_tbm_samples
        result.initial_mask_us = sum(all_initial_masks) / len(all_initial_masks)
    
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
