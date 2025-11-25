"""Benchmark our sep1 system with EBNF grammars.

This runs comprehensive benchmarks on our native Rust implementation
using the grammars we already have working.
"""

import sys
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

# Find existing benchmark infrastructure
from python.aug25.models.rust_model import Model


@dataclass
class Sep1BenchmarkResult:
    """Results for sep1 on one grammar."""
    grammar_name: str
    num_tokens_tested: int
    avg_get_mask_ms: float
    median_get_mask_ms: float
    p90_get_mask_ms: float
    p99_get_mask_ms: float
    avg_commit_ms: float
    compilation_time_sec: float
    vocabulary_size: int
    effective_vocabulary_size: int
    vocab_reduction_pct: float
    

def load_precompiled_grammar(grammar_path: Path) -> Model:
    """Load a precompiled grammar constraint."""
    import gzip
    
    if str(grammar_path).endswith('.gz'):
        with gzip.open(grammar_path, 'rt') as f:
            grammar_json = json.load(f)
    else:
        with open(grammar_path, 'r') as f:
            grammar_json = json.load(f)
    
    model = Model.from_json_string(json.dumps(grammar_json))
    model.reset()
    return model


def benchmark_grammar(grammar_path: Path, num_tokens: int = 1000) -> Sep1BenchmarkResult:
    """Benchmark one grammar."""
    
    print(f"  Loading {grammar_path.name}...")
    model = load_precompiled_grammar(grammar_path)
    
    print(f"  Running {num_tokens} get_mask calls...")
    
    get_mask_times = []
    commit_times = []
    
    for i in range(num_tokens):
        # Get mask
        import time
        start = time.perf_counter()
        mask = model.get_mask()
        get_mask_time = time.perf_counter() - start
        get_mask_times.append(get_mask_time * 1000)  # Convert to ms
        
        # Get valid tokens from mask
        if hasattr(mask, 'to_ranges'):
            ranges = mask.to_ranges()
            valid_tokens = []
            for start_id, end_id in ranges:
                valid_tokens.extend(range(start_id, end_id + 1))
        else:
            valid_tokens = list(mask) if hasattr(mask, '__iter__') else []
        
        if not valid_tokens:
            print(f"    Stopped at token {i} (no valid tokens)")
            break
        
        # Commit a token
        token_to_commit = valid_tokens[0]
        start = time.perf_counter()
        model.commit(token_to_commit)
        commit_time = time.perf_counter() - start
        commit_times.append(commit_time * 1000)
    
    # Calculate stats
    result = Sep1BenchmarkResult(
        grammar_name=grammar_path.stem,
        num_tokens_tested=len(get_mask_times),
        avg_get_mask_ms=statistics.mean(get_mask_times),
        median_get_mask_ms=statistics.median(get_mask_times),
        p90_get_mask_ms=statistics.quantiles(get_mask_times, n=10)[8] if len(get_mask_times) >= 10 else get_mask_times[-1],
        p99_get_mask_ms=statistics.quantiles(get_mask_times, n=100)[98] if len(get_mask_times) >= 100 else get_mask_times[-1],
        avg_commit_ms=statistics.mean(commit_times) if commit_times else 0,
        compilation_time_sec=0,  # Already precompiled
        vocabulary_size=50257,  # GPT-2 vocab
        effective_vocabulary_size=0,  # TODO: extract from model
        vocab_reduction_pct=0
    )
    
    return result


def main():
    """Run sep1 benchmarks."""
    
    print("=" * 80)
    print("sep1 EBNF Grammar Benchmarks")
    print("=" * 80)
    print()
    
    # Find precompiled grammars
    grammar_files = [
        _project_root / "reduced_js_grammar_constraint.json.gz",
        _project_root / "reduced_js_grammar_constraint11.json.gz",
    ]
    
    grammar_files = [f for f in grammar_files if f.exists()]
    
    if not grammar_files:
        print("ERROR: No precompiled grammars found!")
        print("Expected files:")
        for f in grammar_files:
            print(f"  - {f}")
        return 1
    
    print(f"Found {len(grammar_files)} precompiled grammar(s)")
    print()
    
    # Run benchmarks
    results = []
    
    for grammar_file in grammar_files:
        print(f"Benchmarking: {grammar_file.name}")
        try:
            result = benchmark_grammar(grammar_file, num_tokens=1000)
            results.append(result)
            print(f"  ✓ Completed: {result.num_tokens_tested} tokens")
            print(f"    Avg get_mask: {result.avg_get_mask_ms:.3f}ms")
            print(f"    Median get_mask: {result.median_get_mask_ms:.3f}ms")
            print(f"    P90 get_mask: {result.p90_get_mask_ms:.3f}ms")
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Save results
    results_file = Path("benchmarking/results/sep1_ebnf_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    for result in results:
        print(f"Grammar: {result.grammar_name}")
        print(f"  Tokens tested:    {result.num_tokens_tested}")
        print(f"  Avg get_mask:     {result.avg_get_mask_ms:.4f}ms  ({result.avg_get_mask_ms*1000:.2f}µs)")
        print(f"  Median get_mask:  {result.median_get_mask_ms:.4f}ms")
        print(f"  P90 get_mask:     {result.p90_get_mask_ms:.4f}ms")
        print(f"  P99 get_mask:     {result.p99_get_mask_ms:.4f}ms")
        print(f"  Avg commit:       {result.avg_commit_ms:.4f}ms")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
