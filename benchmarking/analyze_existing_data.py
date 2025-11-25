"""Analyze existing benchmark results and create comprehensive report.

This analyzes the benchmark data that already exists in benchmark_results/
from previous runs of the python/aug25 benchmarking infrastructure.
"""

import json
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BenchmarkAnalysis:
    """Analysis of one benchmark run."""
    model_name: str
    constraint_name: str
    total_tokens: int
    get_mask_mean_ms: float
    get_mask_median_ms: float
    get_mask_p90_ms: float
    get_mask_p99_ms: float
    commit_mean_ms: float
    commit_median_ms: float
    successful: bool


def analyze_result_file(result_file: Path) -> BenchmarkAnalysis:
    """Analyze one result JSON file."""
    with open(result_file) as f:
        data = json.load(f)
    
    results = data.get('results', {})
    
    get_mask_times = results.get('get_mask_timings_seconds', [])
    commit_times = results.get('commit_timings_seconds', [])
    
    # Convert to ms
    get_mask_ms = [t * 1000 for t in get_mask_times]
    commit_ms = [t * 1000 for t in commit_times]
    
    # Calculate statistics
    if get_mask_ms:
        sorted_gm = sorted(get_mask_ms)
        n = len(sorted_gm)
        p90_idx = int(n * 0.90)
        p99_idx = int(n * 0.99)
        
        analysis = BenchmarkAnalysis(
            model_name=data.get('model_script', 'unknown'),
            constraint_name=data.get('constraint_file', 'unknown'),
            total_tokens=results.get('total_input_tokens', len(get_mask_ms)),
            get_mask_mean_ms=statistics.mean(get_mask_ms),
            get_mask_median_ms=statistics.median(get_mask_ms),
            get_mask_p90_ms=sorted_gm[p90_idx] if p90_idx < n else sorted_gm[-1],
            get_mask_p99_ms=sorted_gm[p99_idx] if p99_idx < n else sorted_gm[-1],
            commit_mean_ms=statistics.mean(commit_ms) if commit_ms else 0,
            commit_median_ms=statistics.median(commit_ms) if commit_ms else 0,
            successful=True
        )
    else:
        analysis = BenchmarkAnalysis(
            model_name=data.get('model_script', 'unknown'),
            constraint_name=data.get('constraint_file', 'unknown'),
            total_tokens=0,
            get_mask_mean_ms=0,
            get_mask_median_ms=0,
            get_mask_p90_ms=0,
            get_mask_p99_ms=0,
            commit_mean_ms=0,
            commit_median_ms=0,
            successful=False
        )
    
    return analysis


def main():
    """Analyze all existing benchmark results."""
    
    print("=" * 80)
    print("Existing Benchmark Data Analysis")
    print("=" * 80)
    print()
    
    # Find benchmark result directories
    results_dir = Path("benchmark_results")
    
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found!")
        return 1
    
    # Get all result files
    result_files = list(results_dir.glob("*/*_results.json"))
    
    print(f"Found {len(result_files)} benchmark result files")
    print()
    
    # Analyze each
    all_analyses = []
    
    for result_file in sorted(result_files):
        try:
            analysis = analyze_result_file(result_file)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing {result_file.name}: {e}")
    
    # Group by model type
    rust_models = [a for a in all_analyses if 'rust' in a.model_name.lower() and a.successful]
    python_models = [a for a in all_analyses if 'python' in a.model_name.lower() and a.successful]
    bruteforce_models = [a for a in all_analyses if 'brute' in a.model_name.lower() and a.successful]
    
    # Report
    print("RUST MODELS (Our System)")
    print("-" * 80)
    for a in rust_models:
        print(f"\nModel: {a.model_name}")
        print(f"Constraint: {a.constraint_name}")
        print(f"Tokens: {a.total_tokens}")
        print(f"get_mask - Mean: {a.get_mask_mean_ms:.4f}ms, Median: {a.get_mask_median_ms:.4f}ms")
        print(f"get_mask - P90: {a.get_mask_p90_ms:.4f}ms, P99: {a.get_mask_p99_ms:.4f}ms")
        print(f"commit - Mean: {a.commit_mean_ms:.4f}ms, Median: {a.commit_median_ms:.4f}ms")
    
    print("\n")
    print("PYTHON MODELS (Baseline)")
    print("-" * 80)
    for a in python_models:
        print(f"\nModel: {a.model_name}")
        print(f"get_mask - Mean: {a.get_mask_mean_ms:.4f}ms, Median: {a.get_mask_median_ms:.4f}ms")
    
    print("\n")
    print("BRUTEFORCE MODELS (Comparison)")
    print("-" * 80)
    for a in bruteforce_models:
        print(f"\nModel: {a.model_name}")
        print(f"get_mask - Mean: {a.get_mask_mean_ms:.4f}ms, Median: {a.get_mask_median_ms:.4f}ms")
    
    # Calculate speedups
    if rust_models and bruteforce_models:
        rust_avg = statistics.mean([a.get_mask_mean_ms for a in rust_models])
        brute_avg = statistics.mean([a.get_mask_mean_ms for a in bruteforce_models])
        speedup = brute_avg / rust_avg if rust_avg > 0 else 0
        
        print("\n")
        print("SPEEDUP ANALYSIS")
        print("-" * 80)
        print(f"Rust average: {rust_avg:.4f}ms ({rust_avg*1000:.2f}µs)")
        print(f"Bruteforce average: {brute_avg:.4f}ms")
        print(f"Speedup: {speedup:.1f}x")
    
    # Save summary
    summary_file = Path("benchmarking/results/existing_data_analysis.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'rust_models': [vars(a) for a in rust_models],
        'python_models': [vars(a) for a in python_models],
        'bruteforce_models': [vars(a) for a in bruteforce_models]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
