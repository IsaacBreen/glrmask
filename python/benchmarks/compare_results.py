#!/usr/bin/env python3
"""
Compare benchmark results across systems.

Takes JSON results from each system's benchmark script and produces
a comparison table.

Usage:
    python -m python.benchmarks.compare_results \\
        results/sep1_js.json \\
        results/xgrammar_js.json \\
        results/llguidance_js.json \\
        --output comparison.txt
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.benchmarks.common import BenchmarkResult


def load_results(paths: List[Path]) -> List[BenchmarkResult]:
    """Load benchmark results from JSON files."""
    results = []
    for path in paths:
        result = BenchmarkResult.from_json(path)
        results.append(result)
    return results


def format_time(seconds: float, unit: str = "auto") -> str:
    """Format time with appropriate units."""
    if unit == "auto":
        if seconds >= 1:
            return f"{seconds:.2f}s"
        elif seconds >= 0.001:
            return f"{seconds*1000:.1f}ms"
        else:
            return f"{seconds*1e6:.1f}μs"
    elif unit == "ms":
        return f"{seconds*1000:.1f}ms"
    elif unit == "us":
        return f"{seconds*1e6:.1f}μs"
    elif unit == "s":
        return f"{seconds:.2f}s"
    else:
        return str(seconds)


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a comparison table of results."""
    # Header
    systems = [r.system_name for r in results]
    header = ["Metric"] + systems
    
    # GCT rows
    gct_p50 = ["GCT p50"] + [format_time(r.gct_p50_sec) for r in results]
    gct_p99 = ["GCT p99"] + [format_time(r.gct_p99_sec) for r in results]
    
    # TBM rows
    tbm_p50 = ["TBM p50"] + [f"{r.tbm_p50_us:.1f}μs" for r in results]
    tbm_p99 = ["TBM p99"] + [f"{r.tbm_p99_us:.1f}μs" for r in results]
    tbm_mean = ["TBM mean"] + [f"{r.tbm_mean_us:.1f}μs" for r in results]
    
    # Initial mask
    initial = ["Initial mask"] + [f"{r.initial_mask_us:.1f}μs" for r in results]
    
    # Format table
    rows = [header, [], gct_p50, gct_p99, [], tbm_p50, tbm_p99, tbm_mean, [], initial]
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) if row and i < len(row) else 0 
                     for row in rows) 
                 for i in range(len(header))]
    
    # Print
    print("\n" + "=" * (sum(col_widths) + 3 * len(header)))
    print("BENCHMARK COMPARISON")
    print("=" * (sum(col_widths) + 3 * len(header)))
    
    for row in rows:
        if not row:
            print("-" * (sum(col_widths) + 3 * len(header)))
        else:
            cells = [str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]
            print(" | ".join(cells))
    
    print("=" * (sum(col_widths) + 3 * len(header)))
    
    # Print speedup analysis if we have sep1 and others
    sep1_result = next((r for r in results if r.system_name == "sep1"), None)
    if sep1_result:
        print("\nSpeedup vs sep1:")
        for r in results:
            if r.system_name != "sep1" and r.tbm_p50_us > 0:
                speedup = r.tbm_p50_us / sep1_result.tbm_p50_us
                print(f"  {r.system_name}: {speedup:.1f}x slower (TBM p50)")


def generate_latex_table(results: List[BenchmarkResult], grammar_name: str) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{l|" + "r" * len(results) + "}",
        r"\toprule",
    ]
    
    # Header
    systems = [r.system_name for r in results]
    header = " & ".join(["Metric"] + [f"\\textbf{{{s}}}" for s in systems]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # GCT
    gct_row = " & ".join(["GCT"] + [format_time(r.gct_p50_sec) for r in results]) + r" \\"
    lines.append(gct_row)
    
    # TBM p50
    tbm_p50_row = " & ".join(["TBM p50"] + [f"{r.tbm_p50_us:.0f}$\\mu$s" for r in results]) + r" \\"
    lines.append(tbm_p50_row)
    
    # TBM p99
    tbm_p99_row = " & ".join(["TBM p99"] + [f"{r.tbm_p99_us:.0f}$\\mu$s" for r in results]) + r" \\"
    lines.append(tbm_p99_row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Benchmark results on {grammar_name}.}}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("results", nargs="+", type=Path, help="JSON result files to compare")
    parser.add_argument("--output", type=Path, help="Output file for comparison")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    if not results:
        print("No results to compare")
        sys.exit(1)
    
    # Print comparison
    print_comparison_table(results)
    
    # Generate LaTeX if requested
    if args.latex:
        grammar_name = results[0].grammar_name if results else "grammar"
        latex = generate_latex_table(results, grammar_name)
        print("\nLaTeX Table:")
        print(latex)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                f.write(str(r))
                f.write("\n\n")


if __name__ == "__main__":
    main()
