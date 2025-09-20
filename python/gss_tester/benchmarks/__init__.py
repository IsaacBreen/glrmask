"""
GSS Benchmarks Package

This package provides a comprehensive, deterministic benchmarking harness for
Graph-Structured Stack (GSS) implementations that implement the interface
defined in gss_tester.interface.GSS.

Key features:
- Multiple workloads that stress different aspects of a GSS (push-heavy, branching,
  pop/merge collapse, wide merges, apply/prune, and seeded fuzz-like constructions).
- Scaling controls to evaluate asymptotic behavior as the structure grows.
- Timing and memory measurement using Python's standard library (time, tracemalloc).
- Structural metrics derived from to_stacks() and optional implementation-specific
  introspection (e.g., internal node counts for LeveledGSS).
- JSON output suitable for further analysis or visualization.

Run:
    python -m gss_tester.benchmarks.runner --help

Examples:
    # Quick run on bundled implementations
    python -m gss_tester.benchmarks.runner

    # Compare specific implementations with a medium preset
    python -m gss_tester.benchmarks.runner \
        --implementations gss_tester.reference_impl:ReferenceGSS gss_tester.leveled_impl:LeveledGSS \
        --preset medium \
        --output bench_results.json
"""
