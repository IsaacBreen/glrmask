"""
Benchmark suite for GSS implementations.

This package provides:
- workloads: definitions of benchmark workloads that stress different aspects of GSS performance,
  especially scaling and structural sharing across merges.
- runner: a CLI that runs selected workloads against a given implementation and emits JSON.
- analysis: a CLI that reads one or more runner JSON outputs and prints comparisons.
- plotting: optional plotting helpers used by the analyzer when matplotlib is available.
"""
