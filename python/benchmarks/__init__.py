# Benchmarking infrastructure for grammar-constrained decoding systems
#
# This module provides separate benchmark scripts for each system:
# - benchmark_sep1.py: Our system (sep1)
# - benchmark_xgrammar.py: XGrammar
# - benchmark_llguidance.py: llguidance
#
# Each script measures:
# - GCT (Grammar Compilation Time): end-to-end from vocab + grammar to constraint
# - TBM (Time Between Masks): per-token mask computation time
#
# Output format is unified JSON with p50, p99 statistics for both metrics.
