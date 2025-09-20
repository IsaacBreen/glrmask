#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to automate running benchmarks for GSS implementations.
#
# This script should be run from the project root (the directory containing 'python').
#
# Usage:
#   ./python/run_benchmarks.sh [implementation1] [implementation2] ...
#
# Example:
#   # Run with default implementations
#   ./python/run_benchmarks.sh
#
#   # Run with specific implementations and a different preset
#   PRESET=medium ./python/run_benchmarks.sh gss_tester.reference_impl:ReferenceGSS gss_tester.leveled_impl:LeveledGSS
#
# Environment Variables:
#   PRESET:  Benchmark preset to use (small, medium, large). Default: small.
#   REPEATS: Number of times to repeat each workload. Default: 1.
# ==============================================================================

# --- Configuration ---
# Use environment variables if set, otherwise use defaults.
: "${PRESET:="small"}"
: "${REPEATS:=1}"

# --- Argument Handling ---
IMPLEMENTATIONS=("$@")

# If no implementations are provided, the benchmark runner will use its own defaults.
if [ "${#IMPLEMENTATIONS[@]}" -eq 0 ]; then
    echo "No implementations provided; the benchmark runner will use its defaults."
fi

# --- Setup ---
# Create a unique directory for this benchmark run's results.
RESULTS_DIR="benchmark_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
OUTPUT_FILE="${RESULTS_DIR}/gss_bench_results.json"

echo "Benchmark results will be saved to: $OUTPUT_FILE"
echo "---"
if [ "${#IMPLEMENTATIONS[@]}" -gt 0 ]; then
  echo "Implementations: ${IMPLEMENTATIONS[*]}"
else
  echo "Implementations: (default)"
fi
echo "Preset: $PRESET"
echo "Repeats: $REPEATS"
echo "---"


# --- Run Benchmarks ---
echo "Starting benchmark run..."

# Construct the command. If IMPLEMENTATIONS is empty, the --implementations flag is omitted
# and the runner will use its own defaults.
cmd=(python -m gss_tester.benchmarks.runner
    --preset "$PRESET"
    --repeats "$REPEATS"
    --output "$OUTPUT_FILE"
)
if [ "${#IMPLEMENTATIONS[@]}" -gt 0 ]; then
    cmd+=(--implementations "${IMPLEMENTATIONS[@]}")
fi

echo "Executing: ${cmd[*]}"
if "${cmd[@]}"; then
    echo
    echo ">>> Benchmark run finished successfully."
else
    exit_code=$?
    echo
    echo ">>> Benchmark run failed with exit code $exit_code."
    exit $exit_code
fi

echo
echo "---"
echo "Benchmark complete."
echo "Results written to $OUTPUT_FILE"
