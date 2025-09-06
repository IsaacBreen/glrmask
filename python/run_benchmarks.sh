#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to automate running benchmarks for multiple grammar constraint models,
# comparing them against a reference implementation, and generating analysis plots.
#
# This script should be run from the `python` directory.
#
# Usage:
#   ./run_benchmarks.sh <reference_model.py> <competitor1.py> [competitor2.py ...]
#
# Example:
#   ./run_benchmarks.sh aug25/precompute2_model.py aug25/precompute3_model.py
#
# Environment Variables:
#   CONSTRAINT_FILE: Path to the pre-compiled .json.gz constraint file.
#                    (Default: ../.cache/test_vocabs/js_grammar_constraint.json.gz)
#   CODE_FILE:    Path to the code file to use as input.
#                 (Default: ../src/example_code.js)
# ==============================================================================

# --- Configuration ---
# Use environment variables if set, otherwise use defaults.
: "${CONSTRAINT_FILE:="../.cache/test_vocabs/js_grammar_constraint.json.gz"}"
: "${CODE_FILE:="../src/example_code.js"}"

# --- Argument Validation ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <reference_model.py> <competitor1.py> [competitor2.py ...]"
    echo "Error: At least two arguments are required: a reference model and one competitor model."
    exit 1
fi

REFERENCE_MODEL="$1"
shift
COMPETITORS=("$@")

# Check that all provided files exist
for file in "$REFERENCE_MODEL" "${COMPETITORS[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Model file not found: $file"
        exit 1
    fi
done
if [ ! -f "$CONSTRAINT_FILE" ]; then
    echo "Error: Constraint file not found: $CONSTRAINT_FILE"
    exit 1
fi
if [ ! -f "$CODE_FILE" ]; then
    echo "Error: Code file not found: $CODE_FILE"
    exit 1
fi


# --- Setup ---
# Create a unique directory for this benchmark run's results.
RESULTS_DIR="benchmark_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
echo "Benchmark results will be saved in: $RESULTS_DIR"
echo "---"
echo "Reference Model: $REFERENCE_MODEL"
echo "Competitors: ${COMPETITORS[*]}"
echo "Constraint File: $CONSTRAINT_FILE"
echo "Code: $CODE_FILE"
echo "---"


# --- Run Benchmarks ---
ALL_MODELS=("$REFERENCE_MODEL" "${COMPETITORS[@]}")

echo "Starting benchmark runs..."
for model_to_benchmark in "${ALL_MODELS[@]}"; do
    echo
    echo ">>> Running benchmark for: $(basename "$model_to_benchmark")"
    python -m aug25.benchmark_runner \
        --code "$CODE_FILE" \
        --constraint-file "$CONSTRAINT_FILE" \
        --reference "$REFERENCE_MODEL" \
        --competitor "$model_to_benchmark" \
        --output "$RESULTS_DIR"
    echo ">>> Finished benchmark for: $(basename "$model_to_benchmark")"
done
echo
echo "All benchmark runs completed."
echo "---"


# --- Analyze Results ---
echo "Analyzing results and generating plots..."
PLOTS_DIR="${RESULTS_DIR}/analysis"
python -m aug25.benchmark_analyzer \
    "${RESULTS_DIR}"/*.json \
    --output-dir "$PLOTS_DIR"

echo
echo "---"
echo "Benchmark analysis complete."
echo "Summary printed above. Plots are saved in: $PLOTS_DIR"
echo "Full JSON results are in: $RESULTS_DIR"
