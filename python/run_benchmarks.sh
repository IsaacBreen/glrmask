#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to automate running benchmarks for multiple grammar constraint models,
# generating JSON results and analysis plots. This script runs each model once
# (no in-process reference/baseline). The analyzer is later told which result
# is the baseline to perform mask comparisons.
#
# This script should be run from the `python` directory.
#
# Usage:
#   ./run_benchmarks.sh <baseline_model.py> <model1.py> [model2.py ...]
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
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <baseline_model.py> <model1.py> [model2.py ...]"
    echo "Error: At least one model argument is required. The first is treated as the baseline."
    exit 1
fi

BASELINE_MODEL="$1"
shift
COMPETITORS=("$@")

ALL_MODELS=("$BASELINE_MODEL" "${COMPETITORS[@]}")

# Check that all provided files exist
for file in "${ALL_MODELS[@]}"; do
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
echo "Baseline Model: $BASELINE_MODEL"
if [ "${#COMPETITORS[@]}" -gt 0 ]; then
  echo "Other Models: ${COMPETITORS[*]}"
else
  echo "Other Models: (none)"
fi
echo "Constraint File: $CONSTRAINT_FILE"
echo "Code: $CODE_FILE"
echo "---"


# --- Run Benchmarks ---
echo "Starting benchmark runs..."
for model_to_benchmark in "${ALL_MODELS[@]}"; do
    echo
    echo ">>> Running benchmark for: $(basename "$model_to_benchmark")"
    cmd=(python -m aug25.benchmark_runner
        --code "$CODE_FILE"
        --constraint-file "$CONSTRAINT_FILE"
        --model "$model_to_benchmark"
        --output "$RESULTS_DIR")
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
        echo ">>> Finished benchmark for: $(basename "$model_to_benchmark")"
    else
        exit_code=$?
        # Exit code 130 is from SIGINT (Ctrl+C)
        if [ $exit_code -eq 130 ]; then
            echo
            echo ">>> Benchmark for $(basename "$model_to_benchmark") interrupted. Skipping."
        else
            echo
            echo ">>> Benchmark for $(basename "$model_to_benchmark") failed with exit code $exit_code. Skipping."
        fi
    fi
done
echo
echo "All benchmark runs completed."
echo "---"


# --- Analyze Results ---
echo "Analyzing results and generating plots..."
PLOTS_DIR="${RESULTS_DIR}/analysis"
BASELINE_STEM="$(basename "$BASELINE_MODEL" .py)"
cmd=(python -m aug25.benchmark_analyzer
    "${RESULTS_DIR}"/*.json
    --baseline "$BASELINE_STEM"
    --output-dir "$PLOTS_DIR")
echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Benchmark analysis complete."
echo "Baseline: $BASELINE_STEM"
echo "Summary printed above. Plots are saved in: $PLOTS_DIR"
echo "Full JSON results are in: $RESULTS_DIR"
