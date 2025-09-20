#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to automate running GSS benchmarks against multiple implementations.
#
# This script should be run from the `python` directory.
#
# Usage:
#   ./gss_tester/run_benchmarks.sh [--preset small|medium|large] [--repeats N] <impl1> [impl2 ...]
#
# Example:
#   ./gss_tester/run_benchmarks.sh --preset medium gss_tester.reference_impl:ReferenceGSS gss_tester.leveled_impl:LeveledGSS
#
# Each implementation is specified as 'path.to.module:ClassName'.
# ==============================================================================

# --- Setup PYTHONPATH to find gss_tester ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${PYTHON_SRC_ROOT}:${PYTHONPATH:-}"

# --- Argument Parsing ---
PRESET="small"
REPEATS=1
RAW_IMPLS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            RAW_IMPLS+=("$1")
            shift
            ;;
    esac
done

if [ "${#RAW_IMPLS[@]}" -eq 0 ]; then
    echo "Usage: $0 [--preset small|medium|large] [--repeats N] <impl1> [impl2 ...]"
    echo "Error: At least one implementation is required."
    exit 1
fi

# --- Implementation Path Parsing ---
# Convert file paths (e.g., gss_tester/leveled_impl.py) to module:class format.
PARSED_IMPLS=()
for impl_path in "${RAW_IMPLS[@]}"; do
    if [[ "$impl_path" == *.py ]]; then
        # Handle file path: derive module and class name by convention.
        # Module name: strip path up to 'python/', remove '.py' suffix, replace '/' with '.'
        module_name=$(echo "$impl_path" | sed -e 's#^.*python/##' -e 's/\.py$//' -e 's#/#.#g')
        # Class name: e.g., 'leveled_impl' -> 'LeveledGSS'
        base_name=$(basename "$impl_path" .py)
        class_name_base=$(echo "$base_name" | sed 's/_impl$//')
        class_name="$(tr '[:lower:]' '[:upper:]' <<< "${class_name_base:0:1}")${class_name_base:1}GSS"
        # The benchmark runner prefers module:Class format
        PARSED_IMPLS+=("${module_name}:${class_name}")
    else
        # Assume it's already in module:Class or module.Class format
        PARSED_IMPLS+=("$impl_path")
    fi
done


# --- Setup ---
RESULTS_DIR="gss_bench_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
OUTPUT_FILE="${RESULTS_DIR}/results.json"

echo "Benchmark results will be saved in: $OUTPUT_FILE"
echo "---"
echo "Implementations: ${PARSED_IMPLS[*]}"
echo "Preset: $PRESET"
echo "Repeats: $REPEATS"
echo "---"

# --- Run Benchmarks ---
echo "Starting benchmark run..."

cmd=(python -m gss_tester.benchmarks.runner
    --implementations "${PARSED_IMPLS[@]}"
    --preset "$PRESET"
    --repeats "$REPEATS"
    --output "$OUTPUT_FILE"
)

echo "${cmd[*]}"
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
echo "Full JSON results are in: $OUTPUT_FILE"
