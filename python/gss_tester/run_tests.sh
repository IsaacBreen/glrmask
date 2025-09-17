#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_tests.sh
#
# A script to automate running the GSS test specification against multiple
# implementations, generating JSON results, and analyzing them for consistency.
#
# This script should be run from the `python` directory.
#
# Usage:
#   ./gss_tester/run_tests.sh <reference_impl> <impl1> [impl2 ...]
#
# Example:
#   ./gss_tester/run_tests.sh gss_tester.reference_impl.ReferenceGSS gss_tester.fast_impl.FastGSS
#
# Each implementation is specified as 'path.to.module.ClassName'.
# ==============================================================================

# --- Setup PYTHONPATH to find gss_tester ---
# The script is in 'python/gss_tester', so the python source root is one level up.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${PYTHON_SRC_ROOT}:${PYTHONPATH:-}"

# --- Argument Validation ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <reference_impl> <impl1> [impl2 ...]"
    echo "Error: At least one implementation is required. The first is treated as the reference."
    exit 1
fi

REFERENCE_IMPL="$1"
shift
COMPETITORS=("$@")

ALL_IMPLS=("$REFERENCE_IMPL" "${COMPETITORS[@]}")

# --- Setup ---
# Create a unique directory for this test run's results.
RESULTS_DIR="gss_test_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
echo "Test results will be saved in: $RESULTS_DIR"
echo "---"
echo "Reference Implementation: $REFERENCE_IMPL"
if [ "${#COMPETITORS[@]}" -gt 0 ]; then
  echo "Other Implementations: ${COMPETITORS[*]}"
else
  echo "Other Implementations: (none)"
fi
echo "---"

# --- Run Tests ---
echo "Starting test runs..."
REFERENCE_IMPL_CANONICAL="" # Will be set to the canonical name of the reference impl
for full_impl_path in "${ALL_IMPLS[@]}"; do
    echo
    echo ">>> Running test spec for: $full_impl_path"

    # --- Convert file path to module and class names ---
    # This logic handles both module paths (e.g., gss_tester.ref.RefGSS) and
    # file paths (e.g., python/gss_tester/reference_impl.py).
    if [[ "$full_impl_path" == *.py ]]; then
        # Handle file path: derive module and class name by convention.
        # Module name: strip path up to 'python/', remove '.py' suffix(es), replace '/' with '.'
        module_name=$(echo "$full_impl_path" | sed -e 's#^.*python/##' -e 's#\(\.py\)*$##' -e 's#/#.#g')
        # Class name: e.g., 'reference_impl' -> 'ReferenceGSS'
        # Use sed to robustly strip one or more '.py' suffixes.
        base_name=$(basename "$full_impl_path" | sed 's#\(\.py\)*$##')
        class_name_base=$(echo "$base_name" | sed 's/_impl$//')
        class_name="$(tr '[:lower:]' '[:upper:]' <<< "${class_name_base:0:1}")${class_name_base:1}GSS"
        full_impl_name="${module_name}.${class_name}"
    else
        # Handle module.ClassName format
        full_impl_name="$full_impl_path"
        module_name="${full_impl_name%.*}"
        class_name="${full_impl_name##*.}"
    fi

    # The first implementation is the reference; capture its canonical name.
    if [ -z "$REFERENCE_IMPL_CANONICAL" ]; then
        REFERENCE_IMPL_CANONICAL="$full_impl_name"
    fi

    output_file="${RESULTS_DIR}/${full_impl_name}.json"

    cmd=(python -m gss_tester.runner
        "$module_name"
        "$class_name"
        --output "$output_file")
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
        echo ">>> Finished test spec for: $full_impl_path"
    else
        exit_code=$?
        # Exit code 130 is from SIGINT (Ctrl+C)
        if [ $exit_code -eq 130 ]; then
            echo
            echo ">>> Test for $full_impl_path interrupted. Skipping."
        else
            echo
            echo ">>> Test for $full_impl_path failed with exit code $exit_code. Skipping."
        fi
    fi
done
echo
echo "All test runs completed."
echo "---"

# --- Analyze Results ---
echo "Analyzing results for consistency..."
REFERENCE_RESULT_FILE="${RESULTS_DIR}/${REFERENCE_IMPL_CANONICAL}.json"

# Check if the reference file was actually created
if [ ! -f "$REFERENCE_RESULT_FILE" ]; then
    echo "Error: Reference result file was not generated: $REFERENCE_RESULT_FILE"
    echo "Skipping analysis."
    exit 1
fi

cmd=(python -m gss_tester.analyzer
    "${RESULTS_DIR}"/*.json
    --reference "$REFERENCE_RESULT_FILE")
echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Consistency analysis complete."
echo "Reference: $REFERENCE_IMPL"
echo "Summary printed above."
echo "Full JSON results are in: $RESULTS_DIR"
