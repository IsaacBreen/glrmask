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
for full_impl_name in "${ALL_IMPLS[@]}"; do
    echo
    echo ">>> Running test spec for: $full_impl_name"

    # Extract module and class name. Assumes format is module.path.ClassName
    module_name="${full_impl_name%.*}"
    class_name="${full_impl_name##*.}"
    output_file="${RESULTS_DIR}/${full_impl_name}.json"

    cmd=(python -m gss_tester.runner
        "$module_name"
        "$class_name"
        --output "$output_file")
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
        echo ">>> Finished test spec for: $full_impl_name"
    else
        exit_code=$?
        # Exit code 130 is from SIGINT (Ctrl+C)
        if [ $exit_code -eq 130 ]; then
            echo
            echo ">>> Test for $full_impl_name interrupted. Skipping."
        else
            echo
            echo ">>> Test for $full_impl_name failed with exit code $exit_code. Skipping."
        fi
    fi
done
echo
echo "All test runs completed."
echo "---"

# --- Analyze Results ---
echo "Analyzing results for consistency..."
REFERENCE_RESULT_FILE="${RESULTS_DIR}/${REFERENCE_IMPL}.json"

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
