#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_tests.sh
#
# A script to automate running the GSS test specification against multiple
# implementations, generating JSON results, and analyzing them for consistency.
#
# This script can be run from the `python` directory or the project root.
# It also automates the C++ build process for pybind11 modules.
#
# Usage:
#   # From project root:
#   ./python/gss_tester/run_tests.sh <reference_impl> <impl1> [impl2 ...]
#
#   # From python/ directory:
#   ./gss_tester/run_tests.sh <reference_impl> <impl1> [impl2 ...]
#
# Each implementation is specified as 'path.to.module.ClassName' or a .py path.
# ==============================================================================

# --- Setup PYTHONPATH to find gss_tester ---
# SCRIPT_DIR will be the absolute path to the directory containing this script.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# PYTHON_SRC_ROOT should be the absolute path to the 'python' directory.
# If SCRIPT_DIR is /path/to/project/python/gss_tester, then PYTHON_SRC_ROOT is /path/to/project/python
PYTHON_SRC_ROOT=$(dirname "$SCRIPT_DIR")
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
RESULTS_DIR="$PYTHON_SRC_ROOT/gss_test_results/$(date +"%Y-%m-%d_%H-%M-%S")"
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

# --- Automated C++ Build Process ---
if [ "${SKIP_CPP:-0}" == "1" ]; then
  echo "SKIP_CPP is set. Skipping C++ module build."
  echo "---"
else
  echo "Automating C++ module build..."
  C_PLUS_PLUS_MODULES_DIR="$PYTHON_SRC_ROOT/aug25/models"
  C_PLUS_PLUS_BUILD_DIR="$C_PLUS_PLUS_MODULES_DIR/build"
  C_PLUS_PLUS_OUTPUT_DIR="$PYTHON_SRC_ROOT" # Compiled .so files go directly into python/

  # 1. Clean previous build artifacts
  echo "  Cleaning previous C++ build artifacts..."
  rm -rf "$C_PLUS_PLUS_BUILD_DIR"

  # 2. Configure the build using CMake
  echo "  Configuring C++ build with CMake..."
  cmake -S "$C_PLUS_PLUS_MODULES_DIR" -B "$C_PLUS_PLUS_BUILD_DIR"

  # 3. Build the C++ modules
  echo "  Building C++ modules..."
  cmake --build "$C_PLUS_PLUS_BUILD_DIR"

  # 4. Copy the compiled module to a location where Python can find it
  echo "  Copying compiled C++ modules to Python path..."
  # Find the exact name of the compiled shared library (e.g., leveled_gss_cpp.cpython-312-darwin.so)
  # and copy it to the PYTHON_SRC_ROOT.
  find "$C_PLUS_PLUS_BUILD_DIR" -name "leveled_gss_cpp.*.so" -exec cp {} "$C_PLUS_PLUS_OUTPUT_DIR" \;
  find "$C_PLUS_PLUS_BUILD_DIR" -name "precompute3_engine.*.so" -exec cp {} "$C_PLUS_PLUS_OUTPUT_DIR" \;
  echo "C++ module build complete."
  echo "---"
fi

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
        # The sed command needs to strip the PYTHON_SRC_ROOT prefix if present.
        local_path="${full_impl_path#$PYTHON_SRC_ROOT/}" # Remove PYTHON_SRC_ROOT prefix if it exists
        module_name=$(echo "$local_path" | sed -e 's#\(\.py\)*$##' -e 's#/#.#g')
        # Class name: e.g., 'reference_impl' -> 'ReferenceGSS'
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

    cmd=(python -m gss_tester.tests.runner
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

# Collect all generated JSON files for analysis
ALL_RESULT_FILES=()
while IFS= read -r -d $'\0' file; do
    ALL_RESULT_FILES+=("$file")
done < <(find "$RESULTS_DIR" -maxdepth 1 -name '*.json' -print0)

if [ ${#ALL_RESULT_FILES[@]} -eq 0 ]; then
    echo "No result files found for analysis. Skipping."
    exit 1
fi

cmd=(python -m gss_tester.tests.analyzer
    "${ALL_RESULT_FILES[@]}"
    --reference "$REFERENCE_RESULT_FILE")
echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Consistency analysis complete."
echo "Reference: $REFERENCE_IMPL"
echo "Summary printed above."
echo "Full JSON results are in: $RESULTS_DIR"