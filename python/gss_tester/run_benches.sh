#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benches.sh
#
# Orchestrates running GSS benchmark workloads across one or more implementations,
# saving JSON results, and producing comparative analyses and plots.
#
# This script can be run from the `python` directory or the project root.
# It also automates the C++ build process for pybind11 modules.
#
# Usage (Preset Mode):
#   ./gss_tester/run_benches.sh <preset> <impl1> [impl2 ...] [-- [runner_args]]
#
# Usage (Sweep Mode for scaling analysis):
#   ./gss_tester/run_benches.sh <preset> <impl> -- --sweep-workload <name> ...
#
# Notes:
#   - Each implementation is specified as 'module.ClassName' or a .py path.
#   - All arguments after the implementation list (or after a '--') are passed
#     directly to the benchmark runner script.
#
# Examples:
#   # Run 'small' preset workloads AND all predefined 'small' sweeps
#   ./gss_tester/run_benches.sh small gss_tester.implementations.reference_impl.ReferenceGSS gss_tester.fast_impl.FastGSS
#
#   # Run workloads with 'push_scaling' in their name (substring match)
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS -- --include push_scaling
#
#   # Run only the workload named exactly 'push_scaling'
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS -- --only push_scaling
#
#   # Run a single manual scaling sweep (does not run other preset workloads/sweeps)
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS -- --sweep-workload push_scaling --sweep-axis prefix_depth --sweep-values 10 50 100 200
# ==============================================================================

# --- Setup PYTHONPATH to find gss_tester ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="${PYTHON_SRC_ROOT}:${PYTHONPATH:-}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <preset> <impl1> [impl2 ...] [-- runner_args...]"
  exit 1
fi

PRESET="$1"
shift

# Collect implementations until we see a flag starting with '--' or we run out.
IMPLS=()
EXTRA_ARGS=()
while (( "$#" )); do
  if [[ "$1" == "--" ]]; then
    shift # Consume the '--'
    EXTRA_ARGS=("$@")
    break
  elif [[ "$1" == --* ]]; then
    EXTRA_ARGS=("$@")
    break
  else
    IMPLS+=("$1")
  fi
  shift
done

if [ "${#IMPLS[@]}" -eq 0 ]; then
  echo "Error: No implementations provided."
  exit 1
fi

# Check if the user is asking for a manual sweep or providing workload filters.
is_manual_sweep=false
has_filtering_args=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--sweep-workload" ]]; then
    is_manual_sweep=true
  fi
  if [[ "$arg" == "--only" || "$arg" == "--include" || "$arg" == "--exclude" ]]; then
    has_filtering_args=true
  fi
done

RESULTS_DIR="$PYTHON_SRC_ROOT/gss_bench_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
echo "Benchmark results will be saved in: $RESULTS_DIR"
echo "---"
echo "Preset: $PRESET"
echo "Implementations: ${IMPLS[*]}"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "Extra args to runner: ${EXTRA_ARGS[*]}"
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

  echo "  Cleaning previous C++ build artifacts..."
  rm -rf "$C_PLUS_PLUS_BUILD_DIR"
  echo "  Configuring C++ build with CMake..."
  cmake -S "$C_PLUS_PLUS_MODULES_DIR" -B "$C_PLUS_PLUS_BUILD_DIR"
  echo "  Building C++ modules..."
  cmake --build "$C_PLUS_PLUS_BUILD_DIR"
  echo "  Copying compiled C++ modules to Python path..."
  find "$C_PLUS_PLUS_BUILD_DIR" -name "leveled_gss_cpp.*.so" -exec cp {} "$C_PLUS_PLUS_OUTPUT_DIR" \;
  find "$C_PLUS_PLUS_BUILD_DIR" -name "precompute3_engine.*.so" -exec cp {} "$C_PLUS_PLUS_OUTPUT_DIR" \;
  echo "C++ module build complete."
  echo "---"
fi

for full_impl_path in "${IMPLS[@]}"; do
  if [[ "$full_impl_path" == *.py ]]; then
    local_path="${full_impl_path#$PYTHON_SRC_ROOT/}" # Remove PYTHON_SRC_ROOT prefix if it exists
    module_name=$(echo "$local_path" | sed -e 's#\(\.py\)*$##' -e 's#/#.#g')
    base_name=$(basename "$full_impl_path" | sed 's#\(\.py\)*$##')
    class_name_base=$(echo "$base_name" | sed 's/_impl$//')
    class_name="$(tr '[:lower:]' '[:upper:]' <<< "${class_name_base:0:1}")${class_name_base:1}GSS"
    full_impl_name="${module_name}.${class_name}"
  else
    full_impl_name="$full_impl_path"
    module_name="${full_impl_name%.*}"
    class_name="${full_impl_name##*.}"
  fi

  if $is_manual_sweep; then
    # --- Run Manual Sweep ---
    echo
    echo ">>> Running manual sweep for: $full_impl_path"
    output_file="${RESULTS_DIR}/${full_impl_name}.manual_sweep.json"
    cmd=(python -m gss_tester.benchmarks.runner
        "$module_name" "$class_name" --preset "$PRESET" --output "$output_file" "${EXTRA_ARGS[@]}")
    echo "${cmd[*]}"
    if "${cmd[@]}"; then
        echo ">>> Finished manual sweep for: $full_impl_path"
    else
        exit_code=$?
        [ $exit_code -eq 130 ] && echo ">>> Sweep for $full_impl_path interrupted." || echo ">>> Sweep for $full_impl_path failed."
    fi
  else
    # --- Run Preset Workloads ---
    echo
    echo ">>> Running preset workloads for: $full_impl_path"
    output_file_preset="${RESULTS_DIR}/${full_impl_name}.preset.json"
    preset_cmd=(python -m gss_tester.benchmarks.runner
        "$module_name" "$class_name" --preset "$PRESET" --output "$output_file_preset" "${EXTRA_ARGS[@]}")
    echo "${preset_cmd[*]}"
    if "${preset_cmd[@]}"; then
        echo ">>> Finished preset workloads for: $full_impl_path"
    else
        exit_code=$?
        [ $exit_code -eq 130 ] && echo ">>> Preset run for $full_impl_path interrupted." || echo ">>> Preset run for $full_impl_path failed."
    fi

    # --- Run Preset Sweeps (only if no filtering args are present) ---
    if $has_filtering_args; then
        echo
        echo ">>> Skipping preset sweeps due to filtering arguments (--only, --include, or --exclude)."
    else
        echo
        echo ">>> Checking for preset sweeps for preset '$PRESET'..."
        sweeps_to_run=$(python -m gss_tester.benchmarks.runner --preset "$PRESET" --list-sweeps)

        if [ -z "$sweeps_to_run" ]; then
        echo "No predefined sweeps for preset '$PRESET'."
        else
        # Use a while loop to read line by line (in case of multiple sweeps)
        while IFS=';' read -r workload axis values; do
            echo
            echo ">>> Running preset sweep for '$workload' on axis '$axis'"
            output_file_sweep="${RESULTS_DIR}/${full_impl_name}.sweep.${workload}.${axis}.json"

            sweep_cmd=(python -m gss_tester.benchmarks.runner
                "$module_name"
                "$class_name"
                --preset "$PRESET"
                --output "$output_file_sweep"
                --sweep-workload "$workload"
                --sweep-axis "$axis"
                --sweep-values $values # No quotes to allow word splitting
            )
            echo "${sweep_cmd[*]}"
            if "${sweep_cmd[@]}"; then
                echo ">>> Finished sweep for: $workload"
            else
                exit_code=$?
                [ $exit_code -eq 130 ] && echo ">>> Sweep for $workload interrupted." || echo ">>> Sweep for $workload failed."
            fi
        done <<< "$sweeps_to_run"
        fi
    fi
  fi
done

echo
echo "All benchmark runs completed."
echo "---"

# Analyze and plot
echo "Analyzing results..."

# Check if any result files were created before analyzing
if [ -z "$(find "$RESULTS_DIR" -maxdepth 1 -name '*.json' -print -quit)" ]; then
    echo "No result files were generated. Skipping analysis."
else
    cmd=(python -m gss_tester.benchmarks.analyzer
        "${RESULTS_DIR}"/*.json
        --outdir "${RESULTS_DIR}/analysis")
    echo "${cmd[*]}"
    "${cmd[@]}"

    if command -v realpath &> /dev/null; then
        RELATIVE_RESULTS_DIR=$(realpath --relative-to=. "$RESULTS_DIR")
    else
        RELATIVE_RESULTS_DIR="$RESULTS_DIR"
    fi

    echo
    echo "---"
    echo "Benchmark analysis complete."
    echo "Summary printed above. Plots and analysis are in: ${RELATIVE_RESULTS_DIR}/analysis"
    echo "Full JSON results are in: $RELATIVE_RESULTS_DIR"
fi
