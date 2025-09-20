#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benches.sh
#
# Orchestrates running GSS benchmark workloads across one or more implementations,
# saving JSON results, and producing comparative analyses and plots.
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
#   # Run only the 'push_scaling' workload from the 'tiny' preset
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS -- --include push_scaling
#
#   # Run a single manual scaling sweep (does not run other preset workloads/sweeps)
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS -- --sweep-workload push_scaling --sweep-axis prefix_depth --sweep-values 10 50 100 200
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT="${SCRIPT_DIR}/.."
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
  if [[ "$1" == --* ]]; then
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

# Check if the user is asking for a manual sweep.
is_manual_sweep=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--sweep-workload" ]]; then
    is_manual_sweep=true
    break
  fi
done

RESULTS_DIR="gss_bench_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
echo "Benchmark results will be saved in: $RESULTS_DIR"
echo "---"
echo "Preset: $PRESET"
echo "Implementations: ${IMPLS[*]}"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "Extra args to runner: ${EXTRA_ARGS[*]}"
fi
echo "---"

for full_impl_path in "${IMPLS[@]}"; do
  if [[ "$full_impl_path" == *.py ]]; then
    module_name=$(echo "$full_impl_path" | sed -e 's#^.*python/##' -e 's#\(\.py\)*$##' -e 's#/#.#g')
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

    # --- Run Preset Sweeps ---
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

    echo
    echo "---"
    echo "Benchmark analysis complete."
    echo "Summary printed above. Plots saved to: ${RESULTS_DIR}/analysis/plots"
    echo "Full JSON results are in: $RESULTS_DIR"
fi
      else
          echo
          echo ">>> Bench for $full_impl_path failed with exit code $exit_code. Skipping."
      fi
  fi
done

echo
echo "All benchmark runs completed."
echo "---"

# Analyze and plot
echo "Analyzing results..."
cmd=(python -m gss_tester.benchmarks.analyzer
    "${RESULTS_DIR}"/*.json
    --outdir "${RESULTS_DIR}/analysis")
echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Benchmark analysis complete."
echo "Summary printed above. Plots saved to: ${RESULTS_DIR}/analysis/plots"
echo "Full JSON results are in: $RESULTS_DIR"
