#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benches.sh
#
# Orchestrates running GSS benchmark workloads across one or more implementations,
# saving JSON results, and producing comparative analyses and plots.
#
# Usage:
#   ./gss_tester/run_benches.sh <preset> <impl1> [impl2 ...] [-- include/exclude args...]
#
# Examples:
#   ./gss_tester/run_benches.sh tiny gss_tester.implementations.reference_impl.ReferenceGSS
#   ./gss_tester/run_benches.sh small gss_tester.implementations.reference_impl.ReferenceGSS gss_tester.fast_impl.FastGSS
#
# Notes:
#   - Each implementation is specified as 'module.ClassName' or a .py path.
#   - You can pass additional args after implementations to filter workloads:
#       --include push_scaling merge_surface_changes
#       --exclude fuzz
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${PYTHON_SRC_ROOT}:${PYTHONPATH:-}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <preset:{tiny|small|medium|large}> <impl1> [impl2 ...] [-- include/exclude args...]"
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
  echo
  echo ">>> Running benches for: $full_impl_path"

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

  output_file="${RESULTS_DIR}/${full_impl_name}.json"
  cmd=(python -m gss_tester.benchmarks.runner
      "$module_name"
      "$class_name"
      --preset "$PRESET"
      --output "$output_file"
      "${EXTRA_ARGS[@]}"
  )
  echo "${cmd[*]}"
  if "${cmd[@]}"; then
      echo ">>> Finished benches for: $full_impl_path"
  else
      exit_code=$?
      if [ $exit_code -eq 130 ]; then
          echo
          echo ">>> Bench for $full_impl_path interrupted. Skipping."
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
