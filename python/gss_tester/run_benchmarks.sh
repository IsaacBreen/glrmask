#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to run the GSS benchmark suite against one or more implementations
# and analyze the results (including optional plots).
#
# Usage examples:
#   ./gss_tester/run_benchmarks.sh gss_tester.implementations.reference_impl.ReferenceGSS
#   ./gss_tester/run_benchmarks.sh gss_tester.implementations.reference_impl.ReferenceGSS --preset tiny,small
#   ./gss_tester/run_benchmarks.sh python/gss_tester/implementations/reference_impl.py -w split_modify_merge_shared -p tiny
#
# Notes:
# - Each implementation can be specified as 'module.Class' or as a .py file path.
# - Results are written under gss_bench_results/<timestamp>/.
# - After running, an analysis step prints a summary and, if matplotlib is available,
#   saves PNG plots in gss_bench_analysis/<timestamp>/.
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SRC_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${PYTHON_SRC_ROOT}:${PYTHONPATH:-}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <impl1> [impl2 ...] [--preset tiny,small,medium,large] [--only names] [--exclude names] [--repeat N] [--mem]"
    exit 1
fi

# Defaults
PRESETS=("tiny")
ONLY_FILTER=""
EXCLUDE_FILTER=""
REPEAT="1"
MEM_FLAG=""
ANALYZE_PLOTS="1"

# Collect implementations and options
IMPLS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preset|-p)
            shift
            IFS=',' read -r -a PRESETS <<< "$1"
            ;;
        --only|-w)
            shift
            ONLY_FILTER="$1"
            ;;
        --exclude|-x)
            shift
            EXCLUDE_FILTER="$1"
            ;;
        --repeat|-r)
            shift
            REPEAT="$1"
            ;;
        --mem|-m)
            MEM_FLAG="--mem"
            ;;
        --no-plots)
            ANALYZE_PLOTS="0"
            ;;
        *)
            IMPLS+=("$1")
            ;;
    esac
    shift || true
done

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="gss_bench_results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "Benchmark results will be saved in: $RESULTS_DIR"
echo "---"
echo "Implementations: ${IMPLS[*]}"
echo "Presets: ${PRESETS[*]}"
echo "Only workloads: ${ONLY_FILTER:-<all>}"
echo "Exclude workloads: ${EXCLUDE_FILTER:-<none>}"
echo "Repeat: ${REPEAT}"
echo "Memory profiling: $([[ -n "$MEM_FLAG" ]] && echo yes || echo no)"
echo "---"

# Helper: derive module and class from input which can be module.Class or file path
derive_module_class() {
    local input="$1"
    if [[ "$input" == *.py ]]; then
        local module_name
        module_name=$(echo "$input" | sed -e 's#^.*python/##' -e 's#\(\.py\)*$##' -e 's#/#.#g')
        local base_name
        base_name=$(basename "$input" | sed 's#\(\.py\)*$##')
        local class_name_base
        class_name_base=$(echo "$base_name" | sed 's/_impl$//')
        local first_char uppercase_first
        first_char="${class_name_base:0:1}"
        uppercase_first="$(tr '[:lower:]' '[:upper:]' <<< "$first_char")"
        local class_name="${uppercase_first}${class_name_base:1}GSS"
        echo "${module_name} ${class_name}"
    else
        local module_name="${input%.*}"
        local class_name="${input##*.}"
        echo "${module_name} ${class_name}"
    fi
}

BENCH_FILES=()
for impl in "${IMPLS[@]}"; do
    read -r MODULE_NAME CLASS_NAME <<< "$(derive_module_class "$impl")"
    FULL_IMPL_NAME="${MODULE_NAME}.${CLASS_NAME}"

    for PRESET in "${PRESETS[@]}"; do
        OUT_FILE="${RESULTS_DIR}/${FULL_IMPL_NAME}__${PRESET}.json"
        echo
        echo ">>> Running benchmarks for: ${FULL_IMPL_NAME} [${PRESET}]"
        CMD=(python -m gss_tester.benchmarks.runner
            "$MODULE_NAME" "$CLASS_NAME"
            --output "$OUT_FILE"
            --preset "$PRESET"
            --repeat "$REPEAT"
        )
        if [[ -n "$ONLY_FILTER" ]]; then
            CMD+=(--only "$ONLY_FILTER")
        fi
        if [[ -n "$EXCLUDE_FILTER" ]]; then
            CMD+=(--exclude "$EXCLUDE_FILTER")
        fi
        if [[ -n "$MEM_FLAG" ]]; then
            CMD+=("$MEM_FLAG")
        fi

        echo "${CMD[*]}"
        if "${CMD[@]}"; then
            BENCH_FILES+=("$OUT_FILE")
            echo ">>> Completed: $FULL_IMPL_NAME [$PRESET]"
        else
            exit_code=$?
            if [ $exit_code -eq 130 ]; then
                echo ">>> Benchmark interrupted for $FULL_IMPL_NAME [$PRESET]."
            else
                echo ">>> Benchmark failed with exit code $exit_code for $FULL_IMPL_NAME [$PRESET]."
            fi
        fi
    done
done

echo
echo "All benchmarks completed."
echo "---"

if [[ "${#BENCH_FILES[@]}" -eq 0 ]]; then
    echo "No benchmark files were generated. Skipping analysis."
    exit 0
fi

echo "Analyzing results..."
ANALYSIS_DIR="gss_bench_analysis/${TIMESTAMP}"
mkdir -p "$ANALYSIS_DIR"
ANALYZE_CMD=(python -m gss_tester.benchmarks.analysis "${BENCH_FILES[@]}" --out-dir "$ANALYSIS_DIR")
if [[ "$ANALYZE_PLOTS" = "0" ]]; then
    ANALYZE_CMD+=(--no-plots)
fi
echo "${ANALYZE_CMD[*]}"
"${ANALYZE_CMD[@]}"

echo
echo "---"
echo "Analysis complete."
echo "Summary and plots (if enabled) in: $ANALYSIS_DIR"
echo "Raw JSON results in: $RESULTS_DIR"
