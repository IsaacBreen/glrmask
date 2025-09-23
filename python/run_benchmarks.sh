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
# This script should be run from the project root directory.
#
# Usage:
#   ./run_benchmarks.sh <baseline_model.py> <model1.py> [model2.py ...]
#
# Example:
#   ./run_benchmarks.sh aug25/precompute2_model.py aug25/precompute3_model.py
#
# Environment Variables:
#   CONSTRAINT_FILE: Path to the pre-compiled .json.gz constraint file.
#                    (Default: ./.cache/test_vocabs/js_grammar_constraint.json.gz)
#   CODE_FILE:    Path to the code file to use as input.
#                 (Default: ./src/example_code.js)
# ==============================================================================

# --- Configuration ---
# Use environment variables if set, otherwise use defaults.
: "${CONSTRAINT_FILE:="./.cache/test_vocabs/js_grammar_constraint.json.gz"}"
: "${CODE_FILE:="./src/example_code.js"}"

# --- PYTHONPATH setup ---
# The script is run from the project root. The python modules are in the 'python' directory.
# We add the 'python' directory to PYTHONPATH so that compiled extension modules (like leveled_gss_cpp)
# can be found via top-level imports.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

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


# --- Automated C++ Build Process ---
echo "Ensuring C++ extensions are built..."

# Determine project root, assuming this script is in the 'python' directory.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 1) Setup: download Boost headers locally and install pybind11
BOOST_VERSION="1.83.0"
BOOST_VER_UNDERSCORES="1_83_0"
BOOST_DIR="${PROJECT_ROOT}/.build/boost_${BOOST_VER_UNDERSCORES}"
mkdir -p "${PROJECT_ROOT}/.build"

if [ ! -d "$BOOST_DIR" ]; then
  BOOST_TGZ="${PROJECT_ROOT}/.build/boost_${BOOST_VER_UNDERSCORES}.tar.gz"
  echo "Downloading Boost ${BOOST_VERSION} headers..."
  curl -L -o "$BOOST_TGZ" "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VER_UNDERSCORES}.tar.gz"
  echo "Extracting Boost..."
  tar -C "${PROJECT_ROOT}/.build" -xzf "$BOOST_TGZ"
fi

echo "Installing pybind11..."
python3 -m pip install --quiet pybind11

# 2) Build the C++ extensions in place
echo "Building C++ extensions..."

# Compute extension suffix and includes via pybind11
EXT_SUFFIX="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
CXX="${CXX:-c++}"
CXXFLAGS="-O3 -DNDEBUG -march=native -flto -std=c++17 -shared -fPIC"
LDFLAGS="-flto"
if [[ "$(uname)" == "Darwin" ]]; then
  LDFLAGS="${LDFLAGS} -undefined dynamic_lookup"
fi

# Optional ASan (opt-in via SANITIZE=1 env var)
if [[ "${SANITIZE:-0}" == "1" ]]; then
  echo "Building with AddressSanitizer (SANITIZE=1)"
  CXXFLAGS="-g -O1 -fsanitize=address -fno-omit-frame-pointer -std=c++17 -shared -fPIC"
  LDFLAGS="${LDFLAGS} -fsanitize=address"
fi

# Change to python directory to run build commands with relative paths
ORIGINAL_CWD=$(pwd)
cd "$SCRIPT_DIR"

# Compile extensions. Paths are relative to the 'python' directory.
${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
  "aug25/models/icl_rangeset.cpp" -o "aug25/models/icl_rangeset${EXT_SUFFIX}" ${LDFLAGS}

${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
  "aug25/models/leveled_gss_py.cpp" -o "leveled_gss_cpp${EXT_SUFFIX}" ${LDFLAGS}

${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
  "aug25/models/precompute3_engine.cpp" -o "aug25/models/precompute3_engine${EXT_SUFFIX}" ${LDFLAGS}

# Change back to original directory
cd "$ORIGINAL_CWD"

echo "Build complete."
echo "---"

# --- Run Benchmarks ---
echo "Starting benchmark runs..."
# Define the ASAN library path at the top of the loop for clarity
ASAN_LIB="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"

for model_to_benchmark in "${ALL_MODELS[@]}"; do
    echo
    echo ">>> Running benchmark for: $(basename "$model_to_benchmark")"
    cmd=(python -m python.aug25.benchmark_runner
        --code "$CODE_FILE"
        --constraint-file "$CONSTRAINT_FILE"
        --model "$model_to_benchmark"
        --output "$RESULTS_DIR")
    echo "${cmd[*]}"
    # Prepend the environment variable ONLY for the C++ model
    if [[ "$model_to_benchmark" == *"precompute3_model_cpp.py"* ]]; then
        echo ">>> Running with AddressSanitizer..."
        if DYLD_INSERT_LIBRARIES="$ASAN_LIB" "${cmd[@]}"; then
            echo ">>> Finished benchmark for: $(basename "$model_to_benchmark")"
        else
            exit_code=$?
            echo
            echo ">>> Benchmark for $(basename "$model_to_benchmark") failed with exit code $exit_code. Skipping."
        fi
    else
        if "${cmd[@]}"; then
            echo ">>> Finished benchmark for: $(basename "$model_to_benchmark")"
        else
            exit_code=$?
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
cmd=(python -m python.aug25.benchmark_analyzer
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
