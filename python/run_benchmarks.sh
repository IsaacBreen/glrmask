#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_benchmarks.sh
#
# A script to automate running benchmarks for multiple model and constraint
# file pairs, generating JSON results and analysis plots.
#
# This script should be run from the project root directory.
#
# Usage:
#   Usage (Mode 1: model:constraint pairs):
#     ./run_benchmarks.sh <model1.py:constraint1.json.gz> [model2.py:constraint2.json.gz ...]
#
#   Usage (Mode 2: legacy, with CONSTRAINT_FILE env var):
#     CONSTRAINT_FILE=c.json.gz ./run_benchmarks.sh <model1.py> [model2.py ...]
#
# Example:
#   ./run_benchmarks.sh model.py:c1.json.gz model.py:c2.json.gz
#
# Environment Variables:
#   CONSTRAINT_FILE: Path to a constraint file. Used as a default in legacy mode.
#   CODE_FILE:       Path to the code file to use as input.
#                    (Default: ./src/example_code.js)
#   REPEAT:          Number of times to run each benchmark. (Default: 1)
#   AGG_METHOD:      Aggregation method for analyzer (mean, median, min, max).
#                    If unset, runs are plotted individually. (Default: "")
# ==============================================================================

# --- Configuration ---
: "${CONSTRAINT_FILE:=""}"
: "${CODE_FILE:="./src/example_code.js"}"
: "${SKIP_CPP_BUILD:=0}" # Set to 1 to disable C++ compilation
: "${SKIP_RUST_BUILD:=0}" # Set to 1 to disable Rust compilation
: "${REPEAT:=1}"
: "${AGG_METHOD:=""}"
: "${SKIP_PLOTS:=0}" # Set to 1 to skip plot generation

# --- PYTHONPATH setup ---
# The script is run from the project root. The python modules are in the 'python' directory.
# We add the 'python' directory to PYTHONPATH so that compiled extension modules (like leveled_gss_cpp)
# can be found via top-level imports.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- Argument Validation ---
if [ "$#" -lt 1 ]; then
    echo "Usage 1: $0 <model1.py:constraint1.json.gz> [model2.py:constraint2.json.gz ...]"
    echo "Usage 2: CONSTRAINT_FILE=c.json.gz $0 <model1.py> [model2.py ...]"
    echo "Error: At least one model argument is required."
    exit 1
fi

ALL_PAIRS=()
# Check if we are in pair mode (arg contains ':') or legacy mode
if [[ "$1" == *":"* ]]; then
    echo "Detected 'model:constraint' pair syntax."
    ALL_PAIRS=("$@")
else
    echo "Detected legacy model-only syntax. Using CONSTRAINT_FILE env var."
    if [ -z "$CONSTRAINT_FILE" ]; then
        echo "Error: CONSTRAINT_FILE must be set when using legacy model-only syntax."
        exit 1
    fi
    if [ ! -f "$CONSTRAINT_FILE" ]; then
        echo "Error: Constraint file from environment not found: $CONSTRAINT_FILE"
        exit 1
    fi
    for model_file in "$@"; do
        ALL_PAIRS+=("${model_file}:${CONSTRAINT_FILE}")
    done
fi

BASELINE_PAIR="${ALL_PAIRS[0]}"

# Check that all provided files exist
for pair in "${ALL_PAIRS[@]}"; do
    MODEL_FILE="${pair%%:*}"
    CONSTRAINT_FILE_ARG="${pair#*:}"
    if [ "$MODEL_FILE" == "$CONSTRAINT_FILE_ARG" ]; then # handles case where there is no ':'
        echo "Error: Invalid argument format '$pair'. Expected 'model.py:constraint.json.gz'."
        exit 1
    fi
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Model file not found: $MODEL_FILE"
        exit 1
    fi
    if [ ! -f "$CONSTRAINT_FILE_ARG" ]; then
        echo "Error: Constraint file not found: $CONSTRAINT_FILE_ARG"
        exit 1
    fi
done
if [ ! -f "$CODE_FILE" ]; then
    echo "Error: Code file not found: $CODE_FILE"
    exit 1
fi


# --- Setup ---
# Create a unique directory for this benchmark run's results.
# If a directory with the same timestamp already exists, append a counter.
BASE_RESULTS_DIR="benchmark_results/$(date +"%Y-%m-%d_%H-%M-%S")"
RESULTS_DIR="$BASE_RESULTS_DIR"
COUNTER=1
while [ -d "$RESULTS_DIR" ]; do
  RESULTS_DIR="${BASE_RESULTS_DIR}_${COUNTER}"
  COUNTER=$((COUNTER + 1))
done

mkdir -p "$RESULTS_DIR"
echo "Benchmark results will be saved in: $RESULTS_DIR"
echo "Benchmark pairs: ${ALL_PAIRS[*]}"
echo "Code: $CODE_FILE"
echo "Repeat count: $REPEAT"
if [[ -n "$AGG_METHOD" ]]; then
  echo "Aggregation: $AGG_METHOD"
fi
echo "---"


if [[ "${SKIP_CPP_BUILD}" != "1" ]]; then
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

  # Control C++ stats collection. Default to enabled.
  : "${ENABLE_CPP_STATS:=1}"
  if [[ "${ENABLE_CPP_STATS}" == "1" ]]; then
    echo "Building C++ with stats enabled (ENABLE_CPP_STATS=1)"
    CXXFLAGS="${CXXFLAGS} -DENABLE_STATS"
  else
    echo "Building C++ with stats disabled (ENABLE_CPP_STATS=0)"
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
else
  echo "Skipping C++ extension build (SKIP_CPP_BUILD=1)."
  echo "---"
fi

# --- Automated Rust Build Process ---
if [ "${SKIP_RUST_BUILD:-0}" == "1" ]; then
  echo "SKIP_RUST_BUILD is set. Skipping Rust module build."
  echo "---"
else
  echo "Automating Rust module build..."
  RUST_PROJECT_DIR="$SCRIPT_DIR/leveled_rs"
  if [ -d "$RUST_PROJECT_DIR" ]; then
    echo "  Building Rust modules with maturin..."
    # Assuming maturin is installed in the environment. 'maturin develop' builds and installs the wheel in the current venv.
    (cd "$RUST_PROJECT_DIR" && maturin develop)
    echo "Rust module build complete."
  else
    echo "Rust project directory not found at $RUST_PROJECT_DIR. Skipping build."
  fi
  echo "---"
fi

# --- Run Benchmarks ---
echo "Starting benchmark runs..."
# Define the ASAN library path at the top of the loop for clarity
ASAN_LIB="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"

for pair in "${ALL_PAIRS[@]}"; do
    MODEL_FILE="${pair%%:*}"
    CONSTRAINT_FILE_ARG="${pair#*:}"

    echo
    echo ">>> Running benchmark for: $(basename "$MODEL_FILE") with $(basename "$CONSTRAINT_FILE_ARG")"
    cmd=(python -m python.aug25.benchmark_runner
        --code "$CODE_FILE"
        --constraint-file "$CONSTRAINT_FILE_ARG"
        --model "$MODEL_FILE"
        --output "$RESULTS_DIR"
        --repeat "$REPEAT")
    echo "${cmd[*]}"
    # Prepend the environment variable ONLY for the C++ model
    if [[ "$MODEL_FILE" == *"precompute3_model_cpp.py"* ]]; then
        echo ">>> Running with AddressSanitizer..."
        if DYLD_INSERT_LIBRARIES="$ASAN_LIB" "${cmd[@]}"; then
            echo ">>> Finished benchmark for: $(basename "$MODEL_FILE")"
        else
            exit_code=$?
            echo
            echo ">>> Benchmark for $(basename "$MODEL_FILE") failed with exit code $exit_code. Skipping."
        fi
    else
        if "${cmd[@]}"; then
            echo ">>> Finished benchmark for: $(basename "$MODEL_FILE")"
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

BASELINE_MODEL_FILE="${BASELINE_PAIR%%:*}"
BASELINE_CONSTRAINT_FILE="${BASELINE_PAIR#*:}"
BASELINE_MODEL_STEM="$(basename "$BASELINE_MODEL_FILE" .py)"
BASELINE_CONSTRAINT_STEM=$(basename "${BASELINE_CONSTRAINT_FILE}")
BASELINE_CONSTRAINT_STEM=${BASELINE_CONSTRAINT_STEM%.json.gz}
BASELINE_CONSTRAINT_STEM=${BASELINE_CONSTRAINT_STEM%.json}
BASELINE_KEY="${BASELINE_MODEL_STEM}__${BASELINE_CONSTRAINT_STEM}"

cmd=(python -m python.aug25.benchmark_analyzer
    "${RESULTS_DIR}"/*.json
    --baseline "$BASELINE_KEY"
    --output-dir "$PLOTS_DIR")

if [[ -n "$AGG_METHOD" ]]; then
  cmd+=(--agg-method "$AGG_METHOD")
fi

if [[ "${SKIP_PLOTS}" == "1" ]]; then
  cmd+=(--skip-plots)
fi

echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Benchmark analysis complete."
echo "Baseline: $BASELINE_KEY"
echo "Summary printed above. Plots are saved in: $PLOTS_DIR"
echo "Full JSON results are in: $RESULTS_DIR"