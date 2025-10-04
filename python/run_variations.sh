#!/bin/bash
set -euo pipefail

# ==============================================================================
# run_variations.sh
#
# A script to identify hot get_mask steps and evaluate model variations against
# that hot point. It reuses the same build logic as run_benchmarks.sh and runs
# a single model file with an internal list of variations.
#
# Usage:
#   ./run_variations.sh <model.py>
#
# Example:
#   ./run_variations.sh python/aug25/models/precompute3_model_pure_python_opt3.py
#
# Environment Variables:
#   CONSTRAINT_FILE: Path to the pre-compiled .json.gz constraint file.
#                    (Default: ./.cache/test_vocabs/js_grammar_constraint.json.gz)
#   CODE_FILE:       Path to the code file to use as input. (Default: ./src/example_code.js)
#   SKIP_CPP_BUILD:  Set to 1 to disable C++ compilation (Default: 0)
#   DETECT_REPEAT:   Number of input passes to select hot step. (Default: 3)
#   EVAL_REPEAT:     Number of repeated get_mask runs for evaluation. (Default: 10)
#   AGG_METHOD:      Aggregation method (min, mean, median, max). (Default: max)
#   METRIC:          Metric to optimize (edges_traversed | main_loop_ms). (Default: edges_traversed)
#   HOT_STEPS:       Number of steps to consider hot (currently only the first is evaluated). (Default: 1)
# ==============================================================================

# --- Configuration ---
: "${CONSTRAINT_FILE:="./.cache/test_vocabs/js_grammar_constraint.json.gz"}"
: "${CODE_FILE:="./src/example_code.js"}"
: "${SKIP_CPP_BUILD:=0}"
: "${DETECT_REPEAT:=3}"
: "${EVAL_REPEAT:=10}"
: "${AGG_METHOD:=max}"
: "${METRIC:=edges_traversed}"
: "${HOT_STEPS:=1}"

# --- PYTHONPATH setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- Argument Validation ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model.py>"
    exit 1
fi

MODEL_FILE="$1"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi
if [ ! -f "$CONSTRAINT_FILE" ]; then
    echo "Error: Constraint file not found: $CONSTRAINT_FILE"
    exit 1
fi
if [ ! -f "$CODE_FILE" ]; then
    echo "Error: Code file not found: $CODE_FILE"
    exit 1
fi

echo "Optimization coordinator"
echo "---"
echo "Model: $MODEL_FILE"
echo "Constraint File: $CONSTRAINT_FILE"
echo "Code: $CODE_FILE"
echo "Detect repeats: $DETECT_REPEAT"
echo "Eval repeats: $EVAL_REPEAT"
echo "Agg method: $AGG_METHOD"
echo "Metric: $METRIC"
echo "Hot steps: $HOT_STEPS"
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

echo "Starting optimization coordination..."
RESULTS_DIR="variations_results/$(date +"%Y-%m-%d_%H-%M-%S")"
mkdir -p "$RESULTS_DIR"
cmd=(python -m python.aug25.optimization_coordinator
    --constraint-file "$CONSTRAINT_FILE"
    --code "$CODE_FILE"
    --model "$MODEL_FILE"
    --detect-repeat "$DETECT_REPEAT"
    --eval-repeat "$EVAL_REPEAT"
    --agg-method "$AGG_METHOD"
    --metric "$METRIC"
    --hot-steps "$HOT_STEPS"
    --output "$RESULTS_DIR")
echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Optimization coordination complete. Results in: $RESULTS_DIR"
