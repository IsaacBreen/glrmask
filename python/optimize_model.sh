#!/bin/bash
set -euo pipefail

# ==============================================================================
# optimize_model.sh
#
# A script to find the most expensive `get_mask` calls for a given model and
# input, and then benchmark different optimization "variations" on that specific
# expensive call.
#
# This script should be run from the project root directory.
#
# Usage:
#   ./python/optimize_model.sh <model.py>
#
# Example:
#   ./python/optimize_model.sh python/aug25/models/precompute3_model_pure_python_opt3.py
#
# Environment Variables:
#   CONSTRAINT_FILE: Path to the pre-compiled .json.gz constraint file.
#   CODE_FILE:       Path to the code file to use as input.
#   FIND_STEPS_REPEAT: Number of times to run through the code to find expensive steps. (Default: 2)
#   BENCHMARK_REPEAT:  Number of times to run `get_mask` for each variation. (Default: 5)
#   NUM_STEPS_TO_FIND: The number of most expensive steps to analyze. (Default: 1)
#   AGG_METHOD:      Aggregation method for finding expensive steps (min, mean, max). (Default: "min")
# ==============================================================================

# --- Configuration ---
# Use environment variables if set, otherwise use defaults.
: "${CONSTRAINT_FILE:="./.cache/test_vocabs/js_grammar_constraint.json.gz"}"
: "${CODE_FILE:="./src/example_code.js"}"
: "${SKIP_CPP_BUILD:=0}"
: "${FIND_STEPS_REPEAT:=2}"
: "${BENCHMARK_REPEAT:=5}"
: "${NUM_STEPS_TO_FIND:=1}"
: "${AGG_METHOD:="min"}"

# --- PYTHONPATH setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- Argument Validation ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model.py>"
    echo "Error: Exactly one model argument is required."
    exit 1
fi

MODEL_FILE="$1"

# Check that all provided files exist
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


# --- Setup ---
echo "--- Model Optimization Analysis ---"
echo "Model: $MODEL_FILE"
echo "Constraint File: $CONSTRAINT_FILE"
echo "Code: $CODE_FILE"
echo "Find steps repetitions: $FIND_STEPS_REPEAT"
echo "Benchmark repetitions: $BENCHMARK_REPEAT"
echo "Number of steps to find: $NUM_STEPS_TO_FIND"
echo "Aggregation method: $AGG_METHOD"
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

# --- Run Optimizer ---
echo "Starting model optimization analysis..."
cmd=(python -m python.aug25.optimize_model
    --model "$MODEL_FILE"
    --code "$CODE_FILE"
    --constraint-file "$CONSTRAINT_FILE"
    --find-steps-repeat "$FIND_STEPS_REPEAT"
    --benchmark-repeat "$BENCHMARK_REPEAT"
    --num-steps-to-find "$NUM_STEPS_TO_FIND"
    --agg-method "$AGG_METHOD"
)

echo "${cmd[*]}"
"${cmd[@]}"

echo
echo "---"
echo "Analysis complete."
