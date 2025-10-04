#!/bin/bash
set -euo pipefail

# ==============================================================================
# optimize_model.sh
#
# A script to run the model optimization framework, which identifies expensive
# get_mask() calls and tries different optimization variations.
#
# Usage:
#   ./optimize_model.sh [options]
#
# Example:
#   SKIP_CPP_BUILD=1 CONSTRAINT_FILE="..." CODE_FILE="..." \\
#     bash python/optimize_model.sh
#
# Environment Variables:
#   MODEL_FILE:       Path to the model .py file.
#                     (Default: python/aug25/models/precompute3_model_pure_python_opt3.py)
#   CONSTRAINT_FILE:  Path to the pre-compiled .json.gz constraint file.
#                     (Default: ./.cache/test_vocabs/js_grammar_constraint.json.gz)
#   CODE_FILE:        Path to the code file to use as input.
#                     (Default: ./src/example_code.js)
#   SKIP_CPP_BUILD:   Set to 1 to disable C++ compilation. (Default: 0)
#   WARMUP_REPS:      Number of warmup repetitions to identify expensive steps. (Default: 2)
#   TEST_REPS:        Number of test repetitions per variation. (Default: 10)
#   NUM_STEPS:        Number of expensive steps to select. (Default: 1)
#   AGG_METHOD:       Aggregation method (mean, median, min, max). (Default: mean)
# ==============================================================================

# --- Configuration ---
: "${MODEL_FILE:="python/aug25/models/precompute3_model_pure_python_opt3.py"}"
: "${CONSTRAINT_FILE:="./.cache/test_vocabs/js_grammar_constraint.json.gz"}"
: "${CODE_FILE:="./src/example_code.js"}"
: "${SKIP_CPP_BUILD:=0}"
: "${WARMUP_REPS:=2}"
: "${TEST_REPS:=10}"
: "${NUM_STEPS:=1}"
: "${AGG_METHOD:="mean"}"

# --- PYTHONPATH setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}"

# --- Argument Validation ---
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

echo "Model Optimization Framework"
echo "---"
echo "Model: $MODEL_FILE"
echo "Constraint: $CONSTRAINT_FILE"
echo "Code: $CODE_FILE"
echo "Warmup reps: $WARMUP_REPS"
echo "Test reps: $TEST_REPS"
echo "Num steps: $NUM_STEPS"
echo "Agg method: $AGG_METHOD"
echo "---"

if [[ "${SKIP_CPP_BUILD}" != "1" ]]; then
  echo "Ensuring C++ extensions are built..."

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

  python3 -m pip install --quiet pybind11

  EXT_SUFFIX="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
  PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
  CXX="${CXX:-c++}"
  CXXFLAGS="-O3 -DNDEBUG -march=native -flto -std=c++17 -shared -fPIC"
  LDFLAGS="-flto"
  if [[ "$(uname)" == "Darwin" ]]; then
    LDFLAGS="${LDFLAGS} -undefined dynamic_lookup"
  fi

  cd "${PROJECT_ROOT}/python"
  ${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
    "aug25/models/icl_rangeset.cpp" -o "aug25/models/icl_rangeset${EXT_SUFFIX}" ${LDFLAGS}
  ${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
    "aug25/models/leveled_gss_py.cpp" -o "leveled_gss_cpp${EXT_SUFFIX}" ${LDFLAGS}
  ${CXX} ${CXXFLAGS} ${PYBIND_INCLUDES} -I"${BOOST_DIR}" \
    "aug25/models/precompute3_engine.cpp" -o "aug25/models/precompute3_engine${EXT_SUFFIX}" ${LDFLAGS}
  cd -

  echo "Build complete."
else
  echo "Skipping C++ extension build (SKIP_CPP_BUILD=1)."
fi

echo "---"
echo "Running optimization framework..."
python3 -m python.aug25.optimize_model \
  --model "$MODEL_FILE" \
  --code "$CODE_FILE" \
  --constraint-file "$CONSTRAINT_FILE" \
  --warmup-reps "$WARMUP_REPS" \
  --test-reps "$TEST_REPS" \
  --num-steps "$NUM_STEPS" \
  --agg-method "$AGG_METHOD"
