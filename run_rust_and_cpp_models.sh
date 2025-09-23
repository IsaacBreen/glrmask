#!/usr/bin/env bash
set -euo pipefail

# This script builds the Boost.ICL-backed C++ extension and runs the benchmarks
# comparing the Rust baseline and the C++-accelerated Python model.
#
# Usage:
#   bash run_rust_and_cpp_models.sh
#
# Requirements:
#   - curl, tar, a C++17 compiler (clang++ or g++)
#   - Python with pip
#
# It downloads Boost headers locally (no system install needed), builds the extensions in-place,
# and then runs the provided run_benchmarks.sh script.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 1) Ensure we are in the repo root and Python folder exists
if [ ! -d "python" ]; then
  echo "Error: expected 'python' directory at repo root. Run this script from the repo root."
  exit 1
fi

# 2) Setup: download Boost headers locally and install pybind11
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

echo "Installing pybind11 (local user)..."
python3 -m pip install --user --quiet pybind11

# 3) Build the C++ extensions in place
echo "Building C++ extensions ..."
cd "${PROJECT_ROOT}/python"

# Compute extension suffix and includes via pybind11
EXT_SUFFIX="$(python3 - << 'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
CXX="${CXX:-c++}"
LDFLAGS=""
if [[ "$(uname)" == "Darwin" ]]; then
  LDFLAGS="-undefined dynamic_lookup"
fi

# Compile icl_rangeset (Boost.ICL-backed RangeSet)
${CXX} -g -O0 -fsanitize=address -std=c++17 -shared -fPIC \
  ${PYBIND_INCLUDES} \
  -I"${BOOST_DIR}" \
  "aug25/models/icl_rangeset.cpp" \
  -o "aug25/models/icl_rangeset${EXT_SUFFIX}" \
  ${LDFLAGS}

# Compile precompute3_engine (C++ commit/get_mask engine)
${CXX} -g -O0 -fsanitize=address -std=c++17 -shared -fPIC \
  ${PYBIND_INCLUDES} \
  "aug25/models/precompute3_engine.cpp" \
  -o "aug25/models/precompute3_engine${EXT_SUFFIX}" \
  ${LDFLAGS}

echo "Build complete:"
echo " - python/aug25/models/icl_rangeset${EXT_SUFFIX}"
echo " - python/aug25/models/precompute3_engine${EXT_SUFFIX}"

# 4) Run benchmarks comparing Rust baseline vs C++-accelerated model
echo
echo "Running benchmarks..."
cd "$PROJECT_ROOT"
bash python/run_benchmarks.sh "python/aug25/models/rust_model.py" "python/aug25/models/precompute3_model_cpp.py"
