#!/usr/bin/env bash
set -euo pipefail

# This script runs INSIDE the container (manylinux_2_28 or node:22-alpine)
# to build, test, validate runtime dependencies, and package Linux artifacts.

TARGET="${TARGET:-${1:-}}"
TAG="${TAG:-${2:-}}"

if [ -z "$TARGET" ] || [ -z "$TAG" ]; then
  echo "Usage: build-linux-container.sh <target-triple> <platform-tag>" >&2
  exit 1
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/native-ci}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export CARGO_INCREMENTAL=0
RUNNER_TEMP="${RUNNER_TEMP:-/tmp}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

cd "$GITHUB_WORKSPACE"
git config --global --add safe.directory "$GITHUB_WORKSPACE" 2>/dev/null || true

echo "=== Linux container build for $TAG ($TARGET) ==="

# Detect musl vs glibc container
if [ -f /etc/alpine-release ]; then
  IS_MUSL=true
else
  IS_MUSL=false
fi

# Toolchain setup
if [ "$IS_MUSL" = true ]; then
  echo "Setting up Alpine musl toolchains..."
  apk add --no-cache bash git build-base cmake ninja python3 py3-pip curl binutils
  if [ ! -f /usr/bin/python ] && [ -f /usr/bin/python3 ]; then
    ln -sf /usr/bin/python3 /usr/bin/python
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  rustup target add "$TARGET"
  # musl Rust targets default to a static C runtime, which cannot be used
  # for a dlopen-able cdylib. Scope this flag to the actual target, leaving
  # host build scripts and CPU portability unchanged.
  TARGET_ENV="$(printf '%s' "$TARGET" | tr '[:lower:]-' '[:upper:]_')"
  export "CARGO_TARGET_${TARGET_ENV}_RUSTFLAGS=-C target-feature=-crt-static"
else
  echo "Setting up manylinux_2_28 (glibc 2.28) toolchains..."
  # manylinux_2_28 has Python in /opt/python/cp312-cp312/bin or system python3
  if [ -d /opt/python/cp312-cp312/bin ]; then
    export PATH="/opt/python/cp312-cp312/bin:$PATH"
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  rustup target add "$TARGET"
  # Install Node 24 for building/testing if not present
  if ! command -v node >/dev/null 2>&1; then
    ARCH="$(uname -m)"
    NODE_ARCH="x64"
    if [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; fi
    curl -fsSL "https://nodejs.org/dist/v24.2.0/node-v24.2.0-linux-${NODE_ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1
  fi
  python -m pip install --upgrade pip cmake ninja
fi

echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo "Node: $(node --version)"
echo "Python: $(python --version)"

# 1. Install Node dependencies
npm ci --omit=optional --ignore-scripts --prefix bindings/node

# 2. Validate source contracts
python bindings/ci/check.py

# 3. Build native release libraries (Cargo)
cargo build --release --locked -p glrmask-c -p glrmask-node --target "$TARGET"

# 4. Verify runtime dependencies and symbol versions
LIB_DIR="$GITHUB_WORKSPACE/$CARGO_TARGET_DIR/$TARGET/release"
python bindings/ci/check.py --library "$LIB_DIR/libglrmask_c.so" --platform "$TAG"

# 5. Validate and exercise the locally built native Node addon.
# The distribution-wide thin root and eight platform packages are packed and
# smoke-tested by the aggregator after every CI native target has uploaded
# its checksum-verified binary and manifest; no per-target fat package exists.
cd bindings/node
npm run build -- --target "$TARGET"
python "$GITHUB_WORKSPACE/bindings/ci/check.py" --addon "glrmask.$TAG.node" --platform "$TAG"
npm run check
npm test
cd "$GITHUB_WORKSPACE"

# 6. Static/shared CMake installations and relocated C/C++ consumers
python bindings/cpp/scripts/validate-install.py \
  --library-dir "$LIB_DIR" \
  --work-dir "$RUNNER_TEMP/cpp-validation"

# 7. Cross-language masks and artifact exchange (parity)
EXE="$RUNNER_TEMP/cpp-validation/build-static/glrmask_cpp_tests"
node bindings/node/scripts/parity.cjs "$EXE" "$RUNNER_TEMP/native-parity"

# 8. Build self-contained C++ archives
python bindings/ci/package.py \
  --install-work "$RUNNER_TEMP/cpp-validation" \
  --platform "$TAG" \
  --output "$RUNNER_TEMP/cpp-packages"

# 9. Node 22 ABI compatibility
if [ "$IS_MUSL" = false ]; then
  # On glibc, test with Node 22 to confirm ABI compatibility
  ARCH="$(uname -m)"
  NODE_ARCH="x64"
  if [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; fi
  mkdir -p /tmp/node22
  curl -fsSL "https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-${NODE_ARCH}.tar.gz" | tar -xz -C /tmp/node22 --strip-components=1
  (cd bindings/node && PATH="/tmp/node22/bin:$PATH" npm test)
else
  # On node:22-alpine, base node is already Node 22
  (cd bindings/node && npm test)
fi

# Ensure output permissions are accessible by host runner
chmod -R a+rwx "$GITHUB_WORKSPACE/bindings/node" "$RUNNER_TEMP" 2>/dev/null || true

echo "=== Linux container build completed successfully for $TAG ==="
