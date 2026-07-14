#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3.12}
PYTHON_BIN=$(command -v "$PYTHON_BIN")

fail() {
  printf 'release dry-run: %s\n' "$*" >&2
  exit 1
}

[[ -z $(git status --porcelain) ]] || fail "worktree is not clean"


metadata=$(
  "$PYTHON_BIN" - <<'PY'
import pathlib
import tomllib

root = pathlib.Path.cwd()
with (root / "Cargo.toml").open("rb") as f:
    cargo = tomllib.load(f)
with (root / "python" / "Cargo.toml").open("rb") as f:
    py_cargo = tomllib.load(f)
with (root / "python" / "pyproject.toml").open("rb") as f:
    pyproject = tomllib.load(f)

project = pyproject["project"]
print(cargo["package"]["version"])
print(py_cargo["package"]["version"])
print(project["version"])
print(project["name"])
print("\n".join(project.get("license-files", [])))
PY
)

root_version=$(printf '%s\n' "$metadata" | sed -n '1p')
binding_version=$(printf '%s\n' "$metadata" | sed -n '2p')
python_version=$(printf '%s\n' "$metadata" | sed -n '3p')
python_name=$(printf '%s\n' "$metadata" | sed -n '4p')
license_file_metadata=$(printf '%s\n' "$metadata" | sed -n '5,$p')

[[ "$root_version" == "$binding_version" ]] || fail "root Cargo version $root_version != python/Cargo.toml version $binding_version"
[[ "$root_version" == "$python_version" ]] || fail "root Cargo version $root_version != Python project version $python_version"

for name in LICENSE-MIT LICENSE-APACHE; do
  [[ -f "$ROOT/$name" ]] || fail "missing root $name"
  [[ -f "$ROOT/python/licenses/$name" ]] || fail "missing python/licenses/$name"
  cmp -s "$ROOT/$name" "$ROOT/python/licenses/$name" || fail "$name differs between repository root and python/licenses"
  printf '%s\n' "$license_file_metadata" | grep -Fxq "licenses/$name" || fail "python/pyproject.toml does not declare license-files entry for licenses/$name"
done

work=$(mktemp -d "${TMPDIR:-/tmp}/glrmask-release-dry-run.XXXXXX")
trap 'rm -rf "$work"' EXIT
cargo_target="$work/cargo-target"
dist="$work/dist"
tool_venv="$work/tools"
install_venv="$work/install"
mkdir -p "$dist"

printf 'Release artifact dry-run for %s %s\n' "$python_name" "$root_version"
printf 'Temporary output: %s\n' "$work"

CARGO_TARGET_DIR="$cargo_target" cargo +stable publish --dry-run --locked -p glrmask

"$PYTHON_BIN" -m venv "$tool_venv"
"$tool_venv/bin/python" -m pip install --upgrade pip
"$tool_venv/bin/python" -m pip install 'maturin>=1,<2' twine

(
  cd python
  RUSTUP_TOOLCHAIN=stable "$tool_venv/bin/maturin" build \
    --release \
    --sdist \
    --compatibility pypi \
    --out "$dist" \
    -i "$PYTHON_BIN"
)

"$tool_venv/bin/python" -m twine check "$dist"/*

"$tool_venv/bin/python" - "$dist" "$python_name" "$root_version" <<'PY'
import pathlib
import sys
import tarfile
import zipfile

root = pathlib.Path(sys.argv[1])
name = sys.argv[2].replace("-", "_")
version = sys.argv[3]
expected = {"LICENSE-MIT", "LICENSE-APACHE"}

sdists = list(root.glob("*.tar.gz"))
wheels = list(root.glob("*.whl"))
if len(sdists) != 1:
    raise SystemExit(f"expected exactly one sdist, found {len(sdists)}")
if len(wheels) != 1:
    raise SystemExit(f"expected exactly one host wheel, found {len(wheels)}")

with tarfile.open(sdists[0], "r:gz") as tf:
    members = {pathlib.PurePosixPath(n) for n in tf.getnames()}
    for license_name in expected:
        if not any(p.name == license_name for p in members):
            raise SystemExit(f"sdist missing {license_name}")

with zipfile.ZipFile(wheels[0]) as zf:
    names = [pathlib.PurePosixPath(n) for n in zf.namelist()]
    for license_name in expected:
        if not any(p.name == license_name and "licenses" in p.parts for p in names):
            raise SystemExit(f"wheel missing dist-info license file {license_name}")

print(f"artifact license inspection passed: {sdists[0].name}, {wheels[0].name}")
PY

wheel=$(find "$dist" -maxdepth 1 -name '*.whl' -print -quit)
[[ -n "$wheel" ]] || fail "no wheel produced"
"$PYTHON_BIN" -m venv "$install_venv"
"$install_venv/bin/python" -m pip install "$wheel"
"$install_venv/bin/python" - <<'PY'
import glrmask

vocab = glrmask.Vocab.from_dict({b"hello": 0, b" ": 1, b"world": 2})
constraint = glrmask.Constraint.from_ebnf('start ::= "hello" " " "world"', vocab)
state = constraint.start()
assert state.mask().tolist() == [True, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
state.commit_token(2)
assert state.is_finished()
print("installed-wheel first-run smoke test passed")
PY

printf 'release artifact dry-run passed for %s %s\n' "$python_name" "$root_version"
