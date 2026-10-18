# Windows ARM64 worker note (compat/windows-arm64-810748)

Scope: implement Windows ARM64 native support (C/C++ static+shared installers/ABI,
Node NAPI bindings, Python wheel, GitHub Actions native ARM runners + runtime
consumer tests, package metadata, docs CI). No merge/tag/publish.

## Findings

- Rust/C sources are already arch-agnostic: `bindings/cpp/build.rs` only adjusts a
  macOS install-name; `bindings/node/build.rs` is plain `napi_build::setup()`.
  `bindings/cpp/CMakeLists.txt` selects `glrmask_c.dll` / import-lib layout for all
  of WIN32, so no CMake change is needed for ARM64.
- `bindings/node/platform.cjs` already maps `aarch64-pc-windows-msvc` to tag
  `win32-arm64-msvc`, but `releaseTargets` omits it, so `assemble.cjs` rejects and
  `check-package.cjs --all` never requires a Windows ARM64 binary.
- Gaps to close: `releaseTargets`, `native-bindings.yml` matrix + hardcoded
  x64-only MSVC DevShell/linker env, `python-wheels.yml` matrix (no windows-arm64),
  `check-python-wheel-set.py` PLATFORMS, `check-cpp-release-archives.py` TARGETS,
  `release.yml` npm platform loops / wheel count (25->30) / C++ archive count
  (10->12), `docs/releasing.md` counts, Node/C++ README platform matrix wording.
- The exact-count gates (`-eq 25`, `!= 10`, `check-python-wheel-set.py`,
  `check-cpp-release-archives.py`) are what keep a half-added platform from
  silently passing, so they must move together with the matrices.
- Host here is Windows x64: an ARM64 binary cannot be executed locally. Local
  verification = `cargo check` for the ARM64 target (no link), synthetic-fixture
  runs of the packaging/validation scripts, and the unchanged x64 test path.
  Real execution evidence must come from the `windows-11-arm` CI runners.

## Changes

- `bindings/node/platform.cjs`: added `win32-arm64-msvc` to `releaseTargets`.
- `bindings/node/README.md`: six-target matrix wording (Windows x64+arm64).
- `bindings/cpp/README.md`: Windows x64+arm64 matrix wording.
- `.github/workflows/native-bindings.yml`: added
  `windows-11-arm / aarch64-pc-windows-msvc / win32-arm64-msvc` entry; MSVC
  DevShell arch and `CARGO_TARGET_*_LINKER` now derive from matrix fields
  (`msvc_arch`, target triple) instead of hardcoded x64.
- `.github/workflows/python-wheels.yml`: added `windows-arm64` entry
  (`windows-11-arm`, `target: arm64`, `python-arch: arm64`); fixed stale 25-wheel
  comment.
- `scripts/check-python-wheel-set.py`: added `windows-arm64` (`win_arm64`).
- `scripts/check-cpp-release-archives.py`: added `win32-arm64-msvc` to TARGETS;
  message now says 6-platform.
- `.github/workflows/release.yml`: npm loops cover all six platforms; wheel
  counts 25->30 (both jobs); C++ archive count 10->12; recovery label updated.
- `docs/releasing.md`: six-platform npm, 30 wheels + sdist, twelve C/C++ assets.

## Verification (Windows x64 host; ARM64 binaries cannot execute here)

- `rustup target add aarch64-pc-windows-msvc`: std installed OK.
- `cargo check -p glrmask-c -p glrmask-node --target aarch64-pc-windows-msvc`:
  blocked locally, no C compiler on this host (no VS/cl.exe; zstd-sys and blake3
  build scripts require one). No Rust sources changed, so nothing arch-specific
  to typecheck; full compile+test must run on CI runners.
- `check-python-wheel-set.py` with 30 synthetic wheels (6 platforms x 5
  interpreters incl. `win_arm64`): `validated 30 wheels and 1 sdist`.
- `check-cpp-release-archives.py` with 12 synthetic archives incl.
  `win32-arm64-msvc` static+shared: `Complete 6-platform ... 12 archives`.
- Negative test: hiding `cpp-native-win32-arm64-msvc` makes the checker fail
  (exit 1), so a half-added platform cannot silently pass.
- `bindings/ci/check.py` (no --library): passed, 61 declared C symbols.
- All three edited workflows parse as valid YAML (PyYAML safe_load).
- `scripts/sync-guide.py --check` fails identically on clean main63ce9e
  (stale `language-docs.json`, `__init__.pyi`): pre-existing, unrelated.
- No Node.js on this host, so `assemble.cjs`/`check-package.cjs` six-target
  logic verified by diff review only (one-line `releaseTargets` addition
  consumed generically).
- Full execution proof requires CI on `windows-11-arm` runners after push.

## Status

- [x] Implementation edits complete, committed on this branch.
- [ ] CI run on `windows-11-arm` (native-bindings + python-wheels) still needed;
      cannot execute ARM64 binaries on this x64 host.
- [ ] No merge/tag/publish (per task constraints).
