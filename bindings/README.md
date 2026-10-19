# Native language bindings

The C++/C guide is [cpp/README.md](cpp/README.md); the native Node.js JavaScript/TypeScript guide is [node/README.md](node/README.md). Both use the same Rust engine. Browser WebAssembly and model/tokenizer adapters are separate integrations.

`core` is a private Rust ownership/validation helper, not another parser. It retains each compiled constraint with a safe owner/dependent `self_cell`, implements independent state snapshots, validates foreign inputs, and contains unwinding panics. C ABI code and Node-API code add only their language boundary and error/ownership conventions. The existing core algorithms, build defaults and public Rust/Python semantics are unchanged.

The current public lifecycle is source `Grammar` -> `PreparedGrammar` -> runnable `Constraint` -> per-sequence `ConstraintState`. Source/prepared binding mutates an independent snapshot handle transactionally. Bindings mirror the current Rust contract, not a separate proposed naming or composition redesign.

## Maintainer verification

Use an owned Cargo target directory. Normal production builds use `--release`; `--profile bindings-dev` is the incremental implementation loop. Do not mix those profiles when reporting performance.

```sh
cargo build --locked --release -p glrmask-c -p glrmask-node
cargo test --locked --release -p glrmask-binding-core -p glrmask-c
python bindings/ci/check.py
npm ci --ignore-scripts --prefix bindings/node
npm run build --prefix bindings/node
npm run check --prefix bindings/node
python bindings/cpp/scripts/validate-install.py --library-dir target/release --work-dir /new/owned/cpp-validation
node bindings/node/scripts/parity.cjs /new/owned/cpp-validation/build-static/glrmask_cpp_tests /new/owned/parity
```

On Windows, the parity executable has `.exe`. The install-validation script refuses reused output directories, verifies both static and shared variants, moves each installation, and compiles unrelated C/C++ consumers through `find_package`. `check.py --library PATH` additionally verifies every declared C symbol against the actual shared library.

The Node package's `prepack` guard validates versions, checksums, declaration re-exports, licenses and the current-platform binary. `scripts/test-installed.cjs PACKAGE.tgz NEW_DIRECTORY` tests a real package outside the source tree with install scripts disabled and no Rust toolchain on the runtime PATH. The parity helper compares masks/state transitions and exchanges embedded/external-vocabulary artifacts in both directions for all four optimization modes.

`.github/workflows/native-bindings.yml` runs Windows x64, Linux glibc x64/arm64 (glibc 2.28 portable container), Linux musl x64/arm64 (Alpine container), and macOS x64/arm64 release builds. It tests the same addon on Node 24 and Node 22, validates CMake packages and two-way artifacts, and assembles checksummed universal npm and per-platform C++ artifacts. No workflow publishes a registry package, creates a release tag, or bumps versions. Ordinary release authorization remains separate.

## Documentation ownership

The language READMEs are canonical for the hosted C++ and JavaScript API pages. The website imports them from one immutable library commit, rewrites relative source links to that revision, and records the source hash. Common public introduction and link prose remain under `docs/content`, generated with `scripts/sync-guide.py`. Do not edit generated root/package readmes or a hosted copy independently.
