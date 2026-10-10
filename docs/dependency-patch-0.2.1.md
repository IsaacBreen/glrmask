# 0.2.1 dependency patch (unreleased)

Prepared from official v0.2.0 commit 9ef50c988199e2c7bcba6c29d44e5ecae5ceb2f3.
This is a dependency and binding compatibility patch, with no matcher algorithm,
optimization-default, serialized-format, or benchmark claim change. Root and
Python package versions are provisionally 0.2.1; all helper crates remain 0.1.1.

## Reviewed fixes

| Dependency | Locked version | Purpose |
| --- | --- | --- |
| PyO3 family | 0.29.3 | RUSTSEC-2026-0176 and RUSTSEC-2026-0177; Windows linking fixes |
| rust-numpy | 0.29.0 | Compatible PyO3 release; rejects misaligned Rust slice access |
| rand | 0.8.6 | RUSTSEC-2026-0097; root dependency minimum also raised |
| anyhow | 1.0.103 | RUSTSEC-2026-0190 |
| crossbeam-epoch | 0.9.20 | RUSTSEC-2026-0204 |

The public bindings do not directly use the reported PyO3 iterator operations or
closure constructor. Version presence alone does not establish exploitability.
The NumPy change is directly relevant to packed-mask mutable slices. Misaligned
buffers now produce ValueError; aligned int32 buffers retain in-place behavior.

Python detachment scopes and checked casts are preserved. Explicit gil_used=true
retains the pre-upgrade GIL requirement, and from_py_object retains Clone class
extraction rather than adopting a new PyO3 default. Public Python signatures and
exception types are retained; PyO3 can change argument-error diagnostics/notes.
No free-threaded support is newly claimed. The CPython 3.9–3.13 wheel matrix and
numpy>=1.21 declaration are unchanged. NumPy 1.x has no Python 3.13 wheels.
The oldest tested NumPy 1.x versions by interpreter are 1.21.6 (3.9/3.10),
1.23.5 (3.11), and 1.26.4 (3.12); these checks do not test every older patch.
PyO3/rust-numpy require Rust 1.83; the root already uses edition 2024 (Rust 1.85+).
Qualification uses Rust 1.95.0, also avoiding the documented older-toolchain
Mach-O string-pool issue on this Mac.

## Outstanding advisory findings

The complete locked registry graph was checked against the public OSV API.
The following remain, and the patch must not be described as an advisory-clean
full dependency graph:

- bincode 1.3.3: RUSTSEC-2025-0141 (unmaintained). Migrating serialization to
  a different major version is outside this binding patch.
- im 15.1.0: RUSTSEC-2023-0126 (OrdSet aliasing unsoundness) and
  RUSTSEC-2026-0248 (unmaintained), with no patched version.
- bitmaps 2.1.0: RUSTSEC-2026-0247 (unmaintained), with no patched version.
- sized-chunks 0.6.5: RUSTSEC-2026-0251 (unmaintained) and
  RUSTSEC-2026-0255 (panic-safety unsoundness), with no patched version.

These require separate, evidence-backed data-structure/serialization work;
reachability has not been established by this patch. Transitive fixes apply to
the supplied lockfile: downstream Rust consumers with their own older lockfile
must update it. No direct dependencies were added solely to pin transitives.

Primary references:
- https://pyo3.rs/v0.29.0/migration.html
- https://github.com/PyO3/pyo3/releases/tag/v0.29.3
- https://github.com/PyO3/rust-numpy/releases/tag/v0.29.0
- https://rustsec.org/advisories/RUSTSEC-2026-0176.html
- https://rustsec.org/advisories/RUSTSEC-2026-0177.html
- https://rustsec.org/advisories/RUSTSEC-2026-0097.html
- https://rustsec.org/advisories/RUSTSEC-2026-0190.html
- https://rustsec.org/advisories/RUSTSEC-2026-0204.html
- https://rustsec.org/advisories/RUSTSEC-2026-0255.html

## Qualification

The Python wheel workflow builds with --locked, installs each exact wheel in a
fresh virtual environment, runs the public examples and alignment/mode smoke,
and repeats supported installs with NumPy 1.x. The sdist is installed from its
exact generated archive. Matrix completeness is checked before aggregation.
Local qualification includes the Rust workspace regressions, maintained Python
suites, exact public surface/mask comparisons with installed 0.2.0, and loading
0.2.0 external-vocabulary artifacts in the candidate. Final results, source
commit, artifact hashes, and gaps are recorded with the preparation evidence;
a successful compile alone does not constitute qualification.
