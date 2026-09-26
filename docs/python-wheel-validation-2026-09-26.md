# Python artifact validation — September 26, 2026

The final integration was checked beyond `cargo check -p _glrmask`: an actual
macOS wheel was built, installed separately, imported, and exercised through the
checked-in Python tests. No installed development package was replaced.

## Source and successful result

The tested production code is byte-identical to integration commit
`1c93a53852f2d1ed74303ec42bd199b3454fc60f`. A validation checkout reconstructed its
`7d0f388ff` crash/build keepers plus `d26f65fc7` boundary changes. The following
Git tree/blob identities were checked against the actual integrated commit:

| Path | Git object |
| --- | --- |
| `src` | `5eea8cb28c6ba282c57b28ab53bdbe54c81e1624` |
| `crates` | `4f56a1b3f16151f3676cd2360a29cae91edbc106` |
| `python` | `37bbc42792511f0173eaa63cb9604e2c2312c0fe` |
| `Cargo.toml` | `327ae69d1a51fe100c798110023fb8112aef8e67` |
| `Cargo.lock` | `edb76d9f344fc4d0f62be23486b2280711ff2e26` |

On macOS 27.0 arm64, Rust 1.95.0 / LLVM 22.1.2, CPython 3.12, and the Apple C/C++
toolchain, the wheel passed all **42 Python tests**. A separate clean temporary
virtual-environment installation also passed the public smoke test and all 42
tests, using NumPy 2.5.3 and pytest 9.1.1. This is local macOS arm64 coverage, not
a claim that the complete wheel matrix or Python 3.9–3.13 matrix ran locally.

The validated wheel SHA256 is
`1eed79864e1055ca3b4db70cc8caf09ee4df39177fc939feefc8add12e21ffd0`.
The direct Python test log SHA256 is
`79d0059236333601c0358bb7b72aea5e067b9ee6edfb2a47b27c99e430a0fd8a`.
The independent clean-install test log SHA256 is
`0fe1bef21501006b1710ceda736a2c2f11e911948e43fd8ff859a35943cdbbb5`.

## Failures retained, not hidden

1. The first local build inherited a Conda `CC` pointing to a nonexistent
   compiler. Selecting `xcrun`'s Apple C/C++ tools for that build fixed the build
   environment without changing the user's shell configuration.
2. The first successfully built wheel could not load: macOS reported a
   misaligned `LINKEDIT` string pool. The compiler's dylib and packaged extension
   were byte-identical, so packaging had not corrupted the binary. Their
   `LC_SYMTAB.stroff` was 35,579,476 (not divisible by eight). Rebuilding with
   `CARGO_PROFILE_RELEASE_STRIP=none` produced an aligned string table and loaded.
   This matches the upstream stripping issue
   [rust-lang/rust#157750](https://github.com/rust-lang/rust/issues/157750) and
   [llvm/llvm-project#203678](https://github.com/llvm/llvm-project/issues/203678).
3. With the old local `1.91.0-nightly (2025-08-25)` compiler, the now-loadable
   extension still crashed while compiling a small GLRM grammar, in the static
   terminal-interchangeability path. The crash reproduced in a fresh process and
   with one Rayon worker. The **same source**, rebuilt using installed Rust
   1.95.0, passed both that reproducer and the complete Python suite. This is
   evidence of a toolchain-sensitive failure, not a general proof about all
   intermediate compiler versions or the absence of undefined behavior.

## Kept packaging checks

The macOS wheel workflow explicitly selects the validated stable compiler and
disables symbol stripping. Windows and Linux masking/build policies are unchanged.
The clean-install artifact gate can run the full Python regression suite with
`--tests`, removes ambient Python search-path overrides, and disables unrelated
pytest plugin autoloading. The smoke test now includes the small fused-vocabulary
GLRM compile that the old toolchain failed. Crate changes trigger the wheel
workflow as well as facade changes.

These are build-configuration, test, and documentation changes. They do not alter
the Rust compiler algorithms, masking defaults, or the final integrated benchmark
raws. The unsuccessful local wheel artifacts and diagnostic logs remain separate
from the successful validation evidence. No package was published to PyPI and no
release tag was created by this validation.
