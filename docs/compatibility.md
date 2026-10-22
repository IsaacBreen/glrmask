# Supported platforms and requirements

These requirements apply to the current source tree. For prebuilt packages, use the platform list on the corresponding release page.

## Native platforms

- Windows x64 and arm64 with MSVC.
- macOS 11 or newer on x64 and arm64.
- Linux x64 and arm64 with glibc 2.28 or newer, built in `manylinux_2_28` containers.
- Linux x64 and arm64 with musl 1.2 or newer, including Alpine.

C and C++ consumers need C++17 for the C++ facade and only the C header for the C ABI. Building either from source requires a current stable Rust toolchain, CMake 3.20 or newer, and the platform C/C++ compiler. A native artifact built for one operating system, architecture, or C library does not run on another. Use matching headers and libraries from the same build.

Node.js requires Node.js 22 or newer. The addon uses Node-API and targets the same operating systems, architectures, and Linux C libraries listed above. Building from source additionally requires a stable Rust toolchain and the platform native compiler and linker. The package's platform binary is version-matched to the JavaScript root and loaded through its platform-specific dependency; there is no fallback to a different runtime or binary.

## Python

The extension supports ordinary GIL-enabled CPython 3.9 through 3.15. The wheel build produces one `cp39-abi3` wheel per platform. glibc wheels keep their manylinux2014 / glibc 2.17 baseline; Linux musl variants are built separately. A wheel installation needs no Rust toolchain; a source distribution requires Rust and the platform C compiler.

Free-threaded CPython is not supported. `abi3` is not the free-threaded ABI, and the extension declares that it uses the GIL. `cp314t` and `abi3t` wheels are not part of this support statement.

## Rust

The crate is distributed as source and is built with a current stable Rust toolchain for the application's target. Cargo selects `README.rust.md` as the package README.

## Artifact compatibility

Saved constraints and prepared grammars use versioned Rust core formats, not a language-level interchange standard. They can be exchanged between compatible builds of Rust, Python, C++, and Node, but compatibility across releases, architectures, or unrelated builds is not guaranteed. Persist the source and configuration alongside an artifact when it must be regenerated later. Vocabulary identity is validated exactly when a vocabulary is supplied; equal size is not sufficient.
