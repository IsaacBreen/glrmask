# Supported platforms and installation

The compatibility changes in this branch target the **next release**. Published
0.3.1 distributions retain their original five-platform, Python 3.9–3.13 layout.
No existing package version is overwritten when this change is merged.

## Binary matrix

| Operating system / runtime | x86-64 | ARM64 |
| --- | --- | --- |
| Windows, MSVC | C/C++, Node.js, Python | C/C++, Node.js, Python |
| macOS 11 or later | C/C++, Node.js, Python | C/C++, Node.js, Python |
| Linux, glibc | C/C++, Node.js, Python | C/C++, Node.js, Python |
| Linux, musl 1.2 / Alpine | C/C++, Node.js, Python | C/C++, Node.js, Python |

Native Linux Node/C++ libraries build and execute in `manylinux_2_28` containers,
with an enforced **glibc 2.28** baseline. Python's manylinux wheels retain their
manylinux2014 / glibc 2.17 baseline. musl artifacts are separately built and tested
inside musl containers: they are not renamed glibc artifacts. All native targets
run C/C++ static/shared relocated-consumer tests, Node runtime tests, and installed
npm package tests on the matching architecture and C library.

C/C++ produces **16** archives: static and shared installations for eight targets.
Each archive contains matching headers, libraries, CMake configuration and checksums.
Rust packages remain source distributions; applications compile for their own target.

## Python ABI and runtime support

Ordinary, GIL-enabled CPython **3.9 through 3.15** uses one `cp39-abi3` wheel per
platform. Eight platform wheels replace the previous per-interpreter build matrix;
there is also one source distribution. CI installs the same wheel on Python 3.9,
3.12, 3.14 and 3.15, including musl runtimes. Windows ARM64 is tested on 3.12, 3.14
and 3.15 because older native interpreter/dependency distributions are unavailable.

PyO3 and rust-numpy have been upgraded together. Public masks remain NumPy arrays
and the public grammar/constraint/state API is unchanged. The source distribution
requires Rust and a native compiler; a supported wheel installation does not.

**Free-threaded CPython is not qualified by this change.** `abi3` is not the
free-threaded ABI. The extension explicitly declares `gil_used = true`; neither
`cp314t` nor `abi3t` wheels are published here. Advertising those requires a separate
thread-safety audit and runtime stress tests. The ordinary Python 3.15 wheel remains
supported without making a free-threading claim.

## Node.js package size and loading

`@glrmask/node` becomes a small JS/TypeScript root package. npm selects one exactly
version-pinned `@glrmask/node-<platform>` optional dependency using `os`, `cpu` and,
on Linux, `libc`. Only that platform package carries a native `.node` binary.

Keep optional dependencies enabled. `--omit=optional` intentionally leaves out the
binary and loading fails with an error naming the required package. There is no
silent Python/WASM fallback, post-install download or local compiler invocation.
CommonJS, ESM and TypeScript entry points are retained. Package/binary version
mismatches are rejected. CI checks the root contains no native binary and validates
all platform checksums before any publication.

For local/offline tarball installs, install both the root and matching platform
archive together. The CI `glrmask-node-npm-dist` artifact contains all nine tarballs.
The published package name used by applications remains `@glrmask/node`.

## First publication of split npm packages

npm trusted publishing is configured **per package**. The eight new platform
packages require an initial authorized publication and their own trusted-publisher
registration, using the same `IsaacBreen/glrmask`, `release.yml`, `release` identity.
The existing root's trust configuration does not authorize new package names.
Bootstrap these once before releasing the new thin root; the release workflow
publishes platforms before the root. Do not publish a root whose exact-version
platform dependencies are absent. Subsequent versions use the same OIDC flow.

The existing public 0.3.1 npm package is unaffected until a new version is released.
Browser WebAssembly and independent language version numbers are separate projects.