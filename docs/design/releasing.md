> Design notes from development. Type and option names may be out of date; [GUIDE.md](../../GUIDE.md) describes the current API.

# Publishing GLRMask (trusted releases)

Release workflow: [`.github/workflows/release.yml`](../../.github/workflows/release.yml).

## What publishes, and what does not

Publishing a GitHub Release with a **new stable tag** (`vMAJOR.MINOR.PATCH`) runs the release
workflow. Merely pushing to `main`, opening a PR, or creating a tag does **not** publish
anything. The release tag must point to a commit already on `main`, and these versions
must all match its tag:

- `Cargo.toml` root `glrmask` version
- `python/pyproject.toml` version
- `bindings/node/package.json` `@glrmask/node` version

The workflow reuses the native C++/Node and Python wheel build/test workflows at that
tag, and only publishes after their complete validation jobs pass. It releases:

- **npm**: the thin `@glrmask/node` root plus eight prebuilt
  `@glrmask/node-<platform>` optional-dependency packages (Windows x64/arm64,
  macOS x64/arm64, Linux x64/arm64 × glibc/musl) through OIDC, with no
  shared npm credential or post-install native compilation;
- **PyPI**: eight regular CPython abi3 wheels (manylinux, musllinux, macOS and Windows) plus sdist, using PyPI's short-lived OIDC credential;
- **crates.io**: the `glrmask` root crate and every changed internal crate, in
  dependency order, using `rust-lang/crates-io-auth-action` and a temporary token
  which is revoked after the job;
- **C/C++**: sixteen tested, checksummed GitHub Release assets (eight platforms ×
  static/shared), including headers, native libraries, CMake installation and manifests.

For a rerun, an already-published npm/PyPI/crates.io version is skipped. Existing GitHub
C++ assets are never overwritten: a rerun verifies their recorded SHA-256 digests and
fails if they differ. Failure of a registry publish does not undo a successful publish
to another registry; check each registry independently before retrying.

**Rust workspace crates.** GLRMask's internal crates (e.g. `glrmask-vocab`,
`glrmask-grammar`) have their own versions. `scripts/publish-crates.py` publishes every
workspace crate whose version is not yet on crates.io, dependencies first, after
packaging and verifying all of them together. Before uploading anything it fails if a
version that is already published has different sources, which means a crate changed
without a version bump: bump it, and the `=x.y.z` requirements on it, then retry. Run
`python3 scripts/publish-crates.py` locally to do the same check without publishing.

## One-time registry registration

GitHub repository **`IsaacBreen/glrmask`**, trusted workflow filename **`release.yml`**,
GitHub Actions environment **`release`** for all three OIDC services.

1. **npm**: first approve the initial staged `@glrmask/node@0.3.0` release so the
   actual package version is public. Repeat the same trusted-publisher setup
   for each of the eight `@glrmask/node-<platform>` packages: npm trust is
   per package, so every platform package needs its own **Settings → Trusted
   Publisher** entry before its first OIDC publish. A package must exist on npm before
   it has settings, so first run `node scripts/bootstrap-npm-platform-packages.cjs`
   while logged in to npm (`npm login`); it publishes a `0.0.0-stage` placeholder for
   each platform package that doesn't exist yet. Then open each package's npm **Settings → Trusted
   Publisher** and choose **GitHub Actions**. Use owner `IsaacBreen`, repository
   `glrmask`, workflow **`release.yml`**, environment **`release`**, and enable
   **Allow npm publish** (not stage-only, which would require 2FA every time).
   The account owner must complete an interactive 2FA challenge. npm documentation
   says a newly added trust configuration must be used for a successful first publish
   within two days, or it will expire: register this near the **next version release**.
   A successful trusted publish enables provenance for this public package.
   The release workflow publishes the eight platform packages first (with
   per-package existence checks and retries) and the thin root last, so a
   fresh install always resolves every optional dependency.
   [Official npm instructions](https://docs.npmjs.com/trusted-publishers/).
2. **PyPI**: sign in to PyPI, manage existing project **`glrmask`**, open
   **Publishing → Add a trusted publisher → GitHub Actions**. Enter owner
   `IsaacBreen`, repository `glrmask`, filename `release.yml`, environment
   `release`. This is a separate setting on PyPI, not an npm setting.
   [Official PyPI instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
3. **crates.io**: create a crates.io API token under **Settings → API Tokens** with
   the **`trusted-publishing`** endpoint scope (the UI describes it as "Manage
   trusted publishing configurations") and a crate scope covering the crates (for
   example `glrmask*`), then run from a clean `main` checkout:

   ```sh
   CRATES_IO_TOKEN=... python3 scripts/register-trusted-publishers.py --dry-run
   CRATES_IO_TOKEN=... python3 scripts/register-trusted-publishers.py --apply
   ```

   The script registers owner `IsaacBreen`, repository `glrmask`, workflow
   `release.yml`, environment `release` on every publishable crate: the root
   `glrmask` crate and its internal crates `glrmask-invariant`,
   `glrmask-finite-automata`, `glrmask-weight`, `glrmask-weighted-automata`,
   `glrmask-artifact`, `glrmask-vocab`, `glrmask-lexer`, `glrmask-grammar`,
   `glrmask-glr`, `glrmask-json-schema`, `glrmask-dwa-merge`,
   `glrmask-parser-dwa` and `glrmask-terminal-dwa`. It lists each crate's
   existing configurations and skips crates that already have an identical one,
   so reruns are safe; a crate added later is included automatically once it is
   a publishable workspace member. All of them already have a published version,
   fulfilling the initial-publication prerequisite. The token must belong to a
   crate owner whose crates.io account has a linked, verified GitHub account,
   because crates.io uses it to resolve the repository owner. Revoke the token
   after use.

   Manual fallback: sign in, open the existing **`glrmask`** crate settings →
   Trusted Publishing → GitHub Actions. Enter owner `IsaacBreen`, repository
   `glrmask`, filename `release.yml`, environment `release`. Repeat this for
   every internal crate listed above: crates.io trust is per crate. A crate
   added later must be published once by hand before trusted publishing can be
   enabled.
   [Official crates.io instructions](https://crates.io/docs/trusted-publishing).
4. **C/C++**: **no external publishing account**. GitHub releases receive prebuilt
   binaries using the workflow's scoped, short-lived `GITHUB_TOKEN`. No NuGet,
   Conan Center, or vcpkg registry account is assumed or silently used.

Create/keep the GitHub environment named **`release`** and limit its eligible refs to
stable release tags (`v*`). This restricts the OIDC job identity. We do *not* store
npm, PyPI, crates.io, or GitHub personal publishing credentials in Actions secrets.
A published GitHub Release is the human decision to publish.

## Before publishing a new release

1. Update all four versions (Rust, Python, npm, C++ CMake), documentation and
   changelog on `main`. The C++ CMake project version in
   `bindings/cpp/CMakeLists.txt` must match the `bindings/cpp` crate version.
   Check that the new version is not already published on any registry. Do not
   repurpose the existing `v0.3.0` tag: it predates the merged native bindings.
2. If an internal Rust crate changed, bump its version. `python3 scripts/publish-crates.py`
   checks this.
3. Confirm native and wheel CI are green on the source commit.
4. In GitHub, **Draft a new release**, select or create **`vX.Y.Z`** pointing to the
   desired merged `main` commit, then **Publish release**. The OIDC release workflow
   builds, checks and publishes all four channels. A PR or tag push alone will not.
5. Inspect the workflow, then verify install from npm and PyPI, root crate on
   crates.io crates, and static/shared C++ GitHub assets. An unavailable registry or failed
   artifact build should fail safely rather than publish unverified binaries.

## Security and maintenance

Only the npm, PyPI and Rust publishing jobs get `id-token: write`; only the C++
GitHub Release uploader gets `contents: write`. All compilers/test runners have
read-only repository permissions. Review workflow edits carefully, and restrict
maintainers' ability to create release tags. Once OIDC is proven, use npm's
**Require two-factor authentication and disallow tokens** if desired; npm's
trusted publisher keeps working under that setting. Crates.io has its own
trusted-publishing-only setting.

For later C++ dependency-manager adoption, ConanCenter/vcpkg would be a separate
package submission and maintenance process. Binary GitHub Release assets are the
currently supported self-contained distribution format.

## Compatibility build layout

See [Supported platforms and installation](compatibility.md) for the complete eight-platform matrix, the ABI3 Python runtime tests, and the thin npm package layout. These source changes require a new release; they do not replace the existing 0.3.1 packages.
