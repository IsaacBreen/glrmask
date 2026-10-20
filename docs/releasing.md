# Publishing GLRMask (trusted releases)

Release workflow: [`.github/workflows/release.yml`](../.github/workflows/release.yml).

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
- **crates.io**: the `glrmask` *root crate*, using `rust-lang/crates-io-auth-action`
  and a temporary token which is revoked after the job;
- **C/C++**: sixteen tested, checksummed GitHub Release assets (eight platforms ×
  static/shared), including headers, native libraries, CMake installation and manifests.

For a rerun, an already-published npm/PyPI/crates.io version is skipped. Existing GitHub
C++ assets are never overwritten: a rerun verifies their recorded SHA-256 digests and
fails if they differ. Failure of a registry publish does not undo a successful publish
to another registry; check each registry independently before retrying.

**Important Rust workspace limit:** GLRMask's internal crates (e.g. `glrmask-vocab`,
`glrmask-grammar`) have their own versions and crates.io release order. This workflow
publishes only the root `glrmask` crate. Publish and verify any changed internal
dependencies separately before tagging a new root version. If automating all workspace
crates, each named crate needs its own trusted-publisher registration and dependency-
order-aware release path; do not assume `cargo publish --workspace` is suitable here.

## Coordinated Rust internal-crate staging for 0.4.0

The root `glrmask@0.4.0` uses APIs not included in published `glrmask-vocab@0.2.0`
and `glrmask-grammar@0.2.0`, so workspace-local tests **do not** establish
publishability. Nine internal source crates are now pinned to **0.3.0** in both
`Cargo.toml` dependencies and `Cargo.lock`; the unchanged four foundational
crates remain at published **0.1.1**. These are distinct immutable crates.io
versions, not repackaged 0.2.0 tarballs.

Run `python3 scripts/check-internal-crate-release.py` to check exact dependency
pins and the acyclic internal publication graph without contacting a registry;
`--require-published` additionally requires every internal crate version to be
visible and non-yanked on crates.io. **A published 0.3.0 version is not guaranteed
to have the required API merely because the version exists.** Once internal
packages are public, `cargo package --locked -p glrmask` must successfully
compile the root crate *against the actual published versions* before a release.
GitHub's release verifier runs both checks **before** npm, PyPI, crates.io or
C++ publication jobs can start.

Publish the nine internal 0.3.0 crates in this dependency order (within each
wave the crates may be published independently):

1. `glrmask-vocab`, `glrmask-artifact` (the unchanged 0.1.1 foundations exist).
2. `glrmask-lexer` (depends on the new vocab).
3. `glrmask-grammar` (depends on the new lexer).
4. `glrmask-glr`, `glrmask-json-schema` (depend on the new grammar/lexer).
5. `glrmask-dwa-merge` (depends on the new artifact and GLR).
6. `glrmask-parser-dwa`, `glrmask-terminal-dwa` (depend on those predecessors).

The opt-in-only workflow `.github/workflows/internal-crate-publish.yml` can
publish **one** explicitly selected internal crate per manually approved
workflow dispatch using crates.io's short-lived GitHub OIDC credential. The
owner must first configure **each of those nine crates'** crates.io **Trusted
Publishing → GitHub Actions** settings for owner `IsaacBreen`, repo `glrmask`,
workflow filename `internal-crate-publish.yml`, and environment `release`.
The release environment must allow the selected release-family tag; trusted
publishing is distinct for each crate. Do not store crates.io tokens in secrets.

Once the reviewed changes have merged to `main`, the owner may create a
**staging Git tag** such as `v0.4.0-staging-crates` pointing to that merged
commit; a tag **without a published GitHub Release** does not run the consumer
release. Dispatch the internal workflow **on that tag**, selecting one crate
and exact version `0.3.0` per run in the above dependency order. The workflow
refuses a non-tag ref, a tag unrelated to root `0.4.0`, an unmerged source
revision, a mismatched crate version, a version already on crates.io, or a
failed `cargo publish --dry-run`. It publishes only after the owner has
registered the correct trusted-publisher identity and GitHub Actions
`release` environment approves the run. Confirm each crate is public and
its downstream dependency resolution works before advancing to the next.

**Do not publish the GitHub `v0.4.0` Release yet** if any of these packages,
all eight npm platform identities/trusted publishers, PyPI root trust or
crates.io root trust is not ready. A partial multi-registry release is not
atomic and cannot be rolled back by deleting already published versions.

## One-time registry registration

GitHub repository **`IsaacBreen/glrmask`**, trusted workflow filename **`release.yml`**,
GitHub Actions environment **`release`** for all three OIDC services.

1. **npm**: first approve the initial staged `@glrmask/node@0.3.0` release so the
   actual package version is public. Repeat the same trusted-publisher setup
   for each of the eight `@glrmask/node-<platform>` packages: npm trust is
   per package, so every platform package needs its own **Settings → Trusted
   Publisher** entry before its first OIDC publish. Then open each package's npm **Settings → Trusted
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
3. **crates.io**: sign in, open the existing **`glrmask`** crate settings →
   Trusted Publishing → GitHub Actions. Enter owner `IsaacBreen`, repository
   `glrmask`, filename `release.yml`, environment `release`.
   The owner already published `glrmask 0.3.0`, fulfilling the initial-publication
   prerequisite.
   [Official crates.io instructions](https://crates.io/docs/trusted-publishing).
4. **C/C++**: **no external publishing account**. GitHub releases receive prebuilt
   binaries using the workflow's scoped, short-lived `GITHUB_TOKEN`. No NuGet,
   Conan Center, or vcpkg registry account is assumed or silently used.

Create/keep the GitHub environment named **`release`** and limit its eligible refs to
stable release tags (`v*`). This restricts the OIDC job identity. We do *not* store
npm, PyPI, crates.io, or GitHub personal publishing credentials in Actions secrets.
A published GitHub Release is the human decision to publish.

## Before publishing a new release

1. Update all three versions (Rust, Python, npm), documentation and changelog on `main`.
   Check that the new version is not already published on any registry. Do not
   repurpose the existing `v0.3.0` tag: it predates the merged native bindings.
2. If internal Rust crates changed version, publish them first in dependency order.
3. Confirm native and wheel CI are green on the source commit.
4. In GitHub, **Draft a new release**, select or create **`vX.Y.Z`** pointing to the
   desired merged `main` commit, then **Publish release**. The OIDC release workflow
   builds, checks and publishes all four channels. A PR or tag push alone will not.
5. Inspect the workflow, then verify install from npm and PyPI, root crate on
   crates.io, and static/shared C++ GitHub assets. An unavailable registry or failed
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
