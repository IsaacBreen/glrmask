# Follow-up release checklist for the vLLM prerequisite APIs

> **Status:** preparation for `<NEXT_VERSION>`. This document does not select a version, publish an artifact, or make the current public `0.1.0` compatible with vLLM.

## Required source gate

The release source must contain the semantically integrated versions of:

- bounded rollback with zero history by default;
- non-mutating `validate_tokens`;
- `is_failed`;
- explicit EOS in Python vocabulary constructors;
- EOS-aware packed-mask extent;
- focused Rust and Python tests for all of the above.

The preserved prerequisite source is `c13e5d857a9366221949bb73f6224342d2330335`. It is validation evidence, not the final release head. The release must use the prerequisite integrated onto the current intended release line and rerun the complete gate there.

## Version surfaces

After the owner selects `<NEXT_VERSION>`, update these three enforced publication surfaces together:

```text
Cargo.toml                  [package].version
python/Cargo.toml           [package].version
python/pyproject.toml       [project].version
```

Then update `Cargo.lock` and replace the changelog placeholder and release date. `scripts/release-artifact-dry-run.sh` rejects a mismatch among these three versions.

The other workspace packages are not separate registry deliverables in this release path. In particular, `glrmask-runtime`, `python-runtime`, `glrmask-wasm`, and `glrmask-browser-artifact` have `publish = false` where applicable. Their internal versions do not determine the crates.io or PyPI version, although they may be advanced for a deliberate workspace-versioning policy.

## Python support and artifact matrix

Python metadata remains:

```text
requires-python = ">=3.9,<3.14"
```

The existing `Python wheels` workflow expects exactly **25 wheels plus one sdist**:

| Python | manylinux x86_64 | manylinux aarch64 | macOS x86_64 | macOS arm64 | Windows x86_64 |
|---|---:|---:|---:|---:|---:|
| 3.9 | wheel | wheel | wheel | wheel | wheel |
| 3.10 | wheel | wheel | wheel | wheel | wheel |
| 3.11 | wheel | wheel | wheel | wheel | wheel |
| 3.12 | wheel | wheel | wheel | wheel | wheel |
| 3.13 | wheel | wheel | wheel | wheel | wheel |

Linux wheels use manylinux2014 compatibility. The workflow also builds one source distribution, runs `twine check`, clean-installs every individually built artifact, and validates the complete 25-wheel matrix with `scripts/check-python-wheel-set.py`.

## Automation boundary

A `v*` tag or manual dispatch runs `.github/workflows/python-wheels.yml`. It builds and validates artifacts, but it does **not**:

- upload to PyPI;
- run `cargo publish`;
- create a GitHub Release.

A patch release can reproduce the `0.1.0` platform coverage, but registry publication and GitHub Release creation remain explicit manual steps after the workflow succeeds.

## Minimum-version wording for vLLM

Before the version is selected:

> The GLRMask backend requires `glrmask >= <NEXT_VERSION>`. Public `glrmask 0.1.0` is incompatible because it does not expose bounded rollback, non-mutating token validation, failed-state inspection, or the explicit-EOS constructor and mask-extent contract.

After publication, replace `<NEXT_VERSION>` with the exact released version in the vLLM dependency metadata, optional-dependency error, RFC body, support matrix, and reproduction instructions.

## Pre-tag gate

Run from a clean release worktree at the exact proposed commit:

```bash
cargo fmt --check
cargo check --workspace --locked
cargo test --locked
cargo package --locked -p glrmask
cargo +stable publish --dry-run --locked -p glrmask
PYTHON_BIN=python3.12 scripts/release-artifact-dry-run.sh
```

Also run focused lifecycle tests and an installed wheel smoke that checks:

```text
Constraint.start(max_rollback_tokens=...)
rollback(n)
validate_tokens(...)
is_failed()
Vocab.from_id_to_bytes(..., eos_token_id=...)
EOS included in mask_len() and admitted only at completion
```

## Tag, build, publish, and verify

Use `<NEXT_VERSION>` and `<RELEASE_COMMIT>` literally until the owner chooses and approves them.

1. Confirm source identity and cleanliness:

   ```bash
   git status --short
   git rev-parse HEAD
   test "$(git rev-parse HEAD)" = "<RELEASE_COMMIT>"
   ```

2. Create and push the annotated tag only after the pre-tag gate passes:

   ```bash
   git tag -a "v<NEXT_VERSION>" <RELEASE_COMMIT> -m "GLRMask <NEXT_VERSION>"
   git push origin "v<NEXT_VERSION>"
   ```

3. Wait for the tag-triggered `Python wheels` workflow to pass all jobs. Download its combined `python-release-artifacts` artifact into an empty `dist/` directory, then recheck locally:

   ```bash
   python -m pip install --upgrade packaging twine
   python -m twine check dist/*
   python scripts/check-python-wheel-set.py dist
   test "$(find dist -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')" = 25
   test "$(find dist -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')" = 1
   shasum -a 256 dist/* > "glrmask-<NEXT_VERSION>-artifacts.sha256"
   ```

4. Publish the Rust crate from the exact tagged source:

   ```bash
   cargo publish --locked -p glrmask
   ```

5. Publish the already validated Python artifact set without rebuilding it:

   ```bash
   python -m twine upload dist/*
   ```

6. Create a non-draft, non-prerelease GitHub Release for `v<NEXT_VERSION>` using the finalized changelog entry and attach the checksum manifest. Do not attach locally rebuilt replacement wheels.

7. Verify the public Python package in a fresh environment with no local or alternate-index fallback:

   ```bash
   python3.12 -m venv /tmp/glrmask-public-python
   PIP_CONFIG_FILE=/dev/null \
     /tmp/glrmask-public-python/bin/python -m pip install \
     --index-url https://pypi.org/simple --no-cache-dir \
     "glrmask==<NEXT_VERSION>"
   /tmp/glrmask-public-python/bin/python -c \
     'import importlib.metadata, glrmask; assert importlib.metadata.version("glrmask") == "<NEXT_VERSION>"; print(glrmask.__file__)'
   ```

8. Verify the public Rust crate in a fresh consumer with an exact dependency:

   ```toml
   [dependencies]
   glrmask = "=<NEXT_VERSION>"
   ```

   Run `cargo generate-lockfile` followed by `cargo build --locked`, inspect `Cargo.lock` for a crates.io source and checksum, then run a rollback/EOS lifecycle smoke.

9. Verify tag identity and registry state:

   ```bash
   git ls-remote origin "refs/tags/v<NEXT_VERSION>^{}"
   cargo info "glrmask@<NEXT_VERSION>"
   python -m pip index versions glrmask --index-url https://pypi.org/simple
   ```

10. Only after public registry smokes pass, replace the vLLM packet's placeholder with the released minimum version and run the frozen backend against the public artifact.

## Artifact and registry checklist

- [ ] Integrated prerequisite branch is reviewed and approved.
- [ ] `<NEXT_VERSION>` selected by the owner.
- [ ] Three enforced version surfaces and `Cargo.lock` agree.
- [ ] Changelog placeholder replaced with the exact version/date.
- [ ] Full Rust/source gate passes at `<RELEASE_COMMIT>`.
- [ ] Local host wheel, sdist, and crate package pass metadata and installed-artifact smoke.
- [ ] Annotated tag peels to `<RELEASE_COMMIT>`.
- [ ] GitHub Actions produces 25 wheels and one sdist from that tag.
- [ ] Complete artifact set passes `twine check` and matrix validation.
- [ ] Checksums recorded before registry upload.
- [ ] crates.io publication succeeds and is not yanked.
- [ ] PyPI publication contains one sdist and all 25 wheels.
- [ ] GitHub Release is live, non-draft, and non-prerelease.
- [ ] Fresh crates.io-only and PyPI-only lifecycle smokes pass.
- [ ] vLLM minimum version and public-artifact reproduction are updated and retested.
