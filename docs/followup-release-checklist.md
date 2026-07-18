# Follow-up release checklist for the vLLM prerequisite APIs

> **Status:** release checklist for `0.1.1`. This document does not publish an artifact. Public `0.1.0` remains incompatible with the vLLM backend.

## Required source gate

The release source must contain the semantically integrated versions of:

- bounded rollback with zero history by default;
- non-mutating `validate_tokens`;
- `is_failed`;
- grammar-level end-token IDs across JSON Schema, EBNF, Lark, and GLRM constructors;
- end-token-aware packed-mask extent;
- focused Rust and Python tests for all of the above.

The old prerequisite branch is validation evidence only. The release must use the lifecycle API integrated with grammar-level end-token semantics on the unified current development line.

## Version surfaces

The three enforced publication surfaces must all read `0.1.1`:

```text
Cargo.toml                  [package].version
python/Cargo.toml           [package].version
python/pyproject.toml       [project].version
```

Keep `Cargo.lock` and the dated changelog entry synchronized with those manifests. `scripts/release-artifact-dry-run.sh` rejects a mismatch among these three versions.

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

> The GLRMask backend requires `glrmask >= 0.1.1`. Public `glrmask 0.1.0` is incompatible because it does not expose bounded rollback, non-mutating token validation, failed-state inspection, or grammar-level end-token constructors.

After publication, use `0.1.1` in the vLLM dependency metadata, optional-dependency error, RFC body, support matrix, and reproduction instructions.

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
Constraint.from_json_schema(..., end_token_ids=[...])
Constraint.from_ebnf(..., end_token_ids=[...])
end token included in mask_len(), admitted after the base language, and committed to complete the augmented grammar
```

## Tag, build, publish, and verify

Use `0.1.1` and `<RELEASE_COMMIT>` until the exact release commit is approved.

1. Confirm source identity and cleanliness:

   ```bash
   git status --short
   git rev-parse HEAD
   test "$(git rev-parse HEAD)" = "<RELEASE_COMMIT>"
   ```

2. Create and push the annotated tag only after the pre-tag gate passes:

   ```bash
   git tag -a "v0.1.1" <RELEASE_COMMIT> -m "GLRMask 0.1.1"
   git push origin "v0.1.1"
   ```

3. Wait for the tag-triggered `Python wheels` workflow to pass all jobs. Download its combined `python-release-artifacts` artifact into an empty `dist/` directory, then recheck locally:

   ```bash
   python -m pip install --upgrade packaging twine
   python -m twine check dist/*
   python scripts/check-python-wheel-set.py dist
   test "$(find dist -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')" = 25
   test "$(find dist -maxdepth 1 -name '*.tar.gz' | wc -l | tr -d ' ')" = 1
   shasum -a 256 dist/* > "glrmask-0.1.1-artifacts.sha256"
   ```

4. Publish the Rust crate from the exact tagged source:

   ```bash
   cargo publish --locked -p glrmask
   ```

5. Publish the already validated Python artifact set without rebuilding it:

   ```bash
   python -m twine upload dist/*
   ```

6. Create a non-draft, non-prerelease GitHub Release for `v0.1.1` using the finalized changelog entry and attach the checksum manifest. Do not attach locally rebuilt replacement wheels.

7. Verify the public Python package in a fresh environment with no local or alternate-index fallback:

   ```bash
   python3.12 -m venv /tmp/glrmask-public-python
   PIP_CONFIG_FILE=/dev/null \
     /tmp/glrmask-public-python/bin/python -m pip install \
     --index-url https://pypi.org/simple --no-cache-dir \
     "glrmask==0.1.1"
   /tmp/glrmask-public-python/bin/python -c \
     'import importlib.metadata, glrmask; assert importlib.metadata.version("glrmask") == "0.1.1"; print(glrmask.__file__)'
   ```

8. Verify the public Rust crate in a fresh consumer with an exact dependency:

   ```toml
   [dependencies]
   glrmask = "=0.1.1"
   ```

   Run `cargo generate-lockfile` followed by `cargo build --locked`, inspect `Cargo.lock` for a crates.io source and checksum, then run a rollback/EOS lifecycle smoke.

9. Verify tag identity and registry state:

   ```bash
   git ls-remote origin "refs/tags/v0.1.1^{}"
   cargo info "glrmask@0.1.1"
   python -m pip index versions glrmask --index-url https://pypi.org/simple
   ```

10. Only after public registry smokes pass, replace the vLLM packet's placeholder with the released minimum version and run the frozen backend against the public artifact.

## Artifact and registry checklist

- [ ] Integrated prerequisite branch is reviewed and approved.
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
