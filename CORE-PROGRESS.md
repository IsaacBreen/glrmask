# Core Module/Expr implementation progress

## Milestone 1: baseline and API contract

- Worktree verified clean at cd22e24640c5878af8bd1180ab9430d5dad3eda2.
- Read AGENTS.md. No delegates, providers, network tasks or integration actions.
- Existing PreparedGrammar is compiled-only and prepare_with ignores optimization;
  Module will use a separate source/preparation graph and the existing compiler/linker.
- CORE-API.md records the wrapper-facing signatures and option separation.
- Development environment: CARGO_INCREMENTAL=1, RUSTC_WRAPPER empty,
  CARGO_BUILD_JOBS=3, target = supplied glrmask-final-api-work-469935/target-dev.
- Tests not yet run.

## Milestone 2: source/preparation graph, Expr and versioned storage

- Added separate owned Module graph with persistent occurrence bindings, atomic binds,
  per-definition options, local prepared work and closed compilation.
- Added immutable Expr choice/sequence with one root and positional `leaf0`, `leaf1`
  paths; operators add no namespace. Empty choice is explicitly rejected; sequence
  empty is epsilon. CORE-API.md updated to the actual positional spelling.
- Added GLRASM01 storage graph with source/preparation interning, backward references,
  topology/resource checks and complete exact-event target vocabulary identity.
- Shared component shell lowering now also supports native Balanced local preparation.
- Canonical UnboundExtern is exported. Old Requirement remains a compatibility alias.
- First offline incremental cargo check is running. The supplied target starts cold;
  dependencies are building from the local cache. No network access requested.

## Milestone 3: initial passing regression batch

Exact command (provided development environment, offline):

```
GLRMASK_EXPLICIT_TARGET_OK=1 CARGO_INCREMENTAL=1 RUSTC_WRAPPER= CARGO_BUILD_JOBS=3 RUST_MIN_STACK=16777216 cargo test --offline -p glrmask --test module_expr_core --test final_public_api --test public_optimization_modes --test prepared_components
```

Output: test-profile rebuild 17.34s; Module/Expr 11 passed; final_public_api
24 passed; prepared_components 13 passed; public_optimization_modes 6 passed.
Total 54 passed, 0 failed. Original cold test build took 1m09s. A first test
iteration found the synthesized epsilon spelling (`eps`, corrected from `empty`).
No release rebuilds or benchmarks run. Existing forcing semantics are untouched.

Added deeper follow-up coverage for lexical scope/mask/forced_tokens parity,
source ownership, atomic vocabulary mismatch, compact DAGs, physical LR/native
storage, and internal Arc identity checks for reused local preparation. These are
in progress. Cached Grammar.module() snapshots stay cold/default; configured or
prepared Module values never write options or compiled work back into Grammar.

## Milestone 4: final capability and artifact checks

- New Module/Expr suite now has 19 tests, including direct Grammar.bind(Expr),
  configured/prepared source-expression rejection, native mixed modes, exact IDs,
  scoped lexer mask/forced_tokens parity, compact DAGs and source ownership.
- Five internal tests cover cheap cached Grammar.module snapshots, independent
  source/options, shared local Arc preparation before/after occurrence edits and
  load, malformed topology/options/exact targets, and retained boundary work.
- The boundary test explicitly decodes deferred retained metadata. GLR artifact
  loading preserves trigger programs in its deferred composition section;
  Module finalization restores that section without compiler calls.
- Raw Grammar.bind Expr conversion is fallible and atomic. Module.bind accepts
  configured/prepared Expr directly. Canonical normal dependency introspection
  is UnboundExtern/unbound/externs; legacy Requirement/requirements is tooling-only.
- Compiler-generated closed Lark inline schemas work across all four local modes.
  FastBuild with both inline schemas and open imports fails explicitly. FastBuild
  plus Exact boundary detail is rejected atomically. Explicit Module attachment
  edges involving FastBuild fail before preparation/final linking; standalone
  FastBuild retains its serialized LR representation.
- No shortest syntax, forcing implementation, Python, binding or documentation
  prose files were edited by this worker. Other workers' shared-worktree edits
  appeared during final verification and will remain outside this commit.

Verification evidence (all logs retained, no network):

- /private/tmp/glrmask-module-final-focused.log: 131 passed across 10 integration
  suites before adding the direct Grammar.bind(Expr) test.
- /private/tmp/glrmask-module-normal-qualified.log: ordinary dependency library
  cargo check --offline -p glrmask --lib passed, 11.62s.
- /private/tmp/glrmask-module-source-qualified.log: final_public_api 24 passed,
  module_expr_core 19 passed; incremental test rebuild 28.07s.
- /private/tmp/glrmask-module-unit-verified.log: five internal tests passed before
  the source-only direct Expr bind addition. Rechecking the final source below.
- Earlier internal fixture failures were corrected: the name `t` is reserved in
  native GLRM, so that fixture now uses `mark`; checking deferred metadata now
  materializes the saved link section rather than assuming eager load fields.

The stable supplied development target and CARGO_INCREMENTAL=1 were retained
throughout. Warm source edit/check times included 11.62s and 13.60s; source
edit/test rebuilds included 14.57s, 15.45s, 17.34s, 22.87s and 28.07s (test
binary rebuilds, not production performance results). Root unit binary rebuilds
also include this repository's large cfg(test) corpus. No benchmark was run.
No production profile or existing caches were invalidated.

Full consolidated canonical correctness qualification and matched production
build/link/TBM acceptance remain parent integration gates. This worker performs
focused regression verification and an owned local commit only.

## Milestone 5: final focused verification and commit scope

Final source passed:

```
GLRMASK_EXPLICIT_TARGET_OK=1 CARGO_INCREMENTAL=1 RUSTC_WRAPPER= CARGO_BUILD_JOBS=3 cargo check --offline -p glrmask --lib
GLRMASK_EXPLICIT_TARGET_OK=1 CARGO_INCREMENTAL=1 RUSTC_WRAPPER= CARGO_BUILD_JOBS=3 RUST_MIN_STACK=16777216 cargo test --offline -p glrmask --lib public_api::module
GLRMASK_EXPLICIT_TARGET_OK=1 CARGO_INCREMENTAL=1 RUSTC_WRAPPER= CARGO_BUILD_JOBS=3 RUST_MIN_STACK=16777216 cargo test --offline -p glrmask --test module_expr_core --test final_public_api --test final_public_api_review --test prepared_components --test public_optimization_modes --test lark_compatibility --test token_range_bindings --test marker_events --test marker_commit --test forcing_semantics
```

Results: normal library check passed (11.62s); 5 internal tests passed (0.07s
runtime; 38.84s unit-binary rebuild); 132 integration tests passed across all 10
suites, 0 failed. This includes 19 new public Module/Expr tests and 113 existing
regression tests. `git diff --check` passed. Final logs:

- /private/tmp/glrmask-module-normal-qualified.log
- /private/tmp/glrmask-module-local-qualified.log
- /private/tmp/glrmask-module-acceptance-focused.log

Owned commit scope (11 files): Cargo.toml; crates/glrmask-vocab/src/lib.rs;
src/import/mod.rs; src/lib.rs; src/public_api.rs; src/public_api/module.rs;
src/public_api/expr.rs; src/public_api/module/artifact.rs;
tests/module_expr_core.rs; CORE-API.md; CORE-PROGRESS.md.

The vocabulary change exposes only a doc-hidden target_vocab accessor needed to
persist ExactTokens' complete target identity. Shared compiler shell lowering is
factored once for ordinary Dynamic, native Balanced and Static callers. No lexer,
masking, commit, forcing or shortest implementation is forked or changed.
CORE-API.md and this progress artifact are excluded from the published package.

No push, merge, tag, publication, network task, provider task or delegation was
performed. Other workers' Python/bindings/docs files remain outside this commit.

## Commit outcome: sandbox-blocked, owned diff preserved

`git add` failed before staging:

```
fatal: Unable to create '/Users/isaacbreen/Projects2/glrmask2/.git/worktrees/glrmask-final-module-469935-20261020/index.lock': Operation not permitted
```

No commit was created and no escalation/bypass was attempted. HEAD remains
cd22e24640c5878af8bd1180ab9430d5dad3eda2. The complete verified owned changes
remain uncommitted in this worktree, alongside other workers' separate edits.
A portable patch containing only the 11 owned files is retained at
/private/tmp/glrmask-module-core-469935.patch for parent integration. Git staging
was empty immediately after the denied operation. Code verification results
above are unchanged; this final update is only the progress artifact.
