# Lazy Resolve Adapter Prototype Note

## Status

Prototype implemented and validated in the negative-lazy worktree.

Commit gate status:

- behavior-off path is clean
- env-on exact comparison passed on two small grammar integration tests
- reduced-JS env-off/env-on exact compare also passed under `GLRMASK_GLR_TABLE_CONSTRUCTION=core-lac`

This slice is now commit-worthy.

## What the prototype does

- adds a dense-id symbolic graph adapter in `parser_dwa.rs`
- symbolic node kinds:
  - `Continuation { td_state }`
  - `TemplateState { terminal, local_state, target_td_state, bundle_id }`
  - `BundleState { bundle_id, local_state, target_td_state }`
- implements `ResolveNegativesView` for the symbolic graph
- reuses existing template NWAs and built bundle NWAs as raw fragment sources
- under `GLRMASK_LAZY_NEGATIVE_PARSER_DWA=1`, builds the symbolic graph and compares read-only resolve-negative analyses against the materialized raw parser NWA:
  - cancellation summary
  - finality summary
  - terminal/default summary
- behavior-off production path remains unchanged

## Validation completed

Passed with env off:

- `cargo fmt --check`
- `cargo check`
- `cargo test --test integration nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix -- --nocapture`
- `cargo test --test integration direct_glrm_minimized_lowered_schema_has_two_stack_split -- --nocapture`

Passed with env on:

- `GLRMASK_LAZY_NEGATIVE_PARSER_DWA=1 cargo test --test integration nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix -- --nocapture`
- `GLRMASK_LAZY_NEGATIVE_PARSER_DWA=1 cargo test --test integration direct_glrm_minimized_lowered_schema_has_two_stack_split -- --nocapture`
- `GLRMASK_GLR_TABLE_CONSTRUCTION=core-lac` reduced-JS env-off/env-on exact compare via `lazy_reduced_js_compare.py`

## Reduced JS result

The reduced-JS compare succeeded:

- parser states matched: `93`
- mask length matched: `4001`
- exact masks matched on `26` checked prefixes
- `8` prefixes were rejected in both builds

The compare used:

- grammar: `/tmp/core_lac_correctness_reduced_js.glrm`
- filtered vocab bytes: `b"()[]{}-+"`
- table construction: `core-lac`
- env toggle: `GLRMASK_LAZY_NEGATIVE_PARSER_DWA=0/1`

## Artifacts

- patch: `lazy_resolve_adapter_prototype.patch`
- reduced-JS compare harness: `lazy_reduced_js_compare.py`
- logs copied on disk into this artifact directory