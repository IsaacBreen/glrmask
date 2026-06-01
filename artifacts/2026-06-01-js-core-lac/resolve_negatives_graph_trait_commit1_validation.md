# Resolve-Negatives Graph Trait Commit 1 Validation

## Scope

Implemented the first compile-preserving extraction from the plan:

- added `ResolveNegativesView`
- added `NwaResolveView<'_>`
- genericized read-only cancellation, finality, and terminal/default-analysis kernels
- kept public `resolve_negative_codes_in_nwa(&mut NWA)` concrete
- kept mutation/removal/default pruning concrete

## Validation

- `cargo fmt --check`
- `cargo check`
- `cargo test --test integration nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix -- --nocapture`
- `cargo test --test integration direct_glrm_minimized_lowered_schema_has_two_stack_split -- --nocapture`

All of the above passed.

## Logs

Copied into this artifact directory:

- `glrmask2_negative_lazy_worker4_fmt_check.log`
- `glrmask2_negative_lazy_worker4_cargo_check.log`
- `glrmask2_negative_lazy_worker4_test_nullable_repeat.log`
- `glrmask2_negative_lazy_worker4_test_direct_glrm_split.log`
- `glrmask2_negative_lazy_worker4_git_status.log`

## Patch

The current slice patch is saved as:

- `resolve_negatives_graph_trait_commit1.patch`

## Reduced JS smoke

Not run in this chunk.

I did not find an obvious cheap reduced-JS smoke command already checked into this worktree, and I did not want to widen the validation surface beyond the concrete resolver extraction without a known low-cost probe.