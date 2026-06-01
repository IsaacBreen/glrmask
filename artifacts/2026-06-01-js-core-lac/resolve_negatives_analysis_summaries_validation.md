# Resolve-Negatives Analysis Summaries Validation

## Scope

Second zero/low-behavior-change refactor chunk after the graph-view extraction:

- extracted explicit summary structs for cancellation output
- extracted explicit summary structs for finality outputs
- extracted explicit summary struct for terminal/default-analysis output
- kept public `resolve_negative_codes_in_nwa(&mut NWA)` behavior and mutation unchanged
- did not touch parser-DWA lazy code

## Validation

Passed:

- `cargo fmt --check`
- `cargo check`
- `cargo test --test integration nullable_repeat_alternative_accepts_nonempty_branch_before_nullable_suffix -- --nocapture`
- `cargo test --test integration direct_glrm_minimized_lowered_schema_has_two_stack_split -- --nocapture`

## Artifacts

Patch saved as:

- `resolve_negatives_analysis_summaries.patch`

Logs copied into this artifact directory on disk:

- `glrmask2_negative_lazy_worker4_analysis_summaries_fmt_check.log`
- `glrmask2_negative_lazy_worker4_analysis_summaries_cargo_check.log`
- `glrmask2_negative_lazy_worker4_analysis_summaries_test_nullable_repeat.log`
- `glrmask2_negative_lazy_worker4_analysis_summaries_test_direct_glrm_split.log`