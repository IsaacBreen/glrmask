# Basic implementer walkthrough

This walkthrough assumes the implementer understands files, functions, and simple Rust syntax, but not automata theory.

## Mental picture

Imagine a maze. The Terminal DWA maze knows which grammar words a token can spell. The parser templates know which stack shapes allow a grammar word. The Parser DWA is a new maze whose corridors are stack states and whose labels tell us which tokens are still possible.

## Open `builder.rs`

This file has 218 lines. Read the opening comment first. Then locate these symbols:

- line 43: `pub(crate) struct ParserDwaBuildInputs<'a> {`
- line 53: `pub(crate) struct ParserDwaBuildOutput {`
- line 59: `pub(crate) fn build_parser_dwa_from_terminal_dwa_with_templates(`
- line 201: `pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `compose_nwa.rs`

This file has 367 lines. Read the opening comment first. Then locate these symbols:

- line 27: `fn dwa_to_nwa(dwa: &DWA) -> NWA {`
- line 48: `fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {`
- line 90: `fn append_weighted_template_redirecting_finals(`
- line 121: `fn append_bundle_redirecting_finals(`
- line 142: `fn append_branch_fragment(`
- line 190: `pub(crate) fn build_parser_nwa_from_terminal_dwa(`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `determinize/epsilon.rs`

This file has 73 lines. Read the opening comment first. Then locate these symbols:

- line 14: `pub(super) fn local_epsilon_closure(`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `determinize/fallback.rs`

This file has 211 lines. Read the opening comment first. Then locate these symbols:

- line 20: `pub(crate) fn determinize_parser_dwa_with_fallbacks(`
- line 25: `fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `determinize/mod.rs`

This file has 18 lines. Read the opening comment first. Then locate these symbols:


What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `determinize/outgoing.rs`

This file has 99 lines. Read the opening comment first. Then locate these symbols:

- line 14: `pub(crate) fn build_possible_outgoing_ids_by_state(`
- line 19: `enum OutgoingIds {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `determinize/support.rs`

This file has 322 lines. Read the opening comment first. Then locate these symbols:

- line 22: `pub(crate) fn determinize_with_supports(`
- line 26: `fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `labels.rs`

This file has 14 lines. Read the opening comment first. Then locate these symbols:

- line 8: `pub(crate) fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `mod.rs`

This file has 64 lines. Read the opening comment first. Then locate these symbols:


What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `optimize.rs`

This file has 252 lines. Read the opening comment first. Then locate these symbols:

- line 18: `fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {`
- line 40: `pub(crate) fn optimize_parser_dwa_defaults(`
- line 229: `pub(crate) fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `options.rs`

This file has 55 lines. Read the opening comment first. Then locate these symbols:

- line 9: `pub(crate) struct ParserDwaOptions {`
- line 32: `fn skip_parser_dwa_minimization_env_override() -> Option<bool> {`
- line 44: `pub(crate) fn should_skip_parser_dwa_minimization(`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `profiling.rs`

This file has 220 lines. Read the opening comment first. Then locate these symbols:

- line 12: `pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {`
- line 16: `pub(crate) fn parser_dwa_compose_detail_enabled() -> bool {`
- line 23: `pub(crate) struct ParserNwaBuildProfile {`
- line 30: `pub(crate) struct ParserDwaComposeDetailProfile {`
- line 75: `pub(crate) struct ParserDwaProfile {`
- line 148: `pub(crate) fn emit_parser_bundle_profile(bundle_id: usize, bundle_profile: &BundleBuildProfile) {`
- line 186: `pub(crate) fn emit_parser_dwa_compose_profiles(detail: &ParserDwaComposeDetailProfile) {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `terminal_projection.rs`

This file has 157 lines. Read the opening comment first. Then locate these symbols:

- line 21: `fn group_terminal_edges_by_target(`
- line 47: `fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {`
- line 54: `fn terminal_template_has_acceptance(template: &NWA) -> bool {`
- line 58: `fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {`
- line 68: `pub(crate) fn build_state_summaries(`
- line 117: `pub(crate) fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Open `types.rs`

This file has 100 lines. Read the opening comment first. Then locate these symbols:

- line 21: `pub(crate) type TerminalBundle = BTreeMap<TerminalID, Weight>;`
- line 24: `pub(crate) type BundleSignature = Vec<(TerminalID, Weight)>;`
- line 28: `pub(crate) type TargetContribs = SmallVec<[(u32, Weight); 4]>;`
- line 32: `pub(crate) fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {`
- line 49: `pub(crate) fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {`
- line 58: `pub(crate) struct Branch {`
- line 65: `pub(crate) struct StateSummary {`
- line 72: `pub(crate) struct StateSummaries {`
- line 81: `pub(crate) struct DeterminizedDwaWithSupports {`
- line 88: `pub(crate) struct CachedClosure {`
- line 96: `pub(crate) enum PossibleOutgoingIds {`

What to verify in plain English:

1. The file should not mention JSON Schema, Python bindings, runtime commit, or runtime mask internals unless it is only in a comment explaining boundaries.
2. The file should not decide public API names.
3. The file should use `Weight` only as a set of possible lexer-state/token pairs.
4. If the file has `DEFAULT_LABEL`, check that it is treated as a fallback label, not a real parser state.

## Final beginner check

After applying the chunk, open `builder.rs`. If it is still huge, the chunk failed. If `builder.rs` mostly calls named helper functions, the chunk succeeded structurally.
