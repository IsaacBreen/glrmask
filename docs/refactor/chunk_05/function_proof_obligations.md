# Function-level proof obligations for Chunk 05

This file maps each nontrivial function to its intended mathematical role, what may change inside it, and what must not change.

## `builder.rs`

### `build_parser_dwa_from_terminal_dwa_with_templates` at line 59

- Role: top-level phase graph.
- Main obligation: Must call phases in order and preserve the denotation of PDWA.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `build_parser_dwa_from_terminal_dwa_with_precomputed_templates` at line 201

- Role: compatibility wrapper.
- Main obligation: Must remain a thin wrapper that returns only the DWA.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `compose_nwa.rs`

### `dwa_to_nwa` at line 27

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `compute_productive_terminal_states` at line 48

- Role: productive continuation analysis.
- Main obligation: Must mark exactly states that can reach a nonempty final through accepting bundles.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `append_weighted_template_redirecting_finals` at line 90

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `append_bundle_redirecting_finals` at line 121

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `append_branch_fragment` at line 142

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `build_parser_nwa_from_terminal_dwa` at line 190

- Role: composition phase.
- Main obligation: Must produce an NWA whose paths correspond to Terminal-DWA paths decorated by stack-effect templates.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `determinize/epsilon.rs`

### `local_epsilon_closure` at line 14

- Role: weighted epsilon closure.
- Main obligation: Must union repeated state contributions and iterate until no pair mask grows.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `determinize/fallback.rs`

### `determinize_parser_dwa_with_fallbacks` at line 20

- Role: fallback semantics determinization.
- Main obligation: Must bake default-edge behavior into deterministic transitions.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `subset_key` at line 25

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `determinize/mod.rs`

## `determinize/outgoing.rs`

### `build_possible_outgoing_ids_by_state` at line 14

- Role: fallback domain analysis.
- Main obligation: Must over/precisely describe parser-state labels from supports; unsound under-approximation is forbidden.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `determinize/support.rs`

### `determinize_with_supports` at line 22

- Role: support-preserving subset construction.
- Main obligation: Must preserve weighted language and record source NWA supports.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `subset_key` at line 26

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `labels.rs`

### `parser_state_label` at line 8

- Role: raw label interpretation.
- Main obligation: Must reject negative/default/internal labels and out-of-range parser states.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `mod.rs`

## `optimize.rs`

### `union_final_weight` at line 18

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `optimize_parser_dwa_defaults` at line 40

- Role: default compression.
- Main obligation: Must replace repeated explicit edges only with subset-equivalent default weights.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `subtract_final_weights_from_outgoing_dwa` at line 229

- Role: final acceptance factoring.
- Main obligation: Must subtract only already-accepted pairs from outgoing transitions.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `options.rs`

### `skip_parser_dwa_minimization_env_override` at line 32

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `should_skip_parser_dwa_minimization` at line 44

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `profiling.rs`

### `elapsed_ms` at line 12

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `parser_dwa_compose_detail_enabled` at line 16

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `emit_parser_bundle_profile` at line 148

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `emit_parser_dwa_compose_profiles` at line 186

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `terminal_projection.rs`

### `group_terminal_edges_by_target` at line 21

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `bundle_signature` at line 47

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `terminal_template_has_acceptance` at line 54

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `terminal_bundle_has_acceptance` at line 58

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `build_state_summaries` at line 68

- Role: terminal projection.
- Main obligation: Must group terminal edges by Terminal-DWA target without losing weights.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `compute_productive_terminal_states` at line 117

- Role: productive continuation analysis.
- Main obligation: Must mark exactly states that can reach a nonempty final through accepting bundles.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

## `types.rs`

### `add_target_contribution` at line 32

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

### `extend_target_contribs` at line 49

- Role: local helper.
- Main obligation: Must preserve the contract of its caller and avoid mixing unrelated parser/lexer/token concepts.
- Inputs must be read according to their names: terminal ids are grammar terminals; token ids are vocabulary tokens; parser state labels are stack symbols; weights are pair masks.
- Allowed changes in later compile-fix pass: imports, formatting, local variable spelling, borrow-shaping, and moving helper calls without semantic changes.
- Forbidden changes in this refactor phase: changing union/intersection/difference meaning, changing start-state choice, dropping final weights, or interpreting default labels as parser states.

