# Parser-DWA phase graph

The Parser-DWA builder is now an explicit phase graph.  This document spells out
inputs, outputs, and forbidden dependencies for each phase.

## Phase 0: input package

File: `builder.rs`

Input struct:

```rust
ParserDwaBuildInputs {
    table,
    grammar,
    terminal_dwa,
    templates,
    vocab,
    id_map,
}
```

Only `table`, `grammar`, `terminal_dwa`, and `templates` are currently used by
the mathematical construction.  `vocab` and `id_map` remain in the compatibility
contract because surrounding compile stages still provide them and may use them
again when options/profile records become more explicit.

Forbidden in this phase:

- inspecting runtime `ConstraintState`;
- looking at JSON Schema ASTs;
- mutating the Terminal DWA;
- building templates.

## Phase 1: compose Terminal DWA with templates

File: `compose_nwa.rs`

Call:

```rust
build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates)
```

Output:

```rust
Option<(NWA, ParserNwaBuildProfile)>
```

`None` means the Terminal-DWA start state cannot reach any accepting parser-side
stack-effect witness.  The builder turns this into an empty DWA.

Subphases:

1. Build terminal summaries.
2. Compute productive states.
3. Allocate continuation states.
4. Cache multi-terminal bundle NWAs.
5. Append branch fragments.
6. Link branch starts to source continuations.
7. Set the start continuation.

## Phase 2: resolve negative labels

File: external helper from `compiler::stages::resolve_negatives`

Why it still exists: template construction may introduce negative labels for
internal representation.  Parser-DWA input symbols must be parser-state ids, so
negative labels must be resolved before parser-state determinization.

Future direction: move this helper under a general weighted-automata cleanup
chunk or a template-DFA subsystem chunk.

## Phase 3: support determinization

File: `determinize.rs`

Call:

```rust
determinize_with_supports(&parser_nwa, Some(table.num_states))
```

Output:

```rust
DeterminizedDwaWithSupports { dwa, supports }
```

The `supports` vector is critical because default/fallback semantics depend on
which source NWA states can be active at a DWA state.  A plain DWA would be
semantically enough for accepting stack prefixes but not enough to optimize
fallbacks safely.

## Phase 4: possible outgoing parser-state labels

File: `determinize.rs`

Call:

```rust
build_possible_outgoing_ids_by_state(&parser_nwa, &supports, table.num_states)
```

Output:

```rust
Vec<PossibleOutgoingIds>
```

This classifies each DWA state as:

- no possible parser-state outgoing labels;
- all parser-state labels possible;
- a specific bitset of possible parser-state labels.

The result is a domain restriction for default-edge optimization.

## Phase 5: default optimization

File: `optimize.rs`

Call:

```rust
optimize_parser_dwa_defaults(&mut parser_dwa_pre_minimize, &possible_by_state, table.num_states)
```

The optimization has three loops:

1. Discover when all possible explicit parser-state labels share a target and a
   common weight component.
2. Lift target-final contributions through default edges.
3. Remove explicit edge weights covered by default weights.

All three loops must preserve the weighted language.

## Phase 6: final-weight subtraction

File: `optimize.rs`

Call:

```rust
subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize)
```

This removes pairs from outgoing transitions when they are already accepted by
the source state's final weight.

## Phase 7: fallback determinization

File: `determinize.rs`

Call:

```rust
determinize_parser_dwa_with_fallbacks(&parser_dwa_pre_minimize, &possible_by_state, table.num_states)
```

This returns a deterministic DWA whose transitions have default semantics made
explicit.

## Phase 8: minimization policy

Files: `options.rs`, `builder.rs`

Call:

```rust
ParserDwaOptions::from_environment(pre_minimize_states, pre_minimize_transitions)
```

If `skip_minimization` is false, the weighted DWA minimizer runs.  Otherwise the
pre-minimized DWA is used directly.

## Phase 9: output package

File: `builder.rs`

Output struct:

```rust
ParserDwaBuildOutput { dwa, profile }
```

The compatibility wrapper returns only `dwa`, but the subsystem now has a place
to carry profile and future diagnostics without adding more positional return
values.


### Determinization subdirectory ownership

The first and second determinization passes are related but not identical.  The
subdirectory split records their different contracts:

| file | owns | does not own |
| --- | --- | --- |
| `determinize/outgoing.rs` | support-set to possible-parser-label analysis | building DWA transitions |
| `determinize/epsilon.rs` | local weighted epsilon closure | global subset worklist policy |
| `determinize/support.rs` | first weighted subset construction with supports | fallback/default semantics |
| `determinize/fallback.rs` | fallback-aware weighted subset construction | epsilon closure over parser NWA templates |

A future edit that changes one of these contracts should not be hidden as a
minor optimization; it changes a distinct mathematical phase.
