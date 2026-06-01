# Mapping to paper terminology

The paper names the online operations Mask and Commit.  This chunk makes the
runtime tree match that naming.

| Paper term | Runtime location | Meaning |
|---|---|---|
| compiled constraint | `runtime/artifact/`, `Constraint` | immutable object queried by runtime |
| active stacks/frontier | `runtime/state/mod.rs` | generated-prefix configuration |
| Mask | `runtime/mask/` | query returning allowed vocabulary tokens |
| Commit | `runtime/commit/` | transition after accepting bytes/tokens |
| Parser DWA | `compile/parser_dwa/`, runtime field | precomputed stack-prefix evaluator |
| terminal sequence | Commit tokenizer/parser bridge | completed lexer boundaries consumed by parser |
| delayed longest-match exclusions | parser accumulators inside GSS | terminal exclusions attached to branches |

The most important wording choice is `can`, not `may`, for exact runtime
predicates.  A predicate such as `stack_can_advance_on` is not a heuristic; it
is a statement about the parser transition relation.  This chunk renames the
local tokenizer-end-state helper accordingly.
