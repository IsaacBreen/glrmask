# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Invariant catalogue

### Grammar analysis invariants

1. The augmented start production is unique and points to the original start.
2. Nullable/FIRST/FOLLOW are computed over the normalized production set.
3. Normalization must preserve the terminal language of the original grammar.
4. Rule deduplication must not drop display metadata needed for diagnostics.
5. Hidden-left-recursion elimination must not introduce unreachable nonterminals without later reachability cleanup.

### Table construction invariants

1. Every state index referenced by an action or goto must be less than `num_states`.
2. Every action row has terminal keys in the table's terminal universe plus EOF where intended.
3. Every goto row key is a nonterminal.
4. `advance.len() == num_states` whenever admission rows are materialized.
5. State remapping must rewrite action targets, goto targets, advance rows, forwarded shifts, and guarded-shift indexes together.

### Optimizer invariants

1. `merge_identical_rows` may merge only rows with equal execution and admission behavior.
2. `prune_unreachable_states` may remove only states unreachable from the start through action/goto targets.
3. `collapse_sr_unit_reductions_with_compatible_gotos` may inline only reductions whose goto behavior is compatible with the shift/reduce cell being collapsed.
4. `quotient_recognizer_stack_suffixes` must respect future goto observations of the suffix states.
5. `canonicalize_stack_shift_predecessors` may replace an interior pushed predecessor only when its goto row is a compatible subset/superset relation.

### Advance invariants

1. `advance_stacks` consumes exactly one completed grammar terminal.
2. The input GSS is persistent; advance must not mutate shared caller state unexpectedly.
3. Guarded stack shifts must evaluate guards against actual lower-stack states.
4. Fast paths must be observationally equal to the general nondeterministic algorithm.
5. `stack_can_advance_on` is exact, not approximate.
6. `stacks_finished` must distinguish empty stack from EOF-accepted stack.
