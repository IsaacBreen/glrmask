# Mathematical contracts for Parser-DWA construction

This document records the invariants each Parser-DWA submodule must preserve.
It is written as a review aid: every implementation edit should be checked
against these contracts before worrying about performance.

## Contract 1: weights denote pair masks

Every `Weight` in this subsystem denotes a set of lexer-state/token pairs
`(q,v)`.  The construction may combine weights by set operations only:

- union = alternative ways to obtain the same target;
- intersection = simultaneous satisfaction of path and edge constraints;
- difference = representation minimization when a final/default weight has been
  lifted elsewhere.

A weight must never be interpreted as a parser-state set, terminal set, or token
id set.  Names must preserve this distinction.

## Contract 2: Terminal bundles preserve Terminal-DWA semantics

For a Terminal-DWA state `s`, `terminal_projection.rs` groups outgoing terminal
edges by target state.  If multiple terminal labels from `s` enter the same
Terminal-DWA target, the grouped bundle records each terminal with its original
pair-mask weight.

This grouping is a representation change only.  It must preserve:

```text
for each source s, terminal t, target u:
  t --w--> u exists before grouping
  iff
  bundle for u contains t with weight w after grouping
```

If two terminal edges with the same terminal and target appear, their weights
must be unioned.

## Contract 3: bundle interning is semantic identity, not approximation

`BundleSignature` may be used to intern equal bundles.  Equality here is exact:
terminal ids and weights must match.  Do not replace this with hashes, summaries,
or approximate cost signatures unless the exact bundle is still checked.

## Contract 4: productivity is language reachability through accepting templates

A Terminal-DWA state is productive if either:

1. its final weight is nonempty; or
2. it has an outgoing branch whose bundle contains at least one terminal with a
   nonempty weight and an accepting stack-effect template, and whose target is
   productive.

This is a reverse graph reachability computation.  It is not a parser-table
reachability computation and it is not a tokenizer reachability computation.

## Contract 5: continuation states represent Terminal-DWA states

In `compose_nwa.rs`, every productive Terminal-DWA state receives exactly one
parser-NWA continuation state.  Template final states for a branch redirect to
the continuation state corresponding to the Terminal-DWA branch target.

This contract is what makes the composition finite and what preserves the
Terminal-DWA path structure.

## Contract 6: template fragments recognize parser stack effects

Template automata are stack-effect recognizers.  Parser-DWA construction treats
them abstractly.  It may splice, weight, cache, and redirect them, but it must
not depend on them being produced by LR, GLR, Earley, or any particular parser
algorithm.  The only required interface is:

- terminal id -> recognizer over parser stack-state labels;
- recognizer final states indicate completion of that terminal's stack effect.

## Contract 7: weighted template splicing uses intersection at path time

When a Terminal-DWA branch has weight `w` and a template edge/final has weight
`u`, the composed path weight is governed by intersection along the eventual
weighted path.  In the implementation, single-terminal template fragments have
edge weights overwritten by the Terminal-DWA branch weight; later determinization
intersects path weights.  Multi-terminal bundle construction handles the union
of alternatives before the same path semantics apply.

## Contract 8: epsilon closure must union equal-state contributions

During NWA determinization, epsilon closure maps NWA states to accumulated path
weights.  If two epsilon paths reach the same state with weights `w1` and `w2`,
the closure state's weight is `w1 ∪ w2`.  A new closure iteration is required
when the union adds new pairs.

## Contract 9: DWA supports are source-NWA supports

`DeterminizedDwaWithSupports.supports[d]` is the list of source NWA states whose
weighted entries form DWA state `d`.  The list is used only to infer possible
parser-state labels for fallback/default handling.  It must not be used as a
semantic replacement for DWA state identity.

## Contract 10: parser-state labels are nonnegative raw labels below table size

A raw automaton label denotes a parser state only when:

```text
label >= 0 && label < table.num_states
```

The helper in `labels.rs` is the only place this conversion should be performed.
Negative labels and `DEFAULT_LABEL` are automata representation labels, not parser
states.

## Contract 11: default-edge optimization preserves weighted language

`optimize_parser_dwa_defaults` may introduce a default edge when all possible
explicit parser-state labels share a target and a common weight component.  The
introduced default weight must be a subset of every explicit edge it replaces or
compresses.  Residual explicit weights are computed by set difference.

## Contract 12: final-weight subtraction preserves acceptance semantics

If a DWA state has final weight `f`, outgoing transition weights may subtract
`f` because acceptance at the current prefix already accounts for those pairs.
This is safe only when runtime evaluation unions/intersects final weights as the
DWA semantics require.  Do not apply this transformation to unrelated weights.

## Contract 13: fallback determinization makes default semantics explicit

After default-edge optimization, deterministic walking with fallbacks can be
viewed as an NFA-like weighted transition relation.  The second determinization
turns that relation back into an ordinary deterministic weighted automaton.
Runtime Mask should not need to reason about nondeterministic fallback sets.

## Contract 14: minimization is optional and semantics-preserving

Minimization may reduce states but must not change the weighted language.  The
current default skips minimization for compile-time performance.  This is a
policy decision, not a denotational decision.

## Contract 15: profile records are observational

Profiling must not affect weights, transitions, final states, start states, or
intermediate cache contents except for timing overhead.  Profile emission belongs
in `profiling.rs`; construction files may record metrics but should not own print
format strings.

## Review formula

For any stack prefix `rho`, compare the new code to the old monolith by asking:

```text
pairs returned after reading rho
= union over Terminal-DWA terminal strings compatible with (q,v)
  of parser template paths realizing rho
```

If a moved function preserves this equality, it is probably conceptually safe.
If a moved function changes what object it thinks it is manipulating, it is not.
